#!/usr/bin/env python
# inference_vc2_dno_vbench.py
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import numpy as np
import torch
import torchvision
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pytorch_lightning as pl

from utils.lora import collapse_lora, monkeypatch_remove_lora
from utils.lora_handler import LoraHandler
from utils.common_utils import load_model_checkpoint
from utils.utils import instantiate_from_config
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from pipeline.t2v_turbo_vc2_pipeline import T2VTurboVC2Pipeline
from reward_fn import get_reward_fn
from utils.user_prompts import load_user_prompts
from benchmark_sharding import (
    build_compbench_tasks,
    build_vbench_tasks,
    claim_next_task,
    compbench_video_filename,
    existing_video_path,
    vbench_video_filename,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_SEED = np.iinfo(np.int32).max

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    return random.randint(0, MAX_SEED) if randomize_seed else seed


def load_seed_map(seed_file: str) -> dict[str, int]:
    seed_map: dict[str, int] = {}
    if seed_file and os.path.exists(seed_file):
        with open(seed_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, seed_str = line.rsplit(':', 1)
                seed_map[key] = int(seed_str)
    return seed_map


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def iter_vbench_prompt_file(path: str) -> Iterator[tuple[str, list[str]]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        data = json.load(open(path))
        if isinstance(data, dict):
            for prompt, neg_prompts in data.items():
                yield prompt, neg_prompts
        elif isinstance(data, list):
            for item in data:
                prompt = item.get("prompt_en") if isinstance(item, dict) else None
                if prompt:
                    yield prompt, []
        else:
            raise ValueError(f"Unsupported VBench JSON format in {path}")
    elif ext == ".txt":
        with open(path, "r") as f:
            for line in f:
                prompt = line.strip()
                if prompt:
                    yield prompt, []
    else:
        raise ValueError(f"Unsupported VBench prompt file extension: {path}")


def iter_compbench_prompt_file(path: str) -> Iterator[tuple[str, str, str, list[str]]]:
    """Yield sample id, prompt, category, and negative prompts from T2V-CompBench JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported T2V-CompBench JSON format in {path}")

    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Unsupported T2V-CompBench item in {path}: {item!r}")
        yield (
            str(item["id"]),
            item["prompt"],
            item["type"],
            item.get("negative_prompt", []),
        )


def safe_stem(text: str, max_len: int = 120) -> str:
    stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)
    stem = "_".join(part for part in stem.split("_") if part)
    return stem[:max_len] or "sample"


class NoisEasierVC2:
    def __init__(self, pipeline: T2VTurboVC2Pipeline, args: argparse.Namespace):
        self.pipeline = pipeline
        self.args = args
        self.output_path = args.output_path
        self.lcm_origin_steps = 200

        # create output directories
        self.video_dir = os.path.join(self.output_path, "all_videos")
        self.metrics_dir = os.path.join(self.output_path, "metrics")
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        # move & freeze pipeline modules
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.param_dtype]
        for m in (pipeline.vae, pipeline.unet, pipeline.text_encoder):
            m.to(device=args.device, dtype=dtype)
            m.requires_grad_(False)
        self.dtype = dtype

        # prepare reward functions
        self.reward_fns: dict[str, tuple] = {}
        if args.use_dno:
            for name, w in zip(args.reward_fns, args.reward_weights):
                if args.reward_precision == "auto":
                    prec = "fp16" if name == "videoscore" else "fp32"
                else:
                    prec = args.reward_precision
                reward_kwargs = {}
                if name == "hpsv2":
                    reward_kwargs = {"margin": args.hps_margin, "tau": args.hps_tau}
                elif name == "viclip":
                    reward_kwargs = {"margin": args.viclip_margin, "tau": args.viclip_tau}
                elif name == "motion":
                    reward_kwargs = {
                        "alpha": args.motion_dynamic_weight,
                        "beta": args.motion_smooth_weight,
                    }
                fn = get_reward_fn(
                    name, precision=prec, device=args.reward_device, **reward_kwargs
                )
                self.reward_fns[name] = (fn, w)

    def save_video(
        self,
        video_tensor: torch.Tensor,
        filename: str,
        fps: int = 8,
        subdir: str | None = None,
    ) -> str:
        video_dir = os.path.join(self.video_dir, subdir) if subdir else self.video_dir
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, filename)
        video = (video_tensor[0] * 255).to(torch.uint8).permute(1, 2, 3, 0)
        torchvision.io.write_video(
            video_path, video.detach().cpu(), fps=fps, video_codec="h264", options={"crf": "10"}
        )
        return video_path

    def find_existing_video(self, filename: str, subdir: str | None = None) -> str | None:
        if not self.args.skip_existing:
            return None
        return existing_video_path(
            run_root=self.args.run_root,
            rank_video_dir=self.video_dir,
            filename=filename,
            subdir=subdir,
            scope=self.args.skip_existing_scope,
            min_bytes=self.args.min_existing_video_bytes,
        )

    def decode_video_differentiable(self, denoised: torch.Tensor) -> torch.Tensor:
        z = denoised.to(self.pipeline.vae.dtype)
        if not self.args.checkpoint_vae_decode:
            return self.pipeline.pretrained_t2v.decode_first_stage_2DAE_differentiable(z)

        z = 1.0 / self.pipeline.pretrained_t2v.scale_factor * z
        b, _, t, _, _ = z.shape
        frames = []
        for i in range(t):
            frame = torch.utils.checkpoint.checkpoint(
                self.pipeline.vae.decode,
                z[:, :, i],
                use_reentrant=False,
            )
            frames.append(frame.unsqueeze(2))
        return torch.cat(frames, dim=2)

    def generate_sample(
        self,
        prompt: str,
        index: int | str,
        seed: int | None = None,
        subdir: str | None = None,
        filename: str | None = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        start_time = time.time()
        with autocast(enabled=True, dtype=self.dtype):
            video = self.pipeline(
                prompt=prompt,
                frames=self.args.num_frames,
                fps=self.args.fps,
                guidance_scale=self.args.guidance_scale,
                motion_gs=self.args.motion_gs,
                use_motion_cond=(self.args.version == "v2"),
                percentage=self.args.percentage,
                num_inference_steps=self.args.num_inference_steps,
                lcm_origin_steps=self.lcm_origin_steps,
                num_videos_per_prompt=1,
            )
        video = (video.clamp(-1, 1) + 1) / 2
        out_filename = filename or f"{prompt}-{index}.mp4"
        out = self.save_video(video, out_filename, fps=self.args.fps, subdir=subdir)
        logger.info(f"Video saved to: {out}")
        logger.info(f"[Vanilla] Generation time: {time.time() - start_time:.4f}s")

    def optimize_sample(
        self,
        prompt_list: list[str],
        index: int | str,
        seed: int | None = None,
        subdir: str | None = None,
        filename: str | None = None,
        metrics_name: str | None = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        # encode prompt
        prompt_embeds = self.pipeline._encode_prompt(
            prompt_list[0],
            device=self.args.device,
            num_videos_per_prompt=1,
            prompt_embeds=None,
        )

        # init noise trajectory
        C = self.pipeline.unet.in_channels
        H, W = self.pipeline.model_config['params']['image_size']
        shape = (self.args.num_inference_steps, 1, C, self.args.num_frames, H, W)
        init_noises = torch.randn(shape, device=self.args.device, dtype=torch.float32)
        noise_optimization = getattr(self.args, "noise_optimization", "full")
        if noise_optimization == "initial":
            initial_noise = init_noises[0].detach().clone().requires_grad_(True)
            fixed_step_noises = init_noises[1:].detach()
            opt_params = [initial_noise]
            logger.info("[DNO] Optimizing initial latent noise only; stepwise noises are frozen.")
        elif noise_optimization == "full":
            noises = init_noises.detach().clone().requires_grad_(True)
            opt_params = [noises]
            logger.info("[DNO] Optimizing full noise trajectory.")
        else:
            raise ValueError(f"Unknown noise_optimization mode: {noise_optimization}")

        # optimizer & scaler
        optimizer = torch.optim.AdamW(opt_params, lr=self.args.lr)
        scaler = GradScaler(enabled=True)

        # scheduler & embeddings
        self.pipeline.scheduler.set_timesteps(
            self.args.num_inference_steps,
            self.lcm_origin_steps,
            device=self.args.device,
        )
        w_embed = self.pipeline.get_w_embedding(
            torch.tensor([self.args.guidance_scale]),
            embedding_dim=256,
            dtype=self.dtype,
        ).to(self.args.device)

        motion_embed = None
        if self.args.version == "v2":
            motion_embed = self.pipeline.get_w_embedding(
                torch.tensor([self.args.motion_gs]),
                embedding_dim=256,
                dtype=self.dtype,
            ).to(self.args.device)

        best_reward = float("-inf")
        best_video = None
        metrics_cols = list(self.reward_fns.keys()) + [
            "total_reward",
            "initial_grad_norm",
            "step_grad_norm",
            "total_grad_norm_before_clip",
            "initial_update_norm",
            "step_update_norm",
        ]
        metrics_list = []
        metric_records = []
        save_metrics = getattr(self.args, "save_metrics", True)

        start_time = time.time()
        for it in range(self.args.dno_steps):
            optimizer.zero_grad()
            with autocast(enabled=True, dtype=self.dtype):
                if noise_optimization == "initial":
                    latent = initial_noise * self.pipeline.scheduler.init_noise_sigma
                else:
                    latent = noises[0] * self.pipeline.scheduler.init_noise_sigma
                for idx, t in enumerate(self.pipeline.scheduler.timesteps):
                    if idx < self.args.num_inference_steps - 1:
                        if noise_optimization == "initial":
                            step_noise = fixed_step_noises[idx]
                        else:
                            step_noise = noises[idx + 1]
                    else:
                        step_noise = None
                    latent, denoised = self.pipeline.forward_step(
                        latents=latent,
                        timeindex=idx,
                        timestep=t.item(),
                        prompt_embeds=prompt_embeds,
                        w_embedding=w_embed,
                        motion_embedding=motion_embed,
                        fps=self.args.fps,
                        step_noise=step_noise,
                        eta=self.args.eta,
                    )
                # decode video
                video = self.decode_video_differentiable(denoised)
                video = (video.clamp(-1, 1) + 1) / 2
                video = video.to(self.args.reward_device)

                # compute rewards
                total_loss = 0.0
                total_reward = 0.0
                reward_dict = {}
                for name, (fn, w) in self.reward_fns.items():
                    r = fn(video, prompt_list)
                    reward_dict[name] = r.item()
                    total_loss += -r * w
                    total_reward += r.item() * w

            # backward & step
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            if noise_optimization == "initial":
                initial_grad_norm = float(initial_noise.grad.detach().float().norm().cpu())
                step_grad_norm = 0.0
            else:
                initial_grad_norm = float(noises.grad[0].detach().float().norm().cpu())
                step_grad_norm = float(noises.grad[1:].detach().float().norm().cpu())
            total_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(opt_params, self.args.max_grad_norm).detach().cpu()
            )
            scaler.step(optimizer)
            scaler.update()

            if noise_optimization == "initial":
                initial_update_norm = float(
                    (initial_noise.detach() - init_noises[0]).float().norm().cpu()
                )
                step_update_norm = 0.0
            else:
                initial_update_norm = float(
                    (noises[0].detach() - init_noises[0]).float().norm().cpu()
                )
                step_update_norm = float(
                    (noises[1:].detach() - init_noises[1:]).float().norm().cpu()
                )

            # track best
            if self.args.save_best and total_reward > best_reward:
                best_reward = total_reward
                best_video = video.detach().cpu()

            # log & save
            logger.info(
                f"[DNO] iter={it}, total_reward={total_reward:.4f}, "
                f"grad={total_grad_norm:.6g}, "
                f"update={initial_update_norm + step_update_norm:.6g} | "
                + ", ".join(f"{k}={v:.4f}" for k, v in reward_dict.items())
            )

            if self.args.save_step_video:
                fname = f"{prompt_list[0]}-{index}-step-{it}.mp4"
                self.save_video(video.cpu(), fname, fps=self.args.fps, subdir=subdir)

            if save_metrics:
                metric_values = {
                    **reward_dict,
                    "total_reward": total_reward,
                    "initial_grad_norm": initial_grad_norm,
                    "step_grad_norm": step_grad_norm,
                    "total_grad_norm_before_clip": total_grad_norm,
                    "initial_update_norm": initial_update_norm,
                    "step_update_norm": step_update_norm,
                }
                row = [metric_values.get(c, 0.0) for c in metrics_cols]
                metrics_list.append(row)
                metric_records.append({"iteration": it, **metric_values})


        # save metrics
        if save_metrics:
            metric_file = metrics_name or f"{prompt_list[0]}-{index}.npy"
            np.save(
                os.path.join(self.metrics_dir, metric_file),
                np.stack(metrics_list, axis=0)
            )
            json_file = f"{os.path.splitext(metric_file)[0]}.json"
            with open(os.path.join(self.metrics_dir, json_file), "w") as handle:
                json.dump(metric_records, handle, indent=2)
        if self.args.save_best and best_video is not None:
            best_filename = filename or f"{prompt_list[0]}-{index}.mp4"
            self.save_video(best_video, best_filename, fps=self.args.fps, subdir=subdir)

        elapsed = time.time() - start_time
        logger.info(f"[DNO] Total Optimization time: {elapsed:.4f}s")

    def run_vbench(self):
        seed_log = open(os.path.join(self.output_path, "seeds.txt"), "w", buffering=1)
        seed_map = {}
        if self.args.seed_file:
            seed_map = load_seed_map(self.args.seed_file)

        processed = 0
        tasks = build_vbench_tasks(
            self.args.prompts_dir,
            iter_vbench_prompt_file,
            self.args.vbench_dimensions,
        )
        if self.args.vbench_dimensions:
            logger.info("Running VBench dimensions: %s", ", ".join(self.args.vbench_dimensions))

        def run_task(task: tuple[int, str, str, list[str], int]):
            task_id, dim, ori_prompt, neg_prompts, idx = task
            filename = vbench_video_filename(ori_prompt, idx)
            existing = self.find_existing_video(filename)
            if existing:
                logger.info("[Rank %s] Skipping existing VBench task %s/%s: %s", self.args.rank, task_id + 1, len(tasks), existing)
                return False
            key = f"{dim}:{ori_prompt}-{idx}"
            seed = seed_map.get(key, random.randint(0, MAX_SEED))
            torch.manual_seed(seed)
            seed_log.write(f"{key}:{seed}\n")
            seed_log.flush()

            logger.info("[Rank %s] VBench task %s/%s (%s idx=%s)", self.args.rank, task_id + 1, len(tasks), dim, idx)
            if self.args.use_dno:
                prompt_list = [ori_prompt]
                if self.args.use_neg_prompt:
                    prompt_list += neg_prompts
                    logger.info("Use Negative Prompts for Optimization...")
                self.optimize_sample(prompt_list, idx, seed, filename=filename)
            else:
                self.generate_sample(ori_prompt, idx, seed, filename=filename)
            return True

        if self.args.shard_strategy == "dynamic":
            queue_dir = os.path.join(self.args.run_root, "_queue")
            progress = tqdm(disable=self.args.rank != 0)
            while True:
                task_id = claim_next_task(queue_dir, "vbench", len(tasks))
                if task_id is None:
                    break
                generated = run_task(tasks[task_id])
                if generated:
                    processed += 1
                progress.update(1)
                if self.args.max_vbench_samples is not None and processed >= self.args.max_vbench_samples:
                    logger.info("Reached max_vbench_samples=%s on rank %s", self.args.max_vbench_samples, self.args.rank)
                    break
            progress.close()
        else:
            for task in tqdm(tasks):
                task_id = task[0]
                if task_id % self.args.world_size != self.args.rank:
                    continue
                generated = run_task(task)
                if generated:
                    processed += 1
                if self.args.max_vbench_samples is not None and processed >= self.args.max_vbench_samples:
                    logger.info("Reached max_vbench_samples=%s on rank %s", self.args.max_vbench_samples, self.args.rank)
                    break
        seed_log.close()
        logger.info("Rank %s completed %s VBench samples.", self.args.rank, processed)
        sys.exit(0)

    def run_compbench(self):
        """Shard-aware loop over T2V-CompBench prompt JSON files."""
        seed_log = open(os.path.join(self.output_path, "seeds.txt"), "w", buffering=1)
        seed_map = {}
        if self.args.seed_file:
            seed_map = load_seed_map(self.args.seed_file)

        processed = 0
        tasks = build_compbench_tasks(self.args.prompts_dir, iter_compbench_prompt_file)

        def run_task(task: tuple[int, str, str, str, list[str]]):
            task_id, sample_id, prompt, category, neg_prompts = task
            filename = compbench_video_filename(sample_id)
            existing = self.find_existing_video(filename, subdir=category)
            if existing:
                logger.info("[Rank %s] Skipping existing CompBench task %s/%s: %s", self.args.rank, task_id + 1, len(tasks), existing)
                return False

            key = f"{category}:{sample_id}-{prompt}"
            seed = seed_map.get(key, random.randint(0, MAX_SEED))
            pl.seed_everything(seed, verbose=False)
            seed_log.write(f"{key}:{seed}\n")
            seed_log.flush()

            logger.info("[Rank %s] CompBench task %s/%s (%s id=%s)", self.args.rank, task_id + 1, len(tasks), category, sample_id)
            if self.args.use_dno:
                prompt_list = [prompt]
                if self.args.use_neg_prompt:
                    prompt_list += neg_prompts
                    logger.info("Use Negative Prompts for Optimization...")
                self.optimize_sample(
                    prompt_list,
                    sample_id,
                    seed,
                    subdir=category,
                    filename=filename,
                    metrics_name=f"{category}_{sample_id}_{safe_stem(prompt)}.npy",
                )
            else:
                self.generate_sample(prompt, sample_id, seed, subdir=category, filename=filename)
            return True

        if self.args.shard_strategy == "dynamic":
            queue_dir = os.path.join(self.args.run_root, "_queue")
            progress = tqdm(disable=self.args.rank != 0)
            while True:
                task_id = claim_next_task(queue_dir, "compbench", len(tasks))
                if task_id is None:
                    break
                generated = run_task(tasks[task_id])
                if generated:
                    processed += 1
                progress.update(1)
                if self.args.max_compbench_samples is not None and processed >= self.args.max_compbench_samples:
                    logger.info("Reached max_compbench_samples=%s on rank %s", self.args.max_compbench_samples, self.args.rank)
                    break
            progress.close()
        else:
            for task in tqdm(tasks, desc="T2V-CompBench tasks"):
                task_id = task[0]
                if task_id % self.args.world_size != self.args.rank:
                    continue
                generated = run_task(task)
                if generated:
                    processed += 1
                if self.args.max_compbench_samples is not None and processed >= self.args.max_compbench_samples:
                    logger.info("Reached max_compbench_samples=%s on rank %s", self.args.max_compbench_samples, self.args.rank)
                    break

        seed_log.close()
        logger.info("Rank %s completed %s CompBench samples.", self.args.rank, processed)
        sys.exit(0)

    def run(self):
        if self.args.run_compbench:
            logger.info("Running T2V-CompBench evaluation...")
            self.run_compbench()
        elif self.args.run_vbench:
            logger.info("Running VBench evaluation...")
            self.run_vbench()
        else:
            prompts = load_user_prompts(self.args.test_prompt, self.args.prompt_file)
            for index, prompt in enumerate(prompts):
                seed = randomize_seed_fn((self.args.seed or 0) + index, self.args.randomize_seed)
                torch.manual_seed(seed)
                stem = f"{index:03d}_{safe_stem(prompt)}"
                if self.args.use_dno:
                    logger.info("Running DNO for prompt %s/%s", index + 1, len(prompts))
                    self.optimize_sample(
                        [prompt], index=index, seed=seed,
                        filename=f"{stem}.mp4", metrics_name=f"{stem}.npy",
                    )
                else:
                    logger.info("Generating prompt %s/%s", index + 1, len(prompts))
                    self.generate_sample(prompt, index=index, seed=seed, filename=f"{stem}.mp4")


def main():
    parser = argparse.ArgumentParser("T2V‑Turbo VC2 + DNO (VBench / T2V-CompBench)")

    # Basic setup
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--randomize_seed', action='store_true')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--reward_device", type=str, default="cuda:0")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--shard_strategy", type=str, choices=["static", "dynamic"], default="static")
    parser.add_argument("--log_all_ranks", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--skip_existing", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--skip_existing_scope", type=str, choices=["rank", "run"], default="run")
    parser.add_argument("--min_existing_video_bytes", type=int, default=1)
    parser.add_argument('--param_dtype', type=str, choices=["fp16", "bf16", "fp32"], default="fp16")


    # Model Setup (VC2)
    parser.add_argument("--unet_dir", type=str, default="checkpoints/vc2_unet_lora.pt")
    parser.add_argument("--base_model_dir", type=str, default="checkpoints/vc2_model.ckpt")
    parser.add_argument("--version", type=str, choices=["v1", "v2"], default="v1")
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--num_inference_steps', type=int, default=4)
    parser.add_argument("--motion_gs", type=float, default=0.05)
    parser.add_argument("--percentage", type=float, default=0.3)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--fps', type=int, default=8)

    # DNO specifics
    parser.add_argument("--use_dno", type=str2bool, default=True)
    parser.add_argument("--dno_steps", type=int, default=25)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--noise_optimization",
        type=str,
        choices=["full", "initial"],
        default="full",
        help="DNO variable set: 'full' optimizes initial and stepwise noises; "
             "'initial' freezes stepwise noises and optimizes only the initial latent noise.",
    )
    parser.add_argument('--save_path', type=str, default='vc2_vbench_dno')
    parser.add_argument('--save_step_video', type=str2bool, default=False)
    parser.add_argument('--save_metrics', type=str2bool, default=False)
    parser.add_argument('--save_best', type=str2bool, default=True)

    parser.add_argument('--reward_fns', nargs='+', type=str,
                        default=["hpsv2", "img_reward", "motion", "viclip"])
    parser.add_argument('--reward_weights', nargs='+', type=float,
                        default=[2.0, 1.0, 1.0, 2.0])
    parser.add_argument('--hps_margin', type=float, default=0.02)
    parser.add_argument('--hps_tau', type=float, default=0.01)
    parser.add_argument('--viclip_margin', type=float, default=0.05)
    parser.add_argument('--viclip_tau', type=float, default=0.05)
    parser.add_argument('--motion_dynamic_weight', type=float, default=1.0)
    parser.add_argument('--motion_smooth_weight', type=float, default=0.5)
    parser.add_argument(
        '--reward_precision',
        type=str,
        choices=["auto", "fp32", "fp16", "bf16"],
        default="fp16",
        help="Reward-model precision used during noise optimization.",
    )
    parser.add_argument(
        '--checkpoint_vae_decode',
        type=str2bool,
        default=True,
        help="Checkpoint each frame's differentiable VAE decode to reduce generator-GPU memory.",
    )

    # Prompt & VBench
    parser.add_argument('--run_vbench', type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument('--run_compbench', type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--prompt", "--test_prompt", dest="test_prompt", type=str,
                        default="A corgi's head depicted as an explosion of a nebula")
    parser.add_argument(
        "--prompt-file", "--prompt_file", dest="prompt_file", type=str, default=None,
        help="Small UTF-8 TXT/JSON prompt list; bypasses benchmark modes.",
    )
    parser.add_argument('--prompts_dir', type=str, default='vbench_prompts/neg_prompts')
    parser.add_argument('--seed_file', type=str, default="vbench_prompts/vbench_seed.txt")
    parser.add_argument('--use_neg_prompt', type=str2bool, default=False)
    parser.add_argument(
        '--vbench_dimensions',
        nargs='+',
        default=None,
        help="Optional VBench dimension name(s) to run, e.g. color overall_consistency.",
    )
    parser.add_argument('--max_vbench_samples', type=int, default=None)
    parser.add_argument('--max_compbench_samples', type=int, default=None)

    args = parser.parse_args()
    if args.run_compbench:
        if args.prompts_dir == parser.get_default("prompts_dir"):
            args.prompts_dir = "comp-t2v-prompts/neg_prompts"
        if args.save_path == parser.get_default("save_path"):
            args.save_path = "vc2_compbench"
        if args.seed_file == parser.get_default("seed_file"):
            args.seed_file = "comp-t2v-prompts/compbench_seed.txt"
    pl.seed_everything(args.seed)

    if args.rank != 0 and not args.log_all_ranks:
        logger.setLevel(logging.WARNING)

    # Prepare output
    ts = args.run_id or time.strftime('%Y-%m-%d-%H-%M-%S')
    args.run_root = os.path.join(args.save_path, ts)
    args.output_path = os.path.join(args.run_root, f"shard_{args.rank}")
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Load model & UNet + LoRA
    from omegaconf import OmegaConf
    config = OmegaConf.load('configs/inference_t2v_512_v2.0.yaml')
    model_config = config.pop('model', OmegaConf.create())
    base = instantiate_from_config(model_config)
    base = load_model_checkpoint(base, args.base_model_dir)

    uconf = model_config['params']['unet_config']
    uconf['params']['use_checkpoint'] = False
    uconf['params']['time_cond_proj_dim'] = 256
    if args.version == 'v2':
        uconf['params']['motion_cond_proj_dim'] = 256
    unet = instantiate_from_config(uconf)

    if "lora" in args.unet_dir.lower():
        unet.load_state_dict(
            base.model.diffusion_model.state_dict(),
            strict=False
        )
        use_unet_lora = True
        lora_manager = LoraHandler(
            version="cloneofsimo",
            use_unet_lora=use_unet_lora,
            save_for_webui=True,
            unet_replace_modules=["UNetModel"]
        )
        lora_manager.add_lora_to_model(
            use_unet_lora,
            unet,
            lora_manager.unet_replace_modules,
            lora_path=args.unet_dir,
            dropout=0.1,
            r=64
        )
        collapse_lora(unet, lora_manager.unet_replace_modules)
        monkeypatch_remove_lora(unet)
    else:
        unet.load_state_dict(torch.load(args.unet_dir, map_location="cpu"))
    unet.eval()
    base.model.diffusion_model = unet

    # Instantiate Pipeline and Runner
    scheduler = T2VTurboScheduler(
        linear_start=model_config['params']['linear_start'],
        linear_end=model_config['params']['linear_end']
    )
    pipeline = T2VTurboVC2Pipeline(base, scheduler, model_config).to(args.device)

    runner = NoisEasierVC2(pipeline, args)
    runner.run()


if __name__ == '__main__':
    main()
