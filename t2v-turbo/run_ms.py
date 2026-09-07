from __future__ import annotations
import argparse
import glob
import json
import logging
import os
import random
import sys
import time
import uuid
from typing import Iterator

import torch
import torchvision
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl

from diffusers.models import AutoencoderKL
from model_scope.unet_3d_condition import UNet3DConditionModel
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from pipeline.t2v_turbo_ms_pipeline import T2VTurboMSPipeline
from utils.common_utils import set_torch_2_attn
from utils.lora import collapse_lora, monkeypatch_remove_lora
from utils.lora_handler import LoraHandler
from utils.user_prompts import load_user_prompts
from reward_fn import get_reward_fn

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
    if not seed_file or not os.path.exists(seed_file):
        return seed_map
    with open(seed_file, 'r') as f:
        for line in f:
            # each line is “dim:prompt-idx:seed”
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
        with open(path, "r") as f:
            data = json.load(f)
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


class NoisEasier:
    def __init__(self, pipeline: T2VTurboMSPipeline, args: argparse.Namespace):
        """
        One‑time setup: move & freeze models, create dirs, build reward & neg‑prompt generators.
        """
        self.pipeline = pipeline
        self.args = args
        self.output_path = args.output_path

        # create output directories
        self.video_dir = os.path.join(args.output_path, "all_videos")
        self.metrics_dir = os.path.join(args.output_path, "metrics")
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        # move & freeze pipeline modules
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.param_dtype]
        for m in (pipeline.vae, pipeline.unet, pipeline.text_encoder):
            m.to(device=args.device, dtype=dtype)
            m.requires_grad_(False)
        self.dtype = dtype

        # prepare reward functions
        self.reward_fns = {}
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

    def decode_video_differentiable(self, denoised: torch.Tensor) -> torch.Tensor:
        z = denoised.to(self.pipeline.vae.dtype) / self.pipeline.vae.config.scaling_factor

        def decode_frame(frame_latent: torch.Tensor) -> torch.Tensor:
            return self.pipeline.vae.decode(frame_latent)[0]

        frames = []
        for i in range(z.shape[2]):
            if self.args.checkpoint_vae_decode:
                frame = checkpoint(decode_frame, z[:, :, i], use_reentrant=False)
            else:
                frame = decode_frame(z[:, :, i])
            frames.append(frame.unsqueeze(2))
        return torch.cat(frames, dim=2)

    def generate_sample(
        self,
        prompt: str,
        index: int | str,
        seed: int = None,
        subdir: str | None = None,
        filename: str | None = None,
    ):
        """Generate one video via vanilla sampling."""
        if seed is None:
            seed = self.args.seed
        generator = torch.Generator(device=self.args.device).manual_seed(seed)
        with autocast(enabled=True, dtype=self.dtype):
            video = self.pipeline(
                prompt=prompt,
                frames=self.args.num_frames,
                guidance_scale=self.args.guidance_scale,
                num_inference_steps=self.args.num_inference_steps,
                num_videos_per_prompt=1,
                generator=generator
            )
        video = (video.clamp(-1, 1) + 1) / 2  # normalize to [0, 1]
        file_name = filename or f"{prompt}-{index}.mp4"
        out = self.save_video(video, file_name, subdir=subdir)
        logger.info(f"Video saved to: {out}")

    def optimize_sample(
        self,
        prompt_list: list[str],
        index: int | str,
        seed: int = None,
        subdir: str | None = None,
        filename: str | None = None,
        metrics_name: str | None = None,
    ):
        """Run DNO on one prompt (or prompt + neg prompts)."""
        if seed is None:
            seed = self.args.seed
        # encode prompt
        prompt_embeds = self.pipeline._encode_prompt(
            prompt_list[0],
            device=self.args.device,
            num_videos_per_prompt=1,
            prompt_embeds=None,
        )

        # init learnable noise
        C = self.pipeline.unet.config.in_channels
        H, W = 256, 256
        shape = (
            self.args.num_inference_steps,
            1, C, self.args.num_frames,
            H // self.pipeline.vae_scale_factor,
            W // self.pipeline.vae_scale_factor,
        )
        generator = torch.Generator(device=self.args.device).manual_seed(seed)
        init_noises = torch.randn(shape, generator=generator, device=self.args.device, dtype=torch.float32)
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

        # scheduler & guidance embedding
        self.pipeline.scheduler.set_timesteps(
            self.args.num_inference_steps,
            self.args.num_inference_steps,
            device=self.args.device,
        )
        w_embed = self.pipeline.get_w_embedding(
            torch.tensor([self.args.guidance_scale]),
            embedding_dim=256,
            dtype=self.dtype,
        ).to(self.args.device)

        best_reward = float("-inf")
        best_video = None
        metrics_cols = list(self.reward_fns.keys()) + [
            "l2_loss",
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

        # optimization loop
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

            # track best
            if self.args.save_best and total_reward > best_reward:
                best_reward = total_reward
                best_video = video.detach().cpu()

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

            # log & save metrics
            logger.info(
                f"[DNO] iter={it}, total_reward={total_reward:.4f}, "
                f"grad={total_grad_norm:.6g}, "
                f"update={initial_update_norm + step_update_norm:.6g} | "
                + ", ".join(f"{k}={v:.4f}" for k, v in reward_dict.items())
            )
            if self.args.save_step_video:
                step_filename = f"{prompt_list[0]}-{index}-step-{it}.mp4"
                self.save_video(video, step_filename, subdir=subdir)

            if save_metrics:
                if noise_optimization == "initial":
                    current_noises = torch.cat(
                        [initial_noise.detach().unsqueeze(0), fixed_step_noises],
                        dim=0,
                    )
                else:
                    current_noises = noises.detach()
                l2_loss = torch.mean((current_noises - init_noises.detach()) ** 2).item()
                metric_values = {
                    **reward_dict,
                    "l2_loss": l2_loss,
                    "total_reward": total_reward,
                    "initial_grad_norm": initial_grad_norm,
                    "step_grad_norm": step_grad_norm,
                    "total_grad_norm_before_clip": total_grad_norm,
                    "initial_update_norm": initial_update_norm,
                    "step_update_norm": step_update_norm,
                }
                row = [metric_values.get(col, 0.0) for col in metrics_cols]
                metrics_list.append(row)
                metric_records.append({"iteration": it, **metric_values})


        # save metrics + best video
        if save_metrics:
            metric_file = metrics_name or f"{prompt_list[0]}-{index}.npy"
            np.save(os.path.join(self.metrics_dir, metric_file),
                    np.stack(metrics_list, axis=0))
            json_file = f"{os.path.splitext(metric_file)[0]}.json"
            with open(os.path.join(self.metrics_dir, json_file), "w") as handle:
                json.dump(metric_records, handle, indent=2)
        if self.args.save_best and best_video is not None:
            best_filename = filename or f"{prompt_list[0]}-{index}.mp4"
            self.save_video(best_video, best_filename, subdir=subdir)

        elapsed = time.time() - start_time
        logger.info(f"[DNO] Total Optimization time: {elapsed:.4f}s")

    def save_video(self, video_tensor, filename, fps=8, crf="10", subdir: str | None = None):
        # video_tensor: (1, C, T, H, W) in range [0, 1]
        video_dir = os.path.join(self.video_dir, subdir) if subdir else self.video_dir
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, filename)
        video = (video_tensor[0] * 255).to(torch.uint8).permute(1, 2, 3, 0)  # (T, H, W, C)
        torchvision.io.write_video(
            video_path, video.detach().cpu(), fps=fps, video_codec="h264", options={"crf": crf}
        )
        return video_path

    def run_vbench(self):
        """Shard‐aware loop over prompt files: either optimize or generate."""
        seed_log = open(os.path.join(self.output_path, "seeds.txt"), "w")
        seed_map = {}
        if self.args.seed_file:
            seed_map = load_seed_map(self.args.seed_file)

        counter = 0
        processed = 0
        prompt_files = sorted(glob.glob(os.path.join(self.args.prompts_dir, "*.json")))
        if not prompt_files:
            prompt_files = sorted(glob.glob(os.path.join(self.args.prompts_dir, "*.txt")))
        if not prompt_files:
            raise FileNotFoundError(f"No VBench prompt files found in {self.args.prompts_dir}")

        for prompt_file in tqdm(prompt_files):
            dim = os.path.splitext(os.path.basename(prompt_file))[0]
            n = 25 if dim.lower() == "temporal_flickering" else 5

            for ori_prompt, neg_prompts in tqdm(iter_vbench_prompt_file(prompt_file), leave=False):
                for idx in range(n):
                    if counter % self.args.world_size != self.args.rank:
                        counter += 1
                        continue

                    # draw seed for each sample
                    key = f"{dim}:{ori_prompt}-{idx}"
                    seed = seed_map.get(key, random.randint(0, MAX_SEED))
                    # torch.manual_seed(seed)
                    seed_log.write(f"{key}:{seed}\n")

                    # build prompt list
                    if self.args.use_dno:
                        prompt_list = [ori_prompt]
                        if self.args.use_neg_prompt:
                            prompt_list += neg_prompts
                            logger.info(f"Use Negative Prompts for Optimization...")
                        self.optimize_sample(prompt_list, idx, seed)
                    else:
                        self.generate_sample(ori_prompt, idx, seed)

                    counter += 1
                    processed += 1
                    if (
                        self.args.max_vbench_samples is not None
                        and processed >= self.args.max_vbench_samples
                    ):
                        seed_log.close()
                        logger.info(
                            "Reached max_vbench_samples=%s on rank %s",
                            self.args.max_vbench_samples,
                            self.args.rank,
                        )
                        sys.exit(0)

        seed_log.close()
        sys.exit(0)

    def run_compbench(self):
        """Shard-aware loop over T2V-CompBench prompt JSON files."""
        seed_log = open(os.path.join(self.output_path, "seeds.txt"), "w")
        seed_map = {}
        if self.args.seed_file:
            seed_map = load_seed_map(self.args.seed_file)

        counter = 0
        processed = 0
        prompt_files = sorted(glob.glob(os.path.join(self.args.prompts_dir, "*.json")))
        if not prompt_files:
            raise FileNotFoundError(f"No T2V-CompBench JSON files found in {self.args.prompts_dir}")

        for prompt_file in tqdm(prompt_files, desc="T2V-CompBench files"):
            for sample_id, prompt, category, neg_prompts in tqdm(
                iter_compbench_prompt_file(prompt_file),
                desc=os.path.basename(prompt_file),
                leave=False,
            ):
                if counter % self.args.world_size != self.args.rank:
                    counter += 1
                    continue

                key = f"{category}:{sample_id}-{prompt}"
                seed = seed_map.get(key, random.randint(0, MAX_SEED))
                pl.seed_everything(seed, verbose=False)
                seed_log.write(f"{key}:{seed}\n")

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
                        filename=f"{sample_id}.mp4",
                        metrics_name=f"{category}_{sample_id}_{safe_stem(prompt)}.npy",
                    )
                else:
                    self.generate_sample(
                        prompt,
                        sample_id,
                        seed,
                        subdir=category,
                        filename=f"{sample_id}.mp4",
                    )

                counter += 1
                processed += 1
                if (
                    self.args.max_compbench_samples is not None
                    and processed >= self.args.max_compbench_samples
                ):
                    seed_log.close()
                    logger.info(
                        "Reached max_compbench_samples=%s on rank %s",
                        self.args.max_compbench_samples,
                        self.args.rank,
                    )
                    sys.exit(0)

        seed_log.close()
        sys.exit(0)

    def run(self):
        """Entry point: benchmark mode or single demo."""
        if self.args.run_compbench:
            logger.info("Running T2V-CompBench evaluation...")
            self.run_compbench()
        elif self.args.run_vbench:
            logger.info("Running VBench evaluation...")
            self.run_vbench()
        else:
            # single‐shot demo
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
    parser = argparse.ArgumentParser("T2V‑Turbo + DNO (VBench / T2V-CompBench)")

    # Basic setup
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--randomize_seed', action='store_true')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--reward_device", type=str, default="cuda:0")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total #GPUs / shards running in parallel")
    parser.add_argument("--rank", type=int, default=0,
                        help="Shard index for this process (0‑based)")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Shared run folder name")


    # Model Setup
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--param_dtype", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32"])
    parser.add_argument(
        "--unet_dir",
        type=str,
        default="checkpoints/ms_unet_lora.pt",
        help="Path to the public T2V-Turbo (MS) LoRA checkpoint.",
    )
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
    parser.add_argument('--save_path', type=str, default='ms_vbench_dno')
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
        default=False,
        help="Checkpoint each frame's differentiable VAE decode to reduce generator-GPU memory.",
    )

    # Prompt Setup
    parser.add_argument('--run_vbench', type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument('--run_compbench', type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--prompt", "--test_prompt", dest="test_prompt", type=str,
                        default="A blue car drives past a white picket fence on a sunny day")
    parser.add_argument(
        "--prompt-file", "--prompt_file", dest="prompt_file", type=str, default=None,
        help="Small UTF-8 TXT/JSON prompt list; bypasses benchmark modes.",
    )
    parser.add_argument('--prompts_dir', type=str, default='vbench_prompts/neg_prompts')
    parser.add_argument('--seed_file', type=str, default="vbench_prompts/vbench_seed.txt")
    parser.add_argument("--use_neg_prompt", type=str2bool, default=False,
                        help="Whether use negative prompts for contrastive learning")
    parser.add_argument('--max_vbench_samples', type=int, default=None)
    parser.add_argument('--max_compbench_samples', type=int, default=None)

    args = parser.parse_args()
    if args.run_compbench:
        if args.prompts_dir == parser.get_default("prompts_dir"):
            args.prompts_dir = "comp-t2v-prompts/neg_prompts"
        if args.save_path == parser.get_default("save_path"):
            args.save_path = "ms_compbench_dno"
        if args.seed_file == parser.get_default("seed_file"):
            args.seed_file = "comp-t2v-prompts/compbench_seed.txt"
    pl.seed_everything(args.seed)

    if args.rank != 0:
        logger.setLevel(logging.WARNING)

    # Prepare output
    ts = args.run_id or time.strftime('%Y-%m-%d-%H-%M-%S')
    args.output_path = os.path.join(args.save_path, ts, f"shard_{args.rank}")
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Build pipeline
    pretrained_model_path = "ali-vilab/text-to-video-ms-1.7b"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    teacher_unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    time_cond_proj_dim = 256
    unet = UNet3DConditionModel.from_config(
        teacher_unet.config,
        time_cond_proj_dim=time_cond_proj_dim,
    )
    unet.load_state_dict(teacher_unet.state_dict(), strict=False)
    del teacher_unet
    set_torch_2_attn(unet)

    # LoRA merge
    lora_manager = LoraHandler(version="cloneofsimo", use_unet_lora=True, save_for_webui=True)
    lora_manager.add_lora_to_model(
        True,
        unet,
        lora_manager.unet_replace_modules,
        lora_path=args.unet_dir,
        dropout=0.1,
        r=32,
    )
    collapse_lora(unet, lora_manager.unet_replace_modules)
    monkeypatch_remove_lora(unet)
    unet.eval()

    # Instantiate Pipeline
    noise_scheduler = T2VTurboScheduler()
    pipeline = T2VTurboMSPipeline(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
    ).to(args.device)

    runner = NoisEasier(pipeline, args)
    runner.run()


if __name__ == "__main__":
    main()
