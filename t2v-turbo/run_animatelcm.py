from __future__ import annotations
import argparse
import glob
import json
import logging
import os
import random
import sys
import time

import torch
import torchvision
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl

# Diffusers Imports for AnimateLCM
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter, AutoencoderTiny
from diffusers.utils import export_to_video

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


def safe_stem(text: str, max_len: int = 120) -> str:
    stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)
    stem = "_".join(part for part in stem.split("_") if part)
    return stem[:max_len] or "sample"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    return random.randint(0, MAX_SEED) if randomize_seed else seed


def load_seed_map(seed_file: str) -> dict[str, int]:
    seed_map = {}
    if os.path.exists(seed_file):
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


def iter_vbench_prompt_file(path: str):
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


def iter_compbench_prompt_file(path: str):
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


class NoisEasier:
    def __init__(self, pipeline: AnimateDiffPipeline, args: argparse.Namespace):
        self.pipeline = pipeline
        self.args = args
        self.output_path = args.output_path

        self.video_dir = os.path.join(args.output_path, "all_videos")
        self.metrics_dir = os.path.join(args.output_path, "metrics")
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.dtype = torch.float16 if args.param_dtype == "fp16" else torch.float32
        self.device = args.device

        # Freeze pipeline
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.motion_adapter.requires_grad_(False)

        # -------------------------------------------------------
        # OPTIMIZATION: Load TinyVAE (TAESD) for fast decoding
        # -------------------------------------------------------
        self.reward_fns = {}
        if self.args.use_dno:
            logger.info("Loading TinyVAE (TAESD) for optimization loop...")
            try:
                self.tiny_vae = AutoencoderTiny.from_pretrained(
                    "madebyollin/taesd",
                    torch_dtype=self.dtype
                ).to(self.device)
                self.tiny_vae.requires_grad_(False)
            except Exception as e:
                logger.warning(f"Failed to load TAESD: {e}. Fallback to standard VAE (slow).")
                self.tiny_vae = None

            logger.info("Loading Reward Functions...")
            for name, w in zip(args.reward_fns, args.reward_weights):
                if args.reward_precision == "auto":
                    prec = "fp16"
                else:
                    prec = args.reward_precision
                logger.info(f"Loading reward: {name} in {prec}...")
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

    def lcm_step_with_learned_noise(
            self,
            latents: torch.Tensor,
            noise_pred: torch.Tensor,
            t: torch.Tensor | int,
            step_index: int,
            total_steps: int,
            step_noise: torch.Tensor | None,
    ):
        """Single LCM update with explicit noise injection."""
        scheduler = self.pipeline.scheduler
        device = latents.device
        dtype = latents.dtype

        t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)

        alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=dtype)
        final_alpha = scheduler.final_alpha_cumprod.to(device=device, dtype=dtype)

        alpha_prod_t = alphas_cumprod[t_int]
        prev_t = scheduler.previous_timestep(t_int)
        alpha_prod_t_prev = alphas_cumprod[prev_t] if prev_t >= 0 else final_alpha
        beta_prod_t = 1.0 - alpha_prod_t

        pt = scheduler.config.prediction_type
        if pt == "epsilon":
            predicted_original_sample = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
        elif pt == "sample":
            predicted_original_sample = noise_pred
        elif pt == "v_prediction":
            predicted_original_sample = alpha_prod_t.sqrt() * latents - beta_prod_t.sqrt() * noise_pred
        else:
            raise ValueError(f"Unsupported prediction_type: {pt}")

        # Check for diffusers version compatibility
        if hasattr(scheduler, "get_scalings_for_boundary_condition_discrete"):
            c_skip, c_out = scheduler.get_scalings_for_boundary_condition_discrete(t_int)
        else:
            sigma = (1 - alpha_prod_t) ** 0.5 / alpha_prod_t ** 0.5
            c_skip = 1.0 / (sigma ** 2 + 1.0)
            c_out = -sigma / (sigma ** 2 + 1.0)

        c_skip = torch.as_tensor(c_skip, device=device, dtype=dtype)
        c_out = torch.as_tensor(c_out, device=device, dtype=dtype)

        denoised = c_out * predicted_original_sample + c_skip * latents

        if step_index == total_steps - 1 or step_noise is None:
            return denoised

        beta_prod_t_prev = 1.0 - alpha_prod_t_prev
        latents_next = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * step_noise
        return latents_next

    def unet_forward_ckpt(self, latent_model_input, t, prompt_embeds):
        """Wrapper for UNet forward so we can use torch.utils.checkpoint."""
        return self.pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
        ).sample

    def tiny_vae_decode_frame(self, frame_latents):
        return self.tiny_vae.decode(frame_latents).sample

    def decode_latents_taesd(self, latents):
        """
        Fast decoding using TinyVAE (TAESD).
        """
        # NO MANUAL SCALING: Matches your finding that raw latents avoid "fried" colors
        b, c, f, h, w = latents.shape
        if self.args.checkpoint_vae_decode and latents.requires_grad:
            frames = []
            for i in range(f):
                frame = checkpoint(
                    self.tiny_vae_decode_frame,
                    latents[:, :, i],
                    use_reentrant=False,
                )
                frames.append(frame.unsqueeze(2))
            return torch.cat(frames, dim=2)

        latents_2d = latents.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        video_2d = self.tiny_vae.decode(latents_2d).sample
        video = video_2d.reshape(b, f, 3, video_2d.shape[2], video_2d.shape[3]).permute(0, 2, 1, 3, 4)
        return video

    def forward_loop(self, latents_init, step_noises_tensors, prompt_embeds, num_steps, use_taesd=False):
        """
        Unified generation loop.
        """
        self.pipeline.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.pipeline.scheduler.timesteps

        latents = latents_init * self.pipeline.scheduler.init_noise_sigma

        uncond_prompt_embeds, text_prompt_embeds = prompt_embeds.chunk(2)

        for i, t in enumerate(timesteps):
            latent_model_input = self.pipeline.scheduler.scale_model_input(latents, t)
            t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=self.device)

            if self.args.use_dno:
                # Stop-gradient CFG: only the conditional branch keeps DNO activations.
                with torch.no_grad():
                    noise_pred_uncond = self.pipeline.unet(
                        latent_model_input,
                        t_tensor,
                        encoder_hidden_states=uncond_prompt_embeds,
                    ).sample
                noise_pred_text = checkpoint(
                    self.unet_forward_ckpt,
                    latent_model_input,
                    t_tensor,
                    text_prompt_embeds,
                    use_reentrant=False
                )
            else:
                latent_model_input = torch.cat([latent_model_input] * 2)
                noise_pred = self.pipeline.unet(
                    latent_model_input, t_tensor, encoder_hidden_states=prompt_embeds
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            noise_pred = noise_pred_uncond + self.args.guidance_scale * (noise_pred_text - noise_pred_uncond)

            step_noise = step_noises_tensors[i] if i < num_steps - 1 else None
            latents = self.lcm_step_with_learned_noise(
                latents=latents,
                noise_pred=noise_pred,
                t=t,
                step_index=i,
                total_steps=num_steps,
                step_noise=step_noise,
            )

        # Decoding
        if use_taesd and self.tiny_vae is not None:
            # FAST PATH: TAESD (Outputs un-normalized ~[-1, 1], same as VAE)
            video = self.decode_latents_taesd(latents)
            video_tensor = (video / 2 + 0.5).clamp(0, 1)
        else:
            # SLOW / HIGH QUALITY PATH: Standard VAE
            video = self.pipeline.decode_latents(
                latents,
                decode_chunk_size=self.args.full_vae_decode_chunk_size,
            )
            video_tensor = (video / 2 + 0.5).clamp(0, 1)

        return video_tensor

    def sample_noise(self, seed):
        """Generate all necessary noise tensors deterministically."""
        generator = torch.Generator(device=self.device).manual_seed(seed)

        C = self.pipeline.unet.config.in_channels
        H = self.args.height or self.pipeline.unet.config.sample_size * 8
        W = self.args.width or self.pipeline.unet.config.sample_size * 8
        H_latent = H // self.pipeline.vae_scale_factor
        W_latent = W // self.pipeline.vae_scale_factor
        num_steps = self.args.num_inference_steps

        shape = (1, C, self.args.num_frames, H_latent, W_latent)

        latents_init = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)
        step_noise_shape = (num_steps - 1, *shape)
        step_noises = torch.randn(step_noise_shape, generator=generator, device=self.device, dtype=torch.float32)

        return latents_init, step_noises

    def prepare_prompt(self, prompt):
        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt, self.device, num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=self.args.negative_prompt
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_embeds = prompt_embeds.repeat_interleave(repeats=self.args.num_frames, dim=0)
        return prompt_embeds

    def generate_sample(
        self,
        prompt: str,
        index: int | str,
        seed: int,
        subdir: str | None = None,
        filename: str | None = None,
    ):
        logger.info(f"Generating Baseline for {prompt} with seed {seed}")
        prompt_embeds = self.prepare_prompt(prompt)
        latents_init, step_noises = self.sample_noise(seed)

        start_time = time.time()
        with torch.no_grad(), autocast(enabled=True, dtype=self.dtype):
            video_tensor = self.forward_loop(latents_init, step_noises, prompt_embeds,
                                             self.args.num_inference_steps, use_taesd=False)

        self.save_video_tensor(video_tensor, filename or f"{prompt}-{index}.mp4", subdir=subdir)
        logger.info(f"[Vanilla] Generation time: {time.time() - start_time:.4f}s")

    def optimize_sample(
        self,
        prompt_list: list[str],
        index: int | str,
        seed: int,
        subdir: str | None = None,
        filename: str | None = None,
    ):
        """Run Joint DNO."""
        prompt = prompt_list[0]
        logger.info(f"Starting Joint DNO for {prompt} with seed {seed}")
        prompt_embeds = self.prepare_prompt(prompt)

        latents_init_data, step_noises_data = self.sample_noise(seed)
        latents_init = latents_init_data.clone().detach().requires_grad_(True)
        if self.args.noise_optimization == "initial":
            step_noises = step_noises_data.detach()
            opt_params = [latents_init]
        elif self.args.noise_optimization == "full":
            step_noises = step_noises_data.clone().detach().requires_grad_(True)
            opt_params = [latents_init, step_noises]
        else:
            raise ValueError(f"Unknown noise_optimization mode: {self.args.noise_optimization}")

        optimizer = torch.optim.AdamW(opt_params, lr=self.args.lr)
        scaler = GradScaler(enabled=True)

        best_reward = float("-inf")
        best_video_tensor = None
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

        start_time = time.time()
        for it in range(self.args.dno_steps):
            # OPTIMIZATION: set_to_none=True is faster than zeroing memory
            optimizer.zero_grad(set_to_none=True)
            video_grad = None

            with autocast(enabled=True, dtype=self.dtype):
                # Use TAESD (use_taesd=True) for speed during optimization
                video_tensor = self.forward_loop(
                    latents_init, step_noises, prompt_embeds,
                    self.args.num_inference_steps, use_taesd=True
                )

                if self.args.reward_backward_mode == "sequential":
                    reward_video_tensor = video_tensor.to(self.args.reward_device).detach().requires_grad_(True)
                else:
                    video_tensor = video_tensor.to(self.args.reward_device)
                    reward_video_tensor = video_tensor

                total_loss = 0.0
                total_reward = 0.0
                reward_dict = {}

                for name, (fn, weight) in self.reward_fns.items():
                    # No skipping - strict per-sample calculation
                    r = fn(reward_video_tensor, prompt_list)
                    reward_dict[name] = r.item()
                    reward_loss = -r * weight
                    total_reward += r.item() * weight

                    if self.args.reward_backward_mode == "joint":
                        total_loss += reward_loss
                    elif self.args.reward_backward_mode == "sequential":
                        scaler.scale(reward_loss).backward()
                        if reward_video_tensor.grad is None:
                            raise RuntimeError(f"Reward {name} produced no video gradient.")
                        grad = reward_video_tensor.grad.detach()
                        video_grad = grad if video_grad is None else video_grad + grad
                        reward_video_tensor.grad = None
                        del grad, reward_loss, r
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise ValueError(f"Unknown reward_backward_mode: {self.args.reward_backward_mode}")

            if self.args.save_best and total_reward > best_reward:
                best_reward = total_reward
                best_video_tensor = video_tensor.detach().cpu()

            if self.args.reward_backward_mode == "joint":
                scaler.scale(total_loss).backward()
            elif self.args.reward_backward_mode == "sequential":
                if video_grad is None:
                    raise RuntimeError("Sequential reward backward produced no video gradient.")
                video_tensor.backward(video_grad.to(video_tensor.device))
            else:
                raise ValueError(f"Unknown reward_backward_mode: {self.args.reward_backward_mode}")
            scaler.unscale_(optimizer)
            initial_grad_norm = float(latents_init.grad.detach().float().norm().cpu())
            step_grad_norm = (
                float(step_noises.grad.detach().float().norm().cpu())
                if step_noises.requires_grad and step_noises.grad is not None
                else 0.0
            )
            total_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(opt_params, self.args.max_grad_norm).detach().cpu()
            )
            scaler.step(optimizer)
            scaler.update()

            initial_update_norm = float(
                (latents_init.detach() - latents_init_data).float().norm().cpu()
            )
            step_update_norm = (
                float((step_noises.detach() - step_noises_data).float().norm().cpu())
                if step_noises.requires_grad
                else 0.0
            )

            logger.info(
                f"[DNO] iter={it}, total_reward={total_reward:.4f}, "
                f"grad={total_grad_norm:.6g}, "
                f"update={initial_update_norm + step_update_norm:.6g} | "
                + ", ".join(f"{k}={v:.4f}" for k, v in reward_dict.items())
            )
            if self.args.save_metrics:
                metric_values = {
                    **reward_dict,
                    "total_reward": total_reward,
                    "initial_grad_norm": initial_grad_norm,
                    "step_grad_norm": step_grad_norm,
                    "total_grad_norm_before_clip": total_grad_norm,
                    "initial_update_norm": initial_update_norm,
                    "step_update_norm": step_update_norm,
                }
                metrics_list.append([metric_values.get(c, 0.0) for c in metrics_cols])
                metric_records.append({"iteration": it, **metric_values})


        if self.args.save_metrics:
            metric_stem = safe_stem(filename or f"{prompt}-{index}")
            np.save(
                os.path.join(self.metrics_dir, f"{metric_stem}.npy"),
                np.stack(metrics_list, axis=0),
            )
            with open(os.path.join(self.metrics_dir, f"{metric_stem}.json"), "w") as handle:
                json.dump(metric_records, handle, indent=2)

        # Final Best Save
        if self.args.save_best and best_video_tensor is not None:
            try:
                del video_tensor
                del total_loss
            except UnboundLocalError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad(), autocast(enabled=True, dtype=self.dtype):
                # Decode best result with Full VAE for max quality
                best_video_tensor = self.forward_loop(latents_init, step_noises, prompt_embeds,
                                                      self.args.num_inference_steps, use_taesd=False)
                best_video_tensor = best_video_tensor.detach().cpu()

            self.save_video_tensor(best_video_tensor, filename or f"{prompt}-{index}.mp4", subdir=subdir)

        elapsed = time.time() - start_time
        logger.info(f"[DNO] Finished in {elapsed:.4f}s")

    def save_video_tensor(self, video_tensor, filename, fps=8, crf="10", subdir: str | None = None):
        video_dir = os.path.join(self.video_dir, subdir) if subdir else self.video_dir
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, filename)
        video_tensor = video_tensor.cpu()
        video = (video_tensor[0] * 255).to(torch.uint8).permute(1, 2, 3, 0)
        torchvision.io.write_video(
            video_path, video, fps=fps, video_codec="h264", options={"crf": crf}
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

    def run_vbench(self):
        seed_log = open(os.path.join(self.output_path, "seeds.txt"), "w", buffering=1)
        seed_map = load_seed_map(self.args.seed_file) if self.args.seed_file else {}
        processed = 0
        tasks = build_vbench_tasks(
            self.args.prompts_dir,
            iter_vbench_prompt_file,
            self.args.vbench_dimensions,
        )
        if self.args.vbench_dimensions:
            logger.info("Running VBench dimensions: %s", ", ".join(self.args.vbench_dimensions))

        def run_task(task: tuple[int, str, str, list[str], int]):
            task_id, dim, key_prompt, neg_prompts, idx = task
            filename = vbench_video_filename(key_prompt, idx)
            existing = self.find_existing_video(filename)
            if existing:
                logger.info("[Rank %s] Skipping existing VBench task %s/%s: %s", self.args.rank, task_id + 1, len(tasks), existing)
                return False

            seed_key = f"{dim}:{key_prompt}-{idx}"
            seed = seed_map.get(seed_key, random.randint(0, MAX_SEED))
            seed_log.write(f"{seed_key}:{seed}\n")
            seed_log.flush()

            logger.info("[Rank %s] VBench task %s/%s (%s idx=%s)", self.args.rank, task_id + 1, len(tasks), dim, idx)
            prompt_list = [key_prompt]
            if self.args.use_neg_prompt:
                prompt_list += neg_prompts
            if self.args.use_dno:
                self.optimize_sample(prompt_list, idx, seed, filename=filename)
            else:
                self.generate_sample(prompt_list[0], idx, seed, filename=filename)
            return True

        if self.args.shard_strategy == "dynamic":
            queue_dir = os.path.join(self.args.run_root, "_queue")
            progress = tqdm(disable=self.args.rank != 0)
            while True:
                task_id = claim_next_task(queue_dir, "vbench", len(tasks))
                if task_id is None:
                    break
                if run_task(tasks[task_id]):
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
                if run_task(task):
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
        if self.args.seed_file and os.path.exists(self.args.seed_file):
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
                    logger.info("Using Negative Prompts for Optimization...")
                self.optimize_sample(prompt_list, sample_id, seed, subdir=category, filename=filename)
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
                if run_task(tasks[task_id]):
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
                if run_task(task):
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
            self.run_vbench()
        else:
            prompts = load_user_prompts(self.args.test_prompt, self.args.prompt_file)
            for index, prompt in enumerate(prompts):
                seed = randomize_seed_fn((self.args.seed or 0) + index, self.args.randomize_seed)
                logger.info("Running prompt %s/%s with seed %s", index + 1, len(prompts), seed)
                stem = f"{index:03d}_{safe_stem(prompt)}"
                if self.args.use_dno:
                    self.optimize_sample(
                        [prompt], index=index, seed=seed,
                        filename=f"{stem}.mp4",
                    )
                else:
                    self.generate_sample(prompt, index=index, seed=seed, filename=f"{stem}.mp4")


def main():
    parser = argparse.ArgumentParser("AnimateLCM + DNO (VBench / T2V-CompBench)")

    # Basic setup
    parser.add_argument('--seed', type=int, default=20260123)
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


    # AnimateLCM Configuration
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--param_dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--negative_prompt", type=str, default="bad quality, worse quality, low resolution")
    parser.add_argument(
        "--full_vae_decode_chunk_size",
        type=int,
        default=1,
        help="Frame chunk size for full VAE decode during baseline generation or final best-video save.",
    )

    # DNO specifics
    parser.add_argument("--use_dno", type=str2bool, default=True)
    parser.add_argument("--dno_steps", type=int, default=25)
    parser.add_argument(
        "--noise_optimization",
        type=str,
        default="full",
        choices=["initial", "full"],
        help="Which diffusion noise tensors to optimize during DNO.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument('--save_path', type=str, default='animatelcm_vbench')
    parser.add_argument('--save_best', type=str2bool, default=True)
    parser.add_argument('--save_metrics', type=str2bool, default=False)
    parser.add_argument(
        "--reward_backward_mode",
        type=str,
        default="joint",
        choices=["joint", "sequential"],
        help="Backpropagate all rewards together, or accumulate per-reward video gradients before one generator backward.",
    )

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
        help="Checkpoint each frame's differentiable TAESD decode to reduce DNO activation memory.",
    )

    # Prompt Setup
    parser.add_argument('--run_vbench', type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument('--run_compbench', type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--prompt", "--test_prompt", dest="test_prompt", type=str,
                        default="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution")
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
            args.save_path = "animatelcm_compbench"
        if args.seed_file == parser.get_default("seed_file"):
            args.seed_file = "comp-t2v-prompts/compbench_seed.txt"
    pl.seed_everything(args.seed)

    if args.rank != 0 and not args.log_all_ranks:
        logger.setLevel(logging.WARNING)

    # Output Setup
    ts = args.run_id or time.strftime('%Y-%m-%d-%H-%M-%S')
    args.run_root = os.path.join(args.save_path, ts)
    args.output_path = os.path.join(args.run_root, f"shard_{args.rank}")
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    logger.info("Loading AnimateLCM Pipeline...")

    adapter = MotionAdapter.from_pretrained(
        "wangfuyun/AnimateLCM",
        torch_dtype=torch.float16 if args.param_dtype == "fp16" else torch.float32
    )

    pipeline = AnimateDiffPipeline.from_pretrained(
        "emilianJR/epiCRealism",
        motion_adapter=adapter,
        torch_dtype=torch.float16 if args.param_dtype == "fp16" else torch.float32
    )

    pipeline.scheduler = LCMScheduler.from_config(
        pipeline.scheduler.config,
        beta_schedule="linear"
    )

    logger.info("Loading LoRA Weights...")
    try:
        pipeline.load_lora_weights(
            "wangfuyun/AnimateLCM",
            weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
            adapter_name="lcm-lora"
        )
        pipeline.set_adapters(["lcm-lora"], [0.8])
    except Exception as e:
        logger.warning(f"Could not load AnimateLCM LoRA: {e}. Check connectivity/paths.")

    pipeline.to(args.device)

    runner = NoisEasier(pipeline, args)
    runner.run()


if __name__ == "__main__":
    main()
