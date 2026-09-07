"""Inference and Direct Noise Optimization for LTXV-2B distilled."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from huggingface_hub import hf_hub_download

from ltx_video.inference import (
    calculate_padding,
    create_latent_upsampler,
    create_ltx_video_pipeline,
    get_device,
    load_pipeline_config,
)
from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.vae_encode import vae_decode
from ltx_video.pipelines.pipeline_ltx_video import (
    SkipLayerStrategy,
)
from ltx_video.schedulers.rf import RectifiedFlowSchedulerOutput
from benchmark_sharding import (
    build_compbench_tasks,
    claim_next_task,
    compbench_video_filename,
    existing_video_path,
    sanitize_filename_component,
)
from prompt_io import load_user_prompts


DEFAULT_PROMPT = (
    "A blue car drives past a white picket fence on a sunny day."
)
DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"
CONFIG_ONLY_KEYS = {
    "checkpoint_path",
    "downscale_factor",
    "first_pass",
    "pipeline_type",
    "precision",
    "prompt_enhancement_words_threshold",
    "prompt_enhancer_image_caption_model_name_or_path",
    "prompt_enhancer_llm_model_name_or_path",
    "second_pass",
    "spatial_upscaler_model_path",
    "stg_mode",
    "text_encoder_model_name_or_path",
}
IMAGE_STYLE_REWARDS = {
    "aesthetic",
    "clip",
    "hpsv2",
    "img_reward",
    "pickscore",
}
MAX_SEED = np.iinfo(np.int32).max


def log(message: str) -> None:
    print(message, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def safe_stem(text: str, max_len: int = 80) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts)[:max_len] or "sample"


def save_video(video: torch.Tensor, path: Path, fps: int) -> None:
    """Save a (B, C, T, H, W) tensor in [0, 1] as an mp4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clip = video[0].detach().clamp(0, 1).permute(1, 2, 3, 0).cpu().float().numpy()
    clip = (clip * 255).round().astype(np.uint8)
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in clip:
            writer.append_data(frame)


def resolve_model_path(filename_or_path: str) -> str:
    if os.path.isfile(filename_or_path):
        return filename_or_path
    local_checkpoint = Path("checkpoints") / filename_or_path
    if local_checkpoint.is_file():
        return str(local_checkpoint)
    return hf_hub_download(
        repo_id="Lightricks/LTX-Video",
        filename=filename_or_path,
        repo_type="model",
    )


def stg_strategy_from_config(config: dict) -> SkipLayerStrategy:
    stg_mode = config.get("stg_mode", "attention_values")
    if stg_mode.lower() in {"stg_av", "attention_values"}:
        return SkipLayerStrategy.AttentionValues
    if stg_mode.lower() in {"stg_as", "attention_skip"}:
        return SkipLayerStrategy.AttentionSkip
    if stg_mode.lower() in {"stg_r", "residual"}:
        return SkipLayerStrategy.Residual
    if stg_mode.lower() in {"stg_t", "transformer_block"}:
        return SkipLayerStrategy.TransformerBlock
    raise ValueError(f"Invalid spatiotemporal guidance mode: {stg_mode}")


def built_in_reward(name: str, video: torch.Tensor, prompt_list: list[str]) -> torch.Tensor:
    del prompt_list
    if name == "brightness":
        return video.mean()
    if name == "saturation":
        channel_max = video.max(dim=1).values
        channel_min = video.min(dim=1).values
        return (channel_max - channel_min).mean()
    if name == "motion":
        if video.shape[2] < 2:
            return torch.zeros((), device=video.device, dtype=video.dtype)
        return (video[:, :, 1:] - video[:, :, :-1]).abs().mean()
    if name == "temporal_smoothness":
        if video.shape[2] < 2:
            return torch.zeros((), device=video.device, dtype=video.dtype)
        return -(video[:, :, 1:] - video[:, :, :-1]).abs().mean()
    raise KeyError(name)


def differentiable_adain_filter(
    latents: torch.Tensor,
    reference_latents: torch.Tensor,
    factor: float = 1.0,
) -> torch.Tensor:
    """AdaIN used by LTX multi-scale, written without in-place slice updates."""
    reduce_dims = (2, 3, 4)
    ref_std, ref_mean = torch.std_mean(reference_latents, dim=reduce_dims, keepdim=True)
    lat_std, lat_mean = torch.std_mean(latents, dim=reduce_dims, keepdim=True)
    normalized = (latents - lat_mean) / lat_std
    filtered = normalized * ref_std + ref_mean
    return torch.lerp(latents, filtered, factor)


def load_seed_map(seed_file: str | None) -> dict[str, int]:
    seed_map: dict[str, int] = {}
    if not seed_file or not os.path.exists(seed_file):
        return seed_map
    with open(seed_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            key, seed_text = line.rsplit(":", 1)
            seed_map[key] = int(seed_text)
    return seed_map


def iter_compbench_prompt_file(path: str) -> Iterator[tuple[str, str, str, list[str]]]:
    """Yield sample id, prompt, category, and negatives from T2V-CompBench JSON."""
    if Path(path).suffix.lower() != ".json":
        return

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

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


class LTXDistilledDNO:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device or get_device())
        self.output_dir = Path(args.output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        benchmark_mode = getattr(args, "run_compbench", False)
        self.metrics_dir = self.output_dir / ("metrics" if benchmark_mode else "dno_metrics")
        self.video_dir = self.output_dir / ("all_videos" if benchmark_mode else "dno_videos")
        self.metrics_dir.mkdir(exist_ok=True)
        self.video_dir.mkdir(exist_ok=True)

        self.pipeline_config = load_pipeline_config(args.pipeline_config)
        self.pipeline_config = dict(self.pipeline_config)
        self.skip_layer_strategy = stg_strategy_from_config(self.pipeline_config)
        self.precision = self.pipeline_config["precision"]

        ckpt_path = resolve_model_path(self.pipeline_config["checkpoint_path"])
        upscaler_path = self.pipeline_config.get("spatial_upscaler_model_path")
        if upscaler_path:
            upscaler_path = resolve_model_path(upscaler_path)

        self.video_pipeline = create_ltx_video_pipeline(
            ckpt_path=ckpt_path,
            precision=self.precision,
            text_encoder_model_name_or_path=self.pipeline_config[
                "text_encoder_model_name_or_path"
            ],
            sampler=self.pipeline_config.get("sampler"),
            device=str(self.device),
            enhance_prompt=False,
            prompt_enhancer_image_caption_model_name_or_path=self.pipeline_config[
                "prompt_enhancer_image_caption_model_name_or_path"
            ],
            prompt_enhancer_llm_model_name_or_path=self.pipeline_config[
                "prompt_enhancer_llm_model_name_or_path"
            ],
        )
        self.latent_upsampler = None
        if self.pipeline_config.get("pipeline_type") == "multi-scale":
            if upscaler_path is None:
                raise ValueError("Multi-scale LTX config requires a spatial upscaler")
            self.latent_upsampler = create_latent_upsampler(
                upscaler_path, str(self.device)
            )

        self._freeze_modules()
        self.common_call_kwargs = {
            key: value
            for key, value in self.pipeline_config.items()
            if key not in CONFIG_ONLY_KEYS
        }

        self.height_padded = ((args.height - 1) // 32 + 1) * 32
        self.width_padded = ((args.width - 1) // 32 + 1) * 32
        self.num_frames_padded = ((args.num_frames - 2) // 8 + 1) * 8 + 1
        self.padding = calculate_padding(
            args.height, args.width, self.height_padded, self.width_padded
        )
        log(
            "Padded dimensions: "
            f"{self.height_padded}x{self.width_padded}x{self.num_frames_padded}"
        )

        self.reward_fns = self._build_reward_fns() if args.use_dno else {}

    def _freeze_modules(self) -> None:
        modules = [
            self.video_pipeline.transformer,
            self.video_pipeline.vae,
            self.video_pipeline.text_encoder,
        ]
        if self.latent_upsampler is not None:
            modules.append(self.latent_upsampler)
        for module in modules:
            if isinstance(module, torch.nn.Module):
                module.eval()
                module.requires_grad_(False)
        if self.args.gradient_checkpointing:
            self.video_pipeline.transformer.gradient_checkpointing = True
            self.video_pipeline.transformer.train()
            self.video_pipeline.transformer.requires_grad_(False)

    def _build_reward_fns(self) -> dict[str, tuple[Callable, float]]:
        if len(self.args.reward_fns) != len(self.args.reward_weights):
            raise ValueError("--reward-fns and --reward-weights must have the same length")
        reward_fns: dict[str, tuple[Callable, float]] = {}
        built_ins = {"brightness", "saturation", "motion", "temporal_smoothness"}
        for name, weight in zip(self.args.reward_fns, self.args.reward_weights):
            if name in built_ins:
                reward_fns[name] = (lambda video, prompts, n=name: built_in_reward(n, video, prompts), weight)
                continue

            from reward_fn import get_reward_fn

            precision = self.args.reward_precision
            if precision == "auto":
                precision = "fp16" if name == "videoscore" else "fp32"
            log(f"Loading external reward model: {name} ({precision})")
            reward_fns[name] = (
                get_reward_fn(name, precision=precision, device=self.args.reward_device),
                weight,
            )
        return reward_fns

    @property
    def transformer_dtype(self) -> torch.dtype:
        return getattr(self.video_pipeline.transformer, "dtype", torch.bfloat16)

    def _first_pass_dimensions(self) -> tuple[int, int]:
        if self.latent_upsampler is None:
            return self.height_padded, self.width_padded
        downscale_factor = self.pipeline_config["downscale_factor"]
        vae_scale = self.video_pipeline.vae_scale_factor
        height = int(self.height_padded * downscale_factor)
        width = int(self.width_padded * downscale_factor)
        return height - (height % vae_scale), width - (width % vae_scale)

    def _latent_shape(self, height: int, width: int) -> tuple[int, int, int, int, int]:
        latent_height = height // self.video_pipeline.vae_scale_factor
        latent_width = width // self.video_pipeline.vae_scale_factor
        latent_frames = self.num_frames_padded // self.video_pipeline.video_scale_factor
        if isinstance(self.video_pipeline.vae, CausalVideoAutoencoder):
            latent_frames += 1
        return (
            1,
            self.video_pipeline.transformer.config.in_channels,
            latent_frames,
            latent_height,
            latent_width,
        )

    def make_initial_noise(self, seed: int | None = None) -> torch.Tensor:
        height, width = self._first_pass_dimensions()
        latent_shape = self._latent_shape(height, width)
        generator = torch.Generator(device=str(self.device)).manual_seed(
            self.args.seed if seed is None else seed
        )
        noise = randn_tensor(
            (latent_shape[0], latent_shape[2] * latent_shape[3] * latent_shape[4], latent_shape[1]),
            generator=generator,
            device=self.device,
            dtype=self.transformer_dtype,
        )
        noise = rearrange(
            noise,
            "b (f h w) c -> b c f h w",
            f=latent_shape[2],
            h=latent_shape[3],
            w=latent_shape[4],
        )
        noise = noise * self.video_pipeline.scheduler.init_noise_sigma
        return noise.detach().float()

    def generator_after_first_pass_noise(self, seed: int) -> torch.Generator:
        height, width = self._first_pass_dimensions()
        latent_shape = self._latent_shape(height, width)
        generator = torch.Generator(device=str(self.device)).manual_seed(seed)
        noise_shape = (
            latent_shape[0],
            latent_shape[2] * latent_shape[3] * latent_shape[4],
            latent_shape[1],
        )
        _ = randn_tensor(
            noise_shape,
            generator=generator,
            device=self.device,
            dtype=self.transformer_dtype,
        )
        return generator

    def call_uses_stochastic_sampling(self, pass_kwargs: dict) -> bool:
        call_kwargs = dict(self.common_call_kwargs)
        call_kwargs.update(pass_kwargs)
        return bool(call_kwargs.get("stochastic_sampling", False))

    def call_timestep_count(self, pass_kwargs: dict) -> int:
        call_kwargs = dict(self.common_call_kwargs)
        call_kwargs.update(pass_kwargs)
        timesteps = call_kwargs.get("timesteps")
        if timesteps is not None:
            return len(timesteps)
        num_inference_steps = call_kwargs.get("num_inference_steps")
        if num_inference_steps is not None:
            return int(num_inference_steps)
        raise ValueError("Cannot infer denoising step count for full noise optimization")

    def patchified_latent_shape(self, height: int, width: int) -> tuple[int, ...]:
        latent_shape = self._latent_shape(height, width)
        probe = torch.empty(
            latent_shape,
            device=self.device,
            dtype=self.transformer_dtype,
        )
        patchified, _ = self.video_pipeline.patchifier.patchify(latents=probe)
        return tuple(patchified.shape)

    def make_step_noises_for_call(
        self,
        *,
        height: int,
        width: int,
        pass_kwargs: dict,
        generator: torch.Generator,
        trainable: bool,
    ) -> list[torch.Tensor]:
        if not self.call_uses_stochastic_sampling(pass_kwargs):
            return []
        shape = self.patchified_latent_shape(height, width)
        noises = []
        for _ in range(self.call_timestep_count(pass_kwargs)):
            noise = randn_tensor(
                shape,
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )
            noises.append(noise.detach().requires_grad_(trainable))
        return noises

    def make_step_noise_plan(
        self,
        *,
        seed: int,
        trainable: bool,
    ) -> tuple[dict[str, list[torch.Tensor]], list[torch.Tensor]]:
        generator = torch.Generator(device=str(self.device)).manual_seed(seed + 1009)
        plan: dict[str, list[torch.Tensor]] = {}
        params: list[torch.Tensor] = []

        def add_call(key: str, height: int, width: int, pass_kwargs: dict) -> None:
            noises = self.make_step_noises_for_call(
                height=height,
                width=width,
                pass_kwargs=pass_kwargs,
                generator=generator,
                trainable=trainable,
            )
            plan[key] = noises
            if trainable:
                params.extend(noises)

        if self.latent_upsampler is None:
            add_call("single", self.height_padded, self.width_padded, {})
        else:
            first_height, first_width = self._first_pass_dimensions()
            add_call(
                "first",
                first_height,
                first_width,
                self.pipeline_config["first_pass"],
            )
            add_call(
                "second",
                first_height * 2,
                first_width * 2,
                self.pipeline_config["second_pass"],
            )
        return plan, params

    @contextmanager
    def use_step_noises(self, step_noises: list[torch.Tensor] | None):
        if not step_noises:
            yield
            return

        scheduler = self.video_pipeline.scheduler
        original_step = scheduler.step
        state = {"index": 0}

        def patched_step(
            model_output,
            timestep,
            sample,
            return_dict=True,
            stochastic_sampling=False,
            **kwargs,
        ):
            if not stochastic_sampling:
                return original_step(
                    model_output,
                    timestep,
                    sample,
                    return_dict=return_dict,
                    stochastic_sampling=stochastic_sampling,
                    **kwargs,
                )

            index = state["index"]
            state["index"] += 1
            if index >= len(step_noises):
                raise RuntimeError(
                    "Not enough step noises were prepared for stochastic sampling"
                )
            step_noise = step_noises[index].to(
                device=sample.device,
                dtype=sample.dtype,
            )
            if tuple(step_noise.shape) != tuple(sample.shape):
                raise ValueError(
                    f"Step noise shape {tuple(step_noise.shape)} does not match "
                    f"sample shape {tuple(sample.shape)}"
                )

            t_eps = 1e-6
            timesteps_padded = torch.cat(
                [
                    scheduler.timesteps,
                    torch.zeros(1, device=scheduler.timesteps.device),
                ]
            )
            if timestep.ndim == 0:
                lower_mask = timesteps_padded < timestep - t_eps
                lower_timestep = timesteps_padded[lower_mask][0]
                dt = timestep - lower_timestep
            else:
                lower_mask = timesteps_padded[:, None, None] < timestep[None] - t_eps
                lower_timestep = lower_mask * timesteps_padded[:, None, None]
                lower_timestep, _ = lower_timestep.max(dim=0)
                dt = (timestep - lower_timestep)[..., None]

            x0 = sample - timestep[..., None] * model_output
            next_timestep = timestep[..., None] - dt
            prev_sample = scheduler.add_noise(x0, step_noise, next_timestep)

            if not return_dict:
                return (prev_sample,)
            return RectifiedFlowSchedulerOutput(prev_sample=prev_sample)

        scheduler.step = patched_step
        try:
            yield
        finally:
            scheduler.step = original_step

    @contextmanager
    def allow_initial_latents_at_timestep_one(self):
        original_prepare_latents = self.video_pipeline.prepare_latents

        def patched_prepare_latents(
            latents,
            media_items,
            timestep,
            latent_shape,
            dtype,
            device,
            generator,
            vae_per_channel_normalize=True,
        ):
            timestep_value = (
                float(timestep.detach().cpu())
                if torch.is_tensor(timestep)
                else float(timestep)
            )
            if latents is not None and media_items is None and timestep_value >= 1.0:
                if tuple(latents.shape) != tuple(latent_shape):
                    raise ValueError(
                        f"Latents have to be of shape {latent_shape} but are {latents.shape}."
                    )
                return latents.to(device=device, dtype=dtype)
            return original_prepare_latents(
                latents=latents,
                media_items=media_items,
                timestep=timestep,
                latent_shape=latent_shape,
                dtype=dtype,
                device=device,
                generator=generator,
                vae_per_channel_normalize=vae_per_channel_normalize,
            )

        self.video_pipeline.prepare_latents = patched_prepare_latents
        try:
            yield
        finally:
            self.video_pipeline.prepare_latents = original_prepare_latents

    def _call_video_pipeline_latent(
        self,
        *,
        height: int,
        width: int,
        pass_kwargs: dict,
        latents: torch.Tensor | None,
        generator: torch.Generator,
        prompt: str,
        negative_prompt: str,
        step_noises: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        call_kwargs = dict(self.common_call_kwargs)
        call_kwargs.update(pass_kwargs)
        with self.use_step_noises(step_noises):
            result = self.video_pipeline.__class__.__call__.__wrapped__(
                self.video_pipeline,
                prompt=prompt,
                prompt_attention_mask=None,
                negative_prompt=negative_prompt,
                negative_prompt_attention_mask=None,
                skip_layer_strategy=self.skip_layer_strategy,
                generator=generator,
                output_type="latent",
                callback_on_step_end=None,
                height=height,
                width=width,
                num_frames=self.num_frames_padded,
                frame_rate=self.args.fps,
                media_items=None,
                conditioning_items=None,
                is_video=True,
                vae_per_channel_normalize=True,
                image_cond_noise_scale=0.0,
                mixed_precision=(self.precision == "mixed_precision"),
                offload_to_cpu=False,
                device=str(self.device),
                enhance_prompt=False,
                latents=latents,
                **call_kwargs,
            )
        return result.images

    def denoise_to_latents(
        self,
        initial_latents: torch.Tensor,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
        step_noise_plan: dict[str, list[torch.Tensor]] | None,
    ) -> torch.Tensor:
        with self.allow_initial_latents_at_timestep_one():
            generator = self.generator_after_first_pass_noise(seed)
            if self.latent_upsampler is None:
                return self._call_video_pipeline_latent(
                    height=self.height_padded,
                    width=self.width_padded,
                    pass_kwargs={},
                    latents=initial_latents,
                    generator=generator,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    step_noises=(
                        None if step_noise_plan is None else step_noise_plan["single"]
                    ),
                )

            first_height, first_width = self._first_pass_dimensions()
            first_latents = self._call_video_pipeline_latent(
                height=first_height,
                width=first_width,
                pass_kwargs=self.pipeline_config["first_pass"],
                latents=initial_latents,
                generator=generator,
                prompt=prompt,
                negative_prompt=negative_prompt,
                step_noises=(
                    None if step_noise_plan is None else step_noise_plan["first"]
                ),
            )
            upsampled_latents = self.latent_upsampler_forward(first_latents)
            upsampled_latents = differentiable_adain_filter(
                latents=upsampled_latents,
                reference_latents=first_latents,
            )
            return self._call_video_pipeline_latent(
                height=first_height * 2,
                width=first_width * 2,
                pass_kwargs=self.pipeline_config["second_pass"],
                latents=upsampled_latents,
                generator=generator,
                prompt=prompt,
                negative_prompt=negative_prompt,
                step_noises=(
                    None if step_noise_plan is None else step_noise_plan["second"]
                ),
            )

    def latent_upsampler_forward(self, latents: torch.Tensor) -> torch.Tensor:
        if self.latent_upsampler is None:
            raise RuntimeError("No latent upsampler is loaded")
        from ltx_video.models.autoencoders.vae_encode import (
            normalize_latents,
            un_normalize_latents,
        )

        latents = un_normalize_latents(
            latents, self.video_pipeline.vae, vae_per_channel_normalize=True
        )
        upsampled = self.latent_upsampler(latents)
        return normalize_latents(
            upsampled, self.video_pipeline.vae, vae_per_channel_normalize=True
        )

    def decode_latents(
        self,
        latents: torch.Tensor,
        *,
        latent_frames: int | None,
        decode_scale: float,
        decode_timestep: float,
        decode_noise_scale: float,
        resize_to_request: bool,
        generator_seed: int,
    ) -> torch.Tensor:
        if latent_frames is not None:
            latents = latents[:, :, :latent_frames]
        if decode_scale <= 0 or decode_scale > 1:
            raise ValueError("decode_scale must be in the interval (0, 1]")
        if decode_scale < 1:
            _, _, frames, height, width = latents.shape
            latents = F.interpolate(
                latents.float(),
                size=(
                    frames,
                    max(1, round(height * decode_scale)),
                    max(1, round(width * decode_scale)),
                ),
                mode="trilinear",
                align_corners=False,
            )

        timestep = None
        if self.video_pipeline.vae.decoder.timestep_conditioning:
            if decode_noise_scale:
                generator = torch.Generator(device=str(latents.device)).manual_seed(generator_seed)
                noise = torch.randn(
                    latents.shape,
                    generator=generator,
                    device=latents.device,
                    dtype=latents.dtype,
                )
                latents = latents * (1 - decode_noise_scale) + noise * decode_noise_scale
            timestep = torch.tensor([decode_timestep], device=latents.device)

        def decode_fn(input_latents: torch.Tensor) -> torch.Tensor:
            return vae_decode(
                input_latents,
                self.video_pipeline.vae,
                is_video=True,
                vae_per_channel_normalize=True,
                timestep=timestep,
            )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=latents.is_cuda and self.precision == "bfloat16",
        ):
            if (
                self.args.checkpoint_vae_decode
                and torch.is_grad_enabled()
                and latents.requires_grad
            ):
                image = torch.utils.checkpoint.checkpoint(
                    decode_fn,
                    latents,
                    use_reentrant=self.args.vae_checkpoint_reentrant,
                    preserve_rng_state=False,
                )
            else:
                image = decode_fn(latents)
        video = self.video_pipeline.image_processor.postprocess(image, output_type="pt")
        if resize_to_request and video.shape[-2:] != (self.height_padded, self.width_padded):
            bsz, channels, frames, height, width = video.shape
            video = rearrange(video, "b c f h w -> (b f) c h w")
            video = F.interpolate(
                video,
                size=(self.height_padded, self.width_padded),
                mode="bilinear",
                align_corners=False,
            )
            video = rearrange(video, "(b f) c h w -> b c f h w", b=bsz, f=frames)
        return self.crop_to_request(video)

    def crop_to_request(self, video: torch.Tensor) -> torch.Tensor:
        pad_left, pad_right, pad_top, pad_bottom = self.padding
        bottom = -pad_bottom if pad_bottom else video.shape[3]
        right = -pad_right if pad_right else video.shape[4]
        return video[
            :,
            :,
            : self.args.num_frames,
            pad_top:bottom,
            pad_left:right,
        ]

    def sample_reward_frames(self, video: torch.Tensor) -> torch.Tensor:
        count = self.args.reward_frame_sample_count
        if count is None or count <= 0 or video.shape[2] <= count:
            return video
        indices = torch.linspace(0, video.shape[2] - 1, steps=count, device=video.device)
        return video.index_select(2, indices.round().long())

    def reward_context(self):
        if self.args.reward_precision != "amp":
            return nullcontext()
        if not str(self.args.reward_device).startswith("cuda") or not torch.cuda.is_available():
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def compute_reward(
        self,
        name: str,
        reward_fn: Callable,
        video: torch.Tensor,
        prompt_list: list[str],
    ) -> torch.Tensor:
        chunk_size = self.args.reward_temporal_chunk_size
        if (
            chunk_size is None
            or chunk_size <= 0
            or name not in IMAGE_STYLE_REWARDS
            or video.ndim != 5
            or video.shape[2] <= chunk_size
        ):
            return reward_fn(video, prompt_list)

        total = torch.zeros((), device=video.device, dtype=torch.float32)
        total_frames = 0
        for start in range(0, video.shape[2], chunk_size):
            end = min(video.shape[2], start + chunk_size)
            reward = reward_fn(video[:, :, start:end], prompt_list)
            frame_count = end - start
            total = total + reward.float() * frame_count
            total_frames += frame_count
        return total / total_frames

    def sync_cuda(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)


    def optimize(
        self,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        prompt_list: list[str] | None = None,
        seed: int | None = None,
        video_subdir: str | None = None,
        video_filename: str | None = None,
        metrics_filename: str | None = None,
    ) -> None:
        prompt = self.args.prompt if prompt is None else prompt
        negative_prompt = (
            self.args.negative_prompt if negative_prompt is None else negative_prompt
        )
        seed = self.args.seed if seed is None else seed
        prompt_list = [prompt] if prompt_list is None else prompt_list

        noise_optimization = self.args.noise_optimization
        train_step_noises = noise_optimization == "full"
        step_noise_plan, step_noise_params = self.make_step_noise_plan(
            seed=seed,
            trainable=train_step_noises,
        )
        step_noise_count = sum(len(noises) for noises in step_noise_plan.values())
        if noise_optimization == "full" and step_noise_count == 0:
            log(
                "[DNO] noise_optimization=full requested, but stochastic_sampling "
                "is disabled for this LTX config; optimizing initial noise only."
            )
        elif noise_optimization == "full":
            log(
                f"[DNO] Optimizing initial latent noise and {step_noise_count} "
                "stochastic step-noise tensors."
            )
        elif step_noise_count > 0:
            log(
                f"[DNO] Optimizing initial latent noise only; {step_noise_count} "
                "stochastic step-noise tensors are frozen."
            )

        initial_latents = self.make_initial_noise(seed).requires_grad_(True)
        initial_latents_start = initial_latents.detach().clone()
        initial_latents_start_model_dtype = initial_latents_start.to(self.transformer_dtype)
        opt_params = [initial_latents, *step_noise_params]
        optimizer = torch.optim.AdamW(
            opt_params,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        best_reward = float("-inf")
        best_latents = None
        metrics = []
        stem = safe_stem(prompt)
        start = time.perf_counter()

        log(f"[DNO] Optimizing first-pass initial latent noise for seed={seed}")
        for iteration in range(self.args.dno_steps):
            self.sync_cuda()
            iteration_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            save_on_cpu = (
                torch.autograd.graph.save_on_cpu(pin_memory=True)
                if self.args.save_activations_on_cpu
                else nullcontext()
            )
            with save_on_cpu:
                final_latents = self.denoise_to_latents(
                    initial_latents,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    step_noise_plan=step_noise_plan,
                )
                reward_video = self.decode_latents(
                    final_latents,
                    latent_frames=self.args.reward_latent_frames,
                    decode_scale=self.args.reward_decode_scale,
                    decode_timestep=self.args.reward_decode_timestep,
                    decode_noise_scale=self.args.reward_decode_noise_scale,
                    resize_to_request=True,
                    generator_seed=seed + 3009,
                ).to(self.args.reward_device)
                reward_video = self.sample_reward_frames(reward_video)

            if iteration == 0:
                log(
                    "[DNO] Reward video tensor "
                    f"shape={tuple(reward_video.shape)} "
                    f"dtype={reward_video.dtype} "
                    f"range=({float(reward_video.min().detach().cpu()):.4f}, "
                    f"{float(reward_video.max().detach().cpu()):.4f})"
                )

            total_loss = torch.zeros((), device=self.args.reward_device)
            total_reward = 0.0
            reward_values = {}
            for name, (reward_fn, weight) in self.reward_fns.items():
                with self.reward_context():
                    reward = self.compute_reward(name, reward_fn, reward_video, prompt_list)
                reward_scalar = float(reward.detach().cpu())
                reward_values[name] = reward_scalar
                total_reward += reward_scalar * weight
                total_loss = total_loss - reward * weight

            if total_reward > best_reward:
                best_reward = total_reward
                best_latents = final_latents.detach().cpu()
            step_latents = (
                final_latents.detach().cpu() if self.args.save_step_videos else None
            )

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                opt_params, self.args.max_grad_norm
            )
            optimizer.step()
            with torch.no_grad():
                fp32_delta = (initial_latents - initial_latents_start).abs()
                model_dtype_delta = (
                    initial_latents.to(self.transformer_dtype).float()
                    - initial_latents_start_model_dtype.float()
                ).abs()
                model_dtype_changed = (
                    initial_latents.to(self.transformer_dtype)
                    != initial_latents_start_model_dtype
                )
            self.sync_cuda()
            iteration_wall_time = time.perf_counter() - iteration_start

            metrics.append(
                {
                    "iteration": iteration,
                    "total_reward": total_reward,
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "wall_time_seconds": iteration_wall_time,
                    "noise_delta_fp32_max_abs": float(fp32_delta.max().detach().cpu()),
                    "noise_delta_fp32_mean_abs": float(fp32_delta.mean().detach().cpu()),
                    "noise_delta_model_dtype_max_abs": float(
                        model_dtype_delta.max().detach().cpu()
                    ),
                    "noise_delta_model_dtype_mean_abs": float(
                        model_dtype_delta.mean().detach().cpu()
                    ),
                    "noise_delta_model_dtype_changed_fraction": float(
                        model_dtype_changed.float().mean().detach().cpu()
                    ),
                    **reward_values,
                }
            )
            reward_text = ", ".join(
                f"{name}={value:.4f}" for name, value in reward_values.items()
            )
            log(
                f"[DNO] iter={iteration} total_reward={total_reward:.4f} "
                f"grad_norm={float(grad_norm):.4f} "
                f"wall_time={iteration_wall_time:.2f}s {reward_text}"
            )

            if step_latents is not None:
                step_dir = self.video_dir / video_subdir if video_subdir else self.video_dir
                step_path = step_dir / (
                    f"{stem}_{seed}_{self.args.height}x"
                    f"{self.args.width}x{self.args.num_frames}_step{iteration:03d}.mp4"
                )
                with torch.no_grad():
                    step_video = self.decode_latents(
                        step_latents.to(self.device),
                        latent_frames=None,
                        decode_scale=1.0,
                        decode_timestep=self.args.final_decode_timestep,
                        decode_noise_scale=self.args.final_decode_noise_scale,
                        resize_to_request=True,
                        generator_seed=seed + 4009,
                    )
                save_video(step_video, step_path, self.args.fps)
                log(f"[DNO] Saved step video: {step_path}")

        if self.args.save_metrics:
            metrics_path = self.metrics_dir / (
                metrics_filename or f"{stem}_metrics.json"
            )
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            log(f"[DNO] Saved metrics: {metrics_path}")

        if self.args.save_best_video and best_latents is not None:
            best_dir = self.video_dir / video_subdir if video_subdir else self.video_dir
            best_path = best_dir / (
                video_filename
                or f"{stem}_{seed}_{self.args.height}x"
                f"{self.args.width}x{self.args.num_frames}_best.mp4"
            )
            with torch.no_grad():
                best_video = self.decode_latents(
                    best_latents.to(self.device),
                    latent_frames=None,
                    decode_scale=1.0,
                    decode_timestep=self.args.final_decode_timestep,
                    decode_noise_scale=self.args.final_decode_noise_scale,
                    resize_to_request=True,
                    generator_seed=seed + 4009,
                )
            save_video(best_video, best_path, self.args.fps)
            log(f"[DNO] Saved best video: {best_path}")
        log(f"[DNO] Best reward: {best_reward:.4f}")
        log(f"[DNO] Optimization time: {time.perf_counter() - start:.1f}s")

    def generate_sample(
        self,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
        video_subdir: str | None = None,
        video_filename: str | None = None,
    ) -> None:
        prompt = self.args.prompt if prompt is None else prompt
        negative_prompt = (
            self.args.negative_prompt if negative_prompt is None else negative_prompt
        )
        seed = self.args.seed if seed is None else seed
        stem = safe_stem(prompt)
        start = time.perf_counter()

        log(f"[LTX] Generating vanilla sample for seed={seed}")
        with torch.no_grad():
            initial_latents = self.make_initial_noise(seed)
            final_latents = self.denoise_to_latents(
                initial_latents,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                step_noise_plan=None,
            )
            video = self.decode_latents(
                final_latents,
                latent_frames=None,
                decode_scale=1.0,
                decode_timestep=self.args.final_decode_timestep,
                decode_noise_scale=self.args.final_decode_noise_scale,
                resize_to_request=True,
                generator_seed=seed + 4009,
            )

        video_dir = self.video_dir / video_subdir if video_subdir else self.video_dir
        video_path = video_dir / (
            video_filename
            or f"{stem}_{seed}_{self.args.height}x"
            f"{self.args.width}x{self.args.num_frames}.mp4"
        )
        save_video(video, video_path, self.args.fps)
        log(f"[LTX] Saved vanilla video: {video_path}")
        log(f"[LTX] Generation time: {time.perf_counter() - start:.1f}s")

    def find_existing_compbench_video(self, filename: str, category: str) -> str | None:
        if not self.args.skip_existing:
            return None
        return existing_video_path(
            run_root=self.args.run_root,
            rank_video_dir=self.video_dir,
            filename=filename,
            subdir=category,
            scope=self.args.skip_existing_scope,
            min_bytes=self.args.min_existing_video_bytes,
        )

    def run_compbench(self) -> None:
        tasks = build_compbench_tasks(
            self.args.prompts_dir,
            iter_compbench_prompt_file,
        )
        seed_map = load_seed_map(self.args.seed_file)
        seed_log_path = self.output_dir / "seeds.txt"
        processed = 0

        def run_task(task: tuple[int, str, str, str, list[str]]) -> bool:
            task_id, sample_id, prompt, category, negative_prompts = task
            filename = compbench_video_filename(sample_id)
            safe_category = sanitize_filename_component(category, max_len=80)
            existing = self.find_existing_compbench_video(filename, safe_category)
            if existing:
                log(
                    f"[Rank {self.args.rank}] Skipping existing CompBench task "
                    f"{task_id + 1}/{len(tasks)}: {existing}"
                )
                return False

            key = f"{category}:{sample_id}-{prompt}"
            seed = seed_map.get(key, random.randint(0, MAX_SEED))
            seed_everything(seed)
            with seed_log_path.open("a", encoding="utf-8") as seed_log:
                seed_log.write(f"{key}:{seed}\n")

            log(
                f"[Rank {self.args.rank}] CompBench task "
                f"{task_id + 1}/{len(tasks)} ({category} id={sample_id})"
            )
            if self.args.use_dno:
                prompt_list = [prompt]
                if self.args.use_neg_prompt:
                    prompt_list.extend(negative_prompts)
                    log("[DNO] Using CompBench negative prompts for reward optimization")

                self.optimize(
                    prompt=prompt,
                    negative_prompt=self.args.negative_prompt,
                    prompt_list=prompt_list,
                    seed=seed,
                    video_subdir=safe_category,
                    video_filename=filename,
                    metrics_filename=(
                        f"{safe_category}_{sample_id}_{safe_stem(prompt, max_len=80)}.json"
                    ),
                )
            else:
                self.generate_sample(
                    prompt=prompt,
                    negative_prompt=self.args.negative_prompt,
                    seed=seed,
                    video_subdir=safe_category,
                    video_filename=filename,
                )
            return True

        if self.args.shard_strategy == "dynamic":
            queue_dir = Path(self.args.run_root) / "_queue"
            while True:
                task_id = claim_next_task(queue_dir, "compbench", len(tasks))
                if task_id is None:
                    break
                if run_task(tasks[task_id]):
                    processed += 1
                if (
                    self.args.max_compbench_samples is not None
                    and processed >= self.args.max_compbench_samples
                ):
                    log(
                        f"[Rank {self.args.rank}] Reached "
                        f"max_compbench_samples={self.args.max_compbench_samples}"
                    )
                    break
        else:
            for task in tasks:
                task_id = task[0]
                if task_id % self.args.world_size != self.args.rank:
                    continue
                if run_task(task):
                    processed += 1
                if (
                    self.args.max_compbench_samples is not None
                    and processed >= self.args.max_compbench_samples
                ):
                    log(
                        f"[Rank {self.args.rank}] Reached "
                        f"max_compbench_samples={self.args.max_compbench_samples}"
                    )
                    break

        log(f"[Rank {self.args.rank}] Completed {processed} CompBench samples.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("LTXV-2B distilled Direct Noise Optimization")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", "--negative_prompt", dest="negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--pipeline-config", "--pipeline_config", dest="pipeline_config", type=str, default="configs/ltxv-2b-0.9.8-distilled-no-enhance.yaml")
    parser.add_argument("--output-path", "--output_path", dest="output_path", type=str, default=f"outputs/dno-{datetime.today().strftime('%Y-%m-%d')}")
    parser.add_argument(
        "--prompt-file", "--prompt_file", dest="prompt_file", type=str, default=None,
        help="Small UTF-8 TXT/JSON prompt list; bypasses benchmark mode.",
    )
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", "--num_frames", dest="num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--run-compbench", "--run_compbench", dest="run_compbench", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--prompts-dir", "--prompts_dir", dest="prompts_dir", type=str, default="comp-t2v-prompts")
    parser.add_argument("--save-path", "--save_path", dest="save_path", type=str, default="ltx_compbench_dno")
    parser.add_argument("--run-id", "--run_id", dest="run_id", type=str, default=None)
    parser.add_argument("--world-size", "--world_size", dest="world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--shard-strategy", "--shard_strategy", dest="shard_strategy", choices=["static", "dynamic"], default="static")
    parser.add_argument("--log-all-ranks", "--log_all_ranks", dest="log_all_ranks", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--skip-existing", "--skip_existing", dest="skip_existing", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--skip-existing-scope", "--skip_existing_scope", dest="skip_existing_scope", choices=["rank", "run"], default="run")
    parser.add_argument("--min-existing-video-bytes", "--min_existing_video_bytes", dest="min_existing_video_bytes", type=int, default=1)
    parser.add_argument("--seed-file", "--seed_file", dest="seed_file", type=str, default="comp-t2v-prompts/compbench_seed.txt")
    parser.add_argument("--max-compbench-samples", "--max_compbench_samples", dest="max_compbench_samples", type=int, default=None)
    parser.add_argument("--use-dno", "--use_dno", dest="use_dno", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--noise-optimization", "--noise_optimization", dest="noise_optimization", choices=["initial", "full"], default="initial")
    parser.add_argument("--use-neg-prompt", "--use_neg_prompt", dest="use_neg_prompt", type=str2bool, nargs="?", const=True, default=False)

    parser.add_argument("--dno-steps", "--dno_steps", dest="dno_steps", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", "--max_grad_norm", dest="max_grad_norm", type=float, default=0.1)
    parser.add_argument("--reward-fns", "--reward_fns", dest="reward_fns", nargs="+", default=["img_reward", "viclip"])
    parser.add_argument("--reward-weights", "--reward_weights", dest="reward_weights", nargs="+", type=float, default=[1.0, 1.0])
    parser.add_argument("--reward-precision", "--reward_precision", dest="reward_precision", choices=["auto", "fp32", "fp16", "bf16", "amp"], default="fp32")
    parser.add_argument("--reward-device", "--reward_device", dest="reward_device", type=str, default="cuda:0")
    parser.add_argument("--reward-latent-frames", "--reward_latent_frames", dest="reward_latent_frames", type=int, default=None)
    parser.add_argument("--reward-frame-sample-count", "--reward_frame_sample_count", dest="reward_frame_sample_count", type=int, default=16)
    parser.add_argument("--reward-temporal-chunk-size", "--reward_temporal_chunk_size", dest="reward_temporal_chunk_size", type=int, default=None)
    parser.add_argument("--reward-decode-scale", "--reward_decode_scale", dest="reward_decode_scale", type=float, default=1.0)
    parser.add_argument("--reward-decode-timestep", "--reward_decode_timestep", dest="reward_decode_timestep", type=float, default=0.0)
    parser.add_argument("--reward-decode-noise-scale", "--reward_decode_noise_scale", dest="reward_decode_noise_scale", type=float, default=0.0)
    parser.add_argument("--final-decode-timestep", "--final_decode_timestep", dest="final_decode_timestep", type=float, default=0.05)
    parser.add_argument("--final-decode-noise-scale", "--final_decode_noise_scale", dest="final_decode_noise_scale", type=float, default=0.025)
    parser.add_argument("--save-metrics", "--save_metrics", dest="save_metrics", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--save-best-video", "--save_best_video", "--save-best", "--save_best", dest="save_best_video", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--no-save-best-video", "--no_save_best_video", dest="save_best_video", action="store_false")
    parser.add_argument("--save-step-videos", "--save_step_videos", "--save-step-video", "--save_step_video", dest="save_step_videos", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--no-save-step-videos", "--no_save_step_videos", dest="save_step_videos", action="store_false")
    parser.add_argument("--gradient-checkpointing", "--gradient_checkpointing", dest="gradient_checkpointing", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--no-gradient-checkpointing", "--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--checkpoint-vae-decode", "--checkpoint_vae_decode", dest="checkpoint_vae_decode", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--no-checkpoint-vae-decode", "--no_checkpoint_vae_decode", dest="checkpoint_vae_decode", action="store_false")
    parser.add_argument("--vae-checkpoint-reentrant", "--vae_checkpoint_reentrant", dest="vae_checkpoint_reentrant", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--no-vae-checkpoint-reentrant", "--no_vae_checkpoint_reentrant", dest="vae_checkpoint_reentrant", action="store_false")
    parser.add_argument("--save-activations-on-cpu", "--save_activations_on_cpu", dest="save_activations_on_cpu", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--no-save-activations-on-cpu", "--no_save_activations_on_cpu", dest="save_activations_on_cpu", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_compbench:
        run_id = args.run_id or time.strftime("%Y-%m-%d-%H-%M-%S")
        args.run_root = str(Path(args.save_path) / run_id)
        args.output_path = str(Path(args.run_root) / f"shard_{args.rank}")
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
        args_path = Path(args.output_path) / "args.json"
        args_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    else:
        args.run_root = str(Path(args.output_path))

    seed_everything(args.seed)
    runner = LTXDistilledDNO(args)
    if args.run_compbench:
        runner.run_compbench()
    else:
        prompts = load_user_prompts(args.prompt, args.prompt_file)
        for index, prompt in enumerate(prompts):
            seed = (args.seed + index) % MAX_SEED
            seed_everything(seed)
            log(f"[LTX] Running prompt {index + 1}/{len(prompts)} with seed={seed}")
            if args.use_dno:
                runner.optimize(prompt=prompt, seed=seed)
            else:
                runner.generate_sample(prompt=prompt, seed=seed)


if __name__ == "__main__":
    main()
