"""Inference and Direct Noise Optimization (DNO) for FastWan2.1-T2V-1.3B."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
import warnings
from dataclasses import replace
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
os.environ.setdefault("FASTVIDEO_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.pop("TORCH_NCCL_AVOID_RECORD_STREAMS", None)
warnings.filterwarnings("ignore", message=r"VideoGenerator\.generate_video\(\.\.\.\) is deprecated.*")

from reward_fn import get_reward_fn
from benchmark_sharding import (
    build_compbench_tasks,
    build_vbench_tasks,
    claim_next_task,
    compbench_video_filename,
    existing_video_path,
    vbench_video_filename,
)
from fastwan_dno_utils import (
    MAX_SEED,
    WanTinyTAEDecoder,
    context_aware_checkpoint,
    ensure_single_process_dist_env,
    iter_compbench_prompt_file,
    iter_vbench_prompt_file,
    load_seed_map,
    log,
    safe_stem,
    save_video,
    str2bool,
)
from prompt_io import load_user_prompts

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress display
    class _NullProgress:
        def __init__(self, iterable=None, **kwargs):
            del kwargs
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or [])

        def update(self, n: int = 1) -> None:
            del n

        def close(self) -> None:
            pass

    def tqdm(iterable=None, **kwargs):
        return _NullProgress(iterable, **kwargs)

DEFAULT_MODEL = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
OUTPUT_PATH = "test_output_fastwant2v"
DEFAULT_PROMPT = (
    "A majestic lion strides across the golden savanna, its powerful frame "
    "glistening under the warm afternoon sun. The tall grass ripples gently in "
    "the breeze. Low angle, steady tracking shot, cinematic."
)
EXACT_FRAME_REWARD_COUNTS = {
    "viclip": 16,
    "intervid2": 16,
}


def load_fastvideo_runtime() -> None:
    """Import GPU-bound FastVideo modules after CLI parsing.

    FastVideo's kernel package probes the active Triton/CUDA driver at import
    time. Delaying these imports keeps ``--help`` usable on login and CPU-only
    nodes while preserving the same runtime imports for actual generation.
    """
    global VideoGenerator, SamplingParam, FastVideoArgs, ForwardBatch
    global build_pipeline, DecodingStage, get_local_torch_device
    global PRECISION_TO_TYPE, align_to, shallow_asdict

    from fastvideo import VideoGenerator as _VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam as _SamplingParam
    from fastvideo.fastvideo_args import FastVideoArgs as _FastVideoArgs
    from fastvideo.pipelines import ForwardBatch as _ForwardBatch
    from fastvideo.pipelines import build_pipeline as _build_pipeline
    from fastvideo.pipelines.stages.decoding import DecodingStage as _DecodingStage
    from fastvideo.distributed import get_local_torch_device as _get_local_torch_device
    from fastvideo.utils import PRECISION_TO_TYPE as _PRECISION_TO_TYPE
    from fastvideo.utils import align_to as _align_to
    from fastvideo.utils import shallow_asdict as _shallow_asdict

    VideoGenerator = _VideoGenerator
    SamplingParam = _SamplingParam
    FastVideoArgs = _FastVideoArgs
    ForwardBatch = _ForwardBatch
    build_pipeline = _build_pipeline
    DecodingStage = _DecodingStage
    get_local_torch_device = _get_local_torch_device
    PRECISION_TO_TYPE = _PRECISION_TO_TYPE
    align_to = _align_to
    shallow_asdict = _shallow_asdict


def make_sampling_param(args: argparse.Namespace) -> SamplingParam:
    sampling_param = SamplingParam.from_pretrained(args.model_path)
    sampling_param.height = args.height
    sampling_param.width = args.width
    sampling_param.num_frames = args.num_frames
    sampling_param.fps = args.fps
    sampling_param.guidance_scale = args.guidance_scale
    if getattr(args, "num_inference_steps", None) is not None:
        sampling_param.num_inference_steps = args.num_inference_steps
    sampling_param.seed = args.seed
    sampling_param.output_path = args.output_path
    sampling_param.save_video = True
    sampling_param.return_frames = False
    return sampling_param


def run_vanilla(args: argparse.Namespace) -> None:
    log(f"Loading model: {args.model_path}")
    log(f"Using VSA sparsity: {args.vsa_sparsity}")
    t0 = time.perf_counter()
    generator = VideoGenerator.from_pretrained(
        args.model_path,
        num_gpus=1,
        use_fsdp_inference=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        VSA_sparsity=args.vsa_sparsity,
    )
    log(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    prompts = load_user_prompts(args.prompt, args.prompt_file)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, prompt in enumerate(prompts):
        seed = (args.seed + index) % MAX_SEED
        sampling_param = make_sampling_param(args)
        sampling_param.seed = seed
        stem = f"{index:03d}_{safe_stem(prompt)}"
        output_file = output_dir / f"{stem}.mp4"
        sampling_param.output_path = str(output_file)

        log(f"Generating prompt {index + 1}/{len(prompts)} with seed={seed} ...")
        t1 = time.perf_counter()
        video = generator.generate_video(
            prompt,
            output_path=str(output_file),
            save_video=True,
            return_frames=False,
            sampling_param=sampling_param,
        )
        elapsed = time.perf_counter() - t1
        log(f"Video generated in {elapsed:.1f}s")
        log(f"Saved video: {video.get('video_path')}")
        log(f"Output size: {video.get('size')}")
        log(f"Reported generation time: {video.get('generation_time'):.1f}s")


class FastWanDNO:
    """Direct noise optimization runner for the FastWan DMD pipeline."""

    def __init__(self, args: argparse.Namespace, *, build_rewards: bool = True):
        self.args = args
        ensure_single_process_dist_env()
        self.device = get_local_torch_device()
        self.output_dir = Path(args.output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        benchmark_mode = getattr(args, "run_vbench", False) or getattr(args, "run_compbench", False)
        self.video_dir = self.output_dir / ("all_videos" if benchmark_mode else "dno_videos")
        self.metrics_dir = self.output_dir / ("metrics" if benchmark_mode else "dno_metrics")
        self.video_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)

        self.fastvideo_args = FastVideoArgs.from_kwargs(
            model_path=args.model_path,
            num_gpus=1,
            use_fsdp_inference=False,
            text_encoder_cpu_offload=args.text_encoder_cpu_offload,
            pin_cpu_memory=False,
            dit_cpu_offload=False,
            dit_layerwise_offload=False,
            vae_cpu_offload=False,
            disable_autocast=False,
            VSA_sparsity=args.vsa_sparsity,
            enable_stage_verification=False,
        )
        log(f"Using VSA sparsity: {self.fastvideo_args.VSA_sparsity}")
        self.pipeline = build_pipeline(self.fastvideo_args)
        self.pipeline.post_init()
        self._freeze_modules()
        self.reward_decoder = self._build_reward_decoder()
        self.reward_fns = self._build_reward_fns() if build_rewards else {}

    def _freeze_modules(self) -> None:
        for module in self.pipeline.modules.values():
            if isinstance(module, torch.nn.Module):
                module.eval()
                module.requires_grad_(False)
        if self.args.dno_gradient_checkpointing:
            transformer = self.pipeline.get_module("transformer", None)
            if transformer is not None and hasattr(transformer, "gradient_checkpointing"):
                transformer.gradient_checkpointing = True
                transformer._gradient_checkpointing_func = context_aware_checkpoint

    def _build_reward_fns(self) -> dict[str, tuple]:
        reward_fns = {}
        if len(self.args.reward_fns) != len(self.args.reward_weights):
            raise ValueError("--reward-fns and --reward-weights must have the same length")
        for name, weight in zip(self.args.reward_fns, self.args.reward_weights):
            precision = self.args.reward_precision
            if precision == "auto":
                precision = "fp16" if name == "videoscore" else "fp32"
            log(f"Loading reward model: {name} ({precision})")
            reward_fns[name] = (get_reward_fn(name, precision=precision, device=self.args.reward_device), weight)
        return reward_fns

    def _resolve_reward_decoder_path(self) -> Path:
        candidates = []
        if self.args.reward_decoder_path:
            candidates.append(Path(self.args.reward_decoder_path))
        env_path = os.environ.get("FASTWAN_TINY_VAE_PATH")
        if env_path:
            candidates.append(Path(env_path))
        filename = f"{self.args.reward_decoder}.safetensors"
        candidates.append(Path("checkpoints/vae") / filename)
        for path in candidates:
            if path.exists():
                return path
        formatted = "\n  ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Could not find {self.args.reward_decoder} checkpoint. Tried:\n  {formatted}\n"
            "Pass --reward-decoder-path or set FASTWAN_TINY_VAE_PATH."
        )

    def _build_reward_decoder(self) -> WanTinyTAEDecoder | None:
        if self.args.reward_decoder == "wan":
            return None

        decoder_path = self._resolve_reward_decoder_path()
        decoder_device = self.args.reward_decoder_device or self.args.reward_device
        if self.args.reward_tiny_decoder_need_scaled == "auto":
            need_scaled = self.args.reward_decoder == "lighttaew2_1"
        else:
            need_scaled = str2bool(self.args.reward_tiny_decoder_need_scaled)
        dtype = PRECISION_TO_TYPE[self.args.reward_tiny_decoder_precision]
        log(
            f"Loading reward decoder: {self.args.reward_decoder} "
            f"({decoder_path}, {self.args.reward_tiny_decoder_precision}, device={decoder_device})"
        )
        decoder = WanTinyTAEDecoder(
            decoder_path,
            dtype=dtype,
            need_scaled=need_scaled,
            parallel=self.args.reward_tiny_decoder_parallel,
        )
        decoder.eval().to(decoder_device)
        return decoder

    def prepare_batch(self, sampling_param: SamplingParam, prompt: str) -> ForwardBatch:
        target_height = align_to(sampling_param.height, 16)
        target_width = align_to(sampling_param.width, 16)
        latent_size = [
            (sampling_param.num_frames - 1) // 4 + 1,
            sampling_param.height // 8,
            sampling_param.width // 8,
        ]
        n_tokens = latent_size[0] * latent_size[1] * latent_size[2]
        sampling_param = copy.deepcopy(sampling_param)
        sampling_param.prompt = prompt.strip()
        sampling_param.height = target_height
        sampling_param.width = target_width

        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            eta=0.0,
            n_tokens=n_tokens,
            VSA_sparsity=self.fastvideo_args.VSA_sparsity,
        )
        for stage_name in (
            "input_validation_stage",
            "prompt_encoding_stage",
            "conditioning_stage",
            "timestep_preparation_stage",
            "latent_preparation_stage",
        ):
            batch = getattr(self.pipeline, stage_name)(batch, self.fastvideo_args)
        if batch.latents is None:
            raise RuntimeError("Latent preparation did not produce latents")
        return batch

    def decode_differentiable(self, latents: torch.Tensor, *, use_reward_decoder: bool = True) -> torch.Tensor:
        decoding_stage = self.pipeline.decoding_stage
        if not isinstance(decoding_stage, DecodingStage):
            raise TypeError(f"Expected DecodingStage, got {type(decoding_stage)}")

        if self.args.reward_decode_scale <= 0 or self.args.reward_decode_scale > 1:
            raise ValueError("--reward-decode-scale must be in the interval (0, 1]")
        if self.args.reward_decode_scale < 1:
            _, _, latent_frames, latent_height, latent_width = latents.shape
            target_height = max(1, round(latent_height * self.args.reward_decode_scale))
            target_width = max(1, round(latent_width * self.args.reward_decode_scale))
            latents = F.interpolate(
                latents.float(),
                size=(latent_frames, target_height, target_width),
                mode="trilinear",
                align_corners=False,
            )

        if use_reward_decoder and self.reward_decoder is not None:
            decoder_device = next(self.reward_decoder.parameters()).device
            video = self.reward_decoder.decode(latents.to(decoder_device))
            return video.clamp(0, 1)

        vae = decoding_stage.vae.to(self.device)
        latents = latents.to(self.device)
        vae_dtype = PRECISION_TO_TYPE[self.fastvideo_args.pipeline_config.vae_precision]
        autocast_enabled = vae_dtype != torch.float32 and not self.fastvideo_args.disable_autocast
        latents = decoding_stage._denormalize_latents(latents)

        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=autocast_enabled):
            if self.fastvideo_args.pipeline_config.vae_tiling:
                vae.enable_tiling()
            if not autocast_enabled:
                latents = latents.to(vae_dtype)
            if self.args.checkpoint_vae_decode:
                video = checkpoint(vae.decode, latents, use_reentrant=False)
            else:
                video = vae.decode(latents)
        return (video / 2 + 0.5).clamp(0, 1)

    def uniform_sample_frames(self, video: torch.Tensor, frame_count: int | None) -> torch.Tensor:
        if frame_count is None or frame_count <= 0 or video.shape[2] <= frame_count:
            return video
        frame_indices = torch.linspace(
            0,
            video.shape[2] - 1,
            steps=frame_count,
            device=video.device,
        ).round().long()
        return video.index_select(2, frame_indices)

    def sample_reward_frames(self, video: torch.Tensor) -> torch.Tensor:
        return self.uniform_sample_frames(video, self.args.reward_frame_sample_count)

    def reward_video_for(self, reward_name: str, video: torch.Tensor, default_video: torch.Tensor) -> torch.Tensor:
        frame_count = EXACT_FRAME_REWARD_COUNTS.get(reward_name)
        if frame_count is None:
            return default_video
        if video.shape[2] < frame_count:
            raise ValueError(
                f"{reward_name} reward requires at least {frame_count} decoded frames, "
                f"but got {video.shape[2]}"
            )
        return self.uniform_sample_frames(video, frame_count)

    def reward_autocast_context(self):
        if self.args.reward_precision != "amp":
            return nullcontext()
        if not str(self.args.reward_device).startswith("cuda") or not torch.cuda.is_available():
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def denoise_differentiable(
        self,
        prepared_batch: ForwardBatch,
        initial_latents: torch.Tensor,
        step_noises: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = replace(
            prepared_batch,
            latents=initial_latents,
            extra={**prepared_batch.extra, "dno_step_noises": step_noises},
        )
        output_batch = self.pipeline.denoising_stage(batch, self.fastvideo_args)
        if output_batch.latents is None:
            raise RuntimeError("Denoising did not produce latents")
        return output_batch.latents

    def optimize(
        self,
        prompt: str,
        sampling_param: SamplingParam,
        *,
        prompt_list: list[str] | None = None,
        subdir: str | None = None,
        filename: str | None = None,
        metrics_name: str | None = None,
    ) -> None:
        prepared_batch = self.prepare_batch(sampling_param, prompt)
        latent_shape = tuple(prepared_batch.latents.shape)
        base_latents = prepared_batch.latents.detach().float().to(self.device)
        num_denoising_steps = len(self.fastvideo_args.pipeline_config.dmd_denoising_steps)
        if num_denoising_steps < 1:
            raise ValueError("FastWan DMD requires at least one denoising step")

        cpu_generator = torch.Generator("cpu").manual_seed(sampling_param.seed + 1009)
        step_noise_values = [
            torch.randn(latent_shape, generator=cpu_generator, dtype=torch.float32).to(self.device)
            for _ in range(max(num_denoising_steps - 1, 0))
        ]
        fixed_step_noises = (
            torch.stack(step_noise_values, dim=0) if step_noise_values else torch.empty(0, *latent_shape,
                                                                                       device=self.device)
        )

        if self.args.noise_optimization == "initial":
            initial_latents = base_latents.detach().clone().requires_grad_(True)
            step_noises = fixed_step_noises.detach()
            opt_params = [initial_latents]
            log("[DNO] Optimizing initial latent noise only")
        elif self.args.noise_optimization == "full":
            initial_latents = base_latents.detach().clone().requires_grad_(True)
            step_noises = fixed_step_noises.detach().clone().requires_grad_(True)
            opt_params = [initial_latents, step_noises]
            log("[DNO] Optimizing initial and stepwise trajectory noises")
        else:
            raise ValueError(f"Unknown noise_optimization mode: {self.args.noise_optimization}")
        if self.args.reward_frame_sample_count is not None and self.args.reward_frame_sample_count > 0:
            log(f"[DNO] Uniformly sampling {self.args.reward_frame_sample_count} decoded frames for reward")

        optimizer = torch.optim.AdamW(opt_params, lr=self.args.lr)
        best_reward = float("-inf")
        best_latents = None
        metrics = []
        prompt_list = prompt_list or [prompt]
        start_time = time.perf_counter()
        initial_start = initial_latents.detach().clone()
        step_start = step_noises.detach().clone()
        video_dir = self.video_dir / subdir if subdir else self.video_dir
        metrics_dir = self.metrics_dir / subdir if subdir else self.metrics_dir
        video_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        for iteration in range(self.args.dno_steps):
            optimizer.zero_grad(set_to_none=True)
            save_on_cpu = (
                torch.autograd.graph.save_on_cpu(pin_memory=True)
                if self.args.dno_save_activations_on_cpu else nullcontext()
            )
            with save_on_cpu:
                final_latents = self.denoise_differentiable(prepared_batch, initial_latents, step_noises)
            reward_latents = final_latents
            if self.args.reward_latent_frames is not None:
                reward_latents = reward_latents[:, :, :self.args.reward_latent_frames]
            video = self.decode_differentiable(reward_latents).to(self.args.reward_device)
            default_reward_video = self.sample_reward_frames(video)

            total_loss = torch.zeros((), device=self.args.reward_device)
            total_reward = 0.0
            reward_dict = {}
            for name, (reward_fn, weight) in self.reward_fns.items():
                reward_video = self.reward_video_for(name, video, default_reward_video)
                with self.reward_autocast_context():
                    reward = reward_fn(reward_video, prompt_list)
                reward_value = float(reward.detach().cpu())
                reward_dict[name] = reward_value
                total_reward += reward_value * weight
                total_loss = total_loss - reward * weight

            if total_reward > best_reward:
                best_reward = total_reward
                best_latents = final_latents.detach().cpu()

            total_loss.backward()
            initial_grad_norm = float(initial_latents.grad.detach().float().norm().cpu())
            step_grad_norm = (
                float(step_noises.grad.detach().float().norm().cpu())
                if step_noises.requires_grad and step_noises.grad is not None
                else 0.0
            )
            total_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(opt_params, self.args.max_grad_norm).detach().cpu()
            )
            optimizer.step()

            initial_update_norm = float(
                (initial_latents.detach() - initial_start).float().norm().cpu()
            )
            step_update_norm = (
                float((step_noises.detach() - step_start).float().norm().cpu())
                if step_noises.requires_grad
                else 0.0
            )

            metrics.append({
                "iteration": iteration,
                "total_reward": total_reward,
                "initial_grad_norm": initial_grad_norm,
                "step_grad_norm": step_grad_norm,
                "total_grad_norm_before_clip": total_grad_norm,
                "initial_update_norm": initial_update_norm,
                "step_update_norm": step_update_norm,
                **reward_dict,
            })
            reward_text = ", ".join(f"{name}={value:.4f}" for name, value in reward_dict.items())
            log(
                f"[DNO] iter={iteration} total_reward={total_reward:.4f} {reward_text} "
                f"grad={total_grad_norm:.6g} update={initial_update_norm + step_update_norm:.6g}"
            )

            if self.args.save_step_video:
                step_stem = safe_stem(Path(filename).stem if filename else prompt)
                step_path = video_dir / f"{step_stem}_step_{iteration:03d}.mp4"
                save_video(video.detach().cpu(), step_path, sampling_param.fps)

        if self.args.save_best_video and best_latents is not None:
            best_path = video_dir / (filename or f"{safe_stem(prompt)}_best.mp4")
            with torch.no_grad():
                best_video = self.decode_differentiable(best_latents.to(self.device), use_reward_decoder=False).cpu()
            save_video(best_video, best_path, sampling_param.fps)
            log(f"[DNO] Saved best video: {best_path}")
        if self.args.save_metrics:
            metrics_path = metrics_dir / (metrics_name or f"{safe_stem(prompt)}_metrics.json")
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            log(f"[DNO] Saved metrics: {metrics_path}")
        log(f"[DNO] Best reward: {best_reward:.4f}")
        log(f"[DNO] Optimization time: {time.perf_counter() - start_time:.1f}s")


class FastWanBenchmarkRunner:
    """VBench / T2V-CompBench generation loop for FastWan."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_path)
        self.video_dir = self.output_dir / "all_videos"
        self.metrics_dir = self.output_dir / "metrics"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.seed_map = load_seed_map(args.seed_file)
        self.seed_log = None

        if args.use_dno:
            log(f"Loading FastWan DNO benchmark pipeline: {self.args.model_path}")
            self.dno_runner = FastWanDNO(args)
            self.generator = None
        else:
            log(f"Loading FastWan benchmark generator: {self.args.model_path}")
            self.dno_runner = None
            self.generator = VideoGenerator.from_pretrained(
                self.args.model_path,
                num_gpus=1,
                use_fsdp_inference=False,
                text_encoder_cpu_offload=args.text_encoder_cpu_offload,
                pin_cpu_memory=False,
                dit_cpu_offload=False,
                vae_cpu_offload=False,
                VSA_sparsity=args.vsa_sparsity,
            )

    def _selected_vbench_dimensions(self) -> set[str] | None:
        requested = getattr(self.args, "vbench_dimensions", None)
        if not requested:
            return None
        dimensions: set[str] = set()
        for value in requested:
            dimensions.update(part.strip() for part in value.split(",") if part.strip())
        return dimensions or None

    def _sampling_param(self, seed: int) -> SamplingParam:
        sampling_param = make_sampling_param(self.args)
        sampling_param.seed = seed
        return sampling_param

    def _write_seed(self, key: str, seed: int) -> None:
        if self.seed_log is None:
            raise RuntimeError("Seed log is not open")
        self.seed_log.write(f"{key}:{seed}\n")
        self.seed_log.flush()

    def find_existing_vbench_video(self, filename: str) -> str | None:
        if not self.args.skip_existing:
            return None
        return existing_video_path(
            run_root=self.args.run_root,
            rank_video_dir=self.video_dir,
            filename=filename,
            scope=self.args.skip_existing_scope,
            min_bytes=self.args.min_existing_video_bytes,
        )

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

    def generate_sample(
        self,
        prompt: str,
        seed: int,
        *,
        subdir: str | None = None,
        filename: str,
    ) -> None:
        if self.generator is None:
            raise RuntimeError("Vanilla generator is not loaded")
        video_dir = self.video_dir / subdir if subdir else self.video_dir
        video_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(video_dir / filename)
        sampling_param = self._sampling_param(seed)
        sampling_param.output_path = output_path
        result = self.generator.generate_video(
            prompt,
            output_path=output_path,
            save_video=True,
            return_frames=False,
            sampling_param=sampling_param,
        )
        log(f"[Benchmark] Saved video: {result.get('video_path')}")

    def optimize_sample(
        self,
        prompt_list: list[str],
        seed: int,
        *,
        subdir: str | None = None,
        filename: str,
        metrics_name: str | None = None,
    ) -> None:
        if self.dno_runner is None:
            raise RuntimeError("DNO runner is not loaded")
        sampling_param = self._sampling_param(seed)
        self.dno_runner.optimize(
            prompt_list[0],
            sampling_param,
            prompt_list=prompt_list,
            subdir=subdir,
            filename=filename,
            metrics_name=metrics_name,
        )

    def run_vbench(self) -> None:
        tasks = build_vbench_tasks(self.args.prompts_dir, iter_vbench_prompt_file)
        selected_dimensions = self._selected_vbench_dimensions()
        if selected_dimensions is not None:
            available_dimensions = {task[1] for task in tasks}
            unknown_dimensions = sorted(selected_dimensions - available_dimensions)
            if unknown_dimensions:
                available_text = ", ".join(sorted(available_dimensions))
                raise ValueError(
                    f"Unknown VBench dimension(s): {', '.join(unknown_dimensions)}. "
                    f"Available dimensions: {available_text}"
                )
            tasks = [task for task in tasks if task[1] in selected_dimensions]
            log(f"Filtering VBench dimensions: {', '.join(sorted(selected_dimensions))} ({len(tasks)} tasks)")
        processed = 0
        seed_path = self.output_dir / "seeds.txt"
        with seed_path.open("w", encoding="utf-8", buffering=1) as seed_log:
            self.seed_log = seed_log

            def run_task(task: tuple[int, str, str, list[str], int]) -> bool:
                task_id, dim, prompt, neg_prompts, idx = task
                filename = vbench_video_filename(prompt, idx)
                existing = self.find_existing_vbench_video(filename)
                if existing:
                    log(f"[Rank {self.args.rank}] Skipping existing VBench task {task_id + 1}/{len(tasks)}: {existing}")
                    return False

                key = f"{dim}:{prompt}-{idx}"
                seed = self.seed_map.get(key, random.randint(0, MAX_SEED))
                self._write_seed(key, seed)
                log(f"[Rank {self.args.rank}] VBench task {task_id + 1}/{len(tasks)} ({dim} idx={idx})")

                if self.args.use_dno:
                    prompt_list = [prompt]
                    if self.args.use_neg_prompt:
                        prompt_list += neg_prompts
                    self.optimize_sample(prompt_list, seed, filename=filename)
                else:
                    self.generate_sample(prompt, seed, filename=filename)
                return True

            if self.args.shard_strategy == "dynamic":
                queue_dir = os.path.join(self.args.run_root, "_queue")
                progress = tqdm(total=len(tasks), disable=self.args.rank != 0)
                while True:
                    task_id = claim_next_task(queue_dir, "vbench", len(tasks))
                    if task_id is None:
                        break
                    if run_task(tasks[task_id]):
                        processed += 1
                    progress.update(1)
                    if self.args.max_vbench_samples is not None and processed >= self.args.max_vbench_samples:
                        log(f"Reached max_vbench_samples={self.args.max_vbench_samples} on rank {self.args.rank}")
                        break
                progress.close()
            else:
                for task in tqdm(tasks, disable=self.args.rank != 0):
                    task_id = task[0]
                    if task_id % self.args.world_size != self.args.rank:
                        continue
                    if run_task(task):
                        processed += 1
                    if self.args.max_vbench_samples is not None and processed >= self.args.max_vbench_samples:
                        log(f"Reached max_vbench_samples={self.args.max_vbench_samples} on rank {self.args.rank}")
                        break

        self.seed_log = None
        log(f"Rank {self.args.rank} completed {processed} VBench samples.")

    def run_compbench(self) -> None:
        tasks = build_compbench_tasks(self.args.prompts_dir, iter_compbench_prompt_file)
        processed = 0
        seed_path = self.output_dir / "seeds.txt"
        with seed_path.open("w", encoding="utf-8", buffering=1) as seed_log:
            self.seed_log = seed_log

            def run_task(task: tuple[int, str, str, str, list[str]]) -> bool:
                task_id, sample_id, prompt, category, neg_prompts = task
                filename = compbench_video_filename(sample_id)
                existing = self.find_existing_compbench_video(filename, category)
                if existing:
                    log(f"[Rank {self.args.rank}] Skipping existing CompBench task {task_id + 1}/{len(tasks)}: {existing}")
                    return False

                key = f"{category}:{sample_id}-{prompt}"
                seed = self.seed_map.get(key, random.randint(0, MAX_SEED))
                self._write_seed(key, seed)
                log(f"[Rank {self.args.rank}] CompBench task {task_id + 1}/{len(tasks)} ({category} id={sample_id})")

                if self.args.use_dno:
                    prompt_list = [prompt]
                    if self.args.use_neg_prompt:
                        prompt_list += neg_prompts
                    self.optimize_sample(
                        prompt_list,
                        seed,
                        subdir=category,
                        filename=filename,
                        metrics_name=f"{category}_{sample_id}_{safe_stem(prompt)}.json",
                    )
                else:
                    self.generate_sample(prompt, seed, subdir=category, filename=filename)
                return True

            if self.args.shard_strategy == "dynamic":
                queue_dir = os.path.join(self.args.run_root, "_queue")
                progress = tqdm(total=len(tasks), disable=self.args.rank != 0)
                while True:
                    task_id = claim_next_task(queue_dir, "compbench", len(tasks))
                    if task_id is None:
                        break
                    if run_task(tasks[task_id]):
                        processed += 1
                    progress.update(1)
                    if self.args.max_compbench_samples is not None and processed >= self.args.max_compbench_samples:
                        log(f"Reached max_compbench_samples={self.args.max_compbench_samples} on rank {self.args.rank}")
                        break
                progress.close()
            else:
                for task in tqdm(tasks, desc="T2V-CompBench tasks", disable=self.args.rank != 0):
                    task_id = task[0]
                    if task_id % self.args.world_size != self.args.rank:
                        continue
                    if run_task(task):
                        processed += 1
                    if self.args.max_compbench_samples is not None and processed >= self.args.max_compbench_samples:
                        log(f"Reached max_compbench_samples={self.args.max_compbench_samples} on rank {self.args.rank}")
                        break

        self.seed_log = None
        log(f"Rank {self.args.rank} completed {processed} CompBench samples.")

    def run(self) -> None:
        if self.args.run_compbench:
            log("Running T2V-CompBench evaluation generation ...")
            self.run_compbench()
        else:
            log("Running VBench evaluation generation ...")
            self.run_vbench()


def run_dno(args: argparse.Namespace) -> None:
    log(f"Loading FastWan DNO pipeline: {args.model_path}")
    t0 = time.perf_counter()
    runner = FastWanDNO(args)
    log(f"DNO pipeline loaded in {time.perf_counter() - t0:.1f}s")
    prompts = load_user_prompts(args.prompt, args.prompt_file)
    for index, prompt in enumerate(prompts):
        seed = (args.seed + index) % MAX_SEED
        sampling_param = make_sampling_param(args)
        sampling_param.seed = seed
        stem = f"{index:03d}_{safe_stem(prompt)}"
        log(f"[DNO] Running prompt {index + 1}/{len(prompts)} with seed={seed}")
        runner.optimize(
            prompt,
            sampling_param,
            filename=f"{stem}.mp4",
            metrics_name=f"{stem}.json",
        )


def run_decoder_compare(args: argparse.Namespace) -> None:
    if args.reward_decoder == "wan":
        raise ValueError("--mode decoder_compare requires --reward-decoder taew2_1 or lighttaew2_1")

    log(f"Loading FastWan pipeline for decoder comparison: {args.model_path}")
    t0 = time.perf_counter()
    runner = FastWanDNO(args, build_rewards=False)
    log(f"Pipeline loaded in {time.perf_counter() - t0:.1f}s")

    sampling_param = make_sampling_param(args)
    compare_dir = runner.output_dir / "decoder_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(args.prompt)

    with torch.no_grad():
        prepared_batch = runner.prepare_batch(sampling_param, args.prompt)
        final_latents = runner.denoise_differentiable(
            prepared_batch,
            prepared_batch.latents.to(runner.device),
            step_noises=None,
        ).detach()

        official_video = runner.decode_differentiable(final_latents, use_reward_decoder=False)
        official_path = compare_dir / f"{stem}_official_wan.mp4"
        save_video(official_video.cpu(), official_path, sampling_param.fps)
        log(f"[Compare] Saved official Wan VAE decode: {official_path}")
        del official_video
        torch.cuda.empty_cache()

        tiny_video = runner.decode_differentiable(final_latents, use_reward_decoder=True)
        tiny_path = compare_dir / f"{stem}_{args.reward_decoder}.mp4"
        save_video(tiny_video.cpu(), tiny_path, sampling_param.fps)
        log(f"[Compare] Saved {args.reward_decoder} decode: {tiny_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FastWan2.1-T2V-1.3B inference / DNO / benchmark generation")
    parser.add_argument("--mode", choices=["vanilla", "dno", "decoder_compare"], default="vanilla")
    parser.add_argument(
        "--model-path", "--model_path", dest="model_path", default=DEFAULT_MODEL,
        help="Official Hugging Face model id or a local FastWan snapshot directory.",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--output-path", "--output_path", dest="output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument(
        "--prompt-file", "--prompt_file", dest="prompt_file", type=str, default=None,
        help="Small UTF-8 TXT/JSON prompt list; bypasses benchmark modes.",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--num-frames", "--num_frames", dest="num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--guidance-scale", "--guidance_scale", dest="guidance_scale", type=float, default=3.0)
    parser.add_argument("--vsa-sparsity", "--vsa_sparsity", dest="vsa_sparsity", type=float, default=0.8)
    parser.add_argument("--num-inference-steps", "--num_inference_steps", dest="num_inference_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0", help="Accepted for launch-script compatibility.")

    parser.add_argument("--dno-steps", "--dno_steps", dest="dno_steps", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", "--max_grad_norm", dest="max_grad_norm", type=float, default=0.1)
    parser.add_argument("--noise-optimization", "--noise_optimization", dest="noise_optimization",
                        choices=["initial", "full"], default="full")
    parser.add_argument("--reward-fns", "--reward_fns", dest="reward_fns", nargs="+", default=["viclip"])
    parser.add_argument("--reward-weights", "--reward_weights", dest="reward_weights", nargs="+", type=float,
                        default=[1.0])
    parser.add_argument("--reward-precision", "--reward_precision", dest="reward_precision",
                        choices=["auto", "fp32", "fp16", "bf16", "amp"], default="amp")
    parser.add_argument("--reward-device", "--reward_device", dest="reward_device", type=str, default="cuda:0")
    parser.add_argument("--save-step-video", "--save_step_video", dest="save_step_video", type=str2bool, default=False)
    parser.add_argument("--save-best-video", "--save_best_video", "--save_best", dest="save_best_video",
                        type=str2bool, default=True)
    parser.add_argument("--save-metrics", "--save_metrics", dest="save_metrics", type=str2bool, default=True)
    parser.add_argument("--text-encoder-cpu-offload", "--text_encoder_cpu_offload", dest="text_encoder_cpu_offload",
                        type=str2bool, default=True)
    parser.add_argument("--dno-gradient-checkpointing", "--dno_gradient_checkpointing",
                        dest="dno_gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--dno-save-activations-on-cpu", "--dno_save_activations_on_cpu",
                        dest="dno_save_activations_on_cpu", type=str2bool, default=True)
    parser.add_argument("--reward-latent-frames", "--reward_latent_frames", dest="reward_latent_frames", type=int,
                        default=None)
    parser.add_argument("--reward-frame-sample-count", "--reward_frame_sample_count",
                        dest="reward_frame_sample_count", type=int, default=None)
    parser.add_argument("--reward-decode-scale", "--reward_decode_scale", dest="reward_decode_scale", type=float,
                        default=1.0)
    parser.add_argument("--checkpoint-vae-decode", "--checkpoint_vae_decode", dest="checkpoint_vae_decode",
                        type=str2bool, default=False)
    parser.add_argument("--reward-decoder", "--reward_decoder", dest="reward_decoder",
                        choices=["wan", "taew2_1", "lighttaew2_1"], default="lighttaew2_1")
    parser.add_argument("--reward-decoder-path", "--reward_decoder_path", dest="reward_decoder_path", type=str,
                        default=None)
    parser.add_argument("--reward-decoder-device", "--reward_decoder_device", dest="reward_decoder_device", type=str,
                        default=None)
    parser.add_argument("--reward-tiny-decoder-precision", "--reward_tiny_decoder_precision",
                        dest="reward_tiny_decoder_precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--reward-tiny-decoder-parallel", "--reward_tiny_decoder_parallel",
                        dest="reward_tiny_decoder_parallel", type=str2bool, default=False)
    parser.add_argument("--reward-tiny-decoder-need-scaled", "--reward_tiny_decoder_need_scaled",
                        dest="reward_tiny_decoder_need_scaled", type=str, default="auto")

    parser.add_argument("--use-dno", "--use_dno", dest="use_dno", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--run-vbench", "--run_vbench", dest="run_vbench", type=str2bool, nargs="?", const=True,
                        default=False)
    parser.add_argument("--run-compbench", "--run_compbench", dest="run_compbench", type=str2bool, nargs="?",
                        const=True, default=False)
    parser.add_argument("--prompts-dir", "--prompts_dir", dest="prompts_dir", type=str,
                        default="vbench_prompts/neg_prompts")
    parser.add_argument("--vbench-dimensions", "--vbench_dimensions", dest="vbench_dimensions", nargs="+",
                        default=None,
                        help="Optional VBench dimension filter, e.g. color, color overall_consistency, or color,overall_consistency.")
    parser.add_argument("--save-path", "--save_path", dest="save_path", type=str, default="fastwan_vbench_dno")
    parser.add_argument("--seed-file", "--seed_file", dest="seed_file", type=str, default=None)
    parser.add_argument("--world-size", "--world_size", dest="world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--run-id", "--run_id", dest="run_id", type=str, default=None)
    parser.add_argument("--shard-strategy", "--shard_strategy", dest="shard_strategy",
                        choices=["static", "dynamic"], default="static")
    parser.add_argument("--log-all-ranks", "--log_all_ranks", dest="log_all_ranks", type=str2bool, nargs="?",
                        const=True, default=False)
    parser.add_argument("--skip-existing", "--skip_existing", dest="skip_existing", type=str2bool, nargs="?",
                        const=True, default=False)
    parser.add_argument("--skip-existing-scope", "--skip_existing_scope", dest="skip_existing_scope",
                        choices=["rank", "run"], default="run")
    parser.add_argument("--min-existing-video-bytes", "--min_existing_video_bytes",
                        dest="min_existing_video_bytes", type=int, default=1)
    parser.add_argument("--use-neg-prompt", "--use_neg_prompt", dest="use_neg_prompt", type=str2bool, default=False)
    parser.add_argument("--max-vbench-samples", "--max_vbench_samples", dest="max_vbench_samples", type=int,
                        default=None)
    parser.add_argument("--max-compbench-samples", "--max_compbench_samples", dest="max_compbench_samples", type=int,
                        default=None)

    args = parser.parse_args()
    if args.mode == "dno":
        args.use_dno = True
    if args.use_dno and args.reward_decoder == "wan":
        raise ValueError("--use_dno requires --reward_decoder taew2_1 or lighttaew2_1")
    if args.run_compbench:
        if args.prompts_dir == parser.get_default("prompts_dir"):
            args.prompts_dir = "comp-t2v-prompts/neg_prompts"
        if args.save_path == parser.get_default("save_path"):
            args.save_path = "fastwan_compbench_dno" if args.use_dno else "fastwan_compbench"
        if args.seed_file is None:
            args.seed_file = "comp-t2v-prompts/compbench_seed.txt"
    elif args.run_vbench and args.save_path == parser.get_default("save_path") and not args.use_dno:
        args.save_path = "fastwan_vbench"
    if args.run_vbench and args.seed_file is None:
        args.seed_file = "vbench_prompts/vbench_seed.txt"
    elif args.run_compbench and args.seed_file is None:
        args.seed_file = "comp-t2v-prompts/compbench_seed.txt"

    if args.run_vbench or args.run_compbench:
        timestamp = args.run_id or time.strftime("%Y-%m-%d-%H-%M-%S")
        args.run_root = os.path.join(args.save_path, timestamp)
        args.output_path = os.path.join(args.run_root, f"shard_{args.rank}")
        os.makedirs(args.output_path, exist_ok=True)
        with open(os.path.join(args.output_path, "args.json"), "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2)
    else:
        args.run_root = args.output_path
    return args


def main() -> None:
    args = parse_args()
    load_fastvideo_runtime()
    if args.run_vbench or args.run_compbench:
        FastWanBenchmarkRunner(args).run()
    elif args.mode == "vanilla" and not args.use_dno:
        run_vanilla(args)
    elif args.mode == "dno" or args.use_dno:
        run_dno(args)
    else:
        run_decoder_compare(args)


if __name__ == "__main__":
    main()
