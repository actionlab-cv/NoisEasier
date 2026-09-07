from __future__ import annotations

import argparse
import json
import os
import socket
from collections.abc import Iterator
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

MAX_SEED = np.iinfo(np.int32).max


def log(message: str) -> None:
    print(message, flush=True)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


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


def iter_vbench_prompt_file(path: str) -> Iterator[tuple[str, list[str]]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            for prompt, neg_prompts in data.items():
                yield prompt, list(neg_prompts or [])
        elif isinstance(data, list):
            for item in data:
                prompt = item.get("prompt_en") if isinstance(item, dict) else None
                if prompt:
                    yield prompt, []
        else:
            raise ValueError(f"Unsupported VBench JSON format in {path}")
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                prompt = line.strip()
                if prompt:
                    yield prompt, []
    else:
        raise ValueError(f"Unsupported VBench prompt file extension: {path}")


def iter_compbench_prompt_file(path: str) -> Iterator[tuple[str, str, str, list[str]]]:
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
            list(item.get("negative_prompt", []) or []),
        )


def safe_stem(text: str, max_len: int = 80) -> str:
    stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)
    stem = "_".join(part for part in stem.split("_") if part)
    return (stem[:max_len] or "sample").strip("_")


def ensure_single_process_dist_env() -> None:
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            os.environ["MASTER_PORT"] = str(sock.getsockname()[1])


def context_aware_checkpoint(function, *args, **kwargs):
    # Keep utility imports usable on CPU/login nodes. Importing FastVideo
    # eagerly also imports its Triton kernels, which require an active driver.
    from fastvideo.forward_context import get_forward_context, set_forward_context

    forward_context = get_forward_context()

    def run_with_forward_context(*inner_args):
        with set_forward_context(
            current_timestep=forward_context.current_timestep,
            attn_metadata=forward_context.attn_metadata,
            forward_batch=forward_context.forward_batch,
        ):
            return function(*inner_args)

    return checkpoint(run_with_forward_context, *args, use_reentrant=False, **kwargs)


def conv2d_3x3(in_channels: int, out_channels: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class WanTinyMemBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: nn.Module):
        super().__init__()
        self.conv = nn.Sequential(
            conv2d_3x3(in_channels * 2, out_channels),
            act,
            conv2d_3x3(out_channels, out_channels),
            act,
            conv2d_3x3(out_channels, out_channels),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        self.act = act

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], dim=1)) + self.skip(x))


class WanTinyTPool(nn.Module):
    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels * stride, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nt, channels, height, width = x.shape
        return self.conv(x.reshape(nt // self.stride, self.stride * channels, height, width))


class WanTinyTGrow(nn.Module):
    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels, channels * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nt, channels, height, width = x.shape
        x = self.conv(x)
        return x.reshape(nt * self.stride, channels, height, width)


def apply_wan_tiny_blocks(
    model: nn.Sequential,
    x: torch.Tensor,
    *,
    parallel: bool,
) -> torch.Tensor:
    """Apply TAEHV-style temporal blocks to an NTCHW tensor with autograd intact."""
    if x.ndim != 5:
        raise ValueError(f"Wan tiny decoder expects an NTCHW tensor, got shape {tuple(x.shape)}")
    batch, frames, channels, height, width = x.shape
    if parallel:
        x = x.reshape(batch * frames, channels, height, width)
        for block in model:
            if isinstance(block, WanTinyMemBlock):
                nt, channels, height, width = x.shape
                frames = nt // batch
                xt = x.reshape(batch, frames, channels, height, width)
                memory = F.pad(xt, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :frames].reshape_as(x)
                x = block(x, memory)
            else:
                x = block(x)
        nt, channels, height, width = x.shape
        return x.view(batch, nt // batch, channels, height, width)

    outputs: list[torch.Tensor] = []
    work_queue: list[tuple[torch.Tensor, int]] = [
        (frame, 0) for frame in x.reshape(batch, frames * channels, height, width).chunk(frames, dim=1)
    ]
    memory: list[torch.Tensor | list[torch.Tensor] | None] = [None] * len(model)

    while work_queue:
        xt, block_index = work_queue.pop(0)
        if block_index == len(model):
            outputs.append(xt)
            continue

        block = model[block_index]
        if isinstance(block, WanTinyMemBlock):
            past = memory[block_index]
            past_tensor = xt * 0 if past is None else past
            xt_next = block(xt, past_tensor)
            memory[block_index] = xt
            work_queue.insert(0, (xt_next, block_index + 1))
        elif isinstance(block, WanTinyTPool):
            pool_memory = memory[block_index]
            if pool_memory is None:
                pool_memory = []
                memory[block_index] = pool_memory
            if not isinstance(pool_memory, list):
                raise TypeError("Invalid temporal pool memory state")
            pool_memory.append(xt)
            if len(pool_memory) > block.stride:
                raise RuntimeError("Temporal pool received more frames than its stride")
            if len(pool_memory) == block.stride:
                pooled = block(torch.cat(pool_memory, dim=1).view(batch * block.stride, xt.shape[1], xt.shape[2], xt.shape[3]))
                memory[block_index] = []
                work_queue.insert(0, (pooled, block_index + 1))
        elif isinstance(block, WanTinyTGrow):
            grown = block(xt)
            for xt_next in reversed(grown.view(batch, block.stride * xt.shape[1], grown.shape[2], grown.shape[3]).chunk(block.stride, dim=1)):
                work_queue.insert(0, (xt_next, block_index + 1))
        else:
            work_queue.insert(0, (block(xt), block_index + 1))

    return torch.stack(outputs, dim=1)


class WanTinyTAEDecoder(nn.Module):
    """Differentiable Wan2.1 TAE decoder for reward proxy decoding."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        dtype: torch.dtype,
        need_scaled: bool,
        parallel: bool,
    ):
        super().__init__()
        self.dtype = dtype
        self.need_scaled = need_scaled
        self.parallel = parallel
        self.frames_to_trim = 3
        self.latents_mean = (
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        )
        self.latents_std = (
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        )
        act = nn.ReLU(inplace=False)
        features = (256, 128, 64, 64)
        self.decoder = nn.Sequential(
            Clamp(),
            conv2d_3x3(16, features[0]),
            act,
            WanTinyMemBlock(features[0], features[0], act),
            WanTinyMemBlock(features[0], features[0], act),
            WanTinyMemBlock(features[0], features[0], act),
            nn.Upsample(scale_factor=2),
            WanTinyTGrow(features[0], 1),
            conv2d_3x3(features[0], features[1], bias=False),
            WanTinyMemBlock(features[1], features[1], act),
            WanTinyMemBlock(features[1], features[1], act),
            WanTinyMemBlock(features[1], features[1], act),
            nn.Upsample(scale_factor=2),
            WanTinyTGrow(features[1], 2),
            conv2d_3x3(features[1], features[2], bias=False),
            WanTinyMemBlock(features[2], features[2], act),
            WanTinyMemBlock(features[2], features[2], act),
            WanTinyMemBlock(features[2], features[2], act),
            nn.Upsample(scale_factor=2),
            WanTinyTGrow(features[2], 2),
            conv2d_3x3(features[2], features[3], bias=False),
            act,
            conv2d_3x3(features[3], 3),
        )
        self.load_weights(checkpoint_path)
        self.to(dtype=self.dtype)
        self.requires_grad_(False)

    def load_weights(self, checkpoint_path: str | Path) -> None:
        from safetensors.torch import load_file

        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.suffix != ".safetensors":
            raise ValueError("Wan tiny reward decoder only supports .safetensors checkpoints")
        state_dict = load_file(str(checkpoint_path), device="cpu")
        decoder_state = {key: value for key, value in state_dict.items() if key.startswith("decoder.")}
        self.load_state_dict(decoder_state)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.need_scaled:
            mean = torch.tensor(self.latents_mean, device=latents.device, dtype=latents.dtype).view(1, 16, 1, 1, 1)
            std = torch.tensor(self.latents_std, device=latents.device, dtype=latents.dtype).view(1, 16, 1, 1, 1)
            latents = latents * std + mean
        video = apply_wan_tiny_blocks(
            self.decoder,
            latents.transpose(1, 2).to(self.dtype),
            parallel=self.parallel,
        )
        video = video[:, self.frames_to_trim:].transpose(1, 2)
        return video.clamp(0, 1).float()


def save_video(video: torch.Tensor, path: str | Path, fps: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = video[0].detach().cpu().float().clamp(0, 1)
    frames = frames.permute(1, 2, 3, 0).numpy()
    frames = (frames * 255).round().astype(np.uint8)
    imageio.mimsave(path, list(frames), fps=fps, format="mp4")
