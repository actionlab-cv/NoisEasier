# SPDX-License-Identifier: Apache-2.0
"""Configuration for the FastWan2.1-T2V-1.3B DMD pipeline."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, T5Config
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


def t5_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    if torch.isnan(hidden_state).any():
        raise ValueError("FastWan text encoder produced NaN hidden states")
    prompt_embeds = [state[:length] for state, length in zip(hidden_state, seq_lens, strict=True)]
    return torch.stack(
        [
            torch.cat([state, state.new_zeros(512 - state.size(0), state.size(1))])
            for state in prompt_embeds
        ],
        dim=0,
    )


@dataclass
class WanT2V480PConfig(PipelineConfig):
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False
    flow_shift: float | None = 3.0

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = field(
        default_factory=lambda: (t5_postprocess_text,)
    )

    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32",))
    warp_denoising_step: bool = True

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True


@dataclass
class FastWan2_1_T2V_480P_Config(WanT2V480PConfig):
    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 757, 522]
    )


__all__ = ["FastWan2_1_T2V_480P_Config", "WanT2V480PConfig"]
