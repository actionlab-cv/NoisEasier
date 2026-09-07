# SPDX-License-Identifier: Apache-2.0
"""FastWan-only model registry used by the NoisEasier release."""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.pipelines.wan import FastWan2_1_T2V_480P_Config
from fastvideo.fastvideo_args import WorkloadType
from fastvideo.logger import init_logger
from fastvideo.utils import maybe_download_model_index, verify_model_config_and_directory

logger = init_logger(__name__)

FASTWAN_MODEL_ID = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
FASTWAN_PIPELINE_NAME = "WanDMDPipeline"
FASTWAN_PRESET_NAME = "fast_wan_t2v_480p"


@dataclasses.dataclass(frozen=True)
class ConfigInfo:
    sampling_param_cls: type[SamplingParam] | None
    pipeline_config_cls: type[PipelineConfig]
    workload_types: tuple[WorkloadType, ...]
    model_family: str
    default_preset: str


@dataclasses.dataclass(frozen=True)
class ModelInfo:
    pipeline_cls: type
    sampling_param_cls: type[SamplingParam]
    pipeline_config_cls: type[PipelineConfig]


_CONFIG_INFO = ConfigInfo(
    sampling_param_cls=None,
    pipeline_config_cls=FastWan2_1_T2V_480P_Config,
    workload_types=(WorkloadType.T2V,),
    model_family="wan",
    default_preset=FASTWAN_PRESET_NAME,
)


def _model_index(model_path: str) -> dict[str, Any]:
    if os.path.exists(model_path):
        return verify_model_config_and_directory(model_path)
    return maybe_download_model_index(model_path)


def _is_fastwan(model_path: str) -> bool:
    if model_path == FASTWAN_MODEL_ID:
        return True
    if os.path.basename(os.path.normpath(model_path)).lower() == FASTWAN_MODEL_ID.rsplit("/", 1)[-1].lower():
        return True
    try:
        return _model_index(model_path).get("_class_name") == FASTWAN_PIPELINE_NAME
    except Exception:
        return False


def _require_fastwan(model_path: str) -> ConfigInfo:
    if not _is_fastwan(model_path):
        raise RuntimeError(
            "This NoisEasier release vendors only the FastWan2.1-T2V-1.3B "
            f"runtime; unsupported model path: {model_path}"
        )
    return _CONFIG_INFO


def get_pipeline_config_classes(
    pipeline_class_name: str,
) -> tuple[type[PipelineConfig], type[SamplingParam]] | None:
    if pipeline_class_name == FASTWAN_PIPELINE_NAME:
        return FastWan2_1_T2V_480P_Config, SamplingParam
    return None


def get_model_info(
    model_path: str,
    pipeline_type: Any = None,
    workload_type: WorkloadType | None = None,
    override_pipeline_cls_name: str | None = None,
) -> ModelInfo:
    del pipeline_type
    if workload_type not in (None, WorkloadType.T2V):
        raise ValueError("FastWan NoisEasier supports text-to-video inference only")

    config = _model_index(model_path)
    pipeline_name = override_pipeline_cls_name or config.get("_class_name")
    if pipeline_name != FASTWAN_PIPELINE_NAME:
        raise ValueError(
            f"Expected {FASTWAN_PIPELINE_NAME} in model_index.json, got {pipeline_name!r}"
        )
    _require_fastwan(model_path)

    from fastvideo.pipelines.basic.wan.wan_dmd_pipeline import WanDMDPipeline

    return ModelInfo(
        pipeline_cls=WanDMDPipeline,
        sampling_param_cls=SamplingParam,
        pipeline_config_cls=FastWan2_1_T2V_480P_Config,
    )


def get_pipeline_config_cls_from_name(model_path: str) -> type[PipelineConfig]:
    return _require_fastwan(model_path).pipeline_config_cls


def get_sampling_param_cls_for_name(model_path: str) -> type[SamplingParam] | None:
    return _require_fastwan(model_path).sampling_param_cls


def get_model_family(model_path: str) -> str | None:
    return _require_fastwan(model_path).model_family


def get_default_preset(model_path: str) -> str | None:
    return _require_fastwan(model_path).default_preset


def get_preset_selection(model_path: str) -> tuple[str | None, str | None]:
    info = _require_fastwan(model_path)
    return info.default_preset, info.model_family


def get_registered_model_paths() -> list[str]:
    return [FASTWAN_MODEL_ID]


def get_registered_models_with_workloads(
    workload_type: str | None = None,
) -> list[dict[str, Any]]:
    workloads = [item.value for item in _CONFIG_INFO.workload_types]
    if workload_type is not None and workload_type.lower() not in workloads:
        return []
    return [{
        "id": FASTWAN_MODEL_ID,
        "label": "FastWan2.1 T2V 1.3B Diffusers",
        "workload_types": workloads,
    }]


def _register_preset() -> None:
    from fastvideo.api.presets import InferencePreset, PresetStageSpec, register_preset

    denoise_stage = PresetStageSpec(
        name="denoise",
        kind="denoising",
        description="FastWan DMD denoising",
        allowed_overrides=frozenset({"num_inference_steps", "guidance_scale"}),
    )
    register_preset(
        InferencePreset(
            name=FASTWAN_PRESET_NAME,
            version=1,
            model_family="wan",
            description="FastWan 2.1 T2V DMD at 480p (3-step)",
            workload_type="t2v",
            stage_schemas=(denoise_stage,),
            defaults={
                "height": 448,
                "width": 832,
                "num_frames": 61,
                "fps": 16,
                "guidance_scale": 3.0,
                "num_inference_steps": 3,
                "negative_prompt": (
                    "Bright tones, overexposed, static, blurred details, subtitles, "
                    "style, works, paintings, images, overall gray, worst quality, "
                    "low quality, JPEG compression residue, ugly, incomplete, "
                    "deformed, disfigured, misshapen limbs, still picture"
                ),
            },
        )
    )


_register_preset()


__all__ = [
    "ConfigInfo",
    "ModelInfo",
    "get_default_preset",
    "get_model_family",
    "get_model_info",
    "get_pipeline_config_cls_from_name",
    "get_registered_model_paths",
    "get_registered_models_with_workloads",
    "get_sampling_param_cls_for_name",
    "get_pipeline_config_classes",
    "get_preset_selection",
]
