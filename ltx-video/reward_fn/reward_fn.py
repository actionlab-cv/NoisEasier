from typing import List
import importlib
import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
import torchvision.transforms.functional as F
import torch.nn.functional as FF
from torchvision.transforms import (
    Normalize,
    Resize,
    InterpolationMode,
    CenterCrop,
    RandomCrop,
)
from einops import rearrange
from typing import Optional, Union

# Image processing
CLIP_RESIZE = Resize((224, 224), interpolation=InterpolationMode.BICUBIC)
CLIP_NORMALIZE = Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)
CENTER_CROP = CenterCrop(224)

ViCLIP_NORMALIZE = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

precision_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "amp": torch.float32,
    }

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REWARD_ROOT = Path(
    os.environ.get("NOISEASIER_REWARD_ROOT", PROJECT_ROOT / "checkpoints" / "rewards")
).expanduser()


def resolve_reward_asset(filename: str, env_name: str, fallback: Optional[str] = None) -> Path:
    """Resolve a configured, project-local, or cached reward asset."""
    if configured := os.environ.get(env_name):
        return Path(configured).expanduser()
    local_path = REWARD_ROOT / filename
    if local_path.is_file() or fallback is None:
        return local_path
    return Path(fallback).expanduser()


def ensure_hpsv2_bpe_vocab() -> None:
    hps_spec = importlib.util.find_spec("hpsv2")
    open_clip_spec = importlib.util.find_spec("open_clip")
    if hps_spec is None or open_clip_spec is None or hps_spec.origin is None or open_clip_spec.origin is None:
        return

    source = Path(open_clip_spec.origin).parent / "bpe_simple_vocab_16e6.txt.gz"
    target = Path(hps_spec.origin).parent / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
    if source.exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def ensure_viclip_bpe_vocab() -> None:
    target = Path(__file__).resolve().parent.parent / "viclip" / "bpe_simple_vocab_16e6.txt.gz"
    if target.exists():
        return
    for package_name in ("open_clip", "clip", "hpsv2"):
        spec = importlib.util.find_spec(package_name)
        if spec is None or spec.origin is None:
            continue
        package_dir = Path(spec.origin).parent
        candidates = [
            package_dir / "bpe_simple_vocab_16e6.txt.gz",
            package_dir / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz",
        ]
        for source in candidates:
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                return


def _load_image_reward_inference_api():
    """Load ImageReward's inference API without its optional ReFL stack.

    The published ImageReward package imports ReFL and wandb from its package
    initializer even though NoisEasier only needs ``ImageReward.utils.load``.
    Loading that submodule directly keeps inference independent of those
    optional training dependencies while retaining the owner's implementation.
    """
    import transformers.modeling_utils as modeling_utils
    from transformers import pytorch_utils

    for name in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(modeling_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))

    if "ImageReward.utils" in sys.modules:
        return sys.modules["ImageReward.utils"]

    package_spec = importlib.util.find_spec("ImageReward")
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise ModuleNotFoundError(
            "ImageReward is required for --reward-fns img_reward. "
            "Install the official image-reward package first."
        )

    if "ImageReward" not in sys.modules:
        package = types.ModuleType("ImageReward")
        package.__file__ = package_spec.origin
        package.__package__ = "ImageReward"
        package.__path__ = list(package_spec.submodule_search_locations)
        sys.modules["ImageReward"] = package

    return importlib.import_module("ImageReward.utils")


def get_reward_fn(reward_fn_name: str, **kwargs):
    if reward_fn_name == "pickscore":
        return get_pick_score_fn(**kwargs)
    elif reward_fn_name == "hpsv2":
        return get_hpsv2_fn(**kwargs)
    elif reward_fn_name == "img_reward":
        return get_img_reward_fn(**kwargs)
    elif reward_fn_name == "aesthetic":
        return get_aesthetic_score_fn(**kwargs)
    elif reward_fn_name == "viclip":
        return get_viclip_score_fn(**kwargs)
    elif reward_fn_name == "intervid2":
        return get_intern_vid2_score_fn(**kwargs)
    elif reward_fn_name == "vjepa":
        return get_vjepa_score_fn(**kwargs)
    elif reward_fn_name == "dino":
        return get_dino_score_fn(**kwargs)
    elif reward_fn_name == "motion":
        return get_motion_score_fn(**kwargs)
    elif reward_fn_name == "clip":
        return get_clip_score_fn(**kwargs)
    elif reward_fn_name == "videoscore":
        from reward_fn.videoscore import get_videoscore_fn
        return get_videoscore_fn(**kwargs)
    else:
        raise ValueError("Invalid reward_fn_name")


def get_pick_score_fn(precision="fp32", device=None):
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval()
    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model.requires_grad_(False)
    if precision == "fp16":
        model.to(torch.float16)
    model = model.to(device)

    def score_fn(image_inputs: torch.Tensor, text_inputs: list[str], return_logits=False):
        if image_inputs.ndim == 5:
            image_inputs = rearrange(image_inputs, "b c t h w -> (b t) c h w")
        pixel_values = CLIP_NORMALIZE(CENTER_CROP(CLIP_RESIZE(image_inputs)))

        # embed
        image_embs = model.get_image_features(pixel_values=pixel_values)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        with torch.no_grad():
            batch = processor(
                text=text_inputs,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            text_embs = model.get_text_features(**batch)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        sim = image_embs @ text_embs.t()

        if text_embs.shape[0] == 1:
            if return_logits:
                sim = sim * model.logit_scale.exp()
            reward = sim[:, 0]
        else:
            # InfoNCE with negatives
            sim = sim * model.logit_scale.exp()
            probs = torch.softmax(sim, dim=1)
            reward = probs[:, 0]
        return reward.mean()

    return score_fn


def get_hpsv2_fn(precision="amp", device=None):
    precision = "amp" if precision == "no" else precision
    assert precision in ["bf16", "fp16", "amp", "fp32"]
    ensure_hpsv2_bpe_vocab()
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    ckpt_path = resolve_reward_asset(
        "HPS_v2_compressed.pt",
        "NOISEASIER_HPSV2_PATH",
        "~/.cache/hpsv2/HPS_v2_compressed.pt",
    )
    assert ckpt_path.is_file(), (
        f"HPSv2 checkpoint not found at {ckpt_path}. See checkpoints/README.md."
    )

    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14",
        "laion2B-s32B-b79K",
        precision=precision,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )

    # Load fine-tuned weights
    dtype = precision_map.get(precision, torch.float32)
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device, dtype=dtype)
    model.eval()
    tokenizer = get_tokenizer("ViT-H-14")

    # gets vae decode as input
    def score_fn(
        image_inputs: torch.Tensor, text_inputs: List[str], return_logits=False
    ):
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]

        if image_inputs.ndim == 5:
            image_inputs = rearrange(image_inputs, "b c t h w -> (b t) c h w")
        # Process pixels. The original MS DNO path generated square 256px
        # frames and skipped resize/crop; FastWan can use arbitrary aspect
        # ratios, so normalize to HPS/OpenCLIP's 224px visual grid here.
        image_inputs = CLIP_NORMALIZE(CENTER_CROP(CLIP_RESIZE(image_inputs)))
        model_dtype = getattr(model.visual, "conv1").weight.dtype
        amp_enabled = precision == "amp" and image_inputs.is_cuda
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            image_features = model.encode_image(image_inputs.to(model_dtype), normalize=True)

        with torch.no_grad():
            text_inputs = tokenizer(text_inputs).to(image_inputs.device)
            text_features = model.encode_text(text_inputs, normalize=True)

        sim = image_features @ text_features.t()

        if text_features.shape[0] == 1:
            if return_logits:
                sim = sim * model.logit_scale.exp()
            reward = sim[:, 0]
        else:
            # InfoNCE with negatives

            pos = sim[:, 0]
            neg = sim[:, 1:]
            m = 0.05
            tau = 0.05
            hardest_soft = tau * torch.logsumexp((neg - pos.unsqueeze(1) + m) / tau, dim=1)
            reward = pos - hardest_soft

        return reward.mean()

    return score_fn


def get_img_reward_fn(precision="fp32", device=None):
    # pip install image-reward
    RM = _load_image_reward_inference_api()
    from torchvision.transforms import Compose, Resize, CenterCrop
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC

    image_reward_path = resolve_reward_asset(
        "ImageReward.pt", "NOISEASIER_IMAGE_REWARD_PATH", "~/.cache/ImageReward/ImageReward.pt"
    )
    med_config_path = resolve_reward_asset(
        "med_config.json", "NOISEASIER_IMAGE_REWARD_CONFIG", "~/.cache/ImageReward/med_config.json"
    )
    if not image_reward_path.is_file() or not med_config_path.is_file():
        raise FileNotFoundError(
            "ImageReward.pt and med_config.json are required. See checkpoints/README.md."
        )
    model = RM.load(
        str(image_reward_path), device=device, med_config=str(med_config_path)
    )
    model.eval()
    model.requires_grad_(False)

    rm_preprocess = Compose(
        [
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            CLIP_NORMALIZE,
        ]
    )

    # gets vae decode as input
    def score_fn(image_inputs: torch.Tensor, text_inputs: List[str], return_logits=False):
        del return_logits
        if image_inputs.ndim == 5:
            image_inputs = rearrange(image_inputs, "b c t h w -> (b t) c h w")

        if precision == "fp16":
            model.to(torch.float16)
        elif precision == "bf16":
            model.to(torch.bfloat16)

        image = rm_preprocess(image_inputs)

        toks = model.blip.tokenizer(
            text_inputs[0],
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}

        score = model.score_gard(
            toks["input_ids"],
            toks["attention_mask"],
            image
        )
        rewards = (score[:, 0] + 2) / 4
        return rewards.mean()

    return score_fn


class ResizeCropMinSize(nn.Module):

    def __init__(self, min_size, interpolation=InterpolationMode.BICUBIC, fill=0):
        super().__init__()
        if not isinstance(min_size, int):
            raise TypeError(f"Size should be int. Got {type(min_size)}")
        self.min_size = min_size
        self.interpolation = interpolation
        self.fill = fill
        self.random_crop = RandomCrop((min_size, min_size))

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size
        scale = self.min_size / float(min(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            img = self.random_crop(img)
        return img


def get_viclip_score_fn(precision="amp", n_frames=8, device=None):
    assert n_frames == 8
    ensure_viclip_bpe_vocab()
    from viclip import get_viclip

    checkpoint_path = resolve_reward_asset(
        "ViClip-InternVid-10M-FLT.pth", "NOISEASIER_VICLIP_PATH"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"ViCLIP checkpoint not found at {checkpoint_path}. See checkpoints/README.md."
        )
    model_dict = get_viclip(size="l", checkpoint_path=str(checkpoint_path))
    vi_clip = model_dict["viclip"]
    vi_clip.eval()
    vi_clip.requires_grad_(False)
    if precision == "fp16":
        vi_clip.to(torch.float16)
    vi_clip.to(device)

    viclip_resize = ResizeCropMinSize(224)

    def score_fn(image_inputs: torch.Tensor, text_inputs: str, return_logits=False):
        del return_logits
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]

        if image_inputs.ndim == 5:
            image_inputs = rearrange(image_inputs, "b c t h w -> b t c h w")
        b, t = image_inputs.shape[:2]
        image_inputs = image_inputs.view(b * t, *image_inputs.shape[2:])
        pixel_values = ViCLIP_NORMALIZE(viclip_resize(image_inputs))
        model_dtype = next(vi_clip.parameters()).dtype
        pixel_values = pixel_values.to(dtype=model_dtype)
        pixel_values = pixel_values.view(b, t, *pixel_values.shape[1:])

        if t == n_frames * 2:
            crop1 = pixel_values[:, ::2, ...]
            crop2 = pixel_values[:, 1::2, ...]
        else:
            raise ValueError("Invalid number of frames")

        video_feat1 = vi_clip.get_vid_feat_with_grad(crop1)
        video_feat2 = vi_clip.get_vid_feat_with_grad(crop2)
        video_feat = torch.cat((video_feat1, video_feat2), dim=0)

        with torch.no_grad():
            text_feat = vi_clip.encode_text(text_inputs)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat.to(dtype=video_feat.dtype)

        sim = video_feat @ text_feat.t()

        if len(text_inputs) == 1:
            score = sim[:, 0]
        else:
            # InfoNCE with negatives

            pos = sim[:, 0]
            neg = sim[:, 1:]
            m = 0.05
            tau = 0.05
            hardest_soft = tau * torch.logsumexp((neg - pos.unsqueeze(1) + m) / tau, dim=1)
            score = pos - hardest_soft

        return score.mean()

    return score_fn


def get_intern_vid2_score_fn(precision="amp", n_frames=8, device=None):

    from intern_vid2.demo_config import Config, eval_dict_leaf
    from intern_vid2.demo_utils import setup_internvideo2

    config = Config.from_file("intern_vid2/configs/internvideo2_stage2_config.py")
    config = eval_dict_leaf(config)
    config["inputs"]["video_input"]["num_frames"] = n_frames
    config["inputs"]["video_input"]["num_frames_test"] = n_frames
    config["model"]["vision_encoder"]["num_frames"] = n_frames

    vi_clip, tokenizer = setup_internvideo2(config)
    vi_clip.to(device).eval()
    vi_clip.requires_grad_(False)
    if precision == "fp16":
        vi_clip.to(torch.float16)

    viclip_resize = ResizeCropMinSize(224)

    def score_fn(image_inputs: torch.Tensor, text_inputs: str):
        if image_inputs.ndim == 5:
            image_inputs = rearrange(image_inputs, "b c t h w -> b t c h w")
        b, t = image_inputs.shape[:2]
        image_inputs = image_inputs.view(b * t, *image_inputs.shape[2:])
        pixel_values = ViCLIP_NORMALIZE(viclip_resize(image_inputs))
        model_dtype = next(vi_clip.parameters()).dtype
        pixel_values = pixel_values.to(dtype=model_dtype)
        pixel_values = pixel_values.view(b, t, *pixel_values.shape[1:])

        if t == n_frames * 2:
            pixel_values1 = pixel_values[:, ::2, ...]
            pixel_values2 = pixel_values[:, 1::2, ...]
        else:
            raise ValueError("Invalid number of frames")

        video_features1 = vi_clip.get_vid_feat_with_grad(pixel_values1)
        video_features2 = vi_clip.get_vid_feat_with_grad(pixel_values2)
        video_features = torch.cat((video_features1, video_features2), dim=0)

        with torch.no_grad():
            text = tokenizer(
                text_inputs,
                padding="max_length",
                truncation=True,
                max_length=40,
                return_tensors="pt",
            ).to(device)
            _, text_features = vi_clip.encode_text(text)
            text_features = vi_clip.text_proj(text_features)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.to(dtype=video_features.dtype)

        score = video_features @ text_features.t()
        return score.mean()

    return score_fn


def get_clip_score_fn(precision="amp", device=None):
    assert precision in ["bf16", "fp16", "amp", "fp32"]
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14",
        "laion2B-s32B-b79K",
        precision=precision,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=None,
        force_image_size=None,
        image_mean=None,
        image_std=None,
        image_interpolation=None,
        image_resize_mode=None,  # only effective for inference
        aug_cfg={},
        pretrained_image=False,
        output_dict=True,
    )
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    model.eval()
    model.requires_grad_(False)

    # gets vae decode as input
    def score_fn(image_inputs: torch.Tensor, text_inputs: List[str], return_logits=False):
        # Process pixels and multicrop
        model.to(image_inputs.device)

        if image_inputs.ndim == 5:
            image_inputs = rearrange(image_inputs, "b c t h w -> (b t) c h w")
        image_inputs = CLIP_RESIZE(image_inputs)
        image_inputs = CLIP_NORMALIZE(image_inputs)

        model_dtype = getattr(model.visual, "conv1").weight.dtype
        image_features = model.encode_image(image_inputs.to(model_dtype), normalize=True)

        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]

        with torch.no_grad():
            text_inputs = tokenizer(text_inputs).to(image_inputs.device)
            text_features = model.encode_text(text_inputs, normalize=True)

        sim = image_features @ text_features.t()

        if text_features.shape[0] == 1:
            if return_logits:
                sim = sim * model.logit_scale.exp()
            reward = sim[:, 0]
        else:
            # InfoNCE with negatives
            sim = sim * model.logit_scale.exp()
            probs = torch.softmax(sim, dim=1)
            reward = probs[:, 0]
        return reward.mean()

    return score_fn


def get_aesthetic_score_fn(aesthetic_target=None, device=None, precision=None):
    '''
    Args:
        aesthetic_target: optional target value subtracted from the score.
        device: torch.device, the device to run the model.
        precision: torch.dtype, the data type of the model.
    '''
    from reward_fn.aesthetic_scorer import AestheticScorerDiff
    scorer = AestheticScorerDiff(dtype=precision)
    scorer.requires_grad_(False)
    scorer.to(device)

    def score_fn(image_inputs: torch.Tensor, text_inputs: List[str], return_logits=False):
        del return_logits

        if image_inputs.ndim == 5:
            image_inputs = rearrange(image_inputs, "b c t h w -> (b t) c h w")

        image_inputs = CLIP_RESIZE(image_inputs)
        image_inputs = CLIP_NORMALIZE(image_inputs)

        rewards = scorer(image_inputs)

        if aesthetic_target is not None:
            rewards = rewards - aesthetic_target  # using L1 to keep on same scale

        return rewards.mean()

    return score_fn


def get_motion_score_fn(
    precision="fp32",
    alpha=0.6,
    beta=0.5,
    gamma=100.0,
    resize_height=256,
    resize_width=448,
    device=None,
):
    from torchvision.models.optical_flow import raft_small
    from einops import rearrange
    import torch
    import torch.nn.functional as F

    # 1. Load RAFT Small (Fast & Efficient)
    raft_model = raft_small(pretrained=True).to(device)
    raft_model.eval()
    raft_model.requires_grad_(False)

    # --- Helper: Warp Function ---
    def warp(x, flow):
        B, C, H, W = x.size()
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H, device=x.device),
                                        torch.arange(0, W, device=x.device), indexing='ij')

        vgrid_x = 2.0 * (grid_x + flow[:, 0]) / max(W - 1, 1) - 1.0
        vgrid_y = 2.0 * (grid_y + flow[:, 1]) / max(H - 1, 1) - 1.0
        vgrid = torch.stack((vgrid_x, vgrid_y), dim=3)

        return F.grid_sample(x, vgrid, align_corners=True, padding_mode='border')

    def score_fn(image_inputs: torch.Tensor, text_inputs: str = None, return_logits=False):
        # image_inputs: (B, C, T, H, W) in [0, 1] range
        b, c, t, h, w = image_inputs.shape
        if resize_height is not None and resize_width is not None:
            image_inputs = rearrange(image_inputs, 'b c t h w -> (b t) c h w')
            image_inputs = F.interpolate(
                image_inputs,
                size=(resize_height, resize_width),
                mode="bilinear",
                align_corners=False,
            )
            image_inputs = rearrange(image_inputs, '(b t) c h w -> b c t h w', b=b, t=t)
            h, w = resize_height, resize_width
        scale_factor = (h ** 2 + w ** 2) ** 0.5

        # Inputs for RAFT
        st = image_inputs[:, :, :-1, ...]  # Frame t
        ed = image_inputs[:, :, 1:, ...]  # Frame t+1
        st_flat = rearrange(st, 'b c t h w -> (b t) c h w')
        ed_flat = rearrange(ed, 'b c t h w -> (b t) c h w')

        # Normalize to [-1, 1] for RAFT
        st_norm = 2 * st_flat - 1
        ed_norm = 2 * ed_flat - 1

        # ---------------------------------------------------------
        # PASS 1: Forward Flow (WITH GRADS) -> Dynamics Reward
        # ---------------------------------------------------------
        # Gradients enabled so optimizer learns to create motion
        flow_fwd = raft_model(st_norm, ed_norm)[-1]

        flow_4d = rearrange(flow_fwd, '(b t) c h w -> b c t h w', b=b)
        mean_flow = torch.mean(flow_4d, dim=(-2, -1), keepdim=True)
        residual_flow = flow_4d - mean_flow

        # Dynamics Score
        dynamics_per_pixel = torch.norm(residual_flow, dim=1) / scale_factor
        dynamics_score = torch.tanh(dynamics_per_pixel.mean() * 20.0)

        # Smoothness Score
        if flow_4d.shape[2] > 1:
            flow_diff = flow_4d[:, :, 1:, ...] - flow_4d[:, :, :-1, ...]
            smoothness_penalty = (torch.norm(flow_diff, dim=1) / scale_factor).mean() * 20.0
            smoothness_score = 1 - torch.tanh(smoothness_penalty)
        else:
            smoothness_score = torch.tensor(1.0, device=device)

        # ---------------------------------------------------------
        # PASS 2: Backward Flow (NO GRADS) -> Warp Residual
        # ---------------------------------------------------------
        with torch.no_grad():
            # Backward flow: ed -> st
            flow_bwd = raft_model(ed_norm, st_norm)[-1]

        # Warp Frame t to t+1
        st_warped = warp(st_flat, flow_bwd)

        # Residual (The "Flicker" Map)
        residual_flat = st_warped - ed_flat
        residual = rearrange(residual_flat, '(b t) c h w -> b c t h w', b=b)

        # ---------------------------------------------------------
        # PASS 3: Robust Spectral Penalty (New!)
        # ---------------------------------------------------------
        # 1. Use 'ortho' norm for consistency across different T
        # 2. Use Power (Real^2 + Imag^2) instead of Abs
        fft_out = torch.fft.rfft(residual, dim=2, norm="ortho")
        power_spectrum = fft_out.real ** 2 + fft_out.imag ** 2

        # 3. Frequency Weighting (Soft Cutoff)
        # Instead of hard index 2, we use a ramp that penalizes highest freqs most.
        Fbins = power_spectrum.shape[2]

        # Create a weight vector [0, 0, 0.1, 0.5, ... 1.0]
        # Low freqs get 0 penalty (allow occlusion/motion errors)
        # Highest freqs get max penalty (pure flicker)
        freqs = torch.linspace(0, 1, Fbins, device=power_spectrum.device)
        cutoff = 0.25  # Ignore bottom 25% of frequencies (approx index 0-1 for T=16)

        # Ramp function: 0 below cutoff, rising quadratically to 1.0 at max freq
        weights = ((freqs - cutoff).clamp(min=0) / (1 - cutoff + 1e-8)) ** 2
        weights = weights.view(1, 1, -1, 1, 1)  # Broadcast to (B, C, F, H, W)

        # Weighted Mean of Power
        warp_freq_penalty = (power_spectrum * weights).mean()

        # ---------------------------------------------------------
        # Final Score
        # ---------------------------------------------------------
        total_score = (alpha * dynamics_score) + \
                      (beta * smoothness_score) - \
                      (gamma * warp_freq_penalty)

        return total_score

    return score_fn


def get_dino_score_fn(
    precision: str = "fp32",
    lamda: float = 0.5,
    device: Optional[Union[torch.device, str]] = None
):
    """
    Returns a differentiable `score_fn` that maps a video tensor
    (B, 3, T, H, W) ∈ [0,1] → scalar reward  ∈ (-1,1).
    """
    dino = torch.hub.load(
        'facebookresearch/dinov2',
        'dinov2_vitb14',
        pretrained=True
    ).eval()
    if device is not None:
        dino = dino.to(device)
    for p in dino.parameters():
        p.requires_grad_(False)

    dino_dev = next(dino.parameters()).device
    mean = torch.tensor([0.485, 0.456, 0.406], device=dino_dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dino_dev).view(1, 3, 1, 1)

    def preprocess(video: torch.Tensor):
        """
        video: (B,3,T,H,W) in [0,1]
        → (B·T, 3, 224, 224)  with ImageNet norm
        """
        b, c, t, h, w = video.shape
        x = rearrange(video, 'b c t h w -> (b t) c h w')
        x = FF.interpolate(x, (224, 224), mode='bicubic', align_corners=False)
        return (x - mean) / std

    def score_fn(video, text, return_logits=False):
        """
        video: (B,3,T,H,W) in [0,1]
        returns higher = more spatio-temporally consistent
        """
        b, _, t, _, _ = video.shape
        feats = dino.forward_features(preprocess(video))
        patch = feats["x_norm_patchtokens"].view(b, t, 16, 16, -1)  # (B,T,Hp,Wp,D)
        cls = feats["x_norm_clstoken"].view(b, t, -1)  # (B,T,D)

        # --- spatial term (patch neighbours) ---
        right = FF.cosine_similarity(patch[..., :-1, :, :], patch[..., 1:, :, :], dim=-1)
        bottom = FF.cosine_similarity(patch[..., :, :-1, :], patch[..., :, 1:, :], dim=-1)
        spa_reward = 0.5 * (right.mean() + bottom.mean())

        # --- temporal term (CLS token) ---
        sim_cls = FF.cosine_similarity(cls[:, 1:], cls[:, :-1], dim=-1)  # (B,T-1)
        temp_reward = sim_cls.mean()
        reward = temp_reward
        return reward

    return score_fn



def get_vjepa_score_fn(precision="fp32", device=None):
    from reward_fn.vjepa.vjepa_scorer import VJEPARewardHelper

    dtype = precision_map.get(precision, torch.float32)
    config_path = os.path.join(os.path.dirname(__file__), "vjepa/configs/vitl16.yaml")
    reward_model = VJEPARewardHelper(config_path=config_path, device=device, dtype=dtype)

    def score_fn(image_inputs: torch.Tensor, text_inputs: str = None):
        loss = reward_model(image_inputs)
        reward = 1. - loss
        return reward

    return score_fn
