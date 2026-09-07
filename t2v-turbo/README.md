# NoisEasier for T2V-Turbo and AnimateLCM

This directory contains NoisEasier inference runners for T2V-Turbo
(ModelScope), T2V-Turbo (VideoCrafter2), and AnimateLCM.

## Environment

Create and activate the pinned Conda environment:

```bash
conda env create -f environment.yml
conda activate vturbo
```

An editable install for external tooling can be added without changing
dependencies: `python -m pip install -e . --no-deps`.

Run commands from this directory and select a GPU explicitly.
PyTorch 2.1 requires the NumPy 1.x ABI used by this environment. Confirm that
the environment has not drifted before a GPU run:

```bash
python -c "import numpy, torch; assert numpy.__version__ == '1.26.4'; torch.from_numpy(numpy.arange(1))"
```

VC2 uses xformers attention by default. If the installed xformers binary does
not support the selected GPU, prefix the command with
`NOISEASIER_DISABLE_XFORMERS=1` to use native attention. This fallback may be
slower and use more memory.

## Official models and local placement

Download third-party weights only from the model owners' official pages:

- [T2V-Turbo repository](https://github.com/Ji4chenLi/t2v-turbo)
- [T2V-Turbo official model collection](https://huggingface.co/collections/jiachenli-ucsb/t2v-turbo)
- [ModelScope text-to-video 1.7B](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b)
- [VideoCrafter2 repository](https://github.com/AILab-CVC/VideoCrafter)
- [AnimateLCM repository](https://github.com/G-U-N/AnimateLCM)
- [AnimateLCM model card](https://huggingface.co/wangfuyun/AnimateLCM)

After downloading, arrange the two T2V-Turbo variants as follows:

```text
t2v-turbo/checkpoints/
├── ms_unet_lora.pt    # T2V-Turbo-MS unet_lora.pt
├── vc2_unet_lora.pt   # T2V-Turbo-VC2 unet_lora.pt
└── vc2_model.ckpt     # VideoCrafter2 512x320 text-to-video checkpoint
```

The ModelScope base model is resolved from its Hugging Face id. AnimateLCM
follows its official Diffusers setup and resolves `wangfuyun/AnimateLCM` plus
the base model referenced by that model card through the Hugging Face cache.

## Single-prompt quick start

Run these commands after installing the learned reward checkpoints described in
`checkpoints/README.md`. Each command optimizes one prompt.

### T2V-Turbo (ModelScope)

```bash
CUDA_VISIBLE_DEVICES=0 python run_ms.py \
  --unet_dir checkpoints/ms_unet_lora.pt \
  --prompt "A red panda walks through a bamboo forest in gentle rain" \
  --use_dno true --noise_optimization full --dno_steps 5 \
  --reward_fns hpsv2 img_reward motion viclip \
  --reward_weights 2 1 1 2 --reward_precision fp16 \
  --checkpoint_vae_decode true \
  --save_metrics true --save_path outputs/examples-ms --run_id quickstart
```

### T2V-Turbo (VideoCrafter2)

```bash
CUDA_VISIBLE_DEVICES=0 python run_vc2.py \
  --unet_dir checkpoints/vc2_unet_lora.pt \
  --base_model_dir checkpoints/vc2_model.ckpt \
  --prompt "A red panda walks through a bamboo forest in gentle rain" \
  --use_dno true --noise_optimization full --dno_steps 5 \
  --reward_fns hpsv2 img_reward motion viclip \
  --reward_weights 2 1 1 2 --reward_precision fp16 \
  --checkpoint_vae_decode true \
  --save_metrics true --save_path outputs/examples-vc2 --run_id quickstart
```

### AnimateLCM

```bash
CUDA_VISIBLE_DEVICES=0 python run_animatelcm.py \
  --prompt "A red panda walks through a bamboo forest in gentle rain" \
  --height 512 --width 512 \
  --use_dno true --noise_optimization full --dno_steps 5 \
  --reward_fns hpsv2 img_reward motion viclip \
  --reward_weights 2 1 1 2 --reward_precision fp16 \
  --checkpoint_vae_decode true \
  --save_metrics true --save_path outputs/examples-animatelcm --run_id quickstart
```

### A few prompts

All three runners also accept a small TXT/JSON list. The model and rewards load
once, then each prompt receives seed `--seed + index`:

```bash
CUDA_VISIBLE_DEVICES=0 python run_ms.py \
  --unet_dir checkpoints/ms_unet_lora.pt \
  --prompt-file ../examples/prompts.txt
```

## Rewards and evaluation

The runners support HPSv2, ImageReward, ViCLIP, and motion reward. Use
`python RUNNER.py --help` for precision, reward calibration, memory-saving,
sharding, and output options. The default configuration is recorded in
[`configs/noiseasier_t2v.json`](configs/noiseasier_t2v.json).

Prompt metadata and fixed seeds for generation are under `vbench_prompts/` and
`comp-t2v-prompts/`. Generated MP4s can be evaluated with the official
[VBench](https://github.com/Vchitect/VBench) or
[T2V-CompBench](https://github.com/KaiyueSun98/T2V-CompBench) setup.
