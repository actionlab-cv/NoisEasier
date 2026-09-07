# NoisEasier for LTX-Video

This directory contains the NoisEasier runner for the LTX-Video-2B distilled
backbone. It uses an independent Python/PyTorch stack.

## Environment

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate noiseasier-ltx-video
python -m pip install -e ".[inference]" --no-deps
```

## Official model and local placement

Use the model and instructions from the official
[LTX-Video repository](https://github.com/Lightricks/LTX-Video) and
[Lightricks/LTX-Video model repository](https://huggingface.co/Lightricks/LTX-Video).

The runner first checks for locally downloaded files in:

```text
ltx-video/checkpoints/
├── ltxv-2b-0.9.8-distilled.safetensors
└── ltxv-spatial-upscaler-0.9.8.safetensors
```

If they are absent, the same filenames are resolved through the official
Hugging Face repository. Text-encoder assets follow the identifiers in the
official pipeline config.

## Single-prompt quick start

Run from this directory after installing the LTX model plus ImageReward and
ViCLIP checkpoints described in `checkpoints/README.md`.

```bash
CUDA_VISIBLE_DEVICES=0 python run_noiseasier.py \
  --prompt "A red panda walks through a bamboo forest in gentle rain" \
  --use-dno true --noise-optimization initial --dno-steps 5 \
  --reward-fns img_reward viclip --reward-weights 1 1 \
  --reward-precision amp --reward-frame-sample-count 16 \
  --gradient-checkpointing true --checkpoint-vae-decode true \
  --save-activations-on-cpu false \
  --output-path outputs/examples-ltx
```

When `--reward-latent-frames` is unset, the reward decoder receives the
complete latent trajectory. Set it to a positive integer to decode a fixed
number of evenly sampled latent frames.

The optimized MP4 and per-iteration metrics are written under
`outputs/examples-ltx/`. To process a few prompts without reloading the model:

```bash
CUDA_VISIBLE_DEVICES=0 python run_noiseasier.py \
  --prompt-file ../examples/prompts.txt \
  --use-dno true --noise-optimization initial --dno-steps 5 \
  --reward-fns img_reward viclip --reward-weights 1 1 \
  --reward-precision amp --reward-frame-sample-count 16 \
  --gradient-checkpointing true --checkpoint-vae-decode true \
  --save-activations-on-cpu false \
  --output-path outputs/examples-ltx-list
```

## Options and evaluation

Additional settings are recorded in
[`configs/noiseasier_ltx.json`](configs/noiseasier_ltx.json).
Use `python run_noiseasier.py --help` for full-trajectory optimization,
checkpointing, reward precision, and sharding options.

Multi-GPU runs assign one prompt shard to each GPU; a single-prompt command
uses one GPU.

Fixed T2V-CompBench generation prompts and seeds are under
`comp-t2v-prompts/`. Evaluate generated MP4s with the official
[T2V-CompBench](https://github.com/KaiyueSun98/T2V-CompBench) repository.
