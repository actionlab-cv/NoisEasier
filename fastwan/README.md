# NoisEasier for FastWan

This directory contains the NoisEasier runner and required FastVideo components
for the FastWan2.1-T2V-1.3B backbone. It uses an independent Python/PyTorch
stack.

## Environment

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate noiseasier-fastwan
```

Install the bundled VSA kernel followed by the runner:

```bash
uv pip install -e fastvideo-kernel
uv pip install -e .
```

On Linux, the bundled `pyproject.toml` installs the CUDA 12.8 PyTorch wheels
selected by FastVideo. The host NVIDIA driver must support that runtime.
FlashAttention is optional for this runner. ViCLIP is the default learned
reward. ImageReward is optional because its published package metadata pins an
older `timm` than FastVideo; follow the compatibility notes in the checkpoint
guide before enabling it. The runner imports only ImageReward's inference API,
so its optional ReFL/wandb training stack is not required.

## Official model and local placement

Use the upstream [FastVideo repository](https://github.com/hao-ai-lab/FastVideo)
and the official
[FastWan model card](https://huggingface.co/FastVideo/FastWan2.1-T2V-1.3B-Diffusers).
By default, `run_noiseasier.py` resolves that Hugging Face id and uses the
standard cache. To keep an explicit local copy, download the complete snapshot
into the following directory and pass it with `--model-path`:

```text
fastwan/checkpoints/FastWan2.1-T2V-1.3B-Diffusers/
```

The differentiable reward path additionally uses the lightweight
`lighttaew2_1.safetensors` decoder from the official
[lightx2v/Autoencoders repository](https://huggingface.co/lightx2v/Autoencoders).
Place it at:

```text
fastwan/checkpoints/vae/lighttaew2_1.safetensors
```

Then pass `--reward-decoder-path checkpoints/vae/lighttaew2_1.safetensors`, or
set `FASTWAN_TINY_VAE_PATH` to that file.

## Single-prompt quick start

Run from this directory after installing the model, ViCLIP checkpoint, and
lightweight decoder described above:

```bash
CUDA_VISIBLE_DEVICES=0 python run_noiseasier.py \
  --mode dno --noise-optimization full --dno-steps 5 \
  --prompt "A red panda walks through a bamboo forest in gentle rain" \
  --reward-fns viclip --reward-weights 1 \
  --reward-precision amp \
  --reward-decoder lighttaew2_1 \
  --reward-decoder-path checkpoints/vae/lighttaew2_1.safetensors \
  --output-path outputs/examples-fastwan
```

For a local model snapshot, add:

```bash
--model-path checkpoints/FastWan2.1-T2V-1.3B-Diffusers
```

The optimized MP4 and metrics are written under `outputs/examples-fastwan/`.
To process a few prompts without reloading the model:

```bash
CUDA_VISIBLE_DEVICES=0 python run_noiseasier.py \
  --mode dno --prompt-file ../examples/prompts.txt --dno-steps 5 \
  --reward-decoder-path checkpoints/vae/lighttaew2_1.safetensors \
  --output-path outputs/examples-fastwan-list
```

## Learned rewards and evaluation

The runner supports ViCLIP and ImageReward. Install their checkpoints as
described in [`checkpoints/README.md`](checkpoints/README.md).
Additional settings are recorded in
[`configs/noiseasier_fastwan.json`](configs/noiseasier_fastwan.json).

Multi-GPU runs assign one prompt shard to each GPU; a single-prompt command
uses one GPU. The bundled VSA implementation uses differentiable Triton kernels
and does not require a native extension build.

Prompt metadata and fixed seeds are included for generation. Evaluate the
resulting MP4s with the official [VBench](https://github.com/Vchitect/VBench)
or [T2V-CompBench](https://github.com/KaiyueSun98/T2V-CompBench) setup. Use
`python run_noiseasier.py --help` for the full runner interface.
