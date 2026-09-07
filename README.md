# NoisEasier

Official implementation of **NoisEasier: Test-Time Noise Optimization for
Text-to-Video Generation**.

[![Paper](https://img.shields.io/badge/Paper-ECCV%202026-b31b1b)](https://arxiv.org/pdf/2608.30194)
[![Project Page](https://img.shields.io/badge/Project-Page-1673b6)](https://yujiangpu20.github.io/noiseasier/)

Accepted to ECCV 2026.

NoisEasier freezes a pretrained video generator and optimizes its stochastic
inputs at inference time with differentiable rewards. For short-step video
consistency models, the implementation can optimize both the initial latent and
the intermediate perturbations along the denoising trajectory.

This repository provides model adapters, inference runners, prompt metadata,
and environment definitions for five video-generation backbones.

## Supported backbones

| Backbone | Directory | Environment | Source |
|---|---|---|---|
| T2V-Turbo (ModelScope) | `t2v-turbo/` | `vturbo` | [T2V-Turbo](https://github.com/Ji4chenLi/t2v-turbo) |
| T2V-Turbo (VideoCrafter2) | `t2v-turbo/` | `vturbo` | [T2V-Turbo](https://github.com/Ji4chenLi/t2v-turbo), [VideoCrafter2](https://github.com/AILab-CVC/VideoCrafter) |
| AnimateLCM | `t2v-turbo/` | `vturbo` | [AnimateLCM](https://github.com/G-U-N/AnimateLCM) |
| LTX-Video-2B distilled | `ltx-video/` | `noiseasier-ltx-video` | [LTX-Video](https://github.com/Lightricks/LTX-Video) |
| FastWan2.1-T2V-1.3B | `fastwan/` | `noiseasier-fastwan` | [FastVideo](https://github.com/hao-ai-lab/FastVideo) |

The T2V-Turbo and AnimateLCM runners share one dependency stack. LTX-Video and
FastWan use separate environments because they require different Python,
PyTorch, CUDA, and binary-extension versions. Multi-GPU jobs shard prompts
across devices; single-prompt commands use one GPU.

## Layout

```text
NoisEasier/
├── t2v-turbo/   # T2V-Turbo MS/VC2 and AnimateLCM runners
├── ltx-video/   # LTX-Video runner
└── fastwan/     # FastWan runner and required FastVideo subset
```

## Installation and weights

Follow the README inside the selected subproject:

- [`t2v-turbo/README.md`](t2v-turbo/README.md)
- [`ltx-video/README.md`](ltx-video/README.md)
- [`fastwan/README.md`](fastwan/README.md)

Each guide links to the model owner's official repository or model card and
maps the downloaded files to the paths expected by NoisEasier. Hugging Face
model identifiers may also be used directly where the upstream loader supports
them.

## Quick start

NoisEasier runs on one user prompt by default. Each subproject README provides
a complete command using that backbone's learned rewards and model-specific
memory settings. VBench and T2V-CompBench are separate, opt-in reproduction
modes; they are never launched by the commands below.

Run one prompt with `--prompt`:

```bash
cd t2v-turbo
python run_ms.py --prompt "A red panda walks through a bamboo forest in gentle rain."
```

Run a few prompts with one UTF-8 prompt per line:

```bash
python run_ms.py --prompt-file ../examples/prompts.txt
```

The TXT loader ignores blank lines and lines beginning with `#`. JSON files
may be a string list or an object of the form `{"prompts": [...]}`. Model
weights and learned reward checkpoints must be installed first as described in
the selected subproject.

## Evaluation

Generation metadata for VBench and T2V-CompBench is included with the runners.
Evaluate generated MP4s with the official projects:

- [VBench](https://github.com/Vchitect/VBench)
- [T2V-CompBench](https://github.com/KaiyueSun98/T2V-CompBench)

## Acknowledgements

We thank the authors and contributors of the open-source projects that make
this implementation possible:

- Video generation: [T2V-Turbo](https://github.com/Ji4chenLi/t2v-turbo),
  [ModelScope text-to-video](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b),
  [VideoCrafter2](https://github.com/AILab-CVC/VideoCrafter),
  [AnimateLCM](https://github.com/G-U-N/AnimateLCM),
  [LTX-Video](https://github.com/Lightricks/LTX-Video), and
  [FastVideo/FastWan](https://github.com/hao-ai-lab/FastVideo).
- Differentiable rewards and lightweight decoders:
  [HPSv2](https://github.com/tgxs002/HPSv2),
  [ImageReward](https://github.com/THUDM/ImageReward),
  [InternVideo/ViCLIP](https://github.com/OpenGVLab/InternVideo),
  [OpenCLIP](https://github.com/mlfoundations/open_clip),
  [TAESD/TAEHV](https://github.com/madebyollin/taehv), and
  [LightTAE-Wan](https://huggingface.co/lightx2v/Autoencoders).
- Evaluation: [VBench](https://github.com/Vchitect/VBench) and
  [T2V-CompBench](https://github.com/KaiyueSun98/T2V-CompBench).

## Citation

If you find NoisEasier useful, please cite:

```bibtex
@article{pu2026noiseasier,
  title   = {NoisEasier: Test-Time Noise Optimization for Text-to-Video Generation},
  author  = {Pu, Yujiang and Kong, Yu},
  journal = {arXiv preprint arXiv:2608.30194},
  year    = {2026}
}
```
