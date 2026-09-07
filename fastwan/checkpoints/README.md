# FastWan and reward checkpoints

Download the required assets from their owners, review the associated model
cards and licenses, and preserve the layout below.

```text
checkpoints/
├── FastWan2.1-T2V-1.3B-Diffusers/   # complete optional local snapshot
├── vae/
│   └── lighttaew2_1.safetensors
└── rewards/
    ├── HPS_v2_compressed.pt
    ├── ImageReward.pt
    ├── med_config.json
    └── ViClip-InternVid-10M-FLT.pth
```

- FastWan: [FastVideo/FastWan2.1-T2V-1.3B-Diffusers](https://huggingface.co/FastVideo/FastWan2.1-T2V-1.3B-Diffusers)
- LightTAE-Wan: [lightx2v/Autoencoders](https://huggingface.co/lightx2v/Autoencoders)
- ViCLIP / InternVideo: [OpenGVLab/InternVideo](https://github.com/OpenGVLab/InternVideo)
- ImageReward: [THUDM/ImageReward](https://github.com/THUDM/ImageReward)

The model snapshot may instead remain in the Hugging Face cache. Pass explicit
paths through `--model-path` and `--reward-decoder-path`. The reward directory
may be overridden with `NOISEASIER_REWARD_ROOT`, or individual files with
`NOISEASIER_HPSV2_PATH`, `NOISEASIER_IMAGE_REWARD_PATH`,
`NOISEASIER_IMAGE_REWARD_CONFIG`, and `NOISEASIER_VICLIP_PATH`.

The motion reward uses torchvision's official RAFT-Small weights and follows
the standard `TORCH_HOME` cache. Install optional reward packages from their
official repositories.
