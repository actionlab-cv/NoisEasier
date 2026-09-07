# LTX-Video and reward checkpoints

Download the required assets from the owners' official repositories or model
cards, review their licenses, and use this layout:

```text
checkpoints/
├── ltxv-2b-0.9.8-distilled.safetensors
├── ltxv-spatial-upscaler-0.9.8.safetensors
└── rewards/
    ├── HPS_v2_compressed.pt
    ├── ImageReward.pt
    ├── med_config.json
    └── ViClip-InternVid-10M-FLT.pth
```

- LTX-Video: [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video)
- HPSv2: [tgxs002/HPSv2](https://github.com/tgxs002/HPSv2)
- ImageReward: [THUDM/ImageReward](https://github.com/THUDM/ImageReward)
- ViCLIP / InternVideo: [OpenGVLab/InternVideo](https://github.com/OpenGVLab/InternVideo)

LTX files may remain in the Hugging Face cache. Set
`NOISEASIER_REWARD_ROOT` to use another reward directory, or set
`NOISEASIER_HPSV2_PATH`, `NOISEASIER_IMAGE_REWARD_PATH`,
`NOISEASIER_IMAGE_REWARD_CONFIG`, and `NOISEASIER_VICLIP_PATH` individually.
The loaders also recognize the official packages' legacy HPSv2/ImageReward
caches. The motion reward follows torchvision's standard `TORCH_HOME` cache for
its official RAFT-Small weights.
