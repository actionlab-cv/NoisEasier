# Generator and reward checkpoints

Download the required assets from the owners' official repositories or model
cards, review their licenses, and use this layout:

```text
checkpoints/
├── ms_unet_lora.pt
├── vc2_unet_lora.pt
├── vc2_model.ckpt
└── rewards/
    ├── HPS_v2_compressed.pt
    ├── ImageReward.pt
    ├── med_config.json
    └── ViClip-InternVid-10M-FLT.pth
```

Generator sources:

- ModelScope base: [ali-vilab/text-to-video-ms-1.7b](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b) (standard Hugging Face cache)
- T2V-Turbo-MS LoRA: [jiachenli-ucsb/T2V-Turbo-MS](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-MS)
- T2V-Turbo-VC2 LoRA: [jiachenli-ucsb/T2V-Turbo-VC2](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2)
- VideoCrafter2 base: [AILab-CVC/VideoCrafter](https://github.com/AILab-CVC/VideoCrafter)
- AnimateLCM: [G-U-N/AnimateLCM](https://github.com/G-U-N/AnimateLCM) and [wangfuyun/AnimateLCM](https://huggingface.co/wangfuyun/AnimateLCM) (standard Hugging Face cache)

Reward sources:

- HPSv2: [tgxs002/HPSv2](https://github.com/tgxs002/HPSv2)
- ImageReward: [THUDM/ImageReward](https://github.com/THUDM/ImageReward)
- ViCLIP / InternVideo: [OpenGVLab/InternVideo](https://github.com/OpenGVLab/InternVideo)

Set `NOISEASIER_REWARD_ROOT` to use another reward directory, or set
`NOISEASIER_HPSV2_PATH`, `NOISEASIER_IMAGE_REWARD_PATH`,
`NOISEASIER_IMAGE_REWARD_CONFIG`, and `NOISEASIER_VICLIP_PATH` individually.
The loaders also recognize the official packages' legacy HPSv2/ImageReward
caches. The motion reward follows torchvision's standard `TORCH_HOME` cache for
its official RAFT-Small weights.
