# FastWan VSA kernel subset

This directory contains the Python/Triton video-sparse-attention path required
by NoisEasier's FastWan2.1-T2V-1.3B runner. It is adapted from the
[FastVideo](https://github.com/hao-ai-lab/FastVideo) kernel package and retains
its Apache-2.0 license.

Install it from the `fastwan/` directory before installing the runner:

```bash
uv pip install -e fastvideo-kernel
```

The VSA implementation uses Triton and does not require a machine-specific
native extension build.
