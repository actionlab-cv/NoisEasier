"""FastWan video-sparse-attention kernels used by NoisEasier."""

from .block_sparse_attn import block_sparse_attn, block_sparse_attn_from_indices
from .ops import video_sparse_attn
from .version import __version__

__all__ = [
    "video_sparse_attn",
    "block_sparse_attn",
    "block_sparse_attn_from_indices",
    "__version__",
]
