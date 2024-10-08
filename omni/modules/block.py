import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from omni.modules.attention import ATTN_MAP
from omni.modules.config import TransformerConfig
from omni.modules.mlp import MLP_MAP
from omni.modules.norm import NORM_MAP


class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.attn = ATTN_MAP[config.attention](config)
        self.mlp = MLP_MAP[config.mlp](config)
        self.norm1 = NORM_MAP[config.normalization](config)
        self.norm2 = NORM_MAP[config.normalization](config)

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        mask: Float[Tensor, "1 1 seq seq"],
        pos_info,
        kv_cache,
        layer_idx: int = None,
    ):
        """Pre-norm Transformer block."""

        # Self-attention block
        attn_out = self.attn(self.norm1(x), mask, pos_info, kv_cache, layer_idx)
        x = x + attn_out  # add to residual stream

        # MLP block
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out  # add to residual stream

        return x
