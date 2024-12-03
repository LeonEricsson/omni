import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from omni.modules.attention import ATTN_MAP
from omni.modules.config import TransformerConfig
from omni.modules.mlp import MLP_MAP
from omni.modules.norm import NORM_MAP

class nBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.attn = ATTN_MAP[config.attention](config)
        self.mlp = MLP_MAP[config.mlp](config)

        self.alpha_A = nn.Parameter(torch.full((config.d_model,), 5e-2))
        self.alpha_M = nn.Parameter(torch.full((config.d_model,), 5e-2))

        self.norm = lambda x: F.normalize(x, dim=-1)

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        mask: Float[Tensor, "1 1 seq seq"],
        pos_info,
        kv_cache,
    ):
        x_A = self.norm(self.attn(x, mask, pos_info, kv_cache))
        x = self.norm(x + self.alpha_A * (x_A - x)) # eq. 10

        x_M = self.norm(self.mlp(x))
        x = self.norm(x + self.alpha_M * (x_M - x)) # eq. 11

        return x
