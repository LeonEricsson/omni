from typing import List

import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from dyt import DyT

from omni.modules.attention import causal_attention_mask, ATTN_MAP
from omni.modules.mlp import MLP_MAP
from omni.modules.block import Block
from omni.modules.cache import KVCache
from omni.modules.config import TransformerConfig
from omni.modules.pos_embeddings import PositionalEmbedding


class DyTTransformer(nn.Module):
    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        self.position_embedding = PositionalEmbedding(config)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])

        self.norm_out = DyT(config)

        self.vocab_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.weight_tying:
            self.vocab_proj.weight = self.token_emb.weight
            nn.init.normal_(
                self.token_emb.weight, mean=0.0, std=1.0 / (config.d_model**0.5)
            )

        self.register_buffer("causal_mask", causal_attention_mask(config.seq_len))

    def get_kv_cache_layer(self, kv_cache, layer_idx):
        if not kv_cache:
            return None

        return kv_cache[layer_idx]

    def forward(
        self,
        x: Int[Tensor, "batch seq"],
        pad_mask: Int[Tensor, "batch seq"] = None,
        kv_cache: List[KVCache] = [],
    ) -> Float[Tensor, "batch seq vocab_size"]:
        mask = self.causal_mask

        if self.training:
            pad_mask = (pad_mask - 1)[:, None, None, :].float()
            pad_mask = pad_mask.masked_fill(pad_mask == -1, float("-inf"))
            mask = mask + pad_mask

        x = self.token_emb(x)

        x, pos_info = self.position_embedding(x)

        for i, block in enumerate(self.blocks):
            x = block(x, mask, pos_info, self.get_kv_cache_layer(kv_cache, i))

        x = self.norm_out(x)

        x = self.vocab_proj(x)

        return x

class DyTBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.attn = ATTN_MAP[config.attention](config)
        self.mlp = MLP_MAP[config.mlp](config)
        self.norm1 = DyT(config)
        self.norm2 = DyT(config)

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        mask: Float[Tensor, "1 1 seq seq"],
        pos_info,
        kv_cache,
    ):
        """Pre-norm Transformer block."""

        # Self-attention block
        attn_out = self.attn(self.norm1(x), mask, pos_info, kv_cache)
        x = x + attn_out  # add to residual stream

        # MLP block
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out  # add to residual stream

        return x
