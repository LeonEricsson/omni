from typing import List

import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from omni.modules.attention import causal_attention_mask
from omni.modules.block import Block
from omni.modules.cache import KVCache
from omni.modules.config import TransformerConfig
from omni.modules.norm import NORM_MAP
from omni.modules.pos_embeddings import PositionalEmbedding


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        self.position_embedding = PositionalEmbedding(config)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])

        self.norm_out = NORM_MAP[config.normalization](config)

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
