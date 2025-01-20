import torch.nn as nn
from jaxtyping import Float
from jaxtyping import Int
from torch import Tensor

from omni.modules.attention import causal_attention_mask
from omni.modules.block import Block
from omni.modules.config import TransformerConfig
from omni.modules.norm import NORM_MAP
from omni.modules.pos_embeddings import PositionalEmbedding

import torch
class KVCache:
    def __init__(self, config: TransformerConfig, device: str = None, dtype: torch.dtype = None):
        self.num_layers = config.num_layers
        self.num_kv_heads = config.num_kv_heads
        self.seq_len = config.seq_len
        self.device = device
        self.dtype = dtype
        
        head_dim = self.d_model // self.num_heads
        cache_shape = (self.num_layers, self.num_kv_heads, self.seq_len, head_dim)

        self.k = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.v = torch.zeros(cache_shape, device=device, dtype=dtype)
        
        self.cache_lengths = torch.zeros(self.num_layers, device=device, dtype=torch.int16)

    def forward(self, layer_idx, k, v):
        """Update the cache for a single layer, handling overflow by rolling."""
        cache_len = self.cache_lengths[layer_idx].item()
        new_len = k.size(1)  # seq_len dimension for the new keys
        max_len = self.seq_len_cache

        if cache_len + new_len > max_len:
            overflow = cache_len + new_len - max_len
            self.k[layer_idx, :, :-overflow, :] = self.k[layer_idx, :, overflow:, :]
            self.v[layer_idx, :, :-overflow, :] = self.v[layer_idx, :, overflow:, :]
            cache_len = max_len - new_len

        self.k[layer_idx, :, cache_len:cache_len + new_len, :] = k
        self.v[layer_idx, :, cache_len:cache_len + new_len, :] = v

        self.cache_lengths[layer_idx] = min(cache_len + new_len, max_len)

        return self.k[layer_idx, :, :self.cache_lengths[layer_idx], :], \
               self.v[layer_idx, :, :self.cache_lengths[layer_idx], :]


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        self.position_embedding = PositionalEmbedding(config)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])

        self.norm_out = NORM_MAP[config.normalization](config)

        self.vocab_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.register_buffer("causal_mask", causal_attention_mask(config.seq_len))

    def forward(
        self,
        x: Int[Tensor, "batch seq"],
        pad_mask: Int[Tensor, "batch seq"],
        kv_cache: KVCache = None,
    ) -> Float[Tensor, "batch seq vocab_size"]:
        mask = self.causal_mask & pad_mask[:, None, None, :]

        x = self.token_emb(x)

        x, pos_info = self.position_embedding(x)

        for i, block in enumerate(self.blocks):
            x = block(x, mask, pos_info, kv_cache, layer_idx=i)

        x = self.norm_out(x)

        x = self.vocab_proj(x)

        return x
