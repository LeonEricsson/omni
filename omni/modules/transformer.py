import torch.nn as nn
from jaxtyping import Float
from jaxtyping import Int
from torch import Tensor

from omni.modules.attention import causal_attention_mask
from omni.modules.block import Block
from omni.modules.config import TransformerConfig
from omni.modules.norm import NORM_MAP
from omni.modules.pos_embeddings import PositionalEmbedding
from omni.modules.cache import KVCache

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
        pad_mask: Int[Tensor, "batch seq"] = None,
        kv_cache: KVCache = None,
    ) -> Float[Tensor, "batch seq vocab_size"]:
        mask = self.causal_mask
        
        if self.training:
            mask &= pad_mask[:, None, None, :]

        x = self.token_emb(x)

        x, pos_info = self.position_embedding(x)

        for i, block in enumerate(self.blocks):
            x = block(x, mask, pos_info, kv_cache, layer_idx=i)

        x = self.norm_out(x)

        x = self.vocab_proj(x)

        return x
