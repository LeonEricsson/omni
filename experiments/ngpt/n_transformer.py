from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from n_attention import causal_attention_mask, nGQA, nMHA
from n_mlp import nMLPSwiGLU
from torch import Tensor

from omni.modules.cache import KVCache
from omni.modules.pos_embeddings import PositionalEmbedding


@dataclass
class nConfig:
    vocab_size: Int
    seq_len: Int
    d_model: Int
    num_heads: Int
    num_kv_heads: Int
    num_layers: Int
    hidden_dim: Int = None

    pos_encoding_type = "rope"

    mlp_bias: Bool = True
    mlp_dropout: Optional[Float] = None
    attention_bias: Bool = True
    attention_dropout: Optional[Float] = None
    weight_tying: Bool = False
    rope_theta: Float = 10000.0

    alpha_init: Float = 0.125  # on order 1 / num_layers


class nTransformer(nn.Module):
    def __init__(self, config: nConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        self.position_embedding = PositionalEmbedding(config)

        self.blocks = nn.ModuleList([nBlock(config) for _ in range(config.num_layers)])

        self.vocab_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.s_z = nn.Parameter(torch.ones(config.vocab_size))
        self.register_buffer("s_z_scale", torch.tensor(config.d_model**-0.5))

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

    @torch.no_grad()
    def normalize_weights(self):
        self.token_emb.weight.data = F.normalize(self.token_emb.weight.data, dim=-1)
        self.vocab_proj.weight.data = F.normalize(self.vocab_proj.weight.data, dim=-1)

        for block in self.blocks:
            block.normalize_weights()

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

        x = self.vocab_proj(x)

        x = x * self.s_z * self.s_z_scale
        return x


class nBlock(nn.Module):
    def __init__(self, config: nConfig):
        super().__init__()

        self.attn = nMHA(config)
        self.mlp = nMLPSwiGLU(config)

        self.alpha_A = nn.Parameter(torch.full((config.d_model,), config.alpha_init))
        self.alpha_M = nn.Parameter(torch.full((config.d_model,), config.alpha_init))

        self.register_buffer("alpha_scale", torch.tensor(config.d_model**-0.5))

        self.norm = lambda x: F.normalize(x, dim=-1)

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        mask: Float[Tensor, "1 1 seq seq"],
        pos_info,
        kv_cache,
    ):
        alpha_A = self.alpha_A * self.alpha_scale
        alpha_M = self.alpha_M * self.alpha_scale

        x_A = self.norm(self.attn(x, mask, pos_info, kv_cache))
        x = self.norm(x + alpha_A * (x_A - x))  # eq. 10

        x_M = self.norm(self.mlp(x))
        x = self.norm(x + alpha_M * (x_M - x))  # eq. 11

        return x
