from dataclasses import dataclass
from typing import List

import torch.nn as nn
from diff_attention import DifferentialAttention
from jaxtyping import Bool, Float, Int
from torch import Tensor

from omni.modules.activations import ActivationFunction
from omni.modules.attention import causal_attention_mask
from omni.modules.cache import KVCache
from omni.modules.config import TransformerConfig
from omni.modules.mlp import MLP_MAP, MLPType
from omni.modules.norm import NORM_MAP, NormalizationType
from omni.modules.pos_embeddings import PositionalEmbedding, PositionEmbeddingScheme


@dataclass
class DiffConfig:
    vocab_size: Int
    seq_len: Int
    d_model: Int
    num_layers: Int
    head_dim: Int
    num_heads: Int = None
    num_kv_heads: Int = None
    hidden_dim: Int = None

    # components
    pos_encoding_type: PositionEmbeddingScheme = "rope"
    activation_fn: ActivationFunction = "silu"
    mlp: MLPType = "mlp_swiglu"
    normalization: NormalizationType = "rmsnorm"

    mlp_bias: Bool = False
    mlp_dropout: Float = False
    attention_dropout: Float = 0.1
    attention_bias: Bool = True
    weight_tying: Bool = False
    rope_theta: Float = 10000.0
    norm_eps: Float = 1e-5


class DiffBlock(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()

        self.attn = DifferentialAttention(config, layer_idx)
        self.mlp = MLP_MAP[config.mlp](config)
        self.norm1 = NORM_MAP[config.normalization](config)
        self.norm2 = NORM_MAP[config.normalization](config)

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


class DiffTransformer(nn.Module):
    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        self.position_embedding = PositionalEmbedding(config)

        self.blocks = nn.ModuleList(
            [DiffBlock(config, layer_idx) for layer_idx in range(config.num_layers)]
        )

        self.norm_out = NORM_MAP[config.normalization](config)

        self.vocab_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.weight_tying:
            self.vocab_proj.weight = self.token_emb.weight

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
