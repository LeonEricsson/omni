from dataclasses import dataclass

import torch.nn as nn
from jaxtyping import Bool, Float, Int
from mla import MLA
from mla_rope import MLAPositionalEmbedding
from torch import Tensor

from omni.modules.activations import ActivationFunction
from omni.modules.attention import causal_attention_mask
from omni.modules.cache import KVCache
from omni.modules.config import TransformerConfig
from omni.modules.mlp import MLP_MAP, MLPType
from omni.modules.norm import NORM_MAP, NormalizationType


@dataclass
class MLAConfig:
    vocab_size: Int
    seq_len: Int
    d_model: Int
    num_heads: Int
    num_layers: Int
    head_dim: Int
    head_dim_decoupled_qk: Int
    hidden_dim: Int = None

    # components
    activation_fn: ActivationFunction = "silu"
    mlp: MLPType = "mlp_swiglu"
    normalization: NormalizationType = "rmsnorm"

    mlp_bias: Bool = False
    mlp_dropout: Float = False
    rope_theta: Float = 10000.0
    norm_eps: Float = 1e-5
    weight_tying: Bool = False

    d_ckv: Int = None
    d_cq: Int = None
    attention_dropout: Float = 0.0


class MLABlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.attn = MLA(config)
        self.mlp = MLP_MAP[config.mlp](config)
        self.norm1 = NORM_MAP[config.normalization](config)
        self.norm2 = NORM_MAP[config.normalization](config)

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        mask: Float[Tensor, "1 1 seq seq"],
        pos_info,
        kv_cache,
        layer_idx,
    ):
        """Pre-norm Transformer block."""

        # Self-attention block
        attn_out = self.attn(self.norm1(x), mask, pos_info, kv_cache, layer_idx)
        x = x + attn_out  # add to residual stream

        # MLP block
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out  # add to residual stream

        return x


class MLATransformer(nn.Module):
    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        self.position_embedding = MLAPositionalEmbedding(config)

        self.blocks = nn.ModuleList(
            [MLABlock(config) for _ in range(config.num_layers)]
        )

        self.norm_out = NORM_MAP[config.normalization](config)

        self.vocab_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.weight_tying:
            self.vocab_proj.weight = self.token_emb.weight

        self.register_buffer("causal_mask", causal_attention_mask(config.seq_len))

    def forward(
        self,
        x: Int[Tensor, "batch seq"],
        pad_mask: Int[Tensor, "batch seq"] = None,
        kv_cache: KVCache = None,
    ) -> Float[Tensor, "batch seq vocab_size"]:
        mask = self.causal_mask

        if self.training:
            mask = mask & pad_mask[:, None, None, :]

        x = self.token_emb(x)

        x, pos_info = self.position_embedding(x)

        for i, block in enumerate(self.blocks):
            x = block(x, mask, pos_info, kv_cache, layer_idx=i)

        x = self.norm_out(x)

        x = self.vocab_proj(x)

        return x
