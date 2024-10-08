from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex, Float
from torch import Tensor

from omni.modules.pos_embeddings import apply_rope_real

class DifferentialAttention(nn.Module):
    def __init__(self, config):
        """
        Differential attention

        Args:
        config (TransformerConfig): Configuration dataclass containing:
            - num_heads: Number of attention heads
            - d_model: Model dimension
            - bias: Whether to use bias in linear layers
            - dropout: Dropout probability for attention and residual connections

        """
        super().__init__()
        self.n_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.scale = self.head_dim**-0.5

        self.W_QKV = nn.Linear(
            config.d_model, config.d_model * 3, bias=config.attention_bias
        )
        self.W_O = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.res_dropout = nn.Dropout(config.attention_dropout)

        self.flash_attn: bool = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )

        self.pos_encoding_type = config.pos_encoding_type

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        mask: Float[Tensor, "1 1 seq seq"],
        pos_info: Optional[Tensor],
    ):
        batch_size, seq_length, d_model = x.size()

        x = self.W_QKV(x)

        q, k, v = x.chunk(3, dim=-1)

        q = q.reshape(batch_size, seq_length, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_length, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, seq_length, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.forward(layer_idx, k, v)

        if self.pos_encoding_type == "rope":
            freq_cis: Complex[Tensor, "seq half_head_dim"] = pos_info
            q, k = apply_rope_real(q, k, freq_cis)

        if self.flash_attn and not self.pos_encoding_type == "alibi":
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            qk = (q @ k.transpose(2, 3)) / self.scale  # (batch, n_heads, seq, seq)
            if self.pos_encoding_type == "alibi":
                alibi: Float[Tensor, "n_heads seq"] = pos_info
                qk = qk + alibi[None, :, :, None]  # apply bias along key dimension
            qk = qk + mask

            qk = F.softmax(qk, dim=-1)
            qk = self.attn_dropout(qk)

            output = qk @ v

        output = output.transpose(1, 2).reshape(batch_size, seq_length, d_model)

        output = self.W_O(output)
        output = self.res_dropout(output)

        return output