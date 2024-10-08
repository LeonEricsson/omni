from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex, Float
from torch import Tensor

from omni.modules.pos_embeddings import apply_rope_real
from omni.modules.norm import LayerNorm

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
        assert config.d_model % config.num_heads == 0

        self.n_heads = config.num_heads
        self.head_dim = config.d_model // (2 * config.num_heads)
        self.scale = self.head_dim**-0.5

        self.W_QKV = nn.Linear(
            config.d_model, self.head_dim * config.num_heads * 2 * 3, bias=config.attention_bias
        )

        self.W_O = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)

        self.group_norms = nn.ModuleList([LayerNorm(dim=3) for _ in range(config.num_heads)])
        #self.attn_dropout = nn.Dropout(config.attention_dropout)
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

        q = q.reshape(batch_size, seq_length, self.n_heads, 2 * self.head_dim)
        k = k.reshape(batch_size, seq_length, self.n_heads, 2 * self.head_dim)
        v = v.reshape(batch_size, seq_length, self.n_heads, 2 * self.head_dim)

        q = q.transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q1, q2 = q.split(dim=-1)
        k1, k2 = k.split(dim=-1)

        if self.pos_encoding_type == "rope":
            freq_cis: Complex[Tensor, "seq half_head_dim"] = pos_info
            q1, k1 = apply_rope_real(q1, k1, freq_cis)
            q2, k2 = apply_rope_real(q2, k2, freq_cis)

        qk1 = (q1 @ k1.transpose(2, 3)) / self.scale  # (batch, n_heads, seq, seq)
        qk1 = qk1 + mask
        
        qk2 = (q1 @ k1.transpose(2, 3)) / self.scale  # (batch, n_heads, seq, seq)
        qk2 = qk2 + mask

        output = (F.softmax(qk1) - self.lbda * F.softmax(qk2)) @ v # (batch, n_heads, seq, 2*head_dim)
        #qk = self.attn_dropout(qk)

        for i, norm in enumerate(self.group_norms):
            output[:, i] = norm(output[:, i])

        output = output * (1 - self.lbda_init)

        output = output.transpose(1, 2).reshape(batch_size, seq_length, d_model)

        output = self.W_O(output)
        output = self.res_dropout(output)

        return output