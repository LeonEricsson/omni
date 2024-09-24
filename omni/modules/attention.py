from typing import Literal
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex
from jaxtyping import Float
from jaxtyping import Int
from torch import Tensor

from omni.modules.pos_embeddings import apply_rope

AttentionType = Literal["mha", "gqa"]


def causal_attention_mask(sequence_length: Int) -> Float[Int, "1 1 seq seq"]:
    mask = torch.tril(
        torch.ones((1, 1, sequence_length, sequence_length), dtype=torch.int32)
    )
    return mask * 1 + (1 - mask) * -10000


class GQA(nn.Module):
    """
    Grouped Query Attention (GQA) module that reduces key/value heads while maintaining query heads.
    GQA generalizes Multi-Head Attention by allowing fewer key/value heads than query heads,
    where each key/value head is shared across multiple query heads. GQA is equivalent to
    MHA when num_kv_heads == num_heads.

    Args:
    config: Configuration containing:
        - num_heads: Number of query attention heads
        - num_kv_heads: Number of key/value attention heads (must divide num_heads)
        - d_model: Model dimension
        - attention_bias: Whether to use bias in linear projections
        - attention_dropout: Dropout probability for attention and residual connections
        - pos_encoding_type: Type of positional encoding
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.d_model // config.num_heads
        self.scale = self.head_dim**-0.5

        self.W_Q = nn.Linear(
            config.d_model,
            (self.head_dim * self.num_heads),
            bias=config.attention_bias,
        )
        self.W_KV = nn.Linear(
            config.d_model,
            2 * (self.head_dim * config.num_kv_heads),
            bias=config.attention_bias,
        )

        self.W_O = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.attention_bias,
        )

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

        q = self.W_Q(x)
        kv = self.W_KV(x)

        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k = torch.repeat_interleave(k, self.kv_groups, dim=1)
        v = torch.repeat_interleave(v, self.kv_groups, dim=1)

        if self.pos_encoding_type == "rope":
            freq_cis: Complex[Tensor, "seq half_head_dim"] = pos_info
            q, k = apply_rope(q, k, freq_cis)

        if self.flash_attn:
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
            qk = qk + mask

            qk = F.softmax(qk, dim=-1)
            qk = self.attn_dropout(qk)

            output = qk @ v

        output = output.transpose(1, 2).reshape(batch_size, seq_length, d_model)

        output = self.W_O(output)
        output = self.res_dropout(output)

        return output


class MHA(nn.Module):
    def __init__(self, config):
        """
        Multi-Head Attention implementing scaled dot-product attention.

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

        if self.pos_encoding_type == "rope":
            freq_cis: Complex[Tensor, "seq half_head_dim"] = pos_info
            q, k = apply_rope(q, k, freq_cis)

        if self.flash_attn:
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
            qk = qk + mask

            qk = F.softmax(qk, dim=-1)
            qk = self.attn_dropout(qk)

            output = qk @ v

        output = output.transpose(1, 2).reshape(batch_size, seq_length, d_model)

        output = self.W_O(output)
        output = self.res_dropout(output)

        return output


ATTN_MAP = {
    "mha": MHA,
    "gqa": GQA,
}
