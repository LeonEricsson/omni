from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex, Float, Int
from torch import Tensor

from omni.modules.pos_embeddings import apply_rope_real

AttentionType = Literal["mha", "gqa"]


def causal_attention_mask(sequence_length: int, dtype=torch.float32):
    mask = torch.tril(torch.ones((1, 1, sequence_length, sequence_length), dtype=dtype))
    mask = mask.masked_fill(mask == 0, float("-inf")) 
    return mask


class GQA(nn.Module):
    """
    Grouped Query Attention (GQA) module that reduces key/value heads while maintaining query heads.
    GQA generalizes Multi-Head Attention by allowing fewer key/value heads than query heads,
    where each key/value head is shared across multiple query heads. GQA is equivalent to
    MHA when num_kv_heads == num_heads. GQA is equivalent to MQA when num_kv_heads == 1.

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
        assert config.num_heads % config.num_kv_heads == 0
        assert config.d_model % config.num_heads == 0

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
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
        kv_cache,
    ):
        batch_size, seq_length, d_model = x.size()

        q = self.W_Q(x).unflatten(-1, (self.num_heads, self.head_dim))
        kv = self.W_KV(x).unflatten(-1, (2, self.num_kv_heads, self.head_dim))
        k, v = kv[:, :, 0], kv[:, :, 1]

        q = q.transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.forward(k, v)

        k = torch.repeat_interleave(k, self.kv_groups, dim=1)
        v = torch.repeat_interleave(v, self.kv_groups, dim=1)

        if self.pos_encoding_type == "rope":
            freq_cis: Complex[Tensor, "seq half_head_dim"] = pos_info
            q, k = apply_rope_real(q, k, freq_cis)

        # to support single step inference
        start = k.shape[2] - q.shape[2]
        end = k.shape[2]
        mask = mask[:, :, start:end, :k.shape[2]]

        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
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
        assert config.d_model % config.num_heads == 0

        self.n_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.scale = self.head_dim**-0.5

        self.W_QKV = nn.Linear(
            config.d_model,
            self.head_dim * config.num_heads * 3,
            bias=config.attention_bias,
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
        kv_cache,
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
            k, v = kv_cache.forward(k, v)

        if self.pos_encoding_type == "rope":
            freq_cis: Complex[Tensor, "seq half_head_dim"] = pos_info
            q, k = apply_rope_real(q, k, freq_cis)

        # to support single step inference
        start = k.shape[2] - q.shape[2]
        end = k.shape[2]
        mask = mask[:, :, start:end, :k.shape[2]]

        if self.flash_attn and not self.pos_encoding_type == "alibi":
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                attn_mask=mask
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


ATTN_MAP = {
    "mha": MHA,
    "gqa": GQA,
}
