from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex, Float
from torch import Tensor

from omni.modules.pos_embeddings import apply_rope_real


class MLA(nn.Module):
    """
    Multi-head Latent Attention attempts to reduce the KV cache without performance comprimise by
    implementing a low-rank key-value joint compression strategy. Emperically it achivies superior
    performance to MHA, while significantly reducing the memory requirements during inference.

    Args:
    config: Configuration containing:
        - num_heads (int): Number of query attention heads
        - d_model (int): Model dimension
        - head_dim (int): Dimension of each attention head. As opposed to past attention mechanisms, the per-head dimension
        does not increase the KV cache. As such, head_dim is typically set to > d_model // num_heads. DeepSeek use
        head_dim = 3 * d_model // num_heads.
        - d_ckv (int, optional): Low-rank (latent) KV dimension. d_ckv << (head_dim * num_heads). Defaults to 4 * head_dim.
        - d_cq (int, optional): Low-rank (latent) Q dimension. d_cq << (head_dim * num_heads). Defaults to 12 * num_heads.
        - head_dim_decoupled_qk (int, optional): Decoupled QK per-head dimension. Defaults to head_dim // 2.
        - attention_bias (bool): Whether to use bias in linear projections
        - attention_dropout (float): Dropout probability for attention and residual connections
        - pos_encoding_type: (PositionEmbeddingScheme) Type of positional encoding. Has to be "rope".

    References:
         - "DeepSeek V2" (https://arxiv.org/abs/2405.04434)
         - "DeepSeek V3" (https://arxiv.org/abs/2412.19437)
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        if config.d_ckv is None:
            config.d_ckv = 4 * self.head_dim

        if config.d_cq is None:
            config.d_cq = 12 * self.head_dim

        if config.head_dim_decoupled_qk is None:
            self.head_dim_decoupled_qk = self.head_dim // 2

        self.scale = (self.head_dim + self.head_dim_decoupled_qk) ** -0.5

        self.W_DQ = nn.Linear(
            config.d_model,
            config.d_cq,
            bias=config.attention_bias,
        )

        self.W_UQ = nn.Linear(
            config.d_cq,
            (self.head_dim * self.num_heads),
            bias=config.attention_bias,
        )

        self.W_DWK = nn.Linear(
            config.d_model,
            config.d_ckv,
            bias=config.attention_bias,
        )

        self.W_UK = nn.Linear(
            config.d_ckv,
            (self.head_dim * self.num_heads),
            bias=config.attention_bias,
        )

        self.W_UV = nn.Linear(
            config.d_ckv,
            (self.head_dim * self.num_heads),
            bias=config.attention_bias,
        )

        self.W_QR = nn.Linear(
            config.d_model,
            (self.head_dim * self.num_heads),
            bias=config.attention_bias,
        )

        self.W_KR = nn.Linear(
            config.d_model,
            self.head_dim,
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

        assert config.pos_encoding_type is "rope"

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        mask: Float[Tensor, "1 1 seq seq"],
        pos_info: Optional[Tensor],
    ):
        batch_size, seq_length, d_model = x.size()

        # Compressed "content" dimensions
        c_Q = self.W_DQ(x)
        c_KV = self.W_DWK(x)  # cache during inference

        q_C = self.W_UQ(c_Q)
        k_C = self.W_UK(c_KV)
        v = self.W_UV(c_KV)

        q_C = q_C.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k_C = k_C.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_length, self.num_heads, self.head_dim)

        q_C = q_C.transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        k_C = k_C.transpose(1, 2)
        v = v.transpose(1, 2)

        # Decoupled RoPE dimensions
        q_R = self.W_QR(c_Q)  # (batch, seq, num_heads * head_dim)
        q_R = q_R.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        q_R = q_R.transpose(1, 2)
        k_R = self.W_KR(x).unsqueeze(
            1
        )  # (batch, 1, seq, head_dim) cache during inference

        freq_cis: Complex[Tensor, "seq half_head_dim"] = pos_info
        q_R, k_R = apply_rope_real(q_R, k_R, freq_cis)

        # bring it together for the attention
        k = torch.cat([k_C, k_R], dim=-1)
        q = torch.cat([q_C, q_R], dim=-1)

        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
                scale=self.scale,
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
