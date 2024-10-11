from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex, Float
from mla_rope import apply_rope_real
from torch import Tensor

from omni.modules.norm import RMSNorm


class KVCacheMLA:
    def __init__(self, config, device: str = None, dtype: torch.dtype = None):
        self.max_seq_len = config.seq_len
        self.device = device
        self.dtype = dtype

        d_ckv = config.d_ckv
        head_dim_decoupled_qk = config.head_dim_decoupled_qk

        if d_ckv is None:
            d_ckv = 4 * config.head_dim

        if head_dim_decoupled_qk is None:
            head_dim_decoupled_qk = config.head_dim // 2

        self.c_KV = torch.zeros(
            (1, config.num_layers, self.max_seq_len, d_ckv), device=device, dtype=dtype
        )
        self.k_R = torch.zeros(
            (1, config.num_layers, self.max_seq_len, head_dim_decoupled_qk),
            device=device,
            dtype=dtype,
        )

        self.cache_lengths = torch.zeros(
            config.num_layers, device=device, dtype=torch.int16
        )

    def forward(
        self,
        layer_idx: int,
        c_KV: Float[Tensor, "batch seq d_ckv"],
        k_R: Float[Tensor, "batch seq head_dim_decoupled_qk"],
    ):
        """Update the cache for a single layer, handling overflow by rolling."""
        cache_len = self.cache_lengths[layer_idx].item()
        new_len = k_R.size(1)
        max_len = self.max_seq_len

        if cache_len + new_len > max_len:
            overflow = cache_len + new_len - max_len
            self.k_R[:, layer_idx, :-overflow, :] = self.k_R[:, layer_idx, overflow:, :]
            self.c_KV[:, layer_idx, :-overflow, :] = self.c_KV[
                :, layer_idx, overflow:, :
            ]
            cache_len = max_len - new_len

        self.k_R[:, layer_idx, cache_len : cache_len + new_len, :] = k_R
        self.c_KV[:, layer_idx, cache_len : cache_len + new_len, :] = c_KV

        self.cache_lengths[layer_idx] = min(cache_len + new_len, max_len)

        return (
            self.k_R[:, layer_idx, : self.cache_lengths[layer_idx], :],
            self.c_KV[:, layer_idx, : self.cache_lengths[layer_idx], :],
        )


class MLA(nn.Module):
    """
    Multi-head Latent Attention attempts to reduce the KV cache without performance comprimise by
    implementing a low-rank key-value joint compression strategy. Emperically it achivies superior
    performance to MHA, while significantly reducing the memory requirements during inference.

    Args:
    config: Configuration containing:
        - num_heads (int): Number of query attention heads
        - d_model (int): Model dimension
        - head_dim (int): Dimension of each attention head. As opposed to past attention mechanisms, the per-head dimension does not increase the KV cache. As such, head_dim is typically set to > d_model // num_heads. DeepSeek use head_dim = 3 * d_model // num_heads.
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
        self.head_dim_decoupled_qk = config.head_dim_decoupled_qk
        d_ckv = config.d_ckv
        d_cq = config.d_cq

        if self.head_dim is None:
            self.head_dim = 3 * config.d_model // self.num_heads

        if d_ckv is None:
            d_ckv = 4 * self.head_dim

        if d_cq is None:
            d_cq = 12 * self.head_dim

        if self.head_dim_decoupled_qk is None:
            self.head_dim_decoupled_qk = self.head_dim // 2

        self.scale = (self.head_dim + self.head_dim_decoupled_qk) ** -0.5

        self.W_DQ = nn.Linear(
            config.d_model,
            d_cq,
            bias=False,
        )

        self.W_UQ = nn.Linear(
            d_cq,
            (self.head_dim * self.num_heads),
            bias=False,
        )

        self.W_DWK = nn.Linear(
            config.d_model,
            d_ckv,
            bias=False,
        )

        self.W_UK = nn.Linear(
            d_ckv,
            (self.head_dim * self.num_heads),
            bias=False,
        )

        self.W_UV = nn.Linear(
            d_ckv,
            (self.head_dim * self.num_heads),
            bias=False,
        )

        self.W_QR = nn.Linear(
            d_cq,
            (self.head_dim_decoupled_qk * self.num_heads),
            bias=False,
        )

        self.W_KR = nn.Linear(
            config.d_model,
            self.head_dim_decoupled_qk,
            bias=False,
        )

        self.W_O = nn.Linear(
            (self.head_dim * self.num_heads),
            config.d_model,
            bias=False,
        )

        self.q_norm = RMSNorm(dim=d_cq)
        self.kv_norm = RMSNorm(dim=d_ckv)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.res_dropout = nn.Dropout(config.attention_dropout)

        self.flash_attn: bool = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        mask: Float[Tensor, "1 1 seq seq"],
        pos_info: Optional[Tensor],
        kv_cache,
        layer_idx,
    ):
        batch_size, seq_length, _ = x.size()

        c_Q = self.q_norm(self.W_DQ(x))

        # Compressed "content" dimensions
        c_KV = self.kv_norm(self.W_DWK(x))  # cache during inference

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
        q_R = self.W_QR(c_Q)  # (batch, seq, num_heads * head_dim_decoupled_qk)
        q_R = q_R.reshape(
            batch_size, seq_length, self.num_heads, self.head_dim_decoupled_qk
        )
        q_R = q_R.transpose(1, 2)
        k_R = self.W_KR(x).unsqueeze(
            1
        )  # (batch, 1, seq, head_dim_decoupled_qk) cache during inference

        freq_cis: Complex[Tensor, "seq half_head_dim"] = pos_info
        q_R, k_R = apply_rope_real(q_R, k_R, freq_cis)

        # bring it together for the attention
        k = torch.cat([k_C, k_R.expand(-1, self.num_heads, -1, -1)], dim=-1)
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
            qk = qk + mask[:, :, :seq_length, :seq_length]
            qk = F.softmax(qk, dim=-1)
            qk = self.attn_dropout(qk)

            output = qk @ v  # (batch, n_heads, seq, head_dim)

        output = output.transpose(1, 2).reshape(
            batch_size, seq_length, self.num_heads * self.head_dim
        )

        output = self.W_O(output)
        output = self.res_dropout(output)

        return output


class MLAInference(MLA):
    """
    Multi-head Latent Attention attempts to reduce the KV cache without performance comprimise by
    implementing a low-rank key-value joint compression strategy. Emperically it achivies superior
    performance to MHA, while significantly reducing the memory requirements during inference.

    References:
         - "DeepSeek V2" (https://arxiv.org/abs/2405.04434)
         - "DeepSeek V3" (https://arxiv.org/abs/2412.19437)
    """

    def __init__(self, config):
        super().__init__(config)

    def fuse_weights(self):
        """Lazily compute and store fused weights for inference."""

        # Fuse Q with K and V up-projections
        W_UK = self.W_UK.weight.detach()
        W_UQ = self.W_UQ.weight.detach()

        W_UQ_UK = W_UQ.T @ W_UK  # (d_cq, d_ckv)

        W_UV = self.W_UV.weight.detach()
        W_O = self.W_O.weight.detach()

        W_U_OV = W_UV.T @ W_O.T  # (d_ckv, d_model)

        self.register_buffer("W_UQ_UK", W_UQ_UK)
        self.register_buffer("W_U_OV", W_U_OV)

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        mask: Float[Tensor, "1 1 seq seq"],
        pos_info: Optional[Tensor],
        kv_cache: KVCacheMLA,
        layer_idx: Optional[int],
    ):
        # print(x.shape)
        batch_size, seq_length, _ = x.size()

        if not hasattr(self, "W_UQ_UK") or not hasattr(self, "W_U_OV"):
            self.fuse_weights()

        c_Q = self.q_norm(self.W_DQ(x))

        # Compressed "content" dimensions
        c_KV = self.W_DWK(x)
        c_KV = self.kv_norm(c_KV)  # (batch, seq, d_ckv)

        q_C = (c_Q @ self.W_UQ_UK).unsqueeze(1)  # (batch, 1, seq, d_ckv)

        # Decoupled RoPE dimensions
        q_R = self.W_QR(c_Q)  # (batch, seq, num_heads * head_dim_decoupled_qk)
        q_R = q_R.reshape(
            batch_size, seq_length, self.num_heads, self.head_dim_decoupled_qk
        )
        q_R = q_R.transpose(1, 2)
        k_R = self.W_KR(x)  # (batch, seq, head_dim_decoupled_qk)

        if kv_cache is not None:
            c_KV, k_R = kv_cache.forward(layer_idx, c_KV, k_R)

        c_KV = c_KV.unsqueeze(1)
        k_R = k_R.unsqueeze(1)

        v = c_KV.expand(-1, self.num_heads, -1, -1)

        freq_cis: Complex[Tensor, "seq half_head_dim"] = pos_info
        q_R, k_R = apply_rope_real(q_R, k_R, freq_cis)

        # bring it together for the attention

        k = torch.cat([c_KV, k_R], dim=-1)
        q = torch.cat([q_C.expand(-1, self.num_heads, -1, -1), q_R], dim=-1)

        ## There is something wrong here with the shapes, not sure how to handle the heads.
        ## Given the fused weight matrices we have "lost" the head_dim and now we've instead got
        # the compressed dimensions? q_R has num heads but q_C does not because of the fused
        # matrix W_UQ_UK?

        print(k.shape, q.shape, v.shape)
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
            qk = (q @ k.transpose(2, 3)) / self.scale  # (batch, 1, seq, seq)
            qk = qk + mask[:, :, qk.shape[-1], : qk.shape[-1]]
            qk = F.softmax(qk, dim=-1)
            qk = self.attn_dropout(qk)

            output = qk @ v  # (batch, 1, seq, d_ckv)

        output = output.squeeze(1)

        output = output @ self.W_U_OV
        output = self.res_dropout(output)

        print("output", output.shape)
        return output
