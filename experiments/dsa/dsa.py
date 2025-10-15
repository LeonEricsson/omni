from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex, Float
from mla_rope import apply_rope_real
from torch import Tensor

from omni.modules.norm import RMSNorm, LayerNorm

class Indexer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.index_num_heads = config.index_num_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk

        self.W_UQI = nn.Linear(config.d_cq, self.index_num_heads * self.index_head_dim)
        self.W_DKI = nn.Linear(config.d_model, self.index_head_dim)
        self.k_norm = LayerNorm(dim=self.index_head_dim)
        self.weights_proj = nn.Linear(config.d_model, config.index_num_heads)
        self.scale = self.index_head_dim ** -0.5

        self.register_buffer("k_cache", torch.zeros(config.max_batch_size, config.seq_len, self.index_head_dim))
    
    def forward(self,
        x: Float[Tensor, "batch seq d_model"],
        c_Q: Float[Tensor, "batch seq d_cq"],
        mask: Float[Tensor, "1 1 seq seq"],
        freq_cis,
        pos: int,
    ):
        batch_size, seq_length, _ = x.size()

        q_I = self.W_UQI(c_Q)
        q_I = q_I.reshape(batch_size, seq_length, self.index_num_heads, self.index_head_dim)
        q_rope, q_nope = torch.split(q_I, [self.rope_head_dim, self.index_head_dim - self.rope_head_dim], dim=-1)

        k_I = self.k_norm(self.W_DKI(x))
        k_rope, k_nope = torch.split(k_I, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

        q_rope, k_rope = apply_rope_real(q_rope, k_rope, freq_cis)

        q_I = torch.cat([q_rope, q_nope], dim=-1)
        k_I = torch.cat([k_rope, k_nope], dim=-1)

        self.k_cache[:batch_size, pos] = k_I
        k_history = self.k_cache[:batch_size, :pos] # (bsz, end_pos, head_dim)

        # --- Index Score Calculation  ---
        # I = sum_heads(w * ReLU(q @ k.T))
            
        scores = torch.matmul(q_I, k_history.transpose(-2, -1))
        scores = F.relu(scores, inplace=True)
        weights = self.weights_proj(x) * (self.n_heads ** -0.5)
        
        # sum the weighted scores across the heads to get the final index score
        index_score = torch.einsum('bshs,bsh->bs', scores, weights)
        index_score *= self.softmax_scale
        
        # --- Top-k Selection ---
        # Same as original: apply mask and find the top-k indices
        if mask is not None:
            index_score += mask

        # Retrieve the top-k indices along the sequence dimension
        topk_indices = torch.topk(index_score, min(self.index_topk, end_pos), dim=-1, largest=True).indices

        return topk_indices


class DSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads

        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.scale = self.qk_head_dim ** -0.5
        
        self.q_norm = RMSNorm(dim=config.d_cq)
        self.kv_norm = RMSNorm(dim=config.d_ckv)

        self.W_DQ = nn.Linear(
            config.d_model,
            config.d_cq,
            bias=False,
        )
        self.W_UQ = nn.Linear(
            config.d_cq,
            (self.qk_nope_head_dim * self.num_heads),
            bias=False,
        )
        self.W_UQR = nn.Linear(
            config.d_cq,
            (self.qk_rope_head_dim * self.num_heads),
            bias=False,
        )
        self.W_DKV = nn.Linear(
            config.d_model,
            config.d_ckv,
            bias=False,
        )
        self.W_UK = nn.Linear(
            config.d_ckv,
            (self.qk_nope_head_dim * self.num_heads),
            bias=False,
        )

        self.W_UV = nn.Linear(
            config.d_ckv,
            (self.v_head_dim * self.num_heads),
            bias=False,
        )


        self.W_DKR = nn.Linear(
            config.d_model,
            self.qk_rope_head_dim,
            bias=False,
        )

        self.W_O = nn.Linear(
            (self.v_head_dim * self.num_heads),
            config.d_model,
            bias=False,
        )
    
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.res_dropout = nn.Dropout(config.attention_dropout)

        self.register_buffer("kv_cache", torch.zeros(config.max_batch_size, config.seq_len, config.d_ckv))
        self.register_buffer("rope_cache", torch.zeros(config.max_batch_size, config.seq_len, self.qk_rope_head_dim))

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        mask: Float[Tensor, "1 1 seq seq"],
        pos_info: Optional[Tensor],
        pos: int,
    ):
        batch_size, seq_length, _ = x.size()

        c_Q = self.q_norm(self.W_DQ(x))

        # Compressed "content" dimensions
        c_KV = self.kv_norm(self.W_DKV(x))  # cache during inference

        q_C = self.W_UQ(c_Q)
        k_C = self.W_UK(c_KV)
        v = self.W_UV(c_KV)

        q_C = q_C.reshape(batch_size, seq_length, self.num_heads, self.qk_nope_head_dim)
        k_C = k_C.reshape(batch_size, seq_length, self.num_heads, self.qk_nope_head_dim)
        v = v.reshape(batch_size, seq_length, self.num_heads, self.v_head_dim)

        q_C = q_C.transpose(1, 2)
        k_C = k_C.transpose(1, 2)
        v = v.transpose(1, 2)

        # Decoupled RoPE dimensions
        q_R = self.W_UQR(c_Q)
        q_R = q_R.reshape(
            batch_size, seq_length, self.num_heads, self.qk_rope_head_dim
        )
        q_R = q_R.transpose(1, 2)
        k_R = self.W_DKR(x).unsqueeze(1) # missing num_head dimension

        freq_cis: Complex[Tensor, "seq half_head_dim"] = pos_info
        q_R, k_R = apply_rope_real(q_R, k_R, freq_cis)

        # kv cache (size reduction through qk_head_dim << (num_kv_heads * head_dim))
        self.kv_cache[:batch_size, pos] = c_KV
        self.rope_cache[:batch_size, pos] = k_R.squeeze(1)

        # combine content and rope dimensions for attention
        q = torch.cat([q_C, q_R], dim=-1)
        k = torch.cat([k_C, k_R.expand(-1, self.num_heads, -1, -1)], dim=-1)

        qk = (q @ k.transpose(2, 3)) / self.scale  # (batch, n_heads, seq, seq)
        qk = qk + mask[:, :, :seq_length, :seq_length]
        qk = F.softmax(qk, dim=-1)

        output = qk @ v  # (batch, n_heads, seq, head_dim)

        output = output.transpose(1, 2).reshape(
            batch_size, seq_length, self.num_heads * self.v_head_dim
        )

        output = self.W_O(output)
        return output
