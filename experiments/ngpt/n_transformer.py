import inspect
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from n_attention import causal_attention_mask, nGQA, nMHA
from n_mlp import nMLPSwiGLU
from torch import Tensor

from omni.modules.cache import KVCache
from omni.modules.pos_embeddings import PositionalEmbedding, PositionEmbeddingScheme


@dataclass
class nConfig:
    vocab_size: Int
    seq_len: Int
    d_model: Int
    num_heads: Int
    num_kv_heads: Int
    num_layers: Int
    hidden_dim: Int = None

    pos_encoding_type: PositionEmbeddingScheme = "rope"

    mlp_bias: Bool = False
    mlp_dropout: Float = 0.0
    attention_bias: Bool = False
    attention_dropout: Float = 0.0
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

        s_z_init = 1.0
        s_z_scale = config.d_model**-0.5
        self.s_z = nn.Parameter(
            s_z_scale * torch.ones(config.vocab_size, dtype=torch.float32)
        )
        self.register_buffer("s_z_init", torch.tensor(s_z_init))
        self.register_buffer("s_z_scale", torch.tensor(s_z_scale))

        if config.weight_tying:
            self.vocab_proj.weight = self.token_emb.weight
            nn.init.normal_(
                self.token_emb.weight, mean=0.0, std=1.0 / (config.d_model**0.5)
            )

        self.register_buffer("causal_mask", causal_attention_mask(config.seq_len))

    def get_kv_cache_layer(self, kv_cache, layer_idx) -> KVCache:
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

        s_z = self.s_z * (self.s_z_init / self.s_z_scale)
        x = s_z * x

        return x

    def configure_optimizers(
        self, weight_decay, learning_rate, betas, device_type
    ) -> torch.optim.Optimizer:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [
            p for _, p in param_dict.items() if p.dim() >= 2
        ]  # weights in matmuls & embeddings
        nodecay_params = [
            p for _, p in param_dict.items() if p.dim() < 2
        ]  # biases, layer norms, and scalar params

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(
            f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, fused=use_fused
        )


class nBlock(nn.Module):
    def __init__(self, config: nConfig):
        super().__init__()

        self.attn = nMHA(config)
        self.mlp = nMLPSwiGLU(config)

        alpha_scale = config.d_model**-0.5
        self.alpha_attn = nn.Parameter(
            alpha_scale * torch.ones(config.d_model, dtype=torch.float32)
        )
        self.alpha_mlp = nn.Parameter(
            alpha_scale * torch.ones(config.d_model, dtype=torch.float32)
        )
        self.register_buffer("alpha_init", torch.tensor(config.alpha_init))
        self.register_buffer("alpha_scale", torch.tensor(alpha_scale))

        self.norm = lambda x: F.normalize(x, dim=-1)

    def normalize_weights(self):
        self.attn.normalize_weights()
        self.mlp.normalize_weights()

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        mask: Float[Tensor, "1 1 seq seq"],
        pos_info,
        kv_cache,
    ):

        # alpha_A = torch.abs(self.alpha_attn * (self.alpha_init / self.alpha_scale))
        # x_A = self.norm(self.attn(x, mask, pos_info, kv_cache))
        # x = self.norm(x + alpha_A * (x_A - x))  # eq. 10

        # alpha_M = torch.abs(self.alpha_mlp * (self.alpha_init / self.alpha_scale))
        # x_M = self.norm(self.mlp(x))
        # x = self.norm(x + alpha_M * (x_M - x))  # eq. 11

        alpha_A = torch.abs(self.alpha_attn * (self.alpha_init / self.alpha_scale))
        x_A = self.attn(x, mask, pos_info, kv_cache)
        x_A_norm = self.norm(x_A)
        x_norm = self.norm(x)
        x = self.norm(x_norm + alpha_A * (x_A_norm - x_norm))  # eq. 10

        alpha_M = torch.abs(self.alpha_mlp * (self.alpha_init / self.alpha_scale))
        x_M = self.mlp(x)
        x_M_norm = self.norm(x_M)
        x_norm = self.norm(x)
        x = self.norm(x_norm + alpha_M * (x_M_norm - x_norm))  # eq. 11

        return x
