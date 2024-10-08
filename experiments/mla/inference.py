from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import Bool

from omni.modules.config import TransformerConfig
from omni.modules.attention import causal_attention_mask
from omni.modules.mlp import MLP_MAP
from omni.modules.norm import NORM_MAP
from omni.modules.activations import ActivationFunction
from omni.modules.mlp import MLPType
from omni.modules.norm import NormalizationType
from omni.preprocessing.tokenizer import AutoTokenizer
from omni.utils.system import auto_device
from omni.inference.inference import Inference
from omni.modules.cache import KVCache

from mla import MLAInference, KVCacheMLA
from mla_rope import MLAPositionalEmbedding


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
    
    d_ckv: Int = None
    d_cq: Int = None
    attention_dropout: Float = 0.0


class MLABlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.attn = MLAInference(config)
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

        self.blocks = nn.ModuleList([MLABlock(config) for _ in range(config.num_layers)])

        self.norm_out = NORM_MAP[config.normalization](config)

        self.vocab_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.register_buffer("causal_mask", causal_attention_mask(config.seq_len))

    def fuse_mla_weights(self):
        """Fuse MLA weights for faster inference."""
        for block in self.blocks:
            block.attn.fuse_weights()

    def forward(
        self,
        x: Int[Tensor, "batch seq"],
        pad_mask: Int[Tensor, "batch seq"] = None,
        kv_cache: KVCache = None,
    ) -> Float[Tensor, "batch seq vocab_size"]:
        mask = self.causal_mask
        
        if self.training:
            mask &= pad_mask[:, None, None, :]

        x = self.token_emb(x)

        x, pos_info = self.position_embedding(x)

        for i, block in enumerate(self.blocks):
            x = block(x, mask, pos_info, kv_cache, layer_idx=i)

        x = self.norm_out(x)

        x = self.vocab_proj(x)

        return x
    

def main():
    tokenizer = AutoTokenizer.create("EleutherAI/gpt-neo-125m")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    mla_config = MLAConfig(
    vocab_size=50258,
    seq_len=512,
    d_model=256,
    head_dim=64,
    head_dim_decoupled_qk=32,
    num_heads=8,
    num_layers=4,
    activation_fn="silu",
    mlp_bias=False,
    mlp_dropout=0.1,
    attention_dropout=0.1,
    mlp="mlp_swiglu",
    normalization="rmsnorm",
    )

    model = MLATransformer(mla_config)

    checkpoint = torch.load("checkpoints/MLA_20250125_121321/init.ckpt", weights_only=True)
    model.load_state_dict(checkpoint["model"])

    device = auto_device()
    inference = Inference(model, tokenizer, device=device, temperature=0, max_length=100)
    kv_cache = KVCacheMLA(mla_config, device=device)

    prompt = "Once upon a time "

    print(prompt, end="", flush=True)
    import time
    start_time = time.perf_counter()

    # Generate text
    for token in inference.generate_nonkvcache(prompt):
        print(tokenizer.decode([token]), end="", flush=True)
    
    # for token in inference.generate(prompt, kv_cache):
    #     print(tokenizer.decode([token]), end="", flush=True)

    end_time = time.perf_counter()
    tokens_per_sec = 100 / (end_time - start_time)
    print(f"\n{tokens_per_sec:.2f} tokens per second")


if __name__ == "__main__":
    main()