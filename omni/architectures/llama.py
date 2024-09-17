from dataclasses import dataclass

from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int


@dataclass
class LlamaConfig:
    vocab_size: Int
    seq_len: Int
    d_model: Int
    hidden_dim: Int
    num_heads: Int
    num_kv_heads: Int
    num_layers: Int
    rope_theta: Float
    norm_eps: Float
    activation_fn: str = "silu"  # relu, gelu, silu, tanh
    mlp_bias: Bool = False
    mlp_dropout: Float = False
    attention_bias: Bool = False
    attention_dropout: Float = 0.0
    pos_encoding_type: str = "rope"
    mlp: str = "mlp_swiglu"
    normalization: str = "rmsnorm"
    attention: str = "gqa"
