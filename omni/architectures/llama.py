from dataclasses import dataclass

from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int

from omni.modules.activations import ActivationFunction
from omni.modules.attention import AttentionType
from omni.modules.mlp import MLPType
from omni.modules.norm import NormalizationType
from omni.modules.pos_embeddings import PositionEmbeddingScheme


@dataclass
class LlamaConfig:
    vocab_size: Int
    seq_len: Int
    d_model: Int
    hidden_dim: Int
    num_heads: Int
    num_kv_heads: Int
    num_layers: Int

    # components
    pos_encoding_type: PositionEmbeddingScheme = "rope"
    activation_fn: ActivationFunction = "silu"
    mlp: MLPType = "mlp_swiglu"
    normalization: NormalizationType = "rmsnorm"
    attention: AttentionType = "gqa"

    mlp_bias: Bool = False
    mlp_dropout: Float = False
    attention_bias: Bool = False
    attention_dropout: Float = 0.0
    rope_theta: Float = 10000.0
    norm_eps: Float = 1e-5