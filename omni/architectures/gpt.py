from dataclasses import dataclass

from jaxtyping import Bool, Float, Int

from omni.modules.activations import ActivationFunction
from omni.modules.attention import AttentionType
from omni.modules.mlp import MLPType
from omni.modules.norm import NormalizationType
from omni.modules.pos_embeddings import PositionEmbeddingScheme


@dataclass
class GPTConfig:
    vocab_size: Int
    seq_len: Int
    d_model: Int
    hidden_dim: Int
    num_heads: Int
    num_layers: Int

    # components
    pos_encoding_type: PositionEmbeddingScheme = "absolute"
    activation_fn: ActivationFunction = "gelu"
    mlp: MLPType = "mlp"
    normalization: NormalizationType = "layernorm"
    attention: AttentionType = "mha"

    mlp_bias: Bool = True
    mlp_dropout: Float = 0.1
    attention_bias: Bool = True
    attention_dropout: Float = 0.1
    weight_tying: Bool = False
    norm_eps: Float = 1e-5
