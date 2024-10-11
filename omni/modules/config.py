from dataclasses import dataclass
from typing import Optional

from jaxtyping import Bool, Float, Int

from omni.modules.activations import ActivationFunction
from omni.modules.attention import AttentionType
from omni.modules.mlp import MLPType
from omni.modules.norm import NormalizationType
from omni.modules.pos_embeddings import PositionEmbeddingScheme


@dataclass
class TransformerConfig:
    vocab_size: Int
    seq_len: Int
    d_model: Int
    hidden_dim: Int
    num_heads: Int
    num_kv_heads: Int
    num_layers: Int

    # components
    pos_encoding_type: PositionEmbeddingScheme
    activation_fn: ActivationFunction
    mlp: MLPType
    normalization: NormalizationType
    attention: AttentionType

    mlp_bias: Bool = True
    mlp_dropout: Optional[Float] = None
    attention_bias: Bool = True
    attention_dropout: Optional[Float] = None
    weight_tying: Bool = False
    rope_theta: Float = 10000.0
    norm_eps: Float = 1e-5
