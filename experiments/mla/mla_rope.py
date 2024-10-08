from typing import Literal, Tuple, TypeAlias, get_args

import torch.nn as nn
import torch
from torch import Tensor
from jaxtyping import Float
from jaxtyping import Int


MHATensor: TypeAlias = Float[Tensor, "batch heads seq head_dim"]
RotationTensor: TypeAlias = Float[Tensor, "seq dim_half"]

def precompute_freqs_cis_real(
    head_dim: Int, max_seq_length: Int, theta: Float = 10000.0
) -> Tuple[RotationTensor, RotationTensor]:
    """
    For each position in the sequence and each pair of dimensions in the head, computes
    a rotation in the complex plane of the form cos(m*θᵢ) + i*sin(m*θᵢ), where:
    - m is the position in the sequence
    - θᵢ = 1/θ^(2i/d) is the base frequency for the i-th dimension pair

    Split into real and imaginary components for float output.
    Args:
        head_dim: Size of attention head dimension (must be even)
        max_seq_length: Maximum sequence length to precompute rotations for
        theta: Base frequency scaling factor (default: 10000.0)
    """
    assert head_dim % 2 == 0

    freqs = 1.0 / torch.pow(
        theta, torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    )
    positions = torch.arange(0, max_seq_length)
    rotations = torch.outer(positions, freqs)

    cos_rotations = torch.cos(rotations)
    sin_rotations = torch.sin(rotations)

    return cos_rotations, sin_rotations


def apply_rope_real(
    q: MHATensor,
    k: MHATensor,
    rotations: Tuple[RotationTensor, RotationTensor],
) -> Tuple[MHATensor, MHATensor]:
    """
    Applies rotary position embeddings to query and key tensors using real-valued cosine and sine components.

    Args:
        q: Query tensor of shape (batch, n_heads, seq_length, head_dim).
        k: Key tensor of shape (batch, n_heads, seq_length, head_dim).
        rotations: Cosine and sine rotation tuple of tensors of shape (seq_length, head_dim // 2).

    Returns:
        Rotated query and key tensors.
    """
    q_f = q.float()
    k_f = k.float()

    # split into even (real) and odd (imaginary) components
    q_real, q_imag = q_f[:, :, :, ::2], q_f[:, :, :, 1::2]
    k_real, k_imag = k_f[:, :, :, ::2], k_f[:, :, :, 1::2]

    cos_rotations, sin_rotations = rotations

    def apply_rotation(real, imag, cos_rot, sin_rot, length):
        cos_rot = cos_rot[None, None, :length, :]
        sin_rot = sin_rot[None, None, :length, :]
        
        rotated_real = real * cos_rot - imag * sin_rot
        rotated_imag = real * sin_rot + imag * cos_rot
        
        return rotated_real, rotated_imag
    
    q_rotated_real, q_rotated_imag = apply_rotation(
        q_real, q_imag, cos_rotations, sin_rotations, q.size(2)
    )
    k_rotated_real, k_rotated_imag = apply_rotation(
        k_real, k_imag, cos_rotations, sin_rotations, k.size(2)
    )

    # interleave real/imaginary parts and reshape to original shape
    rotated_q = torch.stack((q_rotated_real, q_rotated_imag), dim=4).view(
        q.size()
    )
    rotated_k = torch.stack((k_rotated_real, k_rotated_imag), dim=4).view(
        k.size()
    )

    return rotated_q.type_as(q), rotated_k.type_as(k)


PositionEmbeddingScheme = Literal["rope", "absolute", "learned", "alibi"]


class MLAPositionalEmbedding(nn.Module):
    VALID_TYPES = set(get_args(PositionEmbeddingScheme))

    def __init__(self, config):
        super().__init__()
        cos_rotations, sin_rotations = precompute_freqs_cis_real(
            config.head_dim_decoupled_qk,
            config.seq_len,
            config.rope_theta,
        )
        self.register_buffer("cos_rotations", cos_rotations)
        self.register_buffer("sin_rotations", sin_rotations)


    def forward(self, x: Float[Tensor, "batch seq d_model"]):
        pos_info = (self.cos_rotations, self.sin_rotations)
        return x, pos_info
