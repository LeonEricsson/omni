from typing import Literal, Tuple, TypeAlias, get_args

import torch
import torch.nn as nn
from jaxtyping import Complex, Float, Int
from torch import Tensor

MHATensor: TypeAlias = Float[Tensor, "batch heads seq head_dim"]
RotationTensor: TypeAlias = Float[Tensor, "seq dim_half"]


def _create_absolute_positions(d_model: int, seq_len: int) -> torch.Tensor:
    """
    Create sinusoidal position embeddings.
    Returns: shape [1, seq_len, d_model]
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float()
        * (-torch.log(torch.tensor(10000.0)) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0)


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

    def apply_rotation(real, imag, cos_rot, sin_rot, pos_idx):
        cos_rot = cos_rot[None, None, pos_idx : pos_idx + 1, :]
        sin_rot = sin_rot[None, None, pos_idx : pos_idx + 1, :]

        rotated_real = real * cos_rot - imag * sin_rot
        rotated_imag = real * sin_rot + imag * cos_rot

        return rotated_real, rotated_imag

    seq_position = k.shape[2] - 1
    q_rotated_real, q_rotated_imag = apply_rotation(
        q_real, q_imag, cos_rotations, sin_rotations, seq_position
    )
    k_rotated_real, k_rotated_imag = apply_rotation(
        k_real, k_imag, cos_rotations, sin_rotations, seq_position
    )

    # interleave real/imaginary parts and reshape to original shape
    rotated_q = torch.stack((q_rotated_real, q_rotated_imag), dim=4).view(q.size())
    rotated_k = torch.stack((k_rotated_real, k_rotated_imag), dim=4).view(k.size())

    return rotated_q.type_as(q), rotated_k.type_as(k)


def precompute_freqs_cis(
    head_dim: Int, max_seq_length: Int, theta: Float = 10000.0
) -> Complex[Tensor, "seq dim_half"]:
    """
    For each position in the sequence and each pair of dimensions in the head, computes
    a rotation in the complex plane of the form cos(m*θᵢ) + i*sin(m*θᵢ), where:
    - m is the position in the sequence
    - θᵢ = 1/θ^(2i/d) is the base frequency for the i-th dimension pair

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
    freqs_cis = torch.polar(torch.ones_like(rotations), rotations)

    return freqs_cis


def apply_rope(
    q: MHATensor, k: MHATensor, freqs_cis: Complex[Tensor, "seq dim_half"]
) -> Tuple[MHATensor, MHATensor]:
    """
    Applies rotary position embeddings to query and key tensors in float32.

    Rotates pairs of dimensions (x,y) in q/k by position-dependent angles. For each pair:
    1. Convert to complex number: z = x + iy
    2. Multiply by rotation e^(im*θ) = cos(m*θ) + i*sin(m*θ)
    3. Convert result back to real pairs
    """
    batch, n_heads, seq_length, head_dim = q.size()
    q_f = q.float()
    k_f = k.float()
    complex_q = torch.complex(q_f[:, :, :, ::2], q_f[:, :, :, 1::2])
    complex_k = torch.complex(k_f[:, :, :, ::2], k_f[:, :, :, 1::2])

    rotated_q = complex_q * freqs_cis[None, None, :, :]
    rotated_k = complex_k * freqs_cis[None, None, :, :]

    # interleave real/imaginary parts and reshape to original shape
    real_q = torch.stack((rotated_q.real, rotated_q.imag), dim=4).view(
        batch, n_heads, seq_length, head_dim
    )
    real_k = torch.stack((rotated_k.real, rotated_k.imag), dim=4).view(
        batch, n_heads, seq_length, head_dim
    )

    return real_q.type_as(q), real_k.type_as(k)


def precompute_alibi_bias(
    seq_length: Int, num_heads: Int
) -> Float[Tensor, "num_heads seq"]:
    ratio = 2 ** (-8 / num_heads)
    slopes = torch.pow(ratio, torch.arange(1, num_heads + 1, dtype=torch.float32))
    positions = -torch.arange(seq_length, dtype=torch.float32)
    return slopes[:, None] * positions[None, :]


PositionEmbeddingScheme = Literal["rope", "absolute", "learned", "alibi"]


class PositionalEmbedding(nn.Module):
    VALID_TYPES = set(get_args(PositionEmbeddingScheme))

    def __init__(self, config):
        super().__init__()
        if config.pos_encoding_type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid position encoding type: {config.pos_encoding_type}. Must be one of {self.VALID_TYPES}"
            )

        self.type = config.pos_encoding_type

        if self.type == "absolute":
            self.register_buffer(
                "pos_embedding",
                _create_absolute_positions(config.d_model, config.seq_len),
            )
        elif self.type == "rope":
            if not hasattr(config, "head_dim"):
                head_dim = config.d_model // config.num_heads
            else:
                head_dim = config.head_dim
                
            cos_rotations, sin_rotations = precompute_freqs_cis_real(
                head_dim,
                config.seq_len,
                config.rope_theta,
            )
            self.register_buffer("cos_rotations", cos_rotations)
            self.register_buffer("sin_rotations", sin_rotations)

        elif self.type == "learned":
            self.pos_embedding = nn.Embedding(config.seq_len, config.d_model)

        elif self.type == "alibi":
            self.register_buffer(
                "alibi_bias", precompute_alibi_bias(config.seq_len, config.num_heads)
            )

    def forward(self, x: Float[Tensor, "batch seq d_model"]):
        pos_info = None

        if self.type == "absolute":
            x = x + self.pos_embedding
        elif self.type == "rope":
            pos_info = (self.cos_rotations, self.sin_rotations)
        elif self.type == "learned":
            seq_indices = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            x = x + self.pos_embedding(seq_indices)
        elif self.type == "alibi":
            pos_info = self.alibi_bias

        return x, pos_info
