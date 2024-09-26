"""
Tests for complex and real RoPE implementation in `omni.modules.pos_embeddings`.
Real RoPE is necessary for PyTorch JIT compatibility, as such this test suit
verifies that the real RoPE implementation is equivalent to the complex one.
"""

import torch
from torch.testing import assert_close

from omni.modules.pos_embeddings import apply_rope
from omni.modules.pos_embeddings import apply_rope_real
from omni.modules.pos_embeddings import precompute_freqs_cis
from omni.modules.pos_embeddings import precompute_freqs_cis_real


def test_precompute_freqs_cis():
    head_dim = 64
    max_seq_length = 128
    theta = 10000.0

    freqs_cis_complex = precompute_freqs_cis(head_dim, max_seq_length, theta)
    cos_rotations, sin_rotations = precompute_freqs_cis_real(
        head_dim, max_seq_length, theta
    )

    assert_close(freqs_cis_complex.real, cos_rotations, rtol=1e-5, atol=1e-6)
    assert_close(freqs_cis_complex.imag, sin_rotations, rtol=1e-5, atol=1e-6)


def test_apply_rope():
    batch, n_heads, seq_length, head_dim = 2, 8, 128, 64
    q = torch.rand(batch, n_heads, seq_length, head_dim)
    k = torch.rand(batch, n_heads, seq_length, head_dim)

    freqs_cis_complex = precompute_freqs_cis(head_dim, seq_length)
    cos_rotations, sin_rotations = precompute_freqs_cis_real(head_dim, seq_length)

    q_complex, k_complex = apply_rope(q, k, freqs_cis_complex)
    q_real, k_real = apply_rope_real(q, k, (cos_rotations, sin_rotations))

    assert_close(q_complex, q_real, rtol=1e-5, atol=1e-6)
    assert_close(k_complex, k_real, rtol=1e-5, atol=1e-6)
