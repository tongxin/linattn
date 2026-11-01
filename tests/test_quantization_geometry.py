import torch

from linattn.kv.quant import QuantSpec, roundtrip


def test_constant_group_is_exact_under_absmax_group_scaling():
    """A constant group maps to the top codebook value exactly.

    E2M1 and E4M3 use an absmax group scale. If every element in a group is
    identical, the group maximum itself is exactly representable after scaling,
    so the quantization residual is zero up to float32 round-off.
    """
    for bits in (4, 8):
        x = torch.full((2, 4, 16, 8), 1.1, dtype=torch.float64)
        reconstructed = roundtrip(x, QuantSpec(bits=bits, group_size=16)).to(torch.float64)
        assert torch.allclose(reconstructed, x, atol=3e-8, rtol=3e-8)


def test_identical_values_give_coherent_not_cancelling_residual():
    x = torch.full((2, 4, 16, 8), 1.1, dtype=torch.float64)
    q = roundtrip(x, QuantSpec(bits=4, group_size=16)).to(torch.float64)
    residual = q - x
    assert torch.allclose(residual, residual[..., :1, :].expand_as(residual), atol=3e-8)
