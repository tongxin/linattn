from __future__ import annotations

import torch

from experiments.approximation_certificates import (
    distribution_value_nonidentifiability,
    quantized_kv_certificate,
    softmax_logit_perturbation_bound,
    sparse_support_certificate,
    total_variation,
)


def test_sparse_decomposition_is_exact_and_bound_holds():
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(3, 5, 9, dtype=torch.float64), -1)
    v = torch.randn(3, 5, 9, 7, dtype=torch.float64)
    support = torch.zeros(3, 5, 9, dtype=torch.bool)
    support[..., torch.tensor([0, 3, 4, 8])] = True
    result = sparse_support_certificate(p, v, support)
    assert torch.allclose(result["error"], result["exact_omitted"], atol=1e-12, rtol=1e-12)
    assert torch.all(result["error_norm"] <= result["bound"] + 1e-10)
    assert torch.allclose(result["attention_tv"], result["omitted_mass"], atol=1e-12)


def test_empty_support_is_causal_identity_failure_not_a_bound_failure():
    p = torch.softmax(torch.randn(2, 6), -1)
    v = torch.randn(2, 6, 4)
    support = torch.zeros(2, 6, dtype=torch.bool)
    result = sparse_support_certificate(p, v, support)
    assert torch.all(result["error_norm"] <= result["bound"] + 1e-6)


def test_distribution_error_does_not_identify_value_error():
    result = distribution_value_nonidentifiability()
    assert result["attention_tv"] > 0
    assert result["same_value_error"] < 1e-12
    assert result["orthogonal_value_error"] > 0.4


def test_quantized_key_tv_bound_and_error_decomposition():
    torch.manual_seed(1)
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 3, 4, 8)
    v = torch.randn(2, 3, 4, 8)
    k_quant = (k * 4).round() / 4
    v_quant = (v * 4).round() / 4
    result = quantized_kv_certificate(q, k, v, k_quant, v_quant)
    assert torch.all(result["attention_tv"] <= result["attention_tv_bound"] + 1e-6)
    assert torch.allclose(
        result["total_error"], result["key_error"] + result["value_error"], atol=1e-6
    )
    assert torch.all(result["total_error"].norm(dim=-1) <= result["total_bound"] + 1e-5)


def test_logit_perturbation_bound_matches_projection_bound():
    logits = torch.randn(4, 17)
    p = torch.softmax(logits, -1)
    delta = torch.randn_like(logits) * 0.05
    q = torch.softmax(logits + delta, -1)
    bound = softmax_logit_perturbation_bound(delta)
    assert torch.all(total_variation(p, q) <= bound + 1e-7)
