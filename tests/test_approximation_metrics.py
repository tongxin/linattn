from __future__ import annotations

import torch

from experiments.approximation_metrics import (
    attention_operator_metrics,
    cosine_distance,
    hellinger,
    jensen_shannon,
    mass_recall,
    relative_l2,
    topk_agreement,
    total_variation,
)


def test_identical_inputs_have_zero_error():
    x = torch.randn(3, 8)
    p = torch.softmax(x, -1)
    assert relative_l2(x, x) == 0.0
    assert cosine_distance(x, x) < 1e-12
    assert total_variation(p, p) == 0.0
    assert jensen_shannon(p, p) < 1e-10
    assert hellinger(p, p) == 0.0


def test_distribution_metrics_are_symmetric_and_bounded():
    p = torch.softmax(torch.randn(4, 16), -1)
    q = torch.softmax(torch.randn(4, 16), -1)
    assert abs(total_variation(p, q) - total_variation(q, p)) < 1e-7
    assert abs(jensen_shannon(p, q) - jensen_shannon(q, p)) < 1e-7
    assert abs(hellinger(p, q) - hellinger(q, p)) < 1e-7
    assert 0.0 <= total_variation(p, q) <= 1.0
    assert 0.0 <= hellinger(p, q) <= 1.0


def test_mass_recall_is_zero_for_empty_support():
    p = torch.softmax(torch.randn(1, 16), -1)
    assert mass_recall(p, torch.zeros_like(p, dtype=torch.bool)) == 0.0
    assert mass_recall(p, torch.ones_like(p, dtype=torch.bool)) == 1.0


def test_scale_and_rank_intuition():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert abs(relative_l2(x, 2.0 * x) - 1.0) < 1e-7
    assert cosine_distance(x, 2.0 * x) < 1e-7
    assert topk_agreement(x, x, 2) == 1.0


def test_operator_metrics_are_finite():
    p = torch.softmax(torch.randn(2, 32), -1)
    q = torch.softmax(torch.randn(2, 32), -1)
    v = torch.randn(32, 8)
    rows = attention_operator_metrics(p, q, p @ v, q @ v, k=4)
    assert set(rows) == {
        "operator_rel_l2",
        "operator_cosine_distance",
        "attention_tv",
        "attention_js",
        "attention_hellinger",
        "topk_agreement",
    }
    assert all(torch.isfinite(torch.tensor(v)).item() for v in rows.values())
