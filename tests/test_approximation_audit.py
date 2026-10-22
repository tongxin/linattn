from __future__ import annotations

import torch

from experiments.approximation_audit import (
    FILLER,
    SUM_QUERY,
    TinyCausalLM,
    TinyConfig,
    sample_batch,
    spearman,
)


def test_synthetic_tasks_have_causal_answer_positions():
    gen = torch.Generator().manual_seed(0)
    for task in ("retrieval", "dispersed"):
        tokens, targets = sample_batch(task, 8, 48, gen)
        assert tokens.shape == (8, 48)
        assert targets.shape == (8,)
        assert torch.all((targets >= 0) & (targets < 16))
        if task == "dispersed":
            assert torch.all(tokens[:, -1] == SUM_QUERY)
        assert torch.all(tokens[:, 0] == FILLER)


def test_tiny_model_returns_per_layer_intervention_metadata():
    model = TinyCausalLM(TinyConfig(seq_len=32, d_model=32))
    tokens = torch.randint(0, model.cfg.vocab, (2, 32))
    logits, hidden, attentions, qkvs = model(tokens)
    assert logits.shape == (2, 32, model.cfg.vocab)
    assert hidden.shape == (2, 32, 32)
    assert len(attentions) == len(qkvs) == model.cfg.n_layers
    q, k, v = qkvs[0]
    assert q.shape == k.shape == v.shape == (2, 4, 32, 8)


def test_spearman_detects_monotone_and_reversed_orders():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-12
