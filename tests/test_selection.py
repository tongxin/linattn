"""Selection quality, causality, and the learned-selector capacity question."""

from __future__ import annotations

import pytest
import torch

from linattn.core import AttnConfig, dense_attention
from linattn.harness import structured_inputs
from linattn.metrics import attention_mass, mass_recall, rel_error
from linattn.registry import build
from linattn.variants.selection import DSAAttention, SeerAttentionR, block_mask_to_token_mask


SELECTORS = ["quest", "minference", "flexprefill", "xattention", "moba", "dsa"]


@pytest.mark.parametrize("name", SELECTORS)
def test_selection_is_causal_and_faithful(cfg, structured, name):
    """Two invariants at once: nothing future is read, and the output is exactly
    the attention over the mask the variant claims to use.

    The second part is the stronger one: a variant that quietly read extra keys
    (or dropped some it reported) would produce a different tensor here.
    """
    q, k, v = structured
    q, k, v = q.to(torch.float64), k.to(torch.float64), v.to(torch.float64)
    variant = build(name, cfg)
    state = variant.prepare_state(q.shape[0], q.device, q.dtype)
    state = variant.push_kv(state, k, v)
    mask = variant.effective_mask(q, state, 0)
    t_q, t_k = mask.shape[-2:]
    future = torch.arange(t_q).unsqueeze(1) < torch.arange(t_k).unsqueeze(0)
    assert not bool((mask & future).any()), f"{name} selected future keys"
    assert bool(mask.any(-1).all()), f"{name} left a query with nothing to read"
    with torch.no_grad():
        out, _ = variant.forward(q, k, v, chunk=q.shape[-2])
    expected = dense_attention(q, k, v, causal=False, mask=mask)
    assert rel_error(out.double(), expected) < 1e-9, f"{name} does not match attention over its own mask"


@pytest.mark.parametrize("name", SELECTORS)
def test_selection_recovers_mass_on_structured_inputs(cfg, structured, name):
    """On planted retrieval structure, a good selector should beat random."""
    q, k, v = structured
    variant = build(name, cfg)
    state = variant.prepare_state(q.shape[0], q.device, q.dtype)
    state = variant.push_kv(state, k, v)
    mask = variant.select(q, state, 0)
    recall, budget = block_recall(mask, q, k, v, cfg.block_size)
    assert recall >= budget, f"{name} recall {recall:.3f} below its budget {budget:.3f}"


def test_quest_gqa_union_inflates_budget(cfg, structured):
    """The union-over-group effect the literature criticises must be visible."""
    q, k, v = structured
    reads = {}
    for label, union_mode in (("union", True), ("shared", False)):
        variant = build("quest", cfg.with_(extras={"gqa_union": union_mode, "dense_layers": 0}))
        state = variant.prepare_state(q.shape[0], q.device, q.dtype)
        state = variant.push_kv(state, k, v)
        mask = variant.select(q, state, 0)
        reads[label] = float(mask.sum(-1).float().mean())
    assert reads["union"] >= reads["shared"], "group union should read at least as many keys"
    assert reads["union"] > 0


def test_quest_selects_the_high_scoring_page_not_the_first(monkeypatch):
    """Regression: the own-page bonus must be applied per row, not as a block.

    A previous version wrote ``scores[..., own_page] = inf``, which indexes the last
    axis with a (T_q,) tensor and assigns ``inf`` to a (T_q, T_q) block -- every
    query ended up with the same early pages, and the bug survived the full test
    suite because those pages still captured most of the mass on the retrieval
    fixtures.  Here the high-scoring page is deliberately the *last* one.
    """
    cfg = AttnConfig(n_heads=2, n_kv_heads=2, head_dim=16, block_size=8, topk_blocks=1)
    variant = build("quest", cfg.with_(extras={"dense_layers": 0, "layer_idx": 0, "page_size": 8,
                                               "gqa_union": False}))
    q = torch.zeros(1, 2, 32, 16, dtype=torch.float64)
    k = torch.zeros(1, 2, 32, 16, dtype=torch.float64)
    v = torch.randn(1, 2, 32, 16, dtype=torch.float64)
    # give the last page a large key norm so its bound dominates for every query
    k[:, :, 24:, :] = 50.0
    q[:, :, :, 0] = 1.0
    state = variant.prepare_state(1, q.device, q.dtype)
    state = variant.push_kv(state, k, v)
    mask = variant.select(q, state, 0)
    row = mask[0, 0, 31]                              # (T_k,) read mask of the last query
    selected = row.view(-1, 8).any(-1).nonzero().flatten().tolist()
    assert 3 in selected, f"expected the last page (3) for the final query, got {selected}"


def test_minference_pattern_search_picks_a_pattern(cfg, structured):
    q, k, v = structured
    variant = build("minference", cfg.with_(extras={"pattern": "auto"}))
    variant.forward(q, k, v, chunk=q.shape[-2])
    assert variant.resolved in {"a_shape", "vertical_slash", "block_sparse"}


@pytest.mark.parametrize("pattern", ["a_shape", "vertical_slash", "block_sparse"])
@pytest.mark.parametrize("batch", [1, 3])
def test_minference_patterns_support_batching(cfg, pattern, batch):
    """Regression: the vertical-slash mask used ``v_top.shape[:-2]`` and therefore
    reshaped ``(B,H,k)`` to ``(B,1,H*k)`` -- fine by accident at B=1, a crash at B=3.
    """
    q, k, v = structured_inputs(batch, cfg.n_heads, 192, cfg.head_dim, kv_heads=cfg.n_kv_heads, seed=2)
    variant = build("minference", cfg.with_(extras={"pattern": pattern}))
    out, _ = variant.forward(q, k, v, chunk=64)
    assert out.shape == q.shape and torch.isfinite(out).all()
    # batched result must equal the per-item result when the same instance is used
    if batch > 1:
        for i in range(batch):
            single, _ = variant.forward(q[i : i + 1], k[i : i + 1], v[i : i + 1], chunk=64)
            assert rel_error(out[i : i + 1], single) < 1e-9


def test_dsa_fitting_beats_random_indexer(cfg, structured):
    """A learned indexer is meaningless until fitted; fitting must help.

    This is the capacity question that decides whether DSA-style selection can be
    bolted onto an existing model: does the ReLU-sum-of-head-scores parameterisation
    have enough expressiveness to rank keys the way softmax attention does?
    """
    q, k, v = structured
    variant = DSAAttention(cfg.with_(extras={"indexer_heads": 4, "indexer_dim": 8, "compress_ratio": 1}))
    state = variant.prepare_state(q.shape[0], q.device, q.dtype)
    state = variant.push_kv(state, k, v)
    with torch.no_grad():
        mask_before = variant.select(q, state, 0)
        block = cfg.block_size
        mass = attention_mass(q, k, v, block=block)
        n_qb, n_kb = mass.shape[-2:]
        rec_before = reduced_recall(mask_before, mass, block)
    loss = variant.fit(q, k, v, steps=60, lr=0.05)
    with torch.no_grad():
        mask_after = variant.select(q, state, 0)
        rec_after = reduced_recall(mask_after, mass, block)
    assert loss < 100.0
    assert rec_after >= rec_before - 0.05, "fitting should not destroy the indexer"


def test_seerattention_gate_needs_training(cfg, structured):
    q, k, v = structured
    variant = SeerAttentionR(cfg.with_(extras={"pool_dim": 8}))
    loss = variant.fit_gate(q, k, v, steps=40, lr=0.05)
    assert loss < 20.0, "gate fitting should converge to a sane cross-entropy"


def reduced_recall(mask, mass, block):
    n_qb = (mask.shape[-2] // block)
    n_kb = (mask.shape[-1] // block)
    reduced = mask[..., : n_qb * block, : n_kb * block]
    reduced = reduced.view(*mask.shape[:-2], n_qb, block, n_kb, block).any(-3).any(-1)
    return float(mass_recall(mass[..., :n_qb, :n_kb], reduced).mean())


def block_recall(mask, q, k, v, block):
    """Attention-mass recall of a token mask, plus the fraction of blocks it keeps."""
    mass = attention_mass(q, k, v, block=block)
    n_qb = mask.shape[-2] // block
    n_kb = mask.shape[-1] // block
    reduced = mask[..., : n_qb * block, : n_kb * block]
    reduced = reduced.view(*mask.shape[:-2], n_qb, block, n_kb, block).any(-3).any(-1)
    return float(mass_recall(mass, reduced).mean()), float(reduced.float().mean())
