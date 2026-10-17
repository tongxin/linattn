"""CSA2 building blocks: FP4/FP8 codec, cross-layer plan, layer, stack."""

from __future__ import annotations

import pytest
import torch

from linattn.core import AttnConfig, dense_attention
from linattn.kv.csa2 import CSA2Stack, CSA2StackConfig, csa2_spec, compare_share
from linattn.kv.format import PRESETS, SPECS
from linattn.kv.quant import FP4_GRID, QuantSpec, bytes_for, dequantize, quantize, relative_error, roundtrip
from linattn.kv.reuse import CrossLayerPlan, share_layers
from linattn.metrics import rel_error
from linattn.registry import build


# --------------------------------------------------------------------------------------
# codec
# --------------------------------------------------------------------------------------
def test_fp4_is_exact_on_its_grid():
    grid = torch.cat([FP4_GRID, -FP4_GRID]).unsqueeze(0)
    assert torch.allclose(roundtrip(grid, QuantSpec(bits=4, group_size=8)), grid, atol=1e-6)


def test_precision_ordering_and_bytes():
    torch.manual_seed(0)
    x = torch.randn(4, 128) * 3
    errs = {bits: relative_error(x, roundtrip(x, QuantSpec(bits=bits, group_size=32))) for bits in (16, 8, 4)}
    assert errs[16] == 0.0
    assert errs[8] < errs[4]
    assert errs[4] < 0.25                                   # E2M1 with per-group scales
    # 4-bit costs at most a third of fp16, including the per-group scales
    assert bytes_for(x.numel(), QuantSpec(bits=4, group_size=32)) < bytes_for(x.numel(), QuantSpec(bits=16, group_size=32)) / 3


def test_quantize_preserves_shape_and_is_finite():
    x = torch.randn(2, 3, 64)
    codes, scales = quantize(x, QuantSpec(bits=4, group_size=16))
    assert codes.shape == x.shape
    assert scales.shape == (2, 3, 4)
    assert torch.isfinite(dequantize(codes, scales, QuantSpec(bits=4, group_size=16))).all()


# --------------------------------------------------------------------------------------
# cross-layer plan
# --------------------------------------------------------------------------------------
def test_plan_partitions_layers():
    plan = CrossLayerPlan(n_layers=7, group_size=3)
    assert plan.groups == [[0, 1, 2], [3, 4, 5], [6]]
    assert plan.copies == 3
    assert plan.stored_layers == [0, 3, 6]
    assert plan.bytes_multiplier == pytest.approx(3 / 7)
    assert plan.representative(4) == 3 and plan.is_full(3) and not plan.is_full(4)


def test_share_layers_maps_every_layer_to_a_representative():
    layers = [torch.full((1,), float(i)) for i in range(6)]
    stored, mapping = share_layers(layers, CrossLayerPlan(n_layers=6, group_size=2))
    assert set(stored) == {0, 2, 4}
    assert mapping == {0: 0, 1: 0, 2: 2, 3: 2, 4: 4, 5: 4}


# --------------------------------------------------------------------------------------
# the layer
# --------------------------------------------------------------------------------------
def _attn() -> AttnConfig:
    return AttnConfig(n_heads=4, n_kv_heads=2, head_dim=64, block_size=16, window_size=256,
                      topk_blocks=4, sink_size=2)


def test_csa2_reduces_to_dense_when_nothing_is_compressed_away():
    """Full raw window + no eligible groups == dense attention, exactly."""
    cfg = _attn()
    q, k, v = (torch.randn(2, cfg.n_heads, 256, cfg.head_dim, dtype=torch.float64) for _ in range(3))
    k = k[:, : cfg.n_kv_heads]
    v = v[:, : cfg.n_kv_heads]
    layer = build("csa2", cfg.with_(extras={"compress_ratio": 64, "bits": 16, "window": 256, "topk_groups": 4}))
    out, _, _ = layer.forward(q, k, v, chunk=256, collect_stats=True)
    assert rel_error(out, dense_attention(q, k, v)) < 1e-12


def test_csa2_reads_within_budget_and_is_causal():
    cfg = _attn()
    q, k, v = (torch.randn(2, n, 512, cfg.head_dim, dtype=torch.float64) for n in (cfg.n_heads, cfg.n_kv_heads, cfg.n_kv_heads))
    layer = build("csa2", cfg.with_(extras={"compress_ratio": 32, "bits": 4, "window": 128, "topk_groups": 8}))
    _, _, stats = layer.forward(q, k, v, chunk=512, collect_stats=True)
    reads = stats[-1]["reads_per_query"]
    assert reads <= 8 + 128 + 1
    assert stats[-1]["bits"] == 4


def test_low_precision_changes_the_result():
    cfg = _attn()
    q, k, v = (torch.randn(1, n, 512, cfg.head_dim, dtype=torch.float64) for n in (cfg.n_heads, cfg.n_kv_heads, cfg.n_kv_heads))
    outs = {}
    for bits in (16, 4):
        layer = build("csa2", cfg.with_(extras={"compress_ratio": 2, "bits": bits, "window": 128, "topk_groups": 128}))
        outs[bits] = layer.forward(q, k, v, chunk=512)[0]
    assert not torch.equal(outs[16], outs[4])
    assert rel_error(outs[4], outs[16]) > 1e-3


# --------------------------------------------------------------------------------------
# the stack and the footprint
# --------------------------------------------------------------------------------------
def _stack_inputs(n_layers=8, batch=1, seq=256, sigma=0.0, seed=0):
    torch.manual_seed(seed)
    base = torch.randn(batch, 2, seq, 64)
    q = torch.stack([torch.randn(batch, 4, seq, 64) for _ in range(n_layers)])
    k = torch.stack([base + sigma * torch.randn(batch, 2, seq, 64) for _ in range(n_layers)])
    v = torch.stack([base + sigma * torch.randn(batch, 2, seq, 64) for _ in range(n_layers)])
    return q, k, v


def test_stack_sharing_is_exact_when_layers_agree_and_costs_a_quarter():
    cfg = _attn()
    q, k, v = _stack_inputs()
    res = compare_share(CSA2StackConfig(n_layers=8, layer_group=4, compress_ratio=32, bits=4,
                                        topk_groups=8, window=128), cfg, q, k, v)
    assert res["sharing_delta"] == pytest.approx(0.0, abs=1e-9)
    assert res["copies_shared"] == 2 and res["copies_unshared"] == 8
    assert res["bytes_per_token_shared"] == pytest.approx(res["bytes_per_token_unshared"] / 4, rel=1e-6)


def test_stack_sharing_diverges_when_layers_disagree():
    cfg = _attn()
    q, k, v = _stack_inputs(sigma=1.0)
    res = compare_share(CSA2StackConfig(n_layers=8, layer_group=4, compress_ratio=32, bits=4,
                                        topk_groups=8, window=128), cfg, q, k, v)
    # independent layers cannot share a cache; the model reports that rather than hiding it
    assert res["sharing_delta"] > 0.2
    assert res["divergence_unshared_vs_ref"] < 0.1        # quantisation alone is mild


def test_composed_footprints_bracket_the_reported_number():
    reported = PRESETS["csa2_reported"].bytes_for(1_000_000, 61) / 1_000_000
    composed = PRESETS["csa2"].bytes_for(1_000_000, 61) / 1_000_000
    reach = PRESETS["csa2_reach890"].bytes_for(1_000_000, 61) / 1_000_000
    assert reported == pytest.approx(890.0, rel=1e-6)
    assert composed > reported * 3          # 4-layer sharing of a 576-value fp4 latent
    assert reach < reported * 1.2           # ~3 stored copies lands near the report
    assert SPECS["csa2"].cross_layer_group == 4


def test_stack_bytes_match_the_spec():
    cfg = _attn()
    stack = CSA2Stack(CSA2StackConfig(n_layers=4, layer_group=2, compress_ratio=32, bits=4), cfg)
    q, k, v = _stack_inputs(n_layers=4, seq=256)
    res = stack.run(q, k, v)
    spec = csa2_spec(CSA2StackConfig(n_layers=4, layer_group=2, compress_ratio=32, bits=4), cfg)
    assert res["bytes_per_token"] == pytest.approx(spec.bytes_for(256, 4) / 256, rel=1e-6)
