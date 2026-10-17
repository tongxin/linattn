"""MLX/Metal parity tests.

Skipped automatically when ``mlx.core`` cannot import -- which includes Linux and
headless macOS sessions without a Metal device (MLX raises
``[metal::load_device] No Metal device available``).  Tolerances are set from the
device's own fp32 behaviour: MLX's generic matmul on this Metal device carries
~1e-3 relative error (vs ~1e-6 on its CPU backend, measured in
``bench/bench_mlx.py``), so attention and delta-rule checks use a loose tolerance on
GPU and a tight one on the CPU stream.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from linattn.core import AttnConfig, dense_attention
from linattn.harness import structured_inputs
from linattn.kernels.mlx import metal_available, mlx_available, mlx_import_error, mx
from linattn.metrics import rel_error
from linattn.registry import build

pytestmark = pytest.mark.skipif(
    not mlx_available(), reason=f"mlx unavailable ({mlx_import_error()})"
)

GPU_TOL = 6e-3          # device fp32 matmul is ~1e-2; SDPA is ~1e-3
CPU_TOL = 1e-5          # MLX's CPU backend is true fp32


@pytest.fixture
def cfg() -> AttnConfig:
    return AttnConfig(n_heads=4, n_kv_heads=2, head_dim=64, block_size=16, window_size=64,
                      topk_blocks=4, topk_tokens=128, sink_size=2, seed=0)


def _torch_side(name: str, cfg: AttnConfig):
    extras = {"dense_layers": 0, "layer_idx": 0, "page_size": cfg.block_size, "gqa_union": False}
    return build(name, cfg.with_(extras=extras))


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("name", ["dense", "sliding_window", "quest"])
def test_adapter_matches_torch(name, batch, cfg):
    from linattn.kernels.mlx.adapter import build_mlx

    q, k, v = structured_inputs(batch, cfg.n_heads, 256, cfg.head_dim, kv_heads=cfg.n_kv_heads, seed=0)
    mlx_out, _, mlx_stats = build_mlx(name, cfg).forward(q, k, v, chunk=256, collect_stats=True)
    torch_out, _, torch_stats = _torch_side(name, cfg).forward(q, k, v, chunk=256, collect_stats=True)
    assert rel_error(mlx_out, torch_out) < GPU_TOL, f"{name}: MLX and torch disagree"
    # the read budget must agree too: a kernel that reads more than the selector says
    # is exactly the failure mode the accounting exists to catch
    if "reads_per_query" in torch_stats[-1]:          # dense does not report a budget
        assert abs(mlx_stats[-1]["reads_per_query"] - torch_stats[-1]["reads_per_query"]) < 1.0


def test_metal_decode_kernel_matches_torch():
    from linattn.kernels.mlx import ops
    from linattn.kernels.mlx.metal_decode import decode_attention_torch, metal_decode_attention

    if not metal_available():
        pytest.skip("no Metal device")
    rng = np.random.default_rng(0)
    b, h, hkv, tk, d, page = 1, 4, 2, 512, 64, 16
    q = rng.standard_normal((b, h, 1, d)).astype(np.float32)
    k = rng.standard_normal((b, hkv, tk, d)).astype(np.float32)
    v = rng.standard_normal((b, hkv, tk, d)).astype(np.float32)
    ids = rng.integers(0, tk // page, (b, h, 8)).astype(np.int32)
    out, used = metal_decode_attention(ops.to_mlx(q), ops.to_mlx(k), ops.to_mlx(v), ops.to_mlx(ids),
                                      page_size=page, n_kv_heads=hkv)
    assert used, "the custom Metal kernel should be selected for D=64"
    ref = decode_attention_torch(*(torch.from_numpy(a) for a in (q, k, v)),
                                 torch.from_numpy(ids.astype(np.int64)), page_size=page)
    # the kernel uses explicit fp32 FMAs, so it is *tighter* than generic MLX matmul
    assert rel_error(torch.from_numpy(np.array(out)), ref) < 1e-5


def test_block_sparse_attention_matches_torch():
    from linattn.kernels.mlx import ops
    from linattn.kernels.triton.block_sparse import block_sparse_attention_torch

    rng = np.random.default_rng(1)
    b, h, tq, tk, d, bs = 1, 2, 24, 64, 32, 16
    q, k, v = (rng.standard_normal((b, h, n, d)).astype(np.float32) for n in (tq, tk, tk))
    ids = rng.integers(0, tk // bs, (b, h, tq, 2)).astype(np.int32)
    got = ops.block_sparse_attention(ops.to_mlx(q), ops.to_mlx(k), ops.to_mlx(v), ops.to_mlx(ids), block_size=bs)
    want = block_sparse_attention_torch(*(torch.from_numpy(a) for a in (q, k, v)), torch.from_numpy(ids.astype(np.int64)),
                                        block_size=bs)
    assert rel_error(torch.from_numpy(np.array(got)), want) < 1e-5


def test_indexer_topk_matches_torch_values():
    from linattn.kernels.mlx import ops
    from linattn.kernels.triton.fused_indexer import indexer_topk_torch

    rng = np.random.default_rng(2)
    q_idx = rng.standard_normal((1, 8, 3, 8)).astype(np.float32)
    k_idx = rng.standard_normal((1, 32, 3, 8)).astype(np.float32)
    w = rng.random((1, 8, 3)).astype(np.float32)
    got, _ = ops.indexer_topk(ops.to_mlx(q_idx), ops.to_mlx(k_idx), ops.to_mlx(w), 5)
    want = indexer_topk_torch(*(torch.from_numpy(a) for a in (q_idx, k_idx, w)), 5)[0]
    # scores are O(10); the GPU's reduced-precision matmul puts ~1e-3 relative
    # error on them (see the numerics section of reports/mlx_*.md)
    assert rel_error(torch.from_numpy(np.array(got)), want) < 3e-3


def test_chunked_delta_rule_matches_torch_on_cpu_stream():
    """MLX's CPU backend is exact fp32, so the delta rule can be checked tightly."""
    from linattn.kernels.mlx import ops
    from linattn.kernels.triton.delta_chunk import delta_chunk_torch

    rng = np.random.default_rng(3)
    b, h, c, dk, dv = 1, 2, 16, 8, 8
    q = rng.standard_normal((b, h, c, dk)).astype(np.float32)
    k = rng.standard_normal((b, h, c, dk))
    k = (k / np.linalg.norm(k, axis=-1, keepdims=True)).astype(np.float32)   # delta-rule invariant
    v = rng.standard_normal((b, h, c, dv)).astype(np.float32)
    beta = rng.random((b, h, c)).astype(np.float32)
    g = (-rng.random((b, h, c, dk)) * 0.2).astype(np.float32)
    st = np.zeros((b, h, dk, dv), dtype=np.float32)
    with mx.stream(mx.cpu):
        out, new_state = ops.chunked_delta_rule(*(ops.to_mlx(a) for a in (q, k, v, beta, g, st)))
        out, new_state = np.array(out), np.array(new_state)
    want_out, want_state = delta_chunk_torch(*(torch.from_numpy(a) for a in (q, k, v, beta, g, st)),
                                             use_erase=True)
    assert rel_error(torch.from_numpy(out), want_out) < CPU_TOL
    assert rel_error(torch.from_numpy(new_state), want_state) < CPU_TOL


def test_csa_selection_stays_within_budget(cfg):
    """The compressed-index path must not read more than its page budget."""
    from linattn.kernels.mlx import ops
    from linattn.kernels.mlx.variants import mlx_forward

    q, k, v = structured_inputs(1, cfg.n_heads, 512, cfg.head_dim, kv_heads=cfg.n_kv_heads, seed=4)
    out, stats = mlx_forward("dsa", ops.to_mlx(q), ops.to_mlx(k), ops.to_mlx(v),
                             page_size=cfg.block_size, topk_tokens=cfg.topk_tokens,
                             indexer_proj=np.full((cfg.n_heads, 4, cfg.head_dim, 16), 0.01, dtype=np.float32))
    assert out.shape == (1, cfg.n_heads, 512, cfg.head_dim)
    assert stats["reads_per_query"] <= cfg.topk_tokens + cfg.block_size
    assert not np.isnan(np.array(out)).any()


def test_mlx_causal_alignment_matches_torch():
    """MLX's "causal" string is bottom-right aligned; our explicit mask is not."""
    from linattn.kernels.mlx import ops

    rng = np.random.default_rng(5)
    b, h, tq, tk, d = 1, 2, 8, 12, 16
    q, k, v = (rng.standard_normal((b, h, n, d)).astype(np.float32) for n in (tq, tk, tk))
    got = ops.dense_attention(ops.to_mlx(q), ops.to_mlx(k), ops.to_mlx(v), offset=tk - tq)
    want = dense_attention(*(torch.from_numpy(a) for a in (q, k, v)), causal=True)
    assert rel_error(torch.from_numpy(np.array(got)), want) < GPU_TOL
