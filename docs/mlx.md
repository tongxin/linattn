# Running the variants on Apple Silicon (MLX / Metal)

The reference implementations are PyTorch, which on a MacBook means CPU: PyTorch's
MPS path has no fused attention kernel and no fast top-k, so the sparse-attention
variants cannot be exercised at a realistic scale there. MLX targets Metal directly
and provides the three things these variants need -- a fused SDPA, `topk`/`argsort`,
and a custom-kernel API -- so it is the right backend for validating them on a
laptop GPU.

```bash
make install                  # torch + pytest
uv pip install -e ".[mlx]"    # or: uv pip install mlx
make check                    # MLX tests skip automatically if mlx is missing
pytest tests/test_mlx.py -q   # the Metal parity suite
make bench-mlx                # latency + numerics -> reports/mlx_decode.md
python -m linattn.cli backends    # reports mlx_installed / mlx_metal
python -m linattn.cli cost --seq 65536 --variants mlx_quest mlx_dsa
```

`cost` lists each algorithm variant once and accepts explicit `mlx_*` backend names
through `--variants`; their analytic cost delegates to the matching torch
reference, so the table is available on a machine where MLX is not installed.

## What is implemented

| piece | file | what it validates |
| --- | --- | --- |
| primitives | `kernels/mlx/ops.py` | dense / sliding-window / block-sparse attention, indexer top-k, chunked delta rule -- each against the repo's torch reference |
| custom Metal kernel | `kernels/mlx/metal_decode.py` | paged decode attention: one SIMD group per (batch, head), lanes split the head dim, online softmax in registers, one read per KV byte |
| variants | `kernels/mlx/variants.py` | `dense`, `sliding_window`, `quest`, and `dsa` as *compressed* (CSA-style) page indexing |
| harness adapter | `kernels/mlx/adapter.py` | `build_mlx("quest")` returns an `AttentionVariant`, so `linattn.harness.compare`, the CLI, benchmarks and tests drive Metal unchanged |

```python
from linattn.core import AttnConfig
from linattn.harness import compare
from linattn.kernels.mlx.adapter import build_mlx
from linattn.registry import build

cfg = AttnConfig(n_heads=8, n_kv_heads=4, head_dim=64, block_size=16, topk_blocks=8)
rows = compare([build_mlx("quest", cfg), build("quest", cfg)], seq=2048)
```

## Parity, as measured on this machine

| op | MLX vs torch | note |
| --- | --- | --- |
| dense / sliding / block-sparse attention | ~1e-3 | MLX fused SDPA, fp32 |
| Quest selection + attention (through the adapter) | < 1e-3, identical read budget | after the page-selection fix below |
| custom Metal decode kernel | **4.2e-7** | explicit fp32 FMAs in registers |
| indexer top-k values | ~7e-4 | GPU matmul path |
| chunked delta rule (MLX CPU stream) | < 1e-5 | exact fp32 |

**Device numerics.** MLX's *generic* fp32 matmul on this Metal device carries
~8e-4 relative error, versus ~1e-6 on MLX's own CPU backend (`bench/bench_mlx.py`
measures both). The hand-written Metal kernel is *more* accurate than the generic
matmul path because it does its own fp32 FMAs -- a concrete reminder that kernel
numerics are a design choice. The tests therefore use a loose tolerance on GPU and a
tight one on the CPU stream.

## Three semantics you must not get wrong

1. **MLX's `mask="causal"` is bottom-right aligned.** With `T_q ≠ T_k`, query `i`
   may read keys up to `i + (T_k - T_q)`; PyTorch's `is_causal=True` and this
   repository's `causal_mask` are top-left (`j <= i`). `ops.dense_attention` uses an
   explicit mask whenever the shapes differ.
2. **Selection must still be intersected with causality.** A page selector can rank
   a partially-future page highly; the torch path intersects with the causal mask,
   and `block_sparse_attention(..., offset=...)` does the same -- otherwise the MLX
   path reads future tokens and the parity test compares two different computations.
3. **The delta rule needs L2-normalised keys.** The chunk matrix is
   `M[i,j] = beta_i (k_i G_i)·(k_j/G_j)`, so unnormalised keys make the triangular
   solve exponentially ill-conditioned (we measured |y| ~ 1e24 with ‖k‖ ≈ 8.5, after
   which two correct implementations disagree by 1e-2).

## A bug this backend found

Porting Quest to MLX surfaced a real bug in the *torch* implementation: the
"always read your own page" rule was written as `scores[..., own_page] = inf`, which
indexes the last axis with a `(T_q,)` tensor and assigns `inf` to a `(T_q, T_q)`
block. Every query ended up selecting the same early pages, and the full test suite
missed it because those pages still captured most of the mass on the retrieval
fixtures. The fix is a diagonal scatter, `tests/test_selection.py` now pins the
behaviour with a deliberately late high-scoring page, and the MLX and torch paths
agree *exactly* afterwards.

## Limitations

* The adapter crosses torch → numpy → MLX once per chunk, so use it for
  **validation**, not serving; `bench/bench_mlx.py` times the MLX-only path.
* `dsa` on the MLX path is **compressed** page indexing (CSA), not token-level
  selection, because the Metal kernel wants a fixed-width page list. Token-level DSA
  is exercised by the torch path.
* The MLX chunked delta rule covers the symmetric (scalar-gate) recurrence; GDN-2's
  asymmetric erase/write gates are implemented in torch (`variants/recurrent.py`,
  validated against the recurrence to 2.8e-16) and are not yet ported to MLX.
* Head dim must be a multiple of 32 and ≤ 128 for the Metal kernel; other shapes fall
  back to the ops path.
* In a headless or sandboxed macOS session MLX raises
  `[metal::load_device] No Metal device available` at import; that is why the MLX
  tests skip with a reason there. Run them from a normal terminal.
