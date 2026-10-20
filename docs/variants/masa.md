# `masa`

**Family:** `frontier` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `MASAAttention`

Wan et al., MASA: Sparse Attention Is Matrix Approximation (2026) -- https://arxiv.org/abs/2610.10871

## Mechanism

Matrix Approximation Sparse Attention: score units by output-error reduction.

Wan et al., *Sparse Attention Is Matrix Approximation, Not Choosing from a Bag
of Values* (2026), https://arxiv.org/abs/2610.10871

For a candidate block ``u`` with contribution ``C_u = (M_u * A) V``, MASA uses
``2 <O, C_u> - ||C_u||^2``, the reduction in squared ``||A V||_F`` error.  The
implementation uses the exact profile attention map as the reference's analysis
oracle and selects top blocks per query.  It is deliberately expensive: the point
is to measure the attainable gain of an output-aware score before building a
cheap estimator for it.

## Usage

```python
# analysis oracle: uses the exact profile attention map
attn = build("masa", AttnConfig(block_size=64, topk_blocks=16,
                                extras={"block": 64, "topk_blocks": 16}))
```

## Cost model

Analytic; storage precision is shown per row, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | storage precision | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- | --- |
| 32,768 | 128.0 | fp16 KV | 18,282 | 1,028 |
| 131,072 | 512.0 | fp16 KV | 285,882 | 1,028 |
| 1,048,576 | 4096.0 | fp16 KV | 18,172,797 | 1,028 |

## What to watch out for

The important negative result is conceptual: selecting the largest attention
entries is the identity-Gram special case of preserving ``A V``, not the
general objective. MASA's score is exact but expensive here -- it deliberately
peeks at the profile attention map. Use it as an upper-bound/calibration
instrument for a cheap selector, not as a deployable kernel.

## Validation

```bash
python -m pytest tests -q -k masa
python -m linattn.cli select --variant masa --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
