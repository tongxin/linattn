# `sfa`

**Family:** `frontier` &nbsp;|&nbsp; **Training required:** yes &nbsp;|&nbsp; **Class:** `SFAAttention`

Xie et al., Sparse Feature Attention (ICLR 2026) -- https://arxiv.org/abs/2603.22300

## Mechanism

Sparse Feature Attention: sparsify the feature axis, not the token axis.

Xie et al., *Scaling Attention via Feature Sparsity* (ICLR 2026),
https://arxiv.org/abs/2603.22300

Faithful core: represent queries and keys as ``k``-sparse codes and keep every
token.  The reference learns the codes and provides FlashSFA; here the code is
the top-``k`` magnitude coordinates, which is the inference-time semantics and
gives an exact reduction to dense attention when ``k == d``.

## Usage

```python
attn = build("sfa", AttnConfig(head_dim=128,
                               extras={"feature_keep": 32}))   # every token is kept
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | attn FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 129.0 | 17,592 | 32,768 |
| 131,072 | 516.0 | 281,475 | 131,072 |
| 1,048,576 | 4128.0 | 18,014,399 | 1,048,576 |

## What to watch out for

This is the complementary axis to every mask in the repository: all tokens
remain readable, but each query/key carries a sparse feature code. At
``feature_keep == head_dim`` the implementation is exactly dense attention.
The learned codebook and FlashSFA overlap kernel are not reproduced; the
top-magnitude code is a transparent inference-time stand-in.

## Validation

```bash
python -m pytest tests -q -k sfa
python -m linattn.cli select --variant sfa --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
