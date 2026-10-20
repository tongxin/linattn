# `flexprefill`

**Family:** `selection` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `FlexPrefillAttention`

Lai et al., FlexPrefill (2025) -- https://arxiv.org/abs/2502.20766

## Mechanism

FlexPrefill: context-aware, budget-adaptive block selection.

Lai et al., *FlexPrefill: A Context-Aware Sparse Attention Mechanism for
Efficient Long-Sequence Inference* (2025), https://arxiv.org/abs/2502.20766

Two stages, both reproduced:

1. **pattern estimation** -- a representative query block (the last
   ``block_size`` queries) is scored against the whole cache; if the block
   distribution implied by the probe is close (Jensen-Shannon distance) to the
   distribution implied by walking the candidate keys, the head is treated as
   *structured* (one global estimate reused for every query block), otherwise
   as *query-aware* (each query block estimates its own blocks from a
   subsample of keys per block);
2. **cumulative-attention index selection** -- blocks are taken in descending
   estimated score until the accumulated estimated mass reaches
   ``threshold`` (0.9 for query-aware heads, 0.95 for structured ones in the
   paper).  The read budget is data-dependent, which is the point of the
   method.

## Usage

```python
attn = build("flexprefill", AttnConfig(block_size=64, window_size=128,
                                       extras={"alpha_query_aware": 0.9, "alpha_structured": 0.95,
                                               "tau_js": 0.05}))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 1,100 | 1,536 |
| 131,072 | 512.0 | 7,699 | 1,536 |
| 1,048,576 | 4096.0 | 307,880 | 1,536 |

## What to watch out for

The two thresholds are the method's character: 0.9 for heads whose pattern is
genuinely query-dependent, 0.95 for heads where one estimate serves every
query. In practice most heads in most models resolve to the structured regime,
which is consistent with MInference's finding that static patterns cover a lot
of mass -- and it is a warning against assuming per-query adaptivity is paying
for itself.

## Validation

```bash
python -m pytest tests -q -k flexprefill
python -m linattn.cli select --variant flexprefill --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
