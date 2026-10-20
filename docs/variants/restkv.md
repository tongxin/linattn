# `restkv`

**Family:** `eviction` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `ReSTKVAttention`

An et al., ReST-KV (ICLR 2026) -- https://arxiv.org/abs/2605.08840

## Mechanism

ReST-KV: output reconstruction with spatial--temporal smoothing.

An et al., *ReST-KV: Robust KV Cache Eviction with Layer-wise Output
Reconstruction and Spatial-Temporal Smoothing* (ICLR 2026),
https://arxiv.org/abs/2605.08840

Faithful core: rank by the output discrepancy caused by removing a token,
``p_j ||a - v_j||^2``, smooth the score over time, and pool neighbouring token
scores.  The layer-wise reconstruction solve and adaptive window are simplified
to an EMA plus max-pooling; the paper's kernel is not reproduced.

## Usage

```python
attn = build("restkv", AttnConfig(cache_budget=512, obs_window=8,
                                  extras={"pool_kernel": 7}))
```

## Cost model

Analytic; storage precision is shown per row, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | storage precision | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- | --- |
| 32,768 | 128.0 | fp16 KV | 3,986 | 4,096 |
| 131,072 | 512.0 | fp16 KV | 17,180 | 4,096 |
| 1,048,576 | 4096.0 | fp16 KV | 140,325 | 4,096 |

## What to watch out for

ReST-KV is the reconstruction/smoothing branch of the same 2026 revival.
Its score is a single-token output-discrepancy proxy, then an EMA and a local
max-pool make it robust to layer- and position-specific noise. The layer-wise
reconstruction solve is simplified here; treat this as a reproducible baseline
for the smoother, not as the authors' full system.

## Validation

```bash
python -m pytest tests -q -k restkv
python -m linattn.cli select --variant restkv --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
