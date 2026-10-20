# `performer`

**Family:** `academic` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `PerformerAttention`

Choromanski et al., Performer (ICML 2021) -- https://arxiv.org/abs/2009.14794

## Mechanism

FAVOR+ positive random features: linear time, causal, no attention matrix.

Choromanski et al., *Rethinking Attention with Performers* (ICML 2021),
https://arxiv.org/abs/2009.14794

``phi(x) = exp(-||x||^2/2) * exp(x W)`` with ``W ~ N(0, I)`` is an unbiased
estimator of ``exp(q.k)`` without the ``exp(+||x||^2/2)`` blow-up of the original
formulation. The causal path uses prefix sums of ``phi(k) v^T`` and ``phi(k)``, so
cost is ``O(T m d)`` and the *reads* are the feature dimension, not the cache.

## Usage

```python
attn = build("performer", AttnConfig(extras={"features": 64}))   # causal, linear time
```

## Cost model

Analytic; storage precision is shown per row, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | storage precision | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- | --- |
| 32,768 | 128.0 | fp16 KV | 17,592 | 32 |
| 131,072 | 512.0 | fp16 KV | 281,475 | 32 |
| 1,048,576 | 4096.0 | fp16 KV | 18,014,399 | 32 |

## What to watch out for

The random-feature estimator is unbiased but high-variance: on random inputs we
measured ~0.5 relative error at 16 features improving to ~0.46 at 1024. That is
the honest picture of kernel approximations -- they converge slowly, and their
advantage is asymptotic, which is why production chose *exact* sparse
selection over approximation. Implemented with positive features to avoid the
``exp(+||x||^2/2)`` blow-up.

## Validation

```bash
python -m pytest tests -q -k performer
python -m linattn.cli select --variant performer --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
