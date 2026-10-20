# `dropkv`

**Family:** `eviction` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `DropKVAttention`

Zhang et al., DropKV (NeurIPS 2026) -- https://arxiv.org/abs/2610.22130

## Mechanism

DropKV: evict tokens by decoupled residual-output perturbation.

Zhang et al., *DropKV: Decoupling Residual-Output Perturbation for Near-Optimal
KV-Cache Eviction* (NeurIPS 2026), https://arxiv.org/abs/2610.22130

For each observation-window query, the score is
``(p_j / (1-p_j))^2 * ||a - v_j||^2``; tokens with the smallest score are
evicted.  The paper's fused Triton scorer is not reproduced, but the reference
computes the same quantity without materialising a ``w x n x d`` tensor.

## Usage

```python
attn = build("dropkv", AttnConfig(cache_budget=512, obs_window=8,
                                  extras={"pool_kernel": 11}))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 3,986 | 4,096 |
| 131,072 | 512.0 | 17,180 | 4,096 |
| 1,048,576 | 4096.0 | 140,325 | 4,096 |

## What to watch out for

This is the 2026 answer to the claim that eviction scores are exhausted: rank
by the *output perturbation* of removing a token, not by attention mass.
The score uses the observation window and values, which is why it can beat
attention-only baselines on RULER. The paper's Triton scorer is the deployment
path; this page keeps the reference score and the protected prompt/window
invariant, so the comparison to `random_eviction` remains apples-to-apples.

## Validation

```bash
python -m pytest tests -q -k dropkv
python -m linattn.cli select --variant dropkv --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
