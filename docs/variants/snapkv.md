# `snapkv`

**Family:** `eviction` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `SnapKVAttention`

Li et al., SnapKV (NeurIPS 2024) -- https://arxiv.org/abs/2404.14469

## Mechanism

SnapKV: score keys once from an observation window of recent queries.

Li et al., *SnapKV: LLM Knows What You are Looking for Before Generation*
(NeurIPS 2024), https://arxiv.org/abs/2404.14469

Implementation notes: the observation-window attention is average-pooled along
the key axis with kernel ``cfg.pool_kernel`` (7 for Llama-3 in the paper), the
pooled score decides which *prompt* tokens survive, and the observation window
itself is always retained so that generation can continue.

## Usage

```python
attn = build("snapkv", AttnConfig(cache_budget=2048, obs_window=64,
                                  extras={"pool_kernel": 7}))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | attn FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 3,986 | 4,096 |
| 131,072 | 512.0 | 17,180 | 4,096 |
| 1,048,576 | 4096.0 | 140,325 | 4,096 |

## What to watch out for

Two details drive the results: the observation window (the last tokens before
generation starts) and the pooling kernel that averages neighbouring key
scores. Both are hyperparameters that change the answer, and both are
workload-dependent. SnapKV is a prefill-time compression, so its cost is paid
once rather than per decode step -- the opposite trade-off from Quest/DSA.

## Validation

```bash
python -m pytest tests -q -k snapkv
python -m linattn.cli select --variant snapkv --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
