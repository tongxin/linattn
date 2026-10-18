# `snapkv`

**Family:** `eviction` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `SnapKVAttention`

Li et al., SnapKV (NeurIPS 2024) -- https://arxiv.org/abs/2404.14469

## Mechanism

SnapKV: score keys once from an observation window of recent queries.

Li et al., *SnapKV: LLM Knows What You are Looking for Before Generation*
(NeurIPS 2024), https://arxiv.org/abs/2404.14469

The observation-window attention is average-pooled along the key axis (kernel 7
for Llama-3 in the paper); the window itself is always retained so generation can
continue.

## Usage

```python
attn = build("snapkv", AttnConfig(cache_budget=512, obs_window=32,
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

The prefill-time variant: score once from an observation window, pool the
scores, keep the top keys plus the window. Its cost is paid once rather than
per decode step -- the opposite trade-off from Quest/DSA -- which is why it is
still a reasonable baseline for compression-at-prefill.

## Validation

```bash
python -m pytest tests -q -k snapkv
python -m linattn.cli select --variant snapkv --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
