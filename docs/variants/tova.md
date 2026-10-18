# `tova`

**Family:** `eviction` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `TOVAAttention`

Oren et al., Transformers are Multi-State RNNs (ICLR 2024) -- https://arxiv.org/abs/2401.06104

## Mechanism

TOVA: keep the token the *latest* query attends to most, plus recent tokens.

Oren et al., *Transformers are Multi-State RNNs* (ICLR 2024),
https://arxiv.org/abs/2401.06104

TOVA's point is that the query that matters is the current one, not the average
over history -- a direct rebuttal of H2O's accumulated score.

## Usage

```python
attn = build("tova", AttnConfig(cache_budget=512, window_size=64))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | attn FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 3,986 | 4,096 |
| 131,072 | 512.0 | 17,180 | 4,096 |
| 1,048,576 | 4096.0 | 140,325 | 4,096 |

## What to watch out for

The direct rebuttal to H2O: the query that matters is the *current* one, so
score with the last query only. Cheap, and a useful control for whether
accumulated mass is buying anything.

## Validation

```bash
python -m pytest tests -q -k tova
python -m linattn.cli select --variant tova --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
