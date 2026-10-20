# `h2o`

**Family:** `eviction` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `H2OAttention`

Zhang et al., H2O: Heavy-Hitter Oracle (NeurIPS 2023) -- https://arxiv.org/abs/2306.14048

## Mechanism

Heavy-Hitter Oracle: keep tokens by *accumulated* attention mass.

Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
Large Language Models* (NeurIPS 2023), https://arxiv.org/abs/2306.14048

The score is attention mass received so far -- a past statistic used to predict
future usefulness. This is the canonical example of a heuristic that works on
RULER and is fragile on reasoning chains.

## Usage

```python
attn = build("h2o", AttnConfig(cache_budget=512, window_size=64))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 3,986 | 4,096 |
| 131,072 | 512.0 | 17,180 | 4,096 |
| 1,048,576 | 4096.0 | 140,325 | 4,096 |

## What to watch out for

Re-introduced as academic lineage, not as a recommendation. Accumulated
attention mass is a past statistic used to predict future usefulness; Random
Attention (2026) showed the score barely beats random once the prompt and
recent window are protected. Kept because new selectors need a baseline.

## Validation

```bash
python -m pytest tests -q -k h2o
python -m linattn.cli select --variant h2o --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
