# `h2o`

**Family:** `eviction` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `H2OAttention`

Zhang et al., H2O: Heavy-Hitter Oracle (NeurIPS 2023) -- https://arxiv.org/abs/2306.14048

## Mechanism

Heavy-Hitter Oracle: keep the tokens that received the most attention mass.

Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
Large Language Models* (NeurIPS 2023), https://arxiv.org/abs/2306.14048

## Usage

```python
attn = build("h2o", AttnConfig(cache_budget=1024, window_size=128, sink_size=4))
out, state = attn.forward(q, k, v)     # state["k"] never exceeds cache_budget
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | attn FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 3,986 | 4,096 |
| 131,072 | 512.0 | 17,180 | 4,096 |
| 1,048,576 | 4096.0 | 140,325 | 4,096 |

## What to watch out for

The heavy-hitter score is accumulated attention mass, which is a *past*
statistic used to predict *future* usefulness. It is the canonical example
of an eviction heuristic that works on RULER and is fragile on reasoning
chains -- see `random_eviction` for the control experiment.

## Validation

```bash
python -m pytest tests -q -k h2o
python -m linattn.cli select --variant h2o --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
