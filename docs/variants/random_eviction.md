# `random_eviction`

**Family:** `baseline` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `RandomEvictionBaseline`

Wang et al., Random Attention (2026) -- https://arxiv.org/abs/2609.03430

## Mechanism

Control: evict uniformly at random among the unprotected tokens.

Implements the "no selection signal" baseline from
Wang et al., *Random Attention: Rethinking KV Cache Eviction for Efficient
Reasoning* (2026), https://arxiv.org/abs/2609.03430 -- it keeps the prompt and
the recent window and randomises everything else.  Included because it is the
most uncomfortable, and most reproducible, datapoint in the eviction
literature: on reasoning benchmarks it matches tuned selectors.

## Usage

```python
attn = build("random_eviction", AttnConfig(cache_budget=2048, window_size=128, seed=0))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 3,986 | 4,096 |
| 131,072 | 512.0 | 17,180 | 4,096 |
| 1,048,576 | 4096.0 | 140,325 | 4,096 |

## What to watch out for

This is the most useful page in the eviction family. It keeps the prompt and
the recent window and randomises everything else, and on reasoning workloads
it matches tuned selectors (arXiv:2609.03430). Any eviction method you are
considering should be compared against this baseline; if it does not beat it
clearly, the scoring machinery is not earning its keep.

## Validation

```bash
python -m pytest tests -q -k random_eviction
python -m linattn.cli select --variant random_eviction --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
