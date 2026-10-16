# `pyramidkv`

**Family:** `eviction` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `PyramidKVAttention`

Cai et al., PyramidKV (2024) -- https://arxiv.org/abs/2406.02069

## Mechanism

SnapKV with a depth-dependent budget: more cache in low layers, less in high ones.

Cai et al., *PyramidKV: Dynamic KV Cache Compression based on Pyramidal
Information Funnel* (2024), https://arxiv.org/abs/2406.02069

## Usage

```python
attn = build("pyramidkv", AttnConfig(cache_budget=2048,
                                     extras={"budget_low": 256, "n_layers": 32, "layer_idx": 12}))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | attn FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 3,986 | 4,096 |
| 131,072 | 512.0 | 17,180 | 4,096 |
| 1,048,576 | 4096.0 | 140,325 | 4,096 |

## What to watch out for

PyramidKV is a budget-allocation policy layered on SnapKV: low layers keep
more cache, high layers keep less, because the information funnel narrows with
depth. The effect is real but modest, and it interacts with hybrid
architectures (where the "layer index" spans two different mechanisms), so
treat the profile as a tuning knob rather than a law.

## Validation

```bash
python -m pytest tests -q -k pyramidkv
python -m linattn.cli select --variant pyramidkv --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
