# `csa2`

**Family:** `native` &nbsp;|&nbsp; **Training required:** yes &nbsp;|&nbsp; **Class:** `CSA2Attention`

DeepSeek-AI, DeepSeek-V4.1-Flash: CSA2 + FP4 KV (2026) -- https://arxiv.org/abs/2609.19969

## Mechanism

Compressed, low-precision, sparsely-selected attention (one layer).

## Usage

```python
from linattn.registry import build
attn = build("csa2")
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 4.2 | 352 | 528 |
| 131,072 | 11.0 | 2,233 | 528 |
| 1,048,576 | 74.0 | 79,440 | 528 |

## What to watch out for

This is the **per-layer** CSA2 mechanism: pooled groups, an FP4/FP8 cache with
per-group scales, indexer top-k over groups, and a raw local window kept disjoint
from the pooled groups so no token is counted twice. It is *not* the released
layer: there is no trained indexer, no published layer schedule, and the FP4 path
is a simulated codec (packed byte accounting, FP32 arithmetic after
dequantisation). Cross-layer reuse lives outside the variant, in
`kv/reuse.py` + `kv/csa2.py`, because it is a property of the layer *stack*.
`reports/csa2_model.md` shows the footprint decomposition (the reported 890
B/token is not what one-copy-per-four-layers yields) and the divergence that
sharing introduces as neighbouring layers disagree.

## Validation

```bash
python -m pytest tests -q -k csa2
python -m linattn.cli select --variant csa2 --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
