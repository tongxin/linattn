# `free_solo`

**Family:** `modern` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `FreeSoloAttention`

Shi & Price, FreeSolo (2026) -- https://arxiv.org/abs/2610.17191

## Mechanism

Sliding window + keypoint cells, and nothing else.

Shi & Price, *FreeSolo: Linear-time Sparse Attention with Zero-Shot Length
Extrapolation* (2026), https://arxiv.org/abs/2610.17191

Faithful part: the two branches (local window, long-range *cell* selection) and
the cell-level keypoint score. Approximated part: the paper's FS-L prunes the
prefix with a learned cell scorer and FS-B searches a balanced tree; here the
scorer is ``q . mean(k_cell)`` and the selection is a top-k, which is the
deployable core without the learned pruning.

## Usage

```python
attn = build("free_solo", AttnConfig(window_size=128,
                                     extras={"cell": 64, "topk_cells": 4, "window": 128}))
```

## Cost model

Analytic; storage precision is shown per row, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | storage precision | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- | --- |
| 32,768 | 128.0 | fp16 KV | 17,592 | 1,536 |
| 131,072 | 512.0 | fp16 KV | 281,475 | 1,536 |
| 1,048,576 | 4096.0 | fp16 KV | 18,014,399 | 1,536 |

## What to watch out for

2026 line: a window plus *keypoint cells* selected by score, with length
extrapolation as the goal rather than raw benchmark parity. The cell-level
selection is what makes it linear; the paper's learned prefix pruning and
balanced-tree search are not reproduced here.

## Validation

```bash
python -m pytest tests -q -k free_solo
python -m linattn.cli select --variant free_solo --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
