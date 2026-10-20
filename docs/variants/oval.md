# `oval`

**Family:** `frontier` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `OVALAttention`

Shahbazi et al., OVAL: Output-Aware Local Page Bases (2026) -- https://arxiv.org/abs/2610.06686

## Mechanism

Output-aware local page bases for sparse KV retrieval.

Shahbazi et al., *OVAL: Output-Aware Local Page Bases for KV Cache Retrieval*
(2026), https://arxiv.org/abs/2610.06686

Faithful core: centre each page's keys and values, form
``M_eta = (1-eta)c_k X^T X + eta c_o X^T Y Y^T X``, retain its leading
eigenvectors, and score pages with the page mean plus LogSumExp of the projected
key logits.  The reference uses a per-head output projector for ``G``; this
implementation uses ``G=I`` and a page-local SVD, which preserves the mechanism
while avoiding an unstated model-specific ``W_O``.

## Usage

```python
attn = build("oval", AttnConfig(block_size=16, topk_blocks=8,
                                extras={"page_size": 16, "rank": 8, "eta": 0.5}))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 1,244 | 1,028 |
| 131,072 | 512.0 | 13,222 | 1,028 |
| 1,048,576 | 4096.0 | 721,503 | 1,028 |

## What to watch out for

2026 line in the KV-retrieval layer: page summaries should preserve the
directions that matter to the *output*, not only key energy. The ``eta=0``
endpoint is key-PCA; ``eta=1`` is the output-sensitive basis. The reference
sets ``G=I`` and builds summaries from K/V alone; a production conversion
would use the head's output projector and a calibrated ``eta``.

## Validation

```bash
python -m pytest tests -q -k oval
python -m linattn.cli select --variant oval --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
