# `pisa`

**Family:** `frontier` &nbsp;|&nbsp; **Training required:** yes &nbsp;|&nbsp; **Class:** `PISAAttention`

Tang et al., Block Sparse Attention with Log-Linear Complexity (2026) -- https://arxiv.org/abs/2609.31093

## Mechanism

Pyramid Top-K block selection with log-linear routing.

Tang et al., *Block Sparse Attention with Log-Linear Complexity* (2026),
https://arxiv.org/abs/2609.31093

Faithful core: build a coarse-to-fine hierarchy of pooled keys, select at the
coarsest level, and only expand the children of selected nodes.  Each level
scores a bounded candidate set, so routing does not require every query-block
pair.  The paper uses learned/LogSumExp node scores and fused Triton kernels;
here nodes are mean-pooled keys and the final attention is exact over the
selected tokens.

## Usage

```python
attn = build("pisa", AttnConfig(topk_tokens=1024,
                                extras={"branch": 2, "topk_tokens": 1024}))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 9,348 | 2,052 |
| 131,072 | 512.0 | 142,945 | 2,052 |
| 1,048,576 | 4096.0 | 9,024,860 | 2,052 |

## What to watch out for

PISA's contribution is not another block score; it is removing the
``T x T`` *routing* term. The hierarchy is narrow at the top and only expands
selected children, giving log-linear selection under a fixed token budget.
This implementation uses mean-pooled node keys and exact attention over the
final leaves. The paper's LogSumExp node score and fused Triton kernel are
the next step; do not compare its wall clock to this Python reference.

## Validation

```bash
python -m pytest tests -q -k pisa
python -m linattn.cli select --variant pisa --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
