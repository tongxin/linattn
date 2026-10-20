# `dense`

**Family:** `dense` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `DenseAttention`

Vaswani et al., Attention Is All You Need (2017) -- https://arxiv.org/abs/1706.03762

## Mechanism

Full softmax attention: the oracle, and the reference point for cost models.

## Usage

```python
from linattn.registry import build
from linattn.core import AttnConfig

attn = build("dense", AttnConfig(n_heads=32, n_kv_heads=8, head_dim=128))
out, state = attn.forward(q, k, v)              # prefill
out, state = attn.forward(q_t, k_t, v_t, state=state)   # one decode step
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 17,592 | 32,768 |
| 131,072 | 512.0 | 281,475 | 131,072 |
| 1,048,576 | 4096.0 | 18,014,399 | 1,048,576 |

## What to watch out for

Keep this as the oracle. Every other page's numbers are only interpretable
relative to it, and the test-suite validates every variant against it at an
unrestricted budget.

## Validation

```bash
python -m pytest tests -q -k dense
python -m linattn.cli select --variant dense --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
