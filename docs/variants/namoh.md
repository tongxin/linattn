# `namoh`

**Family:** `modern` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `NAMOHAttention`

Fu et al., NAMOH: native sparse attention from mixture-of-heads (2026) -- https://arxiv.org/abs/2609.38832

## Mechanism

Head-owned token subsets: each head attends only its assigned subsequence.

Fu et al., *Scaling Parameter and Context in Attention: Native Sparse Attention
from Mixture-of-Head* (2026), https://arxiv.org/abs/2609.38832

NAMOH activates ``K`` of ``H`` heads per token and lets each head keep only its
assigned tokens. Here the assignment is a deterministic stride (``token_index %
H``) with a local window so every query has context; the paper learns the
assignment. With ``H = 1`` this is exact causal attention, which the tests use.

## Usage

```python
attn = build("namoh", AttnConfig(window_size=64))   # heads own token subsets
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | attn FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 17,592 | 1,536 |
| 131,072 | 512.0 | 281,475 | 4,608 |
| 1,048,576 | 4096.0 | 18,014,399 | 33,280 |

## What to watch out for

2026 line: sparsity along the *head* axis rather than the token axis -- each
head owns a token subset and attends causally inside it. With one head it is
exact (tested), so the mechanism is a strict generalisation; what it argues
against is the assumption that all heads must see the same tokens.

## Validation

```bash
python -m pytest tests -q -k namoh
python -m linattn.cli select --variant namoh --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
