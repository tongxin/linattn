# `linformer`

**Family:** `academic` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `LinformerAttention`

Sun et al., Linformer (ICML 2020) -- https://arxiv.org/abs/2006.04768

## Mechanism

Low-rank keys/values: ``softmax(Q (E K)^T) (F V)`` in ``O(T E d)``.

Sun et al., *Linformer: Self-Attention with Linear Complexity* (ICML 2020),
https://arxiv.org/abs/2006.04768

**Not causal**: the projection mixes positions, so a query would see a
low-rank summary of future keys. It is the canonical statement of the low-rank
hypothesis that every later feature-axis compression inherits.

## Usage

```python
attn = build("linformer", AttnConfig(extras={"linformer_k": 64}))
# non-causal by design (the projection mixes positions)
```

## Cost model

Analytic; storage precision is shown per row, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | storage precision | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- | --- |
| 32,768 | 128.0 | fp16 KV | 34 | 64 |
| 131,072 | 512.0 | fp16 KV | 137 | 64 |
| 1,048,576 | 4096.0 | fp16 KV | 1,100 | 64 |

## What to watch out for

The low-rank hypothesis in one line: attention is approximately rank-``E``,
so project K/V to ``E`` dimensions and the cost stops growing with context.
Non-causal, as published; it is the ancestor of every feature-axis compression
(MLA included), which is why it stays in the catalogue despite never shipping.

## Validation

```bash
python -m pytest tests -q -k linformer
python -m linattn.cli select --variant linformer --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
