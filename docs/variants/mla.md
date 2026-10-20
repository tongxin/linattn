# `mla`

**Family:** `compressed` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `MLAttention`

DeepSeek-AI, DeepSeek-V2 (2024) -- https://arxiv.org/abs/2405.04434

## Mechanism

Latent KV cache with per-head up-projections.

## Usage

```python
# the cache stores the latent; K/V are materialised inside the kernel
attn = build("mla", AttnConfig(n_heads=128, n_kv_heads=1, head_dim=128,
                               extras={"latent_dim": 512}))
print(attn.compression_ratio())     # ~9x for DeepSeek-V2-style shapes
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 2.0 | 8,814 | 32,768 |
| 131,072 | 8.0 | 140,807 | 131,072 |
| 1,048,576 | 64.0 | 9,007,758 | 1,048,576 |

## What to watch out for

MLA is not sparsity -- and that is the point to take away. Compressing the
feature axis (one latent per token) is the cheapest large win available and it
composes with everything else: DSA sits on top of MLA, and V4.1-Flash adds FP4
and cross-layer reuse on top of the compressed latent. The `compression_ratio`
here reports the KV-footprint reduction for DeepSeek-V2-style shapes.

## Validation

```bash
python -m pytest tests -q -k mla
python -m linattn.cli select --variant mla --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
