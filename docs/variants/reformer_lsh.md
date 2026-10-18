# `reformer_lsh`

**Family:** `academic` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `ReformerLSHAttention`

Kitaev et al., Reformer (ICLR 2020) -- https://arxiv.org/abs/2001.04451

## Mechanism

Angular-LSH buckets: a query reads the keys hashed into its bucket.

Kitaev et al., *Reformer* (ICLR 2020), https://arxiv.org/abs/2001.04451

This is the *semantics* (bucket-equality mask), not the chunked kernel; one
bucket makes it exact, which the tests use. The hash is shared across heads, as a
fixed random projection would be.

## Usage

```python
attn = build("reformer_lsh", AttnConfig(extras={"buckets": 16}))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | attn FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 17,592 | 4,096 |
| 131,072 | 512.0 | 281,475 | 16,384 |
| 1,048,576 | 4096.0 | 18,014,399 | 131,072 |

## What to watch out for

Hashing is the cheapest way to get sub-quadratic attention, and the least
accurate on i.i.d. data (we measured relative error growing from 0 at one
bucket to ~1.9 at 32). Its lasting contribution is the insight that *bucketing*
can be a routing mechanism, which is what later learned indexers (DSA) do
better.

## Validation

```bash
python -m pytest tests -q -k reformer_lsh
python -m linattn.cli select --variant reformer_lsh --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
