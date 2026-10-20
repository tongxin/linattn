# `bigbird`

**Family:** `academic` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `BigBirdAttention`

Zaheer et al., Big Bird (NeurIPS 2020) -- https://arxiv.org/abs/2007.14062

## Mechanism

Window + random + global blocks as an explicit mask.

Zaheer et al., *Big Bird* (NeurIPS 2020), https://arxiv.org/abs/2007.14062

The pattern (not their block-sparse kernel) is what makes BigBird a universal
approximator of sequence functions; the random component is what later work
replaced with learned routing.

## Usage

```python
attn = build("bigbird", AttnConfig(window_size=256,
                                   extras={"window": 256, "n_random": 3, "n_global": 2}))
```

## Cost model

Analytic; storage precision is shown per row, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | storage precision | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- | --- |
| 32,768 | 128.0 | fp16 KV | 17,592 | 517 |
| 131,072 | 512.0 | fp16 KV | 281,475 | 517 |
| 1,048,576 | 4096.0 | fp16 KV | 18,014,399 | 517 |

## What to watch out for

Window + random + global is the sparsity pattern with a universality proof. It
is also the pattern most production systems rejected: random blocks are
hardware-hostile and, unlike a learned selector, cannot adapt to content. The
example of an academic result that is *right* and still not adopted.

## Validation

```bash
python -m pytest tests -q -k bigbird
python -m linattn.cli select --variant bigbird --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
