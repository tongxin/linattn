# `saga`

**Family:** `frontier` &nbsp;|&nbsp; **Training required:** yes &nbsp;|&nbsp; **Class:** `SAGAAttention`

Elata et al., SAGA / More Value per Key (NeurIPS 2026) -- https://arxiv.org/abs/2610.04753

## Mechanism

Sparse Asymmetric Group-Query Attention with approximate top-N selection.

Elata et al., *More Value per Key: Asymmetric Sparse Attention for Faster LLM
Decoding* (NeurIPS 2026), https://arxiv.org/abs/2610.04753

Faithful core: sparse token selection decouples the number of key heads from the
number of value heads, and the memory model reports that asymmetry explicitly.
The paper trains Atop-N and SAGA; this implementation uses the deployable
inference semantics -- exact top-N by the current query -- and lets
``extras["key_heads"]`` choose the key-side allocation.  Value heads stay at
``cfg.n_kv_heads``.

## Usage

```python
attn = build("saga", AttnConfig(n_kv_heads=8, topk_tokens=2048,
                                extras={"key_heads": 2}))
```

## Cost model

Analytic; storage precision is shown per row, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | storage precision | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- | --- |
| 32,768 | 80.0 | fp16 KV | 18,694 | 2,052 |
| 131,072 | 320.0 | fp16 KV | 285,882 | 2,052 |
| 1,048,576 | 2560.0 | fp16 KV | 18,049,652 | 2,052 |

## What to watch out for

2026 correction to the "just read fewer tokens" story: once the selected
probability-value product is small, QK scoring and key metadata dominate, so
allocating fewer heads to keys and more to values can be the right trade.
This reference uses exact top-N for the inference semantics; the paper trains
Atop-N and reports a conversion fine-tune for pretrained GQA models. The
memory table is therefore the point: the key-side heads are configurable
independently of the value-side heads.

## Validation

```bash
python -m pytest tests -q -k saga
python -m linattn.cli select --variant saga --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
