# `sliding_window`

**Family:** `local` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `SlidingWindowAttention`

Xiao et al., StreamingLLM: attention sinks (ICLR 2024) -- https://arxiv.org/abs/2309.17453

## Mechanism

Causal sliding window of ``window_size`` tokens plus ``sink_size`` leading tokens.

References
----------
Beltagy et al., *Longformer* (2020), https://arxiv.org/abs/2004.05150
Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (2024),
https://arxiv.org/abs/2309.17453

## Usage

```python
attn = build("sliding_window", AttnConfig(window_size=4096, sink_size=4))
out, state = attn.forward(q, k, v)
```

## Cost model

Analytic; storage precision is shown per row, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | storage precision | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- | --- |
| 32,768 | 2.0 | fp16 KV | 17,592 | 516 |
| 131,072 | 2.0 | fp16 KV | 281,475 | 516 |
| 1,048,576 | 2.0 | fp16 KV | 18,014,399 | 516 |

## What to watch out for

The cheapest thing that works, and the branch that every production sparse
configuration keeps unconditionally. Beware of measuring it on retrieval
tasks: a sliding window *cannot* retrieve, so its score on RULER is a
measurement of how much of that benchmark is local.

## Validation

```bash
python -m pytest tests -q -k sliding_window
python -m linattn.cli select --variant sliding_window --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
