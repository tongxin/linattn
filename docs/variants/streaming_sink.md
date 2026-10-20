# `streaming_sink`

**Family:** `local` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `StreamingSinkAttention`

Xiao et al., StreamingLLM (ICLR 2024) -- https://arxiv.org/abs/2309.17453

## Mechanism

Attention sinks + window, restricted to exactly ``window_size`` recent tokens.

Same computation as :class:`SlidingWindowAttention`; kept as a separate name
because the literature uses "StreamingLLM-style" for the eviction contract
(drop evicted tokens permanently) rather than for a mask applied at read time.

## Usage

```python
attn = build("streaming_sink", AttnConfig(window_size=1024, sink_size=4))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 2.0 | 17,592 | 516 |
| 131,072 | 2.0 | 281,475 | 516 |
| 1,048,576 | 2.0 | 18,014,399 | 516 |

## What to watch out for

Included because "StreamingLLM-style" is the vocabulary used in serving
stacks for *permanently dropping* evicted tokens, which is subtly different
from masking reads at attention time. The mask-based variant here has the
same arithmetic as the sliding window; the difference is the cache contract.

## Validation

```bash
python -m pytest tests -q -k streaming_sink
python -m linattn.cli select --variant streaming_sink --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
