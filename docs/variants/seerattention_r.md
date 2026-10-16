# `seerattention_r`

**Family:** `selection` &nbsp;|&nbsp; **Training required:** yes &nbsp;|&nbsp; **Class:** `SeerAttentionR`

Gao et al., SeerAttention-R (2025) -- https://arxiv.org/abs/2506.08889

## Mechanism

Learned block pooling gate, adapted for long decoding.

Gao et al., *SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs*
(ICLR 2025) and *SeerAttention-R: Sparse Attention Adaptation for Long
Reasoning* (2025), https://arxiv.org/abs/2506.08889

A ``BlockPool`` averages q and k inside each block, a learned projection maps
the pooled vectors to a low-dimensional space, and a sigmoid gate decides
whether a block is worth reading.  Unlike the training-free selectors above,
the gate carries weights, so this class provides :meth:`fit_gate`, which
trains the gate against the dense block mass.  That makes the capacity of the
learned-gate parameterisation measurable, and it is also why this family is
the one that needs a fine-tuning stage in practice.

## Usage

```python
attn = build("seerattention_r", AttnConfig(block_size=64,
                                           extras={"pool_dim": 16, "seer_topn": 8}))
attn.fit_gate(q_calib, k_calib, v_calib, steps=300)
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | attn FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 825 | 1,536 |
| 131,072 | 512.0 | 3,299 | 1,536 |
| 1,048,576 | 4096.0 | 26,388 | 1,536 |

## What to watch out for

The gate is learned, and this page is the place to see why that matters: a
random gate selects essentially at random, while a fitted one can approach the
dense block distribution. In production this family needs a fine-tuning stage,
which is why it appears in the "can you afford to retrain?" branch of the
[frontier gap analysis](../frontier-gap.md) rather than the plug-and-play one.

## Validation

```bash
python -m pytest tests -q -k seerattention_r
python -m linattn.cli select --variant seerattention_r --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
