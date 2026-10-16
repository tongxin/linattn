# `xattention`

**Family:** `selection` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `XAttention`

Xu et al., XAttention (ICML 2025) -- https://arxiv.org/abs/2503.16428

## Mechanism

XAttention: strided antidiagonal block scoring with a single threshold.

Xu et al., *XAttention: Block Sparse Attention with Antidiagonal Scoring*
(ICML 2025), https://arxiv.org/abs/2503.16428

Why antidiagonals: within a block, a vertical line, a slash line and a dense
blob all intersect the antidiagonals, so summing *only* the strided
antidiagonals (``1/stride`` of the block's entries) recovers a good proxy for
the block's softmax mass at a fraction of the cost of pooling-based scores.
The paper then softmax-normalises the block scores and keeps the minimal set
of blocks whose cumulative score reaches ``threshold`` (0.9-0.95), with
per-head thresholds found offline by dynamic programming; here the threshold
is a config knob and ``threshold_per_head`` allows the offline-searched
values to be supplied.

## Usage

```python
attn = build("xattention", AttnConfig(block_size=64, threshold=0.9,
                                      extras={"xattn_block": 8, "antidiagonal_stride": 4}))
# per-head thresholds found offline by the paper's DP procedure can be supplied:
attn = build("xattention", AttnConfig(block_size=64,
                                      extras={"threshold_per_head": [0.9] * 32, "xattn_block": 8}))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | attn FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 1,100 | 1,536 |
| 131,072 | 512.0 | 7,697 | 1,536 |
| 1,048,576 | 4096.0 | 307,863 | 1,536 |

## What to watch out for

The antidiagonal argument is the most transferable idea in the block-selection
family: a strided sum of 1/stride of the block's entries is a good proxy
because every vertical/slash/dense pattern intersects antidiagonals. The
paper's per-head thresholds come from an offline DP; if you skip that step you
get a uniformly-thresholded variant, which is what the config knob here
defaults to. Its strongest evidence is video, where block structure is
stronger than in text -- do not extrapolate the 13.5x number to language
prefill.

## Validation

```bash
python -m pytest tests -q -k xattention
python -m linattn.cli select --variant xattention --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
