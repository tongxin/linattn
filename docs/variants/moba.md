# `moba`

**Family:** `selection` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `MoBAAttention`

Lu et al., MoBA (2025) -- https://arxiv.org/abs/2502.13189

## Mechanism

Mixture of Block Attention: per-token top-k routing over KV blocks.

Lu et al., *MoBA: Mixture of Block Attention for Long-Context LLMs* (2025),
https://arxiv.org/abs/2502.13189

The gate is the inner product between the query and the **mean-pooled key
block**; each query routes to its ``topk_blocks`` predecessors, always
including its own (causally masked) block.  MoBA can emulate sliding-window
and sink attention as special cases of the routing function, which is the
paper's "less structure" argument.

## Usage

```python
attn = build("moba", AttnConfig(block_size=4096, topk_blocks=12))
```

## Cost model

Analytic; storage precision is shown per row, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | storage precision | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- | --- |
| 32,768 | 128.0 | fp16 KV | 962 | 1,536 |
| 131,072 | 512.0 | fp16 KV | 5,498 | 1,536 |
| 1,048,576 | 4096.0 | fp16 KV | 167,126 | 1,536 |

## What to watch out for

MoBA's selling point is conceptual: sliding-window and sink attention fall out
as special cases of the router, so the model can learn them instead of being
told. The released inference recipe (block size 4096, top-12 blocks, 100B tokens
of continual pretraining with the pattern active) is the configuration that
reports parity with full attention at 1M context; a training-free MoBA-style
router on a dense model will not reproduce that.

## Validation

```bash
python -m pytest tests -q -k moba
python -m linattn.cli select --variant moba --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
