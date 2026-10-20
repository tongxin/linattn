# `jssa`

**Family:** `frontier` &nbsp;|&nbsp; **Training required:** yes &nbsp;|&nbsp; **Class:** `JSSAAttention`

Song et al., JSSA: Joint Subspace Sparse Attention (2026) -- https://arxiv.org/abs/2610.19026

## Mechanism

Joint Subspace Sparse Attention: rank tokens in a shared QK subspace.

Song et al., *Joint Subspace Sparse Attention for Efficient Long-Context
Decoding* (2026), https://arxiv.org/abs/2610.19026

Faithful core: learn a shared low-dimensional query--key subspace and use it for
token ranking, then run exact attention over selected tokens.  The reference
derives the subspace from the leading key covariance directions; the paper adds
a score-preservation objective and a gather-free kernel.  Those training and
kernel details are not reproduced.

## Usage

```python
attn = build("jssa", AttnConfig(topk_tokens=2048, extras={"rank": 16}))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 3,335 | 2,052 |
| 131,072 | 512.0 | 39,729 | 2,052 |
| 1,048,576 | 4096.0 | 2,288,153 | 2,052 |

## What to watch out for

2026 line: token-level selection is finer than pages, but a low-dimensional
ranking score can lose the structure that made the dense inner product useful.
The reference takes the leading key-covariance subspace; the paper learns a
shared subspace with a score-preservation objective and adds a gather-free
kernel. At full rank and full token budget this reduces to dense attention,
which is the correctness check.

## Validation

```bash
python -m pytest tests -q -k jssa
python -m linattn.cli select --variant jssa --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
