# `hla_mixing`

**Family:** `modern` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `HLAMixingAttention`

Chen et al., HLA: hybrid linear attention via chunk-wise dynamic mixing (2026) -- https://arxiv.org/abs/2610.05842

## Mechanism

Chunk-wise dynamic mixing of full attention and a linear state.

Chen et al., *HLA: Expressive Hybrid Linear Attention via Chunk-Wise Dynamic
Mixing* (2026), https://arxiv.org/abs/2610.05842

Within a chunk the layer computes exact causal attention and, in parallel, a
read from a linear-attention state written by previous chunks; a *data-dependent*
scalar gate (from the chunk's mean query norm) mixes the two. This is the
Infini-attention family with the gate made per-chunk and data-dependent, which is
HLA's stated contribution.

## Usage

```python
attn = build("hla_mixing", AttnConfig(extras={"gate_tau": 1.0}))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | attn FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 2.0 | 17,592 | 32,896 |
| 131,072 | 2.0 | 281,475 | 131,200 |
| 1,048,576 | 2.0 | 18,014,399 | 1,048,704 |

## What to watch out for

2026 line: a *data-dependent, per-chunk* gate between exact chunk-local
attention and a linear state read. Same family as Infini-attention, but the
gate is dynamic rather than a learned scalar -- the paper's claim is that the
mixing policy, not the state, is what limits expressivity.

## Validation

```bash
python -m pytest tests -q -k hla_mixing
python -m linattn.cli select --variant hla_mixing --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
