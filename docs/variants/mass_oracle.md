# `mass_oracle`

**Family:** `modern` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `MassOracleAttention`

Tian et al., COBS: block selection is attention-mass ranking (2026) -- https://arxiv.org/abs/2607.09052

## Mechanism

The COBS oracle: select blocks by their *exact* attention mass.

Tian et al., *COBS: Cumulant Order Block Sparse Attention* (2026),
https://arxiv.org/abs/2607.09052

COBS formalises block selection as ranking blocks by attention mass and shows
that *if* selection were exact, block-sparse attention would match dense. This
class implements that oracle: it computes dense attention to measure each block's
mass, then reads only the top blocks. It is an **analysis instrument**, not a
method -- it peeks at the answer -- and exists to bound what any selector could
achieve at a given budget.

## Usage

```python
# analysis instrument (peeks at attention mass); not deployable
attn = build("mass_oracle", AttnConfig(extras={"topk_blocks": 8, "sinks": 2}))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | attn FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 17,592 | 1,028 |
| 131,072 | 512.0 | 281,475 | 1,028 |
| 1,048,576 | 4096.0 | 18,014,399 | 1,028 |

## What to watch out for

Not a method: the COBS oracle. It selects blocks by their exact attention mass
(computed by running dense attention first), so it bounds what *any* selector
could achieve at the same budget. Use it to calibrate a real selector -- if a
deployable method is far below the oracle at equal reads, the gap is in the
selector, not in the idea of sparsity.

## Validation

```bash
python -m pytest tests -q -k mass_oracle
python -m linattn.cli select --variant mass_oracle --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
