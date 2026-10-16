# `minference`

**Family:** `selection` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `MInferenceAttention`

Jiang et al., MInference 1.0 (NeurIPS 2024) -- https://arxiv.org/abs/2407.02490

## Mechanism

Pattern-based prefill sparsity: A-shape, Vertical-Slash, Block-Sparse heads.

Jiang et al., *MInference 1.0: Accelerating Pre-filling for Long-Context LLMs
via Dynamic Sparse Attention* (NeurIPS 2024), https://arxiv.org/abs/2407.02490

Faithful points:

* the Vertical-Slash estimator uses the **last 64 query rows** to locate the
  important vertical lines (columns) and slash lines (diagonals);
* the Block-Sparse estimator mean-pools q and k into 64-token blocks and
  scores blocks with the pooled inner product -- equivalent to mean-pooling
  the attention matrix, because pooling and matmul commute;
* the A-shape head is the static "initial tokens + local window" pattern;
* ``pattern="auto"`` performs the kernel-aware search online, per head,
  picking whichever pattern maximises attention-mass recall at equal read
  budget.  The paper does this offline once per model; doing it per call keeps
  the trade-off measurable instead of assumed.

## Usage

```python
# one pattern per head (the paper searches offline); "auto" searches per call
attn = build("minference", AttnConfig(block_size=64, topk_blocks=8,
                                      extras={"pattern": "vertical_slash", "last_q": 64,
                                              "n_vertical": 4, "n_slash": 4}))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | attn FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 863 | 1,536 |
| 131,072 | 512.0 | 3,505 | 1,536 |
| 1,048,576 | 4096.0 | 31,886 | 1,536 |

## What to watch out for

The pattern search is the interesting part: the paper assigns each head a
pattern offline (`pattern="a_shape" | "vertical_slash" | "block_sparse"`) and
then builds dynamic indices per prompt. `pattern="auto"` here performs the
search online, on the first chunk, by mass recall at equal read budget.
`reports/suite_*.md` shows what happens when you do that: the warm-up chunk is
short, so its attention is local, so the search picks a local pattern and
keeps it for the rest of the sequence. The paper's offline per-head search
over long representative prompts is therefore load-bearing rather than a
detail -- an online version needs a long probe (or a per-head prior) to behave
the same way. Note too the 1x64 block size for vertical-slash: a "vertical
line" is one token wide, which is exactly the kind of pattern that
block-granular kernels handle badly.

## Validation

```bash
python -m pytest tests -q -k minference
python -m linattn.cli select --variant minference --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
