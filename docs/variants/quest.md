# `quest`

**Family:** `selection` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `QuestAttention`

Tang et al., Quest (ICML 2024) -- https://arxiv.org/abs/2406.10774

## Mechanism

Quest: page-level selection using channel-wise key bounds.

For every page the cache stores the channel-wise minimum ``m`` and maximum
``M`` of its keys.  Because ``max(q_d*m_d, q_d*M_d)`` upper-bounds ``q_d*k_d``
for every key in the page, ``sum_d max(q_d m_d, q_d M_d)`` upper-bounds the
page's attention logit.  Selecting the pages with the largest upper bound is a
query-aware approximation of selecting the pages that contain the
highest-attention tokens.

Tang et al., *Quest: Query-Aware Sparsity for Efficient Long-Context LLM
Inference* (ICML 2024), https://arxiv.org/abs/2406.10774

The paper's own caveats are reproduced: the first two layers are kept dense
(their sparsity is low), and the selection is per query head, so under GQA a
group must read the *union* of its heads' pages -- the token budget inflates
by up to the group size.  ``extras["gqa_union"] = False`` switches to a
max-pooled shared selection, which is the fair-comparison setting.

## Usage

```python
attn = build("quest", AttnConfig(block_size=16, topk_blocks=8, extras={"page_size": 16}))
# keep the first two layers dense, as the paper does
attn = build("quest", AttnConfig(block_size=16, topk_blocks=8,
                                 extras={"page_size": 16, "dense_layers": 2, "layer_idx": 6}))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | attn FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 962 | 1,536 |
| 131,072 | 512.0 | 5,498 | 1,536 |
| 1,048,576 | 4096.0 | 167,126 | 1,536 |

## What to watch out for

Quest's bound is not an estimate: channel-wise min/max give a true upper bound
on the page's best logit, which is why it rarely misses the top token. Two
caveats are reproduced in the implementation because they matter in practice:
the first layers should stay dense (their sparsity is low), and under GQA the
per-head page selections must be unioned, so the effective budget can inflate
by the group ratio unless the kernel handles head divergence explicitly.

## Validation

```bash
python -m pytest tests -q -k quest
python -m linattn.cli select --variant quest --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
