# `dsa`

**Family:** `selection` &nbsp;|&nbsp; **Training required:** yes &nbsp;|&nbsp; **Class:** `DSAAttention`

DeepSeek-AI, DeepSeek-V3.2 / DSA + lightning indexer (2025) -- https://huggingface.co/deepseek-ai/DeepSeek-V3.2

## Mechanism

DeepSeek Sparse Attention: learned lightning indexer + top-k token selection.

DeepSeek-AI, *DeepSeek-V3.2* (2025), https://arxiv.org/abs/2512.02556 (tech
report distributed with the release; see also *DeepSeek-V3.2-Exp*).

The indexer score between query ``t`` and key ``s`` is

.. math::  I_{t,s} = \sum_{j=1}^{H_I} w_{t,j}\,\mathrm{ReLU}(q_{t,j}\cdot k_s)

with ``H_I`` indexer heads and learned, per-token head weights ``w``.  Keys
with the top ``k`` scores (2048 in the V3.2 release) are then read by the core
attention, which in DeepSeek's case is MLA over the selected tokens.

``compress_ratio > 1`` implements the **compressed-indexing part** of CSA/CSA2:
the indexer scores pooled keys and the selected groups expand to the tokens they
contain -- the mechanism that makes the indexer cheap enough to sit in front of
every layer.

It is **not** CSA2.  CSA2 (DeepSeek-V4.1-Flash, arXiv:2609.19969) also (a)
reuses one cached KV copy across a group of layers and (b) stores the latent in
FP4.  This variant does neither: it is single-layer and fp32, and the ``csa2``
entry in ``kv/format.py`` is a cost-model constant taken from the report, not an
implementation.  See ``docs/kv-layer.md`` for the component-by-component mapping
and ``docs/roadmap.md`` for the gap.

Because the indexer is a *learned* component, a randomly initialised one is
meaningless; :meth:`fit` trains the indexer projections on calibration inputs
against the dense attention mass, which is what makes the capacity question
("can a low-dimensional ReLU indexer rank keys the way softmax does?")
testable.

## Usage

```python
attn = build("dsa", AttnConfig(block_size=64, topk_tokens=2048,
                               extras={"indexer_heads": 64, "indexer_dim": 128, "compress_ratio": 1}))
attn.fit(q_calib, k_calib, v_calib, steps=300)     # the indexer must be trained
# compressed variant (CSA-style): index pooled keys, expand selected groups
csa = build("dsa", AttnConfig(topk_tokens=2048, extras={"compress_ratio": 8}))
```

## Cost model

Analytic; storage precision is shown per row, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | storage precision | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- | --- |
| 32,768 | 128.0 | fp16 KV | 893 | 1,536 |
| 131,072 | 512.0 | fp16 KV | 4,398 | 1,536 |
| 1,048,576 | 4096.0 | fp16 KV | 96,757 | 1,536 |

## What to watch out for

The lightning indexer is cheap per key but touches every key, so its cost grows
quadratically while the attention it gates grows linearly: at long context the
indexer becomes the bottleneck (the 256 GB intermediate in the naive CSA
implementation is the extreme version of this). Three mitigations are in the
literature and matter more than the selection rule itself: cross-layer index
reuse, hierarchical/compressed indexing, and streaming top-k without
materialising scores. `fit` exists because an unfitted indexer is meaningless:
run it on your own calibration data before believing any selection-recall
number.

## Validation

```bash
python -m pytest tests -q -k dsa
python -m linattn.cli select --variant dsa --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
