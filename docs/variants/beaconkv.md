# `beaconkv`

**Family:** `eviction` &nbsp;|&nbsp; **Training required:** no &nbsp;|&nbsp; **Class:** `BeaconKVAttention`

Kim et al., BeaconKV (ICML 2026) -- https://arxiv.org/abs/2609.04971

## Mechanism

BeaconKV: keep compact representatives of historical query clusters.

Kim et al., *BeaconKV: Key-Value Cache Compression Guided by Beacon Queries
for Efficient Large Reasoning Model Inference* (ICML 2026),
https://arxiv.org/abs/2609.04971

The reference keeps one representative per query cluster so that a later
"thought revisiting" query does not have to be predicted from the most recent
window alone.  This implementation uses farthest-point representatives from the
observation window, scores keys by their attention mass under those beacons, and
retains the prompt, sinks and recent window.  The learned clustering and
long-generation calibration are not reproduced.

## Usage

```python
attn = build("beaconkv", AttnConfig(cache_budget=512, window_size=64,
                                     extras={"n_beacons": 4}))
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 3,986 | 4,096 |
| 131,072 | 512.0 | 17,180 | 4,096 |
| 1,048,576 | 4096.0 | 140,325 | 4,096 |

## What to watch out for

BeaconKV targets the failure mode that made "recency is enough" look false on
long reasoning traces: a later thought can revisit an early plan. Keeping
representatives of past query clusters makes that revisit visible without
retaining the full query history. The reference uses farthest-point
representatives and attention mass; the paper learns the clusters and tunes
them on long generation traces.

## Validation

```bash
python -m pytest tests -q -k beaconkv
python -m linattn.cli select --variant beaconkv --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
