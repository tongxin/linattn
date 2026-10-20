# `nsa`

**Family:** `native` &nbsp;|&nbsp; **Training required:** yes &nbsp;|&nbsp; **Class:** `NSAAttention`

Yuan et al., Native Sparse Attention (ACL 2025) -- https://arxiv.org/abs/2502.11089

## Mechanism

Three-branch native sparse attention (inference semantics of NSA).

## Usage

```python
attn = build("nsa", AttnConfig(block_size=64, window_size=512, topk_blocks=16,
                               compress_ratio=16, extras={"selected_blocks": 16,
                                                          "gate_mode": "learned"}))
attn.fit_gates(q_calib, k_calib, v_calib, steps=200)
```

## Cost model

Analytic, fp16 storage convention, 32 heads / 8 KV heads / head dim 128, block 64, window 512, budget 4096, top-16 blocks, top-2048 tokens:

| T | state (MiB) | prefill attention FLOPs (G) | KV reads/query |
| --- | --- | --- | --- |
| 32,768 | 128.0 | 1,924 | 3,584 |
| 131,072 | 512.0 | 20,891 | 9,728 |
| 1,048,576 | 4096.0 | 1,152,288 | 67,072 |

## What to watch out for

Read the three branches as three different bets: the compressed branch buys
global coverage cheaply, the selected branch buys precision, the window branch
buys locality and stops the other two from being shortcut. The gates are
learned, so `gate_mode="uniform"` is an inference-time proxy and
`fit_gates` is how the branch weights can be recovered without a full
pretraining run. NSA's reported quality wins over full attention are the
strongest evidence in this repository's taxonomy that *native* sparsity can
help rather than merely preserve quality.

## Validation

```bash
python -m pytest tests -q -k nsa
python -m linattn.cli select --variant nsa --seq 1024 --structured
```

Covered by `tests/test_budgets.py` (budget honesty), `tests/test_selection.py` or `tests/test_compressed.py` (quality invariants), and `tests/test_kernels.py` (the accelerated path for this family).

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md) &nbsp;|&nbsp; [evaluation evidence](../evaluation-evidence.md)
