# State mixers (compressed state)

These are the models that replaced the growing KV cache with a fixed-size
recurrent state. They live in `src/linattn/variants/recurrent.py` and are
**not** `AttentionVariant`s: they take `x` of shape `(B, T, d_model)` and own
their q/k/v projections, so they cannot be driven through the `(q, k, v)`
harness. `python -m linattn.cli variants` lists them alongside the attention
variants, and `python -m linattn.cli mixers` prints their state size and
per-token cost.

All four are one recurrence with different gates:

```
S_t = (I - diag(b_t) k_t k_t^T) diag(a_t) S_{t-1} + diag(w_t) k_t v_t^T
o_t = q_t^T S_t
```

with the decay `a_t` scalar or channel-wise and the gates `b_t`/`w_t`
collapsed or separate. That is why one chunkwise kernel with a flag covers
the whole family -- and why GDN-2 is the general case rather than a new
mechanism.

| kind | decay | gates | recurrence | citation |
| --- | --- | --- | --- | --- |
| `ssd` | scalar per head (learned A_log) | write only (beta) | `S_t = diag(a_t) S_{t-1} + b_t k_t v_t^T` | [Dao & Gu, Transformers are SSMs (2024)](https://arxiv.org/abs/2405.21060) |
| `gdn` | scalar per head | one scalar beta for erase and write | `S_t = (I - b_t k_t k_t^T) diag(a_t) S_{t-1} + b_t k_t v_t^T` | [Yang et al., Gated Delta Networks (ICLR 2025)](https://arxiv.org/abs/2412.06464) |
| `kda` | channel-wise (per feature dimension) | one scalar beta for erase and write | `S_t = (I - b_t k_t k_t^T) diag(a_t) S_{t-1} + b_t k_t v_t^T, a_t per channel` | [Kimi Team, Kimi Linear (2025)](https://arxiv.org/abs/2510.26692) |
| `gdn2` | channel-wise | separate channel-wise erase (b) and write (w) gates | `S_t = (I - diag(b_t) k_t k_t^T) diag(a_t) S_{t-1} + diag(w_t) k_t v_t^T` | [Yang et al., Gated DeltaNet-2 (2026)](https://arxiv.org/abs/2605.22791) |

## Usage

```python
from linattn.registry import build_mixer

layer = build_mixer("kda", d_model=2048, n_heads=16, d_state=128, chunk=64)
y_chunked, state = layer.forward_chunked(x)     # the fast path (matmuls + one solve)
y_exact, state2 = layer.forward_recurrent(x)    # the literal per-token recurrence
```

`forward_chunked` and `forward_recurrent` are asserted equal in float64 (and in
gradients) by `tests/test_recurrent.py`, so the chunked path is a verified
re-association of the recurrence rather than an approximation.

## State and cost

| kind | state floats (d_state=16, d_v=64) | state floats (d_state=128, d_v=128) |
| --- | --- | --- |
| `ssd` | 16,384 | 262,144 |
| `gdn` | 16,384 | 262,144 |
| `kda` | 16,384 | 262,144 |
| `gdn2` | 16,384 | 262,144 |

The state size and the per-token FLOPs are **identical across kinds** -- only
the update rule (and therefore quality) differs. What changes is capacity and
expressivity: `reports/state_capacity.md` measures how many associations a
state of a given size can hold before recall collapses.

## Layer interleaving

`variants/hybrid.py` builds `HybridLM`, which interleaves these layers with
full-attention layers (the Kimi Linear 3:1 pattern). `experiments/hybrid_recall.py`
compares dense, hybrid and linear-only stacks on MQAR, and states honestly that
at laptop scale MQAR does not separate the families.

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md)
