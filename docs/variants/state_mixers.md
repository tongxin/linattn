# State mixers (compressed state)

These are the models that replaced the growing KV cache with a fixed-size
recurrent state. They live in `src/linattn/variants/recurrent.py` and are
**not** `AttentionVariant`s: they take `x` of shape `(B, T, d_model)` and own
their q/k/v projections, so they cannot be driven through the `(q, k, v)`
harness. `python -m linattn.cli variants` lists them alongside the attention
variants, and `python -m linattn.cli mixers` prints their state size and
per-token cost.

The first four are one recurrence with different gates:

```
S_t = (I - diag(b_t) k_t k_t^T) diag(a_t) S_{t-1} + diag(w_t) k_t v_t^T
o_t = q_t^T S_t
```

with the decay `a_t` scalar or channel-wise and the gates `b_t`/`w_t`
collapsed or separate. That is why one chunkwise kernel with a flag covers
the whole family -- and why GDN-2 is the general case rather than a new
mechanism. Q-Delta adds query feedback, KDN makes the write gain come from
tracked uncertainty, and SwiLA routes a mixture of delta-rule states.

| kind | decay | gates | recurrence | citation |
| --- | --- | --- | --- | --- |
| `ssd` | scalar per head (learned A_log) | write only (beta) | `S_t = diag(a_t) S_{t-1} + b_t k_t v_t^T` | [Dao & Gu, Transformers are SSMs (2024)](https://arxiv.org/abs/2405.21060) |
| `gdn` | scalar per head | one scalar beta for erase and write | `S_t = (I - b_t k_t k_t^T) diag(a_t) S_{t-1} + b_t k_t v_t^T` | [Yang et al., Gated Delta Networks (ICLR 2025)](https://arxiv.org/abs/2412.06464) |
| `kda` | channel-wise (per feature dimension) | one scalar beta for erase and write | `S_t = (I - b_t k_t k_t^T) diag(a_t) S_{t-1} + b_t k_t v_t^T, a_t per channel` | [Kimi Team, Kimi Linear (2025)](https://arxiv.org/abs/2510.26692) |
| `gdn2` | channel-wise | separate channel-wise erase (b) and write (w) gates | `S_t = (I - diag(b_t) k_t k_t^T) diag(a_t) S_{t-1} + diag(w_t) k_t v_t^T` | [Yang et al., Gated DeltaNet-2 (2026)](https://arxiv.org/abs/2605.22791) |
| `qdelta` | scalar per head | key-write strength plus a learned query-feedback coefficient | `S_t = (I - b_t(k_t + λ_t q_t)k_t^T) diag(a_t) S_{t-1} + b_t k_t v_t^T` | [Park et al., Q-Delta (ICML 2026)](https://arxiv.org/abs/2606.08804) |
| `kdn` | channel-wise | uncertainty-derived Kalman gain; diagonal covariance state | `S_t = (I - κ_t k_t^T) diag(a_t) S_{t-1} + κ_t v_t^T, κ_t = p_t ⊙ k_t / (r_t + k_t^T p_t k_t)` | [Bui et al., Kalman Delta Networks (2026)](https://arxiv.org/abs/2609.07816) |
| `swila` | per expert, per head | soft posterior responsibilities over J memory experts | `S_t^(j) = a_tj S_{t-1}^(j) + b_tj π_tj (v_t - a_tj k_t^T S_{t-1}^(j)) k_t` | [Lee et al., Switching Linear Attention (COLM 2026)](https://arxiv.org/abs/2609.39034) |

## Usage

```python
from linattn.registry import build_mixer

layer = build_mixer("kda", d_model=2048, n_heads=16, d_state=128, chunk=64)
y_chunked, state = layer.forward_chunked(x)     # the fast path (matmuls + one solve)
y_exact, state2 = layer.forward_recurrent(x)    # the literal per-token recurrence
```

For SSD/GDN/KDA/GDN-2, `forward_chunked` and `forward_recurrent` are asserted
equal in float64 (and in gradients) by `tests/test_recurrent.py`, so the
chunked path is a verified re-association of the recurrence. Q-Delta, KDN and
SwiLA currently expose the exact reference recurrence; their papers' WY /
covariance-scan kernels are cited but not reproduced. `forward_chunked` is an
explicit alias for those three rather than a claim of a serial-to-parallel
transformation.

## State and cost

| kind | state floats (d_state=16, d_v=64) | state floats (d_state=128, d_v=128) |
| --- | --- | --- |
| `ssd` | 16,384 | 262,144 |
| `gdn` | 16,384 | 262,144 |
| `kda` | 16,384 | 262,144 |
| `gdn2` | 16,384 | 262,144 |
| `qdelta` | 16,384 | 262,144 |
| `kdn` | 16,448 | 264,192 |
| `swila` | 65,536 | 1,048,576 |

SSD/GDN/KDA/GDN-2 and Q-Delta use the same matrix state. Diagonal KDN adds one
uncertainty value per key channel; SwiLA multiplies the state by its number of
mixture experts. The state table reports those auxiliary costs rather than
hiding them in a fixed-size claim.

## Layer interleaving

`variants/hybrid.py` builds `HybridLM`, which interleaves these layers with
full-attention layers (the Kimi Linear 3:1 pattern). `experiments/hybrid_recall.py`
compares dense, hybrid and linear-only stacks on MQAR, and states honestly that
at laptop scale MQAR does not separate the families.

---

[back to the variant index](index.md) &nbsp;|&nbsp; [survey](../survey.md)
