# Analytic prefill-FLOPs model

The `prefill attn FLOPs @T (G)` column in `python -m linattn.cli cost` is an
**analytic matmul-only estimate**, not a GPU counter and not an end-to-end latency
model. It is implemented in
[`src/linattn/metrics.py`](../src/linattn/metrics.py) and the variants'
`flops()` methods.

The purpose is to compare algorithmic work under a common convention. It does not
claim to reproduce a vendor profiler exactly.

## 1. Symbols and counting convention

| symbol | meaning |
| --- | --- |
| `B` | batch size |
| `H` | query-head count (`n_heads`) |
| `T` | sequence length passed as `--seq`; the `@T` in the header |
| `d` | head dimension |
| `R` | KV entries read per query for a sparse variant |

One multiply-add is counted as **2 FLOPs**. The table reports:

```text
displayed value (G) = F / 10^9
```

This is the standard “two matmuls” accounting for attention:

```text
QKᵀ:  2 · B · H · T · T · d
AV:    2 · B · H · T · T · d
```

so the base dense-attention estimate is:

```text
F_dense = 4 · B · H · T² · d
```

For a sparse selector with read budget `R`:

```text
F_sparse = 4 · B · H · T · R · d + F_selector
```

`F_selector` is added only where the variant implements `selection_flops()`. For
example, Quest charges the page-bound scoring pass, while the basic block mask
itself is not separately charged.

## 2. Important causal caveat

The base dense estimate currently counts the **full** `T × T` score matrix. A
strict causal count would count the lower triangle:

```text
F_causal = 4 · B · H · [T(T+1)/2] · d
         = 2 · B · H · T(T+1) · d
```

That is approximately half the base dense estimate. Some variants override the
base formula and already use the causal triangle; for example, MLA uses
`T(T+1)/2` for its core attention term. The analytic page documents both the code
convention and the causal interpretation so a verification difference can be
traced to the intended model rather than guessed at.

For `B=1`, `H=8`, `T=1024`, `d=64`:

```text
full matrix:   2,147,483,648 FLOPs = 2.147 G
causal matrix: 1,074,741,248 FLOPs = 1.075 G
```

The CLI displays `2.1` for dense in this configuration because it uses the current
full-matrix base formula.

## 3. What is included and excluded

Included:

- QK score matmuls;
- probability-value matmuls;
- variant-specific selector estimates where explicitly implemented;
- MLA latent up-projection in its override.

Not included:

- softmax exponentiation, subtraction, division or max/sum reductions;
- causal-mask construction and boolean-mask materialisation;
- top-k sorting, page metadata maintenance, gather/scatter, or memory traffic;
- Q/K/V input projections and the output projection in ordinary attention variants;
- KV-cache allocation, HBM/DRAM movement, or kernel launch overhead;
- actual achieved FLOP/s on any GPU, CPU or accelerator.

GQA changes KV *bytes* by sharing keys and values across query-head groups. The
analytic attention matmul still charges every query head `H`, because each query
head still forms its own scores and output. The `KV/state @T` and
`storage precision` columns account for the cache footprint separately.

## 4. Variant-specific formulas

The base `AttentionVariant.flops()` is the dense formula above. The main overrides
are:

### Dense and ordinary structural variants

```text
F = 4 · B · H · T² · d
```

### Selection variants

`BlockSelectionAttention` uses its advertised read budget:

```text
F = 4 · B · H · T · R · d + F_selector
```

Typical budgets are:

- top-k blocks: `R = min(T, topk_blocks · block_size + sink_size)`;
- top-k tokens: `R = min(T, topk_tokens + sink_size)`;
- local window: `R = min(T, window_size + sink_size)`;
- CSA2: `R = selected_groups + raw_window`.

The budget is an analytic upper-bound convention, not a per-query measured gather
count. The actual measured read count is reported separately in benchmark records.

### Quest

Quest adds the page-bound scoring pass:

```text
F_selector = 2 · B · H · T · ceil(T / page_size) · d
```

### MInference

The code charges the vertical/slash probe and the pooled block search:

```text
F_selector = 4 · B · H · probe · T · d
           + 4 · B · H · (T / block_size)² · d
```

where `probe = min(last_q, T)`.

### FlexPrefill

```text
F_selector = 4 · B · H · (probe + T) · T · d / block_size
```

### MoBA

```text
F_selector = 2 · B · H · T · ceil(T / block_size) · d
```

### MLA

MLA counts the causal dense core and the latent-to-K/V up-projection:

```text
F = 4 · B · H · [T(T+1)/2] · d
  + 4 · B · T · d_latent · d · H
```

The first term is the causal attention core; the second is the up-projection.

### CSA2

CSA2 inherits the selection formula with `R` equal to selected compressed groups
plus the raw window, and adds the indexer:

```text
F_selector = 2 · B · H · T · ceil(T / compress_ratio) · d_indexer
```

The actual quantised footprint is reported in `KV/state @T` with its precision
column; it is not folded into the FLOPs estimate.

### Other overrides

| variant family | formula used by the implementation |
| --- | --- |
| NSAAttention | `4 · B · H · T · min(T, window + selected_blocks · block_size + ceil(T / compress_ratio)) · d` |
| SFAAttention | `4 · B · H · T² · feature_keep` |
| LinformerAttention | `4 · B · H · T · linformer_k · d` |
| EvictionAttention | `4 · B · H · d · [ (T² − max(0, T − C)²)/2 + max(0, T − C) · C ]`, where `C` is the cache budget |

The eviction formula is an analytic transition from the quadratic prefill phase to
the budgeted-cache phase. It is not a per-request simulation of which tokens were
evicted.

## 5. Reproducing a row by hand

Run the model with an explicit shape:

```bash
python -m linattn.cli cost --seq 1024 --heads 8 --dim 64 --variants dense quest
```

The implementation path is:

1. `efficiency_table()` calls `variant.flops(T, batch)`;
2. `AttentionVariant.flops()` or the variant override returns the integer;
3. the table divides by `10^9` and formats one decimal place;
4. `KV reads/query` comes from `read_budget(T, batch)`, a separate quantity.

If a hand calculation differs, check these four conventions first: full square
versus causal triangle, one FLOP versus two FLOPs per multiply-add, `H` query heads
versus `H_kv` KV heads, and matmul work versus total model work.
