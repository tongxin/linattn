# CSA2 official-config check

The official DeepSeek-V4.1-Flash report and the public `1017` model card are
considerably more specific than the repository's original single-layer surrogate.
This page records the mapping and the places where the current analytic model is
still only an approximation of the released system.

As of October 20, 2026, I did not find an official model or checkpoint named
`DeepSeek-V4.1-Flash-1107`. The public release is V4.1-Flash and its `1017`
minor update. The official model card describes `1017` as a minor update and
points to the same technical report.

## 1. Disclosed official architecture

Source: [`DeepSeek_V41_Tech_Report.pdf`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf),
[`config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/main/config.json),
and the [1017 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

| parameter | official value |
| --- | --- |
| Transformer layers | 40 |
| CED split | 20 encoder + 20 decoder |
| Hidden dimension | 5,120 |
| First two layers | pure SWA |
| Encoder CSA2 layers | 18 layers, compression `m=2` |
| Encoder modes | 3 groups of 6: 1 Full + 5 Reuse in each |
| Decoder CSA2 layers | 20 layers, compression `m=1` |
| Decoder modes | first group: 1 Full + 3 Reuse; remaining four groups: 1 Reindex + 3 Reuse |
| Attention heads | 64 |
| Attention head dim | 512 |
| Query compression dim | 1,280 |
| KV heads in `config.json` | 1 |
| Indexer heads / dim | 32 / 128 |
| Sparse attention top-k | 512 KV entries |
| Hierarchical candidate pool | up to 2,048 blocks × 8 positions = 16,384 |
| SWA window | 128 |
| Main KV precision | FP4 E2M1, E4M3 scale per 16 channels, QAT |
| SWA KV precision | FP8 |
| Reported global KV footprint | 890 bytes/token |

The mode schedule is also encoded in `config.json`:

```text
compress_ratios: 0, 0, then 18 × 2, then 20 × 1
kv_source_layer_ids:   [2, 8, 14, 20]
index_source_layer_ids: [2, 8, 14, 20, 24, 28, 32, 36]
candidate_source_layer_id: 20
```

The trailing zero entries in the 43-entry config array correspond to the
additional next-token prediction layers, not the 40 main Transformer blocks.

## 2. Comparison with the repository model

| property | official V4.1-Flash | current `CSA2Attention` default | consequence |
| --- | --- | --- | --- |
| model depth | 40 layers | one layer wrapper | cannot reproduce the reported global footprint directly |
| compression | `m=2` encoder, `m=1` decoder | default `m=16` | different token/equivalent-entry counts |
| heads / dim | 64 / 512 | 4 / 64 | analytic quantities differ by orders of magnitude |
| KV heads | 1 | 2 in CLI defaults | different cache dimensions |
| indexer | 32 heads × 128 | 4 heads × 16 | different indexer work and selection behavior |
| top-k | 512 entries | 4 groups by CLI default | not the same sparse support |
| SWA | 128, FP8 | 256, raw window in the surrogate | different reads and precision |
| cross-layer modes | Full / Reindex / Reuse schedule | one `cross_layer_group` memory knob | no schedule or index reuse modeled in the layer |
| main KV | FP4 QAT | simulated FP4 after pooling | not QAT, and no released checkpoint |
| global footprint | 890 B/token | literal `csa2_reported` preset or generic composition | the preset is a target, not a derivation |

The important point is not that the defaults are “wrong” for their purpose. They
are deliberately small enough to run on a laptop and to make selection masks
auditable. They are simply **not the official model configuration**.

## 3. What the current cost model can and cannot check

### Can check

- The official per-layer dimensions can be substituted into the analytic
  attention formula.
- The official `top-k=512` plus `window=128` implies a per-layer KV-read budget of
  approximately 640 entries per query for a Full/Reindex layer.
- The official compression changes the number of compressed entries available for
  indexing: roughly `T/2` in encoder layers and `T` in decoder layers.
- The official `index_topk=512` and hierarchical candidate pool bound deeper
  indexer work.

### Cannot yet check

- The exact main-KV latent layout and byte packing that produce 890 B/token. The
  report gives the target and the FP4 format, but the public config does not expose
  every field needed to derive that number from first principles.
- The exact Full/Reindex/Reuse layer costs in one analytic call: the current layer
  class has no mode or candidate-pool state.
- QAT behavior and the released checkpoint's actual numerical error.
- CED's prefill half-depth effect on end-to-end FLOPs.

The existing `csa2_reported` preset is therefore deliberately a **literal target**,
not a composed prediction. The generic `csa2` preset uses an older 61-layer,
576-value latent shape and 4-layer sharing; it should not be presented as the
official V4.1-Flash configuration.

## 4. Minimal official-like analytic check

For a single sparse CSA2 layer with the official attention dimensions and
`topk=512`, `window=128`, the approximate core prefill work is

```text
R = 512 + 128 = 640 KV entries/query
F_core = 4 · B · 64 · T · 640 · 512
```

before indexer, projection, CED, and kernel costs. That is a useful hand-check,
but multiplying it by 40 is not the official model cost: CED computes only the
first half of the layers during prefill, and the Full/Reindex/Reuse modes differ
in indexer work.

The repository's official-config checker is:

```bash
python experiments/csa2_official_check.py
```

It emits the disclosed values, the current wrapper values, and the per-layer
core estimate. It does not claim to reconstruct the 890 B/token composition.

The current checker gives:

| T | official-dim core FLOPs/layer | wrapper, `m=2` | wrapper, `m=1` |
| ---: | ---: | ---: | ---: |
| 32K | 2.75 TFLOPs | 11.54 TFLOPs | 20.34 TFLOPs |
| 128K | 11.0 TFLOPs | 151.7 TFLOPs | 292.5 TFLOPs |
| 1M | 87.96 TFLOPs | 9.10 PFLOPs | 18.10 PFLOPs |

The gap is not a subtle constant-factor disagreement. The wrapper charges its
single-layer indexer against every compressed group on every call. The official
model has only eight layers that perform indexing in total:

- three encoder Full Mode layers;
- one decoder Full Mode layer;
- four decoder Reindex Mode layers.

The other 30 CSA2 layers are Reuse Mode and perform no indexer evaluation. The
Reindex layers also search the 16,384-position candidate pool rather than the
full context. Therefore the table's `wrapper` rows are useful as an upper-bound
sanity check for the naive cost, but not as the official model cost.

## 5. Bottom line

**The official CSA2 configuration is substantially disclosed, but the repository's
analytic model is not yet a faithful multi-layer reproduction of it.**

The current model is suitable for controlled algorithmic comparisons and for
showing where projection assumptions change the footprint. For an official
quality or cost claim, use the released checkpoint/config and model the 40-layer
Full/Reindex/Reuse schedule explicitly.
