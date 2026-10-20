# CSA2 official V4.1-Flash configuration vs the repository surrogate

*Generated 2026-10-20T15:37:16Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

The 1107 tag was not found in the official public metadata as of 2026-10-20; the public release is V4.1-Flash and its 1017 minor update. Official values come from the technical report and public config.json. The wrapper is a single-layer surrogate: it does not model the Full/Reindex/Reuse schedule, CED half-depth prefill, FP8 SWA, or the exact 890 B/token packing. The 890 preset is a target, not a derivation. The official schedule performs indexing in only 8 of 38 CSA2 layers (3 encoder Full, 1 decoder Full, 4 decoder Reindex); the other 30 are Reuse and run no indexer. The wrapper charges an indexer on every call and searches the full compressed prefix, so its cost is a loose upper bound rather than the official model cost.

| check | item | value | unit | note |
| --- | --- | --- | --- | --- |
| official | official layers | 40 | layers | 20 encoder + 20 decoder CED |
| official | official attention heads | 64 | heads | query heads |
| official | official head dim | 512 | dims | main attention |
| official | official indexer | 32 x 128 | heads x dim | indexer Q/K width |
| official | official sparse top-k | 512 | entries/query | main KV entries |
| official | official SWA window | 128 | tokens | FP8 SWA |
| official | official main KV precision | FP4 E2M1 + E4M3 scale/16 | format | QAT |
| official | official reported global KV | 890 | B/token | whole model target |
| wrapper_default | compression ratio | 16 | m | official is m=2 encoder / m=1 decoder |
| wrapper_default | attention heads | 4 | heads | official is 64 |
| wrapper_default | head dim | 64 | dims | official is 512 |
| wrapper_default | KV heads | 4 | heads | official config is 1 |
| wrapper_default | indexer | 4 x 16 | heads x dim | official is 32 x 128 |
| wrapper_default | top-k groups | 4 | groups | official is 512 KV entries |
| wrapper_default | window | 256 | tokens | official is 128; wrapper defaults to raw FP16 window |
| analytic | core FLOPs @T=32768 | 2.749e+12 | FLOPs/layer | 4·H·T·(512+128)·d before indexer/CED/projection |
| wrapper_official_dims | wrapper FLOPs @T=32768, m=2 | 11544872091648 | FLOPs/layer | includes current surrogate indexer estimate |
| wrapper_official_dims | wrapper FLOPs @T=32768, m=1 | 20340965113856 | FLOPs/layer | includes current surrogate indexer estimate |
| analytic | core FLOPs @T=131072 | 1.1e+13 | FLOPs/layer | 4·H·T·(512+128)·d before indexer/CED/projection |
| wrapper_official_dims | wrapper FLOPs @T=131072, m=2 | 151732604633088 | FLOPs/layer | includes current surrogate indexer estimate |
| wrapper_official_dims | wrapper FLOPs @T=131072, m=1 | 292470092988416 | FLOPs/layer | includes current surrogate indexer estimate |
| analytic | core FLOPs @T=1048576 | 8.796e+13 | FLOPs/layer | 4·H·T·(512+128)·d before indexer/CED/projection |
| wrapper_official_dims | wrapper FLOPs @T=1048576, m=2 | 9095160184963072 | FLOPs/layer | includes current surrogate indexer estimate |
| wrapper_official_dims | wrapper FLOPs @T=1048576, m=1 | 18102359439704064 | FLOPs/layer | includes current surrogate indexer estimate |
| kv_preset | csa2_reported @61-layer preset convention | 890 | B/token | DeepSeek-V4.1-Flash reported global KV: 890 B/token across ~61 layers; not a recomputation of the official 40-layer packing |
| kv_preset | csa2 @61-layer preset convention | 5184 | B/token | 576 values/token, fp4, shared over 4 layers; not a recomputation of the official 40-layer packing |
| kv_preset | csa2_reach890 @61-layer preset convention | 972 | B/token | 576 values/token, fp4, shared over 21 layers; not a recomputation of the official 40-layer packing |
