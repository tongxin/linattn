# 2025–2026 frontier variants at T=512 and a 128-entry budget

*Generated 2026-10-18T09:16:03Z by `reporting.py` on Darwin arm64 (torch 2.14.1, cuda=False, triton=False).*

One CPU run on the repository's structured input, not a reproduction of the authors' GPU benchmarks. `masa` is an exact-output-error oracle and is intentionally expensive. `oval` builds page bases in Python on every call; its latency is not representative of a fused kernel. The eviction variants receive the same protected prompt/window policy as the random baseline. Read this table as a sanity check on mechanism and cost accounting, not as evidence that one paper beats another.

| variant | family | rel_err | mass_recall | reads_per_query | state_kib | latency_ms | error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense | dense | 0 | None | 512 | 512 | 4.743 |  |
| saga | frontier | 0.8406 | 1 | 113.9 | 384 | 10.67 |  |
| pisa | frontier | 0.03756 | 1 | 113.9 | 512 | 88.9 |  |
| jssa | frontier | 0.1233 | 1 | 113.9 | 512 | 17.4 |  |
| masa | frontier | 0.181 | 0.998 | 112.2 | 512 | 15.53 |  |
| oval | frontier | 0.1078 | 0.9848 | 111.1 | 512 | 28.33 |  |
| sfa | frontier | 0.9083 | None | 320 | 328 | 6.426 |  |
| random_eviction | baseline | 0.6754 | None | 224.5 | 512 | 2.913 |  |
| h2o | eviction | 0.6536 | None | 224.5 | 512 | 2.702 |  |
| snapkv | eviction | 0.576 | None | 224 | 512 | 2.686 |  |
| tova | eviction | 0.6533 | None | 224.5 | 512 | 2.641 |  |
| dropkv | eviction | 0.6754 | None | 224.5 | 512 | 3.214 |  |
| restkv | eviction | 0.6754 | None | 224.5 | 512 | 3.108 |  |
| beaconkv | eviction | 0.6534 | None | 224.5 | 512 | 3.125 |  |
