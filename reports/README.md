# reports

Generated measurement output. Each report exists twice:

* `<name>.json` -- records plus the environment (platform, torch version, CUDA and
  Triton availability) that produced them;
* `<name>.md` -- the same numbers as a markdown table, with the environment in the
  header.

Nothing here is hand-edited. To regenerate:

```bash
python bench/run_suite.py --seq 2048 --budget 256
python bench/bench_decode.py --seq 65536 --budgets 256 1024 2048 8192
python experiments/hybrid_recall.py --steps 300
python experiments/selector_capacity.py
python experiments/state_capacity.py
make kv            # or: python experiments/kv_economics.py
```

Reports currently checked in:

| file | what it shows |
| --- | --- |
| `suite_<T>_<budget>.md` | every variant at one common budget: relative error vs dense, KV entries read per query, persistent state size, latency |
| `decode_<T>_<dim>.md` | decode time vs bytes read, C kernel (naive and IO-aware tiled) against torch dense and torch-over-the-same-selection |
| `hybrid_recall.md` | MQAR accuracy vs persistent state size for dense / hybrid / linear-only stacks |
| `kv_economics.md` | KV formats x context x hit rate, algorithm leverage (dense/sparse/recurrent), and cache-policy comparison on an agentic trace |
| `csa2_model.md` | CSA2 end to end: footprint (reported vs composed vs reach), sharing divergence vs layer disagreement, and the unchanged attention term |
| `state_capacity.md` | exact associative-recall fidelity of a fixed-size delta-rule state vs. number of associations |
| `selector_capacity.md` | learned-selector cross-entropy and attention-mass recall before and after fitting |

The numbers are CPU numbers on a laptop, and they are meant to be read as
*relative* comparisons (kernel structure, budget honesty, capacity of a selector),
not as absolute performance claims for any accelerator.
