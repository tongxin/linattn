# Reproducing the numbers

This page is the short version of "how do I get the same output as `reports/`", plus
the things that actually go wrong on a fresh machine.

## 1. Environment

| requirement | why | notes |
| --- | --- | --- |
| Python 3.10 - 3.14 | PyTorch wheels exist for these ABIs | **not 3.15**: no `cp315` wheel exists yet, and the resolver only tells you after downloading everything. `pyproject.toml` now bounds `requires-python` so this fails fast. |
| PyTorch >= 2.4 | everything | CPU-only wheel is fine; the reference implementations, benchmarks and all three studies run without a GPU |
| a C compiler (`clang`/`cc`/`gcc`) | `bench/bench_decode.py` compiles `kernels/c/decode_attn.c` at runtime | optional: without one, that benchmark prints a message and exits; `make check` still passes (the C parity tests skip) |
| `pytest` | `make check` | pulled in by `make install` (`.[dev]`) |
| `triton` | only `pytest -m gpu` | CUDA-only; every Triton kernel has a torch fallback, and the GPU tests skip with a reason when no GPU is present |

```bash
make install                 # uv venv --python 3.13 .venv && uv pip install -e ".[dev]"
make PYTHON=python3 check    # or use an existing environment that has torch
```

`make install` is safe to re-run: it reuses an existing `.venv` (and prints its
Python version) rather than prompting to replace it, and re-installs the package
in place. Delete `.venv` to force a fresh environment, and pass `PY=3.12` to pick
the interpreter for a new one. The target ends by importing `linattn`, `torch` and
`pytest`, so it fails loudly if the environment is not actually usable.

There is no `numpy`, `matplotlib`, `pandas` or `transformers` dependency: nothing in
the repository imports them (the manifest was trimmed to match, after an earlier
version listed them speculatively). Plots are intentionally not produced — the
reports are markdown tables so that a number can always be traced to the JSON next
to it.

## 2. The commands

```bash
make check          # ~5 s   75 validation checks, writes nothing
make bench          # ~10 s  reports/suite_<T>_<budget>.{json,md}, reports/decode_<T>_<dim>.{json,md}
make experiments    # ~3 min reports/{selector_capacity,state_capacity,hybrid_recall}.{json,md}
make docs           # regenerate docs/variants/*.md from the class docstrings and cost models
make all            # docs, check, bench, experiments
make clean          # delete the generated reports and caches (keeps .venv)
```

Individual scripts take their own knobs:

```bash
python bench/run_suite.py --seq 8192 --budget 512 --topk-blocks 16 --topk-tokens 1024
python bench/bench_decode.py --seq 262144 --budgets 256 2048 16384
python experiments/state_capacity.py --counts 4 64 512 --states 8 64 256
python experiments/hybrid_recall.py --steps 400 --seq 256 --d-state 16
python experiments/selector_capacity.py --seq 1024 --steps 400
python -m linattn.cli cost --seq 131072 --block-size 64 --budget 4096
python -m linattn.cli select --variant xattention --seq 4096 --structured
```

Runtime scales with `--seq` quadratically in the reference (mask-based) path, which
is deliberate: `bench/run_suite.py --seq 32768` will take minutes and hundreds of MB
because it materialises explicit boolean masks. That is the price of making the
semantics auditable; the accelerated paths are the kernels in `src/linattn/kernels/`.

## 3. Reading a report

Every report is written twice, and the JSON records the environment that produced it:

```jsonc
{
  "name": "suite_2048_256",
  "generated_utc": "2026-10-16T09:11:24Z",
  "environment": { "platform": "Darwin arm64", "torch": "2.14.1", "cuda": false, ... },
  "records": [ { "variant": "quest", "rel_err": 0.1245, "mass_recall": 0.876, "reads_per_query": 368.3, ... } ]
}
```

Three cautions that matter more than the numbers:

1. **Read pairs, not columns.** A selector is defined by recall-at-a-budget;
   `reads_per_query` alone or `mass_recall` alone can each be made to look good
   trivially.
2. **`rel_err` is not an error to minimise blindly.** For a selection method it
   measures how far the sparse output is from dense, which is large by construction
   when the budget is small — that is the trade-off being reported, not a bug.
3. **Recall saturates on retrieval-shaped inputs.** `bench/run_suite.py` plants
   spikes on a recency gradient, and a *randomly initialised* learned selector
   already recovers most of the mass. That is the same effect the eviction
   literature reports, and the reason `docs/survey.md` argues recall alone cannot
   rank selectors.

## 4. Troubleshooting

| symptom | cause | fix |
| --- | --- | --- |
| `No module named torch` | running `python bench/...` without the venv active and without the package installed | `make install` then `source .venv/bin/activate`, or `make PYTHON=python3 bench` with torch available |
| resolver hangs, then reports no matching ABI | Python 3.15 (no torch wheel) | `make install PY=3.13` |
| `pytest -m gpu` reports 0 tests | no CUDA GPU, or `triton` not installed | expected on a laptop: the Triton kernels are then exercised only through their torch fallbacks |
| `bench_decode.py` prints "no C compiler available" | no `clang`/`cc`/`gcc` on `PATH` | install the platform compiler; the rest of the suite is unaffected |
| `kernels/triton/*` imports fail | `triton` missing and no GPU | import paths are guarded; only the `-m gpu` tests need it |
| `uv` fails with `Failed to initialize cache at ~/.cache/uv` | read-only home directory (sandboxed runner, locked-down CI) | point the cache elsewhere: `UV_CACHE_DIR=/tmp/uvcache make install` |
| slow or stalling downloads | restricted or throttled network | use a mirror: `UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple make install`, or pre-download the wheel and `uv pip install <wheel>` |
| reports look different from `reports/` | different machine, thread count, or torch version | compare the `environment` block in the JSON; the numbers are relative (ratios, recall, budgets), not absolute performance claims |

## 5. Continuous integration

`.github/workflows/ci.yml` runs `make check`, small benchmarks and a `-m gpu` job.
It is not part of the first pushed commit because pushing a workflow file requires
the `workflow` scope on the GitHub token:

```bash
gh auth refresh -h github.com -s workflow
git add .github/workflows/ci.yml && git commit -m "ci: validation + benchmark workflow" && git push
```

## 6. Extending

* **Add a variant**: subclass `linattn.core.AttentionVariant` (implement `attend`,
  `memory_bytes`, `flops`, `read_budget`), register it in `src/linattn/registry.py`,
  run `make docs` to generate its page, and add a budget/quality assertion to
  `tests/test_budgets.py`. For selection variants, implement `select` and reuse
  `BlockSelectionAttention`, which gives you causality enforcement, the empty-selection
  fallback and mask-based recall accounting for free.
* **Add a kernel**: implement it next to the existing ones with a `*_torch` fallback
  and a dispatch function, then extend `tests/test_kernels.py`; the parity contract
  (kernel path on GPU, fallback everywhere) is what keeps the numbers comparable.
