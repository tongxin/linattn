# Convenience targets. `make check` is the battery that must pass before a commit.
PYTHON ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: check docs bench experiments all

check:  ## tests + documentation consistency
	$(PYTHON) -m pytest tests -q

docs:  ## regenerate per-variant documentation from the code
	$(PYTHON) scripts/gen_variant_docs.py

bench:  ## quality/cost sweep and the decode-IO benchmark -> reports/
	$(PYTHON) bench/run_suite.py --seq 2048 --budget 256 --topk-blocks 8 --topk-tokens 512
	$(PYTHON) bench/bench_decode.py --seq 65536 --budgets 256 1024 2048 8192

experiments:  ## the three studies behind the survey's empirical claims
	$(PYTHON) experiments/selector_capacity.py
	$(PYTHON) experiments/state_capacity.py
	$(PYTHON) experiments/hybrid_recall.py --steps 250

all: docs check bench experiments
