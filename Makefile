# Everything runs on CPU. `make install` sets the venv up, `make check` is the
# battery that must pass before a commit, `make all` reproduces every number in
# reports/ (a few minutes on a laptop).  Override the interpreter with
# `make PYTHON=python3 <target>` if you prefer to install elsewhere.
PY ?= 3.13
VENV_PY := .venv/bin/python
# use the venv when it exists, otherwise whatever `python3` is on PATH
PYTHON ?= $(shell test -x $(VENV_PY) && echo $(VENV_PY) || echo python3)

export PYTHONPATH := src

.PHONY: help venv install check docs bench bench-frontier csa2-analysis csa2-official-check approx-metrics approx-validation theory-checks error-propagation trained-audit bench-mlx kv experiments all clean

help:  ## list targets
	@grep -hE '^[a-z]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /: /' | column -t -s ':'

venv:  ## ensure .venv exists (idempotent, never prompts)
	@if [ -x $(VENV_PY) ]; then \
		printf 'reusing existing venv: %s (%s)\n' "$(CURDIR)/.venv" "$$($(VENV_PY) -V 2>&1)"; \
	else \
		printf 'creating .venv with Python %s\n' "$(PY)"; \
		uv venv --python $(PY) --allow-existing .venv; \
	fi

install: venv  ## editable install of the package + pytest (safe to re-run)
	uv pip install --python $(VENV_PY) -e ".[dev]"
	@echo
	@$(VENV_PY) -c "import linattn, torch, pytest; print('verified:', 'linattn', linattn.__version__, '| torch', torch.__version__, '| pytest', pytest.__version__)"
	@echo "ready.  try:  make check   |   $(VENV_PY) -m linattn.cli backends"

check:  ## validation suite + documentation consistency (fast)
	$(PYTHON) -m pytest tests -q

docs:  ## regenerate per-variant documentation from the code
	$(PYTHON) scripts/gen_variant_docs.py

bench:  ## quality/cost sweep and the decode-IO benchmark -> reports/
	$(PYTHON) bench/run_suite.py --seq 2048 --budget 256 --topk-blocks 8 --topk-tokens 512
	$(PYTHON) bench/bench_decode.py --seq 65536 --budgets 256 1024 2048 8192

bench-frontier:  ## focused 2025–2026 variant comparison -> reports/frontier_2026.*
	$(PYTHON) bench/run_frontier_2026.py

csa2-analysis:  ## CSA2 pooling/selection/precision error decomposition
	$(PYTHON) experiments/csa2_approximation.py --seq 512

csa2-official-check:  ## compare the repo surrogate with published V4.1-Flash config
	$(PYTHON) experiments/csa2_official_check.py

approx-metrics:  ## run the attention-approximation metric prototype
	$(PYTHON) experiments/approximation_metrics.py

approx-validation:  ## compare approximation metrics across representative variants
	$(PYTHON) experiments/approximation_validation.py

theory-checks:  ## validate sparse bounds, feature convergence and precision
	$(PYTHON) experiments/theory_checks.py

error-propagation:  ## local attention error vs residual-stack drift
	$(PYTHON) experiments/error_propagation.py

trained-audit:  ## matched tiny-checkpoint local-to-task audit
	$(PYTHON) experiments/approximation_audit.py --steps 2200 --seeds 3 --eval-examples 256

bench-mlx:  ## Apple Silicon: MLX/Metal decode benchmark (needs `make install` + mlx)
	$(PYTHON) bench/bench_mlx.py --lengths 4096 16384 65536

kv:  ## KV layer: economics, cache policies and the CSA2 stack model -> reports/
	$(PYTHON) experiments/kv_economics.py
	$(PYTHON) experiments/csa2_model.py

experiments:  ## the three studies behind the survey's empirical claims
	$(PYTHON) experiments/selector_capacity.py
	$(PYTHON) experiments/state_capacity.py
	$(PYTHON) experiments/hybrid_recall.py --steps 250

all: docs check bench bench-frontier csa2-analysis csa2-official-check approx-validation theory-checks error-propagation trained-audit kv experiments  ## everything above, in order

clean:  ## remove generated reports and caches (keeps .venv)
	rm -f reports/suite_*.json reports/suite_*.md reports/decode_*.json reports/decode_*.md
	rm -f reports/hybrid_recall.* reports/selector_capacity.* reports/state_capacity.*
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__
