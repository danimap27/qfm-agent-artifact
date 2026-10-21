PY := .venv/bin/python
ROUTER_PY := .venv-router/bin/python
JULIA_PY := .venv-julia/bin/python
FIG_PY := .venv-figs/bin/python

.PHONY: help install test lint smoke-f1 transfer-f1 router-probe router-probe-julia router-baselines pilot-requests pilot-run pilot-score pilot-stats grid-prep grid-score grid-stats figs-venv figures manuscript-lint clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Create the venv and install dependencies
	uv venv .venv --python 3.12
	uv pip install --python $(PY) -r requirements.txt

test: ## Run the fast test suite
	$(PY) -m pytest tests/ -q

lint: ## Byte-compile the package and tests
	$(PY) -m compileall -q code tests scripts

smoke-f1: ## Run the F1 substrate smoke pretraining
	$(PY) -m code.substrate.train --config configs/f1_smoke.yaml --out results/f1_smoke

transfer-f1: ## Run the held-out transfer protocol (needs the smoke model)
	$(PY) -m code.substrate.transfer --model results/f1_smoke/model.npz --tasks ionosphere,blood_transfusion --out results/f1_transfer

router-probe: ## Run the Laya router probe over the labeled requests
	$(ROUTER_PY) -m code.router.run_probe --probe data/router_probe.json --out results/router_probe_laya

router-probe-build: ## Rebuild the expanded (v3) routing probe from v2
	$(PY) -m code.router.build_probe --out data/router_probe.json

router-stats: ## Decision-layer battery (McNemar + Holm, Wilson, ECE, latency)
	$(PY) -m code.router.router_stats

router-probe-julia: ## Run the Julia-1 router probe over the labeled requests
	$(JULIA_PY) -m code.router.run_probe_julia --probe data/router_probe.json --out results/router_probe_julia

router-baselines: ## Run the classical routing baselines on the probe hold-out
	$(PY) -m code.router.baselines --probe data/router_probe.json --laya results/router_probe_laya/routing_decisions.json --out results/router_baselines

pilot-requests: ## Build the pilot request set from the frozen test splits
	$(PY) -m code.pipeline.build_requests --out results/pilot/requests.json

pilot-run: ## Run the agent pipeline over the pilot requests
	$(PY) -m code.pipeline.run_agent --requests results/pilot/requests.json --out results/pilot

pilot-score: ## Score the pilot explanations against the decision logs
	$(PY) -m code.eval.score --run-dir results/pilot --checkpoint results/sweep_d4_b1900/model.npz

pilot-stats: ## Paired statistics over the pilot metrics
	$(PY) -m code.stats.compare --metrics results/pilot/metrics.json --out results/pilot/stats.json

grid-prep: ## Build the Q1 grid payloads (requests + logs per seed)
	bash scripts/prep_grid_q1.sh

grid-score: ## Assemble, score and pool all collected Q1 cells
	$(PY) scripts/collect_grid_q1.py

grid-stats: ## Grid statistical battery (families, Friedman, effects, pairwise)
	$(PY) -m code.stats.grid_stats

figs-venv: ## Create the figures virtual environment (numpy + matplotlib)
	uv venv .venv-figs --python 3.12
	uv pip install --python $(FIG_PY) matplotlib numpy

figures: ## Render the Q1 grid figures (needs figs-venv)
	$(FIG_PY) -m code.figures.make_figures

manuscript-lint: ## Lint the manuscript against the writing rules
	$(PY) scripts/lint_manuscript.py paper/main.tex

clean: ## Remove caches
	rm -rf .pytest_cache __pycache__ code/__pycache__ code/substrate/__pycache__ code/router/__pycache__ tasks/__pycache__ scripts/__pycache__
