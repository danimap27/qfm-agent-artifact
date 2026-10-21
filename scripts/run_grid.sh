#!/usr/bin/env bash
# F4 grid runner: request seeds, full pipeline, then per-seed and pooled analysis.
# Usage: scripts/run_grid.sh 2 3 4   (request seeds; seed 0 is the deterministic pilot)
set -euo pipefail
cd "$(dirname "$0")/.."

for seed in "$@"; do
  out="results/grid/seed${seed}"
  echo "== seed ${seed} =="
  .venv/bin/python -m code.pipeline.build_requests --out "${out}/requests.json" --sample-seed "${seed}"
  .venv/bin/python -m code.pipeline.run_agent --requests "${out}/requests.json" --out "${out}" --llm-model "${LLM_MODEL:-qwen3.5:4b}" --llm-endpoint "${LLM_ENDPOINT:-http://localhost:11434/api/chat}"
  .venv/bin/python -m code.eval.score --run-dir "${out}" --out "${out}/metrics.json"
  .venv/bin/python -m code.stats.compare --metrics "${out}/metrics.json" --out "${out}/stats.json" --tex "${out}/tab_pilot.tex"
done

.venv/bin/python -m code.stats.pool --root results/grid
.venv/bin/python -m code.stats.compare --metrics results/grid/pooled_metrics.json --out results/grid/pooled_stats.json --tex results/grid/tab_pooled.tex
echo "grid complete"
