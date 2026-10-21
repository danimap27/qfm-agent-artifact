#!/usr/bin/env bash
# Prepare the Q1 grid data: requests and prebuilt decision logs per seed.
# Usage: scripts/prep_grid_q1.sh   (all seeds overridable via SEEDS/PER_TASK/ROOT)
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 42 123 456 789 7 13 21 34 55}"
PER_TASK="${PER_TASK:-17}"
ROOT="${ROOT:-results/grid_q1}"

for seed in $SEEDS; do
  out="${ROOT}/seed${seed}"
  echo "== seed ${seed} =="
  .venv/bin/python -m code.pipeline.build_requests --out "${out}/requests.json" --sample-seed "${seed}" --per-task "${PER_TASK}"
  .venv/bin/python -m code.pipeline.build_logs --requests "${out}/requests.json" --out "${out}"
done

echo "grid q1 data ready under ${ROOT}"
