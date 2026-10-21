#!/bin/bash
#SBATCH --job-name=qfm2x2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-89%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd "$HOME/qfm-2x2"

# python 3.12 venv managed by uv (shared, created in the smoke run)
if [ ! -x "$HOME/venvs/qfm2x2/bin/python" ]; then
  export PATH="$HOME/.local/bin:$PATH"
  uv venv --python 3.12 "$HOME/venvs/qfm2x2"
  source "$HOME/venvs/qfm2x2/bin/activate"
  uv pip install -q numpy scikit-learn requests
else
  source "$HOME/venvs/qfm2x2/bin/activate"
fi

IDX=$SLURM_ARRAY_TASK_ID
MODELS=(qwen35 llama31_8b gemma3_27b)
TAGS=(qwen3.5:latest llama3.1:8b gemma3:27b)
TPLS=(t1_table t2_narrative t3_checklist)
TCODE=(t1 t2 t3)
MODEL=${MODELS[$((IDX / 30))]}
TAG=${TAGS[$((IDX / 30))]}
TPL=${TPLS[$(((IDX % 30) / 10))]}
T=${TCODE[$(((IDX % 30) / 10))]}
SEEDS=(seed0 seed7 seed13 seed21 seed34 seed42 seed55 seed123 seed456 seed789)
SEED=${SEEDS[$((IDX % 10))]}

CELL_IN="grid_inputs/grid_q1/$MODEL/$TPL/$SEED"
OUT="grid_2x2/$MODEL/$TPL/$SEED"
mkdir -p "$OUT"
# No existence guard: run_agent resumes from the first missing pair, so a
# partial cell is completed instead of skipped as done.
cp "$CELL_IN/requests.json" "$CELL_IN/logs.jsonl" "$OUT/"
echo "[$(date)] node=$(hostname) cell=$MODEL/$TPL/$SEED model=$TAG"

export PATH="$HOME/opt/ollama/bin:$PATH"
export OLLAMA_MODELS="$HOME/ollama-models"
export OLLAMA_HOST=127.0.0.1:11434
ollama serve > "$OUT/ollama.log" 2>&1 &
OLLAMA_PID=$!
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break
  sleep 2
done

python -m code.pipeline.run_agent \
  --requests "$OUT/requests.json" \
  --logs "$OUT/logs.jsonl" \
  --out "$OUT" \
  --conditions evidence_only,cite_only \
  --template "$T" \
  --llm-model "$TAG" \
  --llm-endpoint http://127.0.0.1:11434/api/chat

echo "[$(date)] done $MODEL/$TPL/$SEED"
kill "$OLLAMA_PID" || true
