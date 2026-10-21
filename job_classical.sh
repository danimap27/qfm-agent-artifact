#!/bin/bash
#SBATCH --job-name=qfmcls
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-29%6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Classical control: the four evidence x contract cells over logs built from
# the classical heads, template t1, three verbalizers x ten seeds.
set -euo pipefail
cd "$HOME/qfm-2x2"
source "$HOME/venvs/qfm2x2/bin/activate"

IDX=$SLURM_ARRAY_TASK_ID
MODELS=(qwen35 llama31_8b gemma3_27b)
TAGS=(qwen3.5:latest llama3.1:8b gemma3:27b)
SEEDS=(seed0 seed7 seed13 seed21 seed34 seed42 seed55 seed123 seed456 seed789)
MODEL=${MODELS[$((IDX / 10))]}
TAG=${TAGS[$((IDX / 10))]}
SEED=${SEEDS[$((IDX % 10))]}

OUT="classical_control/$MODEL/t1_table/$SEED"
mkdir -p "$OUT"
# No existence guard: run_agent resumes from the first missing pair.
cp "classical_inputs/$SEED/requests.json" "classical_inputs/$SEED/logs.jsonl" "$OUT/"
echo "[$(date)] node=$(hostname) cell=$MODEL/t1_table/$SEED model=$TAG"

PORT=$((11500 + IDX))
export PATH="$HOME/opt/ollama/bin:$PATH"
export OLLAMA_MODELS="$HOME/ollama-models"
export OLLAMA_HOST=127.0.0.1:$PORT
ollama serve > "$OUT/ollama.log" 2>&1 &
OLLAMA_PID=$!
for i in $(seq 1 30); do
  curl -sf "http://127.0.0.1:$PORT/api/tags" >/dev/null && break
  sleep 2
done

python -m code.pipeline.run_agent \
  --requests "$OUT/requests.json" \
  --logs "$OUT/logs.jsonl" \
  --out "$OUT" \
  --conditions grounded,free,evidence_only,cite_only \
  --template t1 \
  --llm-model "$TAG" \
  --llm-endpoint "http://127.0.0.1:$PORT/api/chat"

echo "[$(date)] done $MODEL/t1_table/$SEED records=$(wc -l < "$OUT/explanations.jsonl")"
kill "$OLLAMA_PID" || true
