#!/bin/bash
# Build the Zenodo artifact repository: code at HEAD plus the data archives.
set -euo pipefail
DEST=${1:-../qfm-agent-artifact}
mkdir -p "$DEST/data"
git archive HEAD code tasks tests scripts configs data docs Makefile requirements.txt \
  requirements-lock.txt requirements-extras.txt requirements-router-lock.txt LICENSE REPRODUCIBILITY.md CITATION.cff \
  job_2x2_grid.sh job_classical.sh | tar -x -C "$DEST"
rm -rf "$DEST/scripts/deploy"
cp artifact/README.md artifact/.zenodo.json "$DEST/"
for part in grid_q1 grid_2x2 post_freeze classical_control; do
  [ -d "results/$part" ] && tar --exclude='ollama.log' --exclude='inputs' -cJf "$DEST/data/$part.tar.xz" -C results "$part"
done
tar -cJf "$DEST/data/substrate_and_router.tar.xz" -C results \
  sweep_d4_b1900 router_battery router_baselines router_probe_laya router_probe_julia
(cd "$DEST/data" && sha256sum ./*.tar.xz > SHA256SUMS && ls -la)
