# What Grounding Buys: code and data

Artifact for *What Grounding Buys: Separating Evidence and Citation Contracts in
Language Model Explanations of a Variational Quantum Model* (Martín-Pérez,
Rodríguez-Díaz, Gutiérrez-Avilés, Troncoso, Martínez-Álvarez).

## Contents

| Path | What it holds |
|---|---|
| `code/`, `tasks/`, `scripts/`, `tests/` | Substrate, evidence layer, verbalizer prompts, scorer, statistics, figure generators |
| `data/substrate_and_router.tar.xz` | Frozen eight-qubit checkpoint and the routing battery |
| `data/grid_q1.tar.xz` | Frozen 90-cell grid, 55,080 records, with `analysis/FROZEN.md` and per-file hashes |
| `data/grid_2x2.tar.xz` | Evidence-only and contract-only cells, 18,360 records |
| `data/classical_control.tar.xz` | The four evidence by contract cells over a classical predictor |
| `data/post_freeze.tar.xz` | Analyses run after the freeze, the device-noise audit, and the annotation sheet |
| `data/SHA256SUMS` | Hashes of the archives |

## Reproduce

    tar -xJf data/grid_q1.tar.xz -C results   # likewise for the other archives
    python -m venv .venv && .venv/bin/pip install -r requirements-lock.txt -r requirements-extras.txt
    make test
    python scripts/analyze_2x2.py             # evidence by contract decomposition
    python scripts/value_audit_grid.py        # value audit with its measured chance floor
    python scripts/device_noise_cf_audit.py   # counterfactual audit under device noise

The frozen tree hash in `results/grid_q1/analysis/FROZEN.md` can be checked with
the command printed there. Language model calls replay from the per-cell caches
with `--replay`, so no model needs to be served to reproduce the scores.

## Licenses

Code: MIT (`LICENSE`). Data: CC BY 4.0.
