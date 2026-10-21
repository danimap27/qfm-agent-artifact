#!/usr/bin/env python3
"""Collect, score and pool the Q1 grid cells (protocol sections 4 and 9-D).

For every cell collected from the two nodes (results/f4_collect/<node>/<cell>/)
this script:

1. Assembles the protocol cell layout under results/grid_q1/<model>/<template>/
   seed<K>/ by joining the per-seed payloads (requests.json, logs.jsonl) with
   the cell outputs (explanations.jsonl, llm_cache.jsonl).
2. Scores the cell with code.eval.score (fidelity harness against the frozen
   checkpoint) unless metrics.json already exists.
3. Pools every per-record row into analysis/pooled_metrics.json (the exact
   input format of code/stats/grid_stats.py): each row keeps its seed cluster
   key and a seed-prefixed request id, mirroring code/stats/pool.py.

Deterministic: scoring re-executes against the frozen checkpoint and the pool
is a pure concatenation. Safe to re-run: completed steps are skipped.

Usage: .venv/bin/python scripts/collect_grid_q1.py [--collect-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_SLUGS = {
    "qwen3_5_latest": "qwen35",
    "llama3_1_8b": "llama31_8b",
    "gemma3_27b": "gemma3_27b",
}
TEMPLATE_SLUGS = {
    "t1": "t1_table",
    "t2": "t2_narrative",
    "t3": "t3_checklist",
}


def parse_cell_name(name: str) -> tuple[str, str, str]:
    """gemma3_27b_t1_seed0 -> (gemma3_27b, t1, 0)."""
    model, template, seed_tag = name.rsplit("_", 2)
    if not seed_tag.startswith("seed"):
        raise ValueError(f"unexpected cell name: {name}")
    if model not in MODEL_SLUGS or template not in TEMPLATE_SLUGS:
        raise ValueError(f"unknown model/template in cell name: {name}")
    return model, template, seed_tag[len("seed"):]


def cell_dests(collect_dir: Path, out_root: Path):
    cells = []
    for node_dir in sorted(p for p in collect_dir.iterdir() if p.is_dir()):
        for cell_dir in sorted(p for p in node_dir.iterdir() if p.is_dir()):
            if not (cell_dir / "explanations.jsonl").exists():
                continue
            if not (cell_dir / "COMPLETO").exists():
                continue
            model, template, seed = parse_cell_name(cell_dir.name)
            dest = out_root / MODEL_SLUGS[model] / TEMPLATE_SLUGS[template] / f"seed{seed}"
            cells.append((node_dir.name, cell_dir, model, template, seed, dest))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-dir", default="results/f4_collect")
    parser.add_argument("--payloads-dir", default="results/grid_q1")
    parser.add_argument("--out-root", default="results/grid_q1")
    parser.add_argument("--checkpoint", default="results/sweep_d4_b1900/model.npz")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args()

    collect_dir = Path(args.collect_dir)
    payloads_dir = Path(args.payloads_dir)
    out_root = Path(args.out_root)

    cells = cell_dests(collect_dir, out_root)
    if not cells:
        raise SystemExit(f"no complete cells under {collect_dir}")
    print(f"cells: {len(cells)} (from {len({node for node, *_ in cells})} nodes)")

    for node, cell_dir, model, template, seed, dest in cells:
        dest.mkdir(parents=True, exist_ok=True)
        payload = payloads_dir / f"seed{seed}"
        for src, name in (
            (payload / "requests.json", "requests.json"),
            (payload / "logs.jsonl", "logs.jsonl"),
            (cell_dir / "explanations.jsonl", "explanations.jsonl"),
            (cell_dir / "llm_cache.jsonl", "llm_cache.jsonl"),
        ):
            target = dest / name
            if src.exists() and not target.exists():
                shutil.copy2(src, target)
        metrics = dest / "metrics.json"
        if not metrics.exists() and not args.skip_score:
            print(f"  score {dest.relative_to(out_root)} ...", flush=True)
            subprocess.run(
                [
                    args.python,
                    "-m",
                    "code.eval.score",
                    "--run-dir",
                    str(dest),
                    "--checkpoint",
                    args.checkpoint,
                    "--out",
                    str(metrics),
                ],
                check=True,
            )

    # Pool: mirror code/stats/pool.py with seed-prefixed request ids.
    merged = []
    seeds = []
    for _node, _cell_dir, _model, _template, seed, dest in cells:
        metrics = dest / "metrics.json"
        if not metrics.exists():
            continue
        if seed not in seeds:
            seeds.append(seed)
        payload = json.loads(metrics.read_text())
        for row in payload["per_record"]:
            merged.append({**row, "seed": seed, "request_id": f"s{seed}-{row['request_id']}"})

    analysis_dir = out_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = analysis_dir / "pooled_metrics.json"
    out_path.write_text(json.dumps({"per_record": merged, "seeds": seeds}, indent=2))
    print(f"pooled {len(merged)} records from {len(seeds)} seeds into {out_path}")


if __name__ == "__main__":
    sys.exit(main())
