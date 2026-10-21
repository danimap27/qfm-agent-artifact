"""Pool per-seed metrics into one table for the grid analysis.

Request identifiers repeat across seeds, so each pooled row is keyed by seed
and request to keep the paired comparisons of the statistics stage correct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/grid")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    merged = []
    seeds = []
    for seed_dir in sorted(root.glob("seed*")):
        metrics_path = seed_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        seed = seed_dir.name.removeprefix("seed")
        seeds.append(seed)
        payload = json.loads(metrics_path.read_text())
        for row in payload["per_record"]:
            merged.append({**row, "seed": seed, "request_id": f"s{seed}-{row['request_id']}"})

    out_path = Path(args.out) if args.out else root / "pooled_metrics.json"
    out_path.write_text(json.dumps({"per_record": merged, "seeds": seeds}, indent=2))
    print(f"pooled {len(merged)} records from {len(seeds)} seeds into {out_path}")


if __name__ == "__main__":
    main()
