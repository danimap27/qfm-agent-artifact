"""Write the decision logs for a request set without running any language model.

Log construction needs the substrate checkpoint and the dataset registry
(Qiskit and scikit-learn), so it stays on the local machine. The resulting
logs.jsonl travels with the requests to the remote runner, which then only
needs ``code.pipeline.run_agent --logs``.

Usage: .venv/bin/python -m code.pipeline.build_logs --requests results/grid_q1/seed0/requests.json --out results/grid_q1/seed0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from code.pipeline.logs import build_logs, write_logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", default="results/pilot/requests.json")
    parser.add_argument("--checkpoint", default="results/sweep_d4_b1900/model.npz")
    parser.add_argument("--out", default="results/pilot")
    args = parser.parse_args()

    request_file = json.loads(Path(args.requests).read_text())
    requests = request_file["requests"]
    started = time.time()
    logs = build_logs(requests, request_file.get("split_seed", 0), args.checkpoint)
    path = write_logs(logs, Path(args.out) / "logs.jsonl")
    print(f"wrote {path} with {len(logs)} logs in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
