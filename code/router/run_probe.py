"""Run the Laya router probe over labeled request examples.

The probe asks Laya to route each request in the probe file to one of the
QFM-lite tasks and reports accuracy, per-request latency, and the agreement
of the four request variants of each task. It writes one record per request
to ``routing_decisions.json`` so the classical baselines and the agent
pipeline reuse the exact same decisions. It runs with the router virtual
environment, which carries the Laya dependencies.

Usage: .venv-router/bin/python -m code.router.run_probe --probe data/router_probe.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from code.router.schema import build_task_question, parse_decision


def variant_agreement(records: list[dict]) -> float:
    """Mean share of the most frequent predicted task within each gold task."""
    per_task: dict[str, list[str]] = {}
    for record in records:
        per_task.setdefault(record["gold"], []).append(record["pred"])
    shares = []
    for predictions in per_task.values():
        counts: dict[str, int] = {}
        for prediction in predictions:
            counts[prediction] = counts.get(prediction, 0) + 1
        shares.append(max(counts.values()) / len(predictions))
    return sum(shares) / len(shares)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="data/router_probe.json")
    parser.add_argument("--out", default="results/router_probe_laya")
    parser.add_argument("--model", default="convaiinnovations/laya")
    args = parser.parse_args()

    probe = json.loads(Path(args.probe).read_text())
    requests = probe["requests"]

    import laya

    started = time.time()
    agent = laya.load(args.model)
    print(f"model loaded in {time.time() - started:.1f}s")

    questions = build_task_question()
    records: list[dict] = []
    latencies: list[float] = []
    for item in requests:
        t0 = time.time()
        result = agent.predict(item["text"], questions)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)
        decision = parse_decision(result)
        hit = decision["task"] == item["task"]
        record = {
            "id": item["id"],
            "gold": item["task"],
            "pred": decision["task"],
            "probabilities": decision["probabilities"],
            "confidence": decision["confidence"],
            "latency_ms": round(latency_ms, 1),
            "correct": hit,
        }
        records.append(record)
        print(
            f"{item['id']:<20} gold={item['task']:<18} pred={decision['task']:<18} "
            f"conf={decision['confidence']:.2f} {latency_ms:7.0f} ms {'ok' if hit else 'MISS'}"
        )

    correct = sum(record["correct"] for record in records)
    accuracy = correct / len(records)
    agreement = variant_agreement(records)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "n": len(records),
        "accuracy": round(accuracy, 4),
        "variant_agreement": round(agreement, 4),
        "latency_ms": {
            "median": round(statistics.median(latencies), 1),
            "mean": round(statistics.mean(latencies), 1),
            "max": round(max(latencies), 1),
        },
        "decisions": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    decisions = {
        record["id"]: {
            "task": record["pred"],
            "confidence": record["confidence"],
            "probabilities": record["probabilities"],
        }
        for record in records
    }
    (out_dir / "routing_decisions.json").write_text(json.dumps(decisions, indent=2))
    print(f"\naccuracy {accuracy:.3f} over {len(records)} requests")
    print(f"variant agreement {agreement:.3f}, latency median {statistics.median(latencies):.0f} ms")
    print(f"wrote {out_dir / 'summary.json'} and {out_dir / 'routing_decisions.json'}")


if __name__ == "__main__":
    main()
