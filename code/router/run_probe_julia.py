"""Run the Julia-1 decision-model probe over labeled request examples.

This mirrors ``code.router.run_probe`` (Laya) so both decision-layer models
can be compared on exactly the same requests and metrics. It runs with the
``.venv-julia`` environment, which carries the official SupersonicLabs/Julia-1
runtime and a CPU PyTorch build.

Usage: .venv-julia/bin/python -m code.router.run_probe_julia --probe data/router_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from code.router.schema import build_task_question


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
    parser.add_argument("--out", default="results/router_probe_julia")
    parser.add_argument("--model", default=os.environ.get("JULIA_MODEL_DIR", "models/julia-1"))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--head-length", type=int, default=512)
    args = parser.parse_args()

    probe = json.loads(Path(args.probe).read_text())
    requests = probe["requests"]

    from julia import load_model

    started = time.time()
    engine = load_model(
        args.model,
        device="cpu",
        strict_encoding=True,
        max_length=args.max_length,
        head_length=args.head_length,
    )
    print(f"model loaded in {time.time() - started:.1f}s")

    questions = build_task_question()
    records: list[dict] = []
    latencies: list[float] = []
    for item in requests:
        t0 = time.time()
        result = engine.predict(state=item["text"], questions=questions)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)
        answer = result["answers"]["task"]
        probabilities = {
            key: round(float(value), 4) for key, value in answer.get("probabilities", {}).items()
        }
        confidence = answer.get("max_probability")
        if confidence is None:
            confidence = max(probabilities.values()) if probabilities else 0.0
        pred = answer["choice"]
        hit = pred == item["task"]
        records.append(
            {
                "id": item["id"],
                "gold": item["task"],
                "pred": pred,
                "probabilities": probabilities,
                "confidence": round(float(confidence), 4),
                "latency_ms": round(latency_ms, 1),
                "correct": hit,
            }
        )
        print(
            f"{item['id']:<20} gold={item['task']:<18} pred={pred:<18} "
            f"conf={float(confidence):.2f} {latency_ms:7.0f} ms {'ok' if hit else 'MISS'}"
        )

    correct = sum(record["correct"] for record in records)
    accuracy = correct / len(records)
    agreement = variant_agreement(records)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "SupersonicLabs/Julia-1 (cpu, fp32)",
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
