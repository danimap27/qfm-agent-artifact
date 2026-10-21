"""Decision-layer battery statistics (protocol section 11).

Compares the typed router (Laya) and the graph router (Julia 1) against the
classical baselines on the shared evaluation half of the routing probe:
accuracy with Wilson and task-cluster bootstrap intervals, expected
calibration error, median latency, variant agreement across phrasings, and
exact McNemar tests over the pair family with a single Holm correction.

Reads the JSON artifacts written by code.router.baselines,
code.router.run_probe and code.router.run_probe_julia. Writes a consolidated
JSON plus the LaTeX rows used by the paper.

Usage: .venv/bin/python -m code.router.router_stats
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from code.router.baselines import expected_calibration_error, split_probe
from code.stats.compare import holm


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    phat = correct / total
    denominator = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denominator
    half = z * np.sqrt(phat * (1.0 - phat) / total + z * z / (4 * total * total)) / denominator
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant pair counts."""
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)


def cohens_h(first: float, second: float) -> float:
    """Effect size for the difference of two proportions."""
    return float(2.0 * np.arcsin(np.sqrt(first)) - 2.0 * np.arcsin(np.sqrt(second)))


def cluster_bootstrap_accuracy(correct: np.ndarray, tasks: list[str], seed: int = 0, draws: int = 2000) -> tuple[float, float]:
    """Percentile bootstrap of accuracy, resampling task clusters."""
    rng = random.Random(seed)
    clusters: dict[str, list[int]] = {}
    for index, task in enumerate(tasks):
        clusters.setdefault(task, []).append(index)
    groups = list(clusters.values())
    samples = []
    for _ in range(draws):
        chosen: list[int] = []
        for _ in groups:
            chosen.extend(rng.choice(groups))
        samples.append(float(correct[chosen].mean()))
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def variant_agreement(records: list[dict]) -> float:
    """Mean share of the most frequent prediction within each task."""
    by_task: dict[str, list[str]] = {}
    for record in records:
        by_task.setdefault(record["gold"], []).append(record["pred"])
    shares = []
    for predictions in by_task.values():
        counts: dict[str, int] = {}
        for prediction in predictions:
            counts[prediction] = counts.get(prediction, 0) + 1
        shares.append(max(counts.values()) / len(predictions))
    return float(np.mean(shares))


def load_decisions(path: Path) -> dict[str, dict]:
    summary = json.loads(path.read_text())
    return {record["id"]: record for record in summary["decisions"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="data/router_probe.json")
    parser.add_argument("--laya", default="results/router_probe_laya/summary.json")
    parser.add_argument("--julia", default="results/router_probe_julia/summary.json")
    parser.add_argument("--baselines", default="results/router_baselines/summary.json")
    parser.add_argument("--out", default="results/router_battery")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    requests = json.loads(Path(args.probe).read_text())["requests"]
    train_idx, eval_idx = split_probe(requests)
    eval_items = [requests[i] for i in eval_idx]
    golds = {item["id"]: item["task"] for item in eval_items}
    eval_ids = [item["id"] for item in eval_items]
    tasks = [golds[item_id] for item_id in eval_ids]

    laya = load_decisions(Path(args.laya))
    julia = load_decisions(Path(args.julia))
    baselines = json.loads(Path(args.baselines).read_text())

    baselines_by_id = {
        name: {record["id"]: record for record in records}
        for name, records in baselines["predictions"].items()
    }

    methods: dict[str, dict] = {}
    correctness: dict[str, np.ndarray] = {}
    for name, source in (("laya", laya), ("julia1", julia)):
        records = []
        for item_id in eval_ids:
            record = source[item_id]
            records.append(
                {
                    "id": item_id,
                    "gold": golds[item_id],
                    "pred": record["pred"],
                    "confidence": float(record.get("confidence") or max(record["probabilities"].values())),
                    "correct": record["pred"] == golds[item_id],
                    "latency_ms": record.get("latency_ms"),
                }
            )
        vector = np.array([record["correct"] for record in records])
        correctness[name] = vector
        latencies = [record["latency_ms"] for record in records if record.get("latency_ms") is not None]
        methods[name] = {
            "accuracy": float(vector.mean()),
            "correct": int(vector.sum()),
            "n": len(records),
            "ece": round(expected_calibration_error(np.array([record["confidence"] for record in records]), vector.astype(float)), 4),
            "latency_ms_median": round(float(np.median(latencies)), 1) if latencies else None,
            "variant_agreement": round(variant_agreement(records), 4),
            "concordant_records": records,
        }

    for name, records in baselines_by_id.items():
        if name in methods:
            continue
        vector = np.array([bool(records[item_id]["correct"]) for item_id in eval_ids])
        correctness[name] = vector
        entry = dict(baselines["methods"][name])
        entry["correct"] = int(vector.sum())
        entry["variant_agreement"] = round(
            variant_agreement(
                [
                    {"gold": golds[item_id], "pred": records[item_id]["pred"]}
                    for item_id in eval_ids
                ]
            ),
            4,
        )
        methods[name] = entry

    for name, entry in methods.items():
        low, high = wilson_interval(entry["correct"], entry["n"])
        boot_low, boot_high = cluster_bootstrap_accuracy(correctness[name], tasks, seed=args.seed)
        entry["wilson_ci"] = [round(low, 4), round(high, 4)]
        entry["bootstrap_ci"] = [round(boot_low, 4), round(boot_high, 4)]
        entry["accuracy"] = round(float(entry["accuracy"]), 4)

    pairs = [
        ("laya", "julia1"),
        ("laya", "logreg"),
        ("julia1", "logreg"),
    ]
    pair_rows = []
    raw_ps = []
    for first, second in pairs:
        first_vector = correctness[first]
        second_vector = correctness[second]
        b = int(np.sum(first_vector & ~second_vector))
        c = int(np.sum(~first_vector & second_vector))
        raw_p = mcnemar_exact(b, c)
        raw_ps.append(raw_p)
        risk_difference = float(first_vector.mean() - second_vector.mean())
        pair_rows.append(
            {
                "first": first,
                "second": second,
                "b_first_wrong_only": b,
                "c_second_wrong_only": c,
                "p_raw": raw_p,
                "risk_difference": round(risk_difference, 4),
                "cohens_h": round(cohens_h(float(first_vector.mean()), float(second_vector.mean())), 4),
            }
        )
    adjusted = holm(raw_ps)
    for row, adjusted_p in zip(pair_rows, adjusted):
        row["p_holm"] = adjusted_p

    print(f"evaluation items: {len(eval_ids)}")
    for name, entry in sorted(methods.items(), key=lambda pair: -pair[1]["accuracy"]):
        print(
            f"  {name:<10} accuracy={entry['accuracy']:.3f} "
            f"wilson=[{entry['wilson_ci'][0]:.3f}, {entry['wilson_ci'][1]:.3f}] "
            f"ece={entry['ece'] if entry['ece'] is not None else float('nan'):.3f} "
            f"agreement={entry.get('variant_agreement')} "
            f"latency={entry.get('latency_ms_median')}"
        )
    for row in pair_rows:
        print(
            f"  {row['first']} vs {row['second']}: b={row['b_first_wrong_only']} c={row['c_second_wrong_only']} "
            f"p={row['p_raw']:.4g} p_holm={row['p_holm']:.4g} rd={row['risk_difference']:+.3f} h={row['cohens_h']:+.3f}"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_eval": len(eval_ids),
        "methods": methods,
        "pairs": pair_rows,
        "laya_model": json.loads(Path(args.laya).read_text())["model"],
        "julia_model": json.loads(Path(args.julia).read_text())["model"],
    }
    (out_dir / "router_stats.json").write_text(json.dumps(payload, indent=2))

    display = {
        "laya": "Typed router (Laya)",
        "julia1": "Graph router (Julia 1)",
        "logreg": "Logistic regression",
        "majority": "Majority",
        "random": "Random",
    }
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Method & Accuracy & Wilson 95\\% CI & ECE & Agreement & Latency (ms) \\\\",
        "\\hline",
    ]
    for name in ["laya", "julia1", "logreg", "majority", "random"]:
        if name not in methods:
            continue
        entry = methods[name]
        ece = entry["ece"] if entry["ece"] is not None else float("nan")
        latency = entry["latency_ms_median"]
        latency_text = f"{latency:.1f}" if latency is not None else "--"
        agreement = entry.get("variant_agreement")
        agreement_text = f"{agreement:.3f}" if agreement is not None else "--"
        lines.append(
            f"{display[name]} & {entry['accuracy']:.3f} & "
            f"[{entry['wilson_ci'][0]:.3f}, {entry['wilson_ci'][1]:.3f}] & "
            f"{ece:.3f} & {agreement_text} & {latency_text} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}"]
    (out_dir / "tab_router_methods.tex").write_text("\n".join(lines) + "\n")

    mcnemar_lines = [
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Comparison & $b$ & $c$ & Raw $p$ & Holm $p$ & Risk difference & Cohen's $h$ \\\\",
        "\\hline",
    ]
    for row in pair_rows:
        mcnemar_lines.append(
            f"{display[row['first']]} vs {display[row['second']]} & {row['b_first_wrong_only']} & {row['c_second_wrong_only']} & "
            f"{row['p_raw']:.3g} & {row['p_holm']:.3g} & {row['risk_difference']:+.3f} & {row['cohens_h']:+.3f} \\\\"
        )
    mcnemar_lines += ["\\hline", "\\end{tabular}"]
    (out_dir / "tab_router_mcnemar.tex").write_text("\n".join(mcnemar_lines) + "\n")
    print(f"wrote {out_dir}/router_stats.json and LaTeX rows")


if __name__ == "__main__":
    main()
