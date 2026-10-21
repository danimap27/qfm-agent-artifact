"""Score the pilot explanations (fidelity harness).

Every metric re-executes against the frozen checkpoint, so counterfactual
consistency, comprehensiveness, and sufficiency are exact for the substrate.
Two text metrics run in parallel: the citation-based unsupported rate for the
condition families that cite, and a citation-free contradiction audit for all
conditions. Both are automatic and heuristic: the protocol reserves a
human-audited sample to bound their error. Output: per-record metrics plus
per-condition aggregates, as JSON and CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from code.eval.claims import check_contradictions, score_explanation
from code.evidence.log_builder import occlude_log, predict_one, sabotage_log
from code.substrate.model import QFMLite
from code.verbalizer.prompts_2x2 import CITE_ONLY_FIELDS
from tasks.registry import get_tasks


def build_given_log(condition: str, log: dict) -> dict:
    """The evidence exactly as the condition was shown it."""
    if condition == "sabotage":
        return sabotage_log(log)
    if condition == "occlusion_attributions":
        return occlude_log(log, ["attributions"])
    if condition == "cite_only":
        return {
            "prediction": log["prediction"],
            "attributions": [],
            "counterfactuals": [],
            "ood": {"score": None, "flag": None},
        }
    return log


def verify_counterfactual(model: QFMLite, task: str, x: np.ndarray, claim: dict) -> dict:
    """Execute a parsed counterfactual claim on the substrate."""
    if claim["to_value"] is None:
        return {"verifiable": False, "reason": "no_value"}
    if not 0 <= claim["index"] < len(x):
        return {"verifiable": False, "reason": "feature_out_of_range"}
    if claim["to_value"] < -0.01 or claim["to_value"] > np.pi + 0.01:
        return {"verifiable": False, "reason": "value_out_of_range"}
    if claim["target_class"] is None:
        return {"verifiable": False, "reason": "no_target"}
    perturbed = x.copy()
    perturbed[claim["index"]] = float(np.clip(claim["to_value"], 0.0, np.pi))
    predicted, _ = predict_one(model, task, perturbed)
    return {"verifiable": True, "consistent": bool(predicted == claim["target_class"])}


def comprehensiveness_sufficiency(model: QFMLite, task: str, x: np.ndarray, features: list[int]) -> dict:
    """Exact prediction change when claimed features are removed or isolated."""
    indexes = [value - 1 for value in features if 1 <= value <= len(x)]
    if not indexes:
        return {"n": 0}
    base_class, base_probabilities = predict_one(model, task, x)
    base = float(base_probabilities[base_class])
    removed = x.copy()
    removed[indexes] = 0.0
    _, removed_probabilities = predict_one(model, task, removed)
    kept = np.zeros_like(x)
    kept[indexes] = x[indexes]
    _, kept_probabilities = predict_one(model, task, kept)
    return {
        "n": len(indexes),
        "comprehensiveness": round(base - float(removed_probabilities[base_class]), 4),
        "sufficiency": round(float(kept_probabilities[base_class]), 4),
    }


def single_block_class_check(parsed: dict, log: dict) -> tuple[bool | None, int | None]:
    """Numeric class claims of the single-block condition against the truth.

    Class names cannot be mapped to indices without task metadata, so only
    numeric claims are scored. Returns (correct, claimed_index).
    """
    claimed = parsed.get("claimed_class") or ""
    digits = "".join(character for character in claimed if character.isdigit())
    if not digits:
        return None, None
    claimed_index = int(digits)
    return bool(claimed_index == log["prediction"]["class_index"]), claimed_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="results/pilot")
    parser.add_argument("--checkpoint", default="results/sweep_d4_b1900/model.npz")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    request_file = json.loads((run_dir / "requests.json").read_text())
    requests = {item["id"]: item for item in request_file["requests"]}
    logs = {}
    for line in (run_dir / "logs.jsonl").read_text().splitlines():
        if line.strip():
            log = json.loads(line)
            logs[log["request_id"]] = log
    records = []
    for line in (run_dir / "explanations.jsonl").read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))

    model = QFMLite.load(Path(args.checkpoint))
    names = sorted({item["task"] for item in requests.values()})
    tasks = get_tasks(names, seed=request_file.get("split_seed", 0))
    del tasks  # loaded for side-effect parity with the run, execution uses the request rows

    per_record = []
    for record in records:
        item = requests[record["request_id"]]
        log = logs[record["request_id"]]
        given_log = build_given_log(record["condition"], log)
        text = record["parsed"].get("explanation") or record["response"]
        allowed = CITE_ONLY_FIELDS if record["condition"] == "cite_only" else None
        scoring = score_explanation(text, given_log, allowed_fields=allowed)
        audit = check_contradictions(text, given_log)
        verifications = [
            verify_counterfactual(model, item["task"], np.asarray(item["features"]), claim)
            for claim in scoring["counterfactual_claims"]
        ]
        verifiable = [item for item in verifications if item["verifiable"]]
        consistent = [item for item in verifiable if item.get("consistent")]
        comp_suff = comprehensiveness_sufficiency(
            model, item["task"], np.asarray(item["features"]), scoring["feature_mentions"]
        )
        claimed_class = None
        claimed_class_correct = None
        if record["condition"] == "single_block":
            claimed_class_correct, claimed_class = single_block_class_check(record["parsed"], log)
        per_record.append(
            {
                "request_id": record["request_id"],
                "condition": record["condition"],
                "n_sentences": scoring["n_sentences"],
                "n_unsupported": scoring["n_unsupported"],
                "unsupported_rate": round(scoring["unsupported_rate"], 4),
                "n_checks": audit["n_checks"],
                "n_contradictions": audit["n_contradictions"],
                "contradiction_rate": (
                    round(audit["contradiction_rate"], 4) if audit["contradiction_rate"] is not None else None
                ),
                "checkable_coverage": round(audit["checkable_coverage"], 4),
                "n_cf_claims": len(scoring["counterfactual_claims"]),
                "n_cf_verifiable": len(verifiable),
                "n_cf_consistent": len(consistent),
                "cf_consistency": (round(len(consistent) / len(verifiable), 4) if verifiable else None),
                "feature_mentions": scoring["feature_mentions"],
                "comprehensiveness": comp_suff.get("comprehensiveness"),
                "sufficiency": comp_suff.get("sufficiency"),
                "claimed_task_correct": (
                    (record["parsed"].get("claimed_task") or "").strip().lower() == item["task"]
                    if record["condition"] == "single_block"
                    else None
                ),
                "claimed_class_correct": claimed_class_correct,
                "claimed_class_index": claimed_class,
                "seconds": record["seconds"],
                "text_chars": len(record["response"]),
                "llm_model": record.get("llm_model"),
                "template": record.get("template"),
                "eval_count": (record.get("tokens") or {}).get("eval_count"),
                "prompt_eval_count": (record.get("tokens") or {}).get("prompt_eval_count"),
            }
        )

    def mean_or_none(values) -> float | None:
        numbers = [value for value in values if value is not None]
        return round(float(statistics.mean(numbers)), 4) if numbers else None

    conditions = sorted({item["condition"] for item in per_record})
    aggregates = {}
    for condition in conditions:
        rows = [item for item in per_record if item["condition"] == condition]
        cf_rows = [item for item in rows if item["cf_consistency"] is not None]
        aggregates[condition] = {
            "n": len(rows),
            "mean_unsupported_rate": mean_or_none([item["unsupported_rate"] for item in rows]),
            "mean_contradiction_rate": mean_or_none([item["contradiction_rate"] for item in rows]),
            "mean_checkable_coverage": mean_or_none([item["checkable_coverage"] for item in rows]),
            "mean_cf_consistency": mean_or_none([item["cf_consistency"] for item in cf_rows]),
            "cf_coverage": round(len(cf_rows) / len(rows), 4),
            "mean_comprehensiveness": mean_or_none([item["comprehensiveness"] for item in rows]),
            "mean_sufficiency": mean_or_none([item["sufficiency"] for item in rows]),
            "median_seconds": round(statistics.median([item["seconds"] for item in rows]), 2),
            "mean_seconds": round(statistics.mean([item["seconds"] for item in rows]), 2),
            "mean_chars": round(statistics.mean([item["text_chars"] for item in rows]), 1),
            "claimed_task_accuracy": (
                mean_or_none(
                    [item["claimed_task_correct"] for item in rows if item["claimed_task_correct"] is not None]
                )
                if condition == "single_block"
                else None
            ),
            "claimed_class_accuracy": (
                mean_or_none(
                    [item["claimed_class_correct"] for item in rows if item["claimed_class_correct"] is not None]
                )
                if condition == "single_block"
                else None
            ),
        }

    out_path = Path(args.out) if args.out else run_dir / "metrics.json"
    out_path.write_text(json.dumps({"per_record": per_record, "aggregates": aggregates}, indent=2))
    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_record[0].keys()))
        writer.writeheader()
        for row in per_record:
            writer.writerow(row)

    header = (
        f"{'condition':<26} {'n':>3} {'unsup':>7} {'contra':>7} {'cf_cov':>7} {'cf_ok':>7} "
        f"{'comp':>7} {'suff':>7} {'sec':>7}"
    )
    print(header)
    for condition in conditions:
        row = aggregates[condition]

        def fmt(value):
            return f"{value:.3f}" if isinstance(value, float) else str(value)

        print(
            f"{condition:<26} {row['n']:>3} {fmt(row['mean_unsupported_rate']):>7} "
            f"{fmt(row['mean_contradiction_rate']):>7} {row['cf_coverage']:>7.3f} "
            f"{fmt(row['mean_cf_consistency']):>7} {fmt(row['mean_comprehensiveness']):>7} "
            f"{fmt(row['mean_sufficiency']):>7} {row['median_seconds']:>7.1f}"
        )
    print(f"wrote {out_path} and {csv_path}")


if __name__ == "__main__":
    main()
