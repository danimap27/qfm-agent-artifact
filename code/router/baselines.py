"""Classical routing baselines on the labeled request probe.

The comparison the routing claim needs (C1): a logistic regression on the
request text, a majority baseline, and a uniform baseline, evaluated on the
same hold-out half of the probe that the typed router is evaluated on. The
typed router is zero-shot, so the logistic regression is the only trained
baseline. Expected calibration error is reported for every method with
probabilities.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def split_probe(requests: list[dict]) -> tuple[list[int], list[int]]:
    """First half of each task's requests for training, second half for evaluation."""
    per_task: dict[str, list[int]] = {}
    for index, item in enumerate(requests):
        per_task.setdefault(item["task"], []).append(index)
    train: list[int] = []
    evaluation: list[int] = []
    for task in sorted(per_task):
        indexes = per_task[task]
        half = len(indexes) // 2
        train.extend(indexes[:half])
        evaluation.extend(indexes[half:])
    return train, evaluation


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Weighted average of |accuracy - confidence| over probability bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    total = int(len(confidences))
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidences > low) & (confidences <= high)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_accuracy = float(correct[mask].mean())
        bin_confidence = float(confidences[mask].mean())
        error += (count / total) * abs(bin_accuracy - bin_confidence)
    return float(error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="data/router_probe.json")
    parser.add_argument("--laya", default="results/router_probe_laya/routing_decisions.json")
    parser.add_argument("--out", default="results/router_baselines")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    requests = json.loads(Path(args.probe).read_text())["requests"]
    train_idx, eval_idx = split_probe(requests)
    train_texts = [requests[i]["text"] for i in train_idx]
    train_labels = [requests[i]["task"] for i in train_idx]
    eval_items = [requests[i] for i in eval_idx]
    golds = [item["task"] for item in eval_items]
    labels = sorted({item["task"] for item in requests})

    methods: dict[str, dict] = {}

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    train_matrix = vectorizer.fit_transform(train_texts)
    classifier = LogisticRegression(max_iter=2000, random_state=args.seed)
    classifier.fit(train_matrix, train_labels)
    eval_matrix = vectorizer.transform([item["text"] for item in eval_items])
    probabilities = classifier.predict_proba(eval_matrix)
    timings = []
    for row in range(len(eval_items)):
        single = eval_matrix[row]
        started = time.perf_counter()
        classifier.predict(single)
        timings.append((time.perf_counter() - started) * 1000)
    methods["logreg"] = {
        "pred": list(classifier.predict(eval_matrix)),
        "conf": [float(row.max()) for row in probabilities],
        "latency_ms_median": float(np.median(timings)),
    }

    counts: dict[str, int] = {}
    for label in train_labels:
        counts[label] = counts.get(label, 0) + 1
    majority = max(sorted(counts), key=lambda label: counts[label])
    methods["majority"] = {"pred": [majority] * len(eval_items), "conf": [1.0] * len(eval_items)}

    rng = random.Random(args.seed)
    methods["random"] = {
        "pred": [rng.choice(labels) for _ in eval_items],
        "conf": [1.0 / len(labels)] * len(eval_items),
    }
    methods["oracle"] = {"pred": golds, "conf": [1.0] * len(eval_items)}

    laya_path = Path(args.laya)
    if laya_path.exists():
        decisions = json.loads(laya_path.read_text())
        predicted: list = []
        confidences: list = []
        for item in eval_items:
            decision = decisions.get(item["id"])
            predicted.append(decision["task"] if decision else None)
            confidences.append(float(decision["confidence"]) if decision else 0.0)
        methods["laya"] = {"pred": predicted, "conf": confidences}
    else:
        print(f"note: no Laya decisions at {laya_path}, skipping the typed router row")

    summary: dict[str, dict] = {}
    predictions: dict[str, list] = {}
    for name, method in methods.items():
        correct = np.array([pred == gold for pred, gold in zip(method["pred"], golds)])
        accuracy = float(correct.mean())
        ece = expected_calibration_error(np.asarray(method["conf"], dtype=float), correct.astype(float))
        summary[name] = {
            "accuracy": round(accuracy, 4),
            "ece": round(ece, 4),
            "n": len(eval_items),
            "latency_ms_median": (
                round(float(method["latency_ms_median"]), 3) if method.get("latency_ms_median") is not None else None
            ),
        }
        predictions[name] = [
            {"id": item["id"], "gold": gold, "pred": pred, "confidence": conf, "correct": bool(hit)}
            for item, gold, pred, conf, hit in zip(
                eval_items, golds, method["pred"], method["conf"], correct
            )
        ]
        print(f"{name:<10} accuracy={accuracy:.3f}  ece={ece:.3f}  n={len(eval_items)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"seed": args.seed, "methods": summary, "predictions": predictions}, indent=2)
    )
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
