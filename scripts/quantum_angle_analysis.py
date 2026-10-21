"""Quantum-angle analyses: exact Shapley agreement and finite-shot reliability.

Part A compares the perturbation attributions stored in the decision log with
exact Shapley values under three baselines (zero, midrange, mean of data),
which quantifies how much of the attribution is an artifact of the zeroing
baseline in the angular encoding.

Part B measures how the counterfactual audit degrades when the evidence itself
is estimated from finitely many measurement shots, using the Gaussian limit
of the binomial variance of Pauli-Z estimates, var(z_i) = (1 - z_i^2) / n.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from typing import Any, Dict, List

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from code.evidence.log_builder import counterfactuals, feature_attributions, head_probabilities, head_class_count
from code.evidence.shapley import baseline_mean, baseline_midrange, baseline_zero, shapley_values
from code.substrate.model import QFMLite

CHECKPOINT = pathlib.Path("results/sweep_d4_b1900/model.npz")
REQUESTS = pathlib.Path("results/grid_q1/qwen35/t1_table/seed0/requests.json")

model = QFMLite.load(CHECKPOINT)
request_file = json.loads(REQUESTS.read_text(encoding="utf-8"))
requests = request_file["requests"]
print(f"requests: {len(requests)}")


def predict_proba_factory(task: str):
    def predict_proba(x: np.ndarray) -> np.ndarray:
        z = model.features(np.asarray(x, dtype=float)[None, :])
        logits = model.head_logits(task, z)
        return head_probabilities(logits, head_class_count(model, task))[0]

    return predict_proba


# ---------------------------------------------------------------- Part A
print("== Part A: Shapley agreement over the sampled requests ==")
spearman_zero: List[float] = []
spearman_mid: List[float] = []
spearman_mean: List[float] = []
top1_zero: List[float] = []
top1_mid: List[float] = []
sample = requests[:40]
for item in sample:
    task = item["task"]
    x = np.asarray(item["features"], dtype=float)
    predict_proba = predict_proba_factory(task)
    z = model.features(x[None, :])
    logits = model.head_logits(task, z)
    probs = head_probabilities(logits, head_class_count(model, task))[0]
    predicted = int(np.argmax(probs))
    attrs = feature_attributions(model, task, x, predicted)
    perturb = np.array([row["delta"] for row in attrs])
    # features() input is angles in [0, pi]; mean baseline over the sampled inputs
    all_x = np.asarray([np.asarray(r["features"], dtype=float) for r in requests if r["task"] == task])
    baselines = {
        "zero": baseline_zero(len(x)),
        "midrange": baseline_midrange(len(x)),
        "mean": baseline_mean(all_x),
    }
    shap = {name: shapley_values(predict_proba, x, predicted, base) for name, base in baselines.items()}
    for name, target in (("zero", spearman_zero), ("midrange", spearman_mid), ("mean", spearman_mean)):
        rho = spearmanr(np.abs(perturb), np.abs(shap[name])).statistic
        if rho == rho:
            target.append(float(rho))
    top1_zero.append(float(np.argmax(np.abs(perturb)) == np.argmax(np.abs(shap["zero"]))))
    top1_mid.append(float(np.argmax(np.abs(perturb)) == np.argmax(np.abs(shap["midrange"]))))


def report(name: str, values: List[float]) -> None:
    if values:
        print(f"  {name:44s} mean={np.mean(values):.3f} min={np.min(values):.3f} (n={len(values)})")


report("Spearman |perturbation| vs |Shapley zero|", spearman_zero)
report("Spearman |perturbation| vs |Shapley midrange|", spearman_mid)
report("Spearman |perturbation| vs |Shapley mean|", spearman_mean)
report("top-1 agreement perturbation vs Shapley zero", top1_zero)
report("top-1 agreement perturbation vs Shapley midrange", top1_mid)

# ---------------------------------------------------------------- Part B
print("== Part B: counterfactual audit under finite shots ==")
points = []  # (margin, w2) with w2 = sum_i dw_i^2 (1 - z_i^2)
for item in requests:
    task = item["task"]
    x = np.asarray(item["features"], dtype=float)
    z = model.features(x[None, :])
    logits = model.head_logits(task, z)
    probs = head_probabilities(logits, head_class_count(model, task))[0]
    predicted = int(np.argmax(probs))
    attrs = feature_attributions(model, task, x, predicted)
    order = [row["index"] for row in attrs]
    for cf in counterfactuals(model, task, x, predicted, order, max_cf=2):
        perturbed = x.copy()
        perturbed[cf["index"]] = float(cf["to"])
        z_cf = model.features(perturbed[None, :])[0]
        logits_cf = model.head_logits(task, z_cf[None, :])[0]
        target = int(cf["flips_to_index"])
        if (
            logits_cf.size < 2
            or int(target) >= logits_cf.size
            or int(predicted) >= logits_cf.size
            or int(target) == int(predicted)
        ):
            continue  # single-class head or non-computable pair
        margin = float(logits_cf[target] - logits_cf[predicted])
        w_diff = np.asarray(model.heads[task][target], dtype=float) - np.asarray(
            model.heads[task][predicted], dtype=float
        )
        var = np.maximum(1.0 - z_cf**2, 1e-12)
        w2 = float(np.sum(w_diff[:-1] ** 2 * var))
        points.append((abs(margin), w2))

margins = np.asarray([m for m, _ in points])
w2s = np.asarray([w for _, w in points])
print(f"  counterfactual points: {len(points)}")
print(f"  decision margin at flip: median={np.median(margins):.4f} p10={np.percentile(margins, 10):.4f}")
print(f"  {'shots':>8s} {'P(wrong audit verdict)':>24s}")
shots_curve = {}
for shots in (50, 100, 200, 500, 1000, 2000, 5000):
    sd = np.sqrt(w2s / shots)
    # one-sided: the verdict is wrong when noise pushes the margin below zero,
    # P = Phi(-m / sd); a pairwise bound, since a third class can also overtake
    rate = float(np.mean(0.5 * (1.0 + np.vectorize(math.erf)(-margins / (sd * np.sqrt(2.0))))))
    shots_curve[shots] = rate
    print(f"  {shots:8d} {rate:24.3f}")
need = w2s * (1.644854 / np.maximum(margins, 1e-9)) ** 2  # one-sided 95 percent
print(f"  shots for a 95 percent correct verdict: median={np.median(need):.0f} p90={np.percentile(need, 90):.0f}")

out = pathlib.Path("results/post_freeze/quantum_angle.json")
out.write_text(
    json.dumps(
        {
            "shapley": {
                "spearman_zero": float(np.mean(spearman_zero)),
                "spearman_mid": float(np.mean(spearman_mid)),
                "spearman_mean": float(np.mean(spearman_mean)),
                "top1_zero": float(np.mean(top1_zero)),
                "top1_mid": float(np.mean(top1_mid)),
            },
            "shots": shots_curve,
            "margin_median": float(np.median(margins)),
            "margin_p10": float(np.percentile(margins, 10)),
            "shots_needed_median": float(np.median(need)),
            "shots_needed_p90": float(np.percentile(need, 90)),
            "n_counterfactuals": len(points),
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"written {out}")
