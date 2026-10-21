"""Build the verifiable decision log for one request (schema v1).

Every field is recomputed from the substrate, so the log is exact. Feature
attributions come from single-feature perturbation of the encoded input.
Counterfactuals are searched over a small per-feature grid and recorded only
when re-execution confirms the class flip, which is the same evaluation the
fidelity harness uses, so nothing in the log can drift from what the model
actually computes. The pure-dictionary manipulations live in
``code.evidence.manipulations`` and are re-exported here for compatibility.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Optional

import numpy as np

from code.evidence.manipulations import occlude_log, sabotage_log  # noqa: F401
from code.substrate.model import QFMLite, head_probabilities

CF_GRID = np.linspace(0.0, np.pi, 13)


def feature_names(n_features: int) -> list[str]:
    """Stable human-readable names for the encoded features."""
    return [f"feature {index + 1}" for index in range(n_features)]


def head_class_count(model: QFMLite, task: str) -> int:
    """Number of classes of one task head (a single logit means two classes)."""
    coef = model.heads[task]
    return 2 if coef.shape[0] == 1 else int(coef.shape[0])


def predict_one(model: QFMLite, task: str, x: np.ndarray) -> tuple[int, np.ndarray]:
    """Predicted class and class probabilities for one sample."""
    z = model.features(np.asarray(x, dtype=float)[None, :])
    logits = model.head_logits(task, z)
    probabilities = head_probabilities(logits, head_class_count(model, task))[0]
    return int(np.argmax(probabilities)), probabilities


def feature_attributions(model: QFMLite, task: str, x: np.ndarray, predicted: int) -> list[dict]:
    """Per-feature prediction change under zeroing the angle, ranked by magnitude."""
    base = float(predict_one(model, task, x)[1][predicted])
    names = feature_names(len(x))
    rows = []
    for index in range(len(x)):
        perturbed = np.array(x, dtype=float)
        perturbed[index] = 0.0
        changed = float(predict_one(model, task, perturbed)[1][predicted])
        rows.append(
            {"feature": names[index], "index": index, "delta": base - changed, "method": "perturbation"}
        )
    rows.sort(key=lambda row: -abs(row["delta"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def counterfactuals(
    model: QFMLite, task: str, x: np.ndarray, predicted: int, order: list[int], max_cf: int = 2
) -> list[dict]:
    """Minimal single-feature changes that flip the predicted class, verified by execution."""
    names = feature_names(len(x))
    found: list[dict] = []
    for index in order:
        if len(found) >= max_cf:
            break
        candidates = sorted(CF_GRID, key=lambda value: abs(float(value) - float(x[index])))
        for value in candidates:
            if abs(float(value) - float(x[index])) < 1e-6:
                continue
            perturbed = np.array(x, dtype=float)
            perturbed[index] = value
            flipped, _ = predict_one(model, task, perturbed)
            if flipped != predicted:
                found.append(
                    {
                        "id": f"cf-{len(found) + 1}",
                        "feature": names[index],
                        "index": index,
                        "from": round(float(x[index]), 4),
                        "to": round(float(value), 4),
                        "flips_to": f"class {flipped}",
                        "flips_to_index": int(flipped),
                        "verified": True,
                    }
                )
                break
    return found


class LogBuilder:
    """Builds decision logs against one frozen checkpoint."""

    def __init__(
        self,
        model: QFMLite,
        checkpoint: str,
        train_features: dict[str, np.ndarray],
        task_accuracy: dict[str, float],
    ) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.task_accuracy = task_accuracy
        self.reference: dict[str, tuple[np.ndarray, float]] = {}
        for task, features in train_features.items():
            z = model.features(features)
            if z.shape[0] > 800:
                pick = np.random.default_rng(0).choice(z.shape[0], 800, replace=False)
                z = z[pick]
            squared = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
            np.fill_diagonal(squared, np.inf)
            nearest = np.sqrt(squared.min(axis=1))
            self.reference[task] = (z, float(np.percentile(nearest, 99)))

    def build(
        self,
        request_id: str,
        task: str,
        x: np.ndarray,
        request_text: str,
        routing: dict,
        task_hint: Optional[str] = None,
    ) -> dict:
        """One decision log for one request."""
        predicted, probabilities = predict_one(self.model, task, x)
        attrs = feature_attributions(self.model, task, x, predicted)
        cfs = counterfactuals(self.model, task, x, predicted, order=[row["index"] for row in attrs])
        z = self.model.features(np.asarray(x, dtype=float)[None, :])[0]
        entry = self.reference.get(task)
        if entry is not None:
            sample, threshold = entry
            distance = float(np.sqrt(((sample - z) ** 2).sum(axis=1)).min())
            inside = bool(distance <= threshold)
        else:
            distance, threshold, inside = float("nan"), None, True
        return {
            "schema_version": "1.0",
            "request_id": request_id,
            "request": {"text": request_text, "task_hint": task_hint},
            "routing": routing,
            "prediction": {
                "label": f"class {predicted}",
                "class_index": predicted,
                "probability": round(float(probabilities[predicted]), 4),
                "calibrated": False,
                "head_accuracy_on_task": round(float(self.task_accuracy.get(task, float("nan"))), 4),
            },
            "attributions": attrs,
            "counterfactuals": cfs,
            "ood": {
                "flag": not inside,
                "score": round(distance, 4),
                "threshold": round(threshold, 4) if threshold is not None else None,
                "method": "distance_to_training_manifold",
            },
            "meta": {
                "substrate_checkpoint": self.checkpoint,
                "calibration": "none",
                "evidence_occluded": [],
                "evidence_tampered": False,
                "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            },
        }
