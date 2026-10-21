"""Typed-question schema for the routing layer.

The router asks Laya one choice question per request: which QFM-lite task the
request refers to. The criteria are kept here as plain data so this module is
importable without the heavy router dependencies (torch, transformers) and
without scikit-learn, which is a substrate-side dependency. A unit test keeps
the criteria in sync with the task registry.
"""

from __future__ import annotations

from typing import Any

ROUTER_INSTRUCTIONS = "Which dataset and prediction task does this request refer to?"

TASK_CRITERIA: dict[str, str] = {
    "iris": "morphological measurements of iris flowers for species classification",
    "wine": "chemical measurements of wines for cultivar classification",
    "wdbc": "breast mass cytology measurements for malignancy classification",
    "thyroid": "thyroid function laboratory panels for thyroid disease classification",
    "parkinsons": "biomedical voice measurements for Parkinson's disease screening",
    "wine_quality": "physicochemical tests of wines for quality score prediction",
    "sonar": "sonar signal returns for metal or rock classification",
    "digits": "small low-resolution images of handwritten digits",
    "mnist8": "large collections of handwritten digit images",
    "fashion8": "grayscale images of clothing items for category classification",
    "periodic_sin": "periodic signal forecasting from thresholded sine waves",
    "ionosphere": "radar returns from the ionosphere for structure classification",
    "blood_transfusion": "blood donation service records for donor return prediction",
}


def build_task_question(task_ids: list[str] | None = None) -> dict[str, Any]:
    """Build the typed question that routes a request to a task."""
    names = task_ids if task_ids is not None else list(TASK_CRITERIA)
    return {
        "task": {
            "type": "choice",
            "instructions": ROUTER_INSTRUCTIONS,
            "criteria": {name: TASK_CRITERIA[name] for name in names},
        }
    }


def parse_decision(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Laya response into the router decision schema."""
    answer = result["answers"]["task"]
    probabilities = {
        key: round(float(value), 4) for key, value in answer.get("probabilities", {}).items()
    }
    confidence = answer.get("confidence", answer.get("answer_confidence", 0.0))
    return {
        "task": answer["choice"],
        "probabilities": probabilities,
        "confidence": round(float(confidence), 4),
    }
