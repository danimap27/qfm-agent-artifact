"""Pure-dictionary manipulations of decision logs.

These copies only touch plain dictionaries, so the remote deployment path can
import them without the substrate or scikit-learn dependencies. They are
re-exported by code.evidence.log_builder for import compatibility.
"""

from __future__ import annotations

import json


def occlude_log(log: dict, fields: list[str]) -> dict:
    """Copy of the log with the requested fields removed and the fact recorded."""
    result = json.loads(json.dumps(log))
    for field in fields:
        if field == "attributions":
            result["attributions"] = []
        elif field == "counterfactuals":
            result["counterfactuals"] = []
        elif field == "prediction.probability":
            result["prediction"]["probability"] = None
        elif field == "ood":
            result["ood"] = {"flag": None, "score": None, "method": "occluded"}
        elif field == "routing":
            result["routing"] = {"method": "occluded", "decision": None, "probabilities": {}, "confidence": None, "abstained": None}
    result["meta"]["evidence_occluded"] = list(fields)
    return result


def sabotage_log(log: dict) -> dict:
    """Copy of the log with attribution ranks reversed so the evidence lies."""
    result = json.loads(json.dumps(log))
    rows = list(reversed(result["attributions"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    result["attributions"] = rows
    result["meta"]["evidence_tampered"] = True
    return result
