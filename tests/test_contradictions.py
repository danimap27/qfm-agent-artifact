"""Tests for the contradiction audit."""

from code.eval.claims import check_contradictions


def _log() -> dict:
    return {
        "prediction": {
            "label": "class 1",
            "class_index": 1,
            "probability": 0.82,
            "calibrated": False,
            "head_accuracy_on_task": 0.79,
        },
        "attributions": [
            {"feature": "feature 4", "index": 3, "rank": 1, "delta": 0.31, "method": "perturbation"},
            {"feature": "feature 2", "index": 1, "rank": 2, "delta": 0.19, "method": "perturbation"},
        ],
        "counterfactuals": [
            {
                "id": "cf-1",
                "feature": "feature 4",
                "index": 3,
                "from": 1.2,
                "to": 0.4,
                "flips_to": "class 0",
                "flips_to_index": 0,
                "verified": True,
            }
        ],
        "ood": {"flag": False, "score": 0.12, "method": "distance_to_training_manifold"},
        "request": {"text": "example", "task_hint": None},
        "routing": {
            "method": "oracle",
            "decision": "alpha",
            "probabilities": {},
            "confidence": None,
            "abstained": False,
        },
    }


def test_probability_contradiction():
    audit = check_contradictions("The probability is 0.55 for this case.", _log())
    assert audit["n_checks"] == 1
    assert audit["n_contradictions"] == 1
    assert audit["contradiction_rate"] == 1.0


def test_percent_matches_probability():
    audit = check_contradictions("The model shows a confidence of 82%.", _log())
    assert audit["n_checks"] == 1
    assert audit["n_contradictions"] == 0


def test_class_contradiction_and_flip_allowed():
    bad = check_contradictions("The specimen belongs to class 5.", _log())
    assert bad["n_contradictions"] == 1
    allowed = check_contradictions("Changing the feature would move the prediction to class 0.", _log())
    assert allowed["n_contradictions"] == 0


def test_feature_prominence_contradiction():
    bad = check_contradictions("Feature 2 was the most important signal.", _log())
    assert bad["n_contradictions"] == 1
    good = check_contradictions("Feature 4 was the most important signal.", _log())
    assert good["n_contradictions"] == 0


def test_no_checkable_assertions():
    audit = check_contradictions("The model is very confident.", _log())
    assert audit["n_checks"] == 0
    assert audit["contradiction_rate"] is None
