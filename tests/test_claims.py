"""Tests for the claim scorer."""

from code.eval.claims import field_table, parse_counterfactual, score_explanation, score_sentence


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


def test_supported_sentence():
    result = score_sentence(
        "The prediction is class 1 with probability 0.82 [prediction.probability].", field_table(_log())
    )
    assert result["status"] == "supported"


def test_value_mismatch_on_probability():
    result = score_sentence(
        "The prediction is class 1 with probability 0.55 [prediction.probability].", field_table(_log())
    )
    assert result["status"] == "value_mismatch"


def test_uncited_and_invalid_citation():
    assert score_sentence("The model is very confident.", field_table(_log()))["status"] == "uncited"
    assert (
        score_sentence("It is confident [foo.bar].", field_table(_log()))["status"] == "invalid_citation"
    )


def test_feature_reference_mismatch():
    result = score_sentence(
        "Feature 7 drives the prediction [attributions.rank1].", field_table(_log())
    )
    assert result["status"] == "value_mismatch"


def test_counterfactual_parsing():
    claim = parse_counterfactual(
        "Changing feature 4 from 1.20 to 0.40 would change the prediction to class 0."
    )
    assert claim is not None
    assert claim["index"] == 3
    assert claim["from_value"] == 1.2
    assert claim["to_value"] == 0.4
    assert claim["target_class"] == 0


def test_template_floor_sentence_is_supported():
    sentence = (
        "Changing feature 4 from 1.20 to 0.40 would change the prediction to class 0 [cf-1]."
    )
    assert score_sentence(sentence, field_table(_log()))["status"] == "supported"


def test_score_explanation_rates():
    text = (
        "The prediction is class 1 with probability 0.82 [prediction.probability]. "
        "Nothing else follows."
    )
    result = score_explanation(text, _log())
    assert result["n_sentences"] == 2
    assert result["n_unsupported"] == 1
    assert result["counterfactual_claims"] == []
