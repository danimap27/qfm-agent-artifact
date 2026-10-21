"""Tests for the v3 prompt templates and the single-block scoring helpers."""

from __future__ import annotations

from code.eval.claims import field_table, parse_counterfactuals, score_sentence
from code.eval.score import single_block_class_check
from code.pipeline.run_agent import parse_single_block
from code.verbalizer.prompts import (
    PROMPT_VERSION,
    free_messages,
    grounded_messages,
    prompt_hash,
    single_block_messages,
)

# Frozen v2 hashes: the t1 messages must stay byte-identical, because the
# cached prompt hashes of every earlier run are built on them.
T1_GROUNDED_HASH = "2301a8a8c3e72fc0"
T1_FREE_HASH = "7f0f3cbc33627101"
T1_SINGLE_HASH = "00ccaa3510971828"


def _mini_log() -> dict:
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


def test_prompt_version_is_v3():
    assert PROMPT_VERSION == "v3"


def test_t1_messages_match_frozen_v2_hashes():
    log = _mini_log()
    log["meta"] = {"created": "2026-09-28T09:44:29+00:00", "calibration": "none"}
    assert prompt_hash(grounded_messages("a request", log)) == T1_GROUNDED_HASH
    assert prompt_hash(free_messages("a request", "class 1", 0.82)) == T1_FREE_HASH
    assert prompt_hash(single_block_messages("classify this request", [0.1] * 8, ["iris", "wine"])) == T1_SINGLE_HASH


def test_templates_keep_the_citation_contract_in_grounded():
    log = _mini_log()
    baseline = grounded_messages("a request", log, template="t1")
    for template in ("t2", "t3"):
        messages = grounded_messages("a request", log, template=template)
        system = messages[0]["content"]
        assert "between square brackets" in system
        assert "[cf-1]" in system
        assert "120 words" in system
        assert messages != baseline


def test_templates_keep_the_single_block_parse_contract():
    for template in ("t2", "t3"):
        messages = single_block_messages("classify this request", [0.1] * 8, ["iris", "wine"], template=template)
        system = messages[0]["content"]
        assert "Task:" in system and "Class:" in system and "Explanation:" in system
        assert "iris, wine" in system


def test_unknown_template_is_rejected():
    try:
        grounded_messages("a request", _mini_log(), template="t9")
    except ValueError as error:
        assert "t9" in str(error)
    else:
        raise AssertionError("unknown template must raise")


def test_grounded_messages_ignore_log_creation_time():
    log_a = _mini_log()
    log_b = _mini_log()
    log_a["meta"] = {"created": "2026-09-28T09:44:29+00:00", "calibration": "none"}
    log_b["meta"] = {"created": "2026-12-31T23:59:59+00:00", "calibration": "none"}
    messages_a = grounded_messages("a request", log_a)
    messages_b = grounded_messages("a request", log_b)
    assert messages_a == messages_b
    assert "created" not in messages_a[1]["content"]


def test_single_block_prompt_lists_candidates():
    messages = single_block_messages("classify this request", [0.1] * 8, ["iris", "wine"])
    assert "iris, wine" in messages[0]["content"]
    assert "Task:" in messages[0]["content"]


def test_parse_single_block_fixed_format():
    text = "Task: iris\nClass: 0\nExplanation: The prediction follows from the cited evidence."
    parsed = parse_single_block(text)
    assert parsed["claimed_task"] == "iris"
    assert parsed["claimed_class"] == "0"
    assert parsed["parsed"] is True
    assert parsed["explanation"].startswith("The prediction")


def test_single_block_class_check_numeric_only():
    log = {"prediction": {"class_index": 2, "label": "class 2"}}
    correct, index = single_block_class_check({"claimed_class": "2"}, log)
    assert correct is True and index == 2
    correct, index = single_block_class_check({"claimed_class": "class 1"}, log)
    assert correct is False and index == 1
    correct, index = single_block_class_check({"claimed_class": "setosa"}, log)
    assert correct is None and index is None


def test_counterfactual_id_bracket_normalization():
    sentence = (
        "The model indicates that changing feature 4 from 1.20 to 0.40 would flip the prediction "
        'to class 0 [counterfactuals.id="cf-1"].'
    )
    assert score_sentence(sentence, field_table(_mini_log()))["status"] == "supported"


def test_two_counterfactuals_in_one_sentence():
    sentence = (
        "Changing feature 4 to 0.40 would flip to class 0, while feature 3 to 0.90 would move it "
        "to class 1."
    )
    claims = parse_counterfactuals(sentence)
    assert len(claims) == 2
    assert claims[0]["to_value"] == 0.4 and claims[0]["target_class"] == 0
    assert claims[1]["index"] == 2 and claims[1]["target_class"] == 1
