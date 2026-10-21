"""Tests for the routing schema (no router or substrate dependencies)."""

from code.router.schema import TASK_CRITERIA, build_task_question, parse_decision


def test_criteria_match_the_registry():
    from tasks.registry import HELD_OUT, PRETRAIN_MIX

    assert set(TASK_CRITERIA) == set(PRETRAIN_MIX) | set(HELD_OUT)


def test_question_shape_and_subset():
    question = build_task_question(["iris", "wine"])
    spec = question["task"]
    assert spec["type"] == "choice"
    assert list(spec["criteria"]) == ["iris", "wine"]
    assert spec["instructions"]


def test_parse_decision_normalizes():
    result = {
        "answers": {
            "task": {
                "type": "choice",
                "choice": "wine",
                "probabilities": {"wine": 0.9, "iris": 0.1},
                "confidence": 0.8,
            }
        }
    }
    decision = parse_decision(result)
    assert decision["task"] == "wine"
    assert decision["probabilities"] == {"wine": 0.9, "iris": 0.1}
    assert decision["confidence"] == 0.8
