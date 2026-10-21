"""Tests for the classical routing baselines."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from code.router.baselines import expected_calibration_error, split_probe


def _probe():
    return json.loads(Path("data/router_probe.json").read_text())["requests"]


def test_split_is_balanced_per_task():
    requests = _probe()
    train, evaluation = split_probe(requests)
    assert len(train) == len(evaluation)
    per_task_per_half = len(requests) // 13 // 2
    for half in (train, evaluation):
        counts = Counter(requests[i]["task"] for i in half)
        assert len(counts) == 13
        assert all(count == per_task_per_half for count in counts.values())


def test_ece_bounds():
    overconfident = expected_calibration_error(np.array([0.9, 0.9]), np.array([1.0, 1.0]))
    assert overconfident > 0.0
    calibrated = expected_calibration_error(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert calibrated == 0.0
