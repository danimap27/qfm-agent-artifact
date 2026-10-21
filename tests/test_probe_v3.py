"""Tests for the expanded (v3) routing probe."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from code.router.build_probe import NEW_VARIANTS, build


def _probe() -> dict:
    return json.loads(Path("data/router_probe.json").read_text())


def test_probe_v3_shape() -> None:
    probe = _probe()
    assert probe["version"] == 3
    requests = probe["requests"]
    assert len(requests) >= 150
    counts = Counter(item["task"] for item in requests)
    assert len(counts) == 13
    assert all(count == 12 for count in counts.values())
    identifiers = [item["id"] for item in requests]
    assert len(identifiers) == len(set(identifiers))
    assert all(item["text"].strip() for item in requests)


def test_build_orders_halves_and_keeps_originals() -> None:
    sample = {"requests": []}
    for task in sorted(NEW_VARIANTS):
        for index in range(1, 5):
            sample["requests"].append({"id": f"{task}-{index}", "task": task, "text": f"original {index}"})
    built = build(sample)
    assert len(built["requests"]) == 13 * 12
    iris = [item["id"] for item in built["requests"] if item["task"] == "iris"]
    assert iris == ["iris-1", "iris-2", "iris-5", "iris-6", "iris-7", "iris-8", "iris-3", "iris-4", "iris-9", "iris-10", "iris-11", "iris-12"]
    first_half = [item for item in built["requests"] if item["task"] == "iris"][:6]
    assert {item["id"] for item in first_half} & {"iris-1", "iris-2"} == {"iris-1", "iris-2"}
