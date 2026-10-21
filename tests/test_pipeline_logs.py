"""Tests for prebuilt decision logs, pure manipulations, and the --logs path."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from code.evidence.manipulations import occlude_log, sabotage_log
from code.pipeline.logs import load_logs, write_logs

REPO_ROOT = Path(__file__).resolve().parents[1]


def _log(request_id: str) -> dict:
    return {
        "schema_version": "1.0",
        "request_id": request_id,
        "prediction": {"label": "class 1", "class_index": 1, "probability": 0.8},
        "attributions": [
            {"feature": "feature 1", "index": 0, "rank": 1, "delta": 0.2},
            {"feature": "feature 2", "index": 1, "rank": 2, "delta": 0.1},
        ],
        "counterfactuals": [],
        "meta": {"evidence_occluded": [], "evidence_tampered": False},
    }


def test_write_and_load_logs_roundtrip(tmp_path: Path) -> None:
    logs = {"a-1": _log("a-1"), "b-1": _log("b-1")}
    path = write_logs(logs, tmp_path / "logs.jsonl")
    loaded = load_logs(path)
    assert list(loaded) == ["a-1", "b-1"]
    assert loaded["a-1"] == logs["a-1"]


def test_manipulations_do_not_touch_the_original() -> None:
    log = _log("a-1")
    occluded = occlude_log(log, ["attributions"])
    tampered = sabotage_log(log)
    assert log["attributions"] != []
    assert log["meta"]["evidence_tampered"] is False
    assert occluded["attributions"] == []
    assert occluded["meta"]["evidence_occluded"] == ["attributions"]
    assert [row["rank"] for row in tampered["attributions"]] == [1, 2]
    assert tampered["attributions"][0]["feature"] == "feature 2"
    assert tampered["meta"]["evidence_tampered"] is True


def test_log_builder_reexports_manipulations() -> None:
    from code.evidence.log_builder import occlude_log as via_builder

    assert via_builder is occlude_log


def test_run_agent_with_prebuilt_logs_runs_without_substrate(tmp_path: Path) -> None:
    requests = {
        "split_seed": 0,
        "task_ids": ["iris", "wine"],
        "tasks": ["iris"],
        "requests": [
            {
                "id": "iris-1",
                "task": "iris",
                "text": "Case 4100: routine botanical screening.",
                "features": [0.1] * 8,
                "gold_class": 0,
                "n_classes": 3,
            }
        ],
    }
    (tmp_path / "requests.json").write_text(json.dumps(requests))
    write_logs({"iris-1": _log("iris-1")}, tmp_path / "logs.jsonl")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "code.pipeline.run_agent",
            "--requests",
            str(tmp_path / "requests.json"),
            "--logs",
            str(tmp_path / "logs.jsonl"),
            "--out",
            str(tmp_path / "out"),
            "--conditions",
            "template_floor",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert "loaded 1 prebuilt logs" in result.stdout
    records = [
        json.loads(line)
        for line in (tmp_path / "out" / "explanations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["condition"] == "template_floor"
    assert record["template"] == "t1"
    assert record["llm_model"] == "none"
    assert "[prediction.probability]" in record["response"]
    assert record["tokens"] == {"eval_count": None, "prompt_eval_count": None}
