"""Build and load decision logs for pipeline runs.

Building requires the substrate and the task registry, which import Qiskit
and scikit-learn. Those imports happen inside build_logs, so loading a
prebuilt logs.jsonl works in the remote runner environments, where only the
standard library and numpy are present.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_logs(requests: list[dict], split_seed: int, checkpoint: str | Path, classical: bool = False) -> dict[str, dict]:
    """One decision log per request, against the frozen checkpoint.

    ``classical`` swaps the circuit for the classical control, whose heads are
    refit on the encoded angles, so every other stage stays identical.
    """
    import numpy as np

    from code.evidence.log_builder import LogBuilder
    from code.substrate.classical import load_classical_control
    from code.substrate.model import QFMLite, head_probabilities
    from tasks.registry import get_tasks

    names = sorted({item["task"] for item in requests})
    tasks = get_tasks(names, seed=split_seed)
    model = load_classical_control(checkpoint, split_seed) if classical else QFMLite.load(Path(checkpoint))
    accuracy = {}
    for name, task in tasks.items():
        z = model.features(task.X_test)
        logits = model.head_logits(task.name, z)
        probabilities = head_probabilities(logits, task.n_classes)
        accuracy[name] = float((probabilities.argmax(axis=1) == task.y_test).mean())
    builder = LogBuilder(
        model,
        "classical_control" if classical else Path(checkpoint).parent.name,
        {name: task.X_train for name, task in tasks.items()},
        accuracy,
    )
    logs: dict[str, dict] = {}
    for item in requests:
        routing = {
            "method": "oracle",
            "decision": item["task"],
            "head": f"head_{item['task']}",
            "probabilities": {},
            "confidence": None,
            "abstained": False,
        }
        logs[item["id"]] = builder.build(
            item["id"], item["task"], np.asarray(item["features"]), item["text"], routing
        )
    return logs


def load_logs(path: str | Path) -> dict[str, dict]:
    """Read a logs.jsonl file keyed by request id."""
    logs: dict[str, dict] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            log = json.loads(line)
            logs[log["request_id"]] = log
    return logs


def write_logs(logs: dict[str, dict], path: str | Path) -> Path:
    """Write logs as JSONL, one sorted-key object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(log, sort_keys=True) for log in logs.values()) + "\n")
    return path
