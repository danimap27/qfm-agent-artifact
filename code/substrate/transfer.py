"""Transfer protocol: frozen substrate plus residual adapters on held-out tasks.

For each held-out task the experiment reports two numbers on the same frozen
substrate: a linear probe that trains only the task head on frozen features,
and the full protocol that trains a residual adapter together with the head.
The adapter is rank constrained and its output projection starts at zero, so
the adapted model begins exactly at the frozen model. The gap between probe
and adapter quantifies what low-rank adaptation buys on a task the substrate
has never seen, which is the transfer check of the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression

from code.substrate.adapters import AdapterConfig, ResidualAdapter
from code.substrate.model import QFMLite, cross_entropy, head_probabilities
from code.substrate.spsa import SPSAConfig, run_spsa
from tasks.loaders import TaskData
from tasks.registry import get_tasks

SPSA_DEFAULTS: Dict[str, float] = {"a": 0.4, "c": 0.12, "alpha": 0.602, "gamma": 0.101, "A": 5.0}


def fit_head(model: QFMLite, task: str, z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit the logistic head of one task on top of substrate features."""
    classifier = LogisticRegression(max_iter=2000)
    classifier.fit(np.hstack([z, np.ones((z.shape[0], 1))]), y)
    coef = np.asarray(classifier.coef_, dtype=float)
    model.set_head(task, coef)
    return coef


def head_accuracy(
    model: QFMLite, task: str, z: np.ndarray, y: np.ndarray, n_classes: int
) -> float:
    """Accuracy of the current head on a batch of substrate features."""
    logits = model.head_logits(task, z)
    predicted = head_probabilities(logits, n_classes).argmax(axis=1)
    return float(np.mean(predicted == y))


def train_adapter(
    model: QFMLite,
    task: str,
    data: TaskData,
    config: AdapterConfig,
    seed: int,
    outers: int,
    steps: int,
) -> ResidualAdapter:
    """Alternated training of a residual adapter and the head, substrate frozen.

    The validation accuracy selects the best round and the protocol returns
    the adapter and head of that round, which avoids late-round overfitting on
    small tasks.
    """
    adapter = ResidualAdapter(model.config.n_features, config, seed=seed)
    X_train, y_train = data.X_train, data.y_train
    # The identity adapter (the plain probe) is an explicit candidate, so the
    # protocol never returns a model worse than the probe on validation.
    fit_head(model, task, model.features(X_train), y_train)
    best_val = head_accuracy(model, task, model.features(data.X_val), data.y_val, data.n_classes)
    best_params: np.ndarray = adapter.get_params()
    best_head: np.ndarray = np.array(model.heads[task], copy=True)
    for outer in range(outers):
        z_train = model.features(adapter.forward(X_train))
        coef = fit_head(model, task, z_train, y_train)

        def loss_fn(params: np.ndarray) -> float:
            adapter.set_params(params)
            z = model.features(adapter.forward(X_train))
            logits = np.hstack([z, np.ones((z.shape[0], 1))]) @ coef.T
            return cross_entropy(head_probabilities(logits, data.n_classes), y_train)

        spsa = SPSAConfig(steps=steps, seed=seed, **SPSA_DEFAULTS)
        result = run_spsa(
            loss_fn,
            adapter.get_params(),
            spsa,
            rng=np.random.default_rng([seed, 7000 + outer]),
            k_offset=outer * steps,
        )
        adapter.set_params(result["theta"])
        z_train = model.features(adapter.forward(X_train))
        fit_head(model, task, z_train, y_train)
        val_accuracy = head_accuracy(
            model,
            task,
            model.features(adapter.forward(data.X_val)),
            data.y_val,
            data.n_classes,
        )
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_params = adapter.get_params()
            best_head = np.array(model.heads[task], copy=True)
    adapter.set_params(best_params)
    model.set_head(task, best_head)
    return adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="pretrained model npz")
    parser.add_argument("--tasks", default="ionosphere,blood_transfusion")
    parser.add_argument("--out", type=Path, default=Path("results/f1_transfer"))
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--outers", type=int, default=6)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = QFMLite.load(args.model)
    task_names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    datasets = get_tasks(task_names, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for name in task_names:
        started = time.perf_counter()
        data = datasets[name]
        z_train = model.features(data.X_train)
        fit_head(model, name, z_train, data.y_train)
        probe_val = head_accuracy(model, name, model.features(data.X_val), data.y_val, data.n_classes)
        probe_test = head_accuracy(
            model, name, model.features(data.X_test), data.y_test, data.n_classes
        )

        adapter = train_adapter(
            model, name, data, AdapterConfig(rank=args.rank), args.seed, args.outers, args.steps
        )
        adapter_val = head_accuracy(
            model, name, model.features(adapter.forward(data.X_val)), data.y_val, data.n_classes
        )
        adapter_test = head_accuracy(
            model, name, model.features(adapter.forward(data.X_test)), data.y_test, data.n_classes
        )
        adapter.save(args.out / f"adapter_{name}.npz")

        rows.append(
            {
                "task": name,
                "n_train": int(data.y_train.size),
                "probe_val": round(probe_val, 4),
                "probe_test": round(probe_test, 4),
                "adapter_val": round(adapter_val, 4),
                "adapter_test": round(adapter_test, 4),
                "adapter_params": adapter.n_params,
                "head_params": int(model.heads[name].size),
                "seconds": round(time.perf_counter() - started, 2),
            }
        )
        print(
            f"{name:<18} probe {probe_test:.3f} -> adapter {adapter_test:.3f} "
            f"(val {probe_val:.3f} -> {adapter_val:.3f})"
        )

    with (args.out / "transfer.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    info = {
        "model": str(args.model),
        "tasks": task_names,
        "rank": args.rank,
        "outers": args.outers,
        "steps": args.steps,
        "seed": args.seed,
        "spsa": SPSA_DEFAULTS,
    }
    (args.out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {args.out}/transfer.csv")


if __name__ == "__main__":
    main()
