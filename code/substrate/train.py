"""Multitask pretraining of the QFM-lite substrate (F1).

Alternated optimization: per-task logistic heads are refit on the current
features between SPSA blocks, and each SPSA block updates the circuit
parameters against the multitask cross-entropy with the heads held fixed.
Everything is deterministic given the seed: splits, initialization,
perturbation streams, and head fitting.

Usage:
    python -m code.substrate.train --config configs/f1_smoke.yaml --out results/f1_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

from code.substrate.model import QFMLite, SubstrateConfig, cross_entropy, head_probabilities
from code.substrate.spsa import SPSAConfig, run_spsa
from tasks.registry import get_tasks


def logistic_coef(features: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """Fit one logistic head on ``[features, 1]`` and return its coefficients."""
    augmented = np.hstack([features, np.ones((features.shape[0], 1))])
    classifier = LogisticRegression(fit_intercept=False, max_iter=2000, random_state=seed)
    classifier.fit(augmented, labels)
    return np.asarray(classifier.coef_, dtype=float)


def task_metrics(
    model: QFMLite, name: str, features: np.ndarray, labels: np.ndarray, n_classes: int
) -> tuple[float, float]:
    """Accuracy and cross-entropy of one task under the current heads."""
    logits = model.head_logits(name, features)
    probabilities = head_probabilities(logits, n_classes)
    accuracy = float(np.mean(probabilities.argmax(axis=1) == labels))
    return accuracy, cross_entropy(probabilities, labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument("--out", default=None, help="output directory override")
    parser.add_argument("--seed", type=int, default=None, help="seed override")
    parser.add_argument("--limit", type=int, default=None, help="cap samples per task")
    parser.add_argument("--tasks", default=None, help="comma-separated subset of configured tasks")
    parser.add_argument("--outers", type=int, default=None)
    parser.add_argument("--steps-per-outer", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None, help="ansatz depth override")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.depth is not None:
        config["substrate"]["depth"] = args.depth
    seed = int(config["seed"]) if args.seed is None else args.seed
    out_dir = Path(args.out or config["output"])
    out_dir.mkdir(parents=True, exist_ok=True)

    task_names = list(config["tasks"])
    if args.tasks:
        task_names = [name.strip() for name in args.tasks.split(",") if name.strip()]

    limits = {name: value for name, value in config.get("limits", {}).items() if name in task_names}
    if args.limit is not None:
        limits = {name: args.limit for name in task_names}

    datasets = get_tasks(task_names, seed=seed, limits=limits)
    substrate = SubstrateConfig(seed=seed, **config["substrate"])
    model = QFMLite(substrate)
    encodings = {
        name: (model.encode(dataset.X_train), model.encode(dataset.X_val), model.encode(dataset.X_test))
        for name, dataset in datasets.items()
    }

    training = config["training"]
    outers = int(training["outers"]) if args.outers is None else args.outers
    steps = int(training["spsa_steps_per_outer"]) if args.steps_per_outer is None else args.steps_per_outer

    history = []
    started = time.time()
    for outer in range(outers):
        train_features = {name: model.measure_z(encoding[0]) for name, encoding in encodings.items()}
        for name, dataset in datasets.items():
            model.set_head(name, logistic_coef(train_features[name], dataset.y_train, seed + outer))
        train_loss = float(
            np.mean(
                [
                    task_metrics(model, name, train_features[name], dataset.y_train, dataset.n_classes)[1]
                    for name, dataset in datasets.items()
                ]
            )
        )
        val_accuracy = {
            name: task_metrics(
                model, name, model.measure_z(encodings[name][1]), dataset.y_val, dataset.n_classes
            )[0]
            for name, dataset in datasets.items()
        }
        print(
            f"[outer {outer:02d}] train_ce={train_loss:.4f} "
            + " ".join(f"{name}:{value:.3f}" for name, value in val_accuracy.items()),
            flush=True,
        )
        history.append({"outer": outer, "train_ce": train_loss, "val_accuracy": val_accuracy})

        if outer < outers - 1:

            def loss_fn(theta, _model=model, _encodings=encodings, _datasets=datasets):
                _model.set_theta(theta)
                losses = []
                for name, dataset in _datasets.items():
                    z = _model.measure_z(_encodings[name][0])
                    logits = _model.head_logits(name, z)
                    losses.append(
                        cross_entropy(head_probabilities(logits, dataset.n_classes), dataset.y_train)
                    )
                return float(np.mean(losses))

            spsa = SPSAConfig(steps=steps, seed=seed, **training["spsa"])
            result = run_spsa(
                loss_fn,
                model.theta,
                spsa,
                rng=np.random.default_rng([seed, 1000 + outer]),
                k_offset=outer * steps,
            )
            history[-1]["spsa_final_loss"] = result["final_loss"]
            model.set_theta(result["theta"])

    rows = []
    for name, dataset in datasets.items():
        accuracy_val = task_metrics(
            model, name, model.measure_z(encodings[name][1]), dataset.y_val, dataset.n_classes
        )[0]
        accuracy_test, loss_test = task_metrics(
            model, name, model.measure_z(encodings[name][2]), dataset.y_test, dataset.n_classes
        )
        rows.append(
            {
                "task": name,
                "family": dataset.family,
                "n_train": int(len(dataset.y_train)),
                "n_val": int(len(dataset.y_val)),
                "n_test": int(len(dataset.y_test)),
                "n_features_raw": dataset.n_features_raw,
                "n_classes": dataset.n_classes,
                "val_accuracy": round(accuracy_val, 4),
                "test_accuracy": round(accuracy_test, 4),
                "test_ce": round(loss_test, 4),
            }
        )

    with open(out_dir / "metrics.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - started
    run_info = {
        "seed": seed,
        "outers": outers,
        "steps_per_outer": steps,
        "tasks": list(datasets),
        "n_qubits": substrate.n_qubits,
        "depth": substrate.depth,
        "elapsed_seconds": round(elapsed, 1),
        "qiskit_version": version("qiskit"),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    model.save(out_dir / "model.npz")

    print()
    for row in rows:
        print(
            f"[final] {row['task']:<14} val={row['val_accuracy']:.3f} "
            f"test={row['test_accuracy']:.3f} ce={row['test_ce']:.4f}"
        )
    print(f"[done] {elapsed:.1f}s, outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
