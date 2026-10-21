"""Classical control: the same task heads fitted on the encoded angles directly.

The control keeps every interface of ``QFMLite`` so the evidence layer, the
verbalizer, and the scorer run unchanged, and only removes the circuit: the
features are the eight encoded angles instead of the eight Pauli-Z
expectations, and each head is the same intercept-free logistic regression
over ``[features, 1]`` that the substrate training uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from code.substrate.model import QFMLite
from code.substrate.train import logistic_coef
from tasks.registry import get_tasks


class ClassicalControl(QFMLite):
    """``QFMLite`` with identity features, so the heads see the angles."""

    def features(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)


def load_classical_control(checkpoint: str | Path, split_seed: int, seed: int = 0) -> ClassicalControl:
    """Refit every head of the checkpoint on the raw encoded angles."""
    model = ClassicalControl.load(Path(checkpoint))
    for name, task in get_tasks(sorted(model.heads), seed=split_seed).items():
        model.set_head(name, logistic_coef(task.X_train, task.y_train, seed))
    return model
