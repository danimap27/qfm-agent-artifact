"""QFMLite: the shared variational substrate with per-task linear heads.

The forward path is a dense-matrix formulation that is exact for this
circuit family. The encoding is a product of per-qubit RY rotations, so the
encoded state is a Kronecker product that can be built per sample, and the
ansatz is a fixed unitary for a given parameter vector. The equivalence with
the gate-level statevector simulation is covered by the test suite, which
keeps the fast path honest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Operator

from code.substrate.circuit import build_ansatz_operator, build_circuit


@dataclass
class SubstrateConfig:
    """Substrate hyperparameters."""

    n_qubits: int = 8
    depth: int = 2
    n_features: int = 8
    seed: int = 0


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def head_probabilities(logits: np.ndarray, n_classes: int) -> np.ndarray:
    """Class probabilities from head logits (a single logit for binary tasks)."""
    if n_classes == 2 and logits.shape[1] == 1:
        p_one = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        return np.stack([1.0 - p_one, p_one], axis=1)
    return softmax(logits)


def cross_entropy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean cross-entropy of predicted probabilities against labels."""
    picked = probabilities[np.arange(labels.shape[0]), labels]
    return float(-np.mean(np.log(picked + 1e-12)))


class QFMLite:
    """Shared circuit plus one logistic head per task."""

    def __init__(self, config: SubstrateConfig) -> None:
        if config.n_features != config.n_qubits:
            raise ValueError("the fast path requires n_features == n_qubits")
        self.config = config
        self.circuit, self.x_params, self.theta_params = build_circuit(
            config.n_qubits, config.depth, config.n_features
        )
        self.ansatz, self._ansatz_params = build_ansatz_operator(config.n_qubits, config.depth)
        rng = np.random.default_rng(config.seed)
        self.theta = rng.uniform(-np.pi / 4.0, np.pi / 4.0, size=len(self.theta_params))
        self.heads: dict[str, np.ndarray] = {}
        self._unitary: np.ndarray | None = None
        self._p_one_masks = np.stack(
            [(np.arange(2**config.n_qubits) >> qubit) & 1 for qubit in range(config.n_qubits)]
        )

    def set_theta(self, theta: np.ndarray) -> None:
        """Replace the circuit parameters and invalidate the cached unitary."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (len(self.theta_params),):
            raise ValueError(f"expected {len(self.theta_params)} parameters, got shape {theta.shape}")
        self.theta = theta.copy()
        self._unitary = None

    def unitary(self) -> np.ndarray:
        """Dense unitary of the ansatz for the current parameters."""
        if self._unitary is None:
            bound = self.ansatz.assign_parameters(
                {parameter: value for parameter, value in zip(self._ansatz_params, self.theta)}
            )
            self._unitary = Operator(bound).data
        return self._unitary

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encoded states as a matrix of shape (2**n_qubits, n_samples)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.config.n_features:
            raise ValueError(f"expected features with shape (n, {self.config.n_features}), got {X.shape}")
        samples = X.shape[0]
        psi = np.ones((1, samples), dtype=complex)
        for qubit in range(self.config.n_qubits - 1, -1, -1):
            feature = X[:, qubit]
            rotation = np.stack([np.cos(feature / 2.0), np.sin(feature / 2.0)], axis=0)
            psi = (psi[:, None, :] * rotation[None, :, :]).reshape(psi.shape[0] * 2, samples)
        return psi

    def measure_z(self, psi: np.ndarray) -> np.ndarray:
        """Per-qubit <Z> for encoded states, shape (n_samples, n_qubits)."""
        amplitudes = self.unitary() @ psi
        probabilities = np.abs(amplitudes) ** 2
        p_one = self._p_one_masks @ probabilities
        return (1.0 - 2.0 * p_one).T

    def features(self, X: np.ndarray) -> np.ndarray:
        """Convenience path: encode and measure in one call."""
        return self.measure_z(self.encode(X))

    def set_head(self, task: str, coef: np.ndarray) -> None:
        """Store the coefficient matrix of one task head."""
        self.heads[task] = np.asarray(coef, dtype=float)

    def head_logits(self, task: str, z: np.ndarray) -> np.ndarray:
        """Logits of one task head for a batch of <Z> vectors."""
        augmented = np.hstack([z, np.ones((z.shape[0], 1))])
        return augmented @ self.heads[task].T

    def save(self, path: Path) -> None:
        """Persist parameters, heads, and configuration to a single npz file."""
        payload = {
            "config": np.array(json.dumps(asdict(self.config))),
            "theta": self.theta,
        }
        for task, coef in self.heads.items():
            payload[f"head::{task}"] = coef
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "QFMLite":
        """Rebuild a model from an npz file produced by ``save``."""
        with np.load(path, allow_pickle=False) as data:
            config = SubstrateConfig(**json.loads(str(data["config"])))
            model = cls(config)
            model.set_theta(data["theta"])
            for key in data.files:
                if key.startswith("head::"):
                    model.set_head(key.split("head::", 1)[1], data[key])
        return model
