"""Unit tests for the QFM-lite substrate."""

import numpy as np

from code.substrate.circuit import build_circuit, reference_z_expectations
from code.substrate.model import QFMLite, SubstrateConfig, cross_entropy, head_probabilities
from code.substrate.spsa import SPSAConfig, run_spsa


def test_parameter_counts():
    circuit, x, theta = build_circuit(4, 2, 4)
    assert len(x) == 4
    assert len(theta) == 16
    assert len(circuit.parameters) == 20


def test_fast_path_matches_gate_level_simulation():
    n_qubits, depth, n_features = 3, 2, 3
    model = QFMLite(SubstrateConfig(n_qubits=n_qubits, depth=depth, n_features=n_features, seed=3))
    rng = np.random.default_rng(7)
    model.set_theta(rng.uniform(-1.0, 1.0, size=model.theta.size))
    X = rng.uniform(0.0, np.pi, size=(6, n_features))

    z_fast = model.measure_z(model.encode(X))
    z_reference = np.stack(
        [
            reference_z_expectations(
                model.circuit, model.x_params, model.theta_params, row, model.theta, n_qubits
            )
            for row in X
        ]
    )
    np.testing.assert_allclose(z_fast, z_reference, atol=1e-10)


def test_encoding_is_deterministic_and_shaped():
    model = QFMLite(SubstrateConfig(n_qubits=3, depth=1, n_features=3, seed=0))
    X = np.linspace(0.0, 3.14, 15).reshape(5, 3)
    psi = model.encode(X)
    assert psi.shape == (8, 5)
    np.testing.assert_allclose(psi, model.encode(X))
    assert np.allclose(np.sum(np.abs(psi) ** 2, axis=0), 1.0)


def test_theta_shape_guard():
    model = QFMLite(SubstrateConfig(n_qubits=3, depth=1, n_features=3, seed=0))
    try:
        model.set_theta(np.zeros(3))
    except ValueError:
        return
    raise AssertionError("expected a ValueError for the wrong parameter count")


def test_head_probabilities_and_cross_entropy():
    binary_logits = np.array([[-2.0], [2.0]])
    probabilities = head_probabilities(binary_logits, 2)
    assert probabilities.shape == (2, 2)
    assert probabilities[0, 0] > probabilities[0, 1]
    assert cross_entropy(probabilities, np.array([0, 1])) < 0.15

    multiclass = head_probabilities(np.array([[1.0, 0.0, -1.0]]), 3)
    assert multiclass.shape == (1, 3)
    np.testing.assert_allclose(multiclass.sum(axis=1), 1.0)


def test_spsa_minimizes_quadratic():
    target = np.array([0.3, -0.7, 1.1])
    initial_loss = float(np.sum(target**2))

    def loss(theta):
        return float(np.sum((np.asarray(theta) - target) ** 2))

    config = SPSAConfig(steps=150, a=0.25, c=0.1, seed=1)
    result = run_spsa(loss, np.zeros(3), config)
    assert len(result["history"]) == 150
    assert result["final_loss"] == result["history"][-1]["loss"]
    assert result["final_loss"] < 0.25 * initial_loss
    assert np.allclose(result["theta"], target, atol=0.4)


def test_model_save_load_roundtrip(tmp_path):
    model = QFMLite(SubstrateConfig(n_qubits=3, depth=1, n_features=3, seed=5))
    model.set_head("task_a", np.zeros((2, 4)))
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = QFMLite.load(path)
    assert loaded.config == model.config
    np.testing.assert_allclose(loaded.theta, model.theta)
    assert set(loaded.heads) == {"task_a"}
    np.testing.assert_allclose(loaded.heads["task_a"], model.heads["task_a"])
