"""Tests for the residual adapter and the transfer protocol."""

import numpy as np
from types import SimpleNamespace

from code.substrate.adapters import AdapterConfig, ResidualAdapter
from code.substrate.model import QFMLite, SubstrateConfig


def _tiny_model() -> QFMLite:
    return QFMLite(SubstrateConfig(n_qubits=3, depth=1, n_features=3, seed=0))


def test_adapter_starts_as_identity():
    adapter = ResidualAdapter(4, AdapterConfig(rank=2), seed=0)
    X = np.random.default_rng(0).normal(size=(6, 4))
    np.testing.assert_allclose(adapter.forward(X), X)


def test_adapter_params_round_trip(tmp_path):
    adapter = ResidualAdapter(4, AdapterConfig(rank=2), seed=0)
    params = np.random.default_rng(1).normal(size=adapter.n_params)
    adapter.set_params(params)
    np.testing.assert_allclose(adapter.get_params(), params)
    path = tmp_path / "adapter.npz"
    adapter.save(path)
    restored = ResidualAdapter.load(path)
    np.testing.assert_allclose(restored.get_params(), adapter.get_params())


def test_transfer_training_on_feature_defined_labels():
    from code.substrate.transfer import head_accuracy, train_adapter

    model = _tiny_model()
    rng = np.random.default_rng(3)
    X = rng.uniform(0.0, np.pi, size=(160, 3))
    z = model.features(X)
    y = (z[:, 0] > np.median(z[:, 0])).astype(int)
    data = SimpleNamespace(
        X_train=X[:110], y_train=y[:110], X_val=X[110:], y_val=y[110:], n_classes=2
    )
    adapter = train_adapter(model, "toy", data, AdapterConfig(rank=2), seed=0, outers=3, steps=25)
    accuracy = head_accuracy(model, "toy", model.features(adapter.forward(X[110:])), y[110:], 2)
    assert accuracy >= 0.7
    assert adapter.n_params == (2 * 3) + (3 * 2)
