"""Unit tests for the task loaders and the registry."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from tasks.registry import get_tasks


def _total(dataset):
    return len(dataset.y_train) + len(dataset.y_val) + len(dataset.y_test)


def test_iris_shapes_and_range():
    dataset = get_tasks(["iris"], seed=0)["iris"]
    assert dataset.X_train.shape[1] == 8
    assert dataset.n_classes == 3
    assert dataset.n_features_raw == 4
    assert _total(dataset) == 150
    for block in (dataset.X_train, dataset.X_val, dataset.X_test):
        assert block.shape[1] == 8
        assert block.min() >= -1e-9
        assert block.max() <= np.pi + 1e-9


def test_digits_limit_caps_total():
    dataset = get_tasks(["digits"], seed=0, limits={"digits": 500})["digits"]
    assert _total(dataset) == 500


def test_periodic_task_is_learnable():
    dataset = get_tasks(["periodic_sin"], seed=0, limits={"periodic_sin": 400})["periodic_sin"]
    classifier = DecisionTreeClassifier(max_depth=4, random_state=0)
    classifier.fit(dataset.X_train, dataset.y_train)
    assert classifier.score(dataset.X_train, dataset.y_train) > 0.9


def test_splits_are_deterministic_per_seed():
    first = get_tasks(["wine"], seed=7)["wine"]
    second = get_tasks(["wine"], seed=7)["wine"]
    np.testing.assert_array_equal(first.X_train, second.X_train)
    np.testing.assert_array_equal(first.y_test, second.y_test)


def test_unknown_task_raises():
    try:
        get_tasks(["does_not_exist"], seed=0)
    except KeyError:
        return
    raise AssertionError("expected a KeyError for an unknown task")


def test_average_pool_downsample():
    from tasks.loaders import _average_pool

    images = np.arange(28 * 28, dtype=float).reshape(1, -1)
    pooled = _average_pool(images, side=28, factor=2)
    assert pooled.shape == (1, 196)
    assert pooled[0, 0] == np.mean([0.0, 1.0, 28.0, 29.0])


def test_openml_loader_reads_npz_cache(tmp_path, monkeypatch):
    import tasks.loaders as loaders

    monkeypatch.setattr(loaders, "DATA_CACHE_DIR", tmp_path)
    rng = np.random.default_rng(0)
    np.savez(
        tmp_path / "fake-set.npz",
        X=rng.normal(size=(60, 5)).astype(np.float32),
        y=np.array([0, 1] * 30),
    )
    spec = loaders.TaskSpec("fake", "tabular", loaders.openml_tabular("fake-set"), pca=True)
    dataset = spec.loader(spec, 0, None)
    assert dataset.X_train.shape[1] == 8
    assert _total(dataset) == 60
    assert dataset.n_classes == 2
