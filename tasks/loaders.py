"""Task loaders for the QFM-lite substrate.

Every task returns the same envelope (TaskData): fixed 60/20/20 stratified
splits, features reduced to ``reduce_dim`` dimensions and scaled to [0, pi]
for angle encoding. Splits are deterministic per (task name, global seed),
independent of the order in which tasks are requested. OpenML-backed tasks
download once and cache a compact npz copy under ``data/cache`` so later
runs and offline sessions reuse it.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATA_CACHE_DIR = Path("data/cache")


@dataclass
class TaskSpec:
    """Declarative definition of one substrate task."""

    name: str
    family: str
    loader: Callable[["TaskSpec", int, Optional[int]], "TaskData"]
    reduce_dim: int = 8
    pca: bool = True
    note: str = ""


@dataclass
class TaskData:
    """Splits and metadata for one task, ready for the substrate."""

    name: str
    family: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    n_features_raw: int
    meta: Dict[str, object] = field(default_factory=dict)


def task_rng(seed: int, name: str) -> np.random.Generator:
    """Per-task generator, independent of the task list order."""
    return np.random.default_rng([seed, zlib.crc32(name.encode("utf-8"))])


def _stratify(labels: np.ndarray) -> Optional[np.ndarray]:
    """Return the labels for stratified splitting when every class allows it."""
    counts = np.bincount(labels)
    return labels if counts[counts > 0].min() >= 3 else None


def _prepare_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    spec: TaskSpec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Standardize, optionally reduce with PCA, pad, and scale to [0, pi]."""
    raw_features = X_train.shape[1]
    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = (scaler.transform(block) for block in (X_train, X_val, X_test))

    meta: Dict[str, object] = {"raw_features": raw_features}
    if spec.pca and raw_features > spec.reduce_dim:
        reducer = PCA(n_components=spec.reduce_dim, random_state=seed).fit(X_train)
        X_train, X_val, X_test = (reducer.transform(block) for block in (X_train, X_val, X_test))
        meta["explained_variance"] = float(reducer.explained_variance_ratio_.sum())

    dim = spec.reduce_dim
    padded = []
    for block in (X_train, X_val, X_test):
        if block.shape[1] < dim:
            block = np.hstack([block, np.zeros((block.shape[0], dim - block.shape[1]))])
        padded.append(block[:, :dim])
    X_train, X_val, X_test = padded

    angle = MinMaxScaler(feature_range=(0.0, np.pi)).fit(X_train)
    X_train, X_val, X_test = (
        np.clip(angle.transform(block), 0.0, np.pi) for block in (X_train, X_val, X_test)
    )
    return X_train, X_val, X_test, meta


def _build(
    spec: TaskSpec,
    seed: int,
    limit: Optional[int],
    X: np.ndarray,
    y: np.ndarray,
) -> TaskData:
    """Common pipeline: optional stratified cap, splits, feature preparation."""
    rng = task_rng(seed, spec.name)
    if limit is not None and limit < len(y):
        X, _, y, _ = train_test_split(
            X, y, train_size=limit, stratify=_stratify(y), random_state=int(rng.integers(0, 2**31 - 1))
        )

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=_stratify(y), random_state=int(rng.integers(0, 2**31 - 1))
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=_stratify(y_rest), random_state=int(rng.integers(0, 2**31 - 1))
    )

    X_train, X_val, X_test, meta = _prepare_features(X_train, X_val, X_test, spec, seed)
    meta["limit"] = limit
    return TaskData(
        name=spec.name,
        family=spec.family,
        X_train=X_train,
        y_train=y_train.astype(int),
        X_val=X_val,
        y_val=y_val.astype(int),
        X_test=X_test,
        y_test=y_test.astype(int),
        n_classes=int(len(np.unique(y))),
        n_features_raw=int(meta["raw_features"]),
        meta=meta,
    )


def sklearn_tabular(fetcher: Callable[[], object]) -> Callable[[TaskSpec, int, Optional[int]], TaskData]:
    """Build a loader around an sklearn dataset fetcher."""

    def load(spec: TaskSpec, seed: int, limit: Optional[int] = None) -> TaskData:
        dataset = fetcher()
        X = np.asarray(dataset.data, dtype=float)  # type: ignore[attr-defined]
        y = np.asarray(dataset.target, dtype=int)  # type: ignore[attr-defined]
        return _build(spec, seed, limit, X, y)

    return load


def _label_encode(targets: np.ndarray) -> np.ndarray:
    """Map arbitrary class labels to contiguous integers in sorted order."""
    values = [str(value) for value in np.asarray(targets).tolist()]
    classes = sorted(set(values))
    mapping = {value: index for index, value in enumerate(classes)}
    return np.asarray([mapping[value] for value in values], dtype=int)


def _openml_raw(name: str, version: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Fetch one OpenML dataset as (X, y) with integer labels, no npz cache."""
    from sklearn.datasets import fetch_openml

    dataset = fetch_openml(name=name, version=version, as_frame=False, parser="liac-arff")
    X = np.asarray(dataset.data, dtype=float)
    y = _label_encode(np.asarray(dataset.target))
    return X, y


def openml_tabular(
    name: str, version: int = 1
) -> Callable[[TaskSpec, int, Optional[int]], TaskData]:
    """Build a loader around an OpenML tabular dataset with an npz cache."""

    def load(spec: TaskSpec, seed: int, limit: Optional[int] = None) -> TaskData:
        cache_path = DATA_CACHE_DIR / f"{name.replace('/', '_')}.npz"
        if cache_path.exists():
            with np.load(cache_path) as data:
                X = np.asarray(data["X"], dtype=float)
                y = np.asarray(data["y"], dtype=int)
        else:
            X, y = _openml_raw(name, version)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, X=X.astype(np.float32), y=y)
        return _build(spec, seed, limit, X, y)

    return load


def _average_pool(images: np.ndarray, side: int, factor: int) -> np.ndarray:
    """Average ``factor x factor`` blocks of flattened square images."""
    grid = images.reshape(-1, side, side)
    pooled = np.zeros((grid.shape[0], side // factor, side // factor))
    for row in range(factor):
        for column in range(factor):
            pooled += grid[:, row::factor, column::factor]
    return (pooled / (factor * factor)).reshape(images.shape[0], -1)


def openml_image(
    name: str,
    version: int = 1,
    subset: int = 3000,
    side: int = 28,
    pool: int = 2,
) -> Callable[[TaskSpec, int, Optional[int]], TaskData]:
    """Build a loader for an OpenML image set, pooled and cached as a subset.

    The subset selection uses a fixed internal seed so the cached frames are
    reproducible across machines. Pooling happens before caching, which keeps
    the cache at a few megabytes instead of the full raw set.
    """

    def load(spec: TaskSpec, seed: int, limit: Optional[int] = None) -> TaskData:
        cache_path = DATA_CACHE_DIR / f"{name.lower().replace('/', '_')}-p{pool}-n{subset}.npz"
        if cache_path.exists():
            with np.load(cache_path) as data:
                X = np.asarray(data["X"], dtype=float)
                y = np.asarray(data["y"], dtype=int)
        else:
            X_raw, y_raw = _openml_raw(name, version)
            if subset < y_raw.size:
                X_raw, _, y_raw, _ = train_test_split(
                    X_raw, y_raw, train_size=subset, stratify=y_raw, random_state=0
                )
            X = _average_pool(X_raw, side=side, factor=pool)
            y = np.asarray(y_raw, dtype=int)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, X=X.astype(np.float32), y=y)
        return _build(spec, seed, limit, X, y)

    return load


def make_periodic_sin(spec: TaskSpec, seed: int, limit: Optional[int] = None) -> TaskData:
    """Synthetic task whose label is the sign of a periodic signal.

    The label is the sign of one full sine period, so the only structure it
    requires is the first harmonic. This detail matters for the substrate: an
    expectation measured after angle encoding is linear in the encoded state,
    so a single RY rotation per feature can represent first harmonics only.
    A label built on higher harmonics (for example the sign of sin(2x)),
    whose overlap with the first harmonic is exactly zero, is unreachable by
    construction and the substrate sits at chance level, which is what an
    earlier version of this task measured. Keep periodic tasks inside the
    encoding band or widen the encoding explicitly.
    """
    n_samples = limit or 600
    rng = task_rng(seed, spec.name)
    informative = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    labels = (np.sin(informative) > 0.0).astype(int)
    noise = rng.normal(0.0, 1.0, size=(n_samples, spec.reduce_dim - 1))
    X = np.hstack([informative.reshape(-1, 1), noise])
    return _build(spec, seed, None, X, labels)
