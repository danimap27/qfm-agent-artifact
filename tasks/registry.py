"""Registry of QFM-lite tasks.

The mix covers the three planned families with public or synthetic sources:
tabular (scikit-learn sets plus verified OpenML sets), small image (the 8x8
digits set plus MNIST and Fashion-MNIST pooled to 14x14), and periodic
(synthetic thresholded sine). The held-out tasks feed the transfer protocol
and never appear in pretraining. Dataset identifiers were verified on
2026-09-23 with live fetches; first use downloads each set once and caches a
compact npz copy under ``data/cache``.
"""

from __future__ import annotations

from typing import Dict, Optional

from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine

from tasks.loaders import (
    TaskData,
    TaskSpec,
    make_periodic_sin,
    openml_image,
    openml_tabular,
    sklearn_tabular,
)

SPECS: Dict[str, TaskSpec] = {
    "iris": TaskSpec(
        "iris", "tabular", sklearn_tabular(load_iris), pca=False, note="150 samples, 4 features, 3 classes"
    ),
    "wine": TaskSpec(
        "wine", "tabular", sklearn_tabular(load_wine), note="178 samples, 13 features, 3 classes"
    ),
    "wdbc": TaskSpec(
        "wdbc", "tabular", sklearn_tabular(load_breast_cancer), note="569 samples, 30 features, 2 classes"
    ),
    "thyroid": TaskSpec(
        "thyroid",
        "tabular",
        openml_tabular("thyroid-ann"),
        note="3772 samples, 21 features, 3 classes (OpenML thyroid-ann)",
    ),
    "parkinsons": TaskSpec(
        "parkinsons",
        "tabular",
        openml_tabular("parkinsons"),
        note="195 samples, 22 features, 2 classes (OpenML parkinsons)",
    ),
    "wine_quality": TaskSpec(
        "wine_quality",
        "tabular",
        openml_tabular("wine-quality-red"),
        note="1599 samples, 11 features, 6 classes (OpenML wine-quality-red)",
    ),
    "sonar": TaskSpec(
        "sonar",
        "tabular",
        openml_tabular("sonar"),
        note="208 samples, 60 features, 2 classes (OpenML sonar)",
    ),
    "ionosphere": TaskSpec(
        "ionosphere",
        "tabular",
        openml_tabular("ionosphere"),
        note="351 samples, 34 features, 2 classes, held out (OpenML ionosphere)",
    ),
    "blood_transfusion": TaskSpec(
        "blood_transfusion",
        "tabular",
        openml_tabular("blood-transfusion-service-center"),
        note="748 samples, 4 features, 2 classes, held out (OpenML blood-transfusion-service-center)",
    ),
    "digits": TaskSpec(
        "digits", "image", sklearn_tabular(load_digits), note="8x8 digit images, MNIST-like stand-in"
    ),
    "mnist8": TaskSpec(
        "mnist8",
        "image",
        openml_image("mnist_784", subset=3000),
        note="MNIST pooled 28x28 to 14x14, 3000-sample stratified subset",
    ),
    "fashion8": TaskSpec(
        "fashion8",
        "image",
        openml_image("Fashion-MNIST", subset=3000),
        note="Fashion-MNIST pooled 28x28 to 14x14, 3000-sample stratified subset",
    ),
    "periodic_sin": TaskSpec(
        "periodic_sin", "periodic", make_periodic_sin, pca=False, note="synthetic thresholded sine"
    ),
}

PRETRAIN_MIX = [
    "iris",
    "wine",
    "wdbc",
    "thyroid",
    "parkinsons",
    "wine_quality",
    "sonar",
    "digits",
    "mnist8",
    "fashion8",
    "periodic_sin",
]

HELD_OUT = ["ionosphere", "blood_transfusion"]


def get_tasks(
    names: list[str], seed: int, limits: Optional[Dict[str, int]] = None
) -> Dict[str, TaskData]:
    """Load the requested tasks with per-task sample caps where given."""
    limits = limits or {}
    return {name: SPECS[name].loader(SPECS[name], seed, limits.get(name)) for name in names}
