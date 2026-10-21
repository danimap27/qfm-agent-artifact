"""Tests for the decision-layer battery statistics."""

from __future__ import annotations

import numpy as np

from code.router.router_stats import cluster_bootstrap_accuracy, cohens_h, holm, mcnemar_exact, wilson_interval


def test_wilson_interval_bounds() -> None:
    low, high = wilson_interval(48, 52)
    assert 0.80 < low < 0.923 < high < 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_mcnemar_exact_known_values() -> None:
    assert mcnemar_exact(0, 6) == 0.03125
    assert mcnemar_exact(1, 1) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 12) == mcnemar_exact(12, 5)


def test_holm_is_monotone_and_capped() -> None:
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert holm([0.9, 0.1]) == [0.9, 0.2]
    assert holm([0.6, 0.6]) == [1.0, 1.0]


def test_cohens_h_is_antisymmetric() -> None:
    assert cohens_h(0.9, 0.8) > 0
    assert abs(cohens_h(0.9, 0.8) + cohens_h(0.8, 0.9)) < 1e-12


def test_cluster_bootstrap_stays_in_range() -> None:
    correct = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    tasks = ["a", "a", "b", "b", "c", "c", "d", "d"]
    low, high = cluster_bootstrap_accuracy(correct, tasks, seed=0, draws=300)
    assert 0.0 <= low <= high <= 1.0
    assert low <= correct.mean() <= high
