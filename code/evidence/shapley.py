"""Exact Shapley attributions over the eight substrate features.

Two baseline conventions are supported because the perturbation baseline
theta = 0 is the range minimum of the angular encoding and the |0> pole of the
Bloch sphere: a zero baseline and a mean-of-data baseline. With eight features
the exact Shapley values cost 2^8 = 256 model evaluations per request, which
is trivial on the frozen substrate.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, List

import numpy as np


def shapley_values(
    predict_proba: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    target: int,
    baseline: np.ndarray,
) -> np.ndarray:
    """Exact Shapley values of the target-class probability.

    The value of a coalition is the predicted probability of the target class
    when the coalition's features keep their encoded angles and the remaining
    features take the baseline angles.
    """
    n = len(x)

    def value(mask: int) -> float:
        point = baseline.copy()
        for i in range(n):
            if mask & (1 << i):
                point[i] = x[i]
        return float(predict_proba(point)[target])

    factorial = [1.0]
    for i in range(1, n + 1):
        factorial.append(factorial[-1] * i)
    total = factorial[n]
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(len(others) + 1):
            weight = factorial[size] * factorial[n - size - 1] / total
            for subset in combinations(others, size):
                mask = 0
                for j in subset:
                    mask |= 1 << j
                phi[i] += weight * (value(mask | (1 << i)) - value(mask))
    return phi


def baseline_zero(n: int) -> np.ndarray:
    return np.zeros(n)


def baseline_midrange(n: int) -> np.ndarray:
    """The midpoint of the angular encoding range, away from the |0> pole."""
    return np.full(n, np.pi / 2.0)


def baseline_mean(samples: np.ndarray) -> np.ndarray:
    """Elementwise mean of a reference sample of encoded inputs."""
    return np.asarray(samples, dtype=float).mean(axis=0)
