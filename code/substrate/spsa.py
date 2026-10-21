"""SPSA optimizer for the substrate parameters.

The update rule follows the pattern used in the LLM-MetaOpt codebase:

    theta_{k+1} = theta_k - a_k * g_hat(theta_k)
    a_k = a / (k + 1 + A)^alpha,    c_k = c / (k + 1)^gamma
    g_hat = (L(theta + c_k * delta) - L(theta - c_k * delta)) / (2 c_k) * delta

with Rademacher perturbations delta. The loss recorded per step is the true
loss of the new iterate, not the midpoint proxy, so a decaying schedule
cannot fake progress.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List

import numpy as np


@dataclass
class SPSAConfig:
    """SPSA hyperparameters and schedules."""

    steps: int = 200
    a: float = 0.4
    c: float = 0.12
    alpha: float = 0.602
    gamma: float = 0.101
    A: float = 5.0
    seed: int = 0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def schedules(config: SPSAConfig, step: int) -> tuple[float, float]:
    """Step-size and perturbation schedules at iteration ``step``."""
    a_k = config.a / (step + 1.0 + config.A) ** config.alpha
    c_k = config.c / (step + 1.0) ** config.gamma
    return a_k, c_k


def run_spsa(
    loss_fn: Callable[[np.ndarray], float],
    theta0: np.ndarray,
    config: SPSAConfig,
    rng: np.random.Generator | None = None,
    steps: int | None = None,
    k_offset: int = 0,
) -> Dict[str, object]:
    """Run SPSA from ``theta0`` and return the best-effort trajectory.

    ``k_offset`` continues the decay schedules at a global step so
    consecutive blocks share one trajectory instead of restarting.
    """
    rng = np.random.default_rng(config.seed) if rng is None else rng
    theta = np.asarray(theta0, dtype=float).copy()
    n_steps = config.steps if steps is None else int(steps)
    history: List[Dict[str, float]] = []

    for step in range(n_steps):
        a_k, c_k = schedules(config, k_offset + step)
        delta = rng.choice([-1.0, 1.0], size=theta.size)
        loss_plus = float(loss_fn(theta + c_k * delta))
        loss_minus = float(loss_fn(theta - c_k * delta))
        gradient = (loss_plus - loss_minus) / (2.0 * c_k) * delta
        theta = theta - a_k * gradient
        loss_true = float(loss_fn(theta))
        history.append(
            {
                "step": step,
                "loss": loss_true,
                "grad_norm": float(np.linalg.norm(gradient)),
                "a_k": a_k,
                "c_k": c_k,
            }
        )

    final_loss = history[-1]["loss"] if history else float(loss_fn(theta))
    return {"theta": theta, "history": history, "final_loss": final_loss}
