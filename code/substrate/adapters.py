"""Residual low-rank adapters for transfer to unseen tasks.

The adapter follows the Q-LoRA recipe: the shared substrate stays frozen and
a small trainable correction is added to the input features. The correction
is a rank-constrained map whose output projection starts at zero, so the
adapted model starts exactly at the frozen model and moves away only when
training pays for it. This keeps the parameter budget of a new task at
``2 * rank * n_features`` plus the linear head.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AdapterConfig:
    """Shape of a residual adapter."""

    rank: int = 4
    init_scale: float = 0.5


class ResidualAdapter:
    """Rank-constrained additive correction on the input features."""

    def __init__(
        self,
        n_features: int,
        config: AdapterConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.n_features = int(n_features)
        self.config = config or AdapterConfig()
        rng = np.random.default_rng(seed)
        self.w_in = rng.normal(0.0, self.config.init_scale, size=(self.config.rank, self.n_features))
        self.w_out = np.zeros((self.n_features, self.config.rank))

    @property
    def n_params(self) -> int:
        return self.w_in.size + self.w_out.size

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Return ``X + tanh(X @ W_in.T) @ W_out.T``."""
        X = np.asarray(X, dtype=float)
        hidden = np.tanh(X @ self.w_in.T)
        return X + hidden @ self.w_out.T

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.w_in.ravel(), self.w_out.ravel()])

    def set_params(self, params: np.ndarray) -> None:
        params = np.asarray(params, dtype=float)
        split = self.w_in.size
        self.w_in = params[:split].reshape(self.w_in.shape).copy()
        self.w_out = params[split:].reshape(self.w_out.shape).copy()

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            w_in=self.w_in,
            w_out=self.w_out,
            rank=self.config.rank,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ResidualAdapter":
        data = np.load(Path(path))
        adapter = cls(
            n_features=int(data["w_in"].shape[1]),
            config=AdapterConfig(rank=int(data["rank"])),
        )
        adapter.w_in = np.asarray(data["w_in"], dtype=float)
        adapter.w_out = np.asarray(data["w_out"], dtype=float)
        return adapter
