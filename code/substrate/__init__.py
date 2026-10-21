"""Substrate package: the QFM-lite circuit, model, optimizer, and trainer."""

from code.substrate.circuit import build_ansatz_operator, build_circuit
from code.substrate.model import QFMLite, SubstrateConfig
from code.substrate.spsa import SPSAConfig, run_spsa

__all__ = [
    "QFMLite",
    "SPSAConfig",
    "SubstrateConfig",
    "build_ansatz_operator",
    "build_circuit",
    "run_spsa",
]
