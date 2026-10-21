"""QFM-lite variational circuit.

Angle encoding with one RY rotation per feature followed by a
hardware-efficient ansatz of RY and RZ rotations and a ring of controlled-X
gates. The circuit carries no measurements, so it serves both as the
state-preparation reference path and, without the encoding block, as the
fixed unitary of the fast forward path in ``model.py``.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


def build_circuit(
    n_qubits: int, depth: int, n_features: int
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Return the parameterized circuit plus its encoding and ansatz vectors."""
    if depth < 1:
        raise ValueError("depth must be at least 1")
    x = ParameterVector("x", n_features)
    theta = ParameterVector("theta", 2 * n_qubits * depth)

    circuit = QuantumCircuit(n_qubits)
    for index in range(n_features):
        circuit.ry(x[index], index % n_qubits)
    _append_ansatz(circuit, theta, n_qubits, depth)
    return circuit, x, theta


def build_ansatz_operator(n_qubits: int, depth: int) -> tuple[QuantumCircuit, ParameterVector]:
    """Ansatz-only parameterized circuit used by the dense fast path."""
    theta = ParameterVector("theta", 2 * n_qubits * depth)
    circuit = QuantumCircuit(n_qubits)
    _append_ansatz(circuit, theta, n_qubits, depth)
    return circuit, theta


def _append_ansatz(circuit: QuantumCircuit, theta: ParameterVector, n_qubits: int, depth: int) -> None:
    cursor = 0
    for _ in range(depth):
        for qubit in range(n_qubits):
            circuit.ry(theta[cursor], qubit)
            cursor += 1
            circuit.rz(theta[cursor], qubit)
            cursor += 1
        for qubit in range(n_qubits):
            circuit.cx(qubit, (qubit + 1) % n_qubits)


def reference_z_expectations(
    circuit: QuantumCircuit,
    x_params: ParameterVector,
    theta_params: ParameterVector,
    x_values: np.ndarray,
    theta_values: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """Per-qubit <Z> from a gate-level statevector simulation.

    This is the slow, trusted path. The dense formulation in ``model.py``
    reproduces it exactly and the test suite compares both on random inputs.
    """
    assignments = {parameter: float(value) for parameter, value in zip(x_params, x_values)}
    assignments.update({parameter: float(value) for parameter, value in zip(theta_params, theta_values)})
    state = Statevector.from_instruction(circuit.assign_parameters(assignments))
    probabilities = np.abs(state.data) ** 2
    indices = np.arange(2**n_qubits)
    return np.array(
        [
            1.0 - 2.0 * probabilities[((indices >> qubit) & 1) == 1].sum()
            for qubit in range(n_qubits)
        ]
    )
