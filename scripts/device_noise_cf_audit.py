"""Counterfactual audit under shot noise and calibrated device noise.

Rebuilds the 135 counterfactual points of scripts/quantum_angle_analysis.py
(Part B), prepares each counterfactual input on the frozen circuit, and reads
the eight Pauli-Z expectations from measured bitstrings in two settings:
ideal sampling of the exact state, and an Aer emulation with the noise model
of a calibration snapshot of an IBM Heron r2 device (FakeFez). For every shot
budget the verdict of the audit is recomputed from subsampled shots and
compared with the exact verdict. The run on the physical device is not part
of this script, because it needs an active IBM Quantum account.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from code.evidence.log_builder import counterfactuals, feature_attributions, head_class_count, head_probabilities
from code.substrate.model import QFMLite

CHECKPOINT = pathlib.Path("results/sweep_d4_b1900/model.npz")
REQUESTS = pathlib.Path("results/grid_q1/qwen35/t1_table/seed0/requests.json")
SHOTS = 8192
BUDGETS = (50, 200, 1000, 4000, 8192)
REPEATS = 200

model = QFMLite.load(CHECKPOINT)
requests = json.loads(REQUESTS.read_text(encoding="utf-8"))["requests"]

# Same point set as quantum_angle_analysis.py Part B.
points = []
for item in requests:
    task, x = item["task"], np.asarray(item["features"], dtype=float)
    z = model.features(x[None, :])
    predicted = int(np.argmax(head_probabilities(model.head_logits(task, z), head_class_count(model, task))[0]))
    order = [row["index"] for row in feature_attributions(model, task, x, predicted)]
    for cf in counterfactuals(model, task, x, predicted, order, max_cf=2):
        perturbed = x.copy()
        perturbed[cf["index"]] = float(cf["to"])
        z_cf = model.features(perturbed[None, :])[0]
        logits_cf = model.head_logits(task, z_cf[None, :])[0]
        target = int(cf["flips_to_index"])
        if logits_cf.size < 2 or target >= logits_cf.size or predicted >= logits_cf.size or target == predicted:
            continue
        w = np.asarray(model.heads[task], dtype=float)
        w2 = float(np.sum((w[target] - w[predicted])[:-1] ** 2 * np.maximum(1.0 - z_cf**2, 1e-12)))
        points.append({"task": task, "x": perturbed, "z": z_cf, "exact": int(np.argmax(logits_cf)),
                       "margin": float(abs(logits_cf[target] - logits_cf[predicted])), "w2": w2})
print(f"counterfactual points: {len(points)}")

circuits = []
for p in points:
    bound = model.circuit.assign_parameters(
        {**dict(zip(model.x_params, p["x"])), **dict(zip(model.theta_params, model.theta))}
    )
    bound.measure_all()
    circuits.append(bound)

# Sanity: the circuit reproduces the fast path exactly.
check = Statevector(circuits[0].remove_final_measurements(inplace=False)).probabilities()
z_check = np.array([1.0 - 2.0 * check[((np.arange(check.size) >> q) & 1) == 1].sum() for q in range(8)])
assert np.allclose(z_check, points[0]["z"], atol=1e-9), "circuit and fast path disagree"


def bits_to_matrix(memory: list[str]) -> np.ndarray:
    """Shots x qubits matrix of measured bits, qubit 0 is the last character."""
    return np.array([[int(b) for b in reversed(s)] for s in memory], dtype=np.int8)


def wrong_rate(bits: np.ndarray, point: dict, budget: int, rng: np.random.Generator) -> float:
    wrong = 0
    for _ in range(REPEATS):
        pick = bits[rng.choice(bits.shape[0], budget, replace=False)]
        z_hat = 1.0 - 2.0 * pick.mean(axis=0)
        wrong += int(np.argmax(model.head_logits(point["task"], z_hat[None, :])[0]) != point["exact"])
    return wrong / REPEATS


rng = np.random.default_rng(0)
ideal = AerSimulator(method="statevector", seed_simulator=1)
backend = FakeFez()
# density matrix: one exact noisy evolution per circuit, then cheap sampling
noisy = AerSimulator.from_backend(backend, method="density_matrix", seed_simulator=1)
transpiled = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=7).run(circuits)
two_qubit = [sum(1 for inst in c.data if inst.operation.num_qubits == 2) for c in transpiled]
print(f"transpiled to {backend.name}: two-qubit gates median={np.median(two_qubit):.0f} max={max(two_qubit)}")

results = {}
for label, sim, batch in (("ideal", ideal, circuits), ("device_noise", noisy, transpiled)):
    memories = sim.run(batch, shots=SHOTS, memory=True).result()
    per_point = [bits_to_matrix(memories.get_memory(i)) for i in range(len(points))]
    shrink = np.mean([np.abs(1.0 - 2.0 * b.mean(axis=0)).sum() / max(np.abs(p["z"]).sum(), 1e-12) for b, p in zip(per_point, points)])
    curve = {b: float(np.mean([wrong_rate(bits, p, b, rng) for bits, p in zip(per_point, points)])) for b in BUDGETS}
    results[label] = {"wrong_verdict_rate": curve, "z_magnitude_ratio": float(shrink)}
    print(f"{label:13s} |z| ratio={shrink:.3f}  " + "  ".join(f"{b}:{r:.3f}" for b, r in curve.items()))

analytic = {}
for b in BUDGETS:
    # one-sided pairwise approximation, P = Phi(-m / sd), as in quantum_angle_analysis.py
    rates = [0.5 * math.erfc(p["margin"] / (math.sqrt(p["w2"] / b) * math.sqrt(2.0))) for p in points]
    analytic[b] = float(np.mean(rates))
print("analytic      " + "  ".join(f"{b}:{r:.3f}" for b, r in analytic.items()))

out = pathlib.Path("results/post_freeze/device_noise_cf.json")
out.write_text(json.dumps({
    "backend_snapshot": backend.name,
    "calibration_date": str(backend.properties().last_update_date),
    "n_points": len(points), "shots": SHOTS, "repeats": REPEATS,
    "two_qubit_gates_median": float(np.median(two_qubit)), "two_qubit_gates_max": int(max(two_qubit)),
    "analytic_shot_noise": analytic, **results,
}, indent=2), encoding="utf-8")
print(f"written {out}")
