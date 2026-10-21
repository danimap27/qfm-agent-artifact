"""The classical control keeps the substrate interface and learns from the angles."""

import numpy as np

from code.substrate.classical import load_classical_control
from tasks.registry import get_tasks


def test_classical_control_uses_angles_and_learns():
    model = load_classical_control("results/sweep_d4_b1900/model.npz", split_seed=0)
    task = get_tasks(["iris"], seed=0)["iris"]
    assert np.array_equal(model.features(task.X_test), task.X_test)
    predicted = model.head_logits("iris", model.features(task.X_test)).argmax(axis=1)
    assert (predicted == task.y_test).mean() > 0.8
