"""Regression checks for the audit fixes: Shapley, value audit, citation whitelist."""

import numpy as np

from code.eval.value_audit import check_value_claims
from code.evidence.shapley import shapley_values

LOG = {
    "prediction": {"class_index": 1, "probability": 0.85},
    "attributions": [{"index": 2, "delta": 0.4, "rank": 1}],
    "counterfactuals": [],
}


def test_shapley_efficiency():
    # sum of Shapley values == v(full) - v(baseline), the defining property
    w = np.array([0.3, -0.2, 0.5])
    f = lambda x: np.array([1 - np.tanh(w @ x), np.tanh(w @ x)])  # noqa: E731
    x, base = np.array([1.0, 2.0, 0.5]), np.zeros(3)
    phi = shapley_values(f, x, 1, base)
    assert np.isclose(phi.sum(), f(x)[1] - f(base)[1])


def test_value_audit_reads_percent_as_fraction():
    claim = check_value_claims("The model is 85% confident.", LOG)["claims"][0]
    assert claim["value"] == 0.85 and claim["match_true"]


def test_value_audit_prominence_not_vacuous_on_occluded_log():
    occluded = {**LOG, "attributions": []}
    result = check_value_claims("Feature 3 is the most important.", LOG, occluded)
    assert result["claims"][0]["match_true"] and not result["claims"][0]["match_shown"]
