"""Tests for the decision log builder."""

import numpy as np

from code.evidence.log_builder import LogBuilder, occlude_log, sabotage_log
from code.substrate.model import QFMLite, SubstrateConfig, head_probabilities


def _builder() -> LogBuilder:
    model = QFMLite(SubstrateConfig(n_qubits=8, depth=1, n_features=8, seed=0))
    rng = np.random.default_rng(0)
    for task in ("alpha", "beta"):
        model.set_head(task, rng.normal(0.0, 0.5, size=(2, 9)))
    train = {"alpha": rng.uniform(0.0, np.pi, size=(20, 8))}
    return LogBuilder(model, "tiny", train, {"alpha": 0.5})


def test_log_fields_and_shapes():
    builder = _builder()
    log = builder.build("t-1", "alpha", np.full(8, 1.0), "text", {"method": "oracle"})
    assert log["schema_version"] == "1.0"
    assert log["prediction"]["label"].startswith("class")
    assert 0.0 <= log["prediction"]["probability"] <= 1.0
    assert len(log["attributions"]) == 8
    assert [row["rank"] for row in log["attributions"]] == list(range(1, 9))
    assert log["meta"]["evidence_occluded"] == []
    assert log["meta"]["evidence_tampered"] is False


def test_counterfactuals_execute_as_recorded():
    builder = _builder()
    model = builder.model
    rng = np.random.default_rng(2)
    seen = 0
    for _ in range(5):
        x = rng.uniform(0.0, np.pi, size=8)
        log = builder.build("t-2", "alpha", x, "text", {"method": "oracle"})
        for cf in log["counterfactuals"]:
            seen += 1
            perturbed = x.copy()
            perturbed[cf["index"]] = cf["to"]
            z = model.features(perturbed[None, :])
            probabilities = head_probabilities(model.head_logits("alpha", z), 2)
            assert int(probabilities.argmax(axis=1)[0]) == cf["flips_to_index"]
    assert seen > 0


def test_sabotage_reverses_ranks():
    builder = _builder()
    log = builder.build("t-3", "alpha", np.full(8, 1.0), "text", {"method": "oracle"})
    tampered = sabotage_log(log)
    assert tampered["meta"]["evidence_tampered"] is True
    assert [row["index"] for row in tampered["attributions"]] == list(
        reversed([row["index"] for row in log["attributions"]])
    )


def test_occlusion_records_fields_and_keeps_original():
    builder = _builder()
    log = builder.build("t-4", "alpha", np.full(8, 1.0), "text", {"method": "oracle"})
    occluded = occlude_log(log, ["attributions"])
    assert occluded["attributions"] == []
    assert occluded["meta"]["evidence_occluded"] == ["attributions"]
    assert log["attributions"] != []
