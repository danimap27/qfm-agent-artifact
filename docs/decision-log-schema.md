# Decision log schema (v1)

The decision log is the artifact that makes grounded explanations possible and measurable. It is the interface between the pipeline's internals (routing, prediction, explainability) and the verbalizer. Three principles govern the schema.

1. Every field is verifiable. Predictions and probabilities are recomputed, attributions are recomputed by perturbation, and counterfactuals are executed against the substrate before they are scored. In simulation this verification is exact.
2. Every field is citable. Each entry carries a stable identifier so that an explanation claim can point at the exact evidence it uses.
3. The log records its own manipulation. When a condition occludes or tampers with evidence for experimental purposes, the fact and the affected fields are recorded in the metadata, so no run is ambiguous after the fact.

## Example

```json
{
  "schema_version": "1.0",
  "request_id": "req-000123",
  "request": {
    "text": "Is this thyroid panel consistent with disease recurrence?",
    "task_hint": null
  },
  "routing": {
    "method": "typed_router",
    "decision": "thyroid_panel",
    "head": "head_thyroid",
    "probabilities": {"thyroid_panel": 0.87, "generic_tabular": 0.09, "abstain": 0.04},
    "confidence": 0.87,
    "abstained": false
  },
  "prediction": {
    "label": "recurrence",
    "probability": 0.82,
    "calibrated": true,
    "head_accuracy_on_task": 0.79
  },
  "attributions": [
    {"feature": "tsh", "rank": 1, "delta": 0.31, "method": "perturbation"},
    {"feature": "tumor_size", "rank": 2, "delta": 0.19, "method": "perturbation"}
  ],
  "counterfactuals": [
    {"id": "cf-1", "feature": "tsh", "from": 1.20, "to": 0.40, "flips_to": "benign", "verified": true}
  ],
  "ood": {"flag": false, "score": 0.12, "method": "distance_to_training_manifold"},
  "meta": {
    "substrate_checkpoint": "ckpt-08q-multitask-seed0",
    "calibration": "isotonic-on-validation",
    "evidence_occluded": [],
    "evidence_tampered": false,
    "created": "2026-09-23T12:00:00+02:00"
  }
}
```

## Fields

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | Bumped on any breaking change. |
| `request_id` | string | Stable identifier, one log per request. |
| `request.text` | string | The natural language request, verbatim. |
| `routing.method` | string | `typed_router`, `oracle`, `uniform`, or `single_block`. |
| `routing.decision` | string | Selected task. |
| `routing.probabilities` | object | Calibrated probabilities over tasks plus `abstain`. |
| `routing.abstained` | boolean | True when the router declines to decide. |
| `prediction.label` | string | Predicted class of the selected task. |
| `prediction.probability` | float | Calibrated probability of the predicted class. |
| `attributions[].delta` | float | Prediction change under perturbation of the feature. |
| `counterfactuals[].verified` | boolean | Set by executing the change against the substrate. |
| `ood.flag` | boolean | Out-of-distribution flag for the request. |
| `meta.evidence_occluded` | list | Fields removed for the occlusion conditions. |
| `meta.evidence_tampered` | boolean | True for the sabotage condition. |

## Claim citation format

A grounded explanation cites fields by identifier, for example `[prediction.probability]`, `[attributions.rank1]`, or `[cf-1]`. The claim scorer parses these citations, checks that each cited field exists, and compares the claim against the field's value and semantics. Uncited claims and contradicted claims both count as unsupported.
