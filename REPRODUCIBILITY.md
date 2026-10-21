# Reproducibility policy

The claims of this study are measurement claims, so reproducibility is part of the method, not an appendix. This document states the policy that every experiment in the repository follows.

## Determinism

- Every run records its seeds, and every comparison is paired by seed. The seed list used in the paper tables is fixed before the full experimental grid.
- The substrate is trained and evaluated in simulation with analytic expectation values. Shot noise is never allowed to impersonate an effect, and when sampling is used the number of shots is explicit in the run metadata.

## Language models

- The headline results use open weights served locally, pinned by model tag, quantization, and file hash. Hosted endpoints may appear as an upper bound and are never the main result.
- Every request runs at temperature zero with fixed sampling parameters and a fixed prompt version. Prompts are versioned with a content hash and are part of the method.
- All requests, responses, and failures are cached keyed by the full call payload (model, messages, options, thinking flag, and endpoint), and the pipeline ships a replay mode that serves the whole experiment from disk. Reviewers can reproduce every reported number without an endpoint.
- Prompt serialization excludes wall-clock fields (for example the decision log creation timestamp), so a rebuild of the same experiment produces byte-identical messages and cache keys. A full replay is verified as part of the pilot.
- Latency is measured on the machine that serves the model while the experiment runs, and reported as median and p95 together with the serving configuration.
- Hosted endpoints are declared not bit-reproducible even at temperature zero. This is stated wherever a hosted model appears in a table.

## Budgets and fairness

- All conditions run with identical budgets on identical requests. A call that changed nothing is counted as a call.
- Failed calls are charged: their latency is accumulated and their count is reported next to every latency number.
- No hyperparameter is tuned on the evaluation split, and no run is discarded.

## Artifacts

- Code, per-call logs, and the traced decision logs underlying the paper tables are released with pinned dependency versions and an archived snapshot with a DOI.
- Manuscript tables and figures are generated from stored payloads by scripts in `code/`. No number is transcribed by hand.
- The submission includes a data availability statement and an AI-usage declaration adapted to the venue.

## Environment

- Python environment managed with `uv`, dependency versions pinned in `requirements-lock.txt`.
- Continuous integration runs the full fast test suite on every push.
