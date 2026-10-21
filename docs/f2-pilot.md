# F2/F3 pilot: the agent loop end to end

Status: **v2 run executing on 2026-09-28** on the Titan V node (the v1 run is
archived in `results/pilot_v1`). The pilot validates the pipeline and produces
preliminary effect sizes. It is labeled exploratory in the paper and is
superseded by the full grid (phase F4).

## What runs

- 30 requests: six tasks (iris, wine, wdbc, thyroid, parkinsons, digits), five
  evenly spaced test rows each. Verbatim texts and encoded features live in
  `results/pilot/requests.json`.
- Six conditions per request: `template_floor`, `grounded`, `sabotage`,
  `occlusion_attributions`, `free`, `single_block`.
- Language model: qwen3.5 (9.7B Q4_K_M) served by Ollama on the NVIDIA Titan V
  node over Tailscale (~58 tokens per second, temperature 0, seed 0, thinking
  disabled). Every call is cached with replay in `results/pilot/llm_cache.jsonl`,
  keyed by model, messages, options, thinking flag, and endpoint.
- Routing is the oracle in the pilot. The typed router (Laya) is evaluated in
  its own battery (`make router-probe` and `make router-baselines`) and
  integrates at the grid stage.

## Commands

```
make pilot-requests   # build results/pilot/requests.json
make pilot-run        # run the six conditions over the thirty requests
make pilot-score      # fidelity harness: executed counterfactuals, comp/suff, claim audits
make pilot-stats      # paired tests, Holm correction, bootstrap, LaTeX table
```

## Metrics

Executed counterfactual consistency, comprehensiveness, and sufficiency are
exact because every metric re-executes the substrate. Two text metrics run in
parallel: the citation-based unsupported rate for the conditions that cite,
and a citation-free contradiction audit for all conditions. Both are heuristic
and the protocol reserves a human-audited sample to bound their error.

## v2 prompt and scorer updates (2026-09-28)

- The single-block prompt now receives the candidate task identifiers, so the
  condition measures routing over the known task space instead of name
  guessing.
- The grounded contract asks for one sentence about a counterfactual entry
  (with its identifier cited), which keeps counterfactual consistency
  measurable across the grounded family.
- The scorer normalizes identifier-style citations such as
  `[counterfactuals.id="cf-1"]` to `[cf-1]`, resolves section-level citations
  against the rows they contain, parses several counterfactual edits per
  sentence, and scores numeric class claims of the single-block condition
  against the true class index.

## Known limitations

- The scorer is heuristic by design and the human audit of the protocol is
  pending.
- One language model and one prompt version. Template families and model scale
  belong to the grid stage.
- Oracle routing in the pilot, so routing errors do not propagate into the
  fidelity measurements. The grid stage runs the typed router end to end.
- Serving latency is reported per configuration. On the Titan V node the model
  answers in two to three seconds per explanation, which moves the bottleneck
  to the substrate-side scoring rather than the verbalizer.

## Pilot results (2026-09-28)

Thirty requests, six conditions, qwen3.5 on the Titan V node. Mean unsupported
rate (sentences without support in the cited evidence): template floor 0.000,
grounded 0.008, sabotage 0.103, occlusion of the attribution block 0.244, free
generation 1.000, single block 1.000. Paired comparisons: grounded against
free 0.992 lower with a Holm-corrected p-value of 9.5e-07 and d_z of -21.7,
against occlusion 0.236 lower with p of 4.3e-04. The contradiction audit rises
from 0.100 in grounded to 0.254 under sabotage (p of 0.0027, d_z of -0.95).
Executed counterfactual consistency is 1.000 in grounded and occlusion and
0.963 under sabotage, with counterfactual coverage of 0.83 and higher. The
single-block condition routes 0.833 of requests to the correct task with the
candidate list, and its numeric class claims agree with the substrate in 0.174
of the cases. Median latency is 2.1 seconds in grounded against 1.5 in free.

Two internal iterations are archived and not reported: the v1 pilot used a
weaker citation contract, and the first v2 execution serialized a volatile
creation timestamp into the prompt, which broke exact replay and was fixed
(the fix is recorded in the protocol changelog).

## Preliminary grid (5 seeds, pooled)

Five request seeds (the pilot is seed 0) pool 900 records. Mean unsupported
rate: template floor 0.000, grounded 0.012, sabotage 0.124, occlusion 0.203,
free 1.000, single block 1.000. Paired comparisons over 150 request pairs:
grounded against free 0.988 lower with a Holm-corrected p-value of 3.8e-32
and d_z of -16.3, against occlusion 0.191 lower (p of 5.4e-14) and against
sabotage 0.112 lower (p of 1.7e-09). The contradiction audit places grounded
at 0.108 against 0.242 under sabotage (p of 6.9e-14) and 0.421 for the
single-block condition (p of 5.6e-05 over the 76 checkable pairs). Executed
counterfactual consistency is 1.000 in grounded, 0.992 in occlusion and 0.989
under sabotage with coverage of 0.86 and higher, and the comparison against
sabotage is not significant at this scale (difference 0.012). Single-block
routes 0.827 of requests and agrees with the substrate on 0.162 of its numeric
class claims. The pooled table and statistics are generated by
`scripts/run_grid.sh` into `results/grid/` (a copy of the table lives in
`paper/tables/tab_pooled.tex`).

