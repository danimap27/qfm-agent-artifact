# Grid Q1 — Evaluation Protocol (organization)

Status: AGREED on 2026-09-28 (decisions frozen below). Nothing is launched
until the build phase (A) passes its tests.
Language: repository docs are English by convention.

Frozen decisions (2026-09-28): third model gemma3:27b; 10 seeds
({0, 42, 123, 456, 789, 7, 13, 21, 34, 55}); templates t2/t3 keep the same
information and citation contract (form only); decision-layer battery
extended with Julia 1 (feasibility spike first), probe >= 150 items and
McNemar exact test; graded sabotage ladder deferred as an extension.

## 1. Objective

Scale the F3/F4 evaluation from pilot scale (30 requests, 1 model, 1 template,
1 seed battery) to the Q1 evidence floor defined in the venue evaluation:

- >= 3 prompt templates
- >= 3 open-weights models across >= 2 families
- 100 requests per seed, full 6-condition battery
- 10 request seeds (stretch 20), all metrics as mean +- SD over seeds
- nonparametric statistics with Holm correction over declared families
- multi-objective figures including efficiency (adopted from the QTCL /
  CMES evaluation conventions)

## 2. Experimental design (frozen decisions in bold)

Factors (fully crossed, paired at the request level):

- **model (3)**: `qwen3.5:latest` (Qwen, 9.7B Q4), `llama3.1:8b` (Meta),
  `gemma3:27b` (Google). Alternative to shorten the grid: `gemma3:12b`.
- **template (3)**: `t1_table` (current frozen v3 contract), `t2_narrative`,
  `t3_checklist`. Same information content and same citation contract in all
  three; only presentation and instruction wording change (measurement
  instrument guardrail, validated by a smoke cell before the grid).
- **seed (10)**: {0, 42, 123, 456, 789, 7, 13, 21, 34, 55}. The first five
  are the family used in the QTCL/CMES papers; the protocol section of the
  manuscript states the set before results.
- **condition (6)**: template_floor, grounded, sabotage,
  occlusion_attributions, free, single_block (unchanged from F2/F3).
- **requests (100 per seed)**: rows sampled deterministically from the frozen
  test splits of the six-task mix (per-task allocation 17/17/17/17/16/16).
  Seeds vary the sampled rows only: the substrate (checkpoint d4/1900 and
  split seeds) is frozen and shared, exactly as in the F4 grid.

Counts: per seed 100 x 6 x 3 x 3 = 5,400 records (4,500 LLM calls).
Total: 54,000 records / 45,000 calls over 10 seeds (108k / 90k over 20).

## 3. Execution topology (gpu-pc, both GPUs, LOCAL execution)

Remote execution through the Tailscale link is discarded for the grid
(measured ~3.7 calls/min vs ~35 calls/min per GPU locally). Plan:

1. Pre-build on the homelab: per seed, `requests.json` (sampled rows) and
   `logs.jsonl` (deterministic evidence, no LLM involvement). New helper
   `code.pipeline.build_logs`; `run_agent` gains `--logs` (skip substrate and
   task imports; documented, tested) and `--template`.
2. Deploy to gpu-pc: repo subset + numpy as an unzipped wheel on PYTHONPATH
   (no pip, no sudo), plus the JSONL payloads (small).
3. Models: pulled on the homelab at ~83 MB/s, shipped over the Cloudflare
   pipe with the micro-connection fetch pattern (`?start=` resume, 40 s
   connections, sha256 verified) — the pattern proven on qwen3.5.
4. Two daemons: A on GPU0 :11434, B on GPU1 :11435, `OLLAMA_KEEP_ALIVE=-1`.
5. Two workers (one per daemon), cells = (model, template, seed) = 90 cells,
   500 calls each. Cell lists balanced by expected duration (gemma3:27b
   cells are ~2x slower). Per pass: `timeout 180`; resume through the
   llm_cache; per-cell COMPLETO markers; status file for monitoring.
6. Expected wall time: ~11 h (gemma3:12b) to ~15 h (gemma3:27b) with both
   workers in parallel.
7. Collect: rsync of explanations + caches (tens of MB) with retry loop;
   verify record counts and prompt-hash playback before deleting anything.

### 3.1 Execution split (added 2026-09-28 during phase B)

The grid is federated across two nodes for throughput and redundancy. The
two-node split lowers the wall estimate to ~12 h against ~15 h on gpu-pc
alone, and a node failure still yields a complete half-grid.

- gpu-pc (2 x RTX 3090): seeds 0, 42, 123, 456, 789 for qwen3.5:latest and
  llama3.1:8b (30 cells), plus ALL gemma3:27b cells (30 cells, phase 2 on
  both GPUs).
- Titan V node (12 GB): seeds 7, 13, 21, 34, 55 for qwen3.5:latest and
  llama3.1:8b (30 cells, single GPU, serial queue). gemma3:27b does not
  fit its 12 GB VRAM, so the Titan runs no phase-2 cells.
- Each node keeps per-cell caches and COMPLETO markers; cells are
  deterministic and disjoint, so collection is a union by cell key.
- Cross-hardware note: same ollama version (0.34.4) and frozen prompts on
  both nodes; portability numbers for both GPUs feed the computational
  budget table.
- Template guardrail: the first t2 and t3 cell completed on each node is
  inspected manually (citation-contract check) before the grid is
  accepted.

## 4. Artifacts and layout

    results/grid_q1/<model>/<template>/seed<K>/
        requests.json  logs.jsonl  explanations.jsonl  llm_cache.jsonl  metrics.json
    results/grid_q1/analysis/
        pooled_metrics.json      stats/*.json     tables/*.tex
        figures/*.pdf|*.png      qc-report.md

Model slugs: qwen35, llama31_8b, gemma3_27b. Templates: t1_table,
t2_narrative, t3_checklist. Seeds: seed0 = "0", seed42 = "42", etc.

## 5. Metrics battery (per record unless noted)

Existing (kept): unsupported_rate (primary), contradiction_rate,
checkable_coverage, cf_consistency, cf_coverage, comprehensiveness,
sufficiency, claimed_task/class accuracy (single_block), seconds, chars.

Added for the grid:
- prompt/completion tokens and tok/s per call (client stores them)
- citations per sentence; invalid-citation rate; format parse-fail rate
- injection-repeat rate (sabotage: share of injected claims repeated)
- attribution-omission delta (occlusion vs grounded)
- latency mean/p50/p95 and per-model throughput (calls/min)
- stability: SD of the primary metric across seeds; cross-template
  direction agreement (robustness check, not a test)
- efficiency: seconds and tokens per record, per condition and per model
  (computational-budget table, QTCL convention)

## 6. Statistical analysis plan (pre-registered)

- Unit: request within seed; pooled analyses pair by (seed, request id).
- Omnibus: Friedman across the 6 conditions per metric per model; on
  significance, paired post-hoc.
- Primary family F1: unsupported_rate, grounded vs {free, single_block,
  sabotage, occlusion_attributions} x 3 models, pooled over templates
  = 12 tests, one Holm step-down over the family. Templates analysed in
  parallel per template (robustness, expected same direction, not part of
  the Holm family).
- Secondary families F2..F5: contradiction_rate, cf_consistency,
  comprehensiveness, sufficiency; same 4 x 3 shape, Holm within each
  metric family.
- Reporting: mean +- SD over seeds everywhere (CMES/QTCL convention);
  cluster bootstrap (by seed) 95% CI for pooled mean differences.
- Assumption checks reported in the appendix: Shapiro-Wilk (per cell,
  aggregate statement) and Levene; they justify the nonparametric path.
- Effect sizes: Cohen dz, Cliff delta, rank-biserial r (95% bootstrap CI);
  risk difference for rates. Effect-size table generated automatically.
- Extra (stretch): van Elteren stratified test across models.
- Software: scipy (wilcoxon, friedmanchisquare), custom Holm (existing
  code.stats.compare.holm), seeded bootstrap. All scripts committed.

## 7. Figures (planned set; ~6 main text + appendix)

1. fig_main_unsupported - bars per condition, panels per model + pooled,
   95% CI (cluster bootstrap).
2. fig_contradiction - grounded / sabotage / occlusion per model.
3. fig_fidelity_decomposition - cf coverage + consistency, comprehensiveness
   and sufficiency panels.
4. fig_seed_stability - per-seed dots (grounded vs free) per model.
5. fig_latency - violin per condition (log scale) with throughput annotation.
6. fig_templates - robustness across t1/t2/t3 (grounded unsupported per
   model x template).
7. fig_pvalue_heatmap - pairwise Holm-adjusted p-value matrix across the 6
   conditions (recipe ported from qtcl-experiments/real/generate_extended.py).
8. fig_critical_difference - critical-difference diagram across conditions
   (same recipe family).
9. fig_radar - multi-objective radar per model (faithfulness, coverage,
   counterfactual consistency, efficiency).
Appendix: OPTIONAL pareto (latency vs fidelity), qualitative example.

Tables: main pooled (mean +- SD per condition per model), per-template
appendix, effect sizes, statistics summary (families, raw/adjusted p),
computational budget (efficiency + hardware portability: Titan V vs RTX
3090 numbers already measured), seeds appendix (per-seed primary metric).

Adopted from the authors' previous papers (explicit): five-seed canonical
family and mean +- SD convention, paired Wilcoxon at alpha = 0.05 as the
reported test (CMES/QTCL), Friedman omnibus + Kruskal fallback and
bootstrap CI + Holm batteries (qtcl-experiments/real/statistical_tests.py),
effect-size and computational-budget tables, p-value heatmap and
critical-difference diagram (qtcl-experiments/real/generate_extended.py).

## 8. QC gates (all must pass before results are frozen)

- Cell completeness: 90/90 cells, 5,400 records per seed, no duplicate ids.
- Null audit: no nulls in required fields; parse-fail rate reported.
- Playback: 100 random prompt hashes replay exactly from the caches.
- Sanity: template_floor unsupported == 0; free unsupported ~ 1 by
  construction; grounded strictly below free in every model/template cell.
- Tooling: repo tests green (51+), manuscript lint unaffected,
  figures reproducible from committed inputs (results dir hash recorded).

## 9. Runbook (ops)

A. Build (homelab, ~1 day of work): build_logs helper; run_agent --logs and
   --template; client token capture; stats extensions (Friedman, effect
   sizes, families); figures module; tests; deployment pack scripts.
B. Prep (~1-2 h): pre-build payloads for 10 seeds; wheel pack; model pulls
   (llama3.1:8b, gemma3:27b) and CF-pipe fetches; daemon B setup.
C. Grid (~11-15 h): two workers; monitor status file; watchdog restart.
D. Collect + analyze (~2-3 h): rsync back; score all cells; pooling;
   statistical battery; figures; tables; QC report; vault + repo docs.

## 10. Risks

- Disk on gpu-pc at 93%: +22 GB of models and ~1 GB of results; monitor,
  and delete one blob set after collection if needed.
- Template t2/t3 must not change the measurement contract; guarded by the
  smoke cell and by keeping the citation instruction constant.
- Worker hangs: per-pass timeout + cache resume (proven pattern).
- Machine availability: grid runs unattended overnight; can be paused by
  killing the workers (caches make a resume trivial).
- Reload tax on idle gaps: keep_alive = -1 set on both daemons.

## 11. Decision-layer (non-LLM) battery (added 2026-09-28, for review)

Beyond the LLM grid, the paper evaluates the non-generative decision layer
(the router). Laya (open, Apache 2.0) is the pipeline's decision layer.
Inside the LLM grid the routing is oracle by design, so the decision layer
has its own battery. Status 2026-09-28: the probe was expanded to 156 items
(twelve phrasings per task over the thirteen-task registry, deterministic
build via `code/router/build_probe.py`) and the full battery has been run.

- Entrants: Laya (chosen), Julia 1 (Supersonic Labs, Apache 2.0, 144.3M,
  CPU-runnable, released 2026-09-26), classical baselines (TF-IDF logreg,
  random, majority). Jev (TypeSafe) is closed and not reproducible:
  referenced in related work only, optionally one non-reproducible API row
  in the appendix, clearly labeled.
- Probe: 156 items; per task, the first half trains the classical baselines,
  the second half (78 items) is the shared evaluation split. The original
  v2 items stay inside their original halves as a reproducible subset
  (full-probe accuracies: Laya 0.917 at 687 ms median, agreement 0.994;
  Julia 1 0.846 at 61 ms, agreement 0.846).
- Results on the evaluation half, n = 78, accuracy [Wilson 95 percent
  interval] / ECE / median latency / agreement (`code/router/router_stats.py`):
  logreg 0.974 [0.911, 0.993] / 0.826 / 0.15 ms / 0.974, Laya 0.910
  [0.826, 0.956] / 0.087 / 687 ms / 0.987, Julia 1 0.795 [0.693, 0.870] /
  0.104 / 62.7 ms / 0.821, random 0.154, majority 0.077.
- Paired tests (McNemar exact, Holm over the three-pair family): Laya vs
  Julia 1 p_holm = 0.045, risk difference +0.115. Julia 1 vs logreg
  p_holm = 0.0016, -0.179. Laya vs logreg p = 0.125, not significant,
  -0.064. Reading: the zero-shot typed router matches the trained
  classifier on accuracy while calibrated far better (ECE 0.087 against
  0.826), without training data and under a deterministic typed contract.
  Julia 1 is a strong open-source baseline that stays significantly below
  Laya. Report the promise and the limits as they are, no spin.
- Feasibility spike: DONE 2026-09-28 (single-writer fetch rule learned the
  hard way: never let two fetches append to the same blob, the sha check
  fails on the interleaved stream).
- Related work to cite for the category: arXiv:2609.28940 (JEV and Laya as
  System One decision layers).
