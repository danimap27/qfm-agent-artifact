# Pre-registered experiment protocol (draft v0.1)

> Status: DRAFT, created 2026-09-23. To be frozen before the full experimental grid (phase F4). After the freeze, any change is appended to the changelog at the end of this document, with the date and the reason.

## 1. Objective

Measure whether grounding language-model explanations in verifiable evidence increases explanation faithfulness over free generation, for a pipeline that routes natural language requests to, and explains predictions of, a pretrained variational quantum model.

## 2. Hypotheses

- **C1 Routing fidelity.** A typed router selects task and head with accuracy and calibration no worse than classical baselines on the same requests, and its provenance is reconstructible and sensitive to input perturbations.
- **C2 Explanation faithfulness.** With the same language model, substrate, and requests, the grounded condition scores better than the free condition on intervention metrics (executed counterfactual consistency, comprehensiveness, sufficiency) and produces fewer unsupported claims.
- **C3 Decoupling.** The decoupled pipeline produces fewer unsupported claims and fewer contradictions with the decision log than a single-block model that routes, predicts, and explains in one pass.
- **C4 Evidence dependence.** Faithfulness degrades when evidence fields are occluded one at a time, and explanations in the sabotage condition track the tampered evidence.
- **C5 System.** The loop runs end to end over a pretrained 8-qubit multitask circuit, transfers to unseen tasks through residual adapters, and reports latency and cost per explanation for every serving configuration.

## 3. Conditions

Paired design: identical requests, identical substrate checkpoints, and identical seeds across conditions. The verbalizer is the same language model (temperature zero, cached) in every condition that uses one.

| Condition | Routing | Evidence given to the verbalizer |
|---|---|---|
| `single_block` | decided by the language model | none |
| `free` | oracle | prediction and probability |
| `grounded` | oracle | full decision log, every claim must cite a field |
| `random_routing` | uniform | full decision log |
| `occlusion_<field>` | oracle | decision log minus one field |
| `sabotage` | oracle | decision log with tampered attribution fields |
| `template_floor` | oracle | full log, deterministic template verbalizer (no language model) |

## 4. Metrics

Primary endpoints:

- **P1.** Executed counterfactual consistency: fraction of counterfactual claims that, when applied to the substrate, produce the claimed change.
- **P2.** Comprehensiveness and sufficiency, measured exactly by re-execution: prediction drop when the claimed-important features are removed, and prediction retained when only those features are kept.
- **P3.** Unsupported claim rate: fraction of explanation claims without a supporting citation or contradicted by the cited evidence, scored automatically and audited by human annotators on a stratified sample.

Secondary endpoints: attribution agreement (Kendall tau, top-k overlap), contrary-hint sensitivity, routing accuracy and expected calibration error, out-of-distribution flag precision, explanation coverage and informativeness, and cost (latency median and p95, tokens, failures).

## 5. Statistical analysis plan

- At least five seeds per cell, ten in headline cells. The unit of analysis is one explanation instance. Comparisons are paired by request, seed, and substrate checkpoint.
- Wilcoxon signed-rank tests with Holm-Bonferroni correction across the family of comparisons, reported with effect sizes (Cliff's delta, Cohen's d_z) and bootstrap confidence intervals.
- Full distributions are reported in the appendix. Underpowered cells are labelled exploratory. One pre-registered analysis is used in every table.

## 6. Falsification criteria

- C1 is falsified if the router does not beat the majority baseline or if its calibration is worse than a logistic regression on the same request signal.
- C2 is falsified if the grounded condition shows no improvement over free generation across the P1 to P3 family after correction, or if the only improvement is citation coverage without any change in intervention metrics.
- C3 is falsified if the single-block condition matches the grounded condition on unsupported claims, meaning the decoupling is not the active ingredient.
- C4 is falsified if field occlusion moves no metric (the model is not reading the log) or if the sabotage condition does not change the output (no mediation).
- C5 is a system requirement rather than a statistical claim: the loop must run end to end at the reported cost, and the transfer protocol must be reported with its degradation.

## 7. Reproducibility requirements

Every run follows `REPRODUCIBILITY.md`: pinned dependencies, recorded seeds, prompt versions with hashes, request cache with replay, and model identifiers with quantization for every language model used.

## Changelog

- 2026-09-23: draft v0.1 created.
- 2026-09-28: prompt contract v2 fixed for all conditions after the internal v1 pilot. Grounded requires one sentence about a counterfactual entry with its identifier cited. Single-block receives the candidate task identifiers, so the condition measures routing over the known task space. The scorer normalizes identifier-style citations, accepts section-level citations against the rows they contain, parses several counterfactual edits per sentence, and scores numeric class claims of single-block against the substrate. The contradiction audit joins the comparison family. The internal v1 pilot used a weaker contract and is not reported.
- 2026-09-28: preliminary grid, request seeds 0 to 4 (the deterministic pilot is seed 0). Pooled rows are keyed by seed and request. This preliminary grid is labeled exploratory in the manuscript and does not replace the freeze that precedes the full grid.
- 2026-09-28: serialization of the decision log into prompts excludes volatile metadata (the log creation timestamp). Rebuilding an experiment therefore reproduces byte-identical prompts and cache keys, and `--replay` serves the whole run from disk. The pilot and the preliminary grid were re-run under this contract after the fix.
