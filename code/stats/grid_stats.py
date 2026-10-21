"""Grid-scope statistical battery for the Q1 evaluation (protocol section 6).

Reads the pooled per-record metrics of the grid, evaluates the pre-registered
families F1..F5 (grounded against four conditions, per model, pooled over
templates, one Holm step-down per family), the Friedman omnibus per model and
metric, the per-template robustness runs, the appendix assumption checks
(Shapiro-Wilk on paired differences, Levene across conditions) and the
computational-budget summary. Writes a JSON report and the LaTeX tables for
the manuscript. Deterministic: the cluster bootstrap uses a fixed seed and
resamples seeds (clusters).

Metric families (protocol section 6):
  F1 unsupported_rate      F2 contradiction_rate   F3 cf_consistency
  F4 comprehensiveness     F5 sufficiency
Each family: 4 comparisons x 3 models = 12 tests, Holm within the family.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from code.stats.compare import cohens_dz, holm

METRICS = ["unsupported_rate", "contradiction_rate", "cf_consistency", "comprehensiveness", "sufficiency"]
FAMILY_NAMES = {metric: f"F{index + 1}" for index, metric in enumerate(METRICS)}
COMPARISONS = [
    ("grounded", "free"),
    ("grounded", "single_block"),
    ("grounded", "sabotage"),
    ("grounded", "occlusion_attributions"),
]
CONDITIONS = [
    "template_floor",
    "grounded",
    "sabotage",
    "occlusion_attributions",
    "free",
    "single_block",
]
LLM_CONDITIONS = CONDITIONS[1:]
BOOTSTRAP_ITERATIONS = 10_000
SEED = 0


def cliffs_delta(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Cliff's delta via the Mann-Whitney U identity (ties handled by ranks)."""
    n_a, n_b = values_a.size, values_b.size
    if n_a == 0 or n_b == 0:
        return 0.0
    combined = np.concatenate([values_a, values_b])
    ranks = scipy_stats.rankdata(combined)
    rank_a = ranks[:n_a].sum()
    u = rank_a - n_a * (n_a + 1) / 2.0
    return float(2.0 * u / (n_a * n_b) - 1.0)


def rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation on the nonzero differences."""
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = scipy_stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    total = positive + negative
    return float((positive - negative) / total) if total > 0 else 0.0


def cluster_bootstrap_ci(
    values_a: np.ndarray, values_b: np.ndarray, seeds: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """Percentile CI of the mean difference, resampling seed clusters."""
    differences = values_a - values_b
    unique = np.unique(seeds)
    index_by_seed = {seed: np.where(seeds == seed)[0] for seed in unique}
    means = np.empty(BOOTSTRAP_ITERATIONS)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        picked = rng.choice(unique, size=unique.size, replace=True)
        take = np.concatenate([index_by_seed[item] for item in picked])
        means[iteration] = differences[take].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(differences: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon p-value; all-zero differences are p = 1 by convention."""
    if np.allclose(differences, 0.0):
        return 0.0, 1.0
    statistic, p_value = scipy_stats.wilcoxon(differences, alternative="two-sided")
    return float(statistic), float(p_value)


def friedman_summary(block_table: np.ndarray) -> dict:
    """Friedman omnibus plus mean ranks for a blocks x conditions table."""
    columns = [block_table[:, index] for index in range(block_table.shape[1])]
    chi2, p_value = scipy_stats.friedmanchisquare(*columns)
    rank_means = scipy_stats.rankdata(block_table, axis=1, method="average").mean(axis=0)
    return {"chi2": float(chi2), "p": float(p_value), "n_blocks": int(block_table.shape[0]),
            "mean_ranks": [round(float(value), 3) for value in rank_means]}


def average_by_request(rows: list[dict], metric: str) -> dict[tuple[str, str], float]:
    """One value per (request id, condition): template values averaged per request."""
    accum: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        accum.setdefault((row["request_id"], row["condition"]), []).append(float(value))
    return {key: float(np.mean(values)) for key, values in accum.items()}


def paired_tests(rows: list[dict], metric: str, model: str, rng: np.random.Generator) -> list[dict]:
    """The four grounded comparisons for one metric and model, pooled over templates."""
    selected = [row for row in rows if row.get("llm_model") == model and metric in row]
    by_key = average_by_request(selected, metric)
    request_ids = sorted({key[0] for key in by_key})
    tests = []
    for condition_a, condition_b in COMPARISONS:
        values_a, values_b, seeds = [], [], []
        for request_id in request_ids:
            value_a = by_key.get((request_id, condition_a))
            value_b = by_key.get((request_id, condition_b))
            if value_a is None or value_b is None:
                continue
            values_a.append(value_a)
            values_b.append(value_b)
            seeds.append(request_id.split("-")[0])
        test = {"model": model, "metric": metric, "comparison": f"{condition_a} vs {condition_b}"}
        if len(values_a) < 10:
            test["note"] = "insufficient pairs"
            tests.append(test)
            continue
        array_a = np.asarray(values_a)
        array_b = np.asarray(values_b)
        differences = array_a - array_b
        statistic, p_value = wilcoxon_p(differences)
        low, high = cluster_bootstrap_ci(array_a, array_b, np.asarray(seeds), rng)
        test.update(
            {
                "n": len(values_a),
                "mean_a": round(float(array_a.mean()), 4),
                "mean_b": round(float(array_b.mean()), 4),
                "mean_diff": round(float(differences.mean()), 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
                "p_raw": p_value,
                "statistic": statistic,
                "dz": round(cohens_dz(differences), 3),
                "cliff_delta": round(cliffs_delta(array_a, array_b), 3),
                "rank_biserial": round(rank_biserial(differences), 3),
            }
        )
        tests.append(test)
    return tests


def template_robustness(rows: list[dict], metric: str, model: str) -> dict:
    """Direction of the four comparisons per template (robustness, not a family)."""
    summary = {}
    for template in sorted({row.get("template") for row in rows if row.get("template")}):
        subset = [row for row in rows if row.get("template") == template]
        by_key = {(row["request_id"], row["condition"]): row for row in subset}
        request_ids = sorted({row["request_id"] for row in subset})
        directions = {}
        for condition_a, condition_b in COMPARISONS:
            diffs = []
            for request_id in request_ids:
                row_a = by_key.get((request_id, condition_a))
                row_b = by_key.get((request_id, condition_b))
                if row_a is None or row_b is None:
                    continue
                value_a, value_b = row_a.get(metric), row_b.get(metric)
                if value_a is None or value_b is None:
                    continue
                diffs.append(float(value_a) - float(value_b))
            if len(diffs) >= 10:
                directions[f"{condition_a} vs {condition_b}"] = round(float(np.mean(diffs)), 4)
        summary[template] = directions
    return summary


def friedman_per_model(rows: list[dict], metric: str, model: str, conditions: list[str]) -> dict:
    """Friedman over condition blocks; template values averaged per request."""
    selected = [row for row in rows if row.get("llm_model") in (model, "none")]
    accum: dict[tuple[str, str], list[float]] = {}
    for row in selected:
        value = row.get(metric)
        if value is None or row.get("condition") not in conditions:
            continue
        key = (row["request_id"], row["condition"])
        accum.setdefault(key, []).append(float(value))
    request_ids = sorted({key[0] for key in accum})
    table = []
    for request_id in request_ids:
        block = []
        complete = True
        for condition in conditions:
            values = accum.get((request_id, condition))
            if not values:
                complete = False
                break
            block.append(float(np.mean(values)))
        if complete:
            table.append(block)
    if len(table) < 10:
        return {"note": "insufficient complete blocks", "n_blocks": len(table)}
    result = friedman_summary(np.asarray(table))
    result["conditions"] = conditions
    return result


def shapiro_diagnostics(rows: list[dict], metric: str, model: str) -> dict:
    """Shapiro-Wilk on the paired differences of the four comparisons."""
    selected = [row for row in rows if row.get("llm_model") == model]
    by_key = average_by_request(selected, metric)
    request_ids = sorted({key[0] for key in by_key})
    p_values = []
    for condition_a, condition_b in COMPARISONS:
        diffs = []
        for request_id in request_ids:
            value_a = by_key.get((request_id, condition_a))
            value_b = by_key.get((request_id, condition_b))
            if value_a is None or value_b is None:
                continue
            diffs.append(value_a - value_b)
        if len(diffs) >= 10 and not np.allclose(diffs, 0.0):
            p_values.append(float(scipy_stats.shapiro(np.asarray(diffs)).pvalue))
    return {
        "n_cells": len(p_values),
        "share_p_below_05": round(float(np.mean([p < 0.05 for p in p_values])), 3) if p_values else None,
        "min_p": round(float(np.min(p_values)), 6) if p_values else None,
    }


def levene_diagnostics(rows: list[dict], metric: str, model: str) -> dict:
    """Levene (median-centred) across the six condition groups of one model."""
    groups = []
    for condition in CONDITIONS:
        values = [
            float(row[metric])
            for row in rows
            if row.get("llm_model") == model and row.get("condition") == condition and metric in row
            and row.get(metric) is not None
        ]
        if len(values) >= 10:
            groups.append(np.asarray(values))
    if len(groups) < 2:
        return {"note": "insufficient groups"}
    statistic, p_value = scipy_stats.levene(*groups, center="median")
    return {"statistic": float(statistic), "p": float(p_value), "n_groups": len(groups)}


def budget_summary(rows: list[dict]) -> dict:
    """Per model and condition: latency median/p95 and mean completion tokens."""
    models = sorted({row.get("llm_model") for row in rows if row.get("llm_model")})
    budget = {}
    for model in models:
        budget[model] = {}
        for condition in CONDITIONS:
            cells = [row for row in rows if row.get("llm_model") == model and row.get("condition") == condition]
            seconds = [float(row["seconds"]) for row in cells if row.get("seconds") is not None]
            tokens = [row.get("eval_count") for row in cells if row.get("eval_count")]
            if not seconds:
                continue
            budget[model][condition] = {
                "n": len(seconds),
                "median_seconds": round(float(np.median(seconds)), 2),
                "p95_seconds": round(float(np.percentile(seconds, 95)), 2),
                "mean_tokens": round(float(np.mean(tokens)), 1) if tokens else None,
            }
    return budget


def average_ranks(values: list[float]) -> list[float]:
    """Average ranks within one block, ties sharing the mean rank."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for slot in range(position, end + 1):
            ranks[order[slot]] = average
        position = end + 1
    return ranks


NEMENYI_Q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def pairwise_matrix(rows: list[dict], metric: str, conditions: list[str] | None = None) -> dict:
    """Pairwise Wilcoxon p-values, Holm over the unique pairs, mean ranks and the Nemenyi CD."""
    conditions = list(conditions or CONDITIONS)
    size = len(conditions)
    by_key = average_by_request(rows, metric)
    request_ids = sorted({key[0] for key in by_key})
    p_raw = np.full((size, size), np.nan)
    pairs = []
    for i in range(size):
        for j in range(i + 1, size):
            values_a, values_b = [], []
            for request_id in request_ids:
                value_a = by_key.get((request_id, conditions[i]))
                value_b = by_key.get((request_id, conditions[j]))
                if value_a is None or value_b is None:
                    continue
                values_a.append(value_a)
                values_b.append(value_b)
            if len(values_a) >= 10:
                _, p_value = wilcoxon_p(np.asarray(values_a) - np.asarray(values_b))
            else:
                p_value = float("nan")
            p_raw[i, j] = p_raw[j, i] = p_value
            pairs.append((i, j, p_value))
    valid = [(i, j, p_value) for i, j, p_value in pairs if not math.isnan(p_value)]
    adjusted = holm([p_value for _, _, p_value in valid])
    p_holm = np.full((size, size), np.nan)
    for (i, j, _), value in zip(valid, adjusted):
        p_holm[i, j] = p_holm[j, i] = value
    accumulator: dict[str, list[float]] = {condition: [] for condition in conditions}
    for request_id in request_ids:
        values = []
        complete = True
        for condition in conditions:
            value = by_key.get((request_id, condition))
            if value is None:
                complete = False
                break
            values.append(value)
        if complete:
            for condition, rank in zip(conditions, average_ranks(values)):
                accumulator[condition].append(rank)
    n_blocks = len(accumulator[conditions[0]]) if conditions else 0
    ranks = [
        round(float(np.mean(accumulator[condition])), 3) if accumulator[condition] else None
        for condition in conditions
    ]
    q_value = NEMENYI_Q.get(size)
    cd = q_value * math.sqrt(size * (size + 1) / (6.0 * n_blocks)) if q_value and n_blocks else None
    return {
        "conditions": conditions,
        "p_raw": [[None if math.isnan(value) else float(value) for value in row] for row in p_raw],
        "p_holm": [[None if math.isnan(value) else float(value) for value in row] for row in p_holm],
        "mean_ranks": ranks,
        "n_blocks": n_blocks,
        "nemenyi_cd": round(cd, 3) if cd is not None else None,
    }


def fmt_number(value: float, places: int = 3) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.1e}"
    return f"{value:.{places}f}"


def write_family_table(tests: list[dict], family: str, metric: str, path: Path) -> None:
    lines = [
        f"% Generated by code/stats/grid_stats.py — family {family} ({metric})",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\hline",
        "Model & Comparison & $n$ & Mean A & Mean B & Diff & 95\\% CI & $p_{Holm}$ & $d_z$ & $\\delta$ \\\\",
        "\\hline",
    ]
    for test in tests:
        model = test["model"].replace("_", "\\_")
        comparison = test["comparison"].replace("_", "\\_")
        if "p_raw" in test:
            ci = f"[{test['ci_low']:.3f}, {test['ci_high']:.3f}]"
            lines.append(
                f"{model} & {comparison} & {test['n']} & {test['mean_a']:.3f} & {test['mean_b']:.3f} & "
                f"{test['mean_diff']:.3f} & {ci} & {fmt_number(test['p_holm'])} & {test['dz']:.2f} & "
                f"{test['cliff_delta']:.2f} \\\\"
            )
        else:
            lines.append(f"{model} & {comparison} & -- & -- & -- & -- & -- & -- & -- & -- \\\\")
    lines += ["\\hline", "\\end{tabular}", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/grid_q1/analysis/pooled_metrics.json")
    parser.add_argument("--out", default="results/grid_q1/analysis/stats/grid_stats.json")
    parser.add_argument("--tables-dir", default="results/grid_q1/analysis/tables")
    args = parser.parse_args()

    payload = json.loads(Path(args.metrics).read_text())
    all_rows = [row for row in payload["per_record"] if row.get("llm_model")]
    rows = [row for row in all_rows if row["llm_model"] != "none"]
    if not rows:
        raise SystemExit("no rows with llm_model; score the grid cells first")
    models = sorted({row["llm_model"] for row in rows})
    rng = np.random.default_rng(SEED)

    families: dict[str, list[dict]] = {}
    for metric in METRICS:
        tests = []
        for model in models:
            tests.extend(paired_tests(rows, metric, model, rng))
        computable = [test for test in tests if "p_raw" in test]
        adjusted = holm([test["p_raw"] for test in computable])
        for test, value in zip(computable, adjusted):
            test["p_holm"] = value
        families[FAMILY_NAMES[metric]] = tests

    friedman = {}
    for metric in METRICS:
        friedman[metric] = {}
        for model in models:
            friedman[metric][model] = {
                "six_conditions": friedman_per_model(all_rows, metric, model, CONDITIONS),
                "five_llm_conditions": friedman_per_model(all_rows, metric, model, LLM_CONDITIONS),
            }

    templates = {metric: {model: template_robustness(rows, metric, model) for model in models} for metric in METRICS}
    diagnostics = {
        metric: {model: {"shapiro": shapiro_diagnostics(rows, metric, model), "levene": levene_diagnostics(rows, metric, model)}
                 for model in models}
        for metric in METRICS
    }
    budget = budget_summary(rows)
    pairwise = {metric: pairwise_matrix(all_rows, metric) for metric in METRICS}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "families": families,
                "friedman": friedman,
                "templates": templates,
                "diagnostics": diagnostics,
                "budget": budget,
                "pairwise": pairwise,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "seed": SEED,
                "models": models,
            },
            indent=2,
        )
    )

    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        family = FAMILY_NAMES[metric]
        write_family_table(families[family], family, metric, tables_dir / f"tab_{family.lower()}_{metric}.tex")

    lines = [
        "% Generated by code/stats/grid_stats.py — computational budget",
        "\\begin{tabular}{llrrrr}",
        "\\hline",
        "Model & Condition & $n$ & Median (s) & P95 (s) & Mean tokens \\\\",
        "\\hline",
    ]
    for model, conditions in budget.items():
        for condition, summary in conditions.items():
            model_escaped = model.replace("_", "\\_")
            lines.append(
                f"{model_escaped} & {condition.replace('_', '\\_')} & {summary['n']} & {summary['median_seconds']:.2f} & "
                f"{summary['p95_seconds']:.2f} & {summary['mean_tokens'] if summary['mean_tokens'] is not None else '--'} \\\\"
            )
    lines += ["\\hline", "\\end{tabular}", ""]
    (tables_dir / "tab_budget.tex").write_text("\n".join(lines))

    print(f"{'family':<6} {'metric':<22} {'tests':>5} {'min p_holm':>11}")
    for metric in METRICS:
        family = FAMILY_NAMES[metric]
        computable = [test for test in families[family] if "p_holm" in test]
        minimum = min((test["p_holm"] for test in computable), default=float("nan"))
        print(f"{family:<6} {metric:<22} {len(computable):>5} {fmt_number(minimum):>11}")
    print(f"wrote {out_path} and tables under {tables_dir}")


if __name__ == "__main__":
    main()
