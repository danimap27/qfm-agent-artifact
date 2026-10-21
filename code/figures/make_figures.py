"""Publication figures for the Q1 grid (protocol section 7).

Reads the pooled per-record metrics (and the grid statistics JSON when
available) and renders the planned figure set as PDF and PNG. Deterministic:
bootstraps use a fixed seed. Runs in the dedicated figures environment
(numpy + matplotlib only, no scipy): every statistical input that needs
scipy is precomputed by code/stats/grid_stats.py and read from its JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CONDITIONS = [
    "template_floor",
    "grounded",
    "sabotage",
    "occlusion_attributions",
    "free",
    "single_block",
]
LLM_CONDITIONS = CONDITIONS[1:]

CONDITION_LABELS = {
    "template_floor": "floor",
    "grounded": "grounded",
    "sabotage": "sabotage",
    "occlusion_attributions": "occlusion",
    "free": "free",
    "single_block": "single-block",
}
CONDITION_COLORS = {
    "template_floor": "#9e9e9e",
    "grounded": "#1b7837",
    "sabotage": "#d73027",
    "occlusion_attributions": "#f46d43",
    "free": "#762a83",
    "single_block": "#e08214",
}
MODEL_COLORS = {"qwen": "#1f78b4", "llama": "#33a02c", "gemma": "#e31a1c"}
BOOTSTRAP_ITERATIONS = 5000
SEED = 0


def model_names(rows: list[dict]) -> list[str]:
    """Model names excluding the deterministic floor pseudo-model ("none")."""
    return sorted(
        {
            row.get("llm_model")
            for row in rows
            if row.get("llm_model") and row.get("llm_model") != "none"
        }
    )


def model_color(model: str) -> str:
    lowered = model.lower()
    for key, value in MODEL_COLORS.items():
        if key in lowered:
            return value
    return "#555555"


def bootstrap_mean_ci(values: list[float], rng: np.random.Generator, seeds: list[str] | None = None):
    """Mean with a percentile bootstrap CI; resamples seed clusters when given."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    if seeds is not None and len(set(seeds)) >= 2:
        unique = sorted(set(seeds))
        index_by_seed = {seed: np.where(np.asarray(seeds) == seed)[0] for seed in unique}
        means = np.empty(BOOTSTRAP_ITERATIONS)
        for iteration in range(BOOTSTRAP_ITERATIONS):
            picked = rng.choice(unique, size=len(unique), replace=True)
            take = np.concatenate([index_by_seed[item] for item in picked])
            means[iteration] = array[take].mean()
    else:
        draws = rng.choice(array, size=(BOOTSTRAP_ITERATIONS, array.size), replace=True)
        means = draws.mean(axis=1)
    return float(array.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def group_metric(rows: list[dict], metric: str, fields: tuple[str, ...]) -> dict:
    groups: dict[tuple, dict] = {}
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        key = tuple(row.get(field) for field in fields)
        entry = groups.setdefault(key, {"values": [], "seeds": []})
        entry["values"].append(float(value))
        if row.get("seed") is not None:
            entry["seeds"].append(str(row["seed"]))
    return groups


def mean_cf_coverage(rows: list[dict], condition: str, model: str | None = None) -> float:
    selected = [row for row in rows if row.get("condition") == condition and (model is None or row.get("llm_model") == model)]
    if not selected:
        return float("nan")
    with_claims = sum(1 for row in selected if row.get("cf_consistency") is not None)
    return with_claims / len(selected)


def save_figure(fig, out_dir: Path, name: str, formats: list[str], dpi: int) -> list[Path]:
    written = []
    for extension in formats:
        path = out_dir / f"{name}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def fig_main_unsupported(rows, out_dir, formats, dpi, rng):
    models = model_names(rows)
    panels = models + ["pooled"]
    figure, axes = plt.subplots(1, len(panels), figsize=(2.35 * len(panels), 2.6), constrained_layout=True, sharey=True)
    for axis, model in zip(np.atleast_1d(axes), panels):
        means, lows, highs, positions = [], [], [], []
        for position, condition in enumerate(CONDITIONS):
            if model == "pooled":
                values = [
                    float(row["unsupported_rate"])
                    for row in rows
                    if row.get("condition") == condition and row.get("unsupported_rate") is not None
                ]
                seeds = [str(row.get("seed")) for row in rows if row.get("condition") == condition and row.get("unsupported_rate") is not None]
            else:
                subset = [row for row in rows if row.get("llm_model") == model and row.get("condition") == condition]
                values = [float(row["unsupported_rate"]) for row in subset if row.get("unsupported_rate") is not None]
                seeds = [str(row.get("seed")) for row in subset if row.get("unsupported_rate") is not None]
            mean, low, high = bootstrap_mean_ci(values, rng, seeds if len(set(seeds)) >= 2 else None)
            means.append(mean)
            lows.append(low)
            highs.append(high)
            positions.append(position)
        errors = np.clip(np.vstack([np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)]), 0, None)
        axis.bar(positions, means, yerr=errors, capsize=2, color=[CONDITION_COLORS[c] for c in CONDITIONS], edgecolor="black", linewidth=0.4)
        axis.set_xticks(range(len(CONDITIONS)))
        axis.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], rotation=35, ha="right", fontsize=7)
        title = "pooled" if model == "pooled" else model.split(":")[0]
        axis.set_title(title, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("unsupported rate", fontsize=8)
    return save_figure(figure, out_dir, "fig01_main_unsupported", formats, dpi)


def fig_contradiction(rows, out_dir, formats, dpi, rng):
    models = model_names(rows)
    conditions = ["grounded", "sabotage", "occlusion_attributions"]
    figure, axis = plt.subplots(figsize=(4.6, 2.6), constrained_layout=True)
    width = 0.26
    for offset, condition in zip([-width, 0.0, width], conditions):
        means, errors, positions = [], [], []
        for position, model in enumerate(models):
            subset = [row for row in rows if row.get("llm_model") == model and row.get("condition") == condition]
            values = [float(row["contradiction_rate"]) for row in subset if row.get("contradiction_rate") is not None]
            seeds = [str(row.get("seed")) for row in subset if row.get("contradiction_rate") is not None]
            mean, low, high = bootstrap_mean_ci(values, rng, seeds if len(set(seeds)) >= 2 else None)
            means.append(mean)
            errors.append([max(mean - low, 0.0), max(high - mean, 0.0)])
            positions.append(position + offset)
        axis.bar(positions, means, width=width, yerr=np.asarray(errors).T, capsize=2,
                 color=CONDITION_COLORS[condition], edgecolor="black", linewidth=0.4, label=CONDITION_LABELS[condition])
    axis.set_xticks(range(len(models)))
    axis.set_xticklabels([model.split(":")[0] for model in models], fontsize=8)
    axis.set_ylabel("contradiction rate", fontsize=8)
    axis.legend(fontsize=7, frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    return save_figure(figure, out_dir, "fig02_contradiction", formats, dpi)


def fig_fidelity_decomposition(rows, out_dir, formats, dpi, rng):
    metrics = [
        ("cf_coverage", "CF coverage"),
        ("cf_consistency", "CF consistency"),
        ("comprehensiveness", "comprehensiveness"),
        ("sufficiency", "sufficiency"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(6.4, 4.4), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, metrics):
        means, errors, positions = [], [], []
        for position, condition in enumerate(CONDITIONS):
            if metric == "cf_coverage":
                subset = [row for row in rows if row.get("condition") == condition]
                value = mean_cf_coverage(rows, condition)
                mean, low, high = value, value, value
            else:
                subset = [row for row in rows if row.get("condition") == condition]
                values = [float(row[metric]) for row in subset if row.get(metric) is not None]
                seeds = [str(row.get("seed")) for row in subset if row.get(metric) is not None]
                mean, low, high = bootstrap_mean_ci(values, rng, seeds if len(set(seeds)) >= 2 else None)
            means.append(mean)
            errors.append([max(mean - low, 0.0), max(high - mean, 0.0)])
            positions.append(position)
        axis.bar(positions, means, yerr=np.asarray(errors).T, capsize=1.5,
                 color=[CONDITION_COLORS[c] for c in CONDITIONS], edgecolor="black", linewidth=0.4)
        axis.set_xticks(range(len(CONDITIONS)))
        axis.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], rotation=35, ha="right", fontsize=6.5)
        axis.set_title(label, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    return save_figure(figure, out_dir, "fig03_fidelity_decomposition", formats, dpi)


def fig_seed_stability(rows, out_dir, formats, dpi, rng):
    models = model_names(rows)
    figure, axes = plt.subplots(1, len(models), figsize=(2.4 * len(models), 2.3), constrained_layout=True, sharey=True)
    for axis, model in zip(np.atleast_1d(axes), models):
        for condition in ["grounded", "free"]:
            seeds, means = [], []
            subset = [row for row in rows if row.get("llm_model") == model and row.get("condition") == condition]
            by_seed = {}
            for row in subset:
                if row.get("seed") is None or row.get("unsupported_rate") is None:
                    continue
                by_seed.setdefault(str(row["seed"]), []).append(float(row["unsupported_rate"]))
            for seed in sorted(by_seed, key=lambda value: int(value) if value.isdigit() else value):
                seeds.append(seed)
                means.append(float(np.mean(by_seed[seed])))
            axis.plot(range(len(seeds)), means, marker="o", markersize=3.2, linewidth=1.0,
                      color=CONDITION_COLORS[condition], label=CONDITION_LABELS[condition])
        axis.set_xticks(range(len(seeds)))
        axis.set_xticklabels(seeds, rotation=45, fontsize=6.5)
        axis.set_title(model.split(":")[0], fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("unsupported rate (per seed)", fontsize=8)
    np.atleast_1d(axes)[-1].legend(fontsize=7, frameon=False)
    return save_figure(figure, out_dir, "fig04_seed_stability", formats, dpi)


def fig_latency(rows, out_dir, formats, dpi, rng):
    data, labels, colors = [], [], []
    for condition in CONDITIONS:
        values = [float(row["seconds"]) for row in rows if row.get("condition") == condition and row.get("seconds")]
        if values:
            data.append(values)
            labels.append(CONDITION_LABELS[condition])
            colors.append(CONDITION_COLORS[condition])
    if not data:
        return []
    figure, axis = plt.subplots(figsize=(4.6, 2.6), constrained_layout=True)
    boxes = axis.boxplot(data, positions=range(len(data)), widths=0.55, showfliers=False, patch_artist=True)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.4)
    for element in ["whiskers", "caps", "medians"]:
        for artist in boxes[element]:
            artist.set_color("black")
            artist.set_linewidth(0.5)
    medians = [np.median(values) for values in data]
    for position, median in enumerate(medians):
        axis.text(position, median * 1.15, f"{median:.1f}", ha="center", fontsize=6.5)
    axis.set_yscale("log")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    axis.set_ylabel("seconds per record (log)", fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    return save_figure(figure, out_dir, "fig05_latency", formats, dpi)


def fig_templates(rows, out_dir, formats, dpi, rng):
    models = model_names(rows)
    templates = sorted({row.get("template") for row in rows if row.get("template")})
    if not templates:
        return []
    figure, axis = plt.subplots(figsize=(4.6, 2.6), constrained_layout=True)
    width = 0.8 / max(len(templates), 1)
    template_colors = ["#4477aa", "#66ccee", "#ccbb44"][: len(templates)]
    for index, template in enumerate(templates):
        means, errors, positions = [], [], []
        for position, model in enumerate(models):
            subset = [
                row
                for row in rows
                if row.get("llm_model") == model and row.get("condition") == "grounded" and row.get("template") == template
            ]
            values = [float(row["unsupported_rate"]) for row in subset if row.get("unsupported_rate") is not None]
            seeds = [str(row.get("seed")) for row in subset if row.get("unsupported_rate") is not None]
            mean, low, high = bootstrap_mean_ci(values, rng, seeds if len(set(seeds)) >= 2 else None)
            means.append(mean)
            errors.append([max(mean - low, 0.0), max(high - mean, 0.0)])
            positions.append(position + (index - (len(templates) - 1) / 2) * width)
        axis.bar(positions, means, width=width, yerr=np.asarray(errors).T, capsize=2,
                 color=template_colors[index], edgecolor="black", linewidth=0.4, label=template)
    axis.set_xticks(range(len(models)))
    axis.set_xticklabels([model.split(":")[0] for model in models], fontsize=8)
    axis.set_ylabel("unsupported rate (grounded)", fontsize=8)
    axis.legend(fontsize=7, frameon=False, title="template", title_fontsize=7)
    axis.spines[["top", "right"]].set_visible(False)
    return save_figure(figure, out_dir, "fig06_templates", formats, dpi)


def fig_pvalue_heatmap(stats, out_dir, formats, dpi, rng):
    matrix = stats.get("pairwise", {}).get("unsupported_rate", {}).get("p_holm")
    conditions = stats.get("pairwise", {}).get("unsupported_rate", {}).get("conditions")
    if not matrix or not conditions:
        return []
    values = np.full((len(conditions), len(conditions)), np.nan)
    for i in range(len(conditions)):
        for j in range(len(conditions)):
            value = matrix[i][j]
            if i > j and value is not None:
                values[i, j] = -math.log10(max(value, 1e-300))
    masked = np.ma.masked_invalid(values)
    figure, axis = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
    color_map = plt.get_cmap("YlOrRd").copy()
    color_map.set_bad("white")
    image = axis.imshow(masked, cmap=color_map, vmin=0, vmax=max(3.0, float(np.nanmax(values)) if np.any(~np.isnan(values)) else 3.0))
    labels = [CONDITION_LABELS[c] for c in conditions]
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=7)
    axis.set_yticks(range(len(labels)), labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i > j and not np.isnan(values[i, j]):
                p_value = 10 ** (-values[i, j])
                stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                axis.text(j, i, stars, ha="center", va="center", fontsize=6.5,
                          color="white" if values[i, j] > 2.2 else "black")
    figure.colorbar(image, ax=axis, shrink=0.8, label=r"$-\log_{10}(p_{Holm})$")
    return save_figure(figure, out_dir, "fig07_pvalue_heatmap", formats, dpi)


def fig_critical_difference(stats, out_dir, formats, dpi, rng):
    entry = stats.get("pairwise", {}).get("unsupported_rate", {})
    ranks = entry.get("mean_ranks")
    conditions = entry.get("conditions")
    cd = entry.get("nemenyi_cd")
    if not ranks or not conditions:
        return []
    size = len(conditions)
    figure, axis = plt.subplots(figsize=(5.6, 2.4), constrained_layout=True)
    axis.set_xlim(0.5, size + 0.9)
    axis.set_ylim(-0.6, 1.4)
    axis.axis("off")
    axis.hlines(1.15, 1, size, color="black", linewidth=0.8)
    for tick in range(1, size + 1):
        axis.vlines(tick, 1.08, 1.22, color="black", linewidth=0.8)
        axis.text(tick, 1.28, str(tick), ha="center", fontsize=7)
    axis.text(size + 0.15, 1.28, "rank", fontsize=7, ha="left")
    ordered = sorted(range(size), key=lambda index: ranks[index])
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
    seen_ranks: dict[float, int] = {}
    for slot, index in enumerate(ordered):
        rank = ranks[index]
        seen_ranks[rank] = seen_ranks.get(rank, 0) + 1
        if rank_counts[rank] > 1:
            height = 0.82 if seen_ranks[rank] == 1 else 0.34
        else:
            height = 0.78 if slot % 2 == 0 else 0.5
        axis.plot([ranks[index], ranks[index]], [height, 0.98 if slot % 2 == 0 else 0.72], color="black", linewidth=0.6)
        axis.text(ranks[index], height - 0.06, CONDITION_LABELS[conditions[index]], ha="center", va="top", fontsize=7,
                  color=CONDITION_COLORS[conditions[index]])
    if cd:
        axis.plot([1.0, 1.0 + cd], [-0.35, -0.35], color="black", linewidth=1.2)
        axis.plot([1.0, 1.0], [-0.42, -0.28], color="black", linewidth=0.8)
        axis.plot([1.0 + cd, 1.0 + cd], [-0.42, -0.28], color="black", linewidth=0.8)
        axis.text(1.0 + cd / 2, -0.52, f"CD = {cd:.2f}", ha="center", fontsize=7)
    return save_figure(figure, out_dir, "fig08_critical_difference", formats, dpi)


def fig_radar(rows, out_dir, formats, dpi, rng):
    models = model_names(rows)
    if not models:
        return []
    axes_labels = ["faithfulness", "no contradict.", "CF consist.", "compreh.", "suffic.", "effic."]
    values_by_model = {}
    for model in models:
        subset = [row for row in rows if row.get("llm_model") == model]
        unsupported = [float(row["unsupported_rate"]) for row in subset if row.get("unsupported_rate") is not None]
        contradiction = [float(row["contradiction_rate"]) for row in subset if row.get("contradiction_rate") is not None]
        cf_consistency = [float(row["cf_consistency"]) for row in subset if row.get("cf_consistency") is not None]
        comprehensiveness = [float(row["comprehensiveness"]) for row in subset if row.get("comprehensiveness") is not None]
        sufficiency = [float(row["sufficiency"]) for row in subset if row.get("sufficiency") is not None]
        grounded_seconds = [
            float(row["seconds"])
            for row in subset
            if row.get("condition") == "grounded" and row.get("seconds")
        ]
        values_by_model[model] = [
            1.0 - (float(np.mean(unsupported)) if unsupported else float("nan")),
            1.0 - (float(np.mean(contradiction)) if contradiction else float("nan")),
            float(np.mean(cf_consistency)) if cf_consistency else float("nan"),
            float(np.mean(comprehensiveness)) if comprehensiveness else float("nan"),
            float(np.mean(sufficiency)) if sufficiency else float("nan"),
            1.0 / (float(np.median(grounded_seconds)) + 1e-9) if grounded_seconds else float("nan"),
        ]
    matrix = np.asarray([values_by_model[model] for model in models], dtype=float)
    normalized = np.empty_like(matrix)
    for column in range(matrix.shape[1]):
        column_values = matrix[:, column]
        minimum, maximum = np.nanmin(column_values), np.nanmax(column_values)
        span = maximum - minimum
        normalized[:, column] = (column_values - minimum) / span if span > 0 else 0.5
    angles = np.linspace(0, 2 * math.pi, len(axes_labels), endpoint=False).tolist()
    angles += angles[:1]
    figure, axis = plt.subplots(figsize=(4.4, 3.8), subplot_kw=dict(polar=True), constrained_layout=True)
    for index, model in enumerate(models):
        data = normalized[index].tolist()
        data += data[:1]
        axis.plot(angles, data, linewidth=1.3, marker="o", markersize=3.2, color=model_color(model), label=model.split(":")[0])
        axis.fill(angles, data, alpha=0.18, color=model_color(model))
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(axes_labels, fontsize=7)
    axis.tick_params(axis="x", pad=10)
    axis.set_yticklabels([])
    axis.legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1.22, 1.06))
    return save_figure(figure, out_dir, "fig09_radar", formats, dpi)


def fig_pareto(rows, out_dir, formats, dpi, rng):
    models = model_names(rows)
    if not models:
        return []
    figure, axis = plt.subplots(figsize=(4.6, 2.8), constrained_layout=True)
    points = []
    for model in models:
        for condition in CONDITIONS:
            subset = [
                row
                for row in rows
                if row.get("llm_model") == model and row.get("condition") == condition
            ]
            seconds = [float(row["seconds"]) for row in subset if row.get("seconds")]
            unsupported = [float(row["unsupported_rate"]) for row in subset if row.get("unsupported_rate") is not None]
            if not seconds or not unsupported:
                continue
            x_value = float(np.median(seconds))
            y_value = float(np.mean(unsupported))
            points.append((x_value, y_value))
            axis.scatter([x_value], [y_value], s=26, color=model_color(model),
                         marker="o" if condition == "grounded" else "x",
                         label=f"{model.split(':')[0]} — {CONDITION_LABELS[condition]}" if condition == "grounded" else None)
            if condition == "grounded":
                axis.annotate(model.split(":")[0], (x_value, y_value), textcoords="offset points",
                              xytext=(4, 4), fontsize=6.5, color=model_color(model))
    if points:
        frontier = []
        for x_value, y_value in sorted(points):
            if not frontier or y_value < frontier[-1][1]:
                frontier.append((x_value, y_value))
        axis.plot([x for x, _ in frontier], [y for _, y in frontier], linestyle="--", linewidth=0.8, color="black", label="frontier")
    axis.set_xscale("log")
    axis.set_xlabel("median seconds per record (log)", fontsize=8)
    axis.set_ylabel("unsupported rate", fontsize=8)
    axis.legend(fontsize=6.5, frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    return save_figure(figure, out_dir, "fig10_pareto", formats, dpi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/grid_q1/analysis/pooled_metrics.json")
    parser.add_argument("--stats", default="results/grid_q1/analysis/stats/grid_stats.json")
    parser.add_argument("--out-dir", default="results/grid_q1/analysis/figures")
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 8, "figure.dpi": 110, "savefig.transparent": False, "axes.linewidth": 0.6})

    rows = json.loads(Path(args.metrics).read_text())["per_record"]
    stats_path = Path(args.stats)
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    rng = np.random.default_rng(SEED)

    written = []
    for builder in (
        fig_main_unsupported,
        fig_contradiction,
        fig_fidelity_decomposition,
        fig_seed_stability,
        fig_latency,
        fig_templates,
        fig_pvalue_heatmap,
        fig_critical_difference,
        fig_radar,
        fig_pareto,
    ):
        if builder in (fig_pvalue_heatmap, fig_critical_difference):
            written.extend(builder(stats, out_dir, formats, args.dpi, rng))
        else:
            written.extend(builder(rows, out_dir, formats, args.dpi, rng))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
