"""Tests for the grid-scope statistical battery (code/stats/grid_stats.py)."""

from __future__ import annotations

import numpy as np

from code.stats.grid_stats import (
    average_ranks,
    cliffs_delta,
    cluster_bootstrap_ci,
    friedman_summary,
    paired_tests,
    pairwise_matrix,
    rank_biserial,
    wilcoxon_p,
)


def test_cliffs_delta_extremes() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    assert cliffs_delta(a, b) == 1.0
    assert cliffs_delta(b, a) == -1.0
    reversed_b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert abs(cliffs_delta(a, reversed_b)) < 1e-12


def test_rank_biserial_range_and_extremes() -> None:
    diffs = np.array([1.0, 2.0, -1.0, -2.0, 3.0])
    value = rank_biserial(diffs)
    assert -1.0 <= value <= 1.0
    assert rank_biserial(np.array([1.0, 1.0, 1.0])) == 1.0
    assert rank_biserial(np.array([-1.0, -2.0])) == -1.0
    assert rank_biserial(np.zeros(3)) == 0.0


def test_wilcoxon_all_zero_guard() -> None:
    statistic, p_value = wilcoxon_p(np.zeros(20))
    assert statistic == 0.0
    assert p_value == 1.0


def test_cluster_bootstrap_covers_true_difference() -> None:
    rng = np.random.default_rng(0)
    values_a = rng.normal(1.0, 0.5, 200)
    values_b = values_a - 0.3
    seeds = np.array([index % 10 for index in range(200)])
    low, high = cluster_bootstrap_ci(values_a, values_b, seeds, rng)
    assert low <= 0.3 <= high
    assert low <= high


def test_friedman_perfect_ranking() -> None:
    count = 12
    table = np.vstack([np.zeros(count), np.ones(count), 2.0 * np.ones(count)]).T
    result = friedman_summary(table)
    assert result["p"] < 0.01
    assert result["mean_ranks"] == [1.0, 2.0, 3.0]
    assert result["n_blocks"] == 12


def test_paired_tests_detect_known_direction() -> None:
    rng = np.random.default_rng(1)
    rows = []
    condition_deltas = [
        ("grounded", 0.0),
        ("free", 0.5),
        ("single_block", 0.6),
        ("sabotage", 0.2),
        ("occlusion_attributions", 0.1),
    ]
    for seed in range(4):
        for request in range(30):
            base = float(rng.random())
            for condition, delta in condition_deltas:
                rows.append(
                    {
                        "request_id": f"{seed}-r{request}",
                        "condition": condition,
                        "llm_model": "model-a",
                        "template": "t1",
                        "unsupported_rate": base + delta,
                    }
                )
    tests = paired_tests(rows, "unsupported_rate", "model-a", np.random.default_rng(0))
    assert len(tests) == 4
    grounded_free = [test for test in tests if test["comparison"] == "grounded vs free"][0]
    assert grounded_free["mean_diff"] < 0
    assert grounded_free["p_raw"] < 0.01
    assert grounded_free["cliff_delta"] < -0.5
    assert grounded_free["rank_biserial"] == -1.0


def test_average_ranks_with_ties() -> None:
    assert average_ranks([10.0, 20.0, 20.0, 40.0]) == [1.0, 2.5, 2.5, 4.0]
    assert average_ranks([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_pairwise_matrix_orders_and_detects() -> None:
    rows = []
    shapes = [
        ("template_floor", 0.1),
        ("grounded", 0.0),
        ("sabotage", 0.5),
        ("occlusion_attributions", 0.3),
        ("free", 0.9),
        ("single_block", 0.8),
    ]
    for seed in range(3):
        for request in range(12):
            for condition, value in shapes:
                rows.append(
                    {
                        "request_id": f"s{seed}-r{request}",
                        "condition": condition,
                        "unsupported_rate": value,
                    }
                )
    result = pairwise_matrix(rows, "unsupported_rate")
    assert result["n_blocks"] == 36
    grounded = result["conditions"].index("grounded")
    free = result["conditions"].index("free")
    assert result["p_holm"][grounded][free] < 0.01
    assert result["mean_ranks"][grounded] == 1.0
    assert result["mean_ranks"][free] == 6.0
    assert result["nemenyi_cd"] is not None


def test_paired_tests_averages_templates_per_request() -> None:
    """Request ids repeat across templates: values must be averaged, not collapsed."""
    rows = []
    for seed in range(2):
        for request in range(30):
            for template, grounded_value in (("t1", 0.0), ("t2", 0.2), ("t3", 0.4)):
                rows.append(
                    {
                        "request_id": f"s{seed}-r{request}",
                        "condition": "grounded",
                        "llm_model": "m",
                        "template": template,
                        "unsupported_rate": grounded_value,
                    }
                )
                rows.append(
                    {
                        "request_id": f"s{seed}-r{request}",
                        "condition": "free",
                        "llm_model": "m",
                        "template": template,
                        "unsupported_rate": 1.0,
                    }
                )
    tests = paired_tests(rows, "unsupported_rate", "m", np.random.default_rng(0))
    grounded_free = [test for test in tests if test["comparison"] == "grounded vs free"][0]
    assert grounded_free["n"] == 60  # 2 seeds x 30 requests; templates averaged
    assert abs(grounded_free["mean_a"] - 0.2) < 1e-12  # (0.0 + 0.2 + 0.4) / 3
    assert abs(grounded_free["mean_b"] - 1.0) < 1e-12
