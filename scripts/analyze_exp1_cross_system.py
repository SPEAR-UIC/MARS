#!/usr/bin/env python3
"""Compare exp1a and exp1b heuristic behavior with simple regression analysis.

The analysis matches identical driver tags across the two experiments, normalizes
wait time and utilization against each system's FCFS baseline, and then asks:
do the same heuristic/window choices move performance in similar directions on
both systems?

Outputs are written under results/exp1analysis by default:
  - matched_policy_points.csv
  - family_summary.csv
  - regression_summary.csv
  - cross_system_regression_all_points.png
  - cross_system_regression_family_means.png
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


EXP1A_WAIT_DEFAULT = Path("results/exp1a_plots/table_wait_overall.csv")
EXP1A_UTIL_DEFAULT = Path("results/exp1a_plots/table_util_overall.csv")
EXP1B_WAIT_DEFAULT = Path("results/exp1b_plots/table_wait_overall.csv")
EXP1B_UTIL_DEFAULT = Path("results/exp1b_plots/table_util_overall.csv")
OUTPUT_DIR_DEFAULT = Path("results/exp1analysis")

SYSTEM_A_LABEL = "Theta / exp1a"
SYSTEM_B_LABEL = "Polaris / exp1b"

FIG_W = 12.0
FIG_H = 5.2
DPI = 220
FONT_SIZE = 11
TICK_SIZE = 10
LEGEND_SIZE = 9

SINGLE_COL_FIG_W = 3.45
SINGLE_COL_FIG_H = 2.95
COMBINED_FIG_W = 4.0
COMBINED_FIG_H = 4.0
SINGLE_COL_AXIS_FONT = 7.8
SINGLE_COL_TICK_FONT = 6.2
SINGLE_COL_LEGEND_FONT = 6.4
SINGLE_COL_MARKER_SIZE = 14
SINGLE_COL_MEAN_SIZE = 34

CHARACTERISTIC_FAMILIES = (
    "FCFS",
    "SJF",
    "WFP3",
    "F1",
    "UNICEP",
    "FAT",
    "LRF",
    "LJF",
)
WINDOW_CHARACTERISTIC_FAMILIES = tuple(family for family in CHARACTERISTIC_FAMILIES if family != "FCFS")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple cross-system regression analysis for exp1a vs exp1b."
    )
    parser.add_argument("--exp1a-wait-csv", default=str(EXP1A_WAIT_DEFAULT))
    parser.add_argument("--exp1a-util-csv", default=str(EXP1A_UTIL_DEFAULT))
    parser.add_argument("--exp1b-wait-csv", default=str(EXP1B_WAIT_DEFAULT))
    parser.add_argument("--exp1b-util-csv", default=str(EXP1B_UTIL_DEFAULT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR_DEFAULT))
    return parser.parse_args(argv)


def parse_driver(driver: str) -> tuple[str, int | None]:
    if "-w" not in driver:
        return driver, None
    family, window = driver.rsplit("-w", 1)
    return family, int(window)


def sort_driver_key(driver: str) -> tuple[str, int]:
    family, window = parse_driver(driver)
    return family, -1 if window is None else window


def load_value_table(path: Path, value_column: str) -> OrderedDict[str, float]:
    rows: OrderedDict[str, float] = OrderedDict()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row["Driver"]] = float(row[value_column])
    return rows


def rankdata(values: list[float]) -> list[float]:
    ordered = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for _, idx in ordered[i:j]:
            ranks[idx] = rank
        i = j
    return ranks


def linear_regression(x_values: list[float], y_values: list[float]) -> dict[str, float]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    sxx = float(np.sum((x - x_mean) ** 2))
    sxy = float(np.sum((x - x_mean) * (y - y_mean)))
    syy = float(np.sum((y - y_mean) ** 2))

    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = y_mean - slope * x_mean
    pearson_r = sxy / np.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0
    r2 = pearson_r ** 2

    predicted = intercept + slope * x
    rmse = float(np.sqrt(np.mean((y - predicted) ** 2)))

    rank_x = rankdata(list(x))
    rank_y = rankdata(list(y))
    rank_x_mean = float(np.mean(rank_x))
    rank_y_mean = float(np.mean(rank_y))
    rsxx = float(sum((v - rank_x_mean) ** 2 for v in rank_x))
    rsyy = float(sum((v - rank_y_mean) ** 2 for v in rank_y))
    rsxy = float(sum((vx - rank_x_mean) * (vy - rank_y_mean) for vx, vy in zip(rank_x, rank_y)))
    spearman_rho = rsxy / np.sqrt(rsxx * rsyy) if rsxx > 0 and rsyy > 0 else 0.0

    return {
        "n_points": int(len(x)),
        "intercept": intercept,
        "slope": slope,
        "pearson_r": pearson_r,
        "r2": r2,
        "spearman_rho": spearman_rho,
        "rmse": rmse,
    }


def build_matched_rows(
    wait_a: OrderedDict[str, float],
    util_a: OrderedDict[str, float],
    wait_b: OrderedDict[str, float],
    util_b: OrderedDict[str, float],
) -> list[dict[str, object]]:
    common_drivers = sorted(set(wait_a) & set(util_a) & set(wait_b) & set(util_b), key=sort_driver_key)
    if "FCFS" not in common_drivers:
        raise ValueError("FCFS baseline missing from matched driver set")

    fcfs_wait_a = wait_a["FCFS"]
    fcfs_wait_b = wait_b["FCFS"]
    fcfs_util_a = util_a["FCFS"]
    fcfs_util_b = util_b["FCFS"]

    rows = []
    for driver in common_drivers:
        family, window = parse_driver(driver)
        wait_ratio_a = wait_a[driver] / fcfs_wait_a
        wait_ratio_b = wait_b[driver] / fcfs_wait_b
        wait_reduction_a = 1.0 - wait_ratio_a
        wait_reduction_b = 1.0 - wait_ratio_b
        util_delta_a = util_a[driver] - fcfs_util_a
        util_delta_b = util_b[driver] - fcfs_util_b
        util_ratio_a = util_a[driver] / fcfs_util_a
        util_ratio_b = util_b[driver] / fcfs_util_b
        util_fraction_a = util_a[driver] / 100.0
        util_fraction_b = util_b[driver] / 100.0
        fcfs_util_fraction_a = fcfs_util_a / 100.0
        fcfs_util_fraction_b = fcfs_util_b / 100.0
        util_delta_ratio_a = util_fraction_a - fcfs_util_fraction_a
        util_delta_ratio_b = util_fraction_b - fcfs_util_fraction_b

        rows.append(
            {
                "driver": driver,
                "family": family,
                "window": "" if window is None else window,
                "wait_mean_exp1a_min": wait_a[driver],
                "wait_mean_exp1b_min": wait_b[driver],
                "util_exp1a_pct": util_a[driver],
                "util_exp1b_pct": util_b[driver],
                "wait_ratio_exp1a": wait_ratio_a,
                "wait_ratio_exp1b": wait_ratio_b,
                "wait_reduction_exp1a": wait_reduction_a,
                "wait_reduction_exp1b": wait_reduction_b,
                "util_delta_exp1a_pp": util_delta_a,
                "util_delta_exp1b_pp": util_delta_b,
                "util_ratio_exp1a_vs_fcfs": util_ratio_a,
                "util_ratio_exp1b_vs_fcfs": util_ratio_b,
                "util_fraction_exp1a": util_fraction_a,
                "util_fraction_exp1b": util_fraction_b,
                "util_delta_ratio_exp1a": util_delta_ratio_a,
                "util_delta_ratio_exp1b": util_delta_ratio_b,
            }
        )

    return rows


def build_family_rows(matched_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in matched_rows:
        grouped[str(row["family"])].append(row)

    family_rows = []
    for family in sorted(grouped.keys()):
        rows = grouped[family]
        family_rows.append(
            {
                "family": family,
                "n_points": len(rows),
                "mean_wait_reduction_exp1a": float(np.mean([float(r["wait_reduction_exp1a"]) for r in rows])),
                "mean_wait_reduction_exp1b": float(np.mean([float(r["wait_reduction_exp1b"]) for r in rows])),
                "mean_wait_reduction_diff_b_minus_a": float(
                    np.mean([float(r["wait_reduction_exp1b"]) - float(r["wait_reduction_exp1a"]) for r in rows])
                ),
                "mean_util_delta_exp1a_pp": float(np.mean([float(r["util_delta_exp1a_pp"]) for r in rows])),
                "mean_util_delta_exp1b_pp": float(np.mean([float(r["util_delta_exp1b_pp"]) for r in rows])),
                "mean_util_delta_diff_b_minus_a_pp": float(
                    np.mean([float(r["util_delta_exp1b_pp"]) - float(r["util_delta_exp1a_pp"]) for r in rows])
                ),
                "mean_util_ratio_exp1a_vs_fcfs": float(np.mean([float(r["util_ratio_exp1a_vs_fcfs"]) for r in rows])),
                "mean_util_ratio_exp1b_vs_fcfs": float(np.mean([float(r["util_ratio_exp1b_vs_fcfs"]) for r in rows])),
                "mean_util_delta_ratio_exp1a": float(np.mean([float(r["util_delta_ratio_exp1a"]) for r in rows])),
                "mean_util_delta_ratio_exp1b": float(np.mean([float(r["util_delta_ratio_exp1b"]) for r in rows])),
            }
        )
    return family_rows


def write_csv(path: Path, rows: list[dict[str, object]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def build_regression_rows(matched_rows: list[dict[str, object]], family_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    action_rows = [row for row in matched_rows if row["driver"] != "FCFS"]
    family_non_fcfs = [row for row in family_rows if row["family"] != "FCFS"]

    wait_action = linear_regression(
        [float(row["wait_reduction_exp1a"]) for row in action_rows],
        [float(row["wait_reduction_exp1b"]) for row in action_rows],
    )
    util_action = linear_regression(
        [float(row["util_delta_exp1a_pp"]) for row in action_rows],
        [float(row["util_delta_exp1b_pp"]) for row in action_rows],
    )
    wait_family = linear_regression(
        [float(row["mean_wait_reduction_exp1a"]) for row in family_non_fcfs],
        [float(row["mean_wait_reduction_exp1b"]) for row in family_non_fcfs],
    )
    util_family = linear_regression(
        [float(row["mean_util_delta_exp1a_pp"]) for row in family_non_fcfs],
        [float(row["mean_util_delta_exp1b_pp"]) for row in family_non_fcfs],
    )

    rows = []
    for sample, metric, summary in (
        ("matched_actions", "wait_reduction_vs_fcfs", wait_action),
        ("matched_actions", "util_delta_vs_fcfs_pp", util_action),
        ("family_means", "wait_reduction_vs_fcfs", wait_family),
        ("family_means", "util_delta_vs_fcfs_pp", util_family),
    ):
        rows.append({"sample": sample, "metric": metric, **summary})
    return rows


def build_window_step_rows(matched_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in matched_rows:
        family = str(row["family"])
        if family == "FCFS" or row["window"] == "":
            continue
        grouped[family].append(row)

    step_rows = []
    for family, rows in grouped.items():
        ordered_rows = sorted(rows, key=lambda row: int(row["window"]))
        for start_row, end_row in zip(ordered_rows, ordered_rows[1:]):
            start_window = int(start_row["window"])
            end_window = int(end_row["window"])
            step_rows.append(
                {
                    "family": family,
                    "driver_from": start_row["driver"],
                    "driver_to": end_row["driver"],
                    "window_from": start_window,
                    "window_to": end_window,
                    "window_step": f"{start_window}->{end_window}",
                    "wait_step_exp1a": float(end_row["wait_reduction_exp1a"]) - float(start_row["wait_reduction_exp1a"]),
                    "wait_step_exp1b": float(end_row["wait_reduction_exp1b"]) - float(start_row["wait_reduction_exp1b"]),
                    "util_step_exp1a": float(end_row["util_delta_ratio_exp1a"]) - float(start_row["util_delta_ratio_exp1a"]),
                    "util_step_exp1b": float(end_row["util_delta_ratio_exp1b"]) - float(start_row["util_delta_ratio_exp1b"]),
                }
            )

    step_rows.sort(key=lambda row: (str(row["family"]), int(row["window_from"])))
    return step_rows


def build_window_step_family_rows(step_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in step_rows:
        grouped[str(row["family"])].append(row)

    family_rows = []
    for family in sorted(grouped.keys()):
        rows = grouped[family]
        family_rows.append(
            {
                "family": family,
                "n_steps": len(rows),
                "mean_wait_step_exp1a": float(np.mean([float(row["wait_step_exp1a"]) for row in rows])),
                "mean_wait_step_exp1b": float(np.mean([float(row["wait_step_exp1b"]) for row in rows])),
                "mean_util_step_exp1a": float(np.mean([float(row["util_step_exp1a"]) for row in rows])),
                "mean_util_step_exp1b": float(np.mean([float(row["util_step_exp1b"]) for row in rows])),
            }
        )
    return family_rows


def build_window_step_regression_rows(
    step_rows: list[dict[str, object]],
    family_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    wait_action = linear_regression(
        [float(row["wait_step_exp1a"]) for row in step_rows],
        [float(row["wait_step_exp1b"]) for row in step_rows],
    )
    util_action = linear_regression(
        [float(row["util_step_exp1a"]) for row in step_rows],
        [float(row["util_step_exp1b"]) for row in step_rows],
    )
    wait_family = linear_regression(
        [float(row["mean_wait_step_exp1a"]) for row in family_rows],
        [float(row["mean_wait_step_exp1b"]) for row in family_rows],
    )
    util_family = linear_regression(
        [float(row["mean_util_step_exp1a"]) for row in family_rows],
        [float(row["mean_util_step_exp1b"]) for row in family_rows],
    )

    rows = []
    for sample, metric, summary in (
        ("matched_window_steps", "wait_window_step_transfer", wait_action),
        ("matched_window_steps", "util_window_step_transfer", util_action),
        ("family_mean_window_steps", "wait_window_step_transfer", wait_family),
        ("family_mean_window_steps", "util_window_step_transfer", util_family),
    ):
        rows.append({"sample": sample, "metric": metric, **summary})
    return rows


def family_order(matched_rows: list[dict[str, object]]) -> list[str]:
    ordered = []
    seen = set()
    for row in matched_rows:
        family = str(row["family"])
        if family not in seen:
            seen.add(family)
            ordered.append(family)
    return ordered


def reorder_handles_for_row_major(handles, labels, ncol: int):
    rows = [
        list(zip(handles[i:i + ncol], labels[i:i + ncol]))
        for i in range(0, len(handles), ncol)
    ]
    ordered = []
    max_cols = max((len(row) for row in rows), default=0)
    for col_idx in range(max_cols):
        for row in rows:
            if col_idx < len(row):
                ordered.append(row[col_idx])
    return [handle for handle, _ in ordered], [label for _, label in ordered]


def filter_rows_by_family(rows: list[dict[str, object]], families: tuple[str, ...]) -> list[dict[str, object]]:
    wanted = set(families)
    filtered = [row for row in rows if str(row["family"]) in wanted]
    family_rank = {family: idx for idx, family in enumerate(families)}
    filtered.sort(
        key=lambda row: (
            family_rank[str(row["family"])],
            sort_driver_key(str(row["driver"])) if "driver" in row else (str(row["family"]), -1),
        )
    )
    return filtered


def family_color_map(families: list[str]) -> dict[str, tuple[float, float, float] | str]:
    palette = list(plt.get_cmap("tab20").colors)
    colors: dict[str, tuple[float, float, float] | str] = {}
    idx = 0
    for family in families:
        if family == "FCFS":
            colors[family] = "#000000"
        else:
            colors[family] = palette[idx % len(palette)]
            idx += 1
    return colors


def lookup_regression(regression_rows: list[dict[str, object]], sample: str, metric: str) -> dict[str, object]:
    for row in regression_rows:
        if row["sample"] == sample and row["metric"] == metric:
            return row
    raise KeyError((sample, metric))


def apply_common_style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_axisbelow(True)
    ax.grid(True, color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def apply_single_col_style(ax, xlabel: str, ylabel: str, *, right_y: bool = False) -> None:
    ax.set_xlabel(xlabel, fontsize=SINGLE_COL_AXIS_FONT)
    ax.set_ylabel(ylabel, fontsize=SINGLE_COL_AXIS_FONT)
    ax.tick_params(axis="both", labelsize=SINGLE_COL_TICK_FONT)
    ax.set_axisbelow(True)
    ax.grid(True, color="#e6e6e6", linewidth=0.55, zorder=0)
    ax.axhline(0.0, color="#b3b3b3", linestyle=":", linewidth=0.6, zorder=1)
    ax.axvline(0.0, color="#b3b3b3", linestyle=":", linewidth=0.6, zorder=1)
    ax.spines["top"].set_visible(False)
    if right_y:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        ax.spines["right"].set_visible(False)


def _plot_reference_lines(ax, x_values: list[float], y_values: list[float], regression: dict[str, object]) -> None:
    lim_min = min(min(x_values), min(y_values))
    lim_max = max(max(x_values), max(y_values))
    pad = 0.06 * (lim_max - lim_min if lim_max > lim_min else 1.0)
    lo = lim_min - pad
    hi = lim_max + pad
    ax.plot([lo, hi], [lo, hi], color="#9a9a9a", linestyle="--", linewidth=1.1, label="y = x", zorder=1.4)
    ax.plot(
        [lo, hi],
        [
            float(regression["intercept"]) + float(regression["slope"]) * lo,
            float(regression["intercept"]) + float(regression["slope"]) * hi,
        ],
        color="#222222",
        linewidth=1.6,
        label="Regression fit",
        zorder=1.6,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def plot_all_points_figure(
    matched_rows: list[dict[str, object]],
    family_rows: list[dict[str, object]],
    output_path: Path,
) -> list[dict[str, object]]:
    selected_rows = filter_rows_by_family(matched_rows, CHARACTERISTIC_FAMILIES)
    selected_family_rows = filter_rows_by_family(family_rows, CHARACTERISTIC_FAMILIES)
    selected_families = [family for family in CHARACTERISTIC_FAMILIES if any(row["family"] == family for row in selected_rows)]
    colors = family_color_map(selected_families)
    family_means = {str(row["family"]): row for row in selected_family_rows}
    non_fcfs_rows = [row for row in selected_rows if row["driver"] != "FCFS"]

    wait_reg = linear_regression(
        [float(row["wait_reduction_exp1a"]) for row in non_fcfs_rows],
        [float(row["wait_reduction_exp1b"]) for row in non_fcfs_rows],
    )
    util_reg = linear_regression(
        [float(row["util_delta_ratio_exp1a"]) for row in non_fcfs_rows],
        [float(row["util_delta_ratio_exp1b"]) for row in non_fcfs_rows],
    )
    selected_regression_rows = [
        {"sample": "characteristic_subset", "metric": "wait_reduction_vs_fcfs", **wait_reg},
        {"sample": "characteristic_subset", "metric": "util_delta_ratio_vs_fcfs", **util_reg},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL_FIG_W, SINGLE_COL_FIG_H), dpi=DPI)

    # Wait panel
    wait_x = [float(row["wait_reduction_exp1a"]) for row in non_fcfs_rows]
    wait_y = [float(row["wait_reduction_exp1b"]) for row in non_fcfs_rows]
    for family in selected_families:
        rows = [row for row in non_fcfs_rows if row["family"] == family]
        if not rows:
            continue
        axes[0].scatter(
            [float(row["wait_reduction_exp1a"]) for row in rows],
            [float(row["wait_reduction_exp1b"]) for row in rows],
            s=SINGLE_COL_MARKER_SIZE,
            color=colors[family],
            alpha=0.78,
            edgecolors="none",
            label=family,
            zorder=3,
        )
        mean_row = family_means[family]
        axes[0].scatter(
            [float(mean_row["mean_wait_reduction_exp1a"])],
            [float(mean_row["mean_wait_reduction_exp1b"])],
            s=SINGLE_COL_MEAN_SIZE,
            facecolors="none",
            edgecolors=colors[family],
            linewidths=1.0,
            zorder=3.2,
        )
    axes[0].scatter([0.0], [0.0], marker="*", s=36, color="black", label="FCFS", zorder=3.4)
    _plot_reference_lines(axes[0], wait_x + [0.0], wait_y + [0.0], wait_reg)
    apply_single_col_style(
        axes[0],
        r"$\%\Delta$ Avg Wait w.r.t FCFS" "\n" r"(Theta)",
        r"$\%\Delta$ Avg Wait w.r.t FCFS (Polaris)",
    )
    axes[0].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    axes[0].text(
        0.03,
        0.97,
        f"R^2 = {float(wait_reg['r2']):.2f}\nrho = {float(wait_reg['spearman_rho']):.2f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25", alpha=0.94),
    )

    # Utilization panel
    for family in selected_families:
        rows = [row for row in non_fcfs_rows if row["family"] == family]
        if not rows:
            continue
        axes[1].scatter(
            [float(row["util_delta_ratio_exp1a"]) for row in rows],
            [float(row["util_delta_ratio_exp1b"]) for row in rows],
            s=SINGLE_COL_MARKER_SIZE,
            color=colors[family],
            alpha=0.78,
            edgecolors="none",
            zorder=3,
        )
        mean_row = family_means[family]
        axes[1].scatter(
            [float(mean_row["mean_util_delta_ratio_exp1a"])],
            [float(mean_row["mean_util_delta_ratio_exp1b"])],
            s=SINGLE_COL_MEAN_SIZE,
            facecolors="none",
            edgecolors=colors[family],
            linewidths=1.0,
            zorder=3.2,
        )
    axes[1].scatter([0.0], [0.0], marker="*", s=36, color="black", zorder=3.4)
    util_x = [float(row["util_delta_ratio_exp1a"]) for row in non_fcfs_rows]
    util_y = [float(row["util_delta_ratio_exp1b"]) for row in non_fcfs_rows]
    _plot_reference_lines(axes[1], util_x + [0.0], util_y + [0.0], util_reg)
    apply_single_col_style(
        axes[1],
        r"$\%\Delta$ Utilization w.r.t FCFS" "\n" r"(Theta)",
        r"$\%\Delta$ Utilization w.r.t FCFS (Polaris)",
        right_y=True,
    )
    axes[1].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[1].text(
        0.03,
        0.97,
        f"R^2 = {float(util_reg['r2']):.2f}\nrho = {float(util_reg['spearman_rho']):.2f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25", alpha=0.94),
    )

    handles, labels = axes[0].get_legend_handles_labels()
    legend_map = OrderedDict()
    for handle, label in zip(handles, labels):
        if label in selected_families and label not in legend_map:
            legend_map[label] = handle
    handles = [legend_map[label] for label in selected_families if label in legend_map]
    labels = [label for label in selected_families if label in legend_map]
    handles, labels = reorder_handles_for_row_major(handles, labels, 4)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=SINGLE_COL_LEGEND_FONT,
        frameon=False,
        handlelength=1.3,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    fig.subplots_adjust(bottom=0.29, left=0.15, right=0.88, top=0.98, wspace=0.44)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return selected_regression_rows


def plot_window_step_preservation_figure(
    step_rows: list[dict[str, object]],
    family_rows: list[dict[str, object]],
    output_path: Path,
) -> list[dict[str, object]]:
    selected_rows = filter_rows_by_family(step_rows, WINDOW_CHARACTERISTIC_FAMILIES)
    selected_family_rows = filter_rows_by_family(family_rows, WINDOW_CHARACTERISTIC_FAMILIES)
    selected_families = [
        family for family in WINDOW_CHARACTERISTIC_FAMILIES if any(row["family"] == family for row in selected_rows)
    ]
    colors = family_color_map(selected_families)
    family_means = {str(row["family"]): row for row in selected_family_rows}

    wait_reg = linear_regression(
        [float(row["wait_step_exp1a"]) for row in selected_rows],
        [float(row["wait_step_exp1b"]) for row in selected_rows],
    )
    util_reg = linear_regression(
        [float(row["util_step_exp1a"]) for row in selected_rows],
        [float(row["util_step_exp1b"]) for row in selected_rows],
    )
    selected_regression_rows = [
        {"sample": "characteristic_window_steps", "metric": "wait_window_step_transfer", **wait_reg},
        {"sample": "characteristic_window_steps", "metric": "util_window_step_transfer", **util_reg},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL_FIG_W, SINGLE_COL_FIG_H), dpi=DPI)

    wait_x = [float(row["wait_step_exp1a"]) for row in selected_rows]
    wait_y = [float(row["wait_step_exp1b"]) for row in selected_rows]
    for family in selected_families:
        family_points = [row for row in selected_rows if row["family"] == family]
        axes[0].scatter(
            [float(row["wait_step_exp1a"]) for row in family_points],
            [float(row["wait_step_exp1b"]) for row in family_points],
            s=SINGLE_COL_MARKER_SIZE,
            color=colors[family],
            alpha=0.78,
            edgecolors="none",
            label=family,
            zorder=3,
        )
        mean_row = family_means[family]
        axes[0].scatter(
            [float(mean_row["mean_wait_step_exp1a"])],
            [float(mean_row["mean_wait_step_exp1b"])],
            s=SINGLE_COL_MEAN_SIZE,
            facecolors="none",
            edgecolors=colors[family],
            linewidths=1.0,
            zorder=3.2,
        )
    _plot_reference_lines(axes[0], wait_x, wait_y, wait_reg)
    apply_single_col_style(
        axes[0],
        r"Window-Step Change in" "\n" r"$\%\Delta$ Avg Wait (Theta)",
        r"Window-Step Change in" "\n" r"$\%\Delta$ Avg Wait (Polaris)",
    )
    axes[0].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    axes[0].text(
        0.03,
        0.97,
        f"R^2 = {float(wait_reg['r2']):.2f}\nrho = {float(wait_reg['spearman_rho']):.2f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25", alpha=0.94),
    )

    util_x = [float(row["util_step_exp1a"]) for row in selected_rows]
    util_y = [float(row["util_step_exp1b"]) for row in selected_rows]
    for family in selected_families:
        family_points = [row for row in selected_rows if row["family"] == family]
        axes[1].scatter(
            [float(row["util_step_exp1a"]) for row in family_points],
            [float(row["util_step_exp1b"]) for row in family_points],
            s=SINGLE_COL_MARKER_SIZE,
            color=colors[family],
            alpha=0.78,
            edgecolors="none",
            zorder=3,
        )
        mean_row = family_means[family]
        axes[1].scatter(
            [float(mean_row["mean_util_step_exp1a"])],
            [float(mean_row["mean_util_step_exp1b"])],
            s=SINGLE_COL_MEAN_SIZE,
            facecolors="none",
            edgecolors=colors[family],
            linewidths=1.0,
            zorder=3.2,
        )
    _plot_reference_lines(axes[1], util_x, util_y, util_reg)
    apply_single_col_style(
        axes[1],
        r"Window-Step Change in" "\n" r"$\%\Delta$ Utilization (Theta)",
        r"Window-Step Change in" "\n" r"$\%\Delta$ Utilization (Polaris)",
        right_y=True,
    )
    axes[1].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[1].text(
        0.03,
        0.97,
        f"R^2 = {float(util_reg['r2']):.2f}\nrho = {float(util_reg['spearman_rho']):.2f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25", alpha=0.94),
    )

    handles, labels = axes[0].get_legend_handles_labels()
    legend_map = OrderedDict()
    for handle, label in zip(handles, labels):
        if label in selected_families and label not in legend_map:
            legend_map[label] = handle
    handles = [legend_map[label] for label in selected_families if label in legend_map]
    labels = [label for label in selected_families if label in legend_map]
    handles, labels = reorder_handles_for_row_major(handles, labels, 4)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=SINGLE_COL_LEGEND_FONT,
        frameon=False,
        handlelength=1.3,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    fig.subplots_adjust(bottom=0.29, left=0.15, right=0.88, top=0.98, wspace=0.44)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return selected_regression_rows


def plot_combined_transfer_figure(
    matched_rows: list[dict[str, object]],
    family_rows: list[dict[str, object]],
    step_rows: list[dict[str, object]],
    step_family_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    selected_rows = filter_rows_by_family(matched_rows, CHARACTERISTIC_FAMILIES)
    selected_family_rows = filter_rows_by_family(family_rows, CHARACTERISTIC_FAMILIES)
    selected_families = [family for family in CHARACTERISTIC_FAMILIES if any(row["family"] == family for row in selected_rows)]
    colors = family_color_map(selected_families)
    family_means = {str(row["family"]): row for row in selected_family_rows}
    non_fcfs_rows = [row for row in selected_rows if row["driver"] != "FCFS"]

    selected_step_rows = filter_rows_by_family(step_rows, WINDOW_CHARACTERISTIC_FAMILIES)
    selected_step_family_rows = filter_rows_by_family(step_family_rows, WINDOW_CHARACTERISTIC_FAMILIES)
    step_means = {str(row["family"]): row for row in selected_step_family_rows}

    wait_reg = linear_regression(
        [float(row["wait_reduction_exp1a"]) for row in non_fcfs_rows],
        [float(row["wait_reduction_exp1b"]) for row in non_fcfs_rows],
    )
    util_reg = linear_regression(
        [float(row["util_delta_ratio_exp1a"]) for row in non_fcfs_rows],
        [float(row["util_delta_ratio_exp1b"]) for row in non_fcfs_rows],
    )
    wait_step_reg = linear_regression(
        [float(row["wait_step_exp1a"]) for row in selected_step_rows],
        [float(row["wait_step_exp1b"]) for row in selected_step_rows],
    )
    util_step_reg = linear_regression(
        [float(row["util_step_exp1a"]) for row in selected_step_rows],
        [float(row["util_step_exp1b"]) for row in selected_step_rows],
    )

    fig, axes = plt.subplots(2, 2, figsize=(COMBINED_FIG_W, COMBINED_FIG_H), dpi=DPI)

    for family in selected_families:
        if family == "FCFS":
            continue
        rows = [row for row in non_fcfs_rows if row["family"] == family]
        if not rows:
            continue
        axes[0, 0].scatter(
            [float(row["wait_reduction_exp1a"]) for row in rows],
            [float(row["wait_reduction_exp1b"]) for row in rows],
            s=SINGLE_COL_MARKER_SIZE,
            color=colors[family],
            alpha=0.78,
            edgecolors="none",
            label=family,
            zorder=3,
        )
    axes[0, 0].scatter([0.0], [0.0], marker="*", s=36, color="black", label="FCFS", zorder=3.4)
    _plot_reference_lines(
        axes[0, 0],
        [float(row["wait_reduction_exp1a"]) for row in non_fcfs_rows] + [0.0],
        [float(row["wait_reduction_exp1b"]) for row in non_fcfs_rows] + [0.0],
        wait_reg,
    )
    apply_single_col_style(
        axes[0, 0],
        r"$\%\Delta$ Avg Wait w.r.t FCFS" "\n" r"(Theta)",
        r"$\%\Delta$ Avg Wait" "\n" r"w.r.t FCFS (Polaris)",
    )
    axes[0, 0].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    axes[0, 0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    axes[0, 0].text(
        0.03,
        0.97,
        f"R^2 = {float(wait_reg['r2']):.2f}\nrho = {float(wait_reg['spearman_rho']):.2f}",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25", alpha=0.94),
    )

    for family in selected_families:
        if family == "FCFS":
            continue
        rows = [row for row in non_fcfs_rows if row["family"] == family]
        if not rows:
            continue
        axes[0, 1].scatter(
            [float(row["util_delta_ratio_exp1a"]) for row in rows],
            [float(row["util_delta_ratio_exp1b"]) for row in rows],
            s=SINGLE_COL_MARKER_SIZE,
            color=colors[family],
            alpha=0.78,
            edgecolors="none",
            zorder=3,
        )
    axes[0, 1].scatter([0.0], [0.0], marker="*", s=36, color="black", zorder=3.4)
    _plot_reference_lines(
        axes[0, 1],
        [float(row["util_delta_ratio_exp1a"]) for row in non_fcfs_rows] + [0.0],
        [float(row["util_delta_ratio_exp1b"]) for row in non_fcfs_rows] + [0.0],
        util_reg,
    )
    apply_single_col_style(
        axes[0, 1],
        r"$\%\Delta$ Utilization w.r.t FCFS" "\n" r"(Theta)",
        r"$\%\Delta$ Utilization" "\n" r"w.r.t FCFS (Polaris)",
        right_y=True,
    )
    axes[0, 1].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[0, 1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[0, 1].text(
        0.03,
        0.97,
        f"R^2 = {float(util_reg['r2']):.2f}\nrho = {float(util_reg['spearman_rho']):.2f}",
        transform=axes[0, 1].transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25", alpha=0.94),
    )

    for family in WINDOW_CHARACTERISTIC_FAMILIES:
        rows = [row for row in selected_step_rows if row["family"] == family]
        if not rows:
            continue
        axes[1, 0].scatter(
            [float(row["wait_step_exp1a"]) for row in rows],
            [float(row["wait_step_exp1b"]) for row in rows],
            s=SINGLE_COL_MARKER_SIZE,
            color=colors[family],
            alpha=0.78,
            edgecolors="none",
            zorder=3,
        )
    _plot_reference_lines(
        axes[1, 0],
        [float(row["wait_step_exp1a"]) for row in selected_step_rows],
        [float(row["wait_step_exp1b"]) for row in selected_step_rows],
        wait_step_reg,
    )
    apply_single_col_style(
        axes[1, 0],
        r"Window-Step Change in" "\n" r"$\%\Delta$ Avg Wait (Theta)",
        r"Window-Step Change in" "\n" r"$\%\Delta$ Avg Wait (Polaris)",
    )
    axes[1, 0].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    axes[1, 0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    axes[1, 0].text(
        0.03,
        0.97,
        f"R^2 = {float(wait_step_reg['r2']):.2f}\nrho = {float(wait_step_reg['spearman_rho']):.2f}",
        transform=axes[1, 0].transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25", alpha=0.94),
    )

    for family in WINDOW_CHARACTERISTIC_FAMILIES:
        rows = [row for row in selected_step_rows if row["family"] == family]
        if not rows:
            continue
        axes[1, 1].scatter(
            [float(row["util_step_exp1a"]) for row in rows],
            [float(row["util_step_exp1b"]) for row in rows],
            s=SINGLE_COL_MARKER_SIZE,
            color=colors[family],
            alpha=0.78,
            edgecolors="none",
            zorder=3,
        )
    _plot_reference_lines(
        axes[1, 1],
        [float(row["util_step_exp1a"]) for row in selected_step_rows],
        [float(row["util_step_exp1b"]) for row in selected_step_rows],
        util_step_reg,
    )
    apply_single_col_style(
        axes[1, 1],
        r"Window-Step Change in" "\n" r"$\%\Delta$ Utilization (Theta)",
        r"Window-Step Change in" "\n" r"$\%\Delta$ Utilization (Polaris)",
        right_y=True,
    )
    axes[1, 1].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[1, 1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[1, 1].text(
        0.03,
        0.97,
        f"R^2 = {float(util_step_reg['r2']):.2f}\nrho = {float(util_step_reg['spearman_rho']):.2f}",
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25", alpha=0.94),
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_map = OrderedDict()
    for handle, label in zip(handles, labels):
        if label in selected_families and label not in legend_map:
            legend_map[label] = handle
    legend_handles = [legend_map[label] for label in selected_families if label in legend_map]
    legend_labels = [label for label in selected_families if label in legend_map]
    legend_handles, legend_labels = reorder_handles_for_row_major(legend_handles, legend_labels, 4)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        fontsize=SINGLE_COL_LEGEND_FONT,
        frameon=False,
        handlelength=1.3,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    fig.subplots_adjust(bottom=0.26, left=0.15, right=0.88, top=0.99, wspace=0.44, hspace=0.55)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_family_means_figure(
    family_rows: list[dict[str, object]],
    regression_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    families = [str(row["family"]) for row in family_rows]
    colors = family_color_map(families)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), dpi=DPI)

    wait_reg = lookup_regression(regression_rows, "family_means", "wait_reduction_vs_fcfs")
    wait_x = [float(row["mean_wait_reduction_exp1a"]) for row in family_rows]
    wait_y = [float(row["mean_wait_reduction_exp1b"]) for row in family_rows]
    for row in family_rows:
        family = str(row["family"])
        x = float(row["mean_wait_reduction_exp1a"])
        y = float(row["mean_wait_reduction_exp1b"])
        marker = "*" if family == "FCFS" else "o"
        size = 150 if family == "FCFS" else 70
        axes[0].scatter([x], [y], s=size, marker=marker, color=colors[family], edgecolors="black", linewidths=0.4)
        axes[0].text(x, y, f" {family}", fontsize=8, ha="left", va="center")
    _plot_reference_lines(axes[0], wait_x, wait_y, wait_reg)
    apply_common_style(
        axes[0],
        f"{SYSTEM_A_LABEL} family-mean wait reduction vs FCFS",
        f"{SYSTEM_B_LABEL} family-mean wait reduction vs FCFS",
    )
    axes[0].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    axes[0].set_title("Family Mean Wait Reduction", fontsize=FONT_SIZE + 1)

    util_reg = lookup_regression(regression_rows, "family_means", "util_delta_vs_fcfs_pp")
    util_x = [float(row["mean_util_delta_exp1a_pp"]) for row in family_rows]
    util_y = [float(row["mean_util_delta_exp1b_pp"]) for row in family_rows]
    for row in family_rows:
        family = str(row["family"])
        x = float(row["mean_util_delta_exp1a_pp"])
        y = float(row["mean_util_delta_exp1b_pp"])
        marker = "*" if family == "FCFS" else "o"
        size = 150 if family == "FCFS" else 70
        axes[1].scatter([x], [y], s=size, marker=marker, color=colors[family], edgecolors="black", linewidths=0.4)
        axes[1].text(x, y, f" {family}", fontsize=8, ha="left", va="center")
    _plot_reference_lines(axes[1], util_x, util_y, util_reg)
    apply_common_style(
        axes[1],
        f"{SYSTEM_A_LABEL} family-mean utilization delta vs FCFS (pp)",
        f"{SYSTEM_B_LABEL} family-mean utilization delta vs FCFS (pp)",
    )
    axes[1].set_title("Family Mean Utilization Delta", fontsize=FONT_SIZE + 1)

    fig.subplots_adjust(bottom=0.15, left=0.08, right=0.98, top=0.88, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    exp1a_wait = load_value_table(Path(args.exp1a_wait_csv).resolve(), "Mean")
    exp1a_util = load_value_table(Path(args.exp1a_util_csv).resolve(), "Utilization_Percent")
    exp1b_wait = load_value_table(Path(args.exp1b_wait_csv).resolve(), "Mean")
    exp1b_util = load_value_table(Path(args.exp1b_util_csv).resolve(), "Utilization_Percent")
    output_dir = Path(args.output_dir).resolve()

    matched_rows = build_matched_rows(exp1a_wait, exp1a_util, exp1b_wait, exp1b_util)
    family_rows = build_family_rows(matched_rows)
    regression_rows = build_regression_rows(matched_rows, family_rows)
    window_step_rows = build_window_step_rows(matched_rows)
    window_step_family_rows = build_window_step_family_rows(window_step_rows)
    window_step_regression_rows = build_window_step_regression_rows(window_step_rows, window_step_family_rows)

    write_csv(
        output_dir / "matched_policy_points.csv",
        matched_rows,
        [
            "driver",
            "family",
            "window",
            "wait_mean_exp1a_min",
            "wait_mean_exp1b_min",
            "util_exp1a_pct",
            "util_exp1b_pct",
            "wait_ratio_exp1a",
            "wait_ratio_exp1b",
            "wait_reduction_exp1a",
            "wait_reduction_exp1b",
            "util_delta_exp1a_pp",
            "util_delta_exp1b_pp",
            "util_ratio_exp1a_vs_fcfs",
            "util_ratio_exp1b_vs_fcfs",
            "util_fraction_exp1a",
            "util_fraction_exp1b",
            "util_delta_ratio_exp1a",
            "util_delta_ratio_exp1b",
        ],
    )
    write_csv(
        output_dir / "family_summary.csv",
        family_rows,
        [
            "family",
            "n_points",
            "mean_wait_reduction_exp1a",
            "mean_wait_reduction_exp1b",
            "mean_wait_reduction_diff_b_minus_a",
            "mean_util_delta_exp1a_pp",
            "mean_util_delta_exp1b_pp",
            "mean_util_delta_diff_b_minus_a_pp",
            "mean_util_ratio_exp1a_vs_fcfs",
            "mean_util_ratio_exp1b_vs_fcfs",
            "mean_util_delta_ratio_exp1a",
            "mean_util_delta_ratio_exp1b",
        ],
    )
    write_csv(
        output_dir / "regression_summary.csv",
        regression_rows,
        ["sample", "metric", "n_points", "intercept", "slope", "pearson_r", "r2", "spearman_rho", "rmse"],
    )
    write_csv(
        output_dir / "window_step_policy_points.csv",
        window_step_rows,
        [
            "family",
            "driver_from",
            "driver_to",
            "window_from",
            "window_to",
            "window_step",
            "wait_step_exp1a",
            "wait_step_exp1b",
            "util_step_exp1a",
            "util_step_exp1b",
        ],
    )
    write_csv(
        output_dir / "window_step_family_summary.csv",
        window_step_family_rows,
        [
            "family",
            "n_steps",
            "mean_wait_step_exp1a",
            "mean_wait_step_exp1b",
            "mean_util_step_exp1a",
            "mean_util_step_exp1b",
        ],
    )
    write_csv(
        output_dir / "window_step_regression_summary.csv",
        window_step_regression_rows,
        ["sample", "metric", "n_points", "intercept", "slope", "pearson_r", "r2", "spearman_rho", "rmse"],
    )

    selected_regression_rows = plot_all_points_figure(
        matched_rows,
        family_rows,
        output_dir / "cross_system_regression_all_points.png",
    )
    selected_rows = filter_rows_by_family(matched_rows, CHARACTERISTIC_FAMILIES)
    write_csv(
        output_dir / "characteristic_policy_points.csv",
        selected_rows,
        [
            "driver",
            "family",
            "window",
            "wait_mean_exp1a_min",
            "wait_mean_exp1b_min",
            "util_exp1a_pct",
            "util_exp1b_pct",
            "wait_ratio_exp1a",
            "wait_ratio_exp1b",
            "wait_reduction_exp1a",
            "wait_reduction_exp1b",
            "util_delta_exp1a_pp",
            "util_delta_exp1b_pp",
            "util_ratio_exp1a_vs_fcfs",
            "util_ratio_exp1b_vs_fcfs",
            "util_fraction_exp1a",
            "util_fraction_exp1b",
            "util_delta_ratio_exp1a",
            "util_delta_ratio_exp1b",
        ],
    )
    write_csv(
        output_dir / "characteristic_regression_summary.csv",
        selected_regression_rows,
        ["sample", "metric", "n_points", "intercept", "slope", "pearson_r", "r2", "spearman_rho", "rmse"],
    )
    characteristic_window_step_regression_rows = plot_window_step_preservation_figure(
        window_step_rows,
        window_step_family_rows,
        output_dir / "cross_system_window_step_preservation.png",
    )
    selected_window_step_rows = filter_rows_by_family(window_step_rows, WINDOW_CHARACTERISTIC_FAMILIES)
    write_csv(
        output_dir / "characteristic_window_step_policy_points.csv",
        selected_window_step_rows,
        [
            "family",
            "driver_from",
            "driver_to",
            "window_from",
            "window_to",
            "window_step",
            "wait_step_exp1a",
            "wait_step_exp1b",
            "util_step_exp1a",
            "util_step_exp1b",
        ],
    )
    write_csv(
        output_dir / "characteristic_window_step_regression_summary.csv",
        characteristic_window_step_regression_rows,
        ["sample", "metric", "n_points", "intercept", "slope", "pearson_r", "r2", "spearman_rho", "rmse"],
    )
    plot_combined_transfer_figure(
        matched_rows,
        family_rows,
        window_step_rows,
        window_step_family_rows,
        output_dir / "cross_system_combined_transfer.png",
    )
    plot_family_means_figure(
        family_rows,
        regression_rows,
        output_dir / "cross_system_regression_family_means.png",
    )

    print(f"Wrote analysis outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
