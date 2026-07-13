#!/usr/bin/env python3
"""Plot average wait and utilization versus scheduling window size.

The input CSV is expected to look like results/exp1a_plots/table_wait_overall.csv,
where rows are named like "SJF-w64" and the "Mean" column stores average wait
time in minutes. The matching utilization CSV is expected to look like
table_util_overall.csv, where Utilization_Percent is reported directly in
percent. Optional per-group wait tables can also be loaded for the lower row of
the figure. Rows without a "-w" suffix are treated as fixed baselines and are
repeated across all window sizes.

Usage:
    python3 scripts/plot_avg_wait_vs_window.py
    python3 scripts/plot_avg_wait_vs_window.py \
        results/exp1a_plots/table_wait_overall.csv \
        --util-csv results/exp1a_plots/table_util_overall.csv \
        --output results/exp1a_plots/avg_wait_vs_window_singlecol.png
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


DEFAULT_EXP1A_CSV = Path("results/exp1a_plots/table_wait_overall.csv")
DEFAULT_EXP1B_CSV = Path("results/exp1b_plots/table_wait_overall.csv")
DEFAULT_OUTPUT = Path("results/avg_wait_vs_window_cross_system.png")

POLICIES = ("FCFS", "SJF", "WFP3", "F1", "UNICEP", "FAT", "LRF", "LJF")

FIG_W = 4.8
FIG_H = 3.2
DPI = 300

AXIS_FONT = 11
TICK_FONT = 9.0
LEGEND_FONT = 10.5
LINE_WIDTH = 1.2
MARKER_SIZE = 4.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average wait vs window size for Theta and Polaris."
    )
    parser.add_argument(
        "--exp1a-csv",
        default=str(DEFAULT_EXP1A_CSV),
        help=f"Input Theta CSV path (default: {DEFAULT_EXP1A_CSV})",
    )
    parser.add_argument(
        "--exp1b-csv",
        default=str(DEFAULT_EXP1B_CSV),
        help=f"Input Polaris CSV path (default: {DEFAULT_EXP1B_CSV})",
    )
    parser.add_argument(
        "-o", "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output image path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def split_driver_label(driver: str) -> tuple[str, int | None]:
    """Return (policy_name, window_size_or_none)."""
    if "-w" not in driver:
        return driver, None
    policy, window_text = driver.rsplit("-w", 1)
    return policy, int(window_text)


def load_metric_series(
    csv_path: Path,
    value_column: str,
    transform=lambda value: value,
) -> tuple[list[int], list[str], dict[str, OrderedDict[int, float]]]:
    """Load a per-driver metric series grouped by policy and window size."""
    policies: OrderedDict[str, OrderedDict[int, float]] = OrderedDict()
    baselines: OrderedDict[str, float] = OrderedDict()
    windows: set[int] = set()

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            driver = row["Driver"].strip()
            policy, window = split_driver_label(driver)
            metric_value = transform(float(row[value_column]))

            if policy not in policies:
                policies[policy] = OrderedDict()

            if window is None:
                baselines[policy] = metric_value
            else:
                policies[policy][window] = metric_value
                windows.add(window)

    window_list = sorted(windows)
    if not window_list:
        raise ValueError(f"found no windowed policies in {csv_path}")

    for policy, baseline in baselines.items():
        policies[policy] = OrderedDict((window, baseline) for window in window_list)

    for policy, series in policies.items():
        policies[policy] = OrderedDict(sorted(series.items()))

    return window_list, list(policies.keys()), policies


def validate_metric_coverage(
    metric_name: str,
    expected_windows: list[int],
    expected_policies: list[str],
    series_by_policy: dict[str, OrderedDict[int, float]],
) -> None:
    """Ensure both CSVs cover the same policy/window grid."""
    if list(series_by_policy.keys()) != expected_policies:
        missing = [policy for policy in expected_policies if policy not in series_by_policy]
        extra = [policy for policy in series_by_policy if policy not in expected_policies]
        raise ValueError(
            f"{metric_name} policies do not match wait policies; "
            f"missing={missing}, extra={extra}"
        )

    for policy in expected_policies:
        metric_windows = list(series_by_policy[policy].keys())
        if metric_windows != expected_windows:
            raise ValueError(
                f"{metric_name} windows do not match wait windows for policy {policy}: "
                f"{metric_windows} != {expected_windows}"
            )


def build_color_map(policy_names: list[str]) -> dict[str, tuple[float, float, float] | str]:
    """Assign a stable unique color to each policy."""
    palette = list(plt.get_cmap("tab20").colors)
    color_map: dict[str, tuple[float, float, float] | str] = {}
    idx = 0

    for policy in policy_names:
        if policy == "FCFS":
            color_map[policy] = "#000000"
        else:
            color_map[policy] = palette[idx % len(palette)]
            idx += 1

    return color_map


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


def _log_e_formatter(value: float, _pos: int) -> str:
    """Render compact log labels like 1e1, 2e1, 2e2."""
    if value <= 0:
        return ""

    exponent = int(np.floor(np.log10(value)))
    scale = 10 ** exponent
    mantissa = value / scale

    rounded = round(mantissa)
    if not np.isclose(mantissa, rounded, rtol=0, atol=1e-9):
        return ""

    if rounded not in (1, 2):
        return ""

    return f"{rounded}e{exponent}"


def apply_log_e_notation(ax):
    """Use compact e-notation on log axes with 1x and 2x decade labels."""
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_log_e_formatter))
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def plot_panel(ax, window_sizes, policy_names, series_by_policy, color_map, title):
    x_positions = np.arange(len(window_sizes))
    position_by_window = {window: idx for idx, window in enumerate(window_sizes)}

    if "FCFS" in series_by_policy and series_by_policy["FCFS"]:
        fcfs_val = next(iter(series_by_policy["FCFS"].values()))
        ax.axhline(fcfs_val, color=color_map["FCFS"], linestyle=":", linewidth=LINE_WIDTH, label="FCFS", zorder=2)

    for policy in policy_names:
        if policy == "FCFS" or policy not in series_by_policy:
            continue
        series = series_by_policy[policy]
        xs = [position_by_window[window] for window in series]
        ys = [series[window] for window in series]
        ax.plot(
            xs, ys, label=policy, color=color_map[policy],
            linewidth=LINE_WIDTH, linestyle=":", marker="o", markersize=MARKER_SIZE, zorder=3
        )

    ax.set_xlim(-0.5, len(window_sizes) - 0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(window) for window in window_sizes], rotation=32, ha="right")
    ax.set_xlabel("Window Size", fontsize=AXIS_FONT)
    ax.set_title(title, fontsize=AXIS_FONT)
    
    ax.set_yscale("log")
    apply_log_e_notation(ax)
    
    ax.tick_params(axis="both", labelsize=TICK_FONT)
    ax.grid(True, which="major", axis="y", color="#d0d0d0", linewidth=0.6)
    ax.grid(True, which="minor", axis="y", color="#ebebeb", linewidth=0.45)
    ax.grid(True, which="major", axis="x", color="#f0f0f0", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exp1a_csv = Path(args.exp1a_csv).resolve()
    exp1b_csv = Path(args.exp1b_csv).resolve()
    output_path = Path(args.output).resolve()

    windows_a, policies_a, wait_by_policy_a = load_metric_series(
        exp1a_csv, "Mean", transform=lambda value: value / 60.0
    )
    windows_b, policies_b, wait_by_policy_b = load_metric_series(
        exp1b_csv, "Mean", transform=lambda value: value / 60.0
    )

    cmap = build_color_map(list(POLICIES))

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), dpi=DPI, gridspec_kw={"wspace": 0.25})

    plot_panel(axes[0], windows_a, POLICIES, wait_by_policy_a, cmap, "Theta\n(2021)")
    axes[0].set_ylabel("Overall Avg Wait (h)", fontsize=AXIS_FONT, labelpad=2)

    plot_panel(axes[1], windows_b, POLICIES, wait_by_policy_b, cmap, "Polaris\n(2024)")

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
        
    dict_hl = dict(zip(labels, handles))
    ordered_handles = [dict_hl[p] for p in POLICIES if p in dict_hl]
    ordered_labels = [p for p in POLICIES if p in dict_hl]
    
    ncol = 4
    if ordered_handles:
        ordered_handles, ordered_labels = reorder_handles_for_row_major(ordered_handles, ordered_labels, ncol)

        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=ncol,
            fontsize=LEGEND_FONT,
            frameon=False,
            handlelength=1.6,
            columnspacing=1.0,
            handletextpad=0.4
        )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.38)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved figure to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
