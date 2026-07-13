#!/usr/bin/env python3
"""Plot exp1b average wait and utilization versus window size.

This variant is tailored to the Polaris exp1b outputs and uses the Polaris
size classes directly:
  - S for small jobs
  - L for large jobs

The figure layout is a compact 2x2 grid:
  - overall average wait
  - utilization
  - S average wait
  - L average wait
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from plot_avg_wait_vs_window import (
    add_bottom_legend,
    build_color_map,
    derive_variant_output_path,
    HEURISTIC_LINE_WIDTH,
    HEURISTIC_MARKER_SIZE,
    HEURISTIC_OUTPUT_SUFFIX,
    HEURISTIC_POLICIES,
    load_metric_series,
    plot_policy_lines,
    select_policy_subset,
    validate_metric_coverage,
)


DEFAULT_WAIT_CSV = Path("results/exp1b_plots/table_wait_overall.csv")
DEFAULT_UTIL_CSV = Path("results/exp1b_plots/table_util_overall.csv")
DEFAULT_SMALL_CSV = Path("results/exp1b_plots/table_S_wait.csv")
DEFAULT_LARGE_CSV = Path("results/exp1b_plots/table_L_wait.csv")
DEFAULT_OUTPUT = Path("results/exp1b_plots/avg_wait_vs_window_singlecol.png")

FIG_W = 3.45
FIG_H = 4.95
DPI = 300

AXIS_FONT = 9
TICK_FONT = 6.1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the exp1b overall/S/L wait and utilization summaries."
    )
    parser.add_argument(
        "--wait-csv",
        default=str(DEFAULT_WAIT_CSV),
        help=f"Overall wait CSV (default: {DEFAULT_WAIT_CSV})",
    )
    parser.add_argument(
        "--util-csv",
        default=str(DEFAULT_UTIL_CSV),
        help=f"Utilization CSV (default: {DEFAULT_UTIL_CSV})",
    )
    parser.add_argument(
        "--small-wait-csv",
        default=str(DEFAULT_SMALL_CSV),
        help=f"Small-job wait CSV (default: {DEFAULT_SMALL_CSV})",
    )
    parser.add_argument(
        "--large-wait-csv",
        default=str(DEFAULT_LARGE_CSV),
        help=f"Large-job wait CSV (default: {DEFAULT_LARGE_CSV})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output image path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


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


def apply_compact_log_ticks(ax) -> None:
    """Use compact e-notation on log axes with 1x and 2x decade labels."""
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_log_e_formatter))
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def _configure_wait_panel(ax, ylabel: str) -> None:
    ax.set_yscale("log")
    apply_compact_log_ticks(ax)
    ax.set_ylabel(ylabel, fontsize=AXIS_FONT, labelpad=2)
    ax.tick_params(axis="both", labelsize=TICK_FONT)
    ax.grid(True, which="major", axis="y", color="#d0d0d0", linewidth=0.6)
    ax.grid(True, which="minor", axis="y", color="#ebebeb", linewidth=0.45)


def _configure_util_panel(ax) -> None:
    ax.set_ylabel("Net Utilization (%)", fontsize=AXIS_FONT, labelpad=2)
    ax.tick_params(axis="both", labelsize=TICK_FONT)
    ax.grid(True, which="major", axis="both", color="#e2e2e2", linewidth=0.55)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    wait_csv = Path(args.wait_csv).resolve()
    util_csv = Path(args.util_csv).resolve()
    small_wait_csv = Path(args.small_wait_csv).resolve()
    large_wait_csv = Path(args.large_wait_csv).resolve()
    output_path = Path(args.output).resolve()

    window_sizes, policy_names, overall_wait_by_policy = load_metric_series(
        wait_csv,
        "Mean",
        transform=lambda value: value / 60.0,
    )
    util_windows, util_policies, util_by_policy = load_metric_series(
        util_csv,
        "Utilization_Percent",
    )
    small_windows, small_policies, small_wait_by_policy = load_metric_series(
        small_wait_csv,
        "Mean",
        transform=lambda value: value / 60.0,
    )
    large_windows, large_policies, large_wait_by_policy = load_metric_series(
        large_wait_csv,
        "Mean",
        transform=lambda value: value / 60.0,
    )

    if util_windows != window_sizes:
        raise ValueError(f"utilization windows do not match overall windows: {util_windows} != {window_sizes}")
    if small_windows != window_sizes:
        raise ValueError(f"S windows do not match overall windows: {small_windows} != {window_sizes}")
    if large_windows != window_sizes:
        raise ValueError(f"L windows do not match overall windows: {large_windows} != {window_sizes}")

    validate_metric_coverage("utilization", window_sizes, policy_names, util_by_policy)
    validate_metric_coverage("S wait", window_sizes, policy_names, small_wait_by_policy)
    validate_metric_coverage("L wait", window_sizes, policy_names, large_wait_by_policy)

    if util_policies != policy_names:
        raise ValueError("utilization policy order does not match overall policy order")
    if small_policies != policy_names:
        raise ValueError("S wait policy order does not match overall policy order")
    if large_policies != policy_names:
        raise ValueError("L wait policy order does not match overall policy order")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_W, FIG_H),
        dpi=DPI,
        sharex=True,
        gridspec_kw={"wspace": 0.38, "hspace": 0.18},
    )
    wait_ax = axes[0, 0]
    util_ax = axes[0, 1]
    small_ax = axes[1, 0]
    large_ax = axes[1, 1]

    color_map = build_color_map(policy_names)
    plot_policy_lines(wait_ax, window_sizes, policy_names, overall_wait_by_policy, color_map)
    plot_policy_lines(util_ax, window_sizes, policy_names, util_by_policy, color_map)
    plot_policy_lines(small_ax, window_sizes, policy_names, small_wait_by_policy, color_map)
    plot_policy_lines(large_ax, window_sizes, policy_names, large_wait_by_policy, color_map)

    _configure_wait_panel(wait_ax, "Overall Avg Wait (h)")
    _configure_util_panel(util_ax)
    _configure_wait_panel(small_ax, "Avg Wait (h) - Small Jobs (S)")
    _configure_wait_panel(large_ax, "Avg Wait (h) - Large Jobs (L)")

    for ax in (util_ax, large_ax):
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    util_values = [value for series in util_by_policy.values() for value in series.values()]
    util_min = min(util_values)
    util_max = max(util_values)
    util_pad = max(0.25, 0.08 * (util_max - util_min))
    util_ax.set_ylim(util_min - util_pad, util_max + util_pad)

    for ax in (wait_ax, util_ax):
        ax.tick_params(axis="x", labelbottom=False)

    x_tick_labels = [str(window) for window in window_sizes]
    for ax in (small_ax, large_ax):
        ax.set_xticklabels(x_tick_labels, rotation=32, ha="right")
        ax.set_xlabel("Window Size", fontsize=AXIS_FONT, labelpad=1)

    handles, labels = wait_ax.get_legend_handles_labels()
    add_bottom_legend(fig, handles, labels)

    fig.subplots_adjust(left=0.15, right=0.93, top=0.99, bottom=0.23)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved figure to {output_path}")

    heuristic_output_path = derive_variant_output_path(output_path, HEURISTIC_OUTPUT_SUFFIX)
    heuristic_policy_names, heuristic_series_maps = select_policy_subset(
        policy_names,
        [overall_wait_by_policy, util_by_policy, small_wait_by_policy, large_wait_by_policy],
        HEURISTIC_POLICIES,
    )
    heuristic_wait_by_policy, heuristic_util_by_policy, heuristic_small_wait_by_policy, heuristic_large_wait_by_policy = heuristic_series_maps

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_W, FIG_H),
        dpi=DPI,
        sharex=True,
        gridspec_kw={"wspace": 0.38, "hspace": 0.18},
    )
    wait_ax = axes[0, 0]
    util_ax = axes[0, 1]
    small_ax = axes[1, 0]
    large_ax = axes[1, 1]

    color_map = build_color_map(heuristic_policy_names)
    plot_policy_lines(
        wait_ax,
        window_sizes,
        heuristic_policy_names,
        heuristic_wait_by_policy,
        color_map,
        dotted=True,
        line_width=HEURISTIC_LINE_WIDTH,
        marker_size=HEURISTIC_MARKER_SIZE,
    )
    plot_policy_lines(
        util_ax,
        window_sizes,
        heuristic_policy_names,
        heuristic_util_by_policy,
        color_map,
        dotted=True,
        line_width=HEURISTIC_LINE_WIDTH,
        marker_size=HEURISTIC_MARKER_SIZE,
    )
    plot_policy_lines(
        small_ax,
        window_sizes,
        heuristic_policy_names,
        heuristic_small_wait_by_policy,
        color_map,
        dotted=True,
        line_width=HEURISTIC_LINE_WIDTH,
        marker_size=HEURISTIC_MARKER_SIZE,
    )
    plot_policy_lines(
        large_ax,
        window_sizes,
        heuristic_policy_names,
        heuristic_large_wait_by_policy,
        color_map,
        dotted=True,
        line_width=HEURISTIC_LINE_WIDTH,
        marker_size=HEURISTIC_MARKER_SIZE,
    )

    _configure_wait_panel(wait_ax, "Overall Avg Wait (h)")
    _configure_util_panel(util_ax)
    _configure_wait_panel(small_ax, "Avg Wait (h) - Small Jobs (S)")
    _configure_wait_panel(large_ax, "Avg Wait (h) - Large Jobs (L)")

    for ax in (util_ax, large_ax):
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    util_ax.set_ylim(util_min - util_pad, util_max + util_pad)

    for ax in (wait_ax, util_ax):
        ax.tick_params(axis="x", labelbottom=False)

    x_tick_labels = [str(window) for window in window_sizes]
    for ax in (small_ax, large_ax):
        ax.set_xticklabels(x_tick_labels, rotation=32, ha="right")
        ax.set_xlabel("Window Size", fontsize=AXIS_FONT, labelpad=1)

    handles, labels = wait_ax.get_legend_handles_labels()
    add_bottom_legend(fig, handles, labels, fixed_ncol=5)

    fig.subplots_adjust(left=0.15, right=0.93, top=0.99, bottom=0.23)
    heuristic_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(heuristic_output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved figure to {heuristic_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
