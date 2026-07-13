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


DEFAULT_CSV = Path("results/exp1a_plots/table_wait_overall.csv")
DEFAULT_UTIL_NAME = "table_util_overall.csv"
DEFAULT_OUTPUT_NAME = "avg_wait_vs_window_singlecol.png"
HEURISTIC_OUTPUT_SUFFIX = "_heuristics_dotted"
HEURISTIC_POLICIES = (
    "FCFS",
    "WFP3",
    "SJF",
    "F1",
    "UNICEP",
    "LRF",
    "SCF",
    "LCF",
    "FAT",
    "LJF",
)

# Sized to fit a single paper column with a compact 2x2 grid and bottom legend.
FIG_W = 3.45
FIG_H = 4.95
DPI = 300

AXIS_FONT = 9
TICK_FONT = 6.1
LEGEND_FONT = 9
LINE_WIDTH = 1.2
MARKER_SIZE = 2.8
HEURISTIC_LINE_WIDTH = 0.75
HEURISTIC_MARKER_SIZE = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot average wait time (hours, log scale) against window size from "
            "table_wait_overall.csv."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_CSV),
        help=f"Input CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--util-csv",
        help=(
            "Optional utilization CSV path. Defaults to a sibling "
            f"'{DEFAULT_UTIL_NAME}' next to the wait CSV."
        ),
    )
    parser.add_argument(
        "--small-wait-csv",
        help=(
            "Optional small-job average-wait CSV. Defaults to a sibling "
            "'table_S_wait.csv' or, if unavailable, 'table_T1_wait.csv'."
        ),
    )
    parser.add_argument(
        "--large-wait-csv",
        help=(
            "Optional large-job average-wait CSV. Defaults to a sibling "
            "'table_L_wait.csv' or, if unavailable, 'table_T5_wait.csv'."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output image path. Defaults to the input CSV directory with the name "
            f"'{DEFAULT_OUTPUT_NAME}'."
        ),
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
    palette_index = 0

    for policy in policy_names:
        if policy == "FCFS":
            color_map[policy] = "#000000"
            continue
        color_map[policy] = palette[palette_index % len(palette)]
        palette_index += 1

    return color_map


def infer_group_wait_paths(
    overall_wait_csv: Path,
    small_wait_csv: Path | None,
    large_wait_csv: Path | None,
) -> tuple[Path, str, Path, str]:
    """Infer per-group average-wait tables for the lower-row panels."""
    if small_wait_csv and large_wait_csv:
        return small_wait_csv, small_wait_csv.stem.split("_")[1].upper(), large_wait_csv, large_wait_csv.stem.split("_")[1].upper()

    candidates = [
        (overall_wait_csv.with_name("table_S_wait.csv"), "S", overall_wait_csv.with_name("table_L_wait.csv"), "L"),
        (overall_wait_csv.with_name("table_T1_wait.csv"), "T1", overall_wait_csv.with_name("table_T5_wait.csv"), "T5"),
    ]
    for small_path, small_label, large_path, large_label in candidates:
        if small_path.exists() and large_path.exists():
            return small_path, small_label, large_path, large_label

    raise FileNotFoundError(
        "could not infer small/large wait tables; provide --small-wait-csv and "
        "--large-wait-csv explicitly"
    )


def reorder_handles_for_row_major(handles, labels, ncol):
    """Reorder legend items so matplotlib's column-major fill displays row-major."""
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


def derive_variant_output_path(output_path: Path, suffix: str) -> Path:
    """Append a suffix to a figure filename before its extension."""
    return output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")


def select_policy_subset(
    policy_names: list[str],
    series_maps: list[dict[str, OrderedDict[int, float]]],
    keep_policies: tuple[str, ...] | list[str],
) -> tuple[list[str], list[dict[str, OrderedDict[int, float]]]]:
    """Return the requested policy subset in a stable, explicit order."""
    selected_policies = [policy for policy in keep_policies if policy in policy_names]
    if not selected_policies:
        raise ValueError(f"none of the requested policies were found: {keep_policies}")

    selected_series_maps = []
    for series_map in series_maps:
        selected_series_maps.append(OrderedDict((policy, series_map[policy]) for policy in selected_policies))

    return selected_policies, selected_series_maps


def add_bottom_legend(
    fig,
    handles,
    labels,
    *,
    single_line: bool = False,
    fixed_ncol: int | None = None,
) -> None:
    """Draw a compact bottom legend with two full rows plus centered FCFS."""
    legend_kwargs = dict(
        fontsize=LEGEND_FONT,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.45,
        columnspacing=0.8,
        labelspacing=0.30,
        borderaxespad=0.0,
    )

    if single_line or fixed_ncol is not None:
        ncol = len(handles) if single_line else fixed_ncol
        ordered_handles, ordered_labels = reorder_handles_for_row_major(handles, labels, ncol)
        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.020),
            ncol=ncol,
            **legend_kwargs,
        )
        return

    fcfs_handle = None
    fcfs_label = None
    main_handles = []
    main_labels = []
    for handle, label in zip(handles, labels):
        if label == "FCFS":
            fcfs_handle = handle
            fcfs_label = label
        else:
            main_handles.append(handle)
            main_labels.append(label)

    ncol = min(8, len(main_handles)) if main_handles else 1
    main_handles, main_labels = reorder_handles_for_row_major(main_handles, main_labels, ncol)

    main_legend = fig.legend(
        main_handles,
        main_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.060),
        ncol=ncol,
        **legend_kwargs,
    )
    fig.add_artist(main_legend)

    if fcfs_handle is not None:
        fig.legend(
            [fcfs_handle],
            [fcfs_label],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.020),
            ncol=1,
            **legend_kwargs,
        )


def plot_policy_lines(
    ax,
    window_sizes,
    policy_names,
    series_by_policy,
    color_map,
    *,
    dotted: bool = False,
    line_width: float = LINE_WIDTH,
    marker_size: float = MARKER_SIZE,
):
    """Draw one metric panel for all policies."""
    x_positions = np.arange(len(window_sizes))
    position_by_window = {window: idx for idx, window in enumerate(window_sizes)}

    for policy in policy_names:
        series = series_by_policy[policy]
        xs = [position_by_window[window] for window in series]
        ys = [series[window] for window in series]
        linestyle = ":" if dotted else ("--" if policy == "FCFS" else "-")
        ax.plot(
            xs,
            ys,
            label=policy,
            color=color_map[policy],
            linewidth=line_width,
            linestyle=linestyle,
            marker="o",
            markersize=marker_size,
        )

    ax.set_xlim(-0.2, len(window_sizes) - 0.8)
    ax.set_xticks(x_positions)
    ax.grid(True, which="major", axis="x", color="#f0f0f0", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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


def apply_log_e_notation(ax) -> None:
    """Use compact e-notation on log axes with 1x and 2x decade labels."""
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_log_e_formatter))
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def plot_avg_wait_vs_window(
    window_sizes: list[int],
    policy_names: list[str],
    overall_wait_by_policy: dict[str, OrderedDict[int, float]],
    util_by_policy: dict[str, OrderedDict[int, float]],
    small_wait_by_policy: dict[str, OrderedDict[int, float]],
    small_label: str,
    large_wait_by_policy: dict[str, OrderedDict[int, float]],
    large_label: str,
    output_path: Path,
    *,
    dotted: bool = False,
    line_width: float = LINE_WIDTH,
    marker_size: float = MARKER_SIZE,
    single_line_legend: bool = False,
    legend_ncol_override: int | None = None,
) -> None:
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
    plot_policy_lines(
        wait_ax,
        window_sizes,
        policy_names,
        overall_wait_by_policy,
        color_map,
        dotted=dotted,
        line_width=line_width,
        marker_size=marker_size,
    )
    plot_policy_lines(
        util_ax,
        window_sizes,
        policy_names,
        util_by_policy,
        color_map,
        dotted=dotted,
        line_width=line_width,
        marker_size=marker_size,
    )
    plot_policy_lines(
        small_ax,
        window_sizes,
        policy_names,
        small_wait_by_policy,
        color_map,
        dotted=dotted,
        line_width=line_width,
        marker_size=marker_size,
    )
    plot_policy_lines(
        large_ax,
        window_sizes,
        policy_names,
        large_wait_by_policy,
        color_map,
        dotted=dotted,
        line_width=line_width,
        marker_size=marker_size,
    )

    wait_ax.set_yscale("log")
    apply_log_e_notation(wait_ax)
    wait_ax.set_ylabel("Overall Avg Wait (h)", fontsize=AXIS_FONT, labelpad=2)
    wait_ax.tick_params(axis="both", labelsize=TICK_FONT)
    wait_ax.grid(True, which="major", axis="y", color="#d0d0d0", linewidth=0.6)
    wait_ax.grid(True, which="minor", axis="y", color="#ebebeb", linewidth=0.45)

    util_ax.set_ylabel("Net Utilization (%)", fontsize=AXIS_FONT, labelpad=2)
    util_ax.tick_params(axis="both", labelsize=TICK_FONT)
    util_ax.grid(True, which="major", axis="both", color="#e2e2e2", linewidth=0.55)

    util_values = [value for series in util_by_policy.values() for value in series.values()]
    util_min = min(util_values)
    util_max = max(util_values)
    util_pad = max(0.3, 0.08 * (util_max - util_min))
    util_ax.set_ylim(util_min - util_pad, util_max + util_pad)

    small_ax.set_yscale("log")
    apply_log_e_notation(small_ax)
    small_ax.set_ylabel(f"Avg Wait (h) - Small Jobs ({small_label})", fontsize=AXIS_FONT, labelpad=2)
    small_ax.tick_params(axis="both", labelsize=TICK_FONT)
    small_ax.grid(True, which="major", axis="y", color="#d0d0d0", linewidth=0.6)
    small_ax.grid(True, which="minor", axis="y", color="#ebebeb", linewidth=0.45)

    large_ax.set_yscale("log")
    apply_log_e_notation(large_ax)
    large_ax.set_ylabel(f"Avg Wait (h) - Large Jobs ({large_label})", fontsize=AXIS_FONT, labelpad=2)
    large_ax.tick_params(axis="both", labelsize=TICK_FONT)
    large_ax.grid(True, which="major", axis="y", color="#d0d0d0", linewidth=0.6)
    large_ax.grid(True, which="minor", axis="y", color="#ebebeb", linewidth=0.45)

    for ax in (util_ax, large_ax):
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    for ax in (wait_ax, util_ax):
        ax.tick_params(axis="x", labelbottom=False)

    x_tick_labels = [str(window) for window in window_sizes]
    for ax in (small_ax, large_ax):
        ax.set_xticklabels(x_tick_labels, rotation=32, ha="right")
        ax.set_xlabel("Window Size", fontsize=AXIS_FONT)

    handles, labels = wait_ax.get_legend_handles_labels()
    add_bottom_legend(
        fig,
        handles,
        labels,
        single_line=single_line_legend,
        fixed_ncol=legend_ncol_override,
    )

    fig.subplots_adjust(left=0.15, bottom=0.23, right=0.99, top=0.99)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv_path).resolve()
    util_csv_path = (
        Path(args.util_csv).resolve()
        if args.util_csv
        else csv_path.with_name(DEFAULT_UTIL_NAME)
    )
    small_wait_arg = Path(args.small_wait_csv).resolve() if args.small_wait_csv else None
    large_wait_arg = Path(args.large_wait_csv).resolve() if args.large_wait_csv else None
    small_wait_csv_path, small_label, large_wait_csv_path, large_label = infer_group_wait_paths(
        csv_path,
        small_wait_arg,
        large_wait_arg,
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else csv_path.parent / DEFAULT_OUTPUT_NAME
    )

    window_sizes, policy_names, wait_by_policy = load_metric_series(
        csv_path,
        "Mean",
        transform=lambda value: value / 60.0,
    )
    util_windows, util_policies, util_by_policy = load_metric_series(
        util_csv_path,
        "Utilization_Percent",
    )
    if util_windows != window_sizes:
        raise ValueError(f"utilization windows do not match wait windows: {util_windows} != {window_sizes}")
    validate_metric_coverage("utilization", window_sizes, policy_names, util_by_policy)
    if util_policies != policy_names:
        raise ValueError(f"utilization policy order does not match wait policy order: {util_policies} != {policy_names}")
    small_windows, small_policies, small_wait_by_policy = load_metric_series(
        small_wait_csv_path,
        "Mean",
        transform=lambda value: value / 60.0,
    )
    if small_windows != window_sizes:
        raise ValueError(f"small-group windows do not match overall windows: {small_windows} != {window_sizes}")
    validate_metric_coverage("small-group", window_sizes, policy_names, small_wait_by_policy)
    if small_policies != policy_names:
        raise ValueError(f"small-group policy order does not match overall policy order: {small_policies} != {policy_names}")
    large_windows, large_policies, large_wait_by_policy = load_metric_series(
        large_wait_csv_path,
        "Mean",
        transform=lambda value: value / 60.0,
    )
    if large_windows != window_sizes:
        raise ValueError(f"large-group windows do not match overall windows: {large_windows} != {window_sizes}")
    validate_metric_coverage("large-group", window_sizes, policy_names, large_wait_by_policy)
    if large_policies != policy_names:
        raise ValueError(f"large-group policy order does not match overall policy order: {large_policies} != {policy_names}")

    plot_avg_wait_vs_window(
        window_sizes,
        policy_names,
        wait_by_policy,
        util_by_policy,
        small_wait_by_policy,
        small_label,
        large_wait_by_policy,
        large_label,
        output_path,
    )
    heuristic_output_path = derive_variant_output_path(output_path, HEURISTIC_OUTPUT_SUFFIX)
    heuristic_policy_names, heuristic_series_maps = select_policy_subset(
        policy_names,
        [wait_by_policy, util_by_policy, small_wait_by_policy, large_wait_by_policy],
        HEURISTIC_POLICIES,
    )
    heuristic_wait_by_policy, heuristic_util_by_policy, heuristic_small_wait_by_policy, heuristic_large_wait_by_policy = heuristic_series_maps
    plot_avg_wait_vs_window(
        window_sizes,
        heuristic_policy_names,
        heuristic_wait_by_policy,
        heuristic_util_by_policy,
        heuristic_small_wait_by_policy,
        small_label,
        heuristic_large_wait_by_policy,
        large_label,
        heuristic_output_path,
        dotted=True,
        line_width=HEURISTIC_LINE_WIDTH,
        marker_size=HEURISTIC_MARKER_SIZE,
        legend_ncol_override=5,
    )
    print(f"Saved figure to {output_path}")
    print(f"Saved figure to {heuristic_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
