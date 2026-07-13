#!/usr/bin/env python3
"""Create combined overall exp2a/exp2b bar plots.

Builds side-by-side overall plots for:
  - Theta 2021 (exp2a)
  - Polaris 2024 (exp2b)

Reads the already-generated overall stat tables from:
  results/exp2a_plots/{wait,bsld}/table_*.csv
  results/exp2b_plots/{wait,bsld}/table_*.csv

Outputs:
  results/exp2ab_plots/wait/bar_chart_wait.png
  results/exp2ab_plots/bsld/bar_chart_bsld.png
"""

import argparse
import bisect
import csv
import json
import os
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext
import numpy as np

from exp2_plot import (
    display_tag,
    driver_hatch,
    _HEURISTIC_COLORS,
    _MARS_COLORS,
    _wait_kde_curve,
    _add_centered_two_row_legend,
    _metric_x_positions,
    _add_metric_backgrounds,
    _add_horizontal_guides,
    _collect_drain_queue_samples_by_period_from_table,
    _draw_grouped_drain_queue_boxplots,
    _period_x_positions,
    _draw_period_backgrounds,
    _queue_before_ylim,
    _load_drain_periods_from_table,
    _load_drain_period_labels_from_table,
    _load_driver_columns_from_table,
    _infer_procs_map_for_results_dir,
    _infer_walltimes_map_for_results_dir,
    _build_wildcard_sequence_entries,
    _draw_drain_sequence_bar,
    _load_drain_pct_for_tag,
    tag_slug,
    parse_maintenance_from_events,
    sum_used_core_seconds,
)
from plot_results import parse_events
from exp2_plot import parse_rlscheduler_csv


BAR_STAT_HEADERS = ["P50", "P75", "P95", "P99", "Mean"]
Y_TICKS = [-100, -75, -50, 0, 25, 50, 100]

_GLOBAL_EXCLUDED = frozenset({"MARS-CB", "MARS-IU", "LRF"})


def _load_table(path):
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            driver = row["Driver"]
            rows[driver] = {name: float(row[name]) for name in BAR_STAT_HEADERS}
    return rows


def _build_driver_colors(driver_tags):
    colors = {}
    for tag in driver_tags:
        tag_up = tag.upper()
        if tag_up.startswith("MARS"):
            mars_color = next(
                (c for suffix, c in _MARS_COLORS.items() if tag_up.endswith(suffix.upper())),
                "#f6de6a",
            )
            colors[tag] = mars_color
        else:
            matched = next(
                (color for prefix, color in _HEURISTIC_COLORS.items() if tag_up.startswith(prefix)),
                None,
            )
            colors[tag] = matched if matched is not None else "gray"
    return colors


def _parse_percent_cell(value):
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None


def _compute_improvements(table_rows, baseline_tag=None):
    driver_tags = list(table_rows.keys())
    if baseline_tag is None:
        baseline_tag = next((t for t in driver_tags if t.upper().startswith("WFP")), None)
    if baseline_tag is None:
        raise ValueError("No WFP baseline found in table")

    _excl_up = {t.upper() for t in _GLOBAL_EXCLUDED}
    baseline = [table_rows[baseline_tag][name] for name in BAR_STAT_HEADERS]
    other_tags = [
        tag for tag in driver_tags
        if tag != baseline_tag
        and not any(tag.upper().startswith(e) for e in _excl_up)
    ]
    improvements = {}
    for tag in other_tags:
        vals = [table_rows[tag][name] for name in BAR_STAT_HEADERS]
        improvements[tag] = [
            0.0 if b == 0 else (b - v) / abs(b) * 100.0
            for b, v in zip(baseline, vals)
        ]
    return other_tags, improvements


def _plot_on_ax(ax, other_tags, improvements, driver_colors):
    n_stats = len(BAR_STAT_HEADERS)
    n_tags = len(other_tags)
    x = _metric_x_positions()
    width = 0.88 / max(n_tags, 1)
    handles = []
    labels = []
    for i, tag in enumerate(other_tags):
        offsets = x + (i - (n_tags - 1) / 2.0) * width
        hatch = driver_hatch(tag)
        bars = ax.bar(
            offsets,
            improvements[tag],
            width=width * 0.97,
            color=driver_colors.get(tag, "gray"),
            alpha=0.85,
            hatch=hatch,
            edgecolor="black" if hatch else None,
        )
        handles.append(bars[0])
        labels.append(display_tag(tag))
    return handles, labels


def _style_ax(ax, title):
    x = _metric_x_positions()
    ax.set_ylim(-100, 100)
    ax.set_yticks(Y_TICKS)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.0f}")
    )
    _add_metric_backgrounds(ax, x)
    _add_horizontal_guides(ax, (-100, 100), Y_TICKS, step=25)
    ax.axhline(0, color="black", linewidth=1.0, zorder=1)
    ax.set_xticks(x)
    labels = ax.set_xticklabels(BAR_STAT_HEADERS, fontsize=10, rotation=25, ha="right")
    for lbl in labels:
        lbl.set_fontweight("bold")
    ax.set_xlim(x[0] - 0.72, x[-1] + 0.72)
    ax.tick_params(axis="y", labelsize=10)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.grid(False, axis="y")
    ax.set_title(title, fontsize=11, pad=4)


def _load_drain_util_table(path, excluded_tags=None):
    if excluded_tags is None:
        excluded_tags = set()
    excluded_upper = {tag.upper() for tag in excluded_tags}

    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        driver_tags = [
            tag for tag in fieldnames[2:]
            if tag.upper() not in excluded_upper
        ]

        labels = []
        rows = []
        for idx, row in enumerate(reader, start=1):
            vals = [_parse_percent_cell(row.get(tag)) for tag in driver_tags]
            if not any(v is not None for v in vals):
                continue
            start_str = row.get("Draining_Period_Start", "")
            end_str = row.get("Draining_Period_End", "")
            try:
                start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
                end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
                label = start_dt.strftime("%m/%d") + "\nto\n" + end_dt.strftime("%m/%d")
            except ValueError:
                label = f"W{idx}"
            labels.append(label)
            rows.append(vals)
    return driver_tags, labels, rows


def _load_drain_queue_before_table(path, excluded_tags=None):
    if excluded_tags is None:
        excluded_tags = set()
    excluded_upper = {tag.upper() for tag in excluded_tags}

    driver_tags = []
    seen_tags = set()
    period_order = []
    period_labels = {}
    period_values = {}

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row.get("Scheduler", "")
            if not tag or tag.upper() in excluded_upper:
                continue
            if tag not in seen_tags:
                seen_tags.add(tag)
                driver_tags.append(tag)

            start_str = row.get("Draining_Period_Start", "")
            end_str = row.get("Draining_Period_End", "")
            period_key = (start_str, end_str)
            if period_key not in period_values:
                period_order.append(period_key)
                period_values[period_key] = {}
                try:
                    start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
                    end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
                    label = start_dt.strftime("%m/%d") + "\nto\n" + end_dt.strftime("%m/%d")
                except ValueError:
                    label = f"W{len(period_order)}"
                period_labels[period_key] = label

            try:
                value = float(row.get("Queue_Before_Announcement", ""))
            except (TypeError, ValueError):
                value = None
            period_values[period_key][tag] = value

    labels = [period_labels[key] for key in period_order]
    rows = [
        [period_values[key].get(tag) for tag in driver_tags]
        for key in period_order
    ]
    return driver_tags, labels, rows


def _load_drain_core_hours_table(path, value_column, excluded_tags=None):
    if excluded_tags is None:
        excluded_tags = set()
    excluded_upper = {tag.upper() for tag in excluded_tags}

    driver_tags = []
    seen_tags = set()
    period_order = []
    period_labels = {}
    period_values = {}

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row.get("Scheduler", "")
            if not tag or tag.upper() in excluded_upper:
                continue
            if tag not in seen_tags:
                seen_tags.add(tag)
                driver_tags.append(tag)

            start_str = row.get("Draining_Period_Start", "")
            end_str = row.get("Draining_Period_End", "")
            period_key = (start_str, end_str)
            if period_key not in period_values:
                period_order.append(period_key)
                period_values[period_key] = {}
                try:
                    start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
                    end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
                    label = start_dt.strftime("%m/%d") + "\nto\n" + end_dt.strftime("%m/%d")
                except ValueError:
                    label = f"W{len(period_order)}"
                period_labels[period_key] = label

            try:
                value = float(row.get(value_column, ""))
            except (TypeError, ValueError):
                value = None
            period_values[period_key][tag] = value

    labels = [period_labels[key] for key in period_order]
    rows = [
        [period_values[key].get(tag) for tag in driver_tags]
        for key in period_order
    ]
    return driver_tags, labels, rows


def _add_policy_star_legend(fig, policy_handles, policy_labels,
                            star_handles, star_labels,
                            y_policy=0.965, y_star=0.915,
                            policy_font=9.0, star_font=9.0):
    if policy_handles and policy_labels:
        top_legend = fig.legend(
            policy_handles,
            policy_labels,
            fontsize=policy_font,
            loc="upper center",
            bbox_to_anchor=(0.5, y_policy),
            ncol=len(policy_labels),
            frameon=False,
            columnspacing=0.9,
            handletextpad=0.4,
        )
        fig.add_artist(top_legend)
    if star_handles and star_labels:
        fig.legend(
            star_handles,
            star_labels,
            fontsize=star_font,
            loc="upper center",
            bbox_to_anchor=(0.5, y_star),
            ncol=len(star_labels),
            frameon=False,
            columnspacing=1.2,
            handletextpad=0.45,
        )


def _results_dir_from_plots_dir(plots_dir):
    norm = os.path.normpath(plots_dir)
    if norm.endswith("_plots"):
        return norm[:-6]
    return norm.replace("_plots", "")


def _mars_tag(driver_tags):
    return next((tag for tag in driver_tags if tag.upper().startswith("MARS")), None)


def _highlight_non_mars_best_periods(driver_tags, util_rows, queue_sample_map):
    mars_tag = _mars_tag(driver_tags)
    if mars_tag is None or mars_tag not in driver_tags:
        return set()

    mars_idx = driver_tags.index(mars_tag)
    highlight = set()
    n_periods = len(util_rows)
    for period_idx in range(n_periods):
        util_row = util_rows[period_idx]
        valid_util = [v for v in util_row if v is not None]
        if not valid_util:
            continue
        mars_util = util_row[mars_idx]
        if mars_util is None:
            continue
        util_not_best = mars_util < max(valid_util) - 1e-9
        # Queue size is only a diagnostic panel here; highlight periods are
        # determined solely by whether MARS was best on utilization.
        if util_not_best:
            highlight.add(period_idx)
    return highlight


def _plot_drain_util_panel(ax, driver_tags, window_labels, values_by_window,
                           driver_colors, title,
                           show_xticklabels=True, show_xlabel=True,
                           x_step=1.03, group_width=0.98,
                           highlight_indices=None,
                           background_kwargs=None,
                           xlabel="Drain Period"):
    n_windows = len(values_by_window)
    n_tags = len(driver_tags)
    x = _period_x_positions(n_windows, step=x_step)
    width = group_width / max(n_tags, 1)

    if background_kwargs is None:
        background_kwargs = {}
    _draw_period_backgrounds(
        ax, x, group_width,
        highlight_indices=highlight_indices,
        **background_kwargs,
    )

    handles = []
    labels = []
    for idx, tag in enumerate(driver_tags):
        heights = [row[idx] if row[idx] is not None else 0.0 for row in values_by_window]
        offsets = x + (idx - (n_tags - 1) / 2.0) * width
        hatch = driver_hatch(tag)
        bars = ax.bar(
            offsets,
            heights,
            width=width * 0.95,
            color=driver_colors.get(tag, "gray"),
            alpha=0.88,
            hatch=hatch,
            edgecolor="black" if hatch else None,
            zorder=3,
        )
        handles.append(bars[0])
        labels.append(display_tag(tag))

    top_y = max(max(v for v in row if v is not None) for row in values_by_window)
    for window_idx, row in enumerate(values_by_window):
        valid_vals = [v for v in row if v is not None]
        if not valid_vals:
            continue
        best = max(valid_vals)
        for driver_idx, value in enumerate(row):
            if value is None or abs(value - best) > 1e-9:
                continue
            tag = driver_tags[driver_idx]
            star_color = "red" if tag.upper().startswith("MARS-CU") else "black"
            star_x = x[window_idx] + (driver_idx - (n_tags - 1) / 2.0) * width
            ax.text(
                star_x,
                value + 0.8,
                "★",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
                color=star_color,
                zorder=5,
            )

    ax.set_xticks(x)
    if show_xticklabels:
        xlabels = ax.set_xticklabels(window_labels, fontsize=10.5)
        for lbl in xlabels:
            lbl.set_fontweight("bold")
    else:
        ax.tick_params(axis="x", labelbottom=False, bottom=False)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, max(103.5, top_y + 3.5))
    ax.margins(x=0.005)
    ax.set_xlim(x[0] - group_width * 0.54, x[-1] + group_width * 0.54)
    ax.tick_params(axis="y", labelsize=11.5)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=4)
    return handles, labels


def _plot_drain_queue_before_panel(ax, driver_tags, window_labels, values_by_window,
                                   driver_colors, title,
                                   show_xticklabels=True, show_xlabel=True,
                                   x_step=1.03, group_width=0.98,
                                   highlight_indices=None,
                                   background_kwargs=None):
    n_windows = len(values_by_window)
    n_tags = len(driver_tags)
    x = _period_x_positions(n_windows, step=x_step)
    width = group_width / max(n_tags, 1)

    if background_kwargs is None:
        background_kwargs = {}
    _draw_period_backgrounds(
        ax, x, group_width,
        highlight_indices=highlight_indices,
        **background_kwargs,
    )

    handles = []
    labels = []
    for idx, tag in enumerate(driver_tags):
        heights = [row[idx] if row[idx] is not None else 0.0 for row in values_by_window]
        offsets = x + (idx - (n_tags - 1) / 2.0) * width
        hatch = driver_hatch(tag)
        bars = ax.bar(
            offsets,
            heights,
            width=width * 0.95,
            color=driver_colors.get(tag, "gray"),
            alpha=0.88,
            hatch=hatch,
            edgecolor="black" if hatch else None,
            zorder=3,
        )
        handles.append(bars[0])
        labels.append(display_tag(tag))

    top_y = max(
        max((v for v in row if v is not None), default=0.0)
        for row in values_by_window
    )
    ax.set_xticks(x)
    if show_xticklabels:
        xlabels = ax.set_xticklabels(window_labels, fontsize=10.5)
        for lbl in xlabels:
            lbl.set_fontweight("bold")
    else:
        ax.tick_params(axis="x", labelbottom=False, bottom=False)
    if show_xlabel:
        ax.set_xlabel("Drain Period", fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, _queue_before_ylim(top_y))
    ax.margins(x=0.005)
    ax.set_xlim(x[0] - group_width * 0.54, x[-1] + group_width * 0.54)
    ax.tick_params(axis="y", labelsize=11.5)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=4)
    return handles, labels


def plot_combined_drain_queue_boxplots(theta_table, polaris_table,
                                       theta_results_dir, polaris_results_dir,
                                       out_path):
    excluded = {"MARS-CB", "MARS-CW", "MARS-IU", "Random"}
    theta_tags, theta_labels, theta_samples = _collect_drain_queue_samples_by_period_from_table(
        theta_table, theta_results_dir, excluded_tags=excluded)
    polaris_tags, polaris_labels, polaris_samples = _collect_drain_queue_samples_by_period_from_table(
        polaris_table, polaris_results_dir, excluded_tags=excluded)
    if not theta_tags or not polaris_tags or not theta_labels or not polaris_labels:
        raise ValueError("Missing drain queue samples for combined plot")

    driver_colors = _build_driver_colors(theta_tags)
    theta_n = len(theta_labels)
    polaris_n = len(polaris_labels)
    total_n = theta_n + polaris_n
    per_window_w = 0.52
    fig_w = max(11.2, total_n * per_window_w + 1.9)

    fig = plt.figure(figsize=(fig_w, 5.0))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[theta_n, polaris_n],
        wspace=0.12,
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1], sharey=None),
    ]
    axes[1].sharey(axes[0])

    handles, labels = _draw_grouped_drain_queue_boxplots(
        axes[0], theta_tags, theta_labels, theta_samples, driver_colors,
        show_ylabel=True, tick_font=10)
    axes[0].set_title("Theta 2021", fontsize=11, pad=4)

    _draw_grouped_drain_queue_boxplots(
        axes[1], polaris_tags, polaris_labels, polaris_samples, driver_colors,
        show_ylabel=False, tick_font=10)
    axes[1].set_title("Polaris 2024", fontsize=11, pad=4)

    _add_centered_two_row_legend(
        fig, handles, labels, fontsize=9.0,
        y_top=0.985, row_gap=0.046,
        columnspacing=0.85, handletextpad=0.35,
    )
    fig.subplots_adjust(top=0.82, bottom=0.25, left=0.08, right=0.995, wspace=0.12)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_drain_queue_before_announcement(theta_table, polaris_table, out_path):
    excluded = {"MARS-CB", "MARS-CW", "MARS-IU", "Random"}
    theta_tags, theta_labels, theta_rows = _load_drain_queue_before_table(
        theta_table, excluded_tags=excluded)
    polaris_tags, polaris_labels, polaris_rows = _load_drain_queue_before_table(
        polaris_table, excluded_tags=excluded)
    if not theta_tags or not theta_rows or not polaris_tags or not polaris_rows:
        raise ValueError("Missing queue-before-announcement data for combined plot")

    driver_colors = _build_driver_colors(theta_tags)
    theta_n = len(theta_labels)
    polaris_n = len(polaris_labels)
    total_n = theta_n + polaris_n
    per_window_w = 0.52
    fig_w = max(11.2, total_n * per_window_w + 1.9)

    fig = plt.figure(figsize=(fig_w, 5.1))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[theta_n, polaris_n],
        wspace=0.08,
    )
    ax_t = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[0, 1], sharey=ax_t)

    handles, labels = _plot_drain_queue_before_panel(
        ax_t, theta_tags, theta_labels, theta_rows, driver_colors, "Theta 2021",
        show_xticklabels=True, show_xlabel=True,
    )
    _plot_drain_queue_before_panel(
        ax_p, polaris_tags, polaris_labels, polaris_rows, driver_colors, "Polaris 2024",
        show_xticklabels=True, show_xlabel=True,
    )
    ax_t.set_ylabel("Queue Before Announcement", fontsize=13, fontweight="bold")
    ax_p.tick_params(axis="y", labelleft=False, left=False)

    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(labels),
        frameon=False,
        fontsize=10.2,
        columnspacing=0.95,
        handletextpad=0.42,
    )
    for txt in legend.get_texts():
        txt.set_fontweight("bold")

    fig.subplots_adjust(top=0.81, bottom=0.22, left=0.08, right=0.995, wspace=0.08)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_drain_util(theta_table, polaris_table, out_path, extra_excluded=None):
    excluded = {"MARS-CB", "MARS-CW", "MARS-IU", "Random"}
    if extra_excluded:
        excluded = excluded | set(extra_excluded)
    theta_tags, theta_labels, theta_rows = _load_drain_util_table(theta_table, excluded_tags=excluded)
    polaris_tags, polaris_labels, polaris_rows = _load_drain_util_table(polaris_table, excluded_tags=excluded)
    if not theta_tags or not theta_rows or not polaris_tags or not polaris_rows:
        raise ValueError("Missing drain util data for combined plot")

    driver_colors = _build_driver_colors(list(theta_tags))

    theta_n = len(theta_labels)
    polaris_n = len(polaris_labels)
    total_n = theta_n + polaris_n
    per_window_w = 0.56
    fig_w = max(11.0, total_n * per_window_w + 1.45)

    fig = plt.figure(figsize=(fig_w, 4.65))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[theta_n, polaris_n],
        wspace=0.028,
    )
    ax_t = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[0, 1], sharey=ax_t)

    policy_handles, policy_labels = _plot_drain_util_panel(
        ax_t, theta_tags, theta_labels, theta_rows, driver_colors, "Theta 2021",
        show_xticklabels=True, show_xlabel=True,
        x_step=1.12, group_width=1.02,
    )
    _plot_drain_util_panel(
        ax_p, polaris_tags, polaris_labels, polaris_rows, driver_colors, "Polaris 2024",
        show_xticklabels=True, show_xlabel=True,
        x_step=1.12, group_width=1.02,
    )
    ax_t.set_ylabel("")
    ax_p.tick_params(axis="y", labelleft=False, left=False)

    fig.text(0.049, 0.53, "Utilization (%)",
             rotation=90, va="center", ha="center",
             fontsize=13, fontweight="bold")

    star_handles = [
        Line2D([0], [0], marker="*", linestyle="None", color="black", markersize=15, label="Best"),
        Line2D([0], [0], marker="*", linestyle="None", color="red", markersize=15, label="Best = MARS"),
    ]
    legend_handles = policy_handles + star_handles
    legend_labels = policy_labels + [h.get_label() for h in star_handles]
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=len(legend_labels),
        frameon=False,
        fontsize=11.8,
        columnspacing=1.05,
        handletextpad=0.46,
    )
    for txt in legend.get_texts():
        txt.set_fontweight("bold")
    fig.subplots_adjust(top=0.79, bottom=0.15, left=0.078, right=0.986, wspace=0.028)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


_CST = timezone(timedelta(hours=-6))

_UPTIME_SYS_SIZE = {
    2021: 4096,  # Theta
    2024: 496,   # Polaris
}


def _compute_uptime_periods(results_dir, analysis_start, analysis_end):
    """Return list of (start_ts, end_ts) uptime gaps between scheduled maintenance windows."""
    maint_windows = []
    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        events_path = os.path.join(results_dir, entry.name, "events.csv")
        if os.path.exists(events_path):
            maint_windows = parse_maintenance_from_events(events_path)
            break

    clipped = []
    for mstart, mend, mtype in maint_windows:
        if mtype != "S":
            continue
        cs = max(mstart, analysis_start)
        ce = min(mend, analysis_end)
        if ce > cs:
            clipped.append([cs, ce])

    clipped.sort()
    merged = []
    for cs, ce in clipped:
        if merged and cs <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], ce)
        else:
            merged.append([cs, ce])

    periods = []
    prev_end = analysis_start
    for ms, me in merged:
        if ms > prev_end:
            periods.append((prev_end, ms))
        prev_end = max(prev_end, me)
    if prev_end < analysis_end:
        periods.append((prev_end, analysis_end))
    return periods


def _load_uptime_util_for_results(results_dir, analysis_start, analysis_end,
                                   sys_size, excluded_tags=None):
    """Return (driver_tags, period_labels, values_by_period) for uptime periods.

    values_by_period[period_idx] is a list of util_pct per driver_tag.
    """
    excluded_upper = {t.upper() for t in (excluded_tags or set())}

    def _is_excl(tag):
        tag_up = tag.upper()
        return any(tag_up.startswith(e) for e in excluded_upper)

    periods = _compute_uptime_periods(results_dir, analysis_start, analysis_end)
    procs_map = _infer_procs_map_for_results_dir(results_dir)

    driver_tags = []
    start_maps = {}
    end_maps = {}

    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        tag = entry.name
        if _is_excl(tag):
            continue
        events_path = os.path.join(results_dir, tag, "events.csv")
        if not os.path.exists(events_path):
            continue
        _, start_map, end_map = parse_events(events_path)
        driver_tags.append(tag)
        start_maps[tag] = start_map
        end_maps[tag] = end_map

    period_labels = []
    for pstart, pend in periods:
        s = datetime.fromtimestamp(pstart, tz=_CST)
        e = datetime.fromtimestamp(pend, tz=_CST)
        period_labels.append(s.strftime("%m/%d") + "\nto\n" + e.strftime("%m/%d"))

    values_by_period = []
    for pstart, pend in periods:
        duration = pend - pstart
        row = []
        for tag in driver_tags:
            if duration <= 0:
                row.append(0.0)
                continue
            used = sum_used_core_seconds(start_maps[tag], end_maps[tag],
                                         procs_map, pstart, pend)
            row.append(100.0 * used / (duration * sys_size))
        values_by_period.append(row)

    return driver_tags, period_labels, values_by_period


def plot_combined_uptime_util_bar(theta_results_dir, polaris_results_dir, out_path):
    """Bar chart of utilization per uptime period, styled like bar_chart_util_drain_1day."""
    _EXCL = {"MARS-CB", "MARS-CW", "MARS-IU", "Random", "LRF", "RLScheduler"}

    theta_start  = int(datetime(2021, 1, 1, tzinfo=_CST).timestamp())
    theta_end    = int(datetime(2022, 1, 1, tzinfo=_CST).timestamp())
    polaris_start = int(datetime(2024, 1, 1, tzinfo=_CST).timestamp())
    polaris_end   = int(datetime(2025, 1, 1, tzinfo=_CST).timestamp())

    theta_tags, theta_labels, theta_rows = _load_uptime_util_for_results(
        theta_results_dir, theta_start, theta_end,
        sys_size=_UPTIME_SYS_SIZE[2021], excluded_tags=_EXCL)
    polaris_tags, polaris_labels, polaris_rows = _load_uptime_util_for_results(
        polaris_results_dir, polaris_start, polaris_end,
        sys_size=_UPTIME_SYS_SIZE[2024], excluded_tags=_EXCL)

    if not theta_tags or not theta_rows or not polaris_tags or not polaris_rows:
        raise ValueError("Missing uptime util data for combined plot")

    # Apply uptime-specific display order and reorder rows accordingly
    _UPTIME_POLICY_ORDER = ['WFP', 'MARS-CU', 'FCFS', 'SJF', 'F1',
                             'RLSCHEDULER', 'RANDOM', 'MARS-CW', 'LRF']
    _excl_up = {t.upper() for t in _GLOBAL_EXCLUDED}

    def _uptime_order(tags):
        def _key(tag):
            tag_up = tag.upper()
            for i, prefix in enumerate(_UPTIME_POLICY_ORDER):
                if tag_up.startswith(prefix):
                    return i
            return len(_UPTIME_POLICY_ORDER)
        return [t for t in sorted(tags, key=_key)
                if not any(t.upper().startswith(e) for e in _excl_up)]

    theta_order   = _uptime_order(theta_tags)
    polaris_order = _uptime_order(polaris_tags)

    def _reorder(tags, ordered, rows):
        idx_map = {t: i for i, t in enumerate(tags)}
        new_rows = [[row[idx_map[t]] for t in ordered] for row in rows]
        return ordered, new_rows

    theta_tags,   theta_rows   = _reorder(theta_tags,   theta_order,   theta_rows)
    polaris_tags, polaris_rows = _reorder(polaris_tags, polaris_order, polaris_rows)

    all_tags = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = _build_driver_colors(all_tags)

    theta_n   = len(theta_labels)
    polaris_n = len(polaris_labels)
    total_n   = theta_n + polaris_n
    per_window_w = 0.56
    fig_w = max(11.0, total_n * per_window_w + 1.45)

    fig = plt.figure(figsize=(fig_w, 4.65))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[theta_n, polaris_n],
        wspace=0.028,
    )
    ax_t = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[0, 1], sharey=ax_t)

    policy_handles, policy_labels_leg = _plot_drain_util_panel(
        ax_t, theta_tags, theta_labels, theta_rows, driver_colors, "Theta 2021",
        show_xticklabels=True, show_xlabel=True,
        x_step=1.10, group_width=1.00,
        xlabel="Uptime Period",
    )
    _plot_drain_util_panel(
        ax_p, polaris_tags, polaris_labels, polaris_rows, driver_colors, "Polaris 2024",
        show_xticklabels=True, show_xlabel=True,
        x_step=1.10, group_width=1.00,
        xlabel="Uptime Period",
    )
    ax_t.set_ylabel("")
    ax_p.tick_params(axis="y", labelleft=False, left=False)

    fig.text(0.049, 0.53, "Utilization (%)",
             rotation=90, va="center", ha="center",
             fontsize=13, fontweight="bold")

    star_handles = [
        Line2D([0], [0], marker="*", linestyle="None", color="black", markersize=15, label="Best"),
        Line2D([0], [0], marker="*", linestyle="None", color="red",   markersize=15, label="Best = MARS"),
    ]
    legend_handles = policy_handles + star_handles
    legend_labels  = policy_labels_leg + [h.get_label() for h in star_handles]
    legend = fig.legend(
        legend_handles, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.952),
        ncol=len(legend_labels), frameon=False,
        fontsize=11.8, columnspacing=1.05, handletextpad=0.46,
    )
    for txt in legend.get_texts():
        txt.set_fontweight("bold")
    fig.subplots_adjust(top=0.79, bottom=0.15, left=0.078, right=0.986, wspace=0.028)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _draw_util_boxplot_panel(ax, ordered_tags, util_data, driver_colors, title, show_ylabel, ymin=None):
    """Box plot panel for utilization (%) values per uptime period."""
    data = [np.asarray(util_data.get(t, [0.0]), dtype=float) for t in ordered_tags]
    positions = np.arange(1, len(ordered_tags) + 1)

    parts = ax.boxplot(
        data,
        positions=positions,
        widths=0.58,
        whis=1.5,
        showmeans=True,
        meanline=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=2.5, linestyle="none",
                        alpha=0.45, markeredgewidth=0.0),
        patch_artist=True,
        boxprops=dict(linewidth=1.9, edgecolor="black"),
        whiskerprops=dict(linewidth=1.3, color="black"),
        capprops=dict(linewidth=1.3, color="black"),
        medianprops=dict(linewidth=2.8, color="black"),
        meanprops=dict(linewidth=2.0, color="red", linestyle=":"),
    )
    for box, flier, tag in zip(parts["boxes"], parts["fliers"], ordered_tags):
        color = driver_colors.get(tag, "gray")
        box.set_facecolor(color)
        box.set_alpha(0.82)
        hatch = driver_hatch(tag)
        if hatch:
            box.set_hatch(hatch)
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    if ymin is None:
        all_vals = [v for vals in util_data.values() for v in vals if vals]
        ymin = max(0, np.floor(min(all_vals) - 2)) if all_vals else 0
    ax.set_ylim(ymin, 100)
    ax.tick_params(axis="y", labelsize=13, which="both")
    for lbl in ax.get_yticklabels(which="both"):
        lbl.set_fontweight("bold")
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)

    ax.set_xticks(positions)
    xlbls = ax.set_xticklabels(
        [display_tag(t) for t in ordered_tags],
        fontsize=12, rotation=25, ha="right",
    )
    for lbl in xlbls:
        lbl.set_fontweight("bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=5)
    ax.grid(True, axis="y", which="major", alpha=0.3, zorder=0)
    ax.set_xlim(0.4, len(ordered_tags) + 0.6)


def _load_uptime_util_per_tag(results_dir, analysis_start, analysis_end, sys_size,
                               excluded_tags=None):
    """Return {tag: [util_pct per uptime period]} using all available tags."""
    driver_tags, _, values_by_period = _load_uptime_util_for_results(
        results_dir, analysis_start, analysis_end, sys_size, excluded_tags=excluded_tags)
    result = {tag: [] for tag in driver_tags}
    for row in values_by_period:
        for tag, val in zip(driver_tags, row):
            result[tag].append(val)
    return result


def plot_combined_uptime_util_boxplot(theta_results_dir, polaris_results_dir, out_path,
                                       theta_util_csv=None, polaris_util_csv=None):
    """Box plots of per-uptime-period utilization for all strategies."""
    theta_start   = int(datetime(2021, 1, 1, tzinfo=_CST).timestamp())
    theta_end     = int(datetime(2022, 1, 1, tzinfo=_CST).timestamp())
    polaris_start = int(datetime(2024, 1, 1, tzinfo=_CST).timestamp())
    polaris_end   = int(datetime(2025, 1, 1, tzinfo=_CST).timestamp())

    theta_data   = _load_uptime_util_per_tag(theta_results_dir,
                                              theta_start, theta_end,
                                              _UPTIME_SYS_SIZE[2021],
                                              excluded_tags=_GLOBAL_EXCLUDED)
    polaris_data = _load_uptime_util_per_tag(polaris_results_dir,
                                              polaris_start, polaris_end,
                                              _UPTIME_SYS_SIZE[2024],
                                              excluded_tags=_GLOBAL_EXCLUDED)

    # RLScheduler has no per-period events — use its overall utilization as a single value
    rl_theta   = _read_rl_overall_util(theta_util_csv)
    rl_polaris = _read_rl_overall_util(polaris_util_csv)
    if rl_theta   is not None: theta_data["RLScheduler"]   = [rl_theta]
    if rl_polaris is not None: polaris_data["RLScheduler"] = [rl_polaris]

    theta_tags   = _order_all_tags(list(theta_data.keys()))
    polaris_tags = _order_all_tags(list(polaris_data.keys()))

    all_tags = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = _build_driver_colors(all_tags)

    all_vals = [v for d in (theta_data, polaris_data) for vals in d.values() for v in vals]
    shared_ymin = max(0, np.floor(min(all_vals) - 2)) if all_vals else 0

    n_tags = max(len(theta_tags), len(polaris_tags))
    fig_w = max(7.0, n_tags * 0.72 + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 4.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.22, wspace=0.06)

    _draw_util_boxplot_panel(axes[0], theta_tags,   theta_data,   driver_colors,
                             "Theta 2021",   show_ylabel=True,  ymin=shared_ymin)
    _draw_util_boxplot_panel(axes[1], polaris_tags, polaris_data, driver_colors,
                             "Polaris 2024", show_ylabel=False, ymin=shared_ymin)

    axes[0].set_ylabel("Utilization (%)", fontsize=13, fontweight="bold", labelpad=2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _read_rl_overall_util(util_csv):
    """Return RLScheduler's overall utilization % from table_util.csv, or None."""
    if not util_csv or not os.path.exists(util_csv):
        return None
    with open(util_csv) as f:
        for row in csv.DictReader(f):
            if row.get("Driver", "").upper().startswith("RLSCHEDULER"):
                return float(row["Utilization_Percent"])
    return None


def _load_drain_1day_util_per_tag(csv_path, excluded_tags=None):
    """Return {tag: [util_pct per drain period]} from a table_util_drain_1day.csv."""
    excluded_upper = {t.upper() for t in (excluded_tags or set())}

    def _is_excl(tag):
        tag_up = tag.upper()
        return any(tag_up.startswith(e) for e in excluded_upper)

    result = {}
    if not os.path.exists(csv_path):
        return result
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col, val in row.items():
                if col in ("Draining_Period_Start", "Draining_Period_End"):
                    continue
                if _is_excl(col):
                    continue
                pct = float(val.strip().rstrip("%"))
                result.setdefault(col, []).append(pct)
    return result


def plot_combined_drain_1day_util_boxplot(theta_csv, polaris_csv, out_path,
                                           theta_util_csv=None, polaris_util_csv=None):
    """Box plots of per-drain-period utilization (24 h before downtime) for all strategies."""
    _EXCL = {"MARS-CB", "MARS-IU"}

    theta_data   = _load_drain_1day_util_per_tag(theta_csv,   excluded_tags=_EXCL)
    polaris_data = _load_drain_1day_util_per_tag(polaris_csv, excluded_tags=_EXCL)

    theta_tags   = _order_all_tags(list(theta_data.keys()))
    polaris_tags = _order_all_tags(list(polaris_data.keys()))

    all_tags = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = _build_driver_colors(all_tags)

    all_vals = [v for d in (theta_data, polaris_data) for vals in d.values() for v in vals]
    shared_ymin = max(0, np.floor(min(all_vals) - 2)) if all_vals else 0

    n_tags = max(len(theta_tags), len(polaris_tags))
    fig_w = max(7.0, n_tags * 0.72 + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 4.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.22, wspace=0.06)

    _draw_util_boxplot_panel(axes[0], theta_tags,   theta_data,   driver_colors,
                             "Theta 2021",   show_ylabel=True,  ymin=shared_ymin)
    _draw_util_boxplot_panel(axes[1], polaris_tags, polaris_data, driver_colors,
                             "Polaris 2024", show_ylabel=False, ymin=shared_ymin)

    axes[0].set_ylabel("Utilization (%)", fontsize=13, fontweight="bold", labelpad=2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_drain_util_and_queue(theta_table, polaris_table,
                                       theta_results_dir, polaris_results_dir,
                                       out_path):
    excluded = {"MARS-CB", "MARS-CW", "MARS-IU", "Random"}
    theta_tags, theta_labels, theta_rows = _load_drain_util_table(
        theta_table, excluded_tags=excluded)
    polaris_tags, polaris_labels, polaris_rows = _load_drain_util_table(
        polaris_table, excluded_tags=excluded)
    theta_q_tags, theta_q_labels, theta_q_samples = _collect_drain_queue_samples_by_period_from_table(
        theta_table, theta_results_dir, excluded_tags=excluded)
    polaris_q_tags, polaris_q_labels, polaris_q_samples = _collect_drain_queue_samples_by_period_from_table(
        polaris_table, polaris_results_dir, excluded_tags=excluded)

    if (not theta_tags or not theta_rows or not polaris_tags or not polaris_rows or
            not theta_q_tags or not theta_q_labels or not polaris_q_tags or not polaris_q_labels):
        raise ValueError("Missing drain util/queue data for combined plot")

    legend_tags = list(theta_tags)
    driver_colors = _build_driver_colors(legend_tags)
    theta_highlight = _highlight_non_mars_best_periods(theta_tags, theta_rows, theta_q_samples)
    polaris_highlight = _highlight_non_mars_best_periods(polaris_tags, polaris_rows, polaris_q_samples)
    highlight_bg = {
        "highlight_facecolor": "#f0b6b6",
        "highlight_alpha": 0.55,
        "highlight_edgecolor": None,
    }

    theta_n = len(theta_labels)
    polaris_n = len(polaris_labels)
    total_n = theta_n + polaris_n
    per_window_w = 0.56
    fig_w = max(11.0, total_n * per_window_w + 1.45)

    fig = plt.figure(figsize=(fig_w, 8.15))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[theta_n, polaris_n],
        height_ratios=[1.0, 1.08],
        hspace=0.075,
        wspace=0.028,
    )
    ax_t_util = fig.add_subplot(gs[0, 0])
    ax_p_util = fig.add_subplot(gs[0, 1], sharey=ax_t_util)
    ax_t_q = fig.add_subplot(gs[1, 0], sharex=ax_t_util)
    ax_p_q = fig.add_subplot(gs[1, 1], sharex=ax_p_util, sharey=ax_t_q)
    x_step = 1.12
    group_width = 1.02

    policy_handles, policy_labels = _plot_drain_util_panel(
        ax_t_util, theta_tags, theta_labels, theta_rows, driver_colors, "Theta 2021",
        show_xticklabels=False, show_xlabel=False,
        x_step=x_step, group_width=group_width,
        highlight_indices=theta_highlight,
        background_kwargs=highlight_bg,
    )
    _plot_drain_util_panel(
        ax_p_util, polaris_tags, polaris_labels, polaris_rows, driver_colors, "Polaris 2024",
        show_xticklabels=False, show_xlabel=False,
        x_step=x_step, group_width=group_width,
        highlight_indices=polaris_highlight,
        background_kwargs=highlight_bg,
    )

    _draw_grouped_drain_queue_boxplots(
        ax_t_q, theta_q_tags, theta_q_labels, theta_q_samples, driver_colors,
        show_ylabel=False, tick_font=11,
        show_xticklabels=True, show_xlabel=True,
        x_step=x_step, group_width=group_width,
        highlight_indices=theta_highlight,
        background_kwargs=highlight_bg,
    )
    _draw_grouped_drain_queue_boxplots(
        ax_p_q, polaris_q_tags, polaris_q_labels, polaris_q_samples, driver_colors,
        show_ylabel=False, tick_font=11,
        show_xticklabels=True, show_xlabel=True,
        x_step=x_step, group_width=group_width,
        highlight_indices=polaris_highlight,
        background_kwargs=highlight_bg,
    )
    ax_t_util.set_ylabel("")
    ax_t_q.set_ylabel("")
    ax_p_util.tick_params(axis="y", labelleft=False, left=False)
    ax_p_q.tick_params(axis="y", labelleft=False, left=False)

    fig.text(0.049, 0.665, "Utilization (%)",
             rotation=90, va="center", ha="center",
             fontsize=13, fontweight="bold")
    fig.text(0.049, 0.255, "Queue Size",
             rotation=90, va="center", ha="center",
             fontsize=13, fontweight="bold")

    star_handles = [
        Line2D([0], [0], marker="*", linestyle="None", color="black", markersize=15, label="Best"),
        Line2D([0], [0], marker="*", linestyle="None", color="red", markersize=15, label="Best = MARS"),
    ]
    legend_handles = policy_handles + star_handles
    legend_labels = policy_labels + [h.get_label() for h in star_handles]
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=len(legend_labels),
        frameon=False,
        fontsize=22,
        columnspacing=1.05,
        handletextpad=0.46,
    )
    for txt in legend.get_texts():
        txt.set_fontweight("bold")
    fig.subplots_adjust(top=0.79, bottom=0.11, left=0.078, right=0.986, wspace=0.028, hspace=0.30)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_drain_util_and_queue_before_announcement(theta_util_table, polaris_util_table,
                                                           theta_queue_table, polaris_queue_table,
                                                           out_path):
    excluded = {"MARS-CB", "MARS-CW", "MARS-IU", "Random"}
    theta_tags, theta_labels, theta_rows = _load_drain_util_table(
        theta_util_table, excluded_tags=excluded)
    polaris_tags, polaris_labels, polaris_rows = _load_drain_util_table(
        polaris_util_table, excluded_tags=excluded)
    theta_q_tags, theta_q_labels, theta_q_rows = _load_drain_queue_before_table(
        theta_queue_table, excluded_tags=excluded)
    polaris_q_tags, polaris_q_labels, polaris_q_rows = _load_drain_queue_before_table(
        polaris_queue_table, excluded_tags=excluded)

    if (not theta_tags or not theta_rows or not polaris_tags or not polaris_rows or
            not theta_q_tags or not theta_q_labels or not polaris_q_tags or not polaris_q_labels):
        raise ValueError("Missing drain util/queue-before-announcement data for combined plot")

    driver_colors = _build_driver_colors(list(theta_tags))
    theta_highlight = _highlight_non_mars_best_periods(theta_tags, theta_rows, {})
    polaris_highlight = _highlight_non_mars_best_periods(polaris_tags, polaris_rows, {})
    highlight_bg = {
        "highlight_facecolor": "#f0b6b6",
        "highlight_alpha": 0.55,
        "highlight_edgecolor": None,
    }

    theta_n = len(theta_labels)
    polaris_n = len(polaris_labels)
    total_n = theta_n + polaris_n
    per_window_w = 0.56
    fig_w = max(11.0, total_n * per_window_w + 1.45)

    fig = plt.figure(figsize=(fig_w, 8.15))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[theta_n, polaris_n],
        height_ratios=[1.0, 1.08],
        hspace=0.075,
        wspace=0.028,
    )
    ax_t_util = fig.add_subplot(gs[0, 0])
    ax_p_util = fig.add_subplot(gs[0, 1], sharey=ax_t_util)
    ax_t_q = fig.add_subplot(gs[1, 0], sharex=ax_t_util)
    ax_p_q = fig.add_subplot(gs[1, 1], sharex=ax_p_util, sharey=ax_t_q)
    x_step = 1.12
    group_width = 1.02

    policy_handles, policy_labels = _plot_drain_util_panel(
        ax_t_util, theta_tags, theta_labels, theta_rows, driver_colors, "Theta 2021",
        show_xticklabels=False, show_xlabel=False,
        x_step=x_step, group_width=group_width,
        highlight_indices=theta_highlight,
        background_kwargs=highlight_bg,
    )
    _plot_drain_util_panel(
        ax_p_util, polaris_tags, polaris_labels, polaris_rows, driver_colors, "Polaris 2024",
        show_xticklabels=False, show_xlabel=False,
        x_step=x_step, group_width=group_width,
        highlight_indices=polaris_highlight,
        background_kwargs=highlight_bg,
    )

    _plot_drain_queue_before_panel(
        ax_t_q, theta_q_tags, theta_q_labels, theta_q_rows, driver_colors, "",
        show_xticklabels=True, show_xlabel=True,
        x_step=x_step, group_width=group_width,
        highlight_indices=theta_highlight,
        background_kwargs=highlight_bg,
    )
    _plot_drain_queue_before_panel(
        ax_p_q, polaris_q_tags, polaris_q_labels, polaris_q_rows, driver_colors, "",
        show_xticklabels=True, show_xlabel=True,
        x_step=x_step, group_width=group_width,
        highlight_indices=polaris_highlight,
        background_kwargs=highlight_bg,
    )

    ax_t_util.set_ylabel("")
    ax_t_q.set_ylabel("")
    ax_p_util.tick_params(axis="y", labelleft=False, left=False)
    ax_p_q.tick_params(axis="y", labelleft=False, left=False)

    fig.text(0.049, 0.665, "Utilization (%)",
             rotation=90, va="center", ha="center",
             fontsize=13, fontweight="bold")
    fig.text(0.049, 0.255, "Queue Before Announcement",
             rotation=90, va="center", ha="center",
             fontsize=13, fontweight="bold")

    star_handles = [
        Line2D([0], [0], marker="*", linestyle="None", color="black", markersize=15, label="Best"),
        Line2D([0], [0], marker="*", linestyle="None", color="red", markersize=15, label="Best = MARS"),
    ]
    legend_handles = policy_handles + star_handles
    legend_labels = policy_labels + [h.get_label() for h in star_handles]
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=len(legend_labels),
        frameon=False,
        fontsize=11.8,
        columnspacing=1.05,
        handletextpad=0.46,
    )
    for txt in legend.get_texts():
        txt.set_fontweight("bold")
    fig.subplots_adjust(top=0.79, bottom=0.11, left=0.078, right=0.986, wspace=0.028, hspace=0.30)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_drain_core_hours(theta_util_table, polaris_util_table,
                                   theta_core_table, polaris_core_table,
                                   out_path):
    excluded = {"MARS-CB", "MARS-CW", "MARS-IU", "Random"}
    theta_tags, theta_labels, theta_util_rows = _load_drain_util_table(
        theta_util_table, excluded_tags=excluded)
    polaris_tags, polaris_labels, polaris_util_rows = _load_drain_util_table(
        polaris_util_table, excluded_tags=excluded)
    theta_sched_tags, theta_sched_labels, theta_sched_rows = _load_drain_core_hours_table(
        theta_core_table, "Scheduled_Core_Hours", excluded_tags=excluded)
    pol_sched_tags, pol_sched_labels, pol_sched_rows = _load_drain_core_hours_table(
        polaris_core_table, "Scheduled_Core_Hours", excluded_tags=excluded)
    theta_sub_tags, theta_sub_labels, theta_sub_rows = _load_drain_core_hours_table(
        theta_core_table, "Submitted_Core_Hours", excluded_tags=excluded)
    pol_sub_tags, pol_sub_labels, pol_sub_rows = _load_drain_core_hours_table(
        polaris_core_table, "Submitted_Core_Hours", excluded_tags=excluded)

    if (not theta_tags or not theta_util_rows or not polaris_tags or not polaris_util_rows or
            not theta_sched_tags or not theta_sched_rows or not pol_sched_tags or not pol_sched_rows or
            not theta_sub_tags or not theta_sub_rows or not pol_sub_tags or not pol_sub_rows):
        raise ValueError("Missing drain core-hours data for combined plot")

    driver_colors = _build_driver_colors(list(theta_tags))
    theta_highlight = _highlight_non_mars_best_periods(theta_tags, theta_util_rows, {})
    polaris_highlight = _highlight_non_mars_best_periods(polaris_tags, polaris_util_rows, {})
    highlight_bg = {
        "highlight_facecolor": "#f0b6b6",
        "highlight_alpha": 0.55,
        "highlight_edgecolor": None,
    }

    theta_n = len(theta_labels)
    polaris_n = len(polaris_labels)
    total_n = theta_n + polaris_n
    per_window_w = 0.56
    fig_w = max(11.0, total_n * per_window_w + 1.45)

    fig = plt.figure(figsize=(fig_w, 8.15))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[theta_n, polaris_n],
        height_ratios=[1.0, 1.08],
        hspace=0.075,
        wspace=0.028,
    )
    ax_t_sched = fig.add_subplot(gs[0, 0])
    ax_p_sched = fig.add_subplot(gs[0, 1])
    ax_t_sub = fig.add_subplot(gs[1, 0], sharex=ax_t_sched)
    ax_p_sub = fig.add_subplot(gs[1, 1], sharex=ax_p_sched)

    policy_handles, policy_labels = _plot_drain_queue_before_panel(
        ax_t_sched, theta_sched_tags, theta_sched_labels, theta_sched_rows, driver_colors, "Theta 2021",
        show_xticklabels=False, show_xlabel=False,
        x_step=1.12, group_width=1.02,
        highlight_indices=theta_highlight,
        background_kwargs=highlight_bg,
    )
    _plot_drain_queue_before_panel(
        ax_p_sched, pol_sched_tags, pol_sched_labels, pol_sched_rows, driver_colors, "Polaris 2024",
        show_xticklabels=False, show_xlabel=False,
        x_step=1.12, group_width=1.02,
        highlight_indices=polaris_highlight,
        background_kwargs=highlight_bg,
    )
    _plot_drain_queue_before_panel(
        ax_t_sub, theta_sub_tags, theta_sub_labels, theta_sub_rows, driver_colors, "",
        show_xticklabels=True, show_xlabel=True,
        x_step=1.12, group_width=1.02,
        highlight_indices=theta_highlight,
        background_kwargs=highlight_bg,
    )
    _plot_drain_queue_before_panel(
        ax_p_sub, pol_sub_tags, pol_sub_labels, pol_sub_rows, driver_colors, "",
        show_xticklabels=True, show_xlabel=True,
        x_step=1.12, group_width=1.02,
        highlight_indices=polaris_highlight,
        background_kwargs=highlight_bg,
    )

    ax_t_sched.set_ylabel("")
    ax_t_sub.set_ylabel("")
    ax_p_sched.tick_params(axis="y", labelsize=11.5)
    ax_p_sub.tick_params(axis="y", labelsize=11.5)
    for lbl in ax_p_sched.get_yticklabels():
        lbl.set_fontweight("bold")
    for lbl in ax_p_sub.get_yticklabels():
        lbl.set_fontweight("bold")

    fig.text(0.049, 0.665, "Scheduled Core-Hours",
             rotation=90, va="center", ha="center",
             fontsize=13, fontweight="bold")
    fig.text(0.049, 0.255, "Submitted Core-Hours",
             rotation=90, va="center", ha="center",
             fontsize=13, fontweight="bold")

    star_handles = [
        Line2D([0], [0], marker="*", linestyle="None", color="black", markersize=15, label="Best"),
        Line2D([0], [0], marker="*", linestyle="None", color="red", markersize=15, label="Best = MARS"),
    ]
    legend_handles = policy_handles + star_handles
    legend_labels = policy_labels + [h.get_label() for h in star_handles]
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=len(legend_labels),
        frameon=False,
        fontsize=11.8,
        columnspacing=1.05,
        handletextpad=0.46,
    )
    for txt in legend.get_texts():
        txt.set_fontweight("bold")
    fig.subplots_adjust(top=0.79, bottom=0.11, left=0.078, right=0.986, wspace=0.028, hspace=0.30)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _load_pre_announcement_core_hours(util_table, results_dir, excluded_tags=None):
    """For each drain period, compute core-hours scheduled from jobs that were
    submitted *before* the announcement (i.e. already queued or running at
    announcement time) and that actually started during the drain window."""
    excluded = set() if excluded_tags is None else excluded_tags
    driver_tags_raw = _load_driver_columns_from_table(util_table, excluded_tags=excluded)
    periods = _load_drain_periods_from_table(util_table)
    labels = _load_drain_period_labels_from_table(util_table)
    procs_map = _infer_procs_map_for_results_dir(results_dir)
    walltimes_map = _infer_walltimes_map_for_results_dir(results_dir)

    driver_tags = []
    # period_values[period_idx][tag] = core-hours float
    period_values = [{} for _ in periods]

    for tag in driver_tags_raw:
        events_path = os.path.join(results_dir, tag, "events.csv")
        if not os.path.exists(events_path):
            continue
        driver_tags.append(tag)
        submit_map, start_map, _ = parse_events(events_path)
        for pidx, (d_start, d_end) in enumerate(periods):
            core_h = 0.0
            for jid, start_time in start_map.items():
                if not (d_start <= start_time < d_end):
                    continue
                submit_time = submit_map.get(jid)
                if submit_time is None or submit_time >= d_start:
                    continue
                procs = procs_map.get(jid)
                walltime = walltimes_map.get(jid)
                if procs is not None and walltime is not None:
                    core_h += procs * walltime / 3600.0
            period_values[pidx][tag] = core_h

    rows = [
        [pv.get(tag) for tag in driver_tags]
        for pv in period_values
    ]
    return driver_tags, labels, rows


def _plot_drain_pre_announcement_core_hours_panel(ax, driver_tags, window_labels,
                                                   values_by_window, driver_colors, title,
                                                   show_xticklabels=True, show_xlabel=True,
                                                   show_ylabel_ticks=True,
                                                   x_step=1.12, group_width=1.02,
                                                   highlight_indices=None,
                                                   background_kwargs=None):
    """Bar chart of pre-announcement core-hours per drain period."""
    n_windows = len(values_by_window)
    n_tags = len(driver_tags)
    x = _period_x_positions(n_windows, step=x_step)
    width = group_width / max(n_tags, 1)

    if background_kwargs is None:
        background_kwargs = {}
    _draw_period_backgrounds(ax, x, group_width,
                             highlight_indices=highlight_indices,
                             **background_kwargs)

    handles = []
    labels_out = []
    for idx, tag in enumerate(driver_tags):
        heights = [row[idx] if row[idx] is not None else 0.0 for row in values_by_window]
        offsets = x + (idx - (n_tags - 1) / 2.0) * width
        hatch = driver_hatch(tag)
        bars = ax.bar(
            offsets, heights,
            width=width * 0.95,
            color=driver_colors.get(tag, "gray"),
            alpha=0.88,
            hatch=hatch,
            edgecolor="black" if hatch else None,
            zorder=3,
        )
        handles.append(bars[0])
        labels_out.append(display_tag(tag))

    top_y = max(
        max((v for v in row if v is not None), default=0.0)
        for row in values_by_window
    )
    ax.set_ylim(0.0, max(1.0, top_y * 1.15))
    ax.margins(x=0.005)
    ax.set_xlim(x[0] - group_width * 0.54, x[-1] + group_width * 0.54)
    ax.set_xticks(x)
    if show_xticklabels:
        xlabels = ax.set_xticklabels(window_labels, fontsize=10.5)
        for lbl in xlabels:
            lbl.set_fontweight("bold")
    else:
        ax.tick_params(axis="x", labelbottom=False, bottom=False)
    if show_xlabel:
        ax.set_xlabel("Drain Period", fontsize=13, fontweight="bold")
    if show_ylabel_ticks:
        ax.tick_params(axis="y", labelsize=11.5)
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight("bold")
    else:
        ax.tick_params(axis="y", labelleft=False, left=False)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    if title:
        ax.set_title(title, fontsize=13.5, fontweight="bold", pad=4)
    return handles, labels_out


def plot_combined_drain_util_and_pre_announcement_core_hours(
        theta_util_table, polaris_util_table,
        theta_results_dir, polaris_results_dir,
        out_path):
    """Two-row combined figure.

    Top row:    overall utilization (%) per drain period.
    Bottom row: node-hours scheduled from jobs submitted before the
                announcement (already queued or running at announcement time).
    """
    excluded = {"MARS-CB", "MARS-CW", "MARS-IU", "Random"}

    theta_tags, theta_labels, theta_util_rows = _load_drain_util_table(
        theta_util_table, excluded_tags=excluded)
    polaris_tags, polaris_labels, polaris_util_rows = _load_drain_util_table(
        polaris_util_table, excluded_tags=excluded)
    theta_pre_tags, theta_pre_labels, theta_pre_rows = _load_pre_announcement_core_hours(
        theta_util_table, theta_results_dir, excluded_tags=excluded)
    polaris_pre_tags, polaris_pre_labels, polaris_pre_rows = _load_pre_announcement_core_hours(
        polaris_util_table, polaris_results_dir, excluded_tags=excluded)

    if (not theta_tags or not theta_util_rows or not polaris_tags or not polaris_util_rows
            or not theta_pre_tags or not polaris_pre_tags):
        raise ValueError("Missing data for pre-announcement core-hours combined plot")

    driver_colors = _build_driver_colors(list(theta_tags))
    theta_highlight = _highlight_non_mars_best_periods(theta_tags, theta_util_rows, {})
    polaris_highlight = _highlight_non_mars_best_periods(polaris_tags, polaris_util_rows, {})
    highlight_bg = {
        "highlight_facecolor": "#f0b6b6",
        "highlight_alpha": 0.55,
        "highlight_edgecolor": None,
    }

    theta_n = len(theta_labels)
    polaris_n = len(polaris_labels)
    total_n = theta_n + polaris_n
    per_window_w = 0.56
    fig_w = max(11.0, total_n * per_window_w + 1.45)

    fig = plt.figure(figsize=(fig_w, 8.15))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[theta_n, polaris_n],
        height_ratios=[1.0, 1.08],
        hspace=0.075,
        wspace=0.028,
    )
    ax_t_util = fig.add_subplot(gs[0, 0])
    ax_p_util = fig.add_subplot(gs[0, 1], sharey=ax_t_util)
    # Independent Y scales: Theta has 4096 nodes, Polaris has 496 — very different magnitudes.
    ax_t_pre = fig.add_subplot(gs[1, 0], sharex=ax_t_util)
    ax_p_pre = fig.add_subplot(gs[1, 1], sharex=ax_p_util)
    x_step = 1.12
    group_width = 1.02

    policy_handles, policy_labels_leg = _plot_drain_util_panel(
        ax_t_util, theta_tags, theta_labels, theta_util_rows, driver_colors, "Theta 2021",
        show_xticklabels=False, show_xlabel=False,
        x_step=x_step, group_width=group_width,
        highlight_indices=theta_highlight,
        background_kwargs=highlight_bg,
    )
    _plot_drain_util_panel(
        ax_p_util, polaris_tags, polaris_labels, polaris_util_rows, driver_colors, "Polaris 2024",
        show_xticklabels=False, show_xlabel=False,
        x_step=x_step, group_width=group_width,
        highlight_indices=polaris_highlight,
        background_kwargs=highlight_bg,
    )

    _plot_drain_pre_announcement_core_hours_panel(
        ax_t_pre, theta_pre_tags, theta_pre_labels, theta_pre_rows, driver_colors, "",
        show_xticklabels=True, show_xlabel=True, show_ylabel_ticks=True,
        x_step=x_step, group_width=group_width,
        highlight_indices=theta_highlight,
        background_kwargs=highlight_bg,
    )
    _plot_drain_pre_announcement_core_hours_panel(
        ax_p_pre, polaris_pre_tags, polaris_pre_labels, polaris_pre_rows, driver_colors, "",
        show_xticklabels=True, show_xlabel=True, show_ylabel_ticks=True,
        x_step=x_step, group_width=group_width,
        highlight_indices=polaris_highlight,
        background_kwargs=highlight_bg,
    )

    ax_t_util.set_ylabel("")
    ax_p_util.tick_params(axis="y", labelleft=False, left=False)

    fig.text(0.049, 0.665, "Utilization (%)",
             rotation=90, va="center", ha="center",
             fontsize=13, fontweight="bold")
    fig.text(0.049, 0.255, "Pre-Announce.\nCore-Hours",
             rotation=90, va="center", ha="center",
             fontsize=13, fontweight="bold")

    star_handles = [
        Line2D([0], [0], marker="*", linestyle="None", color="black", markersize=15, label="Best"),
        Line2D([0], [0], marker="*", linestyle="None", color="red", markersize=15, label="Best = MARS"),
    ]
    legend_handles = policy_handles + star_handles
    legend_labels_all = policy_labels_leg + [h.get_label() for h in star_handles]
    legend = fig.legend(
        legend_handles,
        legend_labels_all,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=len(legend_labels_all),
        frameon=False,
        fontsize=11.8,
        columnspacing=1.05,
        handletextpad=0.46,
    )
    for txt in legend.get_texts():
        txt.set_fontweight("bold")
    fig.subplots_adjust(top=0.79, bottom=0.11, left=0.078, right=0.986, wspace=0.028, hspace=0.30)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _load_rlscheduler_entry(results_dir):
    """Find the RLScheduler CSV via the experiment JSON and return (tag, submit_map, start_map, end_map).
    Derives the JSON path as experiments/<basename-of-results-dir>.json."""
    base = os.path.basename(os.path.normpath(results_dir))
    json_path = os.path.join('experiments', base + '.json')
    if not os.path.exists(json_path):
        return None, {}, {}, {}
    with open(json_path) as f:
        cfg = json.load(f)
    for d in cfg.get('drivers', []):
        if d.get('type') == 'rlscheduler':
            csv_path = d.get('csv_path', '')
            tag = d.get('tag', 'RLScheduler')
            if not os.path.exists(csv_path):
                print(f"  [skip] RLScheduler: csv_path not found: {csv_path}")
                return tag, {}, {}, {}
            submit, start, end = parse_rlscheduler_csv(csv_path)
            return tag, submit, start, end
    return None, {}, {}, {}


def _load_wait_hours_for_tags(results_dir, tag_prefixes=None):
    """Return {tag: [wait_hours > 0]} for result subfolders matching tag_prefixes.
    Pass tag_prefixes=None to load all subfolders."""
    data = {}
    if not os.path.isdir(results_dir):
        return data
    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        tag = entry.name
        tag_up = tag.upper()
        if tag_prefixes is not None and not any(
                tag_up.startswith(p.upper()) for p in tag_prefixes):
            continue
        events_path = os.path.join(results_dir, tag, "events.csv")
        if not os.path.exists(events_path):
            continue
        submit_map, start_map, _ = parse_events(events_path)
        waits = [
            (start_map[jid] - submit_map[jid]) / 3600.0
            for jid in set(submit_map) & set(start_map)
            if start_map[jid] - submit_map[jid] > 0
        ]
        data[tag] = waits

    # RLScheduler has no events.csv — load from its own CSV via experiment JSON
    rl_tag, rl_submit, rl_start, _ = _load_rlscheduler_entry(results_dir)
    if rl_tag and rl_submit and rl_start:
        if tag_prefixes is None or any(
                rl_tag.upper().startswith(p.upper()) for p in tag_prefixes):
            data[rl_tag] = [
                (rl_start[jid] - rl_submit[jid]) / 3600.0
                for jid in set(rl_submit) & set(rl_start)
                if rl_start[jid] - rl_submit[jid] > 0
            ]
    return data


def _draw_wait_boxplot_panel(ax, ordered_tags, wait_data, driver_colors,
                              title, show_ylabel):
    data = [np.asarray(wait_data.get(t, [0.001]), dtype=float) for t in ordered_tags]
    positions = np.arange(1, len(ordered_tags) + 1)

    parts = ax.boxplot(
        data,
        positions=positions,
        widths=0.58,
        whis=(5, 95),
        showmeans=True,
        meanline=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=2.0, linestyle='none',
                        alpha=0.35, markeredgewidth=0.0),
        patch_artist=True,
        boxprops=dict(linewidth=1.9, edgecolor='black'),
        whiskerprops=dict(linewidth=1.3, color='black'),
        capprops=dict(linewidth=1.3, color='black'),
        medianprops=dict(linewidth=2.8, color='black'),
        meanprops=dict(linewidth=2.0, color='red', linestyle=':'),
    )
    for box, flier, tag in zip(parts['boxes'], parts['fliers'], ordered_tags):
        color = driver_colors.get(tag, 'gray')
        box.set_facecolor(color)
        box.set_alpha(0.82)
        hatch = driver_hatch(tag)
        if hatch:
            box.set_hatch(hatch)
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.1f}"
        )
    )
    ax.tick_params(axis='y', labelsize=13, which='both')
    for lbl in ax.get_yticklabels(which='both'):
        lbl.set_fontweight('bold')
    if not show_ylabel:
        ax.tick_params(axis='y', labelleft=False)

    ax.set_xticks(positions)
    xlbls = ax.set_xticklabels(
        [display_tag(t) for t in ordered_tags],
        fontsize=12, rotation=25, ha='right',
    )
    for lbl in xlbls:
        lbl.set_fontweight('bold')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=5)
    ax.grid(True, axis='y', which='major', alpha=0.3, zorder=0)
    ax.set_xlim(0.4, len(ordered_tags) + 0.6)


def plot_combined_wait_boxplot(theta_results_dir, polaris_results_dir, out_path,
                               tag_prefixes=("WFP", "MARS-CU")):
    """Box plots of raw wait times for WFP and MARS-CU.
    Left panel = Theta 2021, right panel = Polaris 2024.
    Sized to fit a single column of a two-column paper.
    """
    theta_data   = _load_wait_hours_for_tags(theta_results_dir,   tag_prefixes)
    polaris_data = _load_wait_hours_for_tags(polaris_results_dir, tag_prefixes)

    def _order(data):
        WFP    = sorted(t for t in data if t.upper().startswith('WFP'))
        mars_cw = sorted(t for t in data if t.upper().startswith('MARS-CW'))
        mars_cu = sorted(t for t in data if t.upper().startswith('MARS-CU'))
        return WFP + mars_cw + mars_cu

    theta_tags   = _order(theta_data)
    polaris_tags = _order(polaris_data)

    all_tags = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = _build_driver_colors(all_tags)

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 4.2))
    fig.subplots_adjust(left=0.21, right=0.97, top=0.88, bottom=0.20, wspace=0.08)

    _draw_wait_boxplot_panel(axes[0], theta_tags,   theta_data,   driver_colors,
                             "Theta 2021",   show_ylabel=True)
    _draw_wait_boxplot_panel(axes[1], polaris_tags, polaris_data, driver_colors,
                             "Polaris 2024", show_ylabel=False)

    axes[0].set_ylabel("Wait Time (hours)", fontsize=13, fontweight='bold', labelpad=2)

    # Align y-axes to the same range
    all_vals = [v for d in (theta_data, polaris_data) for vals in d.values() for v in vals if v > 0]
    if all_vals:
        ymin = max(0.01, np.percentile(all_vals, 1) * 0.5)
        ymax = max(np.percentile(all_vals, 99) * 2.0, 1000.0)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _order_all_tags(tags):
    """Canonical display order. Globally excluded tags (MARS-CB, MARS-IU) are dropped."""
    _ORDER = ['RLSCHEDULER', 'RANDOM', 'WFP', 'MARS-CW', 'MARS-CU', 'FCFS', 'SJF', 'F1', 'LRF']
    _excl_up = {t.upper() for t in _GLOBAL_EXCLUDED}

    def _is_excl(tag):
        tag_up = tag.upper()
        return any(tag_up.startswith(e) for e in _excl_up)

    def _key(tag):
        tag_up = tag.upper()
        for i, prefix in enumerate(_ORDER):
            if tag_up.startswith(prefix):
                return i
        return len(_ORDER)

    return [t for t in sorted(tags, key=_key) if not _is_excl(t)]


# Theta: 128 / 129-256 / 257-1024 / 1025-4096
# Polaris: 10-16 / 17-32 / 33-128 / 129-496
def _assign_group_theta(n):
    if n == 128:              return 'S'
    if 129  <= n <= 256:      return 'M'
    if 257  <= n <= 1024:     return 'L'
    if 1025 <= n <= 4096:     return 'XL'
    return None

def _assign_group_polaris(n):
    if 10  <= n <= 16:  return 'S'
    if 17  <= n <= 32:  return 'M'
    if 33  <= n <= 128: return 'L'
    if 129 <= n <= 496: return 'XL'
    return None


def _load_wait_hours_by_group(results_dir, assign_group_fn):
    """Return {tag: {group: [wait_hours]}} for all driver subfolders."""
    GROUPS = ['S', 'M', 'L', 'XL']
    procs_map = _infer_procs_map_for_results_dir(results_dir)
    result = {}
    if not os.path.isdir(results_dir):
        return result
    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        tag = entry.name
        events_path = os.path.join(results_dir, tag, "events.csv")
        if not os.path.exists(events_path):
            continue
        submit_map, start_map, _ = parse_events(events_path)
        by_group = {g: [] for g in GROUPS}
        for jid in set(submit_map) & set(start_map):
            w = (start_map[jid] - submit_map[jid]) / 3600.0
            if w <= 0:
                continue
            g = assign_group_fn(procs_map.get(jid, -1))
            if g is not None:
                by_group[g].append(w)
        result[tag] = by_group

    # RLScheduler — load from its CSV, group using the same procs_map
    rl_tag, rl_submit, rl_start, _ = _load_rlscheduler_entry(results_dir)
    if rl_tag and rl_submit and rl_start:
        by_group = {g: [] for g in GROUPS}
        for jid in set(rl_submit) & set(rl_start):
            w = (rl_start[jid] - rl_submit[jid]) / 3600.0
            if w <= 0:
                continue
            g = assign_group_fn(procs_map.get(jid, -1))
            if g is not None:
                by_group[g].append(w)
        result[rl_tag] = by_group
    return result


def _select_tags_by_prefixes(tags, tag_prefixes):
    ordered_tags = _order_all_tags(list(tags))
    selected = []
    seen = set()
    for prefix in tag_prefixes:
        prefix_up = prefix.upper()
        for tag in ordered_tags:
            if tag in seen:
                continue
            if tag.upper().startswith(prefix_up):
                selected.append(tag)
                seen.add(tag)
    return selected


def _wait_kde_shared_bounds(system_rows, groups, x_lo=1e-1, x_hi=1e4):
    x_grid = np.geomspace(x_lo, x_hi, 500)
    y_hi = 0.0
    for sys_data, tags in system_rows:
        for tag in tags:
            for group in groups:
                dens = _wait_kde_curve(sys_data.get(tag, {}).get(group, []), x_grid)
                if dens.size:
                    y_hi = max(y_hi, float(dens.max()))
    y_hi = max(y_hi * 1.08, 0.1)
    return x_grid, y_hi


def _wait_kde_log_y_bounds(system_rows, groups, x_grid):
    y_lo = None
    y_hi = 0.0
    for sys_data, tags in system_rows:
        for tag in tags:
            for group in groups:
                dens = _wait_kde_curve(sys_data.get(tag, {}).get(group, []), x_grid)
                pos = dens[dens > 0]
                if pos.size:
                    local_min = float(pos.min())
                    local_max = float(pos.max())
                    y_lo = local_min if y_lo is None else min(y_lo, local_min)
                    y_hi = max(y_hi, local_max)
    if y_lo is None:
        return 1e-4, 1.0
    y_lo = 10 ** np.floor(np.log10(y_lo))
    y_hi = 10 ** np.ceil(np.log10(max(y_hi, y_lo * 10.0)))
    return y_lo, y_hi


def _wait_cdf_curve(waits_h):
    arr = np.asarray(waits_h, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return None, None
    arr = np.sort(arr)
    cdf = np.arange(1, arr.size + 1, dtype=float) / arr.size
    return arr, cdf


def _wait_cdf_log_y_bounds(system_rows, groups):
    y_lo = None
    for sys_data, tags in system_rows:
        for tag in tags:
            for group in groups:
                xs, ys = _wait_cdf_curve(sys_data.get(tag, {}).get(group, []))
                if ys is None or ys.size == 0:
                    continue
                local_min = float(ys.min())
                y_lo = local_min if y_lo is None else min(y_lo, local_min)
    if y_lo is None:
        return 1e-4, 1.0
    y_lo = 10 ** np.floor(np.log10(y_lo))
    return y_lo, 1.0


def plot_combined_wait_kde_groups(theta_results_dir, polaris_results_dir, out_path,
                                  tag_prefixes=("WFP", "MARS-CW", "MARS-CU"),
                                  log_y=False):
    """2-row × 4-column wait-time KDE figure by size group for selected policies.

    Top row: Theta 2021
    Bottom row: Polaris 2024
    Columns: S, M, L, XL
    """
    GROUPS = ['S', 'M', 'L', 'XL']

    theta_by_group = _load_wait_hours_by_group(theta_results_dir, _assign_group_theta)
    polaris_by_group = _load_wait_hours_by_group(polaris_results_dir, _assign_group_polaris)

    theta_tags = _select_tags_by_prefixes(theta_by_group.keys(), tag_prefixes)
    polaris_tags = _select_tags_by_prefixes(polaris_by_group.keys(), tag_prefixes)
    all_tags = _select_tags_by_prefixes(list(dict.fromkeys(theta_tags + polaris_tags)), tag_prefixes)
    driver_colors = _build_driver_colors(all_tags)

    system_rows = [
        (theta_by_group, theta_tags),
        (polaris_by_group, polaris_tags),
    ]

    all_waits = [
        wait_h
        for sys_data, tags in system_rows
        for tag in tags
        for group in GROUPS
        for wait_h in sys_data.get(tag, {}).get(group, [])
        if wait_h > 0
    ]
    if not all_waits:
        print(f"  [skip] {os.path.basename(out_path)}: no positive wait-time data")
        return

    x_lo = 1e-1
    x_hi = 1e4
    x_grid, y_hi = _wait_kde_shared_bounds(system_rows, GROUPS, x_lo=x_lo, x_hi=x_hi)
    if log_y:
        y_lo, y_hi = _wait_kde_log_y_bounds(system_rows, GROUPS, x_grid)

    fig = plt.figure(figsize=(16, 4.75))
    gs = fig.add_gridspec(
        2, 4,
        left=0.07, right=0.997, top=0.81, bottom=0.12,
        wspace=0.07, hspace=0.20,
    )
    x_ticks = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    x_formatter = LogFormatterMathtext(base=10)

    first_ax = None
    for row_idx, (sys_data, tags) in enumerate(system_rows):
        for col_idx, group in enumerate(GROUPS):
            ax = fig.add_subplot(gs[row_idx, col_idx],
                                 **({"sharex": first_ax, "sharey": first_ax} if first_ax else {}))
            if first_ax is None:
                first_ax = ax

            for tag in tags:
                waits = sys_data.get(tag, {}).get(group, [])
                dens = _wait_kde_curve(waits, x_grid)
                if not np.any(dens > 0):
                    continue
                ax.plot(
                    x_grid,
                    dens,
                    color=driver_colors.get(tag, "gray"),
                    linewidth=2.8,
                    alpha=0.98,
                    solid_capstyle='round',
                )

            ax.set_xscale('log')
            ax.set_xlim(x_lo, x_hi)
            if log_y:
                ax.set_yscale('log')
                ax.set_ylim(y_lo, y_hi)
            else:
                ax.set_ylim(0.0, y_hi)
            ax.set_xticks(x_ticks)
            ax.xaxis.set_major_formatter(x_formatter)
            ax.grid(True, axis='y', alpha=0.28, linewidth=0.6)
            ax.grid(True, axis='x', alpha=0.14, linewidth=0.45)
            ax.tick_params(axis='both', labelsize=10)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight('bold')

            if row_idx == 0:
                ax.set_title(group, fontsize=13, fontweight='bold', pad=4)
            if col_idx != 0:
                ax.tick_params(axis='y', labelleft=False, left=False)
            if row_idx == 0:
                ax.tick_params(axis='x', labelbottom=False)

    fig.text(0.52, 0.045, "Wait Time (hours)", ha='center', va='center',
             fontsize=12.5, fontweight='bold')
    fig.text(0.017, 0.50, "PDF", rotation=90, va='center', ha='center',
             fontsize=12.5, fontweight='bold')

    legend_handles = [
        Line2D([0], [0], color=driver_colors[tag], linewidth=3.2, label=display_tag(tag))
        for tag in all_tags
    ]
    fig.legend(
        handles=legend_handles,
        labels=[display_tag(tag) for tag in all_tags],
        loc='upper center',
        bbox_to_anchor=(0.53, 0.985),
        ncol=3,
        frameon=False,
        fontsize=11.5,
        handlelength=2.7,
        columnspacing=1.2,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_single_system_wait_kde_groups(results_dir, out_path, assign_group_fn,
                                       tag_prefixes=("WFP", "MARS-CW", "MARS-CU"),
                                       fig_width=8.0, log_y=False):
    """2x2 KDE figure by size group for one system."""
    GROUPS = ['S', 'M', 'L', 'XL']
    by_group = _load_wait_hours_by_group(results_dir, assign_group_fn)
    tags = _select_tags_by_prefixes(by_group.keys(), tag_prefixes)
    if not tags:
        print(f"  [skip] {os.path.basename(out_path)}: no matching tags")
        return

    all_waits = [
        wait_h
        for tag in tags
        for group in GROUPS
        for wait_h in by_group.get(tag, {}).get(group, [])
        if wait_h > 0
    ]
    if not all_waits:
        print(f"  [skip] {os.path.basename(out_path)}: no positive wait-time data")
        return

    driver_colors = _build_driver_colors(tags)
    x_lo = 1e-1
    x_hi = 1e4
    x_grid, y_hi = _wait_kde_shared_bounds([(by_group, tags)], GROUPS, x_lo=x_lo, x_hi=x_hi)
    if log_y:
        y_lo, y_hi = _wait_kde_log_y_bounds([(by_group, tags)], GROUPS, x_grid)
    x_ticks = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    x_formatter = LogFormatterMathtext(base=10)

    fig = plt.figure(figsize=(fig_width, 5.2))
    gs = fig.add_gridspec(
        2, 2,
        left=0.09, right=0.995, top=0.82, bottom=0.12,
        wspace=0.12, hspace=0.22,
    )

    first_ax = None
    for idx, group in enumerate(GROUPS):
        row_idx, col_idx = divmod(idx, 2)
        ax = fig.add_subplot(gs[row_idx, col_idx],
                             **({"sharex": first_ax, "sharey": first_ax} if first_ax else {}))
        if first_ax is None:
            first_ax = ax

        for tag in tags:
            dens = _wait_kde_curve(by_group.get(tag, {}).get(group, []), x_grid)
            if not np.any(dens > 0):
                continue
            ax.plot(
                x_grid,
                dens,
                color=driver_colors.get(tag, "gray"),
                linewidth=3.0,
                alpha=0.98,
                solid_capstyle='round',
            )

        ax.set_xscale('log')
        ax.set_xlim(x_lo, x_hi)
        if log_y:
            ax.set_yscale('log')
            ax.set_ylim(y_lo, y_hi)
        else:
            ax.set_ylim(0.0, y_hi)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(x_formatter)
        ax.grid(True, axis='y', alpha=0.28, linewidth=0.6)
        ax.grid(True, axis='x', alpha=0.14, linewidth=0.45)
        ax.tick_params(axis='both', labelsize=10)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')

        ax.set_title(group, fontsize=13, fontweight='bold', pad=4)
        if col_idx != 0:
            ax.tick_params(axis='y', labelleft=False, left=False)
        if row_idx == 0:
            ax.tick_params(axis='x', labelbottom=False)

    fig.text(0.52, 0.045, "Wait Time (hours)", ha='center', va='center',
             fontsize=12.5, fontweight='bold')
    fig.text(0.024, 0.50, "PDF", rotation=90, va='center', ha='center',
             fontsize=12.5, fontweight='bold')

    legend_handles = [
        Line2D([0], [0], color=driver_colors[tag], linewidth=3.4, label=display_tag(tag))
        for tag in tags
    ]
    fig.legend(
        handles=legend_handles,
        labels=[display_tag(tag) for tag in tags],
        loc='upper center',
        bbox_to_anchor=(0.53, 0.985),
        ncol=max(1, len(tags)),
        frameon=False,
        fontsize=11.5,
        handlelength=2.7,
        columnspacing=1.2,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_wait_cdf_groups(theta_results_dir, polaris_results_dir, out_path,
                                  tag_prefixes=("WFP", "MARS-CW", "MARS-CU"),
                                  log_y=False):
    """2-row × 4-column wait-time CDF figure by size group for selected policies."""
    GROUPS = ['S', 'M', 'L', 'XL']

    theta_by_group = _load_wait_hours_by_group(theta_results_dir, _assign_group_theta)
    polaris_by_group = _load_wait_hours_by_group(polaris_results_dir, _assign_group_polaris)

    theta_tags = _select_tags_by_prefixes(theta_by_group.keys(), tag_prefixes)
    polaris_tags = _select_tags_by_prefixes(polaris_by_group.keys(), tag_prefixes)
    all_tags = _select_tags_by_prefixes(list(dict.fromkeys(theta_tags + polaris_tags)), tag_prefixes)
    driver_colors = _build_driver_colors(all_tags)

    system_rows = [
        (theta_by_group, theta_tags),
        (polaris_by_group, polaris_tags),
    ]

    all_waits = [
        wait_h
        for sys_data, tags in system_rows
        for tag in tags
        for group in GROUPS
        for wait_h in sys_data.get(tag, {}).get(group, [])
        if wait_h > 0
    ]
    if not all_waits:
        print(f"  [skip] {os.path.basename(out_path)}: no positive wait-time data")
        return

    x_lo = 1e-1
    x_hi = 1e4
    if log_y:
        y_lo, y_hi = _wait_cdf_log_y_bounds(system_rows, GROUPS)
    fig = plt.figure(figsize=(16, 4.75))
    gs = fig.add_gridspec(
        2, 4,
        left=0.07, right=0.997, top=0.81, bottom=0.12,
        wspace=0.07, hspace=0.20,
    )
    x_ticks = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    x_formatter = LogFormatterMathtext(base=10)

    first_ax = None
    for row_idx, (sys_data, tags) in enumerate(system_rows):
        for col_idx, group in enumerate(GROUPS):
            ax = fig.add_subplot(gs[row_idx, col_idx],
                                 **({"sharex": first_ax, "sharey": first_ax} if first_ax else {}))
            if first_ax is None:
                first_ax = ax

            for tag in tags:
                xs, ys = _wait_cdf_curve(sys_data.get(tag, {}).get(group, []))
                if xs is None:
                    continue
                ax.plot(
                    xs,
                    ys,
                    color=driver_colors.get(tag, "gray"),
                    linewidth=2.8,
                    alpha=0.98,
                    solid_capstyle='round',
                )

            ax.set_xscale('log')
            ax.set_xlim(x_lo, x_hi)
            if log_y:
                ax.set_yscale('log')
                ax.set_ylim(y_lo, y_hi)
            else:
                ax.set_ylim(0.0, 1.0)
            ax.set_xticks(x_ticks)
            ax.xaxis.set_major_formatter(x_formatter)
            ax.grid(True, axis='y', alpha=0.28, linewidth=0.6)
            ax.grid(True, axis='x', alpha=0.14, linewidth=0.45)
            ax.tick_params(axis='both', labelsize=10)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight('bold')

            if row_idx == 0:
                ax.set_title(group, fontsize=13, fontweight='bold', pad=4)
            if col_idx != 0:
                ax.tick_params(axis='y', labelleft=False, left=False)
            if row_idx == 0:
                ax.tick_params(axis='x', labelbottom=False)

    fig.text(0.52, 0.045, "Wait Time (hours)", ha='center', va='center',
             fontsize=12.5, fontweight='bold')
    fig.text(0.017, 0.50, "CDF", rotation=90, va='center', ha='center',
             fontsize=12.5, fontweight='bold')

    legend_handles = [
        Line2D([0], [0], color=driver_colors[tag], linewidth=3.2, label=display_tag(tag))
        for tag in all_tags
    ]
    fig.legend(
        handles=legend_handles,
        labels=[display_tag(tag) for tag in all_tags],
        loc='upper center',
        bbox_to_anchor=(0.53, 0.985),
        ncol=3,
        frameon=False,
        fontsize=11.5,
        handlelength=2.7,
        columnspacing=1.2,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_single_system_wait_cdf_groups(results_dir, out_path, assign_group_fn,
                                       tag_prefixes=("WFP", "MARS-CW", "MARS-CU"),
                                       fig_width=8.0, log_y=False):
    """2x2 CDF figure by size group for one system."""
    GROUPS = ['S', 'M', 'L', 'XL']
    by_group = _load_wait_hours_by_group(results_dir, assign_group_fn)
    tags = _select_tags_by_prefixes(by_group.keys(), tag_prefixes)
    if not tags:
        print(f"  [skip] {os.path.basename(out_path)}: no matching tags")
        return

    all_waits = [
        wait_h
        for tag in tags
        for group in GROUPS
        for wait_h in by_group.get(tag, {}).get(group, [])
        if wait_h > 0
    ]
    if not all_waits:
        print(f"  [skip] {os.path.basename(out_path)}: no positive wait-time data")
        return

    driver_colors = _build_driver_colors(tags)
    x_lo = 1e-1
    x_hi = 1e4
    if log_y:
        y_lo, y_hi = _wait_cdf_log_y_bounds([(by_group, tags)], GROUPS)
    x_ticks = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    x_formatter = LogFormatterMathtext(base=10)

    fig = plt.figure(figsize=(fig_width, 5.2))
    gs = fig.add_gridspec(
        2, 2,
        left=0.09, right=0.995, top=0.82, bottom=0.12,
        wspace=0.12, hspace=0.22,
    )

    first_ax = None
    for idx, group in enumerate(GROUPS):
        row_idx, col_idx = divmod(idx, 2)
        ax = fig.add_subplot(gs[row_idx, col_idx],
                             **({"sharex": first_ax, "sharey": first_ax} if first_ax else {}))
        if first_ax is None:
            first_ax = ax

        for tag in tags:
            xs, ys = _wait_cdf_curve(by_group.get(tag, {}).get(group, []))
            if xs is None:
                continue
            ax.plot(
                xs,
                ys,
                color=driver_colors.get(tag, "gray"),
                linewidth=3.0,
                alpha=0.98,
                solid_capstyle='round',
            )

        ax.set_xscale('log')
        ax.set_xlim(x_lo, x_hi)
        if log_y:
            ax.set_yscale('log')
            ax.set_ylim(y_lo, y_hi)
        else:
            ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(x_formatter)
        ax.grid(True, axis='y', alpha=0.28, linewidth=0.6)
        ax.grid(True, axis='x', alpha=0.14, linewidth=0.45)
        ax.tick_params(axis='both', labelsize=10)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')

        ax.set_title(group, fontsize=13, fontweight='bold', pad=4)
        if col_idx != 0:
            ax.tick_params(axis='y', labelleft=False, left=False)
        if row_idx == 0:
            ax.tick_params(axis='x', labelbottom=False)

    fig.text(0.52, 0.045, "Wait Time (hours)", ha='center', va='center',
             fontsize=12.5, fontweight='bold')
    fig.text(0.024, 0.50, "CDF", rotation=90, va='center', ha='center',
             fontsize=12.5, fontweight='bold')

    legend_handles = [
        Line2D([0], [0], color=driver_colors[tag], linewidth=3.4, label=display_tag(tag))
        for tag in tags
    ]
    fig.legend(
        handles=legend_handles,
        labels=[display_tag(tag) for tag in tags],
        loc='upper center',
        bbox_to_anchor=(0.53, 0.985),
        ncol=max(1, len(tags)),
        frameon=False,
        fontsize=11.5,
        handlelength=2.7,
        columnspacing=1.2,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_wait_boxplot_all(theta_results_dir, polaris_results_dir, out_path):
    """Box plots of raw wait times for ALL strategies, Theta left / Polaris right."""
    theta_data   = _load_wait_hours_for_tags(theta_results_dir)
    polaris_data = _load_wait_hours_for_tags(polaris_results_dir)

    theta_tags   = _order_all_tags(list(theta_data.keys()))
    polaris_tags = _order_all_tags(list(polaris_data.keys()))

    all_tags = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = _build_driver_colors(all_tags)

    n_tags = max(len(theta_tags), len(polaris_tags))
    fig_w = max(7.0, n_tags * 0.72 + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 4.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.22, wspace=0.06)

    _draw_wait_boxplot_panel(axes[0], theta_tags,   theta_data,   driver_colors,
                             "Theta 2021",   show_ylabel=True)
    _draw_wait_boxplot_panel(axes[1], polaris_tags, polaris_data, driver_colors,
                             "Polaris 2024", show_ylabel=False)

    axes[0].set_ylabel("Wait Time (hours)", fontsize=13, fontweight='bold', labelpad=2)
    all_vals = [v for d in (theta_data, polaris_data) for vals in d.values() for v in vals]
    if all_vals:
        ymin = max(0.01, np.percentile(all_vals, 1) * 0.5)
        ymax = max(np.percentile(all_vals, 99) * 2.0, 1000.0)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_wait_boxplot_groups(theta_results_dir, polaris_results_dir, out_path):
    """2-row × 4-column box plots of raw wait times broken down by job-size group.

    Row 0: Theta 2021  — S, M, L, XL
    Row 1: Polaris 2024 — S, M, L, XL
    """
    GROUPS = ['S', 'M', 'L', 'XL']

    theta_by_group   = _load_wait_hours_by_group(theta_results_dir,   _assign_group_theta)
    polaris_by_group = _load_wait_hours_by_group(polaris_results_dir, _assign_group_polaris)

    theta_tags   = _order_all_tags(list(theta_by_group.keys()))
    polaris_tags = _order_all_tags(list(polaris_by_group.keys()))
    all_tags = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = _build_driver_colors(all_tags)

    n_tags = max(len(theta_tags), len(polaris_tags))
    panel_w = max(2.8, n_tags * 0.62 + 1.2)
    fig = plt.figure(figsize=(panel_w * 4 + 0.6, 7.2))
    gs = fig.add_gridspec(2, 4, wspace=0.07, hspace=0.32,
                          left=0.08, right=0.995, top=0.86, bottom=0.13)

    system_data = [(theta_by_group,   theta_tags,   "Theta 2021",   _assign_group_theta),
                   (polaris_by_group, polaris_tags, "Polaris 2024", _assign_group_polaris)]

    first_ax = None
    for row_idx, (by_group, tags, sys_name, _) in enumerate(system_data):
        for col_idx, group in enumerate(GROUPS):
            ax = fig.add_subplot(gs[row_idx, col_idx],
                                 **({"sharey": first_ax} if first_ax else {}))
            if first_ax is None:
                first_ax = ax

            group_data = {t: by_group.get(t, {}).get(group, []) for t in tags}
            _draw_wait_boxplot_panel(
                ax, tags, group_data, driver_colors,
                title=group if row_idx == 0 else "",
                show_ylabel=(col_idx == 0),
            )
            if col_idx == 0:
                ax.set_ylabel(sys_name, fontsize=13, fontweight='bold', labelpad=4)

    # Unified y-range
    all_vals = [v for d in (theta_by_group, polaris_by_group)
                for gd in d.values() for vals in gd.values() for v in vals]
    if all_vals:
        ymin = max(0.01, np.percentile(all_vals, 1) * 0.5)
        ymax = max(np.percentile(all_vals, 99) * 2.0, 1000.0)
        for ax in fig.axes:
            ax.set_ylim(ymin, ymax)

    fig.text(0.018, 0.50, "Wait Time (hours)", rotation=90,
             va="center", ha="center", fontsize=13, fontweight="bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _load_bsld_for_tags(results_dir, tag_prefixes=None):
    """Return {tag: [bsld > 1]} using actual runtime (end-start) from events.csv."""
    data = {}
    if not os.path.isdir(results_dir):
        return data
    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        tag = entry.name
        tag_up = tag.upper()
        if tag_prefixes is not None and not any(
                tag_up.startswith(p.upper()) for p in tag_prefixes):
            continue
        events_path = os.path.join(results_dir, tag, "events.csv")
        if not os.path.exists(events_path):
            continue
        submit_map, start_map, end_map = parse_events(events_path)
        bslds = []
        for jid in set(submit_map) & set(start_map) & set(end_map):
            wait = start_map[jid] - submit_map[jid]
            runtime = max(end_map[jid] - start_map[jid], 1)
            if wait <= 0:
                continue
            bslds.append((wait + runtime) / runtime)
        data[tag] = bslds

    rl_tag, rl_submit, rl_start, rl_end = _load_rlscheduler_entry(results_dir)
    if rl_tag and rl_submit and rl_start and rl_end:
        if tag_prefixes is None or any(
                rl_tag.upper().startswith(p.upper()) for p in tag_prefixes):
            bslds = []
            for jid in set(rl_submit) & set(rl_start) & set(rl_end):
                wait = rl_start[jid] - rl_submit[jid]
                runtime = max(rl_end[jid] - rl_start[jid], 1)
                if wait <= 0:
                    continue
                bslds.append((wait + runtime) / runtime)
            data[rl_tag] = bslds
    return data


def _load_bsld_by_group(results_dir, assign_group_fn):
    """Return {tag: {group: [bsld]}} for all driver subfolders."""
    GROUPS = ['S', 'M', 'L', 'XL']
    procs_map = _infer_procs_map_for_results_dir(results_dir)
    result = {}
    if not os.path.isdir(results_dir):
        return result
    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        tag = entry.name
        events_path = os.path.join(results_dir, tag, "events.csv")
        if not os.path.exists(events_path):
            continue
        submit_map, start_map, end_map = parse_events(events_path)
        by_group = {g: [] for g in GROUPS}
        for jid in set(submit_map) & set(start_map) & set(end_map):
            wait = start_map[jid] - submit_map[jid]
            runtime = max(end_map[jid] - start_map[jid], 1)
            if wait <= 0:
                continue
            g = assign_group_fn(procs_map.get(jid, -1))
            if g is not None:
                by_group[g].append((wait + runtime) / runtime)
        result[tag] = by_group

    rl_tag, rl_submit, rl_start, rl_end = _load_rlscheduler_entry(results_dir)
    if rl_tag and rl_submit and rl_start and rl_end:
        by_group = {g: [] for g in GROUPS}
        for jid in set(rl_submit) & set(rl_start) & set(rl_end):
            wait = rl_start[jid] - rl_submit[jid]
            runtime = max(rl_end[jid] - rl_start[jid], 1)
            if wait <= 0:
                continue
            g = assign_group_fn(procs_map.get(jid, -1))
            if g is not None:
                by_group[g].append((wait + runtime) / runtime)
        result[rl_tag] = by_group
    return result


def plot_combined_bsld_boxplot(theta_results_dir, polaris_results_dir, out_path,
                               tag_prefixes=("WFP", "MARS-CU")):
    """Box plots of BSLD for WFP / MARS-CW / MARS-CU (single-column figure)."""
    theta_data   = _load_bsld_for_tags(theta_results_dir,   tag_prefixes)
    polaris_data = _load_bsld_for_tags(polaris_results_dir, tag_prefixes)

    def _order(data):
        WFP    = sorted(t for t in data if t.upper().startswith('WFP'))
        mars_cw = sorted(t for t in data if t.upper().startswith('MARS-CW'))
        mars_cu = sorted(t for t in data if t.upper().startswith('MARS-CU'))
        return WFP + mars_cw + mars_cu

    theta_tags   = _order(theta_data)
    polaris_tags = _order(polaris_data)
    all_tags = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = _build_driver_colors(all_tags)

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 4.2))
    fig.subplots_adjust(left=0.21, right=0.97, top=0.88, bottom=0.20, wspace=0.08)

    _draw_wait_boxplot_panel(axes[0], theta_tags,   theta_data,   driver_colors,
                             "Theta 2021",   show_ylabel=True)
    _draw_wait_boxplot_panel(axes[1], polaris_tags, polaris_data, driver_colors,
                             "Polaris 2024", show_ylabel=False)

    axes[0].set_ylabel("Bounded Slowdown", fontsize=13, fontweight='bold', labelpad=2)

    all_vals = [v for d in (theta_data, polaris_data) for vals in d.values() for v in vals if v > 1]
    if all_vals:
        ymin = 1.0
        ymax = max(np.percentile(all_vals, 99) * 2.0, 100.0)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_bsld_boxplot_all(theta_results_dir, polaris_results_dir, out_path):
    """Box plots of BSLD for ALL strategies, Theta left / Polaris right."""
    theta_data   = _load_bsld_for_tags(theta_results_dir)
    polaris_data = _load_bsld_for_tags(polaris_results_dir)

    theta_tags   = _order_all_tags(list(theta_data.keys()))
    polaris_tags = _order_all_tags(list(polaris_data.keys()))
    all_tags = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = _build_driver_colors(all_tags)

    n_tags = max(len(theta_tags), len(polaris_tags))
    fig_w = max(7.0, n_tags * 0.72 + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 4.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.22, wspace=0.06)

    _draw_wait_boxplot_panel(axes[0], theta_tags,   theta_data,   driver_colors,
                             "Theta 2021",   show_ylabel=True)
    _draw_wait_boxplot_panel(axes[1], polaris_tags, polaris_data, driver_colors,
                             "Polaris 2024", show_ylabel=False)

    axes[0].set_ylabel("Bounded Slowdown", fontsize=13, fontweight='bold', labelpad=2)
    all_vals = [v for d in (theta_data, polaris_data) for vals in d.values() for v in vals]
    if all_vals:
        ymin = 1.0
        ymax = max(np.percentile(all_vals, 99) * 2.0, 100.0)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_bsld_boxplot_groups(theta_results_dir, polaris_results_dir, out_path):
    """2-row × 4-column box plots of BSLD broken down by job-size group."""
    GROUPS = ['S', 'M', 'L', 'XL']

    theta_by_group   = _load_bsld_by_group(theta_results_dir,   _assign_group_theta)
    polaris_by_group = _load_bsld_by_group(polaris_results_dir, _assign_group_polaris)

    theta_tags   = _order_all_tags(list(theta_by_group.keys()))
    polaris_tags = _order_all_tags(list(polaris_by_group.keys()))
    all_tags = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = _build_driver_colors(all_tags)

    n_tags = max(len(theta_tags), len(polaris_tags))
    panel_w = max(2.8, n_tags * 0.62 + 1.2)
    fig = plt.figure(figsize=(panel_w * 4 + 0.6, 7.2))
    gs = fig.add_gridspec(2, 4, wspace=0.07, hspace=0.32,
                          left=0.08, right=0.995, top=0.86, bottom=0.13)

    system_data = [(theta_by_group,   theta_tags,   "Theta 2021",   _assign_group_theta),
                   (polaris_by_group, polaris_tags, "Polaris 2024", _assign_group_polaris)]

    first_ax = None
    for row_idx, (by_group, tags, sys_name, _) in enumerate(system_data):
        for col_idx, group in enumerate(GROUPS):
            ax = fig.add_subplot(gs[row_idx, col_idx],
                                 **({"sharey": first_ax} if first_ax else {}))
            if first_ax is None:
                first_ax = ax

            group_data = {t: by_group.get(t, {}).get(group, []) for t in tags}
            _draw_wait_boxplot_panel(
                ax, tags, group_data, driver_colors,
                title=group if row_idx == 0 else "",
                show_ylabel=(col_idx == 0),
            )
            if col_idx == 0:
                ax.set_ylabel(sys_name, fontsize=13, fontweight='bold', labelpad=4)

    all_vals = [v for d in (theta_by_group, polaris_by_group)
                for gd in d.values() for vals in gd.values() for v in vals]
    if all_vals:
        ymin = 1.0
        ymax = max(np.percentile(all_vals, 99) * 2.0, 100.0)
        for ax in fig.axes:
            ax.set_ylim(ymin, ymax)

    fig.text(0.018, 0.50, "Bounded Slowdown", rotation=90,
             va="center", ha="center", fontsize=13, fontweight="bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Tail-wait job-size CDF ────────────────────────────────────────────────────

_THETA_ZONE_X = {
    'S':  (126,  130),
    'M':  (129,  256),
    'L':  (257, 1024),
    'XL': (1025, 4096),
}
_POLARIS_ZONE_X = {
    'S':  (10,  16),
    'M':  (17,  32),
    'L':  (33, 128),
    'XL': (129, 496),
}
_ZONE_COLORS = {'S': 'steelblue', 'M': 'darkorange', 'L': 'seagreen', 'XL': 'crimson'}
_ZONE_DARK   = {'S': '#1a527a',   'M': '#994d00',    'L': '#1a6640',   'XL': '#8b0000'}


def _load_wait_procs_for_tags(results_dir):
    """Return {tag: [(wait_h, procs)]} for all jobs with wait > 0."""
    procs_map = _infer_procs_map_for_results_dir(results_dir)
    data = {}
    if not os.path.isdir(results_dir):
        return data
    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        tag = entry.name
        events_path = os.path.join(results_dir, tag, "events.csv")
        if not os.path.exists(events_path):
            continue
        submit_map, start_map, _ = parse_events(events_path)
        pairs = []
        for jid in set(submit_map) & set(start_map):
            w = (start_map[jid] - submit_map[jid]) / 3600.0
            if w <= 0:
                continue
            p = procs_map.get(jid)
            if p is not None and p > 0:
                pairs.append((w, int(p)))
        data[tag] = pairs

    rl_tag, rl_submit, rl_start, _ = _load_rlscheduler_entry(results_dir)
    if rl_tag and rl_submit and rl_start:
        pairs = []
        for jid in set(rl_submit) & set(rl_start):
            w = (rl_start[jid] - rl_submit[jid]) / 3600.0
            if w <= 0:
                continue
            p = procs_map.get(jid)
            if p is not None and p > 0:
                pairs.append((w, int(p)))
        data[rl_tag] = pairs
    return data


def _draw_tail_wait_size_cdf_panel(ax, ordered_tags, wait_procs_per_tag, driver_colors,
                                    zone_x, wfp3_thresh_h, title, xlim, tail_pctile):
    GROUPS = ['S', 'M', 'L', 'XL']

    # Shaded bands + group labels
    for g in GROUPS:
        x0, x1 = zone_x[g]
        ax.axvspan(x0, x1, alpha=0.13, color=_ZONE_COLORS[g], zorder=0, linewidth=0)
        x_ctr = np.exp(0.5 * (np.log(max(x0, 0.5)) + np.log(max(x1, x0 + 0.1))))
        ax.text(x_ctr, 0.97, g,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top',
                fontsize=12, fontweight='bold',
                color=_ZONE_DARK[g])

    # Empirical CDF of procs for tail jobs
    for tag in ordered_tags:
        pairs = wait_procs_per_tag.get(tag, [])
        tail_procs = sorted(int(p) for w, p in pairs if w > wfp3_thresh_h)
        n = len(tail_procs)
        if n < 2:
            continue
        cdf_y = np.arange(1, n + 1) / n
        ax.step(tail_procs, cdf_y, where='post',
                color=driver_colors[tag],
                linewidth=2.1,
                alpha=0.88,
                label=f"{display_tag(tag)}  (n={n})")

    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.09)
    ax.set_xlabel("Job Size (nodes)", fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=5)
    ax.tick_params(axis='both', labelsize=11)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.grid(True, axis='both', which='major', alpha=0.3, zorder=0)
    ax.axhline(1.0, color='gray', linewidth=0.8, linestyle=':', zorder=1)
    ax.text(0.98, 0.03, f"WFP P{tail_pctile} = {wfp3_thresh_h:.1f} h",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='gray', style='italic')


def plot_combined_tail_wait_job_size_cdf(theta_results_dir, polaris_results_dir, out_path,
                                          tail_pctile=95,
                                          excluded_tags=frozenset()):
    """CDF of job sizes among tail-wait jobs (wait > WFP P95) for all strategies.
    Left panel = Theta 2021, right panel = Polaris 2024.
    X-axis: job size (nodes, log scale) with shaded S/M/L/XL zones.
    """
    excluded_up = {t.upper() for t in excluded_tags}

    def _is_excluded(tag):
        tag_up = tag.upper()
        return any(tag_up.startswith(e) for e in excluded_up)

    theta_data   = _load_wait_procs_for_tags(theta_results_dir)
    polaris_data = _load_wait_procs_for_tags(polaris_results_dir)

    def _wfp3_thresh(data):
        wfp3_tag = next((t for t in data if t.upper().startswith('WFP')), None)
        if wfp3_tag is None:
            return 0.0
        waits = [w for w, _ in data[wfp3_tag]]
        return float(np.percentile(waits, tail_pctile)) if waits else 0.0

    theta_thresh   = _wfp3_thresh(theta_data)
    polaris_thresh = _wfp3_thresh(polaris_data)

    theta_tags   = _order_all_tags([t for t in theta_data   if not _is_excluded(t)])
    polaris_tags = _order_all_tags([t for t in polaris_data if not _is_excluded(t)])
    all_tags = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = _build_driver_colors(all_tags)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.87, bottom=0.36, wspace=0.18)

    _draw_tail_wait_size_cdf_panel(
        axes[0], theta_tags, theta_data, driver_colors,
        _THETA_ZONE_X, theta_thresh, "Theta 2021", (100, 5000), tail_pctile,
    )
    _draw_tail_wait_size_cdf_panel(
        axes[1], polaris_tags, polaris_data, driver_colors,
        _POLARIS_ZONE_X, polaris_thresh, "Polaris 2024", (8, 600), tail_pctile,
    )

    axes[0].set_ylabel("Cumulative Fraction of Tail Jobs", fontsize=13, fontweight='bold')

    # Shared legend below figure — merge handles, Theta first then any Polaris-only
    h_t, l_t = axes[0].get_legend_handles_labels()
    h_p, l_p = axes[1].get_legend_handles_labels()
    seen_keys, all_h, all_l = set(), [], []
    for h, lbl in list(zip(h_t, l_t)) + list(zip(h_p, l_p)):
        key = lbl.split('(')[0].strip()
        if key not in seen_keys:
            seen_keys.add(key)
            all_h.append(h)
            all_l.append(lbl)
    n_cols = min(len(all_l), 5)
    fig.legend(all_h, all_l,
               loc='lower center', ncol=n_cols,
               fontsize=9.5, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.02),
               title=f"Strategy  (n = jobs with wait > WFP P{tail_pctile})",
               title_fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_metric_groups(theta_plots_dir, polaris_plots_dir,
                                metric, metric_label, out_path):
    """2-row × 4-column combined figure for one metric (wait or bsld).

    Row 0: Theta 2021  — columns S, M, L, XL
    Row 1: Polaris 2024 — columns S, M, L, XL
    Designed to span two columns on a paper.
    """
    GROUPS = ['S', 'M', 'L', 'XL']

    def _load(plots_dir, group):
        p = os.path.join(plots_dir, metric, f"table_{metric}_{group}.csv")
        return _load_table(p) if os.path.exists(p) else {}

    theta_tables   = {g: _load(theta_plots_dir,   g) for g in GROUPS}
    polaris_tables = {g: _load(polaris_plots_dir,  g) for g in GROUPS}

    ref_rows = next((t for t in theta_tables.values() if t), {})
    all_driver_tags = list(ref_rows.keys())
    driver_colors = _build_driver_colors(all_driver_tags)

    fig = plt.figure(figsize=(14, 6.8))
    gs = fig.add_gridspec(
        2, 4,
        wspace=0.07, hspace=0.38,
        left=0.075, right=0.995, top=0.82, bottom=0.13,
    )

    # rows: 0 = Theta, 1 = Polaris
    system_tables = [theta_tables, polaris_tables]
    system_names  = ["Theta 2021", "Polaris 2024"]

    handles_out, labels_out = None, None
    first_ax = None

    for row_idx in range(2):
        for col_idx, group in enumerate(GROUPS):
            ax = fig.add_subplot(gs[row_idx, col_idx],
                                 **({"sharey": first_ax} if first_ax else {}))
            if first_ax is None:
                first_ax = ax

            table_rows = system_tables[row_idx].get(group, {})
            if table_rows:
                other_tags, improvements = _compute_improvements(table_rows)
                n_tags = len(other_tags)
                x = _metric_x_positions()
                width = 1.10 / max(n_tags, 1)
                for i, tag in enumerate(other_tags):
                    offsets = x + (i - (n_tags - 1) / 2.0) * width
                    hatch = driver_hatch(tag)
                    bars = ax.bar(
                        offsets, improvements[tag],
                        width=width * 0.95,
                        color=driver_colors.get(tag, "gray"),
                        alpha=0.85,
                        hatch=hatch,
                        edgecolor="black" if hatch else None,
                    )
                    if row_idx == 0 and col_idx == 0:
                        if handles_out is None:
                            handles_out, labels_out = [], []
                        handles_out.append(bars[0])
                        labels_out.append(display_tag(tag))

            # Axis styling
            ax.set_ylim(-100, 100)
            ax.set_yticks(Y_TICKS)
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.0f}")
            )
            _add_metric_backgrounds(ax, _metric_x_positions())
            _add_horizontal_guides(ax, (-100, 100), Y_TICKS, step=25)
            ax.axhline(0, color="black", linewidth=0.9, zorder=1)
            ax.set_xticks(_metric_x_positions())
            if row_idx == 1:
                xlbls = ax.set_xticklabels(BAR_STAT_HEADERS, fontsize=9,
                                           rotation=30, ha="right")
                for lbl in xlbls:
                    lbl.set_fontweight("bold")
            else:
                ax.tick_params(axis="x", labelbottom=False, bottom=False)
            ax.set_xlim(_metric_x_positions()[0] - 0.72,
                        _metric_x_positions()[-1] + 0.72)
            if col_idx == 0:
                ax.tick_params(axis="y", labelsize=9.5)
                for lbl in ax.get_yticklabels():
                    lbl.set_fontweight("bold")
            else:
                ax.tick_params(axis="y", labelleft=False, left=False)

            # Group label (top row only) and system label (leftmost col only)
            if row_idx == 0:
                ax.set_title(group, fontsize=12, fontweight="bold", pad=4)
            if col_idx == 0:
                ax.set_ylabel(system_names[row_idx], fontsize=11,
                              fontweight="bold", labelpad=6)

    # Shared y-label
    fig.text(0.018, 0.50, f"% Improv. vs WFP",
             rotation=90, va="center", ha="center",
             fontsize=11, fontweight="bold")

    # Legend above
    if handles_out:
        legend = fig.legend(
            handles_out, labels_out,
            loc="upper center",
            bbox_to_anchor=(0.535, 0.975),
            ncol=len(labels_out),
            frameon=False,
            fontsize=11,
            columnspacing=1.0,
            handletextpad=0.45,
        )
        for txt in legend.get_texts():
            txt.set_fontweight("bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_overall(theta_table, polaris_table, out_path, metric_label):
    theta_rows = _load_table(theta_table)
    polaris_rows = _load_table(polaris_table)

    # Keep legend/order consistent across both panels.
    all_driver_tags = list(theta_rows.keys())
    driver_colors = _build_driver_colors(all_driver_tags)

    fig, axes = plt.subplots(1, 2, figsize=(8.15, 4.15), sharey=True)

    theta_tags, theta_imp = _compute_improvements(theta_rows)
    handles, labels = _plot_on_ax(axes[0], theta_tags, theta_imp, driver_colors)
    _style_ax(axes[0], "Theta 2021")
    axes[0].set_ylabel("% Improv. w.r.t WFP", fontsize=11)

    polaris_tags, polaris_imp = _compute_improvements(polaris_rows)
    _plot_on_ax(axes[1], polaris_tags, polaris_imp, driver_colors)
    _style_ax(axes[1], "Polaris 2024")

    _add_centered_two_row_legend(
        fig, handles, labels, fontsize=9.25,
        y_top=0.865, row_gap=0.04,
        columnspacing=0.9, handletextpad=0.4,
    )
    fig.supxlabel(metric_label, fontsize=11, y=0.06)
    fig.subplots_adjust(top=0.66, bottom=0.21, left=0.09, right=0.99, wspace=0.16)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _load_drain_overview_data(results_dir, drain_plots_dir, drain_tags, assign_group_fn):
    """Return (drain_pct_by_tag, postdrain_group_by_tag, total_group_by_tag).

    drain_plots_dir : the drain/ subdirectory of the per-system plots dir.
    """
    from exp2_plot import _load_drain_pct_for_tag, tag_slug

    GROUPS   = ['S', 'M', 'L', 'XL']
    procs_map = _infer_procs_map_for_results_dir(results_dir)

    drain_pct_by_tag   = {}
    total_group_by_tag = {}

    for tag in drain_tags:
        drain_pct_by_tag[tag] = _load_drain_pct_for_tag(results_dir, tag)

        events_path = os.path.join(results_dir, tag, 'events.csv')
        totals = {g: 0 for g in GROUPS}
        if os.path.exists(events_path):
            _, start, _ = parse_events(events_path)
            for jid in start:
                g = assign_group_fn(procs_map.get(jid, -1))
                if g in totals:
                    totals[g] += 1
        total_group_by_tag[tag] = totals

    postdrain_group_by_tag = {tag: {g: 0 for g in GROUPS} for tag in drain_tags}
    for tag in drain_tags:
        slug     = tag_slug(tag)
        csv_path = os.path.join(drain_plots_dir,
                                f'table_{slug}_drain_run_sequence_details.csv')
        if not os.path.exists(csv_path):
            continue
        seen = set()
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                for pos in ('First', 'Second', 'Third'):
                    jid_str = row.get(f'{pos}_Run_Job', '').strip()
                    grp     = row.get(f'{pos}_Run_Group', '').strip()
                    if not jid_str or grp not in GROUPS:
                        continue
                    try:
                        jid = int(jid_str)
                    except ValueError:
                        continue
                    if (jid, grp) in seen:
                        continue
                    seen.add((jid, grp))
                    postdrain_group_by_tag[tag][grp] += 1

    return drain_pct_by_tag, postdrain_group_by_tag, total_group_by_tag


def _load_drain_pct_collapsed(results_dir, drain_plots_dir, tag):
    """Return (drain_ep, eligible, pct) using collapsed drain episodes.

    Collapsed = consecutive DRAIN cycles with no jobs starting in between
    are counted as one episode (matches write_driver_drain_run_sequence_table).
    The count comes from the pre-generated sequence-details CSV so it is
    consistent with the bar-chart denominators.
    """
    slug         = tag_slug(tag)
    details_path = os.path.join(drain_plots_dir,
                                f'table_{slug}_drain_run_sequence_details.csv')
    drain_ep = 0
    if os.path.exists(details_path):
        with open(details_path) as f:
            drain_ep = sum(1 for _ in csv.DictReader(f))
    # Denominator: total eligible MCTS cycles (from raw performance/decision CSVs)
    _, eligible, _ = _load_drain_pct_for_tag(results_dir, tag)
    pct = 100.0 * drain_ep / eligible if eligible else 0.0
    return drain_ep, eligible, pct


def _draw_drain_overview_row(axes_left, axes_right, drain_tags, drain_pct_by_tag,
                              postdrain_group_by_tag, total_group_by_tag,
                              driver_colors, left_ymax=None, right_ymax=None,
                              show_legend=True):
    """Draw one system row (left=drain rate, right=post-drain class share)."""
    GROUPS  = ['S', 'M', 'L', 'XL']
    FS_TICK = 8.5
    FS_AXIS = 9.0
    FS_BAR  = 7.5

    def _bold_ticks(ax):
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
            lbl.set_color('black')
        ax.tick_params(labelsize=FS_TICK, colors='black', width=1.2, length=4)

    # ── left: drain rate ─────────────────────────────────────────────────────
    ax = axes_left
    bar_w  = 0.30
    gap_l  = 0.52
    n_left = len(drain_tags)
    offs_l = np.linspace(-(n_left - 1) / 2, (n_left - 1) / 2, n_left) * gap_l
    x_l    = 0.5 + offs_l
    left_pcts   = []
    left_counts = []
    for tag in drain_tags:
        tup = drain_pct_by_tag.get(tag, (0, 0, 0.0))
        left_pcts.append(tup[2])
        left_counts.append((tup[0], tup[1]))

    for i, tag in enumerate(drain_tags):
        ax.bar(x_l[i], left_pcts[i], width=bar_w,
               color=driver_colors.get(tag, 'gray'), alpha=0.88,
               hatch=driver_hatch(tag), edgecolor='black', linewidth=0.8)

    y_pad = max(left_pcts, default=1) * 0.02
    for i, pct in enumerate(left_pcts):
        ax.text(x_l[i], pct + y_pad,
                f'{pct:.1f}%',
                ha='center', va='bottom', rotation=90,
                fontsize=FS_BAR - 0.5, fontweight='bold', color='black')

    ax.set_xticks(x_l)
    lbls = ax.set_xticklabels([display_tag(t) for t in drain_tags],
                               rotation=20, ha='right')
    for lbl in lbls:
        lbl.set_fontweight('bold'); lbl.set_color('black')
    ax.set_xlim(x_l[0] - bar_w * 1.4, x_l[-1] + bar_w * 1.4)
    ax.set_ylabel('Drain Episodes (%)', fontsize=FS_AXIS, fontweight='bold',
                  color='black')
    if left_ymax is not None:
        ax.set_ylim(0, left_ymax)
    else:
        ax.set_ylim(0, max(left_pcts, default=1) * 1.85)
    ax.grid(True, axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    _bold_ticks(ax)

    # ── right: post-drain class share ────────────────────────────────────────
    ax   = axes_right
    bw   = 0.32
    x_r  = np.arange(len(GROUPS), dtype=float)
    n    = len(drain_tags)
    offs = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * bw

    right_pcts = {}
    for tag in drain_tags:
        totals = total_group_by_tag.get(tag, {})
        right_pcts[tag] = {
            g: 100.0 * postdrain_group_by_tag.get(tag, {}).get(g, 0) / totals.get(g, 1)
            if totals.get(g, 0) else 0.0
            for g in GROUPS
        }
        ax.bar(x_r + offs[drain_tags.index(tag)],
               [right_pcts[tag][g] for g in GROUPS],
               width=bw,
               color=driver_colors.get(tag, 'gray'), alpha=0.88,
               hatch=driver_hatch(tag), edgecolor='black', linewidth=0.8,
               label=display_tag(tag))

    global_max = max(
        (right_pcts[t][g] for t in drain_tags for g in GROUPS), default=1)
    y_pad_r = global_max * 0.02
    for tag in drain_tags:
        xi = drain_tags.index(tag)
        for j, g in enumerate(GROUPS):
            v = right_pcts[tag][g]
            ax.text(x_r[j] + offs[xi], v + y_pad_r, f'{v:.1f}%',
                    ha='center', va='bottom', rotation=90,
                    fontsize=FS_BAR - 0.5, fontweight='bold', color='black')

    ax.set_xticks(x_r)
    lbls = ax.set_xticklabels(GROUPS, rotation=30, ha='center')
    for lbl in lbls:
        lbl.set_fontweight('bold'); lbl.set_color('black')
    ax.set_xlim(x_r[0] - bw * 1.6, x_r[-1] + bw * 1.6)
    ax.set_ylabel('Post-drain Jobs (%)', fontsize=FS_AXIS, fontweight='bold',
                  color='black')
    if right_ymax is not None:
        ax.set_ylim(0, right_ymax)
    else:
        ax.set_ylim(0, global_max * 1.85)
    ax.grid(True, axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(fontsize=8.0, loc='upper right', framealpha=0.90, edgecolor='black')
    _bold_ticks(ax)


def _load_drain_combo_stats(drain_plots_dir, tag, results_dir):
    """Load seq_counts, total_drain_episodes and drain-pct tuple from disk.

    Reads:
      table_{slug}_drain_sequence_combos.csv   → seq_counts
      table_{slug}_drain_run_sequence_details.csv → total (rows with First_Run_Group)
      MCTS performance/decision CSVs            → drain-pct tuple via _load_drain_pct_for_tag
    Returns (stats_dict_or_None, drain_pct_tuple_or_None).
    """
    slug         = tag_slug(tag)
    combos_path  = os.path.join(drain_plots_dir,
                                f'table_{slug}_drain_sequence_combos.csv')
    details_path = os.path.join(drain_plots_dir,
                                f'table_{slug}_drain_run_sequence_details.csv')

    if not os.path.exists(combos_path) or not os.path.exists(details_path):
        return None, None

    seq_counts = {}
    with open(combos_path) as f:
        for row in csv.DictReader(f):
            try:
                seq_counts[row['Sequence']] = int(row['Count'])
            except (KeyError, ValueError):
                pass

    total = 0
    with open(details_path) as f:
        for row in csv.DictReader(f):
            if row.get('First_Run_Group', '').strip():
                total += 1

    if not seq_counts or total == 0:
        return None, None

    stats = {
        'tag':                  tag,
        'slug':                 slug,
        'seq_counts':           seq_counts,
        'total_drain_episodes': total,
    }
    drain_pct = _load_drain_pct_for_tag(results_dir, tag)
    return stats, drain_pct


def _inject_sm_lxl_entry(entries, seq_counts):
    """Append '{S/M}-L-*' and '{S/M}-XL-*' aggregate entries and re-sort by count.

    Each count is the sum of seq_counts entries where G1 ∈ {S,M} and G2 = L or XL.
    """
    totals = {}
    for g2 in ('L', 'XL'):
        total = sum(
            cnt for seq, cnt in seq_counts.items()
            if len(seq.split('-')) == 3
            and seq.split('-')[0] in ('S', 'M')
            and seq.split('-')[1] == g2
        )
        if total > 0:
            totals[f'{{S/M}}-{g2}-*'] = total
    if not totals:
        return entries
    result = list(entries) + list(totals.items())
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def plot_combined_drain_sequence_combos_ab(
        theta_results_dir, polaris_results_dir,
        theta_drain_dir, polaris_drain_dir,
        out_path,
        drain_tags=('MARS-CW', 'MARS-CU'),
        max_entries=5):
    """2-row side-by-side figure of post-drain sequence combos.

    Top row    = Theta 2021  (CW left, CU right)
    Bottom row = Polaris 2024 (CW left, CU right)

    Mirrors the layout of drain_sequence_combos_combined.png but stacked.
    """
    drain_tags    = list(drain_tags)
    driver_colors = _build_driver_colors(drain_tags)

    # Load stats for each system × tag
    systems = [
        ('Theta 2021',   theta_results_dir,   theta_drain_dir),
        ('Polaris 2024', polaris_results_dir, polaris_drain_dir),
    ]
    all_stats     = {}   # (sys_label, tag) → stats
    all_drain_pct = {}   # (sys_label, tag) → (drain_ep, eligible, pct)

    for sys_label, results_dir, drain_dir in systems:
        for tag in drain_tags:
            stats, pct = _load_drain_combo_stats(drain_dir, tag, results_dir)
            all_stats[(sys_label, tag)]     = stats
            all_drain_pct[(sys_label, tag)] = pct

    # ── Build shared sequence list per row so CW/CU are comparable ──────────
    def _shared_entries(sys_label):
        cw_s = all_stats.get((sys_label, 'MARS-CW'))
        cu_s = all_stats.get((sys_label, 'MARS-CU'))
        cw_e = _build_wildcard_sequence_entries(
            cw_s['seq_counts'] if cw_s else {}, max_entries=max_entries)
        cu_e = _build_wildcard_sequence_entries(
            cu_s['seq_counts'] if cu_s else {}, max_entries=max_entries)
        cw_e = _inject_sm_lxl_entry(cw_e, cw_s['seq_counts'] if cw_s else {})
        cu_e = _inject_sm_lxl_entry(cu_e, cu_s['seq_counts'] if cu_s else {})
        # Build a shared Y-axis order: CW order defines positions;
        # CU is aligned to the same positions (its own counts, no labels).
        cw_dict = dict(cw_e)
        cu_dict = dict(cu_e)
        shared_seqs = [seq for seq, _ in cw_e]
        for seq, _ in cu_e:
            if seq not in cw_dict:
                shared_seqs.append(seq)
        cw_aligned = [(seq, cw_dict.get(seq, 0)) for seq in shared_seqs]
        cu_aligned = [(seq, cu_dict.get(seq, 0)) for seq in shared_seqs]
        return cw_aligned, cu_aligned

    theta_cw_e,   theta_cu_e   = _shared_entries('Theta 2021')
    polaris_cw_e, polaris_cu_e = _shared_entries('Polaris 2024')

    n_rows = len(systems)
    n_bars = max(len(theta_cw_e), len(polaris_cw_e), max_entries + 2)
    row_h  = max(1.2, n_bars * 0.30 + 0.6)
    fig_h  = row_h * n_rows + 0.4

    fig, axes = plt.subplots(n_rows, 2, figsize=(6.4, fig_h))
    fig.subplots_adjust(left=0.26, right=0.97, top=0.92, bottom=0.10,
                        wspace=0.12, hspace=0.40)

    def _title(tag, pct_tuple):
        dtag = display_tag(tag)
        if pct_tuple:
            drain_ep, eligible, pct = pct_tuple
            return f'{dtag}\nN={drain_ep:,} ({pct:.1f}%)'
        return dtag

    row_data = [
        ('Theta 2021',   theta_cw_e,   theta_cu_e),
        ('Polaris 2024', polaris_cw_e, polaris_cu_e),
    ]
    for row_idx, (sys_label, cw_entries, cu_entries) in enumerate(row_data):
        ax_cw = axes[row_idx][0]
        ax_cu = axes[row_idx][1]

        cw_stats = all_stats.get((sys_label, 'MARS-CW'))
        cu_stats = all_stats.get((sys_label, 'MARS-CU'))
        cw_total = cw_stats['total_drain_episodes'] if cw_stats else 1
        cu_total = cu_stats['total_drain_episodes'] if cu_stats else 1

        cw_pct = all_drain_pct.get((sys_label, 'MARS-CW'))
        cu_pct = all_drain_pct.get((sys_label, 'MARS-CU'))

        _draw_drain_sequence_bar(
            ax_cw, cw_entries, cw_total,
            show_ylabels=True,
            title=_title('MARS-CW', cw_pct),
            bar_color=driver_colors.get('MARS-CW', '#add8e6'),
            hatch=driver_hatch('MARS-CW')
        )
        _draw_drain_sequence_bar(
            ax_cu, cu_entries, cu_total,
            show_ylabels=False,
            title=_title('MARS-CU', cu_pct),
            bar_color=driver_colors.get('MARS-CU', '#add8e6'),
            hatch=driver_hatch('MARS-CU')
        )

        # Row label on the side via ylabel of left panel
        ax_cw.set_ylabel(sys_label, fontsize=12.0, fontweight='bold',
                         labelpad=6, color='black')

        # Only bottom row gets x-axis label
        if row_idx < n_rows - 1:
            ax_cw.set_xlabel('')
            ax_cu.set_xlabel('')

        for ax in (ax_cw, ax_cu):
            ax.set_title(ax.get_title(), fontsize=11.5, fontweight='bold')
            ax.set_xlabel(ax.get_xlabel(), fontsize=11.5, fontweight='bold')
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontsize(10.5)
                lbl.set_fontweight('bold')


    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_drain_overview_ab(
        theta_results_dir, polaris_results_dir,
        theta_drain_dir, polaris_drain_dir,
        out_path,
        drain_tags=('MARS-CW', 'MARS-CU')):
    """2-row combined figure: top=Theta 2021, bottom=Polaris 2024.

    Each row: left=drain rate bar chart, right=post-drain share by size class.
    """
    drain_tags = list(drain_tags)
    driver_colors = _build_driver_colors(drain_tags)

    theta_data   = _load_drain_overview_data(
        theta_results_dir, theta_drain_dir, drain_tags, _assign_group_theta)
    polaris_data = _load_drain_overview_data(
        polaris_results_dir, polaris_drain_dir, drain_tags, _assign_group_polaris)

    fig, axes = plt.subplots(
        2, 2, figsize=(7.0, 4.2),
        gridspec_kw={'width_ratios': [1, 2.4]},
    )
    fig.subplots_adjust(left=0.20, right=0.97, top=0.94, bottom=0.12,
                        wspace=0.32, hspace=0.35)

    _draw_drain_overview_row(
        axes[0][0], axes[0][1],
        drain_tags, *theta_data, driver_colors,
        left_ymax=25, right_ymax=80, show_legend=False
    )
    # Override left-panel ylabel with system name as row label
    axes[0][0].set_ylabel('Theta 2021', fontsize=10.0, fontweight='bold',
                           color='black', labelpad=1)
    axes[0][0].set_title('Drain Rate', fontsize=9.5, fontweight='bold', pad=3)
    axes[0][1].set_title('Post-Drain Share by Class', fontsize=9.5, fontweight='bold', pad=3)

    _draw_drain_overview_row(
        axes[1][0], axes[1][1],
        drain_tags, *polaris_data, driver_colors,
        left_ymax=25, right_ymax=70, show_legend=False
    )
    axes[1][0].set_ylabel('Polaris 2024', fontsize=10.0, fontweight='bold',
                           color='black', labelpad=1)
    axes[1][1].set_xlabel('Size Class', fontsize=8.5, fontweight='bold')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def _get_drain_lxl_ids(results_dir, assign_group_fn, tag):
    """Return the set of L/XL job IDs that appear as targets in DRAIN cycles.

    Reads descisions.csv: any non-NONE job token in a DRAIN row that belongs
    to L or XL is collected.  Works for both heuristic and MCTS policies.
    """
    dec_path  = os.path.join(results_dir, tag, "descisions.csv")
    if not os.path.exists(dec_path):
        return set()
    procs_map = _infer_procs_map_for_results_dir(results_dir)
    ids = set()
    with open(dec_path) as f:
        for row in csv.DictReader(f):
            if row.get("selected_policy", "").strip() != "DRAIN":
                continue
            pjs = row.get("possible_job_sets", "") or ""
            for token in pjs.split("|"):
                token = token.strip()
                if not token or token == "NONE":
                    continue
                for part in token.split(","):
                    part = part.strip()
                    if part and part != "NONE":
                        try:
                            jid = int(part)
                            if assign_group_fn(procs_map.get(jid, -1)) in ("L", "XL"):
                                ids.add(jid)
                        except ValueError:
                            pass
    return ids


def _load_sm_post_lxl_wait_data(results_dir, assign_group_fn, tag,
                                 lxl_filter=None):
    """Return {('post_lxl'|'otherwise', 'S'|'M'): [wait_hours]}.

    Classification rule: walk the ordered Run/Backfill event sequence.
    Whenever a L/XL Run event is followed immediately by a S/M Run event,
    that S/M job is 'post_lxl'.  All other S/M jobs are 'otherwise'.

    lxl_filter: optional set of L/XL job IDs to restrict triggers.
                None  → every L/XL Run event qualifies as a trigger.
                set() → restrict to those specific L/XL jobs (e.g. drain targets
                        for MCTS).
    """
    events_path = os.path.join(results_dir, tag, "events.csv")
    empty = {k: [] for k in (("post_lxl", "S"), ("post_lxl", "M"),
                              ("otherwise", "S"), ("otherwise", "M"))}
    if not os.path.exists(events_path):
        return empty

    procs_map = _infer_procs_map_for_results_dir(results_dir)

    submit_map  = {}   # job_id → first submit time
    first_start = {}   # job_id → first Run/Backfill time (for wait calculation)
    run_seq     = []   # all (sim_time, job_id) Run/Backfill events in file order

    with open(events_path) as f:
        for row in csv.DictReader(f):
            evt = row["event"]
            jid = int(row["id"])
            t   = int(row["sim_time"])
            if evt == "Submit":
                if jid not in submit_map:
                    submit_map[jid] = t
            elif evt in ("Run", "Backfill"):
                if jid not in first_start:
                    first_start[jid] = t
                run_seq.append((t, jid))

    # Walk consecutive pairs in the run sequence
    post_lxl_ids = set()
    for i in range(len(run_seq) - 1):
        jid_cur  = run_seq[i][1]
        jid_next = run_seq[i + 1][1]
        if assign_group_fn(procs_map.get(jid_cur, -1)) not in ("L", "XL"):
            continue
        if lxl_filter is not None and jid_cur not in lxl_filter:
            continue
        if assign_group_fn(procs_map.get(jid_next, -1)) in ("S", "M"):
            post_lxl_ids.add(jid_next)

    data = {("post_lxl", "S"): [], ("post_lxl", "M"): [],
            ("otherwise", "S"): [], ("otherwise", "M"): []}

    for jid, sub_t in submit_map.items():
        if jid not in first_start:
            continue
        grp = assign_group_fn(procs_map.get(jid, -1))
        if grp not in ("S", "M"):
            continue
        wait_h = (first_start[jid] - sub_t) / 3600.0
        if wait_h < 0:
            continue
        kind = "post_lxl" if jid in post_lxl_ids else "otherwise"
        data[(kind, grp)].append(wait_h)

    return data


def plot_sm_post_drain_grid_ab(
        theta_results_dir, polaris_results_dir,
        out_path):
    """2×3 grid: rows = Theta/Polaris, cols = WFP / MARS-CW / LRF.

    S and M jobs only, split into 'post_lxl' vs 'otherwise':
      MARS-CW: post_lxl = S/M whose Run event immediately follows a drain-
               targeted L/XL in the event sequence.
      WFP/LRF: post_lxl = S/M whose Run event immediately follows ANY L/XL
               run in the event sequence.

    Y axis: % of total S+M jobs — shared per row.
    X axis: log scale 10^-1–10^4 — shared.
    """
    from matplotlib.patches import Patch

    bins  = np.logspace(-1, 4, 41)
    OTH_S = "#cccccc"
    OTH_M = "#666666"

    tags       = ["WFP", "MARS-CW"]
    ag_fns     = [_assign_group_theta,   _assign_group_polaris]
    rdirs      = [theta_results_dir,     polaris_results_dir]
    col_colors = [
        ("#aec7e8", "#1f77b4"),
        ("#f4a0a0", "#c0392b"),
    ]
    col_titles = ["WFP", "MARS-CW"]
    row_labels = ["Theta 2021", "Polaris 2024"]
    # MARS-CW uses only drain-targeted L/XL as triggers; heuristics use all L/XL
    is_mcts    = [False, True]

    datasets = []
    for rdir, ag_fn in zip(rdirs, ag_fns):
        for tag, mcts in zip(tags, is_mcts):
            lxl_filter = _get_drain_lxl_ids(rdir, ag_fn, tag) if mcts else None
            datasets.append(_load_sm_post_lxl_wait_data(rdir, ag_fn, tag, lxl_filter))

    OTH = [("otherwise", "S"), ("otherwise", "M")]
    PD  = [("post_lxl",  "S"), ("post_lxl",  "M")]

    theta_total   = max(sum(len(v) for v in datasets[0].values()), 1)
    polaris_total = max(sum(len(v) for v in datasets[2].values()), 1)

    def _peak(data, total):
        stk = np.zeros(len(bins) - 1)
        for k in OTH + PD:
            h, _ = np.histogram(data.get(k, []), bins=bins)
            stk += h
        return float((stk / total * 100).max())

    theta_ymax   = max(_peak(datasets[i],   theta_total)   for i in range(2)) * 1.18
    polaris_ymax = max(_peak(datasets[i+2], polaris_total) for i in range(2)) * 1.18

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88,
                        bottom=0.17, wspace=0.20, hspace=0.40)

    for idx, data in enumerate(datasets):
        row, col = divmod(idx, 2)
        ax     = axes[row][col]
        pd_S, pd_M = col_colors[col]
        total  = theta_total   if row == 0 else polaris_total
        ymax   = theta_ymax    if row == 0 else polaris_ymax
        mcts   = is_mcts[col]

        h_oth_s, _ = np.histogram(data.get(("otherwise", "S"), []), bins=bins)
        h_oth_m, _ = np.histogram(data.get(("otherwise", "M"), []), bins=bins)
        h_pd_s,  _ = np.histogram(data.get(("post_lxl",  "S"), []), bins=bins)
        h_pd_m,  _ = np.histogram(data.get(("post_lxl",  "M"), []), bins=bins)

        pct_oth_s = h_oth_s / total * 100
        pct_oth_m = h_oth_m / total * 100
        pct_pd_s  = h_pd_s  / total * 100
        pct_pd_m  = h_pd_m  / total * 100

        lefts  = bins[:-1]
        widths = np.diff(bins)
        b0 = np.zeros(len(bins) - 1)
        b1 = b0 + pct_oth_s
        b2 = b1 + pct_oth_m
        b3 = b2 + pct_pd_s

        ax.bar(lefts, pct_oth_s, width=widths, bottom=b0, align="edge",
               color=OTH_S, zorder=3)
        ax.bar(lefts, pct_oth_m, width=widths, bottom=b1, align="edge",
               color=OTH_M, zorder=3)
        ax.bar(lefts, pct_pd_s,  width=widths, bottom=b2, align="edge",
               color=pd_S,  zorder=3)
        ax.bar(lefts, pct_pd_m,  width=widths, bottom=b3, align="edge",
               color=pd_M,  zorder=3)

        ax.set_xscale("log")
        ax.set_xlim(1e-1, 1e4)
        ax.set_ylim(0, ymax)

        n_pd    = sum(len(data.get(k, [])) for k in PD)
        n_panel = n_pd + sum(len(data.get(k, [])) for k in OTH)
        pct_pd  = 100.0 * n_pd / n_panel if n_panel else 0.0
        label   = "Post-drain L/XL" if mcts else "Post-L/XL"
        ax.set_title(f"{label}: {n_pd:,}  ({pct_pd:.1f}%)",
                     fontsize=10.0, fontweight="bold", pad=3)

        if row == 0:
            ax.text(0.5, 1.28, col_titles[col], transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax.set_ylabel(
            f"{row_labels[row]}\n% of S+M Jobs" if col == 0 else "",
            fontsize=9.5, fontweight="bold",
        )
        ax.set_xlabel(
            "Wait Time (hours)" if row == 1 else "",
            fontsize=10, fontweight="bold",
        )
        ax.tick_params(axis="both", labelsize=8.5)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        ax.grid(True, alpha=0.25, which="both", zorder=0)

    legend_handles = [
        Patch(facecolor=OTH_S,    label="Otherwise  S"),
        Patch(facecolor=OTH_M,    label="Otherwise  M"),
        Patch(facecolor="#aec7e8", label="WFP Post-L/XL  S"),
        Patch(facecolor="#1f77b4", label="WFP Post-L/XL  M"),
        Patch(facecolor="#f4a0a0", label="MARS-CW Post-drain L/XL  S"),
        Patch(facecolor="#c0392b", label="MARS-CW Post-drain L/XL  M"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9.5,
               frameon=False, handletextpad=0.4, columnspacing=1.2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _get_mars_drain_lxl_job_ids(drain_plots_dir):
    """Return set of L/XL job IDs from MARS-CW drain run-sequence details CSV."""
    path = os.path.join(drain_plots_dir, "table_mars_cw_drain_run_sequence_details.csv")
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path) as f:
        for row in csv.DictReader(f):
            for pos in ("First", "Second", "Third"):
                grp = row.get(f"{pos}_Run_Group", "").strip()
                if grp not in ("L", "XL"):
                    continue
                jid_str = row.get(f"{pos}_Run_Job", "").strip()
                if jid_str:
                    try:
                        ids.add(int(jid_str))
                    except ValueError:
                        pass
    return ids


def _load_lxl_wait_with_highlight(results_dir, assign_group_fn, tag, highlight_ids):
    """Return {('highlight'|'other', 'L'|'XL'): [wait_hours]}.

    highlight_ids: job IDs to highlight (e.g. the MARS-CW drain L/XL jobs).
    All other L/XL jobs are 'other'.
    """
    events_path = os.path.join(results_dir, tag, "events.csv")
    empty = {k: [] for k in (("highlight", "L"), ("highlight", "XL"),
                              ("other",     "L"), ("other",     "XL"))}
    if not os.path.exists(events_path):
        return empty

    procs_map  = _infer_procs_map_for_results_dir(results_dir)
    submit_map = {}
    start_map  = {}

    with open(events_path) as f:
        for row in csv.DictReader(f):
            evt = row["event"]
            jid = int(row["id"])
            t   = int(row["sim_time"])
            if evt == "Submit":
                if jid not in submit_map:
                    submit_map[jid] = t
            elif evt in ("Run", "Backfill"):
                if jid not in start_map:
                    start_map[jid] = t

    data = {("highlight", "L"): [], ("highlight", "XL"): [],
            ("other",     "L"): [], ("other",     "XL"): []}
    for jid, sub_t in submit_map.items():
        if jid not in start_map:
            continue
        grp = assign_group_fn(procs_map.get(jid, -1))
        if grp not in ("L", "XL"):
            continue
        wait_h = (start_map[jid] - sub_t) / 3600.0
        if wait_h < 0:
            continue
        kind = "highlight" if jid in highlight_ids else "other"
        data[(kind, grp)].append(wait_h)
    return data


def _get_mars_drain_sm_job_ids(drain_plots_dir):
    """Return set of S/M job IDs at positions 1/2/3 in MARS-CW drain sequences."""
    path = os.path.join(drain_plots_dir, "table_mars_cw_drain_run_sequence_details.csv")
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path) as f:
        for row in csv.DictReader(f):
            for pos in ("First", "Second", "Third"):
                if row.get(f"{pos}_Run_Group", "").strip() not in ("S", "M"):
                    continue
                jid_str = row.get(f"{pos}_Run_Job", "").strip()
                if jid_str:
                    try:
                        ids.add(int(jid_str))
                    except ValueError:
                        pass
    return ids


def _load_sm_wait_with_highlight(results_dir, assign_group_fn, tag, highlight_ids):
    """Return {('highlight'|'other', 'S'|'M'): [wait_hours]}.

    highlight_ids: job IDs to highlight (e.g. MARS post-drain S/M job IDs).
    """
    events_path = os.path.join(results_dir, tag, "events.csv")
    empty = {k: [] for k in (("highlight", "S"), ("highlight", "M"),
                              ("other",     "S"), ("other",     "M"))}
    if not os.path.exists(events_path):
        return empty

    procs_map  = _infer_procs_map_for_results_dir(results_dir)
    submit_map = {}
    start_map  = {}

    with open(events_path) as f:
        for row in csv.DictReader(f):
            evt = row["event"]
            jid = int(row["id"])
            t   = int(row["sim_time"])
            if evt == "Submit":
                if jid not in submit_map:
                    submit_map[jid] = t
            elif evt in ("Run", "Backfill"):
                if jid not in start_map:
                    start_map[jid] = t

    data = {("highlight", "S"): [], ("highlight", "M"): [],
            ("other",     "S"): [], ("other",     "M"): []}
    for jid, sub_t in submit_map.items():
        if jid not in start_map:
            continue
        grp = assign_group_fn(procs_map.get(jid, -1))
        if grp not in ("S", "M"):
            continue
        wait_h = (start_map[jid] - sub_t) / 3600.0
        if wait_h < 0:
            continue
        kind = "highlight" if jid in highlight_ids else "other"
        data[(kind, grp)].append(wait_h)
    return data


def plot_mars_drain_sm_jobs_cross_policy_grid_ab(
        theta_results_dir, polaris_results_dir,
        theta_drain_dir, polaris_drain_dir,
        out_path):
    """2×3 grid: rows = Theta/Polaris, cols = WFP / MARS-CW / LRF.

    Highlighted (coloured) = S/M jobs at positions 1/2/3 in any MARS-CW drain
    episode.  The same job IDs are looked up in WFP and LRF to show how those
    policies handle the same jobs.  Grey = all other S+M jobs.
    Y axis: % of total S+M jobs — shared per row.
    X axis: log scale 10^-1–10^4 — shared.
    """
    from matplotlib.patches import Patch

    bins   = np.logspace(-1, 4, 41)
    OTH_S  = "#cccccc"
    OTH_M  = "#666666"

    theta_sm_ids   = _get_mars_drain_sm_job_ids(theta_drain_dir)
    polaris_sm_ids = _get_mars_drain_sm_job_ids(polaris_drain_dir)

    tags       = ["WFP", "MARS-CW"]
    ag_fns     = [_assign_group_theta,   _assign_group_polaris]
    rdirs      = [theta_results_dir,     polaris_results_dir]
    drain_sets = [theta_sm_ids,          polaris_sm_ids]
    col_colors = [
        ("#aec7e8", "#1f77b4"),
        ("#f4a0a0", "#c0392b"),
    ]
    col_titles = ["WFP", "MARS-CW"]
    row_labels = ["Theta 2021", "Polaris 2024"]

    datasets = []
    for rdir, ag_fn, sm_ids in zip(rdirs, ag_fns, drain_sets):
        for tag in tags:
            datasets.append(_load_sm_wait_with_highlight(rdir, ag_fn, tag, sm_ids))

    OTH = [("other",     "S"), ("other",     "M")]
    HI  = [("highlight", "S"), ("highlight", "M")]

    theta_total   = max(sum(len(v) for v in datasets[0].values()), 1)
    polaris_total = max(sum(len(v) for v in datasets[2].values()), 1)

    def _peak(data, total):
        stk = np.zeros(len(bins) - 1)
        for k in OTH + HI:
            h, _ = np.histogram(data.get(k, []), bins=bins)
            stk += h
        return float((stk / total * 100).max())

    theta_ymax   = max(_peak(datasets[i],   theta_total)   for i in range(2)) * 1.18
    polaris_ymax = max(_peak(datasets[i+2], polaris_total) for i in range(2)) * 1.18

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88,
                        bottom=0.17, wspace=0.20, hspace=0.40)

    for idx, data in enumerate(datasets):
        row, col    = divmod(idx, 2)
        ax          = axes[row][col]
        hi_S, hi_M  = col_colors[col]
        total       = theta_total   if row == 0 else polaris_total
        ymax        = theta_ymax    if row == 0 else polaris_ymax

        h_oth_s, _ = np.histogram(data.get(("other",     "S"), []), bins=bins)
        h_oth_m, _ = np.histogram(data.get(("other",     "M"), []), bins=bins)
        h_hi_s,  _ = np.histogram(data.get(("highlight", "S"), []), bins=bins)
        h_hi_m,  _ = np.histogram(data.get(("highlight", "M"), []), bins=bins)

        pct_oth_s = h_oth_s / total * 100
        pct_oth_m = h_oth_m / total * 100
        pct_hi_s  = h_hi_s  / total * 100
        pct_hi_m  = h_hi_m  / total * 100

        lefts  = bins[:-1]
        widths = np.diff(bins)
        b0 = np.zeros(len(bins) - 1)
        b1 = b0 + pct_oth_s
        b2 = b1 + pct_oth_m
        b3 = b2 + pct_hi_s

        ax.bar(lefts, pct_oth_s, width=widths, bottom=b0, align="edge",
               color=OTH_S, zorder=3)
        ax.bar(lefts, pct_oth_m, width=widths, bottom=b1, align="edge",
               color=OTH_M, zorder=3)
        ax.bar(lefts, pct_hi_s,  width=widths, bottom=b2, align="edge",
               color=hi_S,  zorder=3)
        ax.bar(lefts, pct_hi_m,  width=widths, bottom=b3, align="edge",
               color=hi_M,  zorder=3)

        ax.set_xscale("log")
        ax.set_xlim(1e-1, 1e4)
        ax.set_ylim(0, ymax)

        n_hi    = sum(len(data.get(k, [])) for k in HI)
        n_panel = n_hi + sum(len(data.get(k, [])) for k in OTH)
        pct_hi  = 100.0 * n_hi / n_panel if n_panel else 0.0
        title_lbl = "Post-drain pos 1/2/3" if col == 1 else "MARS-drain S/M jobs"
        ax.set_title(f"{title_lbl}: {n_hi:,}  ({pct_hi:.1f}%)",
                     fontsize=10.0, fontweight="bold", pad=3)

        if row == 0:
            ax.text(0.5, 1.28, col_titles[col], transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax.set_ylabel(
            f"{row_labels[row]}\n% of S+M Jobs" if col == 0 else "",
            fontsize=9.5, fontweight="bold",
        )
        ax.set_xlabel(
            "Wait Time (hours)" if row == 1 else "",
            fontsize=10, fontweight="bold",
        )
        ax.tick_params(axis="both", labelsize=8.5)
        for obj in ax.get_xticklabels() + ax.get_yticklabels():
            obj.set_fontweight("bold")
        ax.grid(True, alpha=0.25, which="both", zorder=0)

    legend_handles = [
        Patch(facecolor=OTH_S,    label="Other S+M  S"),
        Patch(facecolor=OTH_M,    label="Other S+M  M"),
        Patch(facecolor="#aec7e8", label="WFP: MARS-drain S/M  S"),
        Patch(facecolor="#1f77b4", label="WFP: MARS-drain S/M  M"),
        Patch(facecolor="#f4a0a0", label="MARS-CW: post-drain pos 1/2/3  S"),
        Patch(facecolor="#c0392b", label="MARS-CW: post-drain pos 1/2/3  M"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=9.5,
               frameon=False, handletextpad=0.4, columnspacing=1.2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_mars_drain_jobs_cross_policy_grid_ab(
        theta_results_dir, polaris_results_dir,
        theta_drain_dir, polaris_drain_dir,
        out_path):
    """2×3 grid: rows = Theta/Polaris, cols = WFP / MARS-CW / LRF.

    The highlighted (coloured) jobs are the L/XL jobs that MARS-CW drained for,
    identified once from the MARS drain-sequence details CSV and then looked up
    by job ID in WFP and LRF.  Shows how the same jobs are handled by each policy.
    Grey = all other L/XL jobs.
    Y axis: % of total L+XL jobs — shared per row.
    X axis: log scale 10^-1–10^4 — shared.
    """
    from matplotlib.patches import Patch

    bins   = np.logspace(-1, 4, 41)
    OTH_L  = "#cccccc"
    OTH_XL = "#666666"

    theta_drain_ids   = _get_mars_drain_lxl_job_ids(theta_drain_dir)
    polaris_drain_ids = _get_mars_drain_lxl_job_ids(polaris_drain_dir)

    tags       = ["WFP", "MARS-CW"]
    ag_fns     = [_assign_group_theta,   _assign_group_polaris]
    rdirs      = [theta_results_dir,     polaris_results_dir]
    drain_sets = [theta_drain_ids,       polaris_drain_ids]
    col_colors = [
        ("#aec7e8", "#1f77b4"),
        ("#f4a0a0", "#c0392b"),
    ]
    col_titles = ["WFP", "MARS-CW"]
    row_labels = ["Theta 2021", "Polaris 2024"]

    datasets = []
    for rdir, ag_fn, drain_ids in zip(rdirs, ag_fns, drain_sets):
        for tag in tags:
            datasets.append(_load_lxl_wait_with_highlight(rdir, ag_fn, tag, drain_ids))

    OTH = [("other",     "L"), ("other",     "XL")]
    HI  = [("highlight", "L"), ("highlight", "XL")]

    theta_total   = max(sum(len(v) for v in datasets[0].values()), 1)
    polaris_total = max(sum(len(v) for v in datasets[2].values()), 1)

    def _peak(data, total):
        stk = np.zeros(len(bins) - 1)
        for k in OTH + HI:
            h, _ = np.histogram(data.get(k, []), bins=bins)
            stk += h
        return float((stk / total * 100).max())

    theta_ymax   = max(_peak(datasets[i],   theta_total)   for i in range(2)) * 1.18
    polaris_ymax = max(_peak(datasets[i+2], polaris_total) for i in range(2)) * 1.18

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88,
                        bottom=0.17, wspace=0.20, hspace=0.40)

    for idx, data in enumerate(datasets):
        row, col = divmod(idx, 2)
        ax      = axes[row][col]
        hi_L, hi_XL = col_colors[col]
        total   = theta_total   if row == 0 else polaris_total
        ymax    = theta_ymax    if row == 0 else polaris_ymax

        h_oth_l,  _ = np.histogram(data.get(("other",     "L"),  []), bins=bins)
        h_oth_xl, _ = np.histogram(data.get(("other",     "XL"), []), bins=bins)
        h_hi_l,   _ = np.histogram(data.get(("highlight", "L"),  []), bins=bins)
        h_hi_xl,  _ = np.histogram(data.get(("highlight", "XL"), []), bins=bins)

        pct_oth_l  = h_oth_l  / total * 100
        pct_oth_xl = h_oth_xl / total * 100
        pct_hi_l   = h_hi_l   / total * 100
        pct_hi_xl  = h_hi_xl  / total * 100

        lefts  = bins[:-1]
        widths = np.diff(bins)
        b0 = np.zeros(len(bins) - 1)
        b1 = b0 + pct_oth_l
        b2 = b1 + pct_oth_xl
        b3 = b2 + pct_hi_l

        ax.bar(lefts, pct_oth_l,  width=widths, bottom=b0, align="edge",
               color=OTH_L,  zorder=3)
        ax.bar(lefts, pct_oth_xl, width=widths, bottom=b1, align="edge",
               color=OTH_XL, zorder=3)
        ax.bar(lefts, pct_hi_l,   width=widths, bottom=b2, align="edge",
               color=hi_L,   zorder=3)
        ax.bar(lefts, pct_hi_xl,  width=widths, bottom=b3, align="edge",
               color=hi_XL,  zorder=3)

        ax.set_xscale("log")
        ax.set_xlim(1e-1, 1e4)
        ax.set_ylim(0, ymax)

        n_hi    = sum(len(data.get(k, [])) for k in HI)
        n_panel = n_hi + sum(len(data.get(k, [])) for k in OTH)
        pct_hi  = 100.0 * n_hi / n_panel if n_panel else 0.0
        title_lbl = "Drain" if col == 1 else "MARS-drain jobs"
        ax.set_title(f"{title_lbl}: {n_hi:,}  ({pct_hi:.1f}%)",
                     fontsize=10.0, fontweight="bold", pad=3)

        if row == 0:
            ax.text(0.5, 1.28, col_titles[col], transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax.set_ylabel(
            f"{row_labels[row]}\n% of L+XL Jobs" if col == 0 else "",
            fontsize=9.5, fontweight="bold",
        )
        ax.set_xlabel(
            "Wait Time (hours)" if row == 1 else "",
            fontsize=10, fontweight="bold",
        )
        ax.tick_params(axis="both", labelsize=8.5)
        for obj in ax.get_xticklabels() + ax.get_yticklabels():
            obj.set_fontweight("bold")
        ax.grid(True, alpha=0.25, which="both", zorder=0)

    legend_handles = [
        Patch(facecolor=OTH_L,    label="Other L/XL  L"),
        Patch(facecolor=OTH_XL,   label="Other L/XL  XL"),
        Patch(facecolor="#aec7e8", label="WFP: MARS-drain jobs  L"),
        Patch(facecolor="#1f77b4", label="WFP: MARS-drain jobs  XL"),
        Patch(facecolor="#f4a0a0", label="MARS-CW: Drain  L"),
        Patch(facecolor="#c0392b", label="MARS-CW: Drain  XL"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=9.5,
               frameon=False, handletextpad=0.4, columnspacing=1.2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _load_mars_sm_seq_wait_data(results_dir, assign_group_fn, drain_plots_dir):
    """Return {('seq'|'other', 'S'|'M'): [wait_hours]} for MARS-CW.

    'seq' jobs = any S or M job that appears at position 1 or 2 in any drain
    episode (First_Run_Job or Second_Run_Job where the group is S or M).
    These are the small/medium jobs MARS pairs together right after a drain to
    fill the freed nodes — regardless of what the other position contains.
    """
    details_path = os.path.join(drain_plots_dir,
                                "table_mars_cw_drain_run_sequence_details.csv")
    seq_ids = set()
    if os.path.exists(details_path):
        with open(details_path) as f:
            for row in csv.DictReader(f):
                for pos in ("First", "Second", "Third"):
                    grp = row.get(f"{pos}_Run_Group", "").strip()
                    if grp not in ("S", "M"):
                        continue
                    jid_str = row.get(f"{pos}_Run_Job", "").strip()
                    if jid_str:
                        try:
                            seq_ids.add(int(jid_str))
                        except ValueError:
                            pass

    tag = "MARS-CW"
    events_path = os.path.join(results_dir, tag, "events.csv")
    empty = {k: [] for k in (("seq",   "S"), ("seq",   "M"),
                              ("other", "S"), ("other", "M"))}
    if not os.path.exists(events_path):
        return empty

    procs_map  = _infer_procs_map_for_results_dir(results_dir)
    submit_map = {}
    start_map  = {}

    with open(events_path) as f:
        for row in csv.DictReader(f):
            evt = row["event"]
            jid = int(row["id"])
            t   = int(row["sim_time"])
            if evt == "Submit":
                if jid not in submit_map:
                    submit_map[jid] = t
            elif evt in ("Run", "Backfill"):
                if jid not in start_map:
                    start_map[jid] = t

    data = {("seq",   "S"): [], ("seq",   "M"): [],
            ("other", "S"): [], ("other", "M"): []}
    for jid, sub_t in submit_map.items():
        if jid not in start_map:
            continue
        grp = assign_group_fn(procs_map.get(jid, -1))
        if grp not in ("S", "M"):
            continue
        wait_h = (start_map[jid] - sub_t) / 3600.0
        if wait_h < 0:
            continue
        kind = "seq" if jid in seq_ids else "other"
        data[(kind, grp)].append(wait_h)
    return data


def _load_heuristic_sm_backfill_wait_data(results_dir, assign_group_fn, tag):
    """Return {('backfill'|'primary', 'S'|'M'): [wait_hours]} for a SortPolicy.

    Uses Run vs Backfill event type — same logic as _load_heuristic_backfill_wait_data
    but restricted to S and M job sizes.
    """
    events_path = os.path.join(results_dir, tag, "events.csv")
    empty = {k: [] for k in (("backfill", "S"), ("backfill", "M"),
                              ("primary",  "S"), ("primary",  "M"))}
    if not os.path.exists(events_path):
        return empty

    procs_map    = _infer_procs_map_for_results_dir(results_dir)
    submit_map   = {}
    start_map    = {}
    backfill_ids = set()

    with open(events_path) as f:
        for row in csv.DictReader(f):
            evt = row["event"]
            jid = int(row["id"])
            t   = int(row["sim_time"])
            if evt == "Submit":
                if jid not in submit_map:
                    submit_map[jid] = t
            elif evt == "Run":
                if jid not in start_map:
                    start_map[jid] = t
            elif evt == "Backfill":
                if jid not in start_map:
                    start_map[jid] = t
                backfill_ids.add(jid)

    data = {("backfill", "S"): [], ("backfill", "M"): [],
            ("primary",  "S"): [], ("primary",  "M"): []}
    for jid, sub_t in submit_map.items():
        if jid not in start_map:
            continue
        grp = assign_group_fn(procs_map.get(jid, -1))
        if grp not in ("S", "M"):
            continue
        wait_h = (start_map[jid] - sub_t) / 3600.0
        if wait_h < 0:
            continue
        kind = "backfill" if jid in backfill_ids else "primary"
        data[(kind, grp)].append(wait_h)
    return data


def plot_sm_backfill_seq_grid_ab(
        theta_results_dir, polaris_results_dir,
        theta_drain_dir,   polaris_drain_dir,
        out_path):
    """2×3 grid: rows = Theta/Polaris, cols = WFP / MARS-CW / LRF.

    S and M jobs only:
      WFP / LRF  — primary (grey) vs backfill (colour)
      MARS-CW     — other S/M (grey) vs {S/M}-{S/M}-* sequence jobs (colour)

    Y axis: % of total S+M jobs — shared per row.
    X axis: log scale 10^-1–10^4 — shared.
    """
    from matplotlib.patches import Patch

    bins   = np.logspace(-1, 4, 41)
    PRI_S  = "#cccccc"
    PRI_M  = "#666666"

    col_colors = [
        ("#aec7e8", "#1f77b4"),
        ("#f4a0a0", "#c0392b"),
    ]
    col_titles = ["WFP", "MARS-CW"]
    row_labels = ["Theta 2021", "Polaris 2024"]

    # datasets[row*2 + col]  key tuples differ by column type
    # heuristic: ('primary'/'backfill', 'S'/'M')
    # MARS:      ('other'/'seq',        'S'/'M')
    datasets = []
    for rdir, ag_fn, dr_dir in [
            (theta_results_dir,   _assign_group_theta,   theta_drain_dir),
            (polaris_results_dir, _assign_group_polaris, polaris_drain_dir),
    ]:
        datasets.append(_load_heuristic_sm_backfill_wait_data(rdir, ag_fn, "WFP"))
        datasets.append(_load_mars_sm_seq_wait_data(rdir, ag_fn, dr_dir))

    # Key tuples for grey (non-highlight) and coloured (highlight) stacks
    PRI_KEYS = [
        [("primary", "S"), ("primary", "M")],   # WFP
        [("other",   "S"), ("other",   "M")],   # MARS
        [("primary", "S"), ("primary", "M")],   # WFP polaris
        [("other",   "S"), ("other",   "M")],   # MARS polaris
    ]
    HI_KEYS = [
        [("backfill", "S"), ("backfill", "M")],
        [("seq",      "S"), ("seq",      "M")],
        [("backfill", "S"), ("backfill", "M")],
        [("seq",      "S"), ("seq",      "M")],
    ]

    theta_total   = max(sum(len(v) for v in datasets[0].values()), 1)
    polaris_total = max(sum(len(v) for v in datasets[2].values()), 1)

    def _peak(data, pri_k, hi_k, total):
        stk = np.zeros(len(bins) - 1)
        for k in pri_k + hi_k:
            h, _ = np.histogram(data.get(k, []), bins=bins)
            stk += h
        return float((stk / total * 100).max())

    theta_ymax = max(
        _peak(datasets[i], PRI_KEYS[i], HI_KEYS[i], theta_total)
        for i in range(2)
    ) * 1.18
    polaris_ymax = max(
        _peak(datasets[i+2], PRI_KEYS[i+2], HI_KEYS[i+2], polaris_total)
        for i in range(2)
    ) * 1.18

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88,
                        bottom=0.17, wspace=0.20, hspace=0.40)

    for idx, data in enumerate(datasets):
        row, col  = divmod(idx, 2)
        ax        = axes[row][col]
        hi_S, hi_M = col_colors[col]
        total     = theta_total   if row == 0 else polaris_total
        ymax      = theta_ymax    if row == 0 else polaris_ymax
        pri_k     = PRI_KEYS[idx]
        hi_k      = HI_KEYS[idx]
        is_mcts   = (col == 1)

        h_pri_s, _ = np.histogram(data.get(pri_k[0], []), bins=bins)
        h_pri_m, _ = np.histogram(data.get(pri_k[1], []), bins=bins)
        h_hi_s,  _ = np.histogram(data.get(hi_k[0],  []), bins=bins)
        h_hi_m,  _ = np.histogram(data.get(hi_k[1],  []), bins=bins)

        pct_pri_s = h_pri_s / total * 100
        pct_pri_m = h_pri_m / total * 100
        pct_hi_s  = h_hi_s  / total * 100
        pct_hi_m  = h_hi_m  / total * 100

        lefts  = bins[:-1]
        widths = np.diff(bins)
        b0 = np.zeros(len(bins) - 1)
        b1 = b0 + pct_pri_s
        b2 = b1 + pct_pri_m
        b3 = b2 + pct_hi_s

        ax.bar(lefts, pct_pri_s, width=widths, bottom=b0, align="edge",
               color=PRI_S, zorder=3)
        ax.bar(lefts, pct_pri_m, width=widths, bottom=b1, align="edge",
               color=PRI_M, zorder=3)
        ax.bar(lefts, pct_hi_s,  width=widths, bottom=b2, align="edge",
               color=hi_S,  zorder=3)
        ax.bar(lefts, pct_hi_m,  width=widths, bottom=b3, align="edge",
               color=hi_M,  zorder=3)

        ax.set_xscale("log")
        ax.set_xlim(1e-1, 1e4)
        ax.set_ylim(0, ymax)

        n_hi    = sum(len(data.get(k, [])) for k in hi_k)
        n_panel = n_hi + sum(len(data.get(k, [])) for k in pri_k)
        pct_hi  = 100.0 * n_hi / n_panel if n_panel else 0.0
        hi_label = "Post-drain pos 1/2/3" if is_mcts else "Backfill"
        ax.set_title(f"{hi_label}: {n_hi:,}  ({pct_hi:.1f}%)",
                     fontsize=10.0, fontweight="bold", pad=3)

        if row == 0:
            ax.text(0.5, 1.28, col_titles[col], transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax.set_ylabel(
            f"{row_labels[row]}\n% of S+M Jobs" if col == 0 else "",
            fontsize=9.5, fontweight="bold",
        )
        ax.set_xlabel(
            "Wait Time (hours)" if row == 1 else "",
            fontsize=10, fontweight="bold",
        )
        ax.tick_params(axis="both", labelsize=8.5)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        ax.grid(True, alpha=0.25, which="both", zorder=0)

    legend_handles = [
        Patch(facecolor=PRI_S,    label="Primary / Other  S"),
        Patch(facecolor=PRI_M,    label="Primary / Other  M"),
        Patch(facecolor="#aec7e8", label="WFP Backfill  S"),
        Patch(facecolor="#1f77b4", label="WFP Backfill  M"),
        Patch(facecolor="#f4a0a0", label="MARS-CW post-drain pos 1/2/3  S"),
        Patch(facecolor="#c0392b", label="MARS-CW post-drain pos 1/2/3  M"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9.5,
               frameon=False, handletextpad=0.4, columnspacing=1.2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_backfill_drain_grid_ab(
        theta_results_dir, polaris_results_dir,
        theta_drain_dir,   polaris_drain_dir,
        out_path,
        log_y=False,
        size_filter=None,
        mars_tag="MARS-CW"):
    """2×3 grid: rows = system (Theta/Polaris), cols = policy (WFP, MARS-CW, LRF).

    Y axis: % of jobs in the selected size group — shared per row.
    X axis: log scale 10^-1–10^4 — shared across all panels.
    Y limits: shared within each row so policies are directly comparable.
    Grey stacks = primary / non-drain.  Coloured stacks = backfill / drain.
    log_y:       if True, Y axis is log scale clipped to 10^-1–10^1.
    size_filter: None → show both L and XL stacked; 'L' or 'XL' → single size only.
    """
    from matplotlib.patches import Patch

    bins   = np.logspace(-1, 4, 41)
    # Grey shades: L uses light, XL uses dark; both used when size_filter is None
    PRI_L  = "#cccccc"
    PRI_XL = "#666666"

    # ── load all six datasets ────────────────────────────────────────────────
    wfp3_t = _load_heuristic_backfill_wait_data(
        theta_results_dir,   _assign_group_theta,   "WFP")
    wfp3_p = _load_heuristic_backfill_wait_data(
        polaris_results_dir, _assign_group_polaris, "WFP")
    mcw_t  = _load_mars_cw_drain_wait_data(
        theta_results_dir,   theta_drain_dir,   _assign_group_theta,   mars_tag)
    mcw_p  = _load_mars_cw_drain_wait_data(
        polaris_results_dir, polaris_drain_dir, _assign_group_polaris, mars_tag)

    # ── select key tuples based on size_filter ───────────────────────────────
    sizes = [size_filter] if size_filter in ("L", "XL") else ["L", "XL"]

    def _keys(kind_map, sz_list):
        return [kind_map[s] for s in sz_list]

    H_PRI_MAP = {"L": ("primary",  "L"), "XL": ("primary",  "XL")}
    H_HI_MAP  = {"L": ("backfill", "L"), "XL": ("backfill", "XL")}
    M_PRI_MAP = {"L": ("nondrain", "L"), "XL": ("nondrain", "XL")}
    M_HI_MAP  = {"L": ("drain",    "L"), "XL": ("drain",    "XL")}

    H_PRI = _keys(H_PRI_MAP, sizes)
    H_HI  = _keys(H_HI_MAP,  sizes)
    M_PRI = _keys(M_PRI_MAP, sizes)
    M_HI  = _keys(M_HI_MAP,  sizes)

    # ── per-system totals (denominator = jobs in selected size(s)) ───────────
    def _total(data, pri_k, hi_k):
        return max(sum(len(data.get(k, [])) for k in pri_k + hi_k), 1)

    theta_total   = _total(wfp3_t, H_PRI, H_HI)
    polaris_total = _total(wfp3_p, H_PRI, H_HI)

    def _peak_pct(data, pri_keys, hi_keys, total):
        top = np.zeros(len(bins) - 1)
        for k in pri_keys + hi_keys:
            h, _ = np.histogram(data.get(k, []), bins=bins)
            top += h
        return float((top / total * 100).max())

    theta_ymax = max(
        _peak_pct(wfp3_t, H_PRI, H_HI, theta_total),
        _peak_pct(mcw_t,  M_PRI, M_HI, theta_total),
    ) * 1.18

    polaris_ymax = max(
        _peak_pct(wfp3_p, H_PRI, H_HI, polaris_total),
        _peak_pct(mcw_p,  M_PRI, M_HI, polaris_total),
    ) * 1.18

    # Per-tag hi colours for single-size panels (L and XL use same light shade)
    _MARS_HI_SINGLE = {
        "MARS-CW": {"L": "#ffe566", "XL": "#e6b800"},
        "MARS-CB": {"L": "#fad270", "XL": "#b8860b"},
        "MARS-CU": {"L": "#c5b0d5", "XL": "#9467bd"},
    }
    # Per-tag hi colours for combined L+XL panels
    _MARS_HI_PAIR = {
        "MARS-CW": ("#f4a0a0", "#c0392b"),
        "MARS-CB": ("#fad270", "#b8860b"),
        "MARS-CU": ("#c5b0d5", "#9467bd"),
    }
    mars_hi_l, mars_hi_xl = _MARS_HI_PAIR.get(mars_tag, ("#f4a0a0", "#c0392b"))

    # ── colour selection per size ─────────────────────────────────────────────
    # Grey is always light regardless of size to keep the background uncluttered
    if size_filter == "L":
        pri_grey = PRI_L
        col_hi = {"WFP": "#aec7e8",
                  mars_tag:    _MARS_HI_SINGLE.get(mars_tag, {}).get("L", "#ffe566")}
    elif size_filter == "XL":
        pri_grey = PRI_L
        col_hi = {"WFP": "#aec7e8",
                  mars_tag:    _MARS_HI_SINGLE.get(mars_tag, {}).get("XL", "#e6b800")}
    else:
        pri_grey = None   # unused in combined mode
        col_hi   = {}

    # ── panel definitions ────────────────────────────────────────────────────
    panels = [
        (wfp3_t, H_PRI, H_HI, "#aec7e8",  "#1f77b4",  theta_total,   theta_ymax,   "WFP"),
        (mcw_t,  M_PRI, M_HI, mars_hi_l,  mars_hi_xl, theta_total,   theta_ymax,   mars_tag),
        (wfp3_p, H_PRI, H_HI, "#aec7e8",  "#1f77b4",  polaris_total, polaris_ymax, "WFP"),
        (mcw_p,  M_PRI, M_HI, mars_hi_l,  mars_hi_xl, polaris_total, polaris_ymax, mars_tag),
    ]
    ROW_LABELS = ["Theta 2021", "Polaris 2024"]
    HI_LABELS  = ["Backfill", "Drain", "Backfill", "Drain"]
    size_label = f" ({size_filter})" if size_filter else ""
    y_label    = f"% of {size_filter} Jobs" if size_filter else "% of L+XL Jobs"

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88,
                        bottom=0.17, wspace=0.20, hspace=0.40)

    for idx, (data, pri_keys, hi_keys,
              hi_L, hi_XL, total, ymax, col_title) in enumerate(panels):
        row, col = divmod(idx, 2)
        ax       = axes[row][col]
        hi_label = HI_LABELS[idx]

        lefts  = bins[:-1]
        widths = np.diff(bins)

        if size_filter:
            # Single-size: one grey layer + one coloured layer
            h_pri, _ = np.histogram(data.get(pri_keys[0], []), bins=bins)
            h_hi,  _ = np.histogram(data.get(hi_keys[0],  []), bins=bins)
            pct_pri  = h_pri / total * 100
            pct_hi_v = h_hi  / total * 100
            c_pri = pri_grey
            c_hi  = col_hi.get(col_title, hi_L)
            b0 = np.zeros(len(bins) - 1)
            b1 = b0 + pct_pri
            is_mars   = col_title.startswith("MARS-")
            hi_hatch  = dict(hatch='///', edgecolor='black', linewidth=0.3) if is_mars else dict(edgecolor='none')
            ax.bar(lefts, pct_pri,  width=widths, bottom=b0, align="edge",
                   color=c_pri, zorder=3, edgecolor='none')
            ax.bar(lefts, pct_hi_v, width=widths, bottom=b1, align="edge",
                   color=c_hi,  zorder=3, **hi_hatch)
        else:
            # Combined L+XL: four layers
            h_pri_l,  _ = np.histogram(data.get(pri_keys[0], []), bins=bins)
            h_pri_xl, _ = np.histogram(data.get(pri_keys[1], []), bins=bins)
            h_hi_l,   _ = np.histogram(data.get(hi_keys[0],  []), bins=bins)
            h_hi_xl,  _ = np.histogram(data.get(hi_keys[1],  []), bins=bins)
            pct_pri_l  = h_pri_l  / total * 100
            pct_pri_xl = h_pri_xl / total * 100
            pct_hi_l   = h_hi_l   / total * 100
            pct_hi_xl  = h_hi_xl  / total * 100
            b0 = np.zeros(len(bins) - 1)
            b1 = b0 + pct_pri_l
            b2 = b1 + pct_pri_xl
            b3 = b2 + pct_hi_l
            ax.bar(lefts, pct_pri_l,  width=widths, bottom=b0, align="edge",
                   color=PRI_L,  zorder=3)
            ax.bar(lefts, pct_pri_xl, width=widths, bottom=b1, align="edge",
                   color=PRI_XL, zorder=3)
            ax.bar(lefts, pct_hi_l,   width=widths, bottom=b2, align="edge",
                   color=hi_L,   zorder=3)
            ax.bar(lefts, pct_hi_xl,  width=widths, bottom=b3, align="edge",
                   color=hi_XL,  zorder=3)

        ax.set_xscale("log")
        ax.set_xlim(1e-1, 1e4)
        if log_y:
            ax.set_yscale("log")
            ax.set_ylim(1e-1, 1e1)
        else:
            ax.set_ylim(0, ymax)

        n_hi    = sum(len(data.get(k, [])) for k in hi_keys)
        n_panel = n_hi + sum(len(data.get(k, [])) for k in pri_keys)
        pct_hi  = 100.0 * n_hi / n_panel if n_panel else 0.0
        ax.set_title(f"{hi_label}{size_label}: {n_hi:,}  ({pct_hi:.1f}%)",
                     fontsize=10.0, fontweight="bold", pad=3)

        if row == 0:
            ax.text(0.5, 1.28, col_title, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax.set_ylabel(
            f"{ROW_LABELS[row]}\n{y_label}" if col == 0 else "",
            fontsize=9.5, fontweight="bold",
        )
        ax.set_xlabel(
            "Wait Time (hours)" if row == 1 else "",
            fontsize=10, fontweight="bold",
        )
        ax.tick_params(axis="both", labelsize=8.5)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        ax.grid(True, alpha=0.25, which="both", zorder=0)

    # ── shared legend ────────────────────────────────────────────────────────
    if size_filter:
        c_wfp3 = col_hi["WFP"]
        c_mcw  = col_hi[mars_tag]
        legend_handles = [
            Patch(facecolor=pri_grey, label=f"Primary / Non-drain  {size_filter}"),
            Patch(facecolor=c_wfp3,  label=f"WFP Backfill  {size_filter}"),
            Patch(facecolor=c_mcw,   label=f"{mars_tag} Drain  {size_filter}",
                  hatch='///', edgecolor='black'),
        ]
        ncol = 3
    else:
        legend_handles = [
            Patch(facecolor=PRI_L,    label="Primary / Non-drain  L"),
            Patch(facecolor=PRI_XL,   label="Primary / Non-drain  XL"),
            Patch(facecolor="#aec7e8", label="WFP Backfill  L"),
            Patch(facecolor="#1f77b4", label="WFP Backfill  XL"),
            Patch(facecolor=mars_hi_l,  label=f"{mars_tag} Drain  L"),
            Patch(facecolor=mars_hi_xl, label=f"{mars_tag} Drain  XL"),
        ]
        ncol = 3
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=ncol, fontsize=9.5,
               frameon=False, handletextpad=0.4, columnspacing=1.2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_backfill_drain_4col_ab(
        theta_results_dir, polaris_results_dir,
        theta_drain_dir,   polaris_drain_dir,
        out_path,
        size='L'):
    """2×4 grid: rows = Theta/Polaris, cols = LRF, WFP, MARS-CU, MARS-CW.

    One figure per size class (L or XL).
    Log X, linear Y.
    Light grey = Non-Drain / Non-Backfill.  Red = Drain / Backfill.
    Denominator = total jobs of the given size class under each policy.
    """
    from matplotlib.patches import Patch

    bins  = np.logspace(-1, 4, 41)
    GREY  = "#cccccc"
    RED   = "#e74c3c"   # backfill (heuristics)
    BLUE  = "#26edff"   # drain    (MARS)

    # cols left → right: FCFS, WFP, MARS-CU, MARS-CW
    # (tag, hi_key, pri_key, col_title, hi_color)
    COLS = [
        ("FCFS",       "backfill", "primary",  "FCFS",       RED),
        ("WFP",  "backfill", "primary",  "WFP",        RED),
        ("MARS-CU",    "drain",    "nondrain", "MARS-CU",    BLUE),
        ("MARS-CW",    "drain",    "nondrain", "MARS-CW",    BLUE),
    ]

    def _load(tag, results_dir, drain_dir, assign_fn):
        if tag.startswith("MARS-"):
            return _load_mars_cw_drain_wait_data(results_dir, drain_dir, assign_fn, tag)
        return _load_heuristic_backfill_wait_data(results_dir, assign_fn, tag)

    def _select(data, hi_key, pri_key, sz):
        return (np.asarray(data.get((pri_key, sz), []), dtype=float),
                np.asarray(data.get((hi_key,  sz), []), dtype=float))

    systems = [
        ("Theta 2021",   theta_results_dir,   theta_drain_dir,   _assign_group_theta),
        ("Polaris 2024", polaris_results_dir, polaris_drain_dir, _assign_group_polaris),
    ]

    all_data = {
        (sys_label, tag): _load(tag, res_dir, dr_dir, assign_fn)
        for sys_label, res_dir, dr_dir, assign_fn in systems
        for tag, *_ in COLS
    }

    def _peak(sys_label):
        peaks = []
        for tag, hi_key, pri_key, *_ in COLS:
            pri, hi = _select(all_data[(sys_label, tag)], hi_key, pri_key, size)
            total = max(len(pri) + len(hi), 1)
            top = np.zeros(len(bins) - 1)
            for arr in (pri, hi):
                if len(arr):
                    h, _ = np.histogram(arr, bins=bins)
                    top += h
            peaks.append(float((top / total * 100).max()))
        return max(peaks, default=1) * 1.18

    row_ymaxes = {s[0]: _peak(s[0]) for s in systems}

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 6.8))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88,
                        bottom=0.20, wspace=0.18, hspace=0.40)

    lefts  = bins[:-1]
    widths = np.diff(bins)

    for r, (sys_label, *_) in enumerate(systems):
        ymax = row_ymaxes[sys_label]
        for c, (tag, hi_key, pri_key, col_title, hi_color) in enumerate(COLS):
            ax = axes[r][c]
            pri, hi = _select(all_data[(sys_label, tag)], hi_key, pri_key, size)
            total = max(len(pri) + len(hi), 1)

            h_pri, _ = np.histogram(pri, bins=bins)
            h_hi,  _ = np.histogram(hi,  bins=bins)
            pct_pri = h_pri / total * 100
            pct_hi  = h_hi  / total * 100

            ax.bar(lefts, pct_pri, width=widths, bottom=0,       align="edge",
                   color=GREY,     zorder=3, edgecolor='none')
            ax.bar(lefts, pct_hi,  width=widths, bottom=pct_pri, align="edge",
                   color=hi_color, zorder=3, edgecolor='none')

            ax.set_xscale("log")
            ax.set_xlim(1e-1, 1e4)
            ax.set_ylim(0, ymax)

            n_hi     = len(hi)
            pct_frac = 100.0 * n_hi / total
            hi_label = "Drain" if hi_color == BLUE else "Backfill"
            ax.set_title(f"{hi_label}: {n_hi:,}  ({pct_frac:.1f}%)",
                         fontsize=14, fontweight="bold", pad=3)

            if r == 0:
                ax.text(0.5, 1.28, col_title, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=16, fontweight="bold")

            ax.set_ylabel(
                f"{sys_label}\n% of {size} Jobs" if c == 0 else "",
                fontsize=15, fontweight="bold",
            )
            ax.set_xlabel(
                "Wait Time (hours)" if r == 1 else "",
                fontsize=14, fontweight="bold",
            )
            ax.tick_params(axis="both", labelsize=13)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight("bold")
            ax.grid(True, alpha=0.25, which="both", zorder=0)

    legend_handles = [
        Patch(facecolor=GREY, label="Non-Drain / Non-Backfill", edgecolor='none'),
        Patch(facecolor=RED,  label="Backfill",                 edgecolor='none'),
        Patch(facecolor=BLUE, label="Drain",                    edgecolor='none'),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=16,
               frameon=False, handletextpad=0.6, columnspacing=2.0,
               handleheight=1.6, handlelength=2.6)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: {out_path}")


_ALL_GROUPS = ("S", "M", "L", "XL")


def _load_heuristic_backfill_4grp(results_dir, assign_group_fn, tag):
    """Return {('backfill'|'primary', grp): [wait_hours]} for S/M/L/XL."""
    events_path = os.path.join(results_dir, tag, "events.csv")
    if not os.path.exists(events_path):
        return {(k, g): [] for k in ("backfill", "primary") for g in _ALL_GROUPS}

    procs_map  = _infer_procs_map_for_results_dir(results_dir)
    submit_map = {}
    start_map  = {}
    backfill_ids = set()

    with open(events_path) as f:
        for row in csv.DictReader(f):
            evt = row["event"]
            jid = int(row["id"])
            t   = int(row["sim_time"])
            if evt == "Submit":
                if jid not in submit_map:
                    submit_map[jid] = t
            elif evt == "Run":
                start_map[jid] = t
            elif evt == "Backfill":
                start_map[jid] = t
                backfill_ids.add(jid)

    data = {(k, g): [] for k in ("backfill", "primary") for g in _ALL_GROUPS}
    for jid, sub_t in submit_map.items():
        if jid not in start_map:
            continue
        grp = assign_group_fn(procs_map.get(jid, -1))
        if grp not in _ALL_GROUPS:
            continue
        wait_h = (start_map[jid] - sub_t) / 3600.0
        if wait_h < 0:
            continue
        kind = "backfill" if jid in backfill_ids else "primary"
        data[(kind, grp)].append(wait_h)
    return data


def _load_mars_drain_4grp(results_dir, drain_plots_dir, assign_group_fn, tag="MARS-CW"):
    """Return {('drain'|'nondrain', grp): [wait_hours]} for S/M/L/XL."""
    events_path = os.path.join(results_dir, tag, "events.csv")
    if not os.path.exists(events_path):
        return {(k, g): [] for k in ("drain", "nondrain") for g in _ALL_GROUPS}

    submit_map, start_map, _ = parse_events(events_path)
    procs_map = _infer_procs_map_for_results_dir(results_dir)

    drain_job_ids = set()
    slug = tag_slug(tag)
    details_path = os.path.join(drain_plots_dir,
                                f"table_{slug}_drain_run_sequence_details.csv")
    if os.path.exists(details_path):
        with open(details_path) as f:
            for row in csv.DictReader(f):
                for pos in ("First", "Second", "Third"):
                    jid_str = row.get(f"{pos}_Run_Job", "").strip()
                    if not jid_str:
                        continue
                    try:
                        drain_job_ids.add(int(jid_str))
                    except ValueError:
                        pass

    data = {(k, g): [] for k in ("drain", "nondrain") for g in _ALL_GROUPS}
    for jid, sub_t in submit_map.items():
        if jid not in start_map:
            continue
        grp = assign_group_fn(procs_map.get(jid, -1))
        if grp not in _ALL_GROUPS:
            continue
        wait_h = (start_map[jid] - sub_t) / 3600.0
        if wait_h < 0:
            continue
        kind = "drain" if jid in drain_job_ids else "nondrain"
        data[(kind, grp)].append(wait_h)
    return data


def plot_backfill_drain_grid_per_system(
        results_dir, drain_plots_dir, assign_group_fn,
        system_label, out_path, rows=None):
    """N-row × 4-col histogram + boxplot-strip grid for a single system.

    Each scheduler row has two sub-rows:
      - Histogram (tall): stacked bar chart of wait-time distribution.
      - Strip (short):    single white horizontal boxplot, shared x-axis.
    Columns: S, M, L, XL.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch

    bins   = np.logspace(-1, 4, 41)
    lefts  = bins[:-1]
    widths = np.diff(bins)
    GREY  = "#cccccc"
    RED   = "#e74c3c"
    BLUE  = "#26edff"

    if rows is None:
        rows = [
            ("WFP", "backfill", "primary",  "WFP",     RED,  _load_heuristic_backfill_4grp),
            ("MARS-CW",   "drain",    "nondrain", "MARS-CW", BLUE, _load_mars_drain_4grp),
        ]

    all_data = {}
    for tag, hi_key, pri_key, row_label, hi_color, loader in rows:
        if tag not in all_data:
            if loader is _load_heuristic_backfill_4grp:
                all_data[tag] = loader(results_dir, assign_group_fn, tag)
            else:
                all_data[tag] = loader(results_dir, drain_plots_dir, assign_group_fn, tag)

    def _row_ymax(tag, hi_key, pri_key):
        peaks = []
        for grp in _ALL_GROUPS:
            pri = np.asarray(all_data[tag].get((pri_key, grp), []), dtype=float)
            hi  = np.asarray(all_data[tag].get((hi_key,  grp), []), dtype=float)
            total = max(len(pri) + len(hi), 1)
            stacked = np.zeros(len(bins) - 1)
            for arr in (pri, hi):
                if len(arr):
                    h, _ = np.histogram(arr, bins=bins)
                    stacked += h
            peaks.append(float((stacked / total * 100).max()))
        return max(peaks, default=1) * 1.10

    row_ymaxes = [_row_ymax(tag, hi_key, pri_key)
                  for tag, hi_key, pri_key, *_ in rows]

    n_rows = len(rows)
    # Each scheduler row = 1 histogram sub-row (height 5) + 1 strip sub-row (height 1)
    height_ratios = [5, 1] * n_rows
    fig_h = (3.4 + 0.7) * n_rows + 0.6
    bottom_space = 0.11 if n_rows <= 2 else 0.08

    fig = plt.figure(figsize=(17.0, fig_h))
    gs = gridspec.GridSpec(
        n_rows * 2, 4,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.0,   # no gap within hist+strip pairs; inter-group gap via spine removal
        wspace=0.18,
        left=0.07, right=0.98, top=0.87, bottom=bottom_space,
    )

    # Build paired axes: hist_axes[r][c] and box_axes[r][c]
    hist_axes = [[fig.add_subplot(gs[r * 2,     c]) for c in range(4)] for r in range(n_rows)]
    box_axes  = [[fig.add_subplot(gs[r * 2 + 1, c],
                                  sharex=hist_axes[r][c]) for c in range(4)]
                 for r in range(n_rows)]

    last_row = n_rows - 1
    for r, (tag, hi_key, pri_key, row_label, hi_color, _) in enumerate(rows):
        ymax = row_ymaxes[r]
        for c, grp in enumerate(_ALL_GROUPS):
            ax_h = hist_axes[r][c]
            ax_b = box_axes[r][c]

            pri = np.asarray(all_data[tag].get((pri_key, grp), []), dtype=float)
            hi  = np.asarray(all_data[tag].get((hi_key,  grp), []), dtype=float)
            total = max(len(pri) + len(hi), 1)

            # --- Histogram ---
            h_pri, _ = np.histogram(pri, bins=bins)
            h_hi,  _ = np.histogram(hi,  bins=bins)
            pct_pri = h_pri / total * 100
            pct_hi  = h_hi  / total * 100

            ax_h.bar(lefts, pct_pri, width=widths, bottom=0,       align="edge",
                     color=GREY,     zorder=3, edgecolor='none')
            ax_h.bar(lefts, pct_hi,  width=widths, bottom=pct_pri, align="edge",
                     color=hi_color, zorder=3, edgecolor='none')

            ax_h.set_xscale("log")
            ax_h.set_xlim(1e-1, 1e4)
            ax_h.set_ylim(0, ymax)
            ax_h.grid(True, alpha=0.25, which="both", zorder=0)

            # Hide bottom spine and x-ticks on histogram (strip sits directly below)
            ax_h.spines['bottom'].set_visible(False)
            ax_h.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

            n_hi     = len(hi)
            pct_frac = 100.0 * n_hi / total
            kind_lbl = "Drain" if hi_color == BLUE else "Backfill"
            # Place count/percent inside the top of the histogram (avoids hspace=0 clipping)
            ax_h.text(0.5, 0.97, f"{kind_lbl}: {n_hi:,}  ({pct_frac:.1f}%)",
                      transform=ax_h.transAxes,
                      ha="center", va="top", fontsize=12, fontweight="bold")

            if r == 0:
                ax_h.text(0.5, 1.22, grp, transform=ax_h.transAxes,
                          ha="center", va="bottom", fontsize=16, fontweight="bold")

            ax_h.set_ylabel(
                f"{row_label}\n% of Job Class" if c == 0 else "",
                fontsize=14, fontweight="bold",
            )
            ax_h.tick_params(axis='y', labelsize=12)
            for lbl in ax_h.get_yticklabels():
                lbl.set_fontweight("bold")

            # --- Boxplot strip ---
            ax_b.set_ylim(0, 1)
            ax_b.set_xscale("log")
            ax_b.set_xlim(1e-1, 1e4)

            # Hide top spine (flush with histogram bottom)
            ax_b.spines['top'].set_visible(False)
            ax_b.spines['left'].set_visible(False)
            ax_b.spines['right'].set_visible(False)
            ax_b.set_yticks([])

            all_jobs = np.concatenate([pri, hi])
            if len(all_jobs) >= 5:
                ax_b.boxplot(
                    all_jobs,
                    vert=False,
                    positions=[0.5],
                    widths=[0.55],
                    patch_artist=True,
                    showfliers=False,
                    meanline=True,
                    showmeans=True,
                    boxprops=dict(facecolor="white", linewidth=1.0, zorder=5),
                    whiskerprops=dict(linewidth=1.0, color="black", zorder=5),
                    capprops=dict(linewidth=1.0, color="black", zorder=5),
                    medianprops=dict(linewidth=1.2, color="black", zorder=6),
                    meanprops=dict(linestyle=":", color="red",
                                   linewidth=1.6, zorder=7),
                    manage_ticks=False,
                )

            ax_b.tick_params(axis='x', labelsize=12)
            for lbl in ax_b.get_xticklabels():
                lbl.set_fontweight("bold")

            if r == last_row:
                ax_b.set_xlabel("Wait Time (hours)", fontsize=13, fontweight="bold")
            else:
                ax_b.tick_params(axis='x', which='both', labelbottom=False)

    fig.suptitle(system_label, fontsize=16, fontweight="bold", y=0.98)

    legend_handles = [
        Patch(facecolor=GREY, label="Non-Drain / Non-Backfill", edgecolor='none'),
        Patch(facecolor=RED,  label="Backfill",                 edgecolor='none'),
        Patch(facecolor=BLUE, label="Drain",                    edgecolor='none'),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=15,
               frameon=False, handletextpad=0.6, columnspacing=2.0,
               handleheight=1.5, handlelength=2.4)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _load_backfill_frac_by_size(results_dir, assign_group_fn, tag):
    """Return {size: (n_backfill, n_total)} for S, M, L, XL."""
    events_path = os.path.join(results_dir, tag, "events.csv")
    SIZES  = ['S', 'M', 'L', 'XL']
    bf     = {s: 0 for s in SIZES}
    total  = {s: 0 for s in SIZES}
    if not os.path.exists(events_path):
        return {s: (0, 0) for s in SIZES}
    procs_map = _infer_procs_map_for_results_dir(results_dir)
    seen_start = {}
    backfill_ids = set()
    with open(events_path) as f:
        for row in csv.DictReader(f):
            evt = row["event"]
            jid = int(row["id"])
            if evt in ("Run", "Backfill"):
                seen_start[jid] = True
                if evt == "Backfill":
                    backfill_ids.add(jid)
    for jid in seen_start:
        g = assign_group_fn(procs_map.get(jid, -1))
        if g not in SIZES:
            continue
        total[g] += 1
        if jid in backfill_ids:
            bf[g] += 1
    return {s: (bf[s], total[s]) for s in SIZES}


def plot_backfill_fraction_grouped_ab(
        theta_results_dir, polaris_results_dir,
        out_path):
    """Compact 1×2 (Theta | Polaris) grouped-bar figure for a LaTeX single column.

    Policies: FCFS, LRF, WFP.  One bar per size class (S/M/L/XL), each a
    distinct color.  Bar height = % of jobs in that size class that are backfill.
    Policy name sits below each group on the x-axis.  Legend maps colors to size
    classes and notes they represent backfill %.
    """
    from matplotlib.patches import Patch

    SIZES      = ['S', 'M', 'L', 'XL']
    SIZE_COLORS = {
        'S':  '#3498db',   # blue
        'M':  '#e67e22',   # orange
        'L':  '#2ecc71',   # green
        'XL': '#9b59b6',   # purple
    }
    TAGS = [("FCFS", "FCFS"), ("SJF-w1024", "SJF"), ("LJF-w1024", "LJF"), ("LRF-w1024", "LRF"), ("WFP", "WFP"), ("F1-w256", "F1")]

    systems = [
        ("Theta 2021",   theta_results_dir,   _assign_group_theta),
        ("Polaris 2024", polaris_results_dir, _assign_group_polaris),
    ]

    all_fracs = {
        (sys_label, tag): _load_backfill_frac_by_size(res_dir, assign_fn, tag)
        for sys_label, res_dir, assign_fn in systems
        for tag, _ in TAGS
    }

    bar_w    = 0.10
    gap      = 0.04
    grp_gap  = 0.22
    grp_span = len(SIZES) * (bar_w + gap) - gap

    xs = {}
    for gi, (tag, _) in enumerate(TAGS):
        grp_start = gi * (grp_span + grp_gap)
        for si, sz in enumerate(SIZES):
            xs[(tag, sz)] = grp_start + si * (bar_w + gap)

    grp_centers = [gi * (grp_span + grp_gap) + grp_span / 2 for gi in range(len(TAGS))]
    x_min = -bar_w * 1.5
    x_max = xs[(TAGS[-1][0], SIZES[-1])] + bar_w * 1.5

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.0), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.28,
                        wspace=0.06)

    for c, (sys_label, *_) in enumerate(systems):
        ax = axes[c]
        for tag, _ in TAGS:
            fracs = all_fracs[(sys_label, tag)]
            for sz in SIZES:
                n_bf, n_tot = fracs[sz]
                pct_bf = 100.0 * n_bf / n_tot if n_tot else 0.0
                ax.bar(xs[(tag, sz)], pct_bf, width=bar_w,
                       color=SIZE_COLORS[sz], edgecolor='none', zorder=3)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 100)
        ax.set_xticks(grp_centers)
        ax.set_xticklabels([short for _, short in TAGS],
                           fontsize=8.5, fontweight='bold')
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, axis='y', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(sys_label, fontsize=9, fontweight='bold', pad=3)

        if c == 0:
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_ylabel("% Backfill", fontsize=9, fontweight="bold")
            for lbl in ax.get_yticklabels():
                lbl.set_fontweight('bold')
        else:
            ax.tick_params(axis='y', left=False)

    legend_handles = [
        Patch(facecolor=SIZE_COLORS[sz], label=sz, edgecolor='none')
        for sz in SIZES
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.56, 0.0), ncol=4, fontsize=8.5,
               frameon=False, handletextpad=0.3, columnspacing=0.8,
               handleheight=1.0, handlelength=1.4,
               title="Backfill %", title_fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_mars_backfill_2col_ab(
        theta_results_dir, polaris_results_dir,
        out_path,
        size='L'):
    """2×2 grid: rows = Theta/Polaris, cols = MARS-CB, MARS-CW.

    Classifies L/XL jobs as backfill vs non-backfill (from events.csv Backfill event),
    one figure per size class.  Log X, linear Y.
    Light grey = Non-Backfill.  Red = Backfill.
    """
    from matplotlib.patches import Patch

    bins  = np.logspace(-1, 4, 41)
    GREY  = "#cccccc"
    RED   = "#e74c3c"

    COLS = [
        ("MARS-CB", "MARS-CB"),
        ("MARS-CW", "MARS-CW"),
    ]

    systems = [
        ("Theta 2021",   theta_results_dir,   _assign_group_theta),
        ("Polaris 2024", polaris_results_dir, _assign_group_polaris),
    ]

    all_data = {
        (sys_label, tag): _load_heuristic_backfill_wait_data(res_dir, assign_fn, tag)
        for sys_label, res_dir, assign_fn in systems
        for tag, _ in COLS
    }

    def _peak(sys_label):
        peaks = []
        for tag, _ in COLS:
            data  = all_data[(sys_label, tag)]
            pri   = np.asarray(data.get(("primary",  size), []), dtype=float)
            hi    = np.asarray(data.get(("backfill", size), []), dtype=float)
            total = max(len(pri) + len(hi), 1)
            top   = np.zeros(len(bins) - 1)
            for arr in (pri, hi):
                if len(arr):
                    h, _ = np.histogram(arr, bins=bins)
                    top += h
            peaks.append(float((top / total * 100).max()))
        return max(peaks, default=1) * 1.18

    row_ymaxes = {s[0]: _peak(s[0]) for s in systems}

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.8))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88,
                        bottom=0.20, wspace=0.18, hspace=0.40)

    lefts  = bins[:-1]
    widths = np.diff(bins)

    for r, (sys_label, res_dir, assign_fn) in enumerate(systems):
        ymax = row_ymaxes[sys_label]
        for c, (tag, col_title) in enumerate(COLS):
            ax    = axes[r][c]
            data  = all_data[(sys_label, tag)]
            pri   = np.asarray(data.get(("primary",  size), []), dtype=float)
            hi    = np.asarray(data.get(("backfill", size), []), dtype=float)
            total = max(len(pri) + len(hi), 1)

            h_pri, _ = np.histogram(pri, bins=bins)
            h_hi,  _ = np.histogram(hi,  bins=bins)
            pct_pri = h_pri / total * 100
            pct_hi  = h_hi  / total * 100

            ax.bar(lefts, pct_pri, width=widths, bottom=0,       align="edge",
                   color=GREY, zorder=3, edgecolor='none')
            ax.bar(lefts, pct_hi,  width=widths, bottom=pct_pri, align="edge",
                   color=RED,  zorder=3, edgecolor='none')

            ax.set_xscale("log")
            ax.set_xlim(1e-1, 1e4)
            ax.set_ylim(0, ymax)

            n_hi     = len(hi)
            pct_frac = 100.0 * n_hi / total
            ax.set_title(f"Backfill: {n_hi:,}  ({pct_frac:.1f}%)",
                         fontsize=14, fontweight="bold", pad=3)

            if r == 0:
                ax.text(0.5, 1.28, col_title, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=16, fontweight="bold")

            ax.set_ylabel(
                f"{sys_label}\n% of {size} Jobs" if c == 0 else "",
                fontsize=15, fontweight="bold",
            )
            ax.set_xlabel(
                "Wait Time (hours)" if r == 1 else "",
                fontsize=14, fontweight="bold",
            )
            ax.tick_params(axis="both", labelsize=13)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight("bold")
            ax.grid(True, alpha=0.25, which="both", zorder=0)

    legend_handles = [
        Patch(facecolor=GREY, label="Non-Backfill", edgecolor='none'),
        Patch(facecolor=RED,  label="Backfill",     edgecolor='none'),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=16,
               frameon=False, handletextpad=0.6, columnspacing=2.0,
               handleheight=1.6, handlelength=2.6)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Create combined exp2a/exp2b overall plots.")
    parser.add_argument("--exp2a-plots", default="results/exp2a_plots")
    parser.add_argument("--exp2b-plots", default="results/exp2b_plots")
    parser.add_argument("--output-dir", default="results/exp2ab_plots")
    args = parser.parse_args()

    plot_combined_overall(
        os.path.join(args.exp2a_plots, "wait", "table_wait.csv"),
        os.path.join(args.exp2b_plots, "wait", "table_wait.csv"),
        os.path.join(args.output_dir, "wait", "bar_chart_wait.png"),
        "Wait Time",
    )
    plot_combined_overall(
        os.path.join(args.exp2a_plots, "bsld", "table_bsld.csv"),
        os.path.join(args.exp2b_plots, "bsld", "table_bsld.csv"),
        os.path.join(args.output_dir, "bsld", "bar_chart_bsld.png"),
        "Bounded Slowdown",
    )
    plot_combined_wait_boxplot(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "wait", "boxplot_wait_wfp3_mars_cu_cw.png"),
        tag_prefixes=("WFP", "MARS-CW", "MARS-CU"),
    )
    plot_combined_wait_boxplot_all(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "wait", "boxplot_wait_all.png"),
    )
    plot_combined_wait_boxplot_groups(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "wait", "boxplot_wait_groups.png"),
    )
    plot_single_system_wait_kde_groups(
        _results_dir_from_plots_dir(args.exp2a_plots),
        os.path.join(args.exp2a_plots, "wait", "kde_wait_groups_selected_policies.png"),
        _assign_group_theta,
        fig_width=8.0,
    )
    plot_single_system_wait_kde_groups(
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2b_plots, "wait", "kde_wait_groups_selected_policies.png"),
        _assign_group_polaris,
        fig_width=8.0,
    )
    plot_combined_wait_kde_groups(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "wait", "kde_wait_groups_selected_policies.png"),
    )
    plot_single_system_wait_kde_groups(
        _results_dir_from_plots_dir(args.exp2a_plots),
        os.path.join(args.exp2a_plots, "wait", "kde_wait_groups_selected_policies_logy.png"),
        _assign_group_theta,
        fig_width=8.0,
        log_y=True,
    )
    plot_single_system_wait_kde_groups(
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2b_plots, "wait", "kde_wait_groups_selected_policies_logy.png"),
        _assign_group_polaris,
        fig_width=8.0,
        log_y=True,
    )
    plot_combined_wait_kde_groups(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "wait", "kde_wait_groups_selected_policies_logy.png"),
        log_y=True,
    )
    plot_single_system_wait_cdf_groups(
        _results_dir_from_plots_dir(args.exp2a_plots),
        os.path.join(args.exp2a_plots, "wait", "cdf_wait_groups_selected_policies.png"),
        _assign_group_theta,
        fig_width=8.0,
    )
    plot_single_system_wait_cdf_groups(
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2b_plots, "wait", "cdf_wait_groups_selected_policies.png"),
        _assign_group_polaris,
        fig_width=8.0,
    )
    plot_combined_wait_cdf_groups(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "wait", "cdf_wait_groups_selected_policies.png"),
    )
    plot_single_system_wait_cdf_groups(
        _results_dir_from_plots_dir(args.exp2a_plots),
        os.path.join(args.exp2a_plots, "wait", "cdf_wait_groups_selected_policies_logy.png"),
        _assign_group_theta,
        fig_width=8.0,
        log_y=True,
    )
    plot_single_system_wait_cdf_groups(
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2b_plots, "wait", "cdf_wait_groups_selected_policies_logy.png"),
        _assign_group_polaris,
        fig_width=8.0,
        log_y=True,
    )
    plot_combined_wait_cdf_groups(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "wait", "cdf_wait_groups_selected_policies_logy.png"),
        log_y=True,
    )
    plot_combined_tail_wait_job_size_cdf(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "wait", "cdf_tail_wait_job_size.png"),
        excluded_tags={"MARS-IU", "F1", "SJF", "MARS-CB", "FCFS"},
    )
    plot_combined_bsld_boxplot(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "bsld", "boxplot_bsld_wfp3_mars_cu_cw.png"),
        tag_prefixes=("WFP", "MARS-CW", "MARS-CU"),
    )
    plot_combined_bsld_boxplot_all(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "bsld", "boxplot_bsld_all.png"),
    )
    plot_combined_bsld_boxplot_groups(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "bsld", "boxplot_bsld_groups.png"),
    )
    plot_combined_metric_groups(
        args.exp2a_plots, args.exp2b_plots,
        "wait", "Wait Time",
        os.path.join(args.output_dir, "wait", "bar_chart_wait_groups_combined.png"),
    )
    plot_combined_metric_groups(
        args.exp2a_plots, args.exp2b_plots,
        "bsld", "Bounded Slowdown",
        os.path.join(args.output_dir, "bsld", "bar_chart_bsld_groups_combined.png"),
    )
    plot_combined_drain_util(
        os.path.join(args.exp2a_plots, "util", "table_util_drain_2day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_util_drain_2day.csv"),
        os.path.join(args.output_dir, "util", "bar_chart_util_drain_2day.png"),
    )
    plot_combined_drain_queue_before_announcement(
        os.path.join(args.exp2a_plots, "util", "table_queue_drain_2day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_queue_drain_2day.csv"),
        os.path.join(args.output_dir, "util", "bar_chart_queue_before_announcement_drain_2day.png"),
    )
    plot_combined_drain_queue_boxplots(
        os.path.join(args.exp2a_plots, "util", "table_util_drain_2day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_util_drain_2day.csv"),
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "util", "box_queue_size_drain_2day.png"),
    )
    plot_combined_drain_util_and_queue(
        os.path.join(args.exp2a_plots, "util", "table_util_drain_2day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_util_drain_2day.csv"),
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "util", "combined_util_queue_drain_2day.png"),
    )
    plot_combined_drain_util_and_queue_before_announcement(
        os.path.join(args.exp2a_plots, "util", "table_util_drain_2day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_util_drain_2day.csv"),
        os.path.join(args.exp2a_plots, "util", "table_queue_drain_2day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_queue_drain_2day.csv"),
        os.path.join(args.output_dir, "util", "combined_util_queue_before_announcement_drain_2day.png"),
    )
    plot_combined_drain_core_hours(
        os.path.join(args.exp2a_plots, "util", "table_util_drain_2day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_util_drain_2day.csv"),
        os.path.join(args.exp2a_plots, "util", "table_core_hours_drain_2day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_core_hours_drain_2day.csv"),
        os.path.join(args.output_dir, "util", "combined_core_hours_drain_2day.png"),
    )
    plot_combined_drain_util_and_pre_announcement_core_hours(
        os.path.join(args.exp2a_plots, "util", "table_util_drain_2day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_util_drain_2day.csv"),
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "util", "combined_util_pre_announcement_core_hours_drain_2day.png"),
    )

    # ── 1-day drain equivalents ────────────────────────────────────────────────
    plot_combined_drain_util(
        os.path.join(args.exp2a_plots, "util", "table_util_drain_1day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_util_drain_1day.csv"),
        os.path.join(args.output_dir, "util", "bar_chart_util_drain_1day.png"),
        extra_excluded={"LRF-w1024"},
    )
    plot_combined_uptime_util_bar(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "util", "bar_chart_util_uptime.png"),
    )
    plot_combined_uptime_util_boxplot(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "util", "boxplot_util_uptime.png"),
        theta_util_csv=os.path.join(args.exp2a_plots, "util", "table_util.csv"),
        polaris_util_csv=os.path.join(args.exp2b_plots, "util", "table_util.csv"),
    )
    plot_combined_drain_1day_util_boxplot(
        os.path.join(args.exp2a_plots, "util", "table_util_drain_1day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_util_drain_1day.csv"),
        os.path.join(args.output_dir, "util", "boxplot_util_drain_1day.png"),
        theta_util_csv=os.path.join(args.exp2a_plots, "util", "table_util.csv"),
        polaris_util_csv=os.path.join(args.exp2b_plots, "util", "table_util.csv"),
    )
    plot_combined_drain_queue_before_announcement(
        os.path.join(args.exp2a_plots, "util", "table_queue_drain_1day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_queue_drain_1day.csv"),
        os.path.join(args.output_dir, "util", "bar_chart_queue_before_announcement_drain_1day.png"),
    )
    plot_combined_drain_queue_boxplots(
        os.path.join(args.exp2a_plots, "util", "table_util_drain_1day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_util_drain_1day.csv"),
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "util", "box_queue_size_drain_1day.png"),
    )
    plot_combined_drain_util_and_queue(
        os.path.join(args.exp2a_plots, "util", "table_util_drain_1day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_util_drain_1day.csv"),
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "util", "combined_util_queue_drain_1day.png"),
    )
    plot_combined_drain_util_and_queue_before_announcement(
        os.path.join(args.exp2a_plots, "util", "table_util_drain_1day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_util_drain_1day.csv"),
        os.path.join(args.exp2a_plots, "util", "table_queue_drain_1day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_queue_drain_1day.csv"),
        os.path.join(args.output_dir, "util", "combined_util_queue_before_announcement_drain_1day.png"),
    )
    plot_combined_drain_core_hours(
        os.path.join(args.exp2a_plots, "util", "table_util_drain_1day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_util_drain_1day.csv"),
        os.path.join(args.exp2a_plots, "util", "table_core_hours_drain_1day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_core_hours_drain_1day.csv"),
        os.path.join(args.output_dir, "util", "combined_core_hours_drain_1day.png"),
    )
    plot_combined_drain_util_and_pre_announcement_core_hours(
        os.path.join(args.exp2a_plots, "util", "table_util_drain_1day.csv"),
        os.path.join(args.exp2b_plots, "util", "table_util_drain_1day.csv"),
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "util", "combined_util_pre_announcement_core_hours_drain_1day.png"),
    )
    plot_combined_drain_sequence_combos_ab(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2a_plots, "drain"),
        os.path.join(args.exp2b_plots, "drain"),
        os.path.join(args.output_dir, "drain", "drain_sequence_combos_ab.png"),
    )
    plot_combined_drain_overview_ab(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2a_plots, "drain"),
        os.path.join(args.exp2b_plots, "drain"),
        os.path.join(args.output_dir, "drain", "drain_overview_combined_ab.png"),
    )
    plot_combined_drain_wait_stacked_hist_mars_cw(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2a_plots, "drain"),
        os.path.join(args.exp2b_plots, "drain"),
        os.path.join(args.output_dir, "drain", "drain_wait_stacked_hist_mars_cw_ab.png"),
    )
    plot_combined_drain_wait_stacked_hist_mars_cw(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2a_plots, "drain"),
        os.path.join(args.exp2b_plots, "drain"),
        os.path.join(args.output_dir, "drain", "drain_wait_stacked_hist_mars_cw_ab_linear_y.png"),
        log_y=False,
    )
    plot_combined_drain_wait_stacked_hist_mars_cw(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2a_plots, "drain"),
        os.path.join(args.exp2b_plots, "drain"),
        os.path.join(args.output_dir, "drain", "drain_wait_stacked_hist_mars_cw_ab_linear_xy.png"),
        log_x=False, log_y=False,
    )
    plot_combined_drain_overview_wfp3_ab(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "drain", "drain_overview_wfp3_ab.png"),
    )
    plot_combined_drain_sequence_combos_wfp3_ab(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "drain", "drain_sequence_combos_wfp3_ab.png"),
    )
    # WFP stacked wait histogram — primary (grey) vs backfill (blue)
    _theta_rd   = _results_dir_from_plots_dir(args.exp2a_plots)
    _polaris_rd = _results_dir_from_plots_dir(args.exp2b_plots)
    _h_ymax_theta, _h_ymax_polaris = _heuristic_hist_ymax(
        _theta_rd, _polaris_rd, tags=["WFP"])
    plot_combined_drain_wait_stacked_hist_heuristic_ab(
        _theta_rd, _polaris_rd,
        tag="WFP",
        d_color_L="#aec7e8",   # light blue
        d_color_XL="#1f77b4",  # dark blue
        out_path=os.path.join(args.output_dir, "drain", "drain_wait_stacked_hist_wfp3_ab.png"),
        theta_ymax=_h_ymax_theta, polaris_ymax=_h_ymax_polaris,
    )
    # Combined 2×3 grid: WFP / MARS-CW / LRF × Theta / Polaris
    plot_backfill_drain_4col_ab(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2a_plots, "drain"),
        os.path.join(args.exp2b_plots, "drain"),
        os.path.join(args.output_dir, "drain", "backfill_drain_grid_ab_mars_cu_L.png"),
        size='L',
    )
    plot_backfill_drain_4col_ab(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2a_plots, "drain"),
        os.path.join(args.exp2b_plots, "drain"),
        os.path.join(args.output_dir, "drain", "backfill_drain_grid_ab_mars_cu_XL.png"),
        size='XL',
    )
    plot_mars_backfill_2col_ab(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "drain", "mars_backfill_2col_ab_L.png"),
        size='L',
    )
    plot_mars_backfill_2col_ab(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "drain", "mars_backfill_2col_ab_XL.png"),
        size='XL',
    )
    plot_backfill_fraction_grouped_ab(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "drain", "backfill_fraction_grouped_ab.png"),
    )
    # S+M jobs split by whether they ran during a L/XL drain-trigger window
    plot_sm_post_drain_grid_ab(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "drain", "sm_post_drain_grid_ab.png"),
    )
    # MARS drain L/XL jobs highlighted across WFP / MARS-CW / LRF
    plot_mars_drain_jobs_cross_policy_grid_ab(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2a_plots, "drain"),
        os.path.join(args.exp2b_plots, "drain"),
        os.path.join(args.output_dir, "drain", "mars_drain_jobs_cross_policy_grid_ab.png"),
    )
    # S+M: WFP/LRF backfill vs primary; MARS {S/M}-{S/M}-* seq vs other
    plot_sm_backfill_seq_grid_ab(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2a_plots, "drain"),
        os.path.join(args.exp2b_plots, "drain"),
        os.path.join(args.output_dir, "drain", "sm_backfill_seq_grid_ab.png"),
    )
    # Cross-policy: MARS post-drain S/M job IDs highlighted in WFP and LRF
    plot_mars_drain_sm_jobs_cross_policy_grid_ab(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2a_plots, "drain"),
        os.path.join(args.exp2b_plots, "drain"),
        os.path.join(args.output_dir, "drain", "mars_drain_sm_jobs_cross_policy_grid_ab.png"),
    )

    _3ROW = lambda RED, BLUE: [
        ("WFP", "backfill", "primary",  "WFP",     RED,  _load_heuristic_backfill_4grp),
        ("MARS-CU",   "drain",    "nondrain", "MARS-CU", BLUE, _load_mars_drain_4grp),
        ("MARS-CW",   "drain",    "nondrain", "MARS-CW", BLUE, _load_mars_drain_4grp),
    ]
    # Per-system 2-row × 4-col: WFP backfill/primary + MARS-CW drain/nondrain
    plot_backfill_drain_grid_per_system(
        _results_dir_from_plots_dir(args.exp2a_plots),
        os.path.join(args.exp2a_plots, "drain"),
        _assign_group_theta,
        "Theta 2021",
        os.path.join(args.output_dir, "drain", "backfill_drain_grid_wfp3_marscw_theta.png"),
    )
    plot_backfill_drain_grid_per_system(
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2b_plots, "drain"),
        _assign_group_polaris,
        "Polaris 2024",
        os.path.join(args.output_dir, "drain", "backfill_drain_grid_wfp3_marscw_polaris.png"),
    )
    # Per-system 3-row × 4-col: WFP / MARS-CU / MARS-CW
    _RED  = "#e74c3c"
    _BLUE = "#26edff"
    plot_backfill_drain_grid_per_system(
        _results_dir_from_plots_dir(args.exp2a_plots),
        os.path.join(args.exp2a_plots, "drain"),
        _assign_group_theta,
        "Theta 2021",
        os.path.join(args.output_dir, "drain", "backfill_drain_grid_wfp3_marscu_marscw_theta.png"),
        rows=_3ROW(_RED, _BLUE),
    )
    plot_backfill_drain_grid_per_system(
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.exp2b_plots, "drain"),
        _assign_group_polaris,
        "Polaris 2024",
        os.path.join(args.output_dir, "drain", "backfill_drain_grid_wfp3_marscu_marscw_polaris.png"),
        rows=_3ROW(_RED, _BLUE),
    )

    _drain_resume_csv = os.path.join(args.output_dir, "drain", "drain_resume_times.csv")
    write_drain_resume_times_csv(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        _drain_resume_csv,
    )
    plot_drain_resume_time_boxplot(
        _drain_resume_csv,
        os.path.join(args.output_dir, "drain", "drain_resume_time_boxplot.png"),
    )

    plot_mcts_iterations_branching_ab(
        _results_dir_from_plots_dir(args.exp2a_plots),
        _results_dir_from_plots_dir(args.exp2b_plots),
        os.path.join(args.output_dir, "mcts", "mcts_iterations_branching_ab.png"),
    )


# ── MCTS performance figures ──────────────────────────────────────────────────

_MCTS_NUM_CORES = 250  # parallel trees used in all MARS runs

def _load_mcts_perf_data(results_dir, tags=("MARS-CW", "MARS-CU")):
    """Return {tag: {"iterations": [...], "branching": [...]}} for MCTS cycles only.

    iterations: per-tree count (total / _MCTS_NUM_CORES) for each MCTS cycle.
    branching:  root branching factor from descisions.csv, filtered to MCTS cycles.
    """
    data = {}
    if not os.path.isdir(results_dir):
        return data
    for tag in tags:
        perf_path = os.path.join(results_dir, tag, "performance.csv")
        dec_path  = os.path.join(results_dir, tag, "descisions.csv")
        if not os.path.exists(perf_path):
            continue

        bf_map = {}
        if os.path.exists(dec_path):
            with open(dec_path) as f:
                for row in csv.DictReader(f):
                    cycle = int(row["cycle"])
                    bf_map[cycle] = int(row.get("root_branching_factor", 0))

        iterations  = []
        branching = []
        with open(perf_path) as f:
            for row in csv.DictReader(f):
                cycle = int(row["cycle"])
                total = int(row.get("mcts_iterations", 0))
                bf = bf_map.get(cycle, 0)
                if total > 0 and bf > 2:
                    iterations.append(total / _MCTS_NUM_CORES)
                    branching.append(bf)

        data[tag] = {"iterations": iterations, "branching": branching}
    return data


def plot_mcts_iterations_branching_ab(
        theta_results_dir, polaris_results_dir, out_path,
        tags=("MARS-CW", "MARS-CU")):
    """2×2 figure: rows = iterations / branching-factor CDF, cols = Theta / Polaris.

    Top row   : box plots of per-tree MCTS iterations per scheduling cycle (log scale).
                Shared y-axis across both systems.
    Bottom row: empirical CDF of root branching factor.
                Shared y-axis across both systems.
    """
    theta_data   = _load_mcts_perf_data(theta_results_dir,   tags)
    polaris_data = _load_mcts_perf_data(polaris_results_dir, tags)

    if not theta_data and not polaris_data:
        print("  [mcts perf] no performance.csv / descisions.csv found — skipping")
        return

    TAG_COLORS   = {"MARS-CW": "#f6de6a", "MARS-CU": "#9467bd"}
    TAG_HATCHES  = {"MARS-CW": "///",     "MARS-CU": "---"}

    fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharey="row")
    fig.subplots_adjust(hspace=0.50, wspace=0.15)

    for col, (sys_data, sys_label) in enumerate(
            zip([theta_data, polaris_data], ["Theta 2021", "Polaris 2024"])):

        ax_iter = axes[0][col]
        ax_bf   = axes[1][col]

        # ── Top: per-tree iterations box plot (log y) ─────────────────────────
        boxes, xlabels, present_tags = [], [], []
        for tag in tags:
            iters = sys_data.get(tag, {}).get("iterations", [])
            if iters:
                boxes.append(iters)
                xlabels.append(tag)
                present_tags.append(tag)

        if boxes:
            bp = ax_iter.boxplot(
                boxes, patch_artist=True, widths=0.5,
                medianprops=dict(color="black", linewidth=1.5),
                showfliers=False,
            )
            for patch, tag in zip(bp["boxes"], present_tags):
                patch.set_facecolor(TAG_COLORS.get(tag, "#888888"))
                patch.set_hatch(TAG_HATCHES.get(tag, ""))
                patch.set_edgecolor("black")
                patch.set_alpha(0.85)
            ax_iter.set_xticks(range(1, len(xlabels) + 1))
            ax_iter.set_xticklabels(xlabels, fontsize=9)

        ax_iter.set_title(sys_label, fontsize=10, fontweight="bold")
        ax_iter.set_yscale("log")
        ax_iter.grid(True, axis="y", which="both", alpha=0.3)
        if col == 0:
            ax_iter.set_ylabel("Iterations / Tree / Cycle", fontsize=9)

        # ── Bottom: branching factor CDF ──────────────────────────────────────
        any_bf = False
        for tag in tags:
            bf_vals = sys_data.get(tag, {}).get("branching", [])
            if not bf_vals:
                continue
            any_bf = True
            sorted_bf = np.sort(bf_vals)
            cdf = np.arange(1, len(sorted_bf) + 1) / len(sorted_bf)
            ax_bf.plot(sorted_bf, cdf,
                       color=TAG_COLORS.get(tag, "#888888"),
                       linestyle="--" if TAG_HATCHES.get(tag) == "---" else "-",
                       label=tag, linewidth=1.8)

        if any_bf:
            ax_bf.legend(fontsize=8, framealpha=0.8)

        ax_bf.set_xlabel("Root Branching Factor", fontsize=9)
        ax_bf.set_xlim(0, 50)
        ax_bf.set_ylim(0, 1.05)
        ax_bf.grid(True, alpha=0.3)
        if col == 0:
            ax_bf.set_ylabel("CDF", fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _detect_wfp3_drain_episodes(results_dir, assign_group_fn, tag="WFP"):
    """Detect WFP drain episodes from performance/descisions/events CSVs.

    A drain episode = consecutive DRAIN/BACKFILL cycles (queue > 0) ending when
    the head L/XL job is finally scheduled (WFP or WFP+BACKFILL cycle).

    Returns (episodes, n_eligible) where:
      episodes   – list of dicts: drain_start, drain_end, trigger_group, seq (3 groups)
      n_eligible – WFP + WFP+BACKFILL cycle count (denominator for drain rate)
    """
    perf_path = os.path.join(results_dir, tag, "performance.csv")
    dec_path  = os.path.join(results_dir, tag, "descisions.csv")
    evt_path  = os.path.join(results_dir, tag, "events.csv")

    if not all(os.path.exists(p) for p in (perf_path, dec_path, evt_path)):
        return [], 0

    procs_map = _infer_procs_map_for_results_dir(results_dir)

    perf_rows = {}  # cycle -> (policy, queue_len, sim_time)
    with open(perf_path) as f:
        for row in csv.DictReader(f):
            c = int(row["cycle"])
            perf_rows[c] = (row["selected_policy"], int(row["queue_len"]), int(row["sim_time"]))

    head_job_at = {}  # cycle -> job_id
    with open(dec_path) as f:
        for row in csv.DictReader(f):
            c = int(row["cycle"])
            ps = row.get("possible_job_sets", "").strip('"').strip()
            if ps.startswith("|") and ps.endswith("|"):
                try:
                    head_job_at[c] = int(ps[1:-1])
                except ValueError:
                    pass

    run_events = []  # (sim_time, job_id) — primary runs only
    with open(evt_path) as f:
        for row in csv.DictReader(f):
            if row["event"] == "Run":
                run_events.append((int(row["sim_time"]), int(row["id"])))
    run_events.sort()

    # Derive the policy name label from the tag (e.g. "WFP" → "WFP", "LRF-w1024" → "LRF")
    # and accept both plain and "+BACKFILL" variants as "ran a primary job"
    _base = tag.split("-")[0].upper()
    DRAIN_POL = {"DRAIN", "BACKFILL"}
    RUN_POL   = {_base, _base + "+BACKFILL"}

    episodes   = []
    n_eligible = 0
    in_drain   = False
    drain_start = None

    for c in sorted(perf_rows.keys()):
        policy, q_len, sim_t = perf_rows[c]

        if policy in RUN_POL:
            n_eligible += 1

        if policy in DRAIN_POL and q_len > 0:
            if not in_drain:
                in_drain    = True
                drain_start = c
        elif policy in RUN_POL:
            if in_drain:
                hj = head_job_at.get(c, -1)
                hg = None
                if hj >= 0:
                    hg = assign_group_fn(procs_map.get(hj, -1))
                if hg in ("L", "XL"):
                    idx = bisect.bisect_left(run_events, (sim_t,))
                    seq = []
                    for ev_t, ev_j in run_events[idx:]:
                        g = assign_group_fn(procs_map.get(ev_j, -1))
                        if g is not None:
                            seq.append(g)
                        if len(seq) >= 3:
                            break
                    episodes.append({
                        "drain_start":   drain_start,
                        "drain_end":     c,
                        "trigger_job":   hj,
                        "trigger_group": hg,
                        "seq":           seq,
                    })
            in_drain = False
        else:  # NONE
            in_drain = False

    return episodes, n_eligible


def _wfp3_drain_stats(results_dir, assign_group_fn, tag="WFP"):
    """Return (drain_pct_tuple, postdrain_group, total_group, seq_counts) for WFP."""
    GROUPS = ["S", "M", "L", "XL"]
    episodes, n_eligible = _detect_wfp3_drain_episodes(results_dir, assign_group_fn, tag)

    n_drain = len(episodes)
    pct     = 100.0 * n_drain / n_eligible if n_eligible else 0.0
    drain_pct_tuple = (n_drain, n_eligible, pct)

    postdrain_group = {g: 0 for g in GROUPS}
    seq_counts      = {}

    seen_trigger = set()
    for ep in episodes:
        tg = ep["trigger_group"]
        seq = ep["seq"]
        jkey = (ep["drain_end"], tg)
        if jkey not in seen_trigger:
            seen_trigger.add(jkey)
            if tg in postdrain_group:
                postdrain_group[tg] += 1
        for g in seq[1:3]:
            if g in postdrain_group:
                postdrain_group[g] += 1

        if len(seq) >= 3:
            key = "-".join(seq[:3])
            seq_counts[key] = seq_counts.get(key, 0) + 1

    procs_map = _infer_procs_map_for_results_dir(results_dir)
    evt_path  = os.path.join(results_dir, tag, "events.csv")
    total_group = {g: 0 for g in GROUPS}
    if os.path.exists(evt_path):
        _, start_map, _ = parse_events(evt_path)
        for jid in start_map:
            g = assign_group_fn(procs_map.get(jid, -1))
            if g in total_group:
                total_group[g] += 1

    return drain_pct_tuple, postdrain_group, total_group, seq_counts


def _load_heuristic_backfill_wait_data(results_dir, assign_group_fn, tag):
    """Return {('backfill'|'primary', 'L'|'XL'): [wait_hours]} for a SortPolicy heuristic.

    Classification uses the Run vs Backfill event type recorded in events.csv.
    Backfill = the job ran while a higher-priority job was blocked at the head.
    Primary  = the job ran as the head of the sorted window.
    """
    events_path = os.path.join(results_dir, tag, "events.csv")
    if not os.path.exists(events_path):
        return {k: [] for k in (("backfill", "L"), ("backfill", "XL"),
                                ("primary",  "L"), ("primary",  "XL"))}

    procs_map = _infer_procs_map_for_results_dir(results_dir)

    submit_map   = {}  # job_id → submit_time
    start_map    = {}  # job_id → start_time
    backfill_ids = set()

    with open(events_path) as f:
        for row in csv.DictReader(f):
            evt = row["event"]
            jid = int(row["id"])
            t   = int(row["sim_time"])
            if evt == "Submit":
                if jid not in submit_map:
                    submit_map[jid] = t
            elif evt == "Run":
                start_map[jid] = t
            elif evt == "Backfill":
                start_map[jid] = t
                backfill_ids.add(jid)

    data = {("backfill", "L"): [], ("backfill", "XL"): [],
            ("primary",  "L"): [], ("primary",  "XL"): []}
    for jid, sub_t in submit_map.items():
        if jid not in start_map:
            continue
        grp = assign_group_fn(procs_map.get(jid, -1))
        if grp not in ("L", "XL"):
            continue
        wait_h = (start_map[jid] - sub_t) / 3600.0
        if wait_h < 0:
            continue
        kind = "backfill" if jid in backfill_ids else "primary"
        data[(kind, grp)].append(wait_h)
    return data


def _heuristic_hist_ymax(theta_results_dir, polaris_results_dir, tags):
    """Pre-compute the peak normalized bar height (% of L+XL) across all tags/systems.

    Returns (theta_ymax, polaris_ymax) rounded up to the next nice boundary.
    """
    bins = np.logspace(-1, 4, 41)

    def _peak(results_dir, assign_fn):
        peak = 0.0
        for tag in tags:
            d = _load_heuristic_backfill_wait_data(results_dir, assign_fn, tag)
            total = max(sum(len(v) for v in d.values()), 1)
            for key in d:
                h, _ = np.histogram(d[key], bins=bins)
                # worst-case stacked height at any single bin
                pass
            # compute stacked top at each bin
            pri_l,  _ = np.histogram(d[("primary",  "L")],  bins=bins)
            pri_xl, _ = np.histogram(d[("primary",  "XL")], bins=bins)
            bf_l,   _ = np.histogram(d[("backfill", "L")],  bins=bins)
            bf_xl,  _ = np.histogram(d[("backfill", "XL")], bins=bins)
            top = (pri_l + pri_xl + bf_l + bf_xl) / total * 100
            peak = max(peak, float(top.max()))
        return peak

    def _nice(v):
        import math
        if v <= 0:
            return 5.0
        mag = 10 ** math.floor(math.log10(v))
        return math.ceil(v / mag) * mag * 1.15

    return (_nice(_peak(theta_results_dir,   _assign_group_theta)),
            _nice(_peak(polaris_results_dir, _assign_group_polaris)))


def plot_combined_drain_wait_stacked_hist_heuristic_ab(
        theta_results_dir, polaris_results_dir,
        tag, d_color_L, d_color_XL,
        out_path,
        theta_ymax=None, polaris_ymax=None):
    """Stacked wait-time histogram for a SortPolicy heuristic (Theta left, Polaris right).

    X axis: log scale 10^-1–10^4.
    Y axis: linear, normalized as % of all L+XL jobs (same denominator across policies).
    Grey = primary (head-of-queue ran normally); colored = backfill (ran during drain).

    Pass theta_ymax / polaris_ymax to share Y limits across multiple policy figures.
    """
    from matplotlib.patches import Patch

    theta_data   = _load_heuristic_backfill_wait_data(
        theta_results_dir,   _assign_group_theta,   tag)
    polaris_data = _load_heuristic_backfill_wait_data(
        polaris_results_dir, _assign_group_polaris, tag)

    bins   = np.logspace(-1, 4, 41)
    PRI_L  = "#cccccc"
    PRI_XL = "#666666"

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.22, wspace=0.30)

    for ax, data, sys_label, forced_ymax in [
        (axes[0], theta_data,   "Theta 2021",   theta_ymax),
        (axes[1], polaris_data, "Polaris 2024", polaris_ymax),
    ]:
        total = max(sum(len(v) for v in data.values()), 1)

        pri_l,  _ = np.histogram(data[("primary",  "L")],  bins=bins)
        pri_xl, _ = np.histogram(data[("primary",  "XL")], bins=bins)
        bf_l,   _ = np.histogram(data[("backfill", "L")],  bins=bins)
        bf_xl,  _ = np.histogram(data[("backfill", "XL")], bins=bins)

        # Normalize each bin count to % of all L+XL jobs
        pct_pri_l  = pri_l  / total * 100
        pct_pri_xl = pri_xl / total * 100
        pct_bf_l   = bf_l   / total * 100
        pct_bf_xl  = bf_xl  / total * 100

        lefts  = bins[:-1]
        widths = np.diff(bins)
        b0 = np.zeros(len(bins) - 1)
        b1 = b0 + pct_pri_l
        b2 = b1 + pct_pri_xl
        b3 = b2 + pct_bf_l

        ax.bar(lefts, pct_pri_l,  width=widths, bottom=b0, align="edge", color=PRI_L,      zorder=3)
        ax.bar(lefts, pct_pri_xl, width=widths, bottom=b1, align="edge", color=PRI_XL,     zorder=3)
        ax.bar(lefts, pct_bf_l,   width=widths, bottom=b2, align="edge", color=d_color_L,  zorder=3)
        ax.bar(lefts, pct_bf_xl,  width=widths, bottom=b3, align="edge", color=d_color_XL, zorder=3)

        ax.set_xscale("log")
        ax.set_xlim(1e-1, 1e4)
        if forced_ymax is not None:
            ax.set_ylim(0, forced_ymax)

        n_bf    = len(data[("backfill", "L")]) + len(data[("backfill", "XL")])
        n_total = n_bf + len(data[("primary", "L")]) + len(data[("primary", "XL")])
        pct_bf = 100.0 * n_bf / n_total if n_total else 0.0

        ax.set_title(
            f"{tag}\nN={n_total:,}  (backfill: {n_bf:,}, {pct_bf:.1f}%)",
            fontsize=10.5, fontweight="bold", pad=4,
        )
        ax.text(0.5, 1.30, sys_label, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_xlabel("Wait Time (hours)", fontsize=11, fontweight="bold")
        ax.set_ylabel("% of L+XL Jobs", fontsize=11, fontweight="bold")
        ax.tick_params(axis="both", labelsize=9)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        ax.grid(True, alpha=0.25, which="both", zorder=0)

    legend_handles = [
        Patch(facecolor=PRI_L,      label="Primary  L"),
        Patch(facecolor=PRI_XL,     label="Primary  XL"),
        Patch(facecolor=d_color_L,  label="Backfill  L"),
        Patch(facecolor=d_color_XL, label="Backfill  XL"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=10.0,
               frameon=False, handletextpad=0.4, columnspacing=1.0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _draw_wfp3_overview_row(ax_left, ax_right, drain_pct, post_g, total_g, bar_color,
                             sys_label, tag="WFP"):
    """Draw one system row of the WFP drain overview (single-tag variant)."""
    GROUPS   = ["S", "M", "L", "XL"]
    FS_TICK  = 8.5
    FS_AXIS  = 9.0
    FS_BAR   = 7.5

    def _bold_ticks(ax):
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
            lbl.set_color("black")
        ax.tick_params(labelsize=FS_TICK, colors="black", width=1.2, length=4)

    # ── left: drain rate bar ─────────────────────────────────────────────────
    ax = ax_left
    n_ep, n_elig, pct = drain_pct
    bar_w = 0.45
    ax.bar(0.5, pct, width=bar_w, color=bar_color, alpha=0.88, edgecolor="black", linewidth=0.8)
    ax.text(0.5, pct + pct * 0.03, f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=FS_BAR, fontweight="bold", color="black")
    ax.set_xticks([0.5])
    ax.set_xticklabels([tag], rotation=20, ha="right",
                       fontsize=FS_TICK, fontweight="bold")
    ax.set_xlim(0.5 - bar_w * 1.4, 0.5 + bar_w * 1.4)
    ax.set_ylim(0, max(pct * 1.85, 1.0))
    ax.set_ylabel(sys_label, fontsize=10.0, fontweight="bold", color="black", labelpad=1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    _bold_ticks(ax)

    # ── right: post-drain share by class ────────────────────────────────────
    ax = ax_right
    bw = 0.55
    x  = np.arange(len(GROUPS), dtype=float)
    pcts = [100.0 * post_g.get(g, 0) / total_g.get(g, 1)
            if total_g.get(g, 0) else 0.0
            for g in GROUPS]
    ax.bar(x, pcts, width=bw, color=bar_color, alpha=0.88, edgecolor="black", linewidth=0.8)
    y_pad = max(pcts, default=1) * 0.02
    for xi, v in zip(x, pcts):
        ax.text(xi, v + y_pad, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=FS_BAR, fontweight="bold", color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, fontsize=FS_TICK, fontweight="bold")
    ax.set_xlim(x[0] - bw, x[-1] + bw)
    ax.set_ylim(0, max(pcts, default=1) * 1.85)
    ax.set_ylabel("Post-drain Jobs (%)", fontsize=FS_AXIS, fontweight="bold", color="black")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    _bold_ticks(ax)


def plot_combined_drain_overview_wfp3_ab(
        theta_results_dir, polaris_results_dir, out_path,
        tag="WFP", bar_color=None):
    """2-row drain overview figure for a SortPolicy heuristic."""
    if bar_color is None:
        bar_color = _build_driver_colors([tag]).get(tag, "#888888")

    theta_pct,   theta_post,   theta_total,   _ = _wfp3_drain_stats(
        theta_results_dir,   _assign_group_theta,   tag)
    polaris_pct, polaris_post, polaris_total, _ = _wfp3_drain_stats(
        polaris_results_dir, _assign_group_polaris, tag)

    fig, axes = plt.subplots(
        2, 2, figsize=(5.8, 4.2),
        gridspec_kw={"width_ratios": [1, 2.4]},
    )
    fig.subplots_adjust(left=0.20, right=0.97, top=0.94, bottom=0.12,
                        wspace=0.32, hspace=0.35)

    for row_idx, (sys_label, drain_pct, post_g, total_g) in enumerate([
        ("Theta 2021",   theta_pct,   theta_post,   theta_total),
        ("Polaris 2024", polaris_pct, polaris_post, polaris_total),
    ]):
        _draw_wfp3_overview_row(
            axes[row_idx][0], axes[row_idx][1],
            drain_pct, post_g, total_g, bar_color, sys_label, tag=tag,
        )

    axes[0][0].set_title("Drain Rate",                fontsize=9.5, fontweight="bold", pad=3)
    axes[0][1].set_title("Post-Drain Share by Class", fontsize=9.5, fontweight="bold", pad=3)
    axes[1][1].set_xlabel("Size Class", fontsize=8.5, fontweight="bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_combined_drain_sequence_combos_wfp3_ab(
        theta_results_dir, polaris_results_dir, out_path, max_entries=8):
    """2-row sequence combos figure for WFP (Theta top, Polaris bottom)."""
    tag = "WFP"
    driver_colors = _build_driver_colors([tag])
    bar_color = driver_colors.get(tag, "#888888")

    systems = [
        ("Theta 2021",   theta_results_dir,   _assign_group_theta),
        ("Polaris 2024", polaris_results_dir, _assign_group_polaris),
    ]

    all_data = {}
    for sys_label, rd, assign_fn in systems:
        drain_pct, _, _, seq_counts = _wfp3_drain_stats(rd, assign_fn, tag)
        all_data[sys_label] = (drain_pct, seq_counts)

    row_h = max(1.2, max_entries * 0.30 + 0.6)
    fig, axes = plt.subplots(2, 1, figsize=(5.5, row_h * 2 + 0.4))
    fig.subplots_adjust(left=0.26, right=0.97, top=0.93, bottom=0.09,
                        hspace=0.50)

    for row_idx, (sys_label, rd, _) in enumerate(systems):
        ax = axes[row_idx]
        drain_pct, seq_counts = all_data[sys_label]
        entries = _build_wildcard_sequence_entries(seq_counts, max_entries=max_entries)
        n_ep, _, pct = drain_pct
        total_ep = sum(seq_counts.values()) if seq_counts else 1
        title = f"WFP\nN={n_ep:,} ({pct:.1f}%)"
        _draw_drain_sequence_bar(
            ax, entries, total_ep,
            show_ylabels=True,
            title=title,
            bar_color=bar_color,
            hatch=driver_hatch(tag),
        )
        ax.set_ylabel(sys_label, fontsize=11.0, fontweight="bold", labelpad=6, color="black")
        ax.set_title(ax.get_title(), fontsize=11.0, fontweight="bold")
        ax.set_xlabel(ax.get_xlabel(), fontsize=11.0, fontweight="bold")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontsize(10.0)
            lbl.set_fontweight("bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _load_mars_cw_drain_wait_data(results_dir, drain_plots_dir, assign_group_fn,
                                  tag="MARS-CW"):
    """Return {('drain'|'nondrain', 'L'|'XL'): [wait_hours]} for the given MARS tag."""
    events_path = os.path.join(results_dir, tag, "events.csv")
    if not os.path.exists(events_path):
        return {k: [] for k in (("drain", "L"), ("drain", "XL"),
                                ("nondrain", "L"), ("nondrain", "XL"))}

    submit_map, start_map, _ = parse_events(events_path)
    procs_map = _infer_procs_map_for_results_dir(results_dir)

    drain_job_ids = set()
    slug = tag_slug(tag)
    details_path = os.path.join(drain_plots_dir,
                                f"table_{slug}_drain_run_sequence_details.csv")
    if os.path.exists(details_path):
        with open(details_path) as f:
            for row in csv.DictReader(f):
                for pos in ("First", "Second", "Third"):
                    jid_str = row.get(f"{pos}_Run_Job", "").strip()
                    grp = row.get(f"{pos}_Run_Group", "").strip()
                    if not jid_str or grp not in ("L", "XL"):
                        continue
                    try:
                        drain_job_ids.add(int(jid_str))
                    except ValueError:
                        pass

    data = {("drain", "L"): [], ("drain", "XL"): [],
            ("nondrain", "L"): [], ("nondrain", "XL"): []}
    for jid, sub_t in submit_map.items():
        if jid not in start_map:
            continue
        grp = assign_group_fn(procs_map.get(jid, -1))
        if grp not in ("L", "XL"):
            continue
        wait_h = (start_map[jid] - sub_t) / 3600.0
        if wait_h < 0:
            continue
        kind = "drain" if jid in drain_job_ids else "nondrain"
        data[(kind, grp)].append(wait_h)

    return data


def plot_combined_drain_wait_stacked_hist_mars_cw(
        theta_results_dir, polaris_results_dir,
        theta_drain_dir, polaris_drain_dir,
        out_path,
        log_x=True, log_y=True):
    """1×2 stacked-bar wait-time histogram for L+XL jobs under MARS-CW.

    Left panel  : Theta 2021
    Right panel : Polaris 2024

    Stacking order (bottom → top):
      non-drain L  (light grey)
      non-drain XL (dark grey)
      drain L      (light red)
      drain XL     (dark red)
    X axis: wait time in hours, log scale 10^-1 to 10^4.
    Y axis: job count, log scale.
    """
    from matplotlib.patches import Patch

    theta_data   = _load_mars_cw_drain_wait_data(
        theta_results_dir,   theta_drain_dir,   _assign_group_theta)
    polaris_data = _load_mars_cw_drain_wait_data(
        polaris_results_dir, polaris_drain_dir, _assign_group_polaris)

    if log_x:
        bins = np.logspace(-1, 4, 41)
    else:
        all_w = []
        for d in (theta_data, polaris_data):
            for k in d:
                all_w.extend(d[k])
        max_w = max(all_w) if all_w else 100.0
        bins = np.linspace(0, max_w, 41)

    # light → dark within each category
    ND_L  = "#cccccc"   # non-drain L  (light grey)
    ND_XL = "#666666"   # non-drain XL (dark grey)
    D_L   = "#f4a0a0"   # drain L      (light red)
    D_XL  = "#c0392b"   # drain XL     (dark red)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.22, wspace=0.08)

    systems = [
        (axes[0], theta_data,   "Theta 2021"),
        (axes[1], polaris_data, "Polaris 2024"),
    ]

    for ax, data, sys_label in systems:
        nd_l,  _ = np.histogram(data[("nondrain", "L")],  bins=bins)
        nd_xl, _ = np.histogram(data[("nondrain", "XL")], bins=bins)
        d_l,   _ = np.histogram(data[("drain",    "L")],  bins=bins)
        d_xl,  _ = np.histogram(data[("drain",    "XL")], bins=bins)

        lefts  = bins[:-1]
        widths = np.diff(bins)

        b0 = np.zeros(len(bins) - 1)
        b1 = b0   + nd_l
        b2 = b1   + nd_xl
        b3 = b2   + d_l

        ax.bar(lefts, nd_l,  width=widths, bottom=b0, align="edge", color=ND_L,  zorder=3)
        ax.bar(lefts, nd_xl, width=widths, bottom=b1, align="edge", color=ND_XL, zorder=3)
        ax.bar(lefts, d_l,   width=widths, bottom=b2, align="edge", color=D_L,   zorder=3)
        ax.bar(lefts, d_xl,  width=widths, bottom=b3, align="edge", color=D_XL,  zorder=3)

        if log_x:
            ax.set_xscale("log")
            ax.set_xlim(1e-1, 1e4)
        else:
            ax.set_xlim(bins[0], bins[-1])

        if log_y:
            ax.set_yscale("log")

        n_drain = len(data[("drain", "L")]) + len(data[("drain", "XL")])
        n_total = n_drain + len(data[("nondrain", "L")]) + len(data[("nondrain", "XL")])
        pct = 100.0 * n_drain / n_total if n_total else 0.0

        ax.set_title(
            f"MARS-CW\nN={n_total:,}  (drain: {n_drain:,}, {pct:.1f}%)",
            fontsize=12.0, fontweight="bold", pad=6,
        )
        ax.text(0.5, 1.15, sys_label, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=14, fontweight="bold")

        ax.set_xlabel("Wait Time (hours)", fontsize=13, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Job Count", fontsize=13, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        ax.grid(True, alpha=0.25, which="both", zorder=0)

    legend_handles = [
        Patch(facecolor=ND_L,  label="Non-drain  L"),
        Patch(facecolor=ND_XL, label="Non-drain  XL"),
        Patch(facecolor=D_L,   label="Drain  L"),
        Patch(facecolor=D_XL,  label="Drain  XL"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=12.0,
               frameon=False, handletextpad=0.4, columnspacing=1.0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _load_maintenance_intervals(evt_path):
    """Return sorted list of (start, end) maintenance windows from events.csv.

    Scheduled:   SMA (announcement) → SME (end)
    Unscheduled: UMS (start)        → UME (end)
    """
    events = []
    with open(evt_path) as f:
        for row in csv.DictReader(f):
            ev = row.get("event", "")
            if ev in ("SMA", "SME", "UMS", "UME"):
                events.append((int(row["sim_time"]), ev))
    events.sort()

    intervals = []
    pending_sma = pending_ums = None
    for t, ev in events:
        if ev == "SMA":
            pending_sma = t
        elif ev == "SME" and pending_sma is not None:
            intervals.append((pending_sma, t))
            pending_sma = None
        elif ev == "UMS":
            pending_ums = t
        elif ev == "UME" and pending_ums is not None:
            intervals.append((pending_ums, t))
            pending_ums = None
    return sorted(intervals)


def _in_maintenance(t, intervals):
    """Return True if sim_time t falls inside any maintenance interval."""
    for start, end in intervals:
        if start <= t <= end:
            return True
        if start > t:
            break
    return False


def _collect_drain_resume_times_for_system(results_dir, system_label, tags):
    """Return rows of {system, tag, drain_start_time, first_run_time, gap_seconds} per drain episode.

    For MCTS-based MARS tags the run policies are FCFS and MCTS; drain is DRAIN.
    Episodes whose first drain cycle falls inside a maintenance window
    (SMA→SME or UMS→UME) are excluded — scheduling is suppressed during
    maintenance so the gap would not reflect normal drain behaviour.
    """
    DRAIN_POL = {"DRAIN"}
    RUN_POL   = {"FCFS", "MCTS"}

    rows = []
    for tag in tags:
        perf_path = os.path.join(results_dir, tag, "performance.csv")
        evt_path  = os.path.join(results_dir, tag, "events.csv")
        if not (os.path.exists(perf_path) and os.path.exists(evt_path)):
            continue

        maint_intervals = _load_maintenance_intervals(evt_path)

        perf_rows = {}
        with open(perf_path) as f:
            for row in csv.DictReader(f):
                c = int(row["cycle"])
                perf_rows[c] = (row["selected_policy"].strip().upper(), int(row["sim_time"]))

        run_events = []
        with open(evt_path) as f:
            for row in csv.DictReader(f):
                if row["event"] == "Run":
                    run_events.append((int(row["sim_time"]), int(row["id"])))
        run_events.sort()

        in_drain          = False
        first_drain_sim_t = None

        for c in sorted(perf_rows.keys()):
            policy, sim_t = perf_rows[c]

            if policy in DRAIN_POL:
                if not in_drain:
                    first_drain_sim_t = sim_t
                    in_drain          = True
            elif policy in RUN_POL:
                if in_drain and first_drain_sim_t is not None:
                    if not _in_maintenance(first_drain_sim_t, maint_intervals):
                        idx = bisect.bisect_left(run_events, (sim_t,))
                        if idx < len(run_events):
                            first_run_sim_t = run_events[idx][0]
                            rows.append({
                                "system":           system_label,
                                "tag":              tag,
                                "drain_start_time": first_drain_sim_t,
                                "first_run_time":   first_run_sim_t,
                                "gap_seconds":      first_run_sim_t - first_drain_sim_t,
                            })
                in_drain          = False
                first_drain_sim_t = None
            else:  # NONE
                in_drain          = False
                first_drain_sim_t = None

    return rows


def write_drain_resume_times_csv(
        theta_results_dir, polaris_results_dir, out_path,
        tags=("MARS-CW", "MARS-CB", "MARS-CU")):
    """Write a CSV of last-drain-cycle → first-Run-event gap for every drain episode."""
    systems = [
        ("Theta 2021",   theta_results_dir),
        ("Polaris 2024", polaris_results_dir),
    ]
    all_rows = []
    for label, rd in systems:
        all_rows.extend(_collect_drain_resume_times_for_system(rd, label, tags))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["system", "tag", "drain_start_time", "first_run_time", "gap_seconds"],
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Saved: {out_path}  ({len(all_rows)} episodes)")
    return all_rows


def plot_drain_resume_time_boxplot(csv_path, out_path):
    """Two-panel boxplot (Theta | Polaris) of drain→first-Run gap in minutes."""
    if not os.path.exists(csv_path):
        print(f"  Skipping drain resume boxplot (CSV not found): {csv_path}")
        return

    SYSTEMS = ["Theta 2021", "Polaris 2024"]
    TAG_ORDER = ["MARS-CW", "MARS-CB", "MARS-CU"]

    data_by_system = {s: {} for s in SYSTEMS}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            s   = row["system"]
            tag = row["tag"]
            if s in data_by_system:
                data_by_system[s].setdefault(tag, []).append(float(row["gap_seconds"]) / 60.0)

    tag_colors  = _build_driver_colors(TAG_ORDER)
    tag_hatches = {t: driver_hatch(t) for t in TAG_ORDER}

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.06)

    for ax, system in zip(axes, SYSTEMS):
        groups  = data_by_system[system]
        tags    = [t for t in TAG_ORDER if t in groups]
        data    = [groups[t] for t in tags]
        colors  = [tag_colors[t] for t in tags]
        hatches = [tag_hatches[t] for t in tags]

        bp = ax.boxplot(data, labels=tags, patch_artist=True, showfliers=False,
                        whis=1.5,
                        medianprops=dict(color="black", linewidth=2.8))
        for patch, color, hatch in zip(bp["boxes"], colors, hatches):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            if hatch:
                patch.set_hatch(hatch)
                patch.set_edgecolor("black")

        # dotted red mean line per box
        for i, vals in enumerate(data, start=1):
            mean_val = sum(vals) / len(vals)
            ax.plot([i - 0.4, i + 0.4], [mean_val, mean_val],
                    color="red", linestyle=":", linewidth=1.8)
            ax.text(i + 0.2, mean_val, f"{mean_val:.1f}",
                    color="red", fontsize=12, fontweight="bold",
                    va="bottom", ha="left")
        ax.set_title(system, fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", labelsize=12)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Minutes", fontsize=13, fontweight="bold")


    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
