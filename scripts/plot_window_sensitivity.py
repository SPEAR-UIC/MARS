#!/usr/bin/env python3
"""Plot overall average wait time vs scheduling window size, Theta vs Polaris
side by side, for the window-sweep heuristics defined in exp1a/exp1b configs
(drivers tagged "{POLICY}-w{N}"; drivers without a "-w" suffix, e.g. FCFS,
are treated as fixed baselines and repeated across all window sizes).

Wait time is computed directly from each driver's events.csv (Submit -> Run
or Backfill), not from any precomputed table -- there is no dependency on an
intermediate table_wait_overall.csv or similar file.

Usage:
    python3 scripts/plot_window_sensitivity.py \\
        experiments/exp1a.json experiments/exp1b.json \\
        --output_dir plots_cluster26
"""

import argparse
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_MAINT_EVT = frozenset({
    "SMA", "SMS", "SME", "UMS", "UME",
    "MaintenanceAnnounced", "MaintenanceStart", "MaintenanceEnd",
    "ScheduledMaintenanceAnnounced", "ScheduledMaintenanceStart",
    "ScheduledMaintenanceEnd", "UnscheduledMaintenanceStart",
    "UnscheduledMaintenanceEnd",
})

DEFAULT_POLICIES = ("FCFS", "SJF", "WFP3", "F1", "UNICEP", "FAT", "LRF", "LJF")

# Display-only renames: the underlying driver tag (e.g. "WFP3-w64") still
# drives data lookup; this only changes the legend/line label text.
DISPLAY_LABEL = {"WFP3": "WFP"}


def display_label(policy):
    return DISPLAY_LABEL.get(policy, policy)

FIG_W, FIG_H, DPI = 4.8, 3.2, 300
AXIS_FONT, TICK_FONT, LEGEND_FONT = 11, 9.0, 10.5
LINE_WIDTH, MARKER_SIZE = 1.2, 4.0


def load_config(path):
    with open(path) as f:
        return json.load(f)


def detect_system_label(swf_path):
    swf = (swf_path or "").lower()
    if "theta" in swf:
        return "Theta\n(2021)"
    if "polaris" in swf:
        return "Polaris\n(2024)"
    stem = os.path.splitext(os.path.basename(swf_path or "system"))[0]
    return stem or "System"


def split_driver_label(tag):
    """Return (policy_name, window_size_or_None)."""
    m = re.match(r"^(.*)-w(\d+)$", tag)
    if not m:
        return tag, None
    return m.group(1), int(m.group(2))


def load_mean_wait_hours(results_dir, tag):
    """Mean (Submit -> Run/Backfill) wait time in hours for one driver,
    computed directly from events.csv."""
    events_path = os.path.join(results_dir, tag, "events.csv")
    if not os.path.exists(events_path):
        return None
    submit, start = {}, {}
    with open(events_path) as f:
        for row in csv.DictReader(f):
            event = row.get("event", "")
            if event in _MAINT_EVT:
                continue
            try:
                jid = int(row["id"])
                t = float(row["sim_time"])
            except (KeyError, ValueError, TypeError):
                continue
            if event == "Submit":
                submit.setdefault(jid, t)
            elif event in ("Run", "Backfill"):
                start[jid] = t
    waits = [
        (start[jid] - submit[jid]) / 3600.0
        for jid in submit.keys() & start.keys()
        if start[jid] - submit[jid] > 0
    ]
    if not waits:
        return None
    return float(np.mean(waits))


def load_window_series(results_dir, drivers, policies):
    """Return (window_sizes, {policy: {window: mean_wait_hours}}) for the
    requested policies, built directly from each driver's events.csv."""
    baselines = {}
    series = {p: {} for p in policies}
    windows = set()
    for d in drivers:
        policy, window = split_driver_label(d["tag"])
        if policy not in policies:
            continue
        mean_wait = load_mean_wait_hours(results_dir, d["tag"])
        if mean_wait is None:
            continue
        if window is None:
            baselines[policy] = mean_wait
        else:
            series[policy][window] = mean_wait
            windows.add(window)

    window_list = sorted(windows)
    if not window_list:
        raise ValueError(f"No windowed drivers found for policies {policies} in {results_dir}")
    for policy, value in baselines.items():
        series[policy] = {w: value for w in window_list}
    return window_list, series


def build_color_map(policies):
    palette = list(plt.get_cmap("tab20").colors)
    colors = {}
    idx = 0
    for p in policies:
        if p == "FCFS":
            colors[p] = "#000000"
        else:
            colors[p] = palette[idx % len(palette)]
            idx += 1
    return colors


def _log_e_formatter(value, _pos):
    """Render compact log labels like 1e1, 2e1, 2e2."""
    if value <= 0:
        return ""
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / (10 ** exponent)
    rounded = round(mantissa)
    if not np.isclose(mantissa, rounded, rtol=0, atol=1e-9):
        return ""
    if rounded not in (1, 2):
        return ""
    return f"{rounded}e{exponent}"


def apply_log_e_axis(ax):
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_log_e_formatter))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=tuple(range(3, 10))))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def plot_panel(ax, window_sizes, policies, series_by_policy, color_map, title):
    x_positions = np.arange(len(window_sizes))
    pos_by_window = {w: i for i, w in enumerate(window_sizes)}

    if series_by_policy.get("FCFS"):
        fcfs_val = next(iter(series_by_policy["FCFS"].values()))
        ax.axhline(fcfs_val, color=color_map["FCFS"], linestyle=":",
                   linewidth=LINE_WIDTH, label=display_label("FCFS"), zorder=2)

    for policy in policies:
        if policy == "FCFS" or not series_by_policy.get(policy):
            continue
        s = series_by_policy[policy]
        pts = sorted((pos_by_window[w], v) for w, v in s.items() if w in pos_by_window)
        xs = [p for p, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, label=display_label(policy), color=color_map[policy],
                linewidth=LINE_WIDTH, linestyle=":", marker="o",
                markersize=MARKER_SIZE, zorder=3)

    ax.set_xlim(-0.5, len(window_sizes) - 0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(w) for w in window_sizes], rotation=32, ha="right")
    ax.set_xlabel("Window Size", fontsize=AXIS_FONT)
    ax.set_title(title, fontsize=AXIS_FONT)
    ax.set_yscale("log")
    apply_log_e_axis(ax)
    ax.tick_params(axis="both", labelsize=TICK_FONT)
    ax.grid(True, which="major", axis="y", color="#d0d0d0", linewidth=0.6)
    ax.grid(True, which="minor", axis="y", color="#ebebeb", linewidth=0.45)
    ax.grid(True, which="major", axis="x", color="#f0f0f0", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def reorder_handles_row_major(handles, labels, ncol):
    rows = [list(zip(handles[i:i + ncol], labels[i:i + ncol]))
            for i in range(0, len(handles), ncol)]
    ordered = []
    max_cols = max((len(row) for row in rows), default=0)
    for c in range(max_cols):
        for row in rows:
            if c < len(row):
                ordered.append(row[c])
    return [h for h, _ in ordered], [l for _, l in ordered]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot overall average wait time vs scheduling window size "
                    "(Theta vs Polaris), computed directly from exp1a/exp1b events.csv.")
    parser.add_argument("exp1a_json", help="Path to exp1a.json (Theta window sweep)")
    parser.add_argument("exp1b_json", help="Path to exp1b.json (Polaris window sweep)")
    parser.add_argument("--output_dir", default="plots_cluster26",
                        help="Directory to write window_sensitivity.png into "
                             "(default: plots_cluster26)")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES),
                        help=f"Comma-separated policy names to plot "
                             f"(default: {','.join(DEFAULT_POLICIES)})")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    def abs_path(p):
        return p if os.path.isabs(p) else os.path.join(base_dir, p)

    cfg_a = load_config(args.exp1a_json)
    cfg_b = load_config(args.exp1b_json)
    res_a = abs_path(cfg_a.get("output_dir", "results/exp1a"))
    res_b = abs_path(cfg_b.get("output_dir", "results/exp1b"))
    label_a = detect_system_label(cfg_a.get("swf_path", ""))
    label_b = detect_system_label(cfg_b.get("swf_path", ""))

    windows_a, series_a = load_window_series(res_a, cfg_a["drivers"], policies)
    windows_b, series_b = load_window_series(res_b, cfg_b["drivers"], policies)

    color_map = build_color_map(policies)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), dpi=DPI,
                             gridspec_kw={"wspace": 0.25})

    plot_panel(axes[0], windows_a, policies, series_a, color_map, label_a)
    axes[0].set_ylabel("Overall Avg Wait (h)", fontsize=AXIS_FONT, labelpad=2)
    plot_panel(axes[1], windows_b, policies, series_b, color_map, label_b)

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_labels = [display_label(p) for p in policies if display_label(p) in by_label]
    ordered_handles = [by_label[l] for l in ordered_labels]

    if ordered_handles:
        ordered_handles, ordered_labels = reorder_handles_row_major(
            ordered_handles, ordered_labels, ncol=4)
        fig.legend(
            ordered_handles, ordered_labels,
            loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4,
            fontsize=LEGEND_FONT, frameon=False,
            handlelength=1.6, columnspacing=1.0, handletextpad=0.4,
        )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.38)
    out_path = os.path.join(args.output_dir, "window_sensitivity.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
