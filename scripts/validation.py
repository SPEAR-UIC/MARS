#!/usr/bin/env python3
"""validation.py — Verify the C++ cqsim reimplementation against the
reference Python cqsim (cqsim-python) by comparing per-job event timestamps
on the same input trace, using each side's events.csv.

Data sources
------------
Both sides   : sim_time,event,id  with event in {Submit, Run, Backfill, End,
               SchedulingCycle, ...}
               A job's start time is marked by either "Run" (started in its
               primary scheduling slot) or "Backfill" (started via
               backfilling); "End" marks its finish time.

               Python side : cqsim-python/data/Results/<trace>_events.csv
                              (see cqsim-python/src/IOModule/Output_log.py print_event)
               C++ side    : results/<exp>/<driver_tag>/events.csv

               The C++ simulator additionally logs a "SchedulingCycle" event
               and marks backfilled jobs with "Backfill", neither of which
               cqsim-python's job-event log produces. This script normalizes
               both sides the same way while reading them (see load_events):
               "SchedulingCycle" rows are dropped and "Backfill" is treated
               the same as "Run" (both mark a job's start time), so raw
               events.csv output from either simulator can be compared
               directly -- no separate conversion step is needed.

Ground truth : the shared .swf trace (e.g. data/theta21cln.swf) gives the
               total job count in the trace, used only as a coverage reference.

Job ids are the raw SWF first-column ids and line up 1:1 between the two
simulators, EXCEPT that some jobs finish in one output but not the other
(e.g. never scheduled by simulation end). Those are reported as coverage
diagnostics and excluded from the deviation statistics, which are computed
only on the intersection of completed job ids.

For each matched job we directly compare the three raw event timestamps
each simulator recorded: submit, start (Run/Backfill), and end (End).

Only the Theta 2021 trace is wired up as a default right now -- the Polaris
2024 C++ run hasn't finished yet. Point --cpp-events/--py-events/--swf at
the Polaris files once they're available.

Usage
-----
    # theta21cln, using this repo's checked-in default paths:
    python3 scripts/validation.py

    # explicit paths (e.g. once the polaris24cln C++ run is ready):
    python3 scripts/validation.py \\
        --cpp-events results/cqsimpt-test2/FCFS/events.csv \\
        --py-events cqsim-python/data/Results/polaris24_events.csv \\
        --swf data/polaris24cln.swf \\
        --label polaris24cln --output-dir results/cpp_vs_py
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext
import numpy as np

DPI = 150

# The C++ simulator logs a couple of event kinds the python side doesn't;
# normalize both sides to the same {Submit, Run, End} vocabulary on read.
DROP_EVENTS = {"SchedulingCycle"}
RENAME_EVENTS = {"Backfill": "Run"}


# ─── Parsing ──────────────────────────────────────────────────────────────────

def parse_swf(path):
    """Return {id: {'reqProc', 'reqTime', 'run_orig', 'submit_raw'}}."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            jid = int(parts[0])
            submit = int(parts[1])
            run_orig = int(parts[3])
            req_procs = int(parts[7])
            req_time = float(parts[8])
            out[jid] = {
                "reqProc": req_procs,
                "reqTime": req_time,
                "run_orig": run_orig,
                "submit_raw": submit,
            }
    return out


def load_events(path):
    """Parse an events.csv (sim_time,event,id) from either simulator.

    Drops "SchedulingCycle" rows and treats "Backfill" as "Run" so raw
    C++ output and python output normalize to the same shape.

    Returns (jobs, first_submit):
        jobs         : {id: {'submit', 'start', 'end'}} for jobs with all
                        three events recorded (i.e. completed by sim end).
        first_submit : earliest submit time seen in this file, across ALL
                        submitted jobs (not just completed ones) -- the
                        anchor used to make start times comparable across
                        the two simulators' runs of the same trace.
    """
    submit_t, start_t, end_t = {}, {}, {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ev = row["event"]
            if ev in DROP_EVENTS:
                continue
            ev = RENAME_EVENTS.get(ev, ev)
            jid = int(row["id"])
            t = float(row["sim_time"])
            if ev == "Submit":
                submit_t.setdefault(jid, t)
            elif ev == "Run":
                start_t.setdefault(jid, t)
            elif ev == "End":
                end_t.setdefault(jid, t)

    jobs = {}
    for jid, sub in submit_t.items():
        if jid in start_t and jid in end_t:
            jobs[jid] = {"submit": sub, "start": start_t[jid], "end": end_t[jid]}

    first_submit = min(submit_t.values()) if submit_t else 0.0
    return jobs, first_submit


# Known trace file locations, used by build_polaris_theta_violin() to build
# the combined comparison figure once both traces' C++ runs exist.
TRACE_FILES = {
    "polaris24cln": dict(
        cpp_events="results/cqsimpt-test2/FCFS/events.csv",
        py_events="cqsim-python/data/Results/polaris24_events.csv",
    ),
    "theta21cln": dict(
        cpp_events="results/cqsimpy-test1/FCFS/events.csv",
        py_events="cqsim-python/data/Results/theta21_events.csv",
    ),
}


def compute_start_diff_hours(py_events_path, cpp_events_path):
    """|CQSim C++ start - CQSim Python start| in hours, over matched jobs."""
    py, _ = load_events(py_events_path)
    cpp, _ = load_events(cpp_events_path)
    matched = set(py) & set(cpp)
    if not matched:
        return None
    return np.array([abs(cpp[j]["start"] - py[j]["start"]) for j in matched]) / 3600.0


# ─── Metrics ──────────────────────────────────────────────────────────────────

def summarize(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "p95": float(np.percentile(a, 95)),
    }


def pearson_r(x, y):
    if np.std(x) > 0 and np.std(y) > 0:
        return float(np.corrcoef(x, y)[0, 1])
    return float("nan")


# ─── Plot theming (borrowed from scripts/plot_cluster26.py) ───────────────────
# Bold ticks/labels, light major-axis grid, tight-cropped saves -- matches the
# look of the CLUSTER'26 paper plots so validation figures read as one family.

def _bold_ax(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.tick_params(labelsize=11)
    ax.grid(True, which="major", alpha=0.3, zorder=0)


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_start_relative(py_rel, cpp_rel, r, n, label, output_dir, log_scale=False):
    """Scatter of start time relative to each side's own first submit time,
    with a y = x reference line and the correlation stats in the corner."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(py_rel, cpp_rel, s=4, alpha=0.25, color="#1f77b4", linewidths=0, zorder=2)

    if log_scale:
        pos = np.concatenate([py_rel, cpp_rel])
        pos = pos[pos > 0]
        lo = float(pos.min()) if pos.size else 1e-3
        hi = float(max(py_rel.max(), cpp_rel.max()))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
    else:
        lo = min(py_rel.min(), cpp_rel.min())
        hi = max(py_rel.max(), cpp_rel.max())

    ax.plot([lo, hi], [lo, hi], color="#d62728", lw=1.4, ls="--", label="y = x", zorder=3)
    ax.set_xlabel("Start Time in CQSim-Python", fontsize=12, fontweight="bold", labelpad=4)
    ax.set_ylabel("Start Time in CQSim-C++", fontsize=12, fontweight="bold", labelpad=4)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(
        0.98, 0.03, f"r = {r:.4f}\nr² = {r ** 2:.4f}\nn = {n}",
        transform=ax.transAxes, fontsize=10, ha="right", va="bottom", fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", linewidth=0.8, alpha=0.9),
    )
    _bold_ax(ax)
    fig.tight_layout()

    suffix = "_start_relative_log" if log_scale else "_start_relative"
    fig_path = os.path.join(output_dir, f"{label}{suffix}.png")
    save_fig(fig, fig_path)
    return fig_path


_TRACE_DISPLAY_NAMES = {
    "theta21cln": "Theta 2021",
    "polaris24cln": "Polaris 2024",
}


def plot_start_diff_violin(diff, label, output_dir):
    """Violin plot of a per-job start-time difference metric (e.g. the
    magnitude |CQSim C++ - CQSim Python|); `diff` is whatever values/units
    the caller wants plotted."""
    fig, ax = plt.subplots(figsize=(5, 6.5))
    parts = ax.violinplot([diff], showmeans=True, showmedians=True, showextrema=True)

    for body in parts["bodies"]:
        body.set_facecolor("#1f77b4")
        body.set_edgecolor("black")
        body.set_linewidth(1.2)
        body.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2.4)
    parts["cmeans"].set_color("red")
    parts["cmeans"].set_linewidth(2.0)
    parts["cmeans"].set_linestyle(":")
    for key in ("cbars", "cmins", "cmaxes"):
        parts[key].set_color("black")
        parts[key].set_linewidth(1.2)

    ax.axhline(0, color="#d62728", lw=1.2, ls="--", zorder=1)
    ax.set_xticks([1])
    ax.set_xticklabels([_TRACE_DISPLAY_NAMES.get(label, label)])
    ax.set_ylabel("|Δ Start Time| (hours)",
                   fontsize=12, fontweight="bold", labelpad=4)

    # Percentile + median reference lines, spanning the full width of the
    # axes. Labels alternate left/right (in value order) so close-together
    # values don't overlap each other's text.
    median_v = float(np.median(diff))
    pct_colors = {75: "#ff7f0e", 99: "#8c564b"}
    pct_values = {p: float(np.percentile(diff, p)) for p in pct_colors}
    labeled_lines = [("median", median_v, "black")] + \
        [(f"P{p}", pct_values[p], pct_colors[p]) for p in pct_colors]

    y_trans = ax.get_yaxis_transform()  # x in axes coords, y in data coords
    for i, (name, v, color) in enumerate(sorted(labeled_lines, key=lambda t: t[1])):
        if name != "median":
            ax.axhline(v, color=color, lw=1.3, ls="-.", zorder=1.5)
        side_x, ha = (0.02, "left") if i % 2 == 0 else (0.98, "right")
        ax.text(side_x, v, f"{name}={v:.2f}", transform=y_trans, fontsize=8.5,
                fontweight="bold", color=color, ha=ha, va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.75))

    # Crop the view to P99 -- the extreme tail beyond it otherwise dwarfs the
    # bulk of the distribution. A little headroom keeps the P99 line/label
    # from sitting flush against the top spine.
    ax.set_ylim(0, pct_values[99] * 1.06)

    legend_handles = [
        Line2D([0], [0], color="black", lw=2.4, label="median"),
        Line2D([0], [0], color="red", lw=2.0, ls=":", label="mean"),
    ] + [
        Line2D([0], [0], color=c, lw=1.3, ls="-.", label=f"P{p}")
        for p, c in pct_colors.items()
    ]
    # Legend sits above the axes entirely so it never collides with the
    # in-plot P75/P99 labels, wherever their values happen to land.
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=4, fontsize=12, frameon=True, columnspacing=1.2, handlelength=1.8)

    _bold_ax(ax)
    ax.tick_params(labelsize=13)
    fig.tight_layout()

    fig_path = os.path.join(output_dir, f"{label}_start_diff_violin.png")
    save_fig(fig, fig_path)
    return fig_path


def plot_polaris_theta_violin(diff_by_label, output_dir):
    """Compact two-panel |Δ start time| violin: Polaris 2024 on the left,
    Theta 2021 on the right, sized to sit at roughly a quarter of a row's
    width in a two-column LaTeX paper figure (e.g. next to 3 similar plots).

    Skinnier than plot_start_diff_violin's single-trace version and with
    lighter per-panel annotation (median + P99 only) to stay legible once
    shrunk down on the page.
    """
    order = ["polaris24cln", "theta21cln"]
    positions = {lbl: i + 1 for i, lbl in enumerate(order)}
    face_colors = {"polaris24cln": "#2ca02c", "theta21cln": "#1f77b4"}
    violin_width = 0.5

    fig, ax = plt.subplots(figsize=(3.3, 3.2))

    for lbl in order:
        diff = diff_by_label[lbl]
        pos = positions[lbl]
        parts = ax.violinplot([diff], positions=[pos], widths=violin_width,
                               showmeans=True, showmedians=True, showextrema=True)
        for body in parts["bodies"]:
            body.set_facecolor(face_colors[lbl])
            body.set_edgecolor("black")
            body.set_linewidth(1.0)
            body.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.8)
        parts["cmeans"].set_color("red")
        parts["cmeans"].set_linewidth(1.5)
        parts["cmeans"].set_linestyle(":")
        for key in ("cbars", "cmins", "cmaxes"):
            parts[key].set_color("black")
            parts[key].set_linewidth(0.9)

        median_v = float(np.median(diff))
        p99_v = float(np.percentile(diff, 99))
        half = violin_width / 2 + 0.05
        ax.hlines(p99_v, pos - half, pos + half, color="#8c564b", lw=1.1, ls="-.", zorder=1.5)
        ax.text(pos, p99_v, f"P99={p99_v:.1f}", fontsize=6.3, fontweight="bold",
                color="#8c564b", ha="center", va="bottom")
        ax.text(pos, median_v, f"med={median_v:.1f}", fontsize=6.3, fontweight="bold",
                color="black", ha="center", va="bottom")

    ax.axhline(0, color="#d62728", lw=1.0, ls="--", zorder=1)
    ax.set_xticks([positions[l] for l in order])
    ax.set_xticklabels([_TRACE_DISPLAY_NAMES.get(l, l) for l in order])
    ax.set_xlim(0.5, 2.5)
    top = max(float(np.percentile(diff_by_label[l], 99)) for l in order) * 1.15
    ax.set_ylim(0, top)
    ax.set_ylabel("|Δ Start Time| (hours)", fontsize=9, fontweight="bold", labelpad=3)

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.8, label="median"),
        Line2D([0], [0], color="red", lw=1.5, ls=":", label="mean"),
        Line2D([0], [0], color="#8c564b", lw=1.1, ls="-.", label="P99"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, fontsize=6.5, frameon=True, columnspacing=0.8, handlelength=1.4)

    _bold_ax(ax)
    ax.tick_params(labelsize=8)
    fig.tight_layout()

    fig_path = os.path.join(output_dir, "polaris_theta_start_diff_violin.png")
    save_fig(fig, fig_path)
    return fig_path


def build_polaris_theta_violin(output_dir):
    """Emit the combined Polaris-vs-Theta violin once both traces' raw C++
    events.csv exist. Skips (with a note, not an error) while either side is
    still missing -- e.g. the Polaris C++ run hasn't finished yet."""
    diff_by_label = {}
    for lbl, paths in TRACE_FILES.items():
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            print(f"[skip combined polaris/theta violin] {lbl}: missing {missing[0]}")
            return None
        diff_by_label[lbl] = compute_start_diff_hours(paths["py_events"], paths["cpp_events"])

    fig_path = plot_polaris_theta_violin(diff_by_label, output_dir)
    print(f"wrote {fig_path}")
    return fig_path


# ─── Core comparison ──────────────────────────────────────────────────────────

def compare(label, swf_path, py_events_path, cpp_events_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    ground_truth = parse_swf(swf_path)
    py, py_first_submit = load_events(py_events_path)
    cpp, cpp_first_submit = load_events(cpp_events_path)

    ids_py = set(py)
    ids_cpp = set(cpp)
    ids_gt = set(ground_truth)
    matched = sorted(ids_py & ids_cpp)
    only_py = sorted(ids_py - ids_cpp)
    only_cpp = sorted(ids_cpp - ids_py)

    report_lines = []
    def out(s=""):
        print(s)
        report_lines.append(s)

    out(f"=== {label} ===")
    out(f"trace jobs (swf)     : {len(ids_gt)}")
    out(f"python completed     : {len(ids_py)}")
    out(f"cpp completed        : {len(ids_cpp)}")
    out(f"matched (both)       : {len(matched)}")
    out(f"only in python       : {len(only_py)}"
        + (f"  e.g. {only_py[:5]}" if only_py else ""))
    out(f"only in cpp          : {len(only_cpp)}"
        + (f"  e.g. {only_cpp[:5]}" if only_cpp else ""))

    if not matched:
        out("No overlapping completed jobs -- cannot compute deviation stats.")
        return None

    fields = ["submit", "start", "end"]
    diffs = {f: np.array([cpp[j][f] - py[j][f] for j in matched]) for f in fields}
    py_arr = {f: np.array([py[j][f] for j in matched]) for f in fields}
    cpp_arr = {f: np.array([cpp[j][f] for j in matched]) for f in fields}

    out("\n--- per-job deviation (cpp - python), over matched jobs ---")
    for f in fields:
        d = diffs[f]
        mae = float(np.mean(np.abs(d)))
        rmse = float(np.sqrt(np.mean(d ** 2)))
        corr = pearson_r(py_arr[f], cpp_arr[f])
        out(f"  {f:10s}  MAE={mae:12.3f}  RMSE={rmse:12.3f}  mean_diff={float(np.mean(d)):12.3f}  "
            f"max_abs_diff={float(np.max(np.abs(d))):12.3f}  corr={corr:.4f}")

    out("\n--- aggregate metrics computed identically on both sides ---")
    out(f"{'metric':12s} {'python':>14s} {'cpp':>14s} {'abs diff':>12s} {'rel diff %':>10s}")
    for f in fields:
        p_mean = float(np.mean(py_arr[f]))
        c_mean = float(np.mean(cpp_arr[f]))
        rel = 100.0 * (c_mean - p_mean) / p_mean if p_mean != 0 else float("nan")
        out(f"{f:12s} {p_mean:14.3f} {c_mean:14.3f} {abs(c_mean - p_mean):12.3f} {rel:10.2f}")

    # ─── First point of divergence ─────────────────────────────────────────
    # Walk matched jobs in chronological (python submit) order and report the
    # first one whose start or end time disagrees between the two sides --
    # everything before it lines up exactly, so this is where the two
    # simulators' scheduling decisions first split.
    TOL = 1e-6
    chrono = sorted(matched, key=lambda j: (py[j]["submit"], j))
    first_idx = None
    for i, jid in enumerate(chrono):
        if abs(cpp[jid]["start"] - py[jid]["start"]) > TOL or abs(cpp[jid]["end"] - py[jid]["end"]) > TOL:
            first_idx = i
            break

    out("\n--- first point of divergence (jobs in python-submit-time order) ---")
    if first_idx is None:
        out("No divergence found -- all matched jobs agree on submit/start/end within tolerance.")
    else:
        jid = chrono[first_idx]
        out(f"first mismatch at position {first_idx + 1}/{len(chrono)} in submit order: job id {jid}")
        out(f"  python : submit={py[jid]['submit']:.3f}  start={py[jid]['start']:.3f}  end={py[jid]['end']:.3f}")
        out(f"  cpp    : submit={cpp[jid]['submit']:.3f}  start={cpp[jid]['start']:.3f}  end={cpp[jid]['end']:.3f}")
        out(f"  diff   : start={cpp[jid]['start'] - py[jid]['start']:+.3f}  end={cpp[jid]['end'] - py[jid]['end']:+.3f}")

        out("\n  context (jobs immediately before/after, submit order):")
        out(f"  {'#':>6s} {'id':>8s} {'py_submit':>14s} {'py_start':>14s} {'cpp_start':>14s} {'start_diff':>12s}")
        lo = max(0, first_idx - 3)
        hi = min(len(chrono), first_idx + 6)
        for i in range(lo, hi):
            j = chrono[i]
            marker = " <-- " if i == first_idx else "     "
            out(f"  {i + 1:>6d} {j:>8d} {py[j]['submit']:>14.3f} {py[j]['start']:>14.3f} "
                f"{cpp[j]['start']:>14.3f} {cpp[j]['start'] - py[j]['start']:>+12.3f}{marker}")

    # ─── Plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    def scatter_ax(ax, f, unit=""):
        x, y = py_arr[f], cpp_arr[f]
        ax.scatter(x, y, s=4, alpha=0.25, color="#1f77b4", linewidths=0, zorder=2)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="#d62728", lw=1.4, ls="--", label="y = x", zorder=3)
        ax.set_xlabel(f"python {f} {unit}", fontweight="bold")
        ax.set_ylabel(f"cpp {f} {unit}", fontweight="bold")
        ax.legend(fontsize=8)
        _bold_ax(ax)

    scatter_ax(axes[0, 0], "submit", "(s)")
    scatter_ax(axes[0, 1], "start", "(s)")
    scatter_ax(axes[1, 0], "end", "(s)")

    ax = axes[1, 1]
    ax.hist(diffs["start"], bins=60, color="#2ca02c", alpha=0.8, edgecolor="black", linewidth=0.4, zorder=2)
    ax.axvline(0, color="#d62728", lw=1.4, ls="--")
    ax.set_xlabel("start_cpp - start_py (s)", fontweight="bold")
    ax.set_ylabel("job count", fontweight="bold")
    _bold_ax(ax)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, f"{label}_deviation.png")
    save_fig(fig, fig_path)
    out(f"\nwrote {fig_path}")

    # Relative start time: each side's start time measured from that side's
    # own first submit time, so the two runs line up even if their absolute
    # sim-time epochs differ. Linear and log-log versions -- the log view
    # spreads out the dense cluster of early, short-wait jobs.
    py_rel = py_arr["start"] - py_first_submit
    cpp_rel = cpp_arr["start"] - cpp_first_submit
    r = pearson_r(py_rel, cpp_rel)

    for log_scale in (False, True):
        rel_path = plot_start_relative(py_rel, cpp_rel, r, len(matched), label, output_dir, log_scale=log_scale)
        out(f"wrote {rel_path}")

    # Per-job start-time difference magnitude (|CQSim C++ - CQSim Python|), as a violin.
    violin_path = plot_start_diff_violin(np.abs(diffs["start"]) / 3600.0, label, output_dir)
    out(f"wrote {violin_path}")

    # CDF overlay of start time
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    for arr, name, color in [(py_arr["start"], "python", "#1f77b4"), (cpp_arr["start"], "cpp", "#ff7f0e")]:
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax2.plot(xs, ys, label=name, color=color, lw=1.8)
    ax2.set_xlabel("start time (s)", fontweight="bold")
    ax2.set_ylabel("CDF", fontweight="bold")
    ax2.legend()
    _bold_ax(ax2)
    fig2.tight_layout()
    fig2_path = os.path.join(output_dir, f"{label}_start_cdf.png")
    save_fig(fig2, fig2_path)
    out(f"wrote {fig2_path}")

    report_path = os.path.join(output_dir, f"{label}_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    out(f"wrote {report_path}")

    return {
        "label": label,
        "matched": len(matched),
        "only_py": len(only_py),
        "only_cpp": len(only_cpp),
        "diffs": diffs,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpp-events", default="results/cqsimpy-test1/FCFS/events.csv",
                     help="path to the raw C++ events.csv, e.g. results/cqsimpy-test1/FCFS/events.csv")
    ap.add_argument("--py-events", default="cqsim-python/data/Results/theta21_events.csv",
                     help="path to the python events.csv, e.g. cqsim-python/data/Results/theta21_events.csv")
    ap.add_argument("--swf", default="data/theta21cln.swf",
                     help="path to the shared .swf input trace")
    ap.add_argument("--label", default="theta21cln", help="name used for output files/titles")
    ap.add_argument("--output-dir", default="results/cpp_vs_py", help="where to write plots/report")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    for key, path in (("--cpp-events", args.cpp_events), ("--py-events", args.py_events), ("--swf", args.swf)):
        if not os.path.exists(path):
            print(f"[error] missing file for {key}: {path}", file=sys.stderr)
            sys.exit(1)

    compare(args.label, args.swf, args.py_events, args.cpp_events,
            os.path.join(args.output_dir, args.label))

    # Combined Polaris-vs-Theta figure -- only emitted once both traces'
    # raw C++ events.csv exist (see TRACE_FILES); a no-op until then.
    build_polaris_theta_violin(args.output_dir)


if __name__ == "__main__":
    main()
