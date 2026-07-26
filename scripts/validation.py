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
import numpy as np

DPI = 150

# ── EDIT ME ─────────────────────────────────────────────────────────────────
# How many hours of simulation time (since each trace's first submit) to
# show in the combined Polaris-vs-Theta error/cumulative-submissions plot
# (see plot_error_and_submissions_comparison). Change this to zoom in/out.
ERROR_VS_TIME_WINDOW_HOURS = 720

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


# Known trace file locations, used by the combined Polaris-vs-Theta figure
# builders to build their figures once both traces' C++ runs exist.
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


def compute_deviation_timeseries_hours(py_events_path, cpp_events_path):
    """Per matched job: (hours since this trace's first submit, |CQSim
    C++ start - CQSim Python start| in hours) -- the two series plotted by
    plot_error_and_submissions_comparison."""
    py, py_first_submit = load_events(py_events_path)
    cpp, _ = load_events(cpp_events_path)
    matched = sorted(set(py) & set(cpp))
    if not matched:
        return np.array([]), np.array([])
    sim_hours = np.array([(py[j]["submit"] - py_first_submit) / 3600.0 for j in matched])
    dev_hours = np.array([abs(cpp[j]["start"] - py[j]["start"]) / 3600.0 for j in matched])
    return sim_hours, dev_hours


# ─── Metrics ──────────────────────────────────────────────────────────────────

def pearson_r(x, y):
    if np.std(x) > 0 and np.std(y) > 0:
        return float(np.corrcoef(x, y)[0, 1])
    return float("nan")


def loess_smooth(x, y, frac=0.08, n_eval=300):
    """Locally weighted linear regression (LOESS/LOWESS, degree 1, tricube
    kernel), evaluated at `n_eval` evenly spaced points across x's range.

    No statsmodels dependency -- for each evaluation point, takes the
    nearest `frac`-fraction of points (by x-distance), weights them with
    the tricube kernel, and fits a weighted least-squares line locally.
    Caller should pass already-transformed y (e.g. log10) for heavy-tailed
    data, since this is not a robust/iterative LOESS.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n = len(xs)
    r = max(int(np.ceil(frac * n)), 2)

    x_eval = np.linspace(xs[0], xs[-1], n_eval)
    y_eval = np.empty(n_eval)
    for i, x0 in enumerate(x_eval):
        idx = np.searchsorted(xs, x0)
        lo = max(0, min(idx - r // 2, n - r))
        hi = lo + r
        xw, yw = xs[lo:hi], ys[lo:hi]

        d = np.abs(xw - x0)
        dmax = d.max()
        w = (1 - (d / dmax) ** 3) ** 3 if dmax > 0 else np.ones_like(d)
        w = np.clip(w, 0, None)

        sw, sx, sy = w.sum(), (w * xw).sum(), (w * yw).sum()
        sxx, sxy = (w * xw * xw).sum(), (w * xw * yw).sum()
        denom = sw * sxx - sx * sx
        if abs(denom) < 1e-12 or sw <= 0:
            y_eval[i] = sy / sw if sw > 0 else np.nan
        else:
            b = (sw * sxy - sx * sy) / denom
            a = (sy - b * sx) / sw
            y_eval[i] = a + b * x0
    return x_eval, y_eval


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


_TRACE_DISPLAY_NAMES = {
    "theta21cln": "Theta 2021",
    "polaris24cln": "Polaris 2024",
}

_TRACE_ORDER = ["polaris24cln", "theta21cln"]
_TRACE_FACE_COLORS = {"polaris24cln": "#2ca02c", "theta21cln": "#1f77b4"}


def _style_violin_bodies(parts, color):
    for body in parts["bodies"]:
        body.set_facecolor(color)
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




def plot_polaris_theta_violin_separate_scales(start_diff_by_label, output_dir):
    """|Δ start time| for Polaris and Theta as two side-by-side violins,
    each in its OWN subplot with its OWN y-axis scale -- unlike a single
    shared-scale violin, this keeps Theta's much smaller distribution
    visible instead of being
    flattened to a sliver by Polaris's much larger range. Compact, sized
    to sit at roughly a quarter of a row's width in a two-column LaTeX
    paper figure. One shared legend for both panels (median/mean/P99 mean
    the same thing in both, so a single legend avoids duplicating it).
    """
    violin_width = 0.5
    half = violin_width / 2 + 0.05

    fig, axes = plt.subplots(1, 2, figsize=(3, 3))

    for ax, lbl in zip(axes, _TRACE_ORDER):
        diff = start_diff_by_label[lbl]
        color = _TRACE_FACE_COLORS[lbl]
        parts = ax.violinplot([diff], positions=[1], widths=violin_width,
                               showmeans=True, showmedians=True, showextrema=True)
        _style_violin_bodies(parts, color)

        median_v = float(np.median(diff))
        p99_v = float(np.percentile(diff, 99))
        ax.hlines(p99_v, 1 - half, 1 + half, color="#8c564b", lw=1.1, ls="-.", zorder=1.5)
        ax.text(1, p99_v, f"P99={p99_v:.1f}", fontsize=7, fontweight="bold",
                color="#8c564b", ha="center", va="bottom")
        ax.text(1, median_v, f"med={median_v:.1f}", fontsize=7, fontweight="bold",
                color="black", ha="center", va="bottom")

        ax.axhline(0, color="#d62728", lw=1.0, ls="--", zorder=1)
        ax.set_xticks([1])
        ax.set_xticklabels([_TRACE_DISPLAY_NAMES.get(lbl, lbl)])
        ax.set_xlim(0.5, 1.5)
        # Each panel gets its own scale, sized to its own data --
        # the whole point of splitting these into separate subplots.
        ax.set_ylim(0, p99_v * 1.15)
        _bold_ax(ax)
        ax.tick_params(labelsize=8.5)

    axes[0].set_ylabel("|Δ Start Time| (hours)", fontsize=9.5, fontweight="bold", labelpad=3)

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.8, label="median"),
        Line2D([0], [0], color="red", lw=1.5, ls=":", label="mean"),
        Line2D([0], [0], color="#8c564b", lw=1.1, ls="-.", label="P99"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.6, 0.94),
               ncol=3, fontsize=8, frameon=True, columnspacing=1.0, handlelength=1.6)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "polaris_theta_start_diff_violin.png")
    save_fig(fig, fig_path)
    return fig_path


def build_polaris_theta_violin_separate_scales(output_dir):
    """Emit the separate-scales Polaris/Theta violin figure once both
    traces' raw C++ events.csv exist. Skips (with a note, not an error)
    while either side is still missing."""
    start_diff_by_label = {}
    for lbl, paths in TRACE_FILES.items():
        missing = [paths[k] for k in ("cpp_events", "py_events") if not os.path.exists(paths[k])]
        if missing:
            print(f"[skip separate-scales violin] {lbl}: missing {missing[0]}")
            return None
        start_diff_by_label[lbl] = compute_start_diff_hours(paths["py_events"], paths["cpp_events"])

    fig_path = plot_polaris_theta_violin_separate_scales(start_diff_by_label, output_dir)
    print(f"wrote {fig_path}")
    return fig_path


def _plot_loess_curve(ax, x, y, color, label, x_log, y_log):
    """Fit+plot a LOESS trend of y(x) onto ax, in whichever space (log10 or
    raw) each axis is displaying. Shared by the error panel and the
    rate_loess top-panel variant so both smooth identically."""
    fit_mask = (y > 0) if y_log else np.ones_like(y, dtype=bool)
    if x_log:
        fit_mask &= x > 0
    if fit_mask.sum() < 10:
        return
    x_fit = np.log10(x[fit_mask]) if x_log else x[fit_mask]
    y_fit = np.log10(y[fit_mask]) if y_log else y[fit_mask]
    x_eval, y_smooth = loess_smooth(x_fit, y_fit)
    x_plot = 10 ** x_eval if x_log else x_eval
    y_plot = 10 ** y_smooth if y_log else np.clip(y_smooth, 0, None)
    ax.plot(x_plot, y_plot, color=color, lw=2.4, label=label, zorder=2)


def plot_error_and_submissions_comparison(series_by_label, output_dir, window_hours):
    """Two side-by-side panels sharing a linear simulation-time (days)
    x-axis, Polaris and Theta overlaid in their trace colors:
        left  -- |Δ start time|, LOESS trend only (no raw scatter)
        right -- cumulative jobs submitted so far (step function)
    `window_hours` caps how much of the run is shown -- see the
    ERROR_VS_TIME_WINDOW_HOURS constant near the top of this file (still
    expressed/tunable in hours; only the displayed axis is in days).
    A shared dotted-red vertical line marks Polaris's busiest *sustained*
    (12h rolling-sum) submission window on both panels, since that's what
    lines up with its deviation spike -- not the single tallest 1h bin,
    which turns out to be an isolated early outlier with no downstream
    deviation response.
    """
    window_days = window_hours / 24.0
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.6, 3.6), sharex=True)
    bins = np.arange(0, window_hours + 1.0, 1.0)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for lbl in _TRACE_ORDER:
        sim_hours, dev_hours = series_by_label.get(lbl, (np.array([]), np.array([])))
        if sim_hours.size == 0:
            continue
        mask = (sim_hours > 0) & (sim_hours <= window_hours)
        x, y = sim_hours[mask] / 24.0, dev_hours[mask]
        color = _TRACE_FACE_COLORS[lbl]
        name = _TRACE_DISPLAY_NAMES.get(lbl, lbl)

        _plot_loess_curve(ax_l, x, y, color, name, x_log=False, y_log=False)

        x_sorted = np.sort(x)
        cum_counts = np.arange(1, len(x_sorted) + 1)
        ax_r.step(x_sorted, cum_counts, where="post", color=color, lw=3.2, label=name, zorder=2)

    # LOESS curve on the left panel defaults to lw=2.4; thicken to match
    # the rest of the figure's heavier line weight.
    for line in ax_l.get_lines():
        line.set_linewidth(3.2)

    # Vertical marker at Polaris's busiest sustained submission window
    # (not in the legend -- keep it to just the two trace names). Detected
    # in hours (finer 12h rolling sum) then converted to days for display.
    polaris_hours, _ = series_by_label.get("polaris24cln", (np.array([]), np.array([])))
    if polaris_hours.size:
        p_mask = (polaris_hours > 0) & (polaris_hours <= window_hours)
        p_counts, _ = np.histogram(polaris_hours[p_mask], bins=bins)
        p_rolling = np.convolve(p_counts, np.ones(12), mode="same")
        if p_rolling.max() > 0:
            spike_x = float(bin_centers[np.argmax(p_rolling)]) / 24.0
            for ax_ in (ax_l, ax_r):
                ax_.axvline(spike_x, color="#d62728", linestyle=":", linewidth=2.4, zorder=5)

    ax_l.set_xlim(0, window_days)
    ax_l.set_ylabel("|Δ Start Time| (hours)\n[LOESS trend]", fontsize=13, fontweight="bold", labelpad=4)
    ax_l.set_xlabel(f"Simulation Time\n({window_days:.0f} days)", fontsize=12, fontweight="bold", labelpad=4)
    _bold_ax(ax_l)
    ax_l.tick_params(labelsize=12)

    ax_r.set_xlim(0, window_days)
    ax_r.set_ylabel("Cumulative Jobs\nSubmitted", fontsize=13, fontweight="bold", labelpad=4)
    ax_r.set_xlabel(f"Simulation Time\n({window_days:.0f} days)", fontsize=12, fontweight="bold", labelpad=4)
    _bold_ax(ax_r)
    ax_r.tick_params(labelsize=12)

    # One shared legend for the whole figure instead of one per panel --
    # just the two trace names, not the spike marker.
    handles, labels = ax_l.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.54, 1.0),
               ncol=2, fontsize=12, frameon=True, columnspacing=1.2)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, f"polaris_theta_error_and_submissions_first{int(window_hours)}h_xlinear_ylinear.png")
    save_fig(fig, fig_path)
    return fig_path


def build_error_and_submissions_comparison(output_dir, window_hours=None):
    """Emit the combined Polaris-vs-Theta error/submissions figure once
    both traces' raw C++ events.csv exist. Skips (with a note, not an
    error) while either side is still missing."""
    if window_hours is None:
        window_hours = ERROR_VS_TIME_WINDOW_HOURS

    series_by_label = {}
    for lbl, paths in TRACE_FILES.items():
        missing = [paths[k] for k in ("cpp_events", "py_events") if not os.path.exists(paths[k])]
        if missing:
            print(f"[skip error/submissions comparison] {lbl}: missing {missing[0]}")
            return None
        series_by_label[lbl] = compute_deviation_timeseries_hours(paths["py_events"], paths["cpp_events"])

    fig_path = plot_error_and_submissions_comparison(series_by_label, output_dir, window_hours)
    print(f"wrote {fig_path}")
    return fig_path


# ─── Core comparison ──────────────────────────────────────────────────────────

def compare(label, swf_path, py_events_path, cpp_events_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    ground_truth = parse_swf(swf_path)
    py, _ = load_events(py_events_path)
    cpp, _ = load_events(cpp_events_path)

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

    # Combined Polaris-vs-Theta figures -- only emitted once both traces'
    # raw C++ events.csv exist (see TRACE_FILES); a no-op until then.
    build_error_and_submissions_comparison(args.output_dir)
    build_polaris_theta_violin_separate_scales(args.output_dir)


if __name__ == "__main__":
    main()
