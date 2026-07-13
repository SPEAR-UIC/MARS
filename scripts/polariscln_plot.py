#!/usr/bin/env python3
"""Plotting script for cleaned Polaris trace experiments.

Job-size groups (consistent colors throughout):
  S — Small  : 10–24  nodes   (steelblue)
  M — Medium : 25–99  nodes   (darkorange)
  L — Large  : 100–496 nodes  (seagreen)

Per-driver output files  (<plots_dir>/<driver_tag>/):
  cycletime_vs_time.png      — X: time  Y: cycle time (s)
  freeprocs_vs_time.png      — X: time  Y: free procs
  throughput_jobs_per_day.png — X: day  Y: completed jobs/day with 7-day average
  throughput_proc_hours_per_day.png — X: day  Y: completed proc-hours/day with 7-day average
  cdf_wait_S.png             — wait-time CDF for S group, P25/P50/P75/P99 marked
  cdf_wait_M.png             — wait-time CDF for M group
  cdf_wait_L.png             — wait-time CDF for L group
  cdf_bsld_S.png             — BSLD CDF for S group, P25/P50/P75/P99 marked
  cdf_bsld_M.png             — BSLD CDF for M group
  cdf_bsld_L.png             — BSLD CDF for L group
  queuelength_vs_time.png    — X: time  Y: queue length
  scatter_wait_vs_size.png   — X: node size (log) Y: wait time (h)

Global output files  (<plots_dir>/):
  cdf_wait_cdf.png           — wait-time CDF, all groups, all drivers
  cdf_bsld.png               — BSLD CDF, all groups, all drivers
  throughput_jobs_per_day.png — 7-day average completed jobs/day, all drivers
  throughput_proc_hours_per_day.png — 7-day average completed proc-hours/day, all drivers
  cumulative_completed_jobs.png — cumulative completed jobs vs time, all drivers
  table_overall_wait.csv     — overall wait-time stats (minutes)
  table_overall_bsld.csv     — overall BSLD stats
  cdf_wait_S.png             — wait-time CDF, S group, all drivers
  cdf_wait_M.png             — wait-time CDF, M group, all drivers
  cdf_wait_L.png             — wait-time CDF, L group, all drivers
  cdf_bsld_S.png             — BSLD CDF, S group, all drivers
  cdf_bsld_M.png             — BSLD CDF, M group, all drivers
  cdf_bsld_L.png             — BSLD CDF, L group, all drivers
  table_S_wait.csv           — wait-time stats (minutes): Min/P25/P50/P75/P99/Max per driver
  table_M_wait.csv           — same for M group
  table_L_wait.csv           — same for L group
  table_S_bsld.csv           — BSLD stats: Min/P25/P50/P75/P99/Max per driver
  table_M_bsld.csv           — same for M group
  table_L_bsld.csv           — same for L group

RLScheduler support:
    Add a driver entry with "type": "rlscheduler" and "csv_path" pointing to a
    pre-computed RLScheduler results CSV (columns: job_id, submit_time,
    scheduled_time, run_time).  Per-driver time-series plots are skipped for
    this driver type; all global comparison plots and tables are included.

    Example driver entry in the experiment JSON:
        {"type": "rlscheduler", "tag": "RLScheduler",
         "csv_path": "data/polaris24cln_rlscheduler.csv"}

Usage:
    python3 scripts/polariscln_plot.py experiments/exp1b.json
    python3 scripts/polariscln_plot.py experiments/exp7b.json --plots-dir results/custom_plots
"""

import sys
import os
import json
import csv
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
from plot_results import (
    build_cumulative_completion_series,
    build_daily_time_series,
    parse_events,
    parse_swf,
    rolling_average,
)

# ── Style constants ───────────────────────────────────────────────────────────
FONT_SIZE   = 13
TICK_SIZE   = 12
LEGEND_SIZE = 12
LINE_WIDTH  = 1.8
FIG_W       = 10
FIG_H       = 6
DPI         = 150

# Polaris system size (total nodes)
S_SYS = 496

# ── Polaris job-size groups ───────────────────────────────────────────────────
GROUP_LABELS  = ['S', 'M', 'L']
GROUP_DISPLAY = {
    'S': 'Small (10–24)',
    'M': 'Medium (25–99)',
    'L': 'Large (100–496)',
}
# Fixed group colors — consistent across every plot in the experiment
GROUP_COLORS = {'S': 'steelblue', 'M': 'darkorange', 'L': 'seagreen'}

PERCENTILES = [25, 50, 75, 99]

# Argonne / Chicago local time — matches Polaris submit timestamps
CST = timezone(timedelta(hours=-6))
ANALYSIS_YEAR = 2024

# Driver-comparison palette (tab10, stable assignment by driver index)
_DRIVER_PALETTE = list(plt.cm.tab10.colors)

# Shaped markers for percentiles on multi-line CDF plots
_PCTILE_MARKER = {25: '^', 50: 'D', 75: 's', 99: '*'}
_PCTILE_SIZE   = {25: 7,   50: 8,   75: 7,   99: 10}


# ── Maintenance window helpers ────────────────────────────────────────────────

# Visual style for maintenance spans
_MAINT_SCHED   = dict(color='#a8c8e8', alpha=0.30, hatch='//',  edgecolor='#4477aa', lw=0.5)
_MAINT_UNSCHED = dict(color='#f4b8b8', alpha=0.30, hatch='\\\\', edgecolor='#cc4444', lw=0.5)
_MAINT_LEGEND  = [
    matplotlib.patches.Patch(facecolor='#a8c8e8', alpha=0.70, hatch='//',
                             edgecolor='#4477aa', label='Scheduled maintenance'),
    matplotlib.patches.Patch(facecolor='#f4b8b8', alpha=0.70, hatch='\\\\',
                             edgecolor='#cc4444', label='Unscheduled maintenance'),
]


def analysis_window_bounds():
    start = datetime(ANALYSIS_YEAR, 1, 1, tzinfo=CST)
    end = datetime(ANALYSIS_YEAR + 1, 1, 1, tzinfo=CST)
    return start.timestamp(), end.timestamp()


def filter_maintenance_to_window(maintenance, t_start, t_end):
    filtered = []
    for start, end, mtype in maintenance:
        ov_start = max(start, t_start)
        ov_end = min(end, t_end)
        if ov_end > ov_start:
            filtered.append((ov_start, ov_end, mtype))
    return filtered


def filter_jobs_by_submit_window(submit, start, end, t_start, t_end):
    allowed = {
        job_id
        for job_id, submit_time in submit.items()
        if t_start <= submit_time < t_end
    }
    submit_f = {job_id: submit_time for job_id, submit_time in submit.items() if job_id in allowed}
    start_f = {job_id: start_time for job_id, start_time in start.items() if job_id in allowed}
    end_f = {job_id: end_time for job_id, end_time in end.items() if job_id in allowed}
    return submit_f, start_f, end_f


def sum_used_core_seconds(start_map, end_map, procs_map, t_start, t_end):
    used = 0.0
    for jid, t_e in end_map.items():
        t_s = start_map.get(jid)
        if t_s is None:
            continue
        ov_s = max(t_s, t_start)
        ov_e = min(t_e, t_end)
        if ov_e > ov_s:
            used += (ov_e - ov_s) * procs_map.get(jid, 0)
    return used


def parse_maintenance_from_events(events_csv_path):
    """Derive maintenance windows from an events.csv produced by the simulator.

    Reads SMA/SMS/SME (scheduled) and UMS/UME (unscheduled) rows, where id is
    the maintenance window index.  Scheduled windows are tagged 'S'; unscheduled
    windows are tagged 'U'.

    Returns a list of (start_unix, end_unix, 'S'|'U') tuples sorted by start time.
    """
    sma_times = {}   # window_idx -> announcement sim_time
    sms_times = {}   # window_idx -> start sim_time (scheduled)
    sme_times = {}   # window_idx -> end sim_time (scheduled)
    ums_times = {}   # window_idx -> start sim_time (unscheduled)
    ume_times = {}   # window_idx -> end sim_time (unscheduled)

    try:
        with open(events_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                event = row.get("event", "")
                if event not in ("SMA", "SMS", "SME", "UMS", "UME"):
                    continue
                try:
                    idx = int(row["id"])
                    t   = float(row["sim_time"])
                except (KeyError, ValueError):
                    continue
                if event == "SMA":
                    sma_times[idx] = t
                elif event == "SMS":
                    sms_times[idx] = t
                elif event == "SME":
                    sme_times[idx] = t
                elif event == "UMS":
                    ums_times[idx] = t
                elif event == "UME":
                    ume_times[idx] = t
    except FileNotFoundError:
        print(f"  [warn] events.csv not found: {events_csv_path}")
        return []

    windows = []
    # Scheduled windows: SMS→SME pairs
    for idx in sorted(sms_times):
        if idx not in sme_times:
            continue
        windows.append((int(sms_times[idx]), int(sme_times[idx]), 'S'))
    # Unscheduled windows: UMS→UME pairs
    for idx in sorted(ums_times):
        if idx not in ume_times:
            continue
        windows.append((int(ums_times[idx]), int(ume_times[idx]), 'U'))

    return sorted(windows, key=lambda w: w[0])


def add_maintenance_shading(ax, maintenance, tz):
    """Overlay scheduled (blue) and unscheduled (red) maintenance spans on *ax*.

    maintenance — list of (start_unix, end_unix, 'S'|'U') from load_mlog()
    tz          — timezone used on the x-axis (must match what the time-series uses)

    Adds a small 'S' / 'U' label at the top of each span and returns the
    legend handles so the caller can add them to the legend.
    Returns True if any spans were drawn (so caller knows to add legend entries).
    """
    if not maintenance:
        return False

    ylims = ax.get_ylim()
    label_y = ylims[1] * 0.97  # just below the top edge

    drawn = False
    for start_u, end_u, mtype in maintenance:
        start_dt = datetime.fromtimestamp(start_u, tz=tz)
        end_dt   = datetime.fromtimestamp(end_u,   tz=tz)
        style    = _MAINT_SCHED if mtype == 'S' else _MAINT_UNSCHED
        ax.axvspan(start_dt, end_dt, **style)

        # Centered label at the top of the span
        mid_dt = start_dt + (end_dt - start_dt) / 2
        ax.text(mid_dt, label_y, mtype,
                ha='center', va='top', fontsize=8, fontweight='bold',
                color='#4477aa' if mtype == 'S' else '#cc4444',
                clip_on=True)
        drawn = True

    return drawn


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_backfill_jobs(events_csv_path):
    """Return set of job IDs whose first start event was a Backfill."""
    backfill_ids = set()
    run_ids = set()
    with open(events_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                event  = row["event"]
                job_id = int(row["id"])
            except (KeyError, ValueError, TypeError):
                continue
            if event == "Backfill" and job_id not in run_ids:
                backfill_ids.add(job_id)
            elif event == "Run":
                run_ids.add(job_id)
                backfill_ids.discard(job_id)
    return backfill_ids


def parse_rlscheduler_csv(csv_path):
    """Parse an RLScheduler results CSV into the same (submit, start, end) triple
    as parse_events().

    Expected columns: job_id, submit_time, scheduled_time, run_time
    On duplicate job_id rows the last row wins for start/end (first submit kept).

    Returns:
        submit — {job_id: submit_time}
        start  — {job_id: scheduled_time}
        end    — {job_id: scheduled_time + run_time}
    """
    submit = {}
    start  = {}
    end    = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                job_id  = int(row['job_id'])
                sub_t   = float(row['submit_time'])
                sched_t = float(row['scheduled_time'])
                run_t   = float(row['run_time'])
            except (KeyError, ValueError, TypeError):
                continue
            if job_id not in submit:
                submit[job_id] = sub_t
            start[job_id] = sched_t
            end[job_id]   = sched_t + run_t

    return submit, start, end


def assign_group(num_procs):
    """Return 'S', 'M', 'L', or None."""
    if 10 <= num_procs <= 24:
        return 'S'
    if 25 <= num_procs <= 99:
        return 'M'
    if 100 <= num_procs <= 496:
        return 'L'
    return None


def apply_style(ax, xlabel=None, ylabel=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


def driver_color(idx):
    return _DRIVER_PALETTE[idx % len(_DRIVER_PALETTE)]


def _compute_cdf(values):
    """Return (sorted_values, cdf_fractions)."""
    sv = np.sort(values)
    cdf = np.arange(1, len(sv) + 1) / len(sv)
    return sv, cdf


def _pctile_legend_handles():
    """Legend entries for percentile marker shapes and mean."""
    handles = [
        Line2D([0], [0], marker=_PCTILE_MARKER[p], color='gray',
               markersize=_PCTILE_SIZE[p], linestyle='None', label=f'P{p}',
               markeredgecolor='white', markeredgewidth=0.5)
        for p in PERCENTILES
    ]
    handles.append(Line2D([0], [0], marker='P', color='gray', markersize=9,
                           linestyle='None', label='Mean',
                           markeredgecolor='white', markeredgewidth=0.5))
    return handles


# ── CDF drawing — two modes ───────────────────────────────────────────────────

def _draw_cdf_single(ax, values, color):
    """Single-line CDF: vertical dashed annotations for each percentile and mean.

    Returns {p: value} with the computed percentile values, or {} if no data.
    """
    if not values:
        return {}
    sv, cdf = _compute_cdf(values)
    ax.plot(sv, cdf, color=color, linewidth=LINE_WIDTH)

    pctile_values = {}
    for p in PERCENTILES:
        pv = float(np.percentile(sv, p))
        pctile_values[p] = pv
        idx = max(0, int(np.searchsorted(sv, pv, side='right')) - 1)
        cdf_at_p = float(cdf[min(idx, len(cdf) - 1)])
        ax.axvline(pv, color=color, linestyle='--', alpha=0.55, linewidth=1.1)
        ax.plot(pv, cdf_at_p, 'o', color=color, markersize=5, zorder=5)
        ax.text(pv, 1.02, f'P{p}', color=color, fontsize=10,
                ha='center', va='bottom', clip_on=False)

    mean_v = float(np.mean(sv))
    mean_idx = max(0, int(np.searchsorted(sv, mean_v, side='right')) - 1)
    cdf_at_mean = float(cdf[min(mean_idx, len(cdf) - 1)])
    ax.axvline(mean_v, color=color, linestyle=':', alpha=0.8, linewidth=1.4)
    ax.plot(mean_v, cdf_at_mean, 's', color=color, markersize=6, zorder=5)
    ax.text(mean_v, 1.02, 'Mean', color=color, fontsize=10,
            ha='center', va='bottom', clip_on=False)

    return pctile_values


def _draw_cdf_multi(ax, values, color, label):
    """Multi-line CDF: shaped markers at each percentile and mean, label in legend."""
    if not values:
        return
    sv, cdf = _compute_cdf(values)
    ax.plot(sv, cdf, color=color, linewidth=LINE_WIDTH, label=label)

    for p in PERCENTILES:
        pv = float(np.percentile(sv, p))
        idx = max(0, int(np.searchsorted(sv, pv, side='right')) - 1)
        cdf_at_p = float(cdf[min(idx, len(cdf) - 1)])
        ax.plot(pv, cdf_at_p, _PCTILE_MARKER[p], color=color,
                markersize=_PCTILE_SIZE[p], zorder=5,
                markeredgecolor='white', markeredgewidth=0.5)

    mean_v = float(np.mean(sv))
    mean_idx = max(0, int(np.searchsorted(sv, mean_v, side='right')) - 1)
    cdf_at_mean = float(cdf[min(mean_idx, len(cdf) - 1)])
    ax.plot(mean_v, cdf_at_mean, 'P', color=color, markersize=9, zorder=5,
            markeredgecolor='white', markeredgewidth=0.5)


# ── Per-driver: Cycle Time vs Time ────────────────────────────────────────────

def plot_cycletime_vs_time(tag, results_dir, plots_dir, time_start_limit=None, time_end_limit=None):
    perf_path = os.path.join(results_dir, tag, "performance.csv")
    if not os.path.exists(perf_path):
        print(f"  [skip] {tag}: performance.csv not found")
        return

    times, cycle_s = [], []
    with open(perf_path) as f:
        reader = csv.DictReader(f)
        if "cycle_ms" not in (reader.fieldnames or []):
            print(f"  [skip] {tag}: 'cycle_ms' column not found")
            return
        for row in reader:
            try:
                raw_t = float(row["sim_time"])
                if time_start_limit is not None and raw_t < time_start_limit:
                    continue
                if time_end_limit is not None and raw_t >= time_end_limit:
                    continue
                t = datetime.fromtimestamp(raw_t, tz=CST)
                c = float(row["cycle_ms"]) / 1000.0
            except (ValueError, KeyError, TypeError):
                continue
            times.append(t)
            cycle_s.append(c)

    if not times:
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(times, cycle_s, linewidth=LINE_WIDTH, color='steelblue', alpha=0.85)
    apply_style(ax, xlabel="Time (MM/DD HH:MM)", ylabel="Cycle Time (s)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, "cycletime_vs_time.png"))


# ── Per-driver: Free Procs vs Time ───────────────────────────────────────────

def plot_freeprocs_vs_time(tag, results_dir, plots_dir, maintenance=None,
                           time_start_limit=None, time_end_limit=None):
    perf_path = os.path.join(results_dir, tag, "performance.csv")
    if not os.path.exists(perf_path):
        print(f"  [skip] {tag}: performance.csv not found")
        return

    times, fp = [], []
    with open(perf_path) as f:
        reader = csv.DictReader(f)
        if "util" not in (reader.fieldnames or []):
            print(f"  [skip] {tag}: 'util' column not found")
            return
        for row in reader:
            try:
                raw_t = float(row["sim_time"])
                if time_start_limit is not None and raw_t < time_start_limit:
                    continue
                if time_end_limit is not None and raw_t >= time_end_limit:
                    continue
                t = datetime.fromtimestamp(raw_t, tz=CST)
                v = round(float(row["util"]) * S_SYS)
            except (ValueError, KeyError, TypeError):
                continue
            times.append(t)
            fp.append(v)

    if not times:
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(times, fp, linewidth=LINE_WIDTH, color='darkorange', alpha=0.85, zorder=3)
    apply_style(ax, xlabel="Time (MM/DD HH:MM)", ylabel="Number of Free Procs")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    if maintenance:
        has_spans = add_maintenance_shading(ax, maintenance, CST)
        if has_spans:
            existing_handles, existing_labels = ax.get_legend_handles_labels()
            ax.legend(handles=existing_handles + _MAINT_LEGEND,
                      labels=existing_labels + [h.get_label() for h in _MAINT_LEGEND],
                      fontsize=LEGEND_SIZE - 1, loc='upper left')

    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, "freeprocs_vs_time.png"))


# ── Per-driver: Queue Length vs Time ─────────────────────────────────────────

def plot_queuelength_vs_time(tag, results_dir, plots_dir, maintenance=None,
                             time_start_limit=None, time_end_limit=None):
    perf_path = os.path.join(results_dir, tag, "performance.csv")
    if not os.path.exists(perf_path):
        print(f"  [skip] {tag}: performance.csv not found")
        return

    times, ql = [], []
    with open(perf_path) as f:
        reader = csv.DictReader(f)
        if "queue_len" not in (reader.fieldnames or []):
            print(f"  [skip] {tag}: 'queue_len' column not found")
            return
        for row in reader:
            try:
                raw_t = float(row["sim_time"])
                if time_start_limit is not None and raw_t < time_start_limit:
                    continue
                if time_end_limit is not None and raw_t >= time_end_limit:
                    continue
                t = datetime.fromtimestamp(raw_t, tz=CST)
                v = int(row["queue_len"])
            except (ValueError, KeyError, TypeError):
                continue
            times.append(t)
            ql.append(v)

    if not times:
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(times, ql, linewidth=LINE_WIDTH, color='seagreen', alpha=0.85, zorder=3)
    apply_style(ax, xlabel="Time (MM/DD HH:MM)", ylabel="Queue Length")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    if maintenance:
        has_spans = add_maintenance_shading(ax, maintenance, CST)
        if has_spans:
            existing_handles, existing_labels = ax.get_legend_handles_labels()
            ax.legend(handles=existing_handles + _MAINT_LEGEND,
                      labels=existing_labels + [h.get_label() for h in _MAINT_LEGEND],
                      fontsize=LEGEND_SIZE - 1, loc='upper left')

    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, "queuelength_vs_time.png"))


def _completed_proc_hours_by_job(end_map, procs_map, runtimes_map):
    values = {}
    for jid in end_map:
        try:
            procs = float(procs_map[jid])
            runtime = float(runtimes_map[jid])
        except (KeyError, TypeError, ValueError):
            continue
        values[jid] = max(procs, 0.0) * max(runtime, 0.0) / 3600.0
    return values


def _plot_daily_throughput_series(days, values, output_path, ylabel, maintenance=None,
                                  bar_label=None, bar_color="#7a68a6",
                                  avg_label="7-day average", avg_color="#1f1f1f"):
    if not days:
        return

    avg = rolling_average(values, 7)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.bar(days, values, width=0.85, color=bar_color, alpha=0.55,
           label=bar_label, zorder=2)
    ax.plot(days, avg, color=avg_color, linewidth=LINE_WIDTH,
            label=avg_label, zorder=3)
    ax.set_ylim(bottom=0)
    apply_style(ax, xlabel="Date", ylabel=ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    if maintenance:
        has_spans = add_maintenance_shading(ax, maintenance, CST)
        if has_spans:
            existing_handles, existing_labels = ax.get_legend_handles_labels()
            ax.legend(handles=existing_handles + _MAINT_LEGEND,
                      labels=existing_labels + [h.get_label() for h in _MAINT_LEGEND],
                      fontsize=LEGEND_SIZE - 1, loc='upper left')
    elif bar_label or avg_label:
        ax.legend(fontsize=LEGEND_SIZE - 1, loc='upper left')

    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, output_path)


def plot_throughput_jobs_per_day(tag, end_map, plots_dir, maintenance=None, end_time_limit=None):
    days, values = build_daily_time_series(end_map, CST, end_time_limit=end_time_limit)
    _plot_daily_throughput_series(
        days,
        values,
        os.path.join(plots_dir, tag, "throughput_jobs_per_day.png"),
        ylabel="Completed Jobs / Day",
        maintenance=maintenance,
        bar_label="Daily completed jobs",
        bar_color="#8172b3",
    )


def plot_throughput_proc_hours_per_day(tag, end_map, procs_map, runtimes_map,
                                       plots_dir, maintenance=None, end_time_limit=None):
    proc_hours_by_job = _completed_proc_hours_by_job(end_map, procs_map, runtimes_map)
    days, values = build_daily_time_series(
        end_map,
        CST,
        value_by_job=proc_hours_by_job,
        end_time_limit=end_time_limit,
    )
    _plot_daily_throughput_series(
        days,
        values,
        os.path.join(plots_dir, tag, "throughput_proc_hours_per_day.png"),
        ylabel="Completed Proc-Hours / Day",
        maintenance=maintenance,
        bar_label="Daily completed proc-hours",
        bar_color="#55a868",
        avg_color="#0f5132",
    )


def plot_global_throughput_jobs_per_day(driver_tags, end_maps, driver_colors, plots_dir,
                                        maintenance=None, end_time_limit=None):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    plotted = False
    for tag in driver_tags:
        days, values = build_daily_time_series(
            end_maps.get(tag, {}),
            CST,
            end_time_limit=end_time_limit,
        )
        if not days:
            continue
        ax.plot(days, rolling_average(values, 7),
                linewidth=LINE_WIDTH, color=driver_colors[tag],
                label=tag, zorder=3)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_ylim(bottom=0)
    apply_style(ax, xlabel="Date", ylabel="Completed Jobs / Day (7-day avg)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    if maintenance:
        has_spans = add_maintenance_shading(ax, maintenance, CST)
        if has_spans:
            existing_handles, existing_labels = ax.get_legend_handles_labels()
            ax.legend(handles=existing_handles + _MAINT_LEGEND,
                      labels=existing_labels + [h.get_label() for h in _MAINT_LEGEND],
                      fontsize=LEGEND_SIZE - 1, loc='upper left', ncol=2)
        else:
            ax.legend(fontsize=LEGEND_SIZE - 1, loc='upper left', ncol=2)
    else:
        ax.legend(fontsize=LEGEND_SIZE - 1, loc='upper left', ncol=2)

    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, "throughput_jobs_per_day.png"))


def plot_global_throughput_proc_hours_per_day(driver_tags, end_maps, driver_colors,
                                              procs_map, runtimes_map, plots_dir,
                                              maintenance=None, end_time_limit=None):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    plotted = False
    for tag in driver_tags:
        proc_hours_by_job = _completed_proc_hours_by_job(end_maps.get(tag, {}), procs_map, runtimes_map)
        days, values = build_daily_time_series(
            end_maps.get(tag, {}),
            CST,
            value_by_job=proc_hours_by_job,
            end_time_limit=end_time_limit,
        )
        if not days:
            continue
        ax.plot(days, rolling_average(values, 7),
                linewidth=LINE_WIDTH, color=driver_colors[tag],
                label=tag, zorder=3)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_ylim(bottom=0)
    apply_style(ax, xlabel="Date", ylabel="Completed Proc-Hours / Day (7-day avg)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    if maintenance:
        has_spans = add_maintenance_shading(ax, maintenance, CST)
        if has_spans:
            existing_handles, existing_labels = ax.get_legend_handles_labels()
            ax.legend(handles=existing_handles + _MAINT_LEGEND,
                      labels=existing_labels + [h.get_label() for h in _MAINT_LEGEND],
                      fontsize=LEGEND_SIZE - 1, loc='upper left', ncol=2)
        else:
            ax.legend(fontsize=LEGEND_SIZE - 1, loc='upper left', ncol=2)
    else:
        ax.legend(fontsize=LEGEND_SIZE - 1, loc='upper left', ncol=2)

    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, "throughput_proc_hours_per_day.png"))


def plot_global_cumulative_completed_jobs(driver_tags, end_maps, driver_colors, plots_dir,
                                          maintenance=None, end_time_limit=None):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    plotted = False
    for tag in driver_tags:
        times, cumulative = build_cumulative_completion_series(
            end_maps.get(tag, {}),
            CST,
            end_time_limit=end_time_limit,
        )
        if not times:
            continue
        ax.step(times, cumulative, where='post',
                linewidth=LINE_WIDTH, color=driver_colors[tag],
                label=tag, zorder=3)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_ylim(bottom=0)
    apply_style(ax, xlabel="Date", ylabel="Cumulative Completed Jobs")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    if maintenance:
        has_spans = add_maintenance_shading(ax, maintenance, CST)
        if has_spans:
            existing_handles, existing_labels = ax.get_legend_handles_labels()
            ax.legend(handles=existing_handles + _MAINT_LEGEND,
                      labels=existing_labels + [h.get_label() for h in _MAINT_LEGEND],
                      fontsize=LEGEND_SIZE - 1, loc='upper left', ncol=2)
        else:
            ax.legend(fontsize=LEGEND_SIZE - 1, loc='upper left', ncol=2)
    else:
        ax.legend(fontsize=LEGEND_SIZE - 1, loc='upper left', ncol=2)

    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, "cumulative_completed_jobs.png"))


# ── Per-driver: Per-group wait CDF (one file per group) ──────────────────────

def plot_per_driver_group_wait_cdfs(tag, job_ids, wait_map, procs_map, plots_dir):
    group_waits = {g: [] for g in GROUP_LABELS}
    for jid in job_ids:
        if jid not in wait_map or jid not in procs_map:
            continue
        g = assign_group(procs_map[jid])
        if g:
            group_waits[g].append(wait_map[jid] / 3600.0)

    for g in GROUP_LABELS:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        pv = _draw_cdf_single(ax, group_waits[g], GROUP_COLORS[g])
        apply_style(ax, xlabel="Wait Time (hours)", ylabel="CDF")
        ax.set_ylim(0, 1.12)
        color = GROUP_COLORS[g]
        handles = [Line2D([0], [0], color=color, linewidth=LINE_WIDTH)]
        labels  = [GROUP_DISPLAY[g]]
        for p in PERCENTILES:
            if p in pv:
                handles.append(Line2D([0], [0], color=color, linestyle='--',
                                      linewidth=1.1, marker='o', markersize=5))
                labels.append(f'P{p} = {pv[p]:.2f} h')
        ax.legend(handles, labels, fontsize=LEGEND_SIZE, loc='lower right')
        fig.tight_layout()
        save(fig, os.path.join(plots_dir, tag, f"cdf_wait_{g}.png"))


# ── Per-driver: Per-group BSLD CDF (one file per group) ──────────────────────

def plot_per_driver_group_bsld_cdfs(tag, job_ids, wait_map, walltimes_map,
                                    procs_map, plots_dir):
    group_bsld = {g: [] for g in GROUP_LABELS}
    for jid in job_ids:
        if jid not in wait_map or jid not in procs_map:
            continue
        g = assign_group(procs_map[jid])
        if g is None:
            continue
        walltime = max(walltimes_map.get(jid, 1), 1)
        bsld = (wait_map[jid] + walltime) / walltime
        group_bsld[g].append(bsld)

    for g in GROUP_LABELS:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        pv = _draw_cdf_single(ax, group_bsld[g], GROUP_COLORS[g])
        apply_style(ax, xlabel="Bounded Slowdown (Walltime)", ylabel="CDF")
        ax.set_ylim(0, 1.12)
        color = GROUP_COLORS[g]
        handles = [Line2D([0], [0], color=color, linewidth=LINE_WIDTH)]
        labels  = [GROUP_DISPLAY[g]]
        for p in PERCENTILES:
            if p in pv:
                handles.append(Line2D([0], [0], color=color, linestyle='--',
                                      linewidth=1.1, marker='o', markersize=5))
                labels.append(f'P{p} = {pv[p]:.2f}')
        ax.legend(handles, labels, fontsize=LEGEND_SIZE, loc='lower right')
        fig.tight_layout()
        save(fig, os.path.join(plots_dir, tag, f"cdf_bsld_{g}.png"))


# ── Per-driver: Wait Time vs Size Scatter ─────────────────────────────────────

def plot_per_driver_wait_vs_size_scatter(tag, job_ids, wait_map, procs_map,
                                         plots_dir, driver_cfg=None,
                                         backfill_ids=None):
    """scatter_wait_vs_size.png — y: wait (h), x: node size (log), bg color: group."""
    if driver_cfg is None:
        driver_cfg = {}
    if backfill_ids is None:
        backfill_ids = set()

    # Split into normal vs backfill jobs
    sizes_run, waits_run = [], []
    sizes_bf,  waits_bf  = [], []
    waits_by_group = {g: [] for g in GROUP_LABELS}

    for jid in job_ids:
        if jid not in wait_map or jid not in procs_map:
            continue
        g = assign_group(procs_map[jid])
        if g is None:
            continue
        w_h = wait_map[jid] / 3600.0
        waits_by_group[g].append(w_h)
        if jid in backfill_ids:
            sizes_bf.append(procs_map[jid])
            waits_bf.append(w_h)
        else:
            sizes_run.append(procs_map[jid])
            waits_run.append(w_h)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    if sizes_run:
        ax.scatter(sizes_run, waits_run, s=12, alpha=0.25, color='black',
                   marker='o', zorder=10, edgecolors='none')
    if sizes_bf:
        ax.scatter(sizes_bf, waits_bf, s=12, alpha=0.25, color='black',
                   marker='o', zorder=10, edgecolors='none')

    # Reward=0.5 iso-line: QueueSnapshot reward = A_eff/(wait+A_eff), A_eff=A*(B/S)^n
    # iso-curve: wait_h = (A/3600) * (B/S_sys)^n
    reward_type = driver_cfg.get('reward_type', '')
    A = driver_cfg.get('size_adj_A') or driver_cfg.get('reward_param_a')
    n = driver_cfg.get('size_adj_n') or driver_cfg.get('reward_param_n', 1.0)
    if reward_type in ('queuesnapshot', 'harmonic_wait_sized') and A is not None:
        S_sys = 496
        B_values = np.linspace(10, 496, 500)
        w_half = (A / 3600.0) * (B_values / S_sys) ** n
        ax.plot(B_values, w_half, color='black', linestyle='-', linewidth=1.0, zorder=11)

    # Highlight Polaris Job Classes: S, M, L
    ax.axvspan(10,  24,  color=GROUP_COLORS['S'], alpha=0.15, zorder=1)
    ax.axvspan(25,  99,  color=GROUP_COLORS['M'], alpha=0.15, zorder=1)
    ax.axvspan(100, 496, color=GROUP_COLORS['L'], alpha=0.15, zorder=1)

    # Category Labels and Y-axis limit
    fixed_ymax = 48.0
    ax.set_ylim(bottom=0, top=fixed_ymax)
    label_y = fixed_ymax * 0.9

    _total_scatter = sum(len(v) for v in waits_by_group.values())
    for _g, _pos in [('S', np.sqrt(10*24)), ('M', np.sqrt(25*99)), ('L', np.sqrt(100*496))]:
        _pct = len(waits_by_group[_g]) / _total_scatter * 100 if _total_scatter else 0
        ax.text(_pos, label_y, f'{_g}\n({_pct:.1f}%)', ha='center',
                fontweight='bold', alpha=0.6, fontsize=12, linespacing=1.4)

    # ── Per-zone percentile lines (P50, P75, P90, P99) ───────────────────────
    for g, waits_g in waits_by_group.items():
        if not waits_g:
            continue
        arr = np.array(waits_g)
        x_lo, x_hi = _ZONE_X[g]
        dark_color = _ZONE_DARK[g]
        for p in [50, 75, 90, 99]:
            pv = float(np.percentile(arr, p))
            if pv > fixed_ymax:
                continue
            ax.hlines(pv, x_lo, x_hi, colors=dark_color, linestyles='--',
                      linewidth=1.3, zorder=8)

    apply_style(ax, xlabel="Node Size (log scale)", ylabel="Wait Time (hours)")
    ax.set_xscale('log')
    ax.set_xlim(8, 512)
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, "scatter_wait_vs_size.png"))

# ── Per-driver: BSLD vs Size Scatter ─────────────────────────────────────────

def plot_per_driver_bsld_vs_size_scatter(tag, job_ids, wait_map, procs_map,
                                          walltimes_map, plots_dir,
                                          driver_cfg=None, backfill_ids=None):
    """scatter_bsld_vs_size.png — y: BSLD (walltime-based), x: node size (log)."""
    if driver_cfg is None:
        driver_cfg = {}
    if backfill_ids is None:
        backfill_ids = set()

    sizes_run, bsld_run = [], []
    sizes_bf,  bsld_bf  = [], []
    bsld_by_group = {g: [] for g in GROUP_LABELS}
    all_wt_h = []

    for jid in job_ids:
        if jid not in wait_map or jid not in procs_map:
            continue
        g = assign_group(procs_map[jid])
        if g is None:
            continue
        walltime = max(walltimes_map.get(jid, 1), 1)
        bsld = (wait_map[jid] + walltime) / walltime
        bsld_by_group[g].append(bsld)
        all_wt_h.append(walltime / 3600.0)
        if jid in backfill_ids:
            sizes_bf.append(procs_map[jid])
            bsld_bf.append(bsld)
        else:
            sizes_run.append(procs_map[jid])
            bsld_run.append(bsld)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    if sizes_run:
        ax.scatter(sizes_run, bsld_run, s=12, alpha=0.25, color='black',
                   marker='o', zorder=10, edgecolors='none')
    if sizes_bf:
        ax.scatter(sizes_bf, bsld_bf, s=12, alpha=0.25, color='black',
                   marker='o', zorder=10, edgecolors='none')

    # Reward=0.5 iso-line for the driver's reward function (in BSLD space)
    reward_type = driver_cfg.get('reward_type', '')
    A = driver_cfg.get('reward_param_a')
    n = driver_cfg.get('reward_param_n', 1.0)
    if A is not None:
        S_sys = 496
        B_values = np.linspace(10, 496, 500)
        if reward_type == 'harmonic_bsld_sized':
            # reward=bsld/(bsld+A*(S/B)^(1/n)); reward=0.5 when bsld=A*(S/B)^(1/n)
            bsld_line = A * (S_sys / B_values) ** (1.0 / n)
            ax.plot(B_values, bsld_line, color='black', linestyle='-',
                    linewidth=1.0, zorder=11)
        elif reward_type == 'harmonic_wait_sized':
            # Convert wait threshold to BSLD using median walltime
            wt_median_h = float(np.median(all_wt_h)) if all_wt_h else 1.0
            w_line_h = A * (S_sys / B_values) ** (1.0 / n)
            bsld_line = 1.0 + w_line_h / wt_median_h
            ax.plot(B_values, bsld_line, color='black', linestyle='-',
                    linewidth=1.0, zorder=11)

    # Zone backgrounds
    ax.axvspan(10,  24,  color=GROUP_COLORS['S'], alpha=0.15, zorder=1)
    ax.axvspan(25,  99,  color=GROUP_COLORS['M'], alpha=0.15, zorder=1)
    ax.axvspan(100, 496, color=GROUP_COLORS['L'], alpha=0.15, zorder=1)

    fixed_ymax = 12.0
    ax.set_ylim(bottom=1.0, top=fixed_ymax)
    label_y = 1.0 + (fixed_ymax - 1.0) * 0.9

    _total = sum(len(v) for v in bsld_by_group.values())
    for _g, _pos in [('S', np.sqrt(10*24)), ('M', np.sqrt(25*99)),
                     ('L', np.sqrt(100*496))]:
        _pct = len(bsld_by_group[_g]) / _total * 100 if _total else 0
        ax.text(_pos, label_y, f'{_g}\n({_pct:.1f}%)', ha='center',
                fontweight='bold', alpha=0.6, fontsize=12, linespacing=1.4)

    for g, bsld_g in bsld_by_group.items():
        if not bsld_g:
            continue
        arr = np.array(bsld_g)
        x_lo, x_hi = _ZONE_X[g]
        for p in [50, 75, 90, 99]:
            pv = float(np.percentile(arr, p))
            if pv > fixed_ymax:
                continue
            ax.hlines(pv, x_lo, x_hi, colors=_ZONE_DARK[g], linestyles='--',
                      linewidth=1.3, zorder=8)

    apply_style(ax, xlabel="Node Size (log scale)", ylabel="Bounded Slowdown (Walltime)")
    ax.set_xscale('log')
    ax.set_xlim(8, 512)
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, "scatter_bsld_vs_size.png"))


# ── Per-driver: Job Size vs Wait Time — Kernel Density ───────────────────────

def plot_per_driver_size_wait_kde(tag, job_ids, wait_map, procs_map,
                                   plots_dir, driver_cfg=None,
                                   class_ymax=None):
    """size_wait_kde.png

    Single figure, three subplots (S / M / L).
    Each subplot: 2-D KDE of (job size [log], wait time) for that class.
    Black axes background — density blooms from black to bright colour.
    Reward lines r=1 and r=0.5 overlaid as white curves.
    Y-axis: 0 → per-class P99 across all drivers (passed via class_ymax).
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        print("  [skip] size_wait_kde: scipy not available")
        return

    if driver_cfg is None:
        driver_cfg = {}
    if class_ymax is None:
        class_ymax = {}

    reward_type = driver_cfg.get('reward_type', '')
    A = driver_cfg.get('reward_param_a')
    n = driver_cfg.get('reward_param_n', 1.0)

    S_sys = 496

    # Collect (log10(size), wait_h) per class
    data = {g: {'ls': [], 'w': []} for g in GROUP_LABELS}
    for jid in job_ids:
        if jid not in wait_map or jid not in procs_map:
            continue
        g = assign_group(procs_map[jid])
        if g is None:
            continue
        data[g]['ls'].append(np.log10(procs_map[jid]))
        data[g]['w'].append(wait_map[jid] / 3600.0)

    # Colormaps: low density = black, high density = vivid colour
    _KDE_CMAPS = {'S': 'plasma', 'M': 'inferno', 'L': 'magma'}
    _GC = '#333333'

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W * 1.6, FIG_H + 1))

    for ax, g in zip(axes, GROUP_LABELS):
        ax.set_facecolor('black')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
        ax.tick_params(colors='black', labelsize=TICK_SIZE)

        ls_arr = np.array(data[g]['ls'])
        ws_arr = np.array(data[g]['w'])
        x_lo, x_hi = _ZONE_X[g]
        ymax = class_ymax.get(g, float(np.percentile(ws_arr, 99))
                               if ws_arr.size > 0 else 48.0)

        if ls_arr.size >= 4:
            mask = ws_arr <= ymax
            if mask.sum() >= 4:
                kde = gaussian_kde(
                    np.vstack([ls_arr[mask], ws_arr[mask]]),
                    bw_method='scott')
                xi = np.linspace(np.log10(x_lo), np.log10(x_hi), 300)
                yi = np.linspace(0, ymax, 300)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = kde(np.vstack([Xi.ravel(),
                                    Yi.ravel()])).reshape(Xi.shape)
                ax.pcolormesh(10**Xi, Yi, Zi,
                              cmap=_KDE_CMAPS[g],
                              shading='gouraud', zorder=3)

        if reward_type == 'harmonic_wait_sized' and A is not None:
            B_vals = np.linspace(x_lo, x_hi, 400)
            w_half  = A * (S_sys / B_vals) ** (1.0 / n)
            visible = w_half <= ymax
            if visible.sum() >= 2:
                ax.plot(B_vals[visible], w_half[visible],
                        color='white', linestyle='-',
                        linewidth=1.4, alpha=0.85, zorder=10)

        ax.set_xscale('log')
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0, ymax)
        ax.set_title(GROUP_DISPLAY[g], fontsize=FONT_SIZE - 1)
        ax.set_xlabel("Node Size (log)", fontsize=FONT_SIZE)
        ax.set_ylabel("Wait Time (hours)", fontsize=FONT_SIZE)
        ax.grid(True, color=_GC, alpha=0.5, linewidth=0.5)

    if reward_type == 'harmonic_wait_sized' and A is not None:
        legend_handles = [
            Line2D([0], [0], color='white', linestyle='-', linewidth=1.4,
                   label='Reward=0.5 threshold'),
        ]
        axes[0].legend(handles=legend_handles, fontsize=LEGEND_SIZE - 2,
                       loc='upper right',
                       facecolor='#1a1a1a', edgecolor='#555555',
                       labelcolor='white')

    fig.suptitle(f'Job Size vs Wait Time (KDE) — {tag}', fontsize=FONT_SIZE)
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, "size_wait_kde.png"))


# ── Per-driver: Normal-only and Backfill-only scatter helpers ────────────────

def _draw_scatter_subset(ax, sizes, waits, waits_by_group, driver_cfg, tag,
                         fixed_ymax=48.0, total_trace=None):
    """Shared drawing logic for the normal-only / backfill-only scatter plots."""
    if sizes:
        ax.scatter(sizes, waits, s=12, alpha=0.25, color='black',
                   marker='o', zorder=10, edgecolors='none')

    # Reward=0.5 iso-line: QueueSnapshot reward = A_eff/(wait+A_eff), A_eff=A*(B/S)^n
    # iso-curve: wait_h = (A/3600) * (B/S_sys)^n
    reward_type = driver_cfg.get('reward_type', '')
    A = driver_cfg.get('size_adj_A') or driver_cfg.get('reward_param_a')
    n = driver_cfg.get('size_adj_n') or driver_cfg.get('reward_param_n', 1.0)
    if reward_type in ('queuesnapshot', 'harmonic_wait_sized') and A is not None:
        B_values = np.linspace(10, 496, 500)
        w_half = (A / 3600.0) * (B_values / 496) ** n
        ax.plot(B_values, w_half, color='black', linestyle='-',
                linewidth=1.0, zorder=11)

    # Zone backgrounds and labels
    ax.axvspan(10,  24,  color=GROUP_COLORS['S'], alpha=0.15, zorder=1)
    ax.axvspan(25,  99,  color=GROUP_COLORS['M'], alpha=0.15, zorder=1)
    ax.axvspan(100, 496, color=GROUP_COLORS['L'], alpha=0.15, zorder=1)
    ax.set_ylim(bottom=0, top=fixed_ymax)
    label_y = fixed_ymax * 0.9
    _denom = total_trace if total_trace else sum(len(v) for v in waits_by_group.values())
    for _g, _pos in [('S', np.sqrt(10*24)), ('M', np.sqrt(25*99)), ('L', np.sqrt(100*496))]:
        _pct = len(waits_by_group[_g]) / _denom * 100 if _denom else 0
        ax.text(_pos, label_y, f'{_g}\n({_pct:.1f}%)', ha='center',
                fontweight='bold', alpha=0.6, fontsize=12, linespacing=1.4)

    # Percentile lines P50, P75, P90 per zone (dark zone color, dashed)
    for g, waits_g in waits_by_group.items():
        if not waits_g:
            continue
        arr = np.array(waits_g)
        x_lo, x_hi = _ZONE_X[g]
        for p in [50, 75, 90]:
            pv = float(np.percentile(arr, p))
            if pv > fixed_ymax:
                continue
            ax.hlines(pv, x_lo, x_hi, colors=_ZONE_DARK[g],
                      linestyles='--', linewidth=1.3, zorder=8)

    apply_style(ax, xlabel="Node Size (log scale)", ylabel="Wait Time (hours)")
    ax.set_xscale('log')
    ax.set_xlim(8, 512)


def _draw_violin_row(axes_by_group, walltimes_by_group):
    """Draw one violin per zone into the provided axes dict {g: ax}."""
    for g in GROUP_LABELS:
        ax  = axes_by_group[g]
        wt  = walltimes_by_group[g]
        if wt:
            parts = ax.violinplot([wt], positions=[0], showmedians=True,
                                  showextrema=True, widths=0.7)
            for pc in parts.get('bodies', []):
                pc.set_facecolor(GROUP_COLORS[g])
                pc.set_alpha(0.55)
            for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                if key in parts:
                    parts[key].set_color(_ZONE_DARK[g])
                    parts[key].set_linewidth(1.2)
        ax.set_title(GROUP_DISPLAY[g], fontsize=FONT_SIZE - 2)
        ax.set_xticks([])
        ax.set_ylabel('Walltime (h)', fontsize=FONT_SIZE - 3)
        ax.tick_params(labelsize=TICK_SIZE - 2)
        ax.grid(True, alpha=0.3)


def plot_per_driver_scatter_normal(tag, job_ids, wait_map, procs_map,
                                   plots_dir, driver_cfg=None, backfill_ids=None,
                                   walltimes_map=None):
    """scatter_wait_vs_size_normal.png — normal jobs only + walltime violins."""
    if driver_cfg   is None: driver_cfg   = {}
    if backfill_ids is None: backfill_ids = set()
    if walltimes_map is None: walltimes_map = {}

    sizes, waits = [], []
    waits_by_group    = {g: [] for g in GROUP_LABELS}
    walltimes_by_group = {g: [] for g in GROUP_LABELS}
    total_trace = 0
    for jid in job_ids:
        if jid not in wait_map or jid not in procs_map:
            continue
        g = assign_group(procs_map[jid])
        if g is None:
            continue
        total_trace += 1
        if jid in backfill_ids:
            continue
        w_h  = wait_map[jid] / 3600.0
        wt_h = walltimes_map.get(jid, 0) / 3600.0
        sizes.append(procs_map[jid])
        waits.append(w_h)
        waits_by_group[g].append(w_h)
        walltimes_by_group[g].append(wt_h)

    fig = plt.figure(figsize=(FIG_W, FIG_H + 3))
    gs  = fig.add_gridspec(2, 3, height_ratios=[1, 2.5], hspace=0.45, wspace=0.35)
    ax_vio = {g: fig.add_subplot(gs[0, i]) for i, g in enumerate(GROUP_LABELS)}
    ax_sct = fig.add_subplot(gs[1, :])

    _draw_violin_row(ax_vio, walltimes_by_group)
    _draw_scatter_subset(ax_sct, sizes, waits, waits_by_group, driver_cfg, tag,
                         total_trace=total_trace)
    save(fig, os.path.join(plots_dir, tag, "scatter_wait_vs_size_normal.png"))


def plot_per_driver_scatter_backfill(tag, job_ids, wait_map, procs_map,
                                     plots_dir, driver_cfg=None, backfill_ids=None,
                                     walltimes_map=None):
    """scatter_wait_vs_size_backfill.png — backfill jobs only + walltime violins."""
    if driver_cfg   is None: driver_cfg   = {}
    if backfill_ids is None: backfill_ids = set()
    if walltimes_map is None: walltimes_map = {}

    sizes, waits = [], []
    waits_by_group    = {g: [] for g in GROUP_LABELS}
    walltimes_by_group = {g: [] for g in GROUP_LABELS}
    total_trace = 0
    for jid in job_ids:
        if jid not in wait_map or jid not in procs_map:
            continue
        g = assign_group(procs_map[jid])
        if g is None:
            continue
        total_trace += 1
        if jid not in backfill_ids:
            continue
        w_h  = wait_map[jid] / 3600.0
        wt_h = walltimes_map.get(jid, 0) / 3600.0
        sizes.append(procs_map[jid])
        waits.append(w_h)
        waits_by_group[g].append(w_h)
        walltimes_by_group[g].append(wt_h)

    fig = plt.figure(figsize=(FIG_W, FIG_H + 3))
    gs  = fig.add_gridspec(2, 3, height_ratios=[1, 2.5], hspace=0.45, wspace=0.35)
    ax_vio = {g: fig.add_subplot(gs[0, i]) for i, g in enumerate(GROUP_LABELS)}
    ax_sct = fig.add_subplot(gs[1, :])

    _draw_violin_row(ax_vio, walltimes_by_group)
    _draw_scatter_subset(ax_sct, sizes, waits, waits_by_group, driver_cfg, tag,
                         total_trace=total_trace)
    save(fig, os.path.join(plots_dir, tag, "scatter_wait_vs_size_backfill.png"))


# ── Per-driver: Wait vs Submit Time ──────────────────────────────────────────

def plot_per_driver_wait_vs_submit_time(tag, job_ids, wait_map, submit_map,
                                        procs_map, plots_dir):
    """scatter_wait_vs_submit_time.png — y: wait (h), x: submit datetime, color: S/M/L."""
    data = {g: {'x': [], 'y': []} for g in GROUP_LABELS}

    for jid in job_ids:
        if jid not in wait_map or jid not in submit_map or jid not in procs_map:
            continue
        g = assign_group(procs_map[jid])
        if g is None:
            continue
        data[g]['x'].append(datetime.fromtimestamp(submit_map[jid], tz=CST))
        data[g]['y'].append(wait_map[jid] / 3600.0)

    fig, ax = plt.subplots(figsize=(FIG_W * 1.4, FIG_H))

    for g in GROUP_LABELS:
        if data[g]['x']:
            ax.scatter(data[g]['x'], data[g]['y'], s=8, alpha=0.3,
                       color=GROUP_COLORS[g], edgecolors='none',
                       label=GROUP_DISPLAY[g], zorder=5)

    apply_style(ax, xlabel="Submit Time (DD/MM HH:MM)", ylabel="Wait (h)")
    ax.set_ylim(0, 120)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, "scatter_wait_vs_submit_time.png"))


# ── Global: scatter wait vs size — per-driver subplots, one file per group ───

_GROUP_XLIM  = {'S': (8, 28), 'M': (20, 115), 'L': (85, 560)}
_SCATTER_PCTILES = [25, 50, 75, 90]
_ZONE_X      = {'S': (10, 24), 'M': (25, 99), 'L': (100, 496)}
_ZONE_DARK   = {'S': '#1a527a', 'M': '#994d00', 'L': '#1a6640'}


def _detect_exp_axis(valid_tags, tag_to_driver_cfg):
    """Return 'a_value' if reward_param_a varies across drivers, else 'window_size'."""
    a_vals = {tag_to_driver_cfg.get(t, {}).get('reward_param_a') for t in valid_tags}
    a_vals.discard(None)
    return 'a_value' if len(a_vals) > 1 else 'window_size'


def _subplot_title(tag, cfg, exp_axis):
    if exp_axis == 'a_value':
        a = cfg.get('reward_param_a', '?')
        n = cfg.get('reward_param_n')
        return f'A={a}' + (f', n={n}' if n is not None else '')
    else:
        w = cfg.get('window_size')
        return f'w={w}' if w is not None else tag


# ── Per-driver: Wait KDE per group (with reward threshold bands) ──────────────

def plot_per_driver_wait_kde_by_tier(tag, job_ids, wait_map, procs_map,
                                      plots_dir, driver_cfg=None):
    """wait_kde_by_tier.png — one subplot per group, KDE of wait time 0-120 h.

    Vertical bands show the r=0.5 and r=0.99 reward thresholds across the
    group's node-size range (harmonic_wait_sized drivers only).
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        print("  [skip] wait_kde_by_tier: scipy not available")
        return

    if driver_cfg is None:
        driver_cfg = {}

    S_SYS = 496
    WAIT_XMAX = 120.0
    XS = np.linspace(0, WAIT_XMAX, 600)

    tier_waits = {g: [] for g in GROUP_LABELS}
    for jid in job_ids:
        if jid not in wait_map or jid not in procs_map:
            continue
        g = assign_group(procs_map[jid])
        if g:
            w_h = wait_map[jid] / 3600.0
            tier_waits[g].append(w_h)

    reward_type = driver_cfg.get('reward_type', '')
    A = driver_cfg.get('reward_param_a')
    n = driver_cfg.get('reward_param_n', 1.0)
    has_reward = (reward_type == 'harmonic_wait_sized' and A is not None)

    def _reward_thresh(r, B):
        return (r / (1.0 - r)) * A * (S_SYS / B) ** (1.0 / n)

    N     = len(GROUP_LABELS)
    ncols = min(N, 3)
    nrows = int(np.ceil(N / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.6),
                             sharey=False, squeeze=False)

    for idx, g in enumerate(GROUP_LABELS):
        row, col = divmod(idx, ncols)
        ax  = axes[row][col]
        clr = GROUP_COLORS[g]

        arr = np.array(tier_waits[g])
        arr = arr[arr <= WAIT_XMAX]

        if arr.size >= 4:
            kde = gaussian_kde(arr, bw_method='scott')
            ys  = kde(XS)
            ymax = float(ys.max()) * 1.35
            ax.fill_between(XS, ys, alpha=0.25, color=clr)
            ax.plot(XS, ys, color=clr, linewidth=LINE_WIDTH)

            for p, ls in [(50, '--'), (90, ':'), (99, '-.')]:
                pv = float(np.percentile(arr, p))
                if pv <= WAIT_XMAX:
                    ax.axvline(pv, color=clr, linestyle=ls, linewidth=1.1, alpha=0.8)
                    ax.text(pv, ymax * 0.96, f'P{p}', color=clr,
                            fontsize=7, ha='center', va='top', clip_on=True)
        else:
            ymax = 1.0

        if has_reward:
            x_lo, x_hi = _ZONE_X[g]
            for r_val, r_color, r_label in [
                (0.50, '#e06000', 'r=0.50'),
                (0.99, '#c00000', 'r=0.99'),
            ]:
                t_lo = _reward_thresh(r_val, x_hi)
                t_hi = _reward_thresh(r_val, x_lo)
                t_lo = min(t_lo, WAIT_XMAX)
                t_hi = min(t_hi, WAIT_XMAX)
                if t_lo < WAIT_XMAX:
                    ax.axvspan(t_lo, t_hi, color=r_color, alpha=0.12)
                    ax.axvline(t_lo, color=r_color, linestyle='-', linewidth=1.3)
                    ax.axvline(t_hi, color=r_color, linestyle='--', linewidth=1.0)
                    ax.text((t_lo + min(t_hi, WAIT_XMAX)) / 2, ymax * 0.85,
                            r_label, color=r_color, fontsize=7,
                            ha='center', va='top', clip_on=True)

        ax.set_title(f'{GROUP_DISPLAY[g]}  (n={len(tier_waits[g])})',
                     fontsize=FONT_SIZE - 2)
        ax.set_xlim(0, WAIT_XMAX)
        ax.set_ylim(0, ymax)
        ax.tick_params(labelsize=TICK_SIZE - 2)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Density", fontsize=FONT_SIZE - 2)
        if row == nrows - 1:
            ax.set_xlabel("Wait Time (hours)", fontsize=FONT_SIZE - 2)

    for idx in range(N, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f'Wait KDE by Group — {tag}', fontsize=FONT_SIZE)
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, "wait_kde_by_tier.png"))


def plot_global_wait_kde_by_group(driver_tags, all_wait_by_group, driver_colors,
                                   plots_dir):
    """wait_kde_{S,M,L}.png — 1-D KDE subplots per driver, shared x-axis 0-120 h."""
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        print("  [skip] wait_kde: scipy not available")
        return

    WAIT_XMAX = 120.0
    XS = np.linspace(0, WAIT_XMAX, 600)

    for g in GROUP_LABELS:
        N     = len(driver_tags)
        ncols = min(N, 4)
        nrows = int(np.ceil(N / ncols))
        fw    = ncols * 4.0
        fh    = nrows * 3.5

        fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh),
                                 sharey=True, squeeze=False)

        density_max = 0.0
        for tag in driver_tags:
            values = all_wait_by_group[tag].get(g, [])
            if len(values) < 4:
                continue
            arr = np.array(values)
            arr = arr[arr <= WAIT_XMAX]
            if len(arr) < 4:
                continue
            kde = gaussian_kde(arr, bw_method='scott')
            density_max = max(density_max, float(kde(XS).max()))

        for idx, tag in enumerate(driver_tags):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]

            values = all_wait_by_group[tag].get(g, [])
            if len(values) >= 4:
                arr = np.array(values)
                arr = arr[arr <= WAIT_XMAX]
            else:
                arr = np.array([])

            if arr.size >= 4:
                kde = gaussian_kde(arr, bw_method='scott')
                ys  = kde(XS)
                ax.fill_between(XS, ys, alpha=0.25, color=driver_colors[tag])
                ax.plot(XS, ys, color=driver_colors[tag], linewidth=LINE_WIDTH)
                for p in [50, 90, 99]:
                    pv = float(np.percentile(arr, p))
                    if pv <= WAIT_XMAX:
                        ax.axvline(pv, color=driver_colors[tag],
                                   linestyle='--', linewidth=1.0, alpha=0.7)
                        ax.text(pv, density_max * 0.97, f'P{p}',
                                color=driver_colors[tag], fontsize=7,
                                ha='center', va='top', clip_on=True)

            ax.set_title(tag, fontsize=FONT_SIZE - 3)
            ax.set_xlim(0, WAIT_XMAX)
            if density_max > 0:
                ax.set_ylim(0, density_max * 1.1)
            ax.tick_params(labelsize=TICK_SIZE - 2)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("Density", fontsize=FONT_SIZE - 2)
            if row == nrows - 1:
                ax.set_xlabel("Wait Time (hours)", fontsize=FONT_SIZE - 2)

        for idx in range(N, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(f'Wait Time KDE — {GROUP_DISPLAY[g]}', fontsize=FONT_SIZE)
        fig.tight_layout()
        save(fig, os.path.join(plots_dir, f'wait_kde_{g}.png'))


# ── Global: overall wait CDF across all drivers ──────────────────────────────

def plot_global_overall_wait_cdfs(driver_tags, all_wait, driver_colors, plots_dir):
    """cdf_wait_cdf.png — one line per driver."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for tag in driver_tags:
        values = all_wait.get(tag, [])
        _draw_cdf_multi(ax, values, driver_colors[tag], tag)

    apply_style(ax, xlabel="Wait Time (hours)", ylabel="CDF")
    ax.set_ylim(0, 1.05)
    ax.set_xscale('log')

    handles, lbls = ax.get_legend_handles_labels()
    handles += _pctile_legend_handles()
    lbls    += [f'P{p}' for p in PERCENTILES] + ['Mean']
    ax.legend(handles, lbls, fontsize=LEGEND_SIZE, loc='lower right')

    fig.tight_layout()
    save(fig, os.path.join(plots_dir, "cdf_wait_cdf.png"))


# ── Global: overall BSLD CDF across all drivers ──────────────────────────────

def plot_global_overall_bsld_cdfs(driver_tags, all_bsld, driver_colors, plots_dir):
    """cdf_bsld.png — one line per driver."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for tag in driver_tags:
        values = all_bsld.get(tag, [])
        _draw_cdf_multi(ax, values, driver_colors[tag], tag)

    apply_style(ax, xlabel="Bounded Slowdown (Walltime)", ylabel="CDF")
    ax.set_ylim(0, 1.05)

    handles, lbls = ax.get_legend_handles_labels()
    handles += _pctile_legend_handles()
    lbls    += [f'P{p}' for p in PERCENTILES] + ['Mean']
    ax.legend(handles, lbls, fontsize=LEGEND_SIZE, loc='lower right')

    fig.tight_layout()
    save(fig, os.path.join(plots_dir, "cdf_bsld.png"))


# ── Global: per-group wait CDF across all drivers ────────────────────────────

def plot_global_group_wait_cdfs(driver_tags, all_wait_by_group, driver_colors,
                                plots_dir):
    """cdf_wait_{S,M,L}.png — one line per driver, one file per group."""
    for g in GROUP_LABELS:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        for i, tag in enumerate(driver_tags):
            values = all_wait_by_group[tag].get(g, [])
            _draw_cdf_multi(ax, values, driver_colors[tag], tag)

        apply_style(ax, xlabel="Wait Time (hours)", ylabel="CDF")
        ax.set_ylim(0, 1.05)
        ax.set_xscale('log')

        handles, lbls = ax.get_legend_handles_labels()
        handles += _pctile_legend_handles()
        lbls    += [f'P{p}' for p in PERCENTILES] + ['Mean']
        ax.legend(handles, lbls, fontsize=LEGEND_SIZE, loc='lower right')

        fig.tight_layout()
        save(fig, os.path.join(plots_dir, f"cdf_wait_{g}.png"))


# ── Global: per-group BSLD CDF across all drivers ────────────────────────────

def plot_global_group_bsld_cdfs(driver_tags, all_bsld_by_group, driver_colors,
                                plots_dir):
    """cdf_bsld_{S,M,L}.png — one line per driver, one file per group."""
    for g in GROUP_LABELS:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        for tag in driver_tags:
            values = all_bsld_by_group[tag].get(g, [])
            _draw_cdf_multi(ax, values, driver_colors[tag], tag)

        apply_style(ax, xlabel="Bounded Slowdown (Walltime)", ylabel="CDF")
        ax.set_ylim(0, 1.05)

        handles, lbls = ax.get_legend_handles_labels()
        handles += _pctile_legend_handles()
        lbls    += [f'P{p}' for p in PERCENTILES] + ['Mean']
        ax.legend(handles, lbls, fontsize=LEGEND_SIZE, loc='lower right')

        fig.tight_layout()
        save(fig, os.path.join(plots_dir, f"cdf_bsld_{g}.png"))


# ── Table helpers ─────────────────────────────────────────────────────────────

_STAT_HEADERS = ["Min", "P25", "P50", "P75", "P85", "P95", "P99", "Max",
                 "Mean", "StdDev", "GeometricMean"]


def _compute_stats_row(arr):
    pos   = arr[arr > 0]
    gmean = float(np.exp(np.mean(np.log(pos)))) if pos.size > 0 else 0.0
    return [
        f"{arr.min():.4f}",
        f"{np.percentile(arr, 25):.4f}",
        f"{np.percentile(arr, 50):.4f}",
        f"{np.percentile(arr, 75):.4f}",
        f"{np.percentile(arr, 85):.4f}",
        f"{np.percentile(arr, 95):.4f}",
        f"{np.percentile(arr, 99):.4f}",
        f"{arr.max():.4f}",
        f"{arr.mean():.4f}",
        f"{arr.std():.4f}",
        f"{gmean:.4f}",
    ]


def _write_stat_table(path, driver_tags, data_map, multiplier=1.0):
    os.makedirs(plots_dir_from_path(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Driver"] + _STAT_HEADERS)
        for tag in driver_tags:
            values = data_map.get(tag, [])
            if not values:
                writer.writerow([tag] + ["N/A"] * len(_STAT_HEADERS))
                continue
            arr = np.array(values) * multiplier
            writer.writerow([tag] + _compute_stats_row(arr))
    print(f"  Saved: {path}")


def plots_dir_from_path(path):
    d = os.path.dirname(path)
    return d if d else "."


def write_global_overall_wait_table(driver_tags, all_wait, plots_dir):
    _write_stat_table(os.path.join(plots_dir, "table_wait_overall.csv"),
                      driver_tags, all_wait, multiplier=60.0)


def write_global_overall_bsld_table(driver_tags, all_bsld, plots_dir):
    _write_stat_table(os.path.join(plots_dir, "table_bsld_overall.csv"),
                      driver_tags, all_bsld, multiplier=1.0)


def write_global_wait_tables(driver_tags, all_wait_by_group, plots_dir):
    for g in GROUP_LABELS:
        _write_stat_table(
            os.path.join(plots_dir, f"table_{g}_wait.csv"),
            driver_tags,
            {tag: all_wait_by_group[tag].get(g, []) for tag in driver_tags},
            multiplier=60.0)


def write_global_bsld_tables(driver_tags, all_bsld_by_group, plots_dir):
    for g in GROUP_LABELS:
        _write_stat_table(
            os.path.join(plots_dir, f"table_{g}_bsld.csv"),
            driver_tags,
            {tag: all_bsld_by_group[tag].get(g, []) for tag in driver_tags},
            multiplier=1.0)


def write_global_size_job_tables(label, driver_tags, all_wait_sz, all_bsld_sz,
                                  job_info, plots_dir):
    extra_headers = ["Count", "Mean_Walltime_h", "Mean_Runtime_h"]

    def _write(filename, data_map, multiplier=1.0):
        path = os.path.join(plots_dir, filename)
        os.makedirs(plots_dir, exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Driver"] + _STAT_HEADERS + extra_headers)
            for tag in driver_tags:
                values = data_map.get(tag, [])
                count, mean_wt, mean_rt = job_info.get(tag, (0, 0.0, 0.0))
                extra = [str(count), f"{mean_wt:.4f}", f"{mean_rt:.4f}"]
                if not values:
                    writer.writerow([tag] + ["N/A"] * len(_STAT_HEADERS) + extra)
                    continue
                arr = np.array(values) * multiplier
                writer.writerow([tag] + _compute_stats_row(arr) + extra)
        print(f"  Saved: {path}")

    _write(f"table_{label}_wait.csv", all_wait_sz, multiplier=60.0)
    _write(f"table_{label}_bsld.csv", all_bsld_sz, multiplier=1.0)


def write_global_util_table(driver_tags, end_maps, start_maps, submit_maps,
                            procs_map, plots_dir, window_start=None, window_end=None):
    path = os.path.join(plots_dir, "table_util_overall.csv")
    os.makedirs(plots_dir, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Driver", "Utilization_Percent",
                         "Core_Hours_Used", "Core_Hours_Unused"])
        for tag in driver_tags:
            end    = end_maps.get(tag, {})
            start  = start_maps.get(tag, {})
            submit = submit_maps.get(tag, {})
            if not end or not submit:
                writer.writerow([tag, "N/A", "N/A", "N/A"])
                continue
            if window_start is not None and window_end is not None:
                makespan = window_end - window_start
                used_core_s = sum_used_core_seconds(start, end, procs_map, window_start, window_end)
            else:
                used_core_s = sum(
                    (end[jid] - start[jid]) * procs_map.get(jid, 0)
                    for jid in end if jid in start
                )
                makespan = max(end.values()) - min(submit.values())
            total_core_s  = makespan * S_SYS
            util_pct      = 100.0 * used_core_s / total_core_s if total_core_s > 0 else 0.0
            used_core_h   = used_core_s / 3600.0
            unused_core_h = max(0.0, total_core_s - used_core_s) / 3600.0
            writer.writerow([tag,
                             f"{util_pct:.2f}",
                             f"{used_core_h:.2f}",
                             f"{unused_core_h:.2f}"])
    print(f"  Saved: {path}")


def _parse_maint_raw(events_csv_path):
    sma, sms, sme, ums, ume = {}, {}, {}, {}, {}
    try:
        with open(events_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                event = row.get("event", "")
                if event not in ("SMA", "SMS", "SME", "UMS", "UME"):
                    continue
                try:
                    idx = int(row["id"])
                    t   = float(row["sim_time"])
                except (KeyError, ValueError):
                    continue
                if event == "SMA":   sma[idx] = t
                elif event == "SMS": sms[idx] = t
                elif event == "SME": sme[idx] = t
                elif event == "UMS": ums[idx] = t
                elif event == "UME": ume[idx] = t
    except FileNotFoundError:
        pass
    return sma, sms, sme, ums, ume


def _compute_window_util(start_map, end_map, procs_map, t_start, t_end, sys_size):
    duration = t_end - t_start
    if duration <= 0:
        return 0.0
    used = sum_used_core_seconds(start_map, end_map, procs_map, t_start, t_end)
    return used / (duration * sys_size)


def write_global_util_maintenance_table(driver_tags, end_maps, start_maps,
                                        output_dir, procs_map, plots_dir, tz,
                                        window_start=None, window_end=None):
    date_fmt   = "%Y-%m-%d %H:%M:%S"
    two_days_s = 2 * 24 * 3600
    os.makedirs(plots_dir, exist_ok=True)

    event_order = []  # insertion-ordered list of (mstart_dt, mend_dt, total_h_str, type)
    event_data  = {}  # key -> {tag+"_ann"|"_dur"|"_after": util_str}
    tags_seen   = []

    for tag in driver_tags:
        events_path = os.path.join(output_dir, tag, "events.csv")
        if not os.path.exists(events_path):
            continue
        tags_seen.append(tag)
        sma, sms, sme, ums, ume = _parse_maint_raw(events_path)
        s_map = start_maps.get(tag, {})
        e_map = end_maps.get(tag, {})

        for idx in sorted(sms):
            if idx not in sme:
                continue
            mstart    = sms[idx]
            mend      = sme[idx]
            if window_start is not None and mend <= window_start:
                continue
            if window_end is not None and mstart >= window_end:
                continue
            mstart_dt = datetime.fromtimestamp(mstart, tz=tz).strftime(date_fmt)
            mend_dt   = datetime.fromtimestamp(mend,   tz=tz).strftime(date_fmt)
            total_h   = f"{(mend - mstart) / 3600.0:.4f}"
            ann_str   = "N/A"
            if idx in sma:
                ann_start = sma[idx]
                ann_end = mstart
                if window_start is not None:
                    ann_start = max(ann_start, window_start)
                if window_end is not None:
                    ann_end = min(ann_end, window_end)
                u = _compute_window_util(s_map, e_map, procs_map, ann_start, ann_end, S_SYS)
                ann_str = f"{u * 100:.2f}%"
            dur_start = max(mstart, window_start) if window_start is not None else mstart
            dur_end = min(mend, window_end) if window_end is not None else mend
            after_start = max(mend, window_start) if window_start is not None else mend
            after_end = mend + two_days_s
            if window_end is not None:
                after_end = min(after_end, window_end)
            u_dur   = _compute_window_util(s_map, e_map, procs_map, dur_start, dur_end, S_SYS)
            u_after = _compute_window_util(s_map, e_map, procs_map, after_start, after_end, S_SYS)
            key = (mstart_dt, mend_dt, total_h, "Scheduled")
            if key not in event_data:
                event_order.append(key)
                event_data[key] = {}
            event_data[key][tag + "_ann"]   = ann_str
            event_data[key][tag + "_dur"]   = f"{u_dur * 100:.2f}%"
            event_data[key][tag + "_after"] = f"{u_after * 100:.2f}%"

        for idx in sorted(ums):
            if idx not in ume:
                continue
            mstart    = ums[idx]
            mend      = ume[idx]
            if window_start is not None and mend <= window_start:
                continue
            if window_end is not None and mstart >= window_end:
                continue
            mstart_dt = datetime.fromtimestamp(mstart, tz=tz).strftime(date_fmt)
            mend_dt   = datetime.fromtimestamp(mend,   tz=tz).strftime(date_fmt)
            total_h   = f"{(mend - mstart) / 3600.0:.4f}"
            dur_start = max(mstart, window_start) if window_start is not None else mstart
            dur_end = min(mend, window_end) if window_end is not None else mend
            after_start = max(mend, window_start) if window_start is not None else mend
            after_end = mend + two_days_s
            if window_end is not None:
                after_end = min(after_end, window_end)
            u_dur   = _compute_window_util(s_map, e_map, procs_map, dur_start, dur_end, S_SYS)
            u_after = _compute_window_util(s_map, e_map, procs_map, after_start, after_end, S_SYS)
            key = (mstart_dt, mend_dt, total_h, "Unscheduled")
            if key not in event_data:
                event_order.append(key)
                event_data[key] = {}
            event_data[key][tag + "_ann"]   = "N/A"
            event_data[key][tag + "_dur"]   = f"{u_dur * 100:.2f}%"
            event_data[key][tag + "_after"] = f"{u_after * 100:.2f}%"

    meta_cols = ["Maintenance_Start", "Maintenance_End", "Total_Maintenance_Hours", "Type"]

    def _write_table(filename, suffix):
        path = os.path.join(plots_dir, filename)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(meta_cols + tags_seen)
            for key in event_order:
                mstart_dt, mend_dt, total_h, mtype = key
                row = [mstart_dt, mend_dt, total_h, mtype]
                for tag in tags_seen:
                    row.append(event_data[key].get(tag + suffix, "N/A"))
                writer.writerow(row)
        print(f"  Saved: {path}")

    _write_table("table_util_announcement_to_start.csv", "_ann")
    _write_table("table_util_during_maintenance.csv",    "_dur")
    _write_table("table_util_2days_after_maintenance.csv", "_after")


# ── Tail-wait heatmap ────────────────────────────────────────────────────────

_TAIL_NODE_EDGES  = [10, 25, 100, 128, 256, 497]
_TAIL_NODE_LABELS = ['S\n10–24', 'M\n25–99', 'L1\n100–127',
                     'L2\n128–255', 'L3\n256–496']
_TAIL_WT_EDGES    = [0, 6, 12, 24, 48.01]   # hours
_TAIL_WT_LABELS   = ['0–6h', '6–12h', '12–24h', '24–48h']


def _tail_bin(value, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return None


def _build_tail_grids(job_ids, wait_map, procs_map, walltimes_map):
    n_node = len(_TAIL_NODE_EDGES) - 1
    n_wt   = len(_TAIL_WT_EDGES)   - 1
    count    = np.zeros((n_node, n_wt))
    sum_wait = np.zeros((n_node, n_wt))
    for jid in job_ids:
        if jid not in procs_map or jid not in walltimes_map:
            continue
        ri = _tail_bin(procs_map[jid], _TAIL_NODE_EDGES)
        ci = _tail_bin(walltimes_map[jid] / 3600.0, _TAIL_WT_EDGES)
        if ri is None or ci is None:
            continue
        count[ri, ci]    += 1
        sum_wait[ri, ci] += wait_map[jid] / 3600.0
    return count, sum_wait


def _draw_tail_heatmap(ax, count_grid, sum_wait_grid, title, threshold_h, shared_vmax):
    n_rows, n_cols = count_grid.shape
    img = np.log1p(count_grid)
    im  = ax.imshow(img, aspect='auto', cmap='YlOrRd',
                    vmin=0, vmax=shared_vmax, interpolation='nearest')
    for r in range(n_rows):
        for c in range(n_cols):
            cnt = int(count_grid[r, c])
            if cnt == 0:
                ax.text(c, r, '0', ha='center', va='center',
                        fontsize=8, color='#bbbbbb')
                continue
            avg_w = sum_wait_grid[r, c] / cnt
            tc    = 'white' if img[r, c] > shared_vmax * 0.55 else 'black'
            ax.text(c, r, f'{cnt}\n{avg_w:.1f}h',
                    ha='center', va='center', fontsize=10,
                    color=tc, fontweight='bold', linespacing=1.4)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(_TAIL_WT_LABELS, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(_TAIL_NODE_LABELS, fontsize=9)
    ax.set_title(f'{title}\n(tail > {threshold_h:.1f} h)', fontsize=10, pad=5)
    ax.set_xlabel("Requested Walltime", fontsize=9)
    ax.set_ylabel("Node Count", fontsize=9)
    return im


def write_global_tail_heatmap(driver_tags, all_job_ids_map, all_wait_maps,
                               procs_map, walltimes_map, plots_dir, pctile=80):
    """Save one tail-heatmap PNG per driver into its own output_dir/<tag>/ folder."""
    if not driver_tags:
        return

    cell_w = 2.8
    cell_h = 1.8

    # Build grids for all drivers first so we can share vmax across them.
    all_grids = []
    for tag in driver_tags:
        job_ids  = all_job_ids_map.get(tag, [])
        wait_map = all_wait_maps.get(tag, {})
        n_node = len(_TAIL_NODE_EDGES) - 1
        n_wt   = len(_TAIL_WT_EDGES)   - 1
        if not job_ids:
            all_grids.append((tag, np.zeros((n_node, n_wt)),
                              np.zeros((n_node, n_wt)), 0.0, 0, 0))
            continue
        waits_h     = np.array([wait_map[j] for j in job_ids]) / 3600.0
        threshold_h = float(np.percentile(waits_h, pctile))
        tail_ids    = [j for j in job_ids if wait_map[j] / 3600.0 > threshold_h]
        tail_wmap   = {j: wait_map[j] for j in tail_ids}
        count_g, sum_w_g = _build_tail_grids(tail_ids, tail_wmap,
                                             procs_map, walltimes_map)
        all_grids.append((tag, count_g, sum_w_g, threshold_h,
                          len(tail_ids), len(job_ids)))

    shared_vmax = max((float(np.log1p(g[1].max())) for g in all_grids), default=1.0)
    shared_vmax = max(shared_vmax, 1.0)

    figw = cell_w * len(_TAIL_WT_LABELS) + 1.5
    figh = cell_h * len(_TAIL_NODE_LABELS) + 1.2

    for tag, count_g, sum_w_g, thr_h, n_tail, n_total in all_grids:
        fig, ax = plt.subplots(figsize=(figw, figh))
        pct = 100.0 * n_tail / n_total if n_total else 0
        im = _draw_tail_heatmap(
            ax, count_g, sum_w_g,
            f'{tag}  ({n_tail} / {n_total},  {pct:.0f}%)',
            thr_h, shared_vmax)
        cbar = fig.colorbar(im, ax=ax, orientation='vertical',
                            fraction=0.046, pad=0.04)
        cbar.set_label('log(count + 1)', fontsize=8)
        tv = np.linspace(0, shared_vmax, 5)
        cbar.set_ticks(tv)
        cbar.set_ticklabels([f'{int(np.expm1(v))}' for v in tv], fontsize=7)
        fig.suptitle(
            f'Tail-wait characteristics — above P{int(pctile)}  (common window)\n'
            'Cell: count (top)  /  mean wait in hours (bottom)',
            fontsize=10, y=1.01)
        fig.tight_layout(rect=[0, 0, 1.0, 1.0])
        driver_dir = os.path.join(plots_dir, tag)
        os.makedirs(driver_dir, exist_ok=True)
        path = os.path.join(driver_dir, f'tail_heatmap_p{int(pctile)}.png')
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 1:
        print(f"Usage: {sys.argv[0]} <experiment.json> [--plots-dir <dir>]")
        return 1

    config_path         = argv[0]
    plots_dir_override  = None
    output_dir_override = None
    global_only         = False
    i = 1
    while i < len(argv):
        if argv[i] == "--plots-dir" and i + 1 < len(argv):
            plots_dir_override = argv[i + 1]
            i += 2
        elif argv[i] == "--output-dir" and i + 1 < len(argv):
            output_dir_override = argv[i + 1]
            i += 2
        elif argv[i] == "--global-only":
            global_only = True
            i += 1
        else:
            i += 1

    with open(config_path) as f:
        config = json.load(f)

    swf_path   = config["swf_path"]
    output_dir = output_dir_override or config["output_dir"]
    plots_dir  = plots_dir_override or (output_dir.rstrip("/") + "_plots")

    driver_tags = [
        d.get("tag", d["type"])
        for d in config.get("drivers", [])
        if d.get("plot", True)
    ]
    tag_to_driver_cfg = {
        d.get("tag", d["type"]): d
        for d in config.get("drivers", [])
        if d.get("plot", True)
    }
    if not driver_tags:
        print("No drivers with plot=true found.")
        return 1

    os.makedirs(plots_dir, exist_ok=True)

    print(f"Config:    {config_path}")
    print(f"Results:   {output_dir}")
    print(f"Plots dir: {plots_dir}")
    print(f"Drivers:   {driver_tags}")
    print()

    analysis_start, analysis_end = analysis_window_bounds()
    analysis_start_dt = datetime.fromtimestamp(analysis_start, tz=CST)
    analysis_end_dt = datetime.fromtimestamp(analysis_end, tz=CST)
    print(
        "Analysis window: "
        f"{analysis_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} to "
        f"{analysis_end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} "
        "(jobs filtered by submit time)"
    )
    print()

    # Assign a stable color per driver for global comparison plots
    driver_colors = {tag: driver_color(i) for i, tag in enumerate(driver_tags)}

    # Load maintenance windows from the first available driver's events.csv
    maintenance = []
    for _tag in driver_tags:
        _events_path = os.path.join(output_dir, _tag, "events.csv")
        if os.path.exists(_events_path):
            maintenance = parse_maintenance_from_events(_events_path)
            break
    maintenance = filter_maintenance_to_window(maintenance, analysis_start, analysis_end)
    if maintenance:
        sched_n   = sum(1 for _, _, t in maintenance if t == 'S')
        unsched_n = sum(1 for _, _, t in maintenance if t == 'U')
        print(f"Maintenance: {len(maintenance)} windows  "
              f"({sched_n} scheduled, {unsched_n} unscheduled)")
        print()

    # Parse SWF once
    procs_map, walltimes_map, _, runtimes_map = parse_swf(swf_path)

    # ── Pass 1: load events, skip missing drivers ─────────────────────────────
    valid_tags   = []
    raw_submit_maps = {}
    raw_start_maps  = {}
    raw_end_maps    = {}
    submit_maps  = {}
    start_maps   = {}
    end_maps     = {}

    for tag in driver_tags:
        cfg = tag_to_driver_cfg.get(tag, {})
        if cfg.get("type") == "rlscheduler" or "csv_path" in cfg:
            csv_path = cfg.get("csv_path", "")
            if not os.path.exists(csv_path):
                print(f"[skip] {tag}: csv_path not found: {csv_path}")
                continue
            submit, start, end = parse_rlscheduler_csv(csv_path)
        else:
            events_path = os.path.join(output_dir, tag, "events.csv")
            if not os.path.exists(events_path):
                print(f"[skip] {tag}: events.csv not found")
                continue
            submit, start, end = parse_events(events_path)
        if not end:
            print(f"[skip] {tag}: no completed jobs")
            continue
        raw_submit_maps[tag] = submit
        raw_start_maps[tag]  = start
        raw_end_maps[tag]    = end

        submit_f, start_f, end_f = filter_jobs_by_submit_window(
            submit, start, end, analysis_start, analysis_end
        )
        if not end_f:
            print(f"[warn] {tag}: no jobs in {ANALYSIS_YEAR} window, using all available data")
            submit_f, start_f, end_f = submit, start, end
        submit_maps[tag] = submit_f
        start_maps[tag]  = start_f
        end_maps[tag]    = end_f
        valid_tags.append(tag)

    if not valid_tags:
        print(f"No drivers have data to plot within calendar year {ANALYSIS_YEAR}.")
        return 1

    # ── Detect partial results and determine common windows ───────────────────
    # common_submit_end: min of each driver's latest completed-job submit time.
    #   Used for wait/bsld statistics so every driver is compared on the same
    #   set of submitted jobs.
    # t_window_end: min of each driver's latest job end time.
    #   Kept for time-series plots (throughput, utilisation) indexed by end time.
    driver_t_end      = {tag: max(end_maps[tag].values())    for tag in valid_tags}
    driver_max_submit = {tag: max(submit_maps[tag].values()) for tag in valid_tags}
    job_counts        = {tag: len(end_maps[tag])             for tag in valid_tags}

    counts_equal      = len(set(job_counts.values())) == 1
    partial           = not counts_equal
    t_window_end      = min(driver_t_end.values())
    common_submit_end = min(driver_max_submit.values())

    if partial:
        print("WARNING: partial results — drivers have different job counts; "
              "wait/bsld stats restricted to common submit-time window.")
        for tag in valid_tags:
            print(f"  {tag}: {job_counts[tag]} jobs, "
                  f"max submit = {driver_max_submit[tag]:.0f}")
        print(f"  Common submit window end: t = {common_submit_end:.0f}\n")
    else:
        print(f"All {len(valid_tags)} drivers have {list(job_counts.values())[0]} "
              f"completed jobs — full results.\n")

    # ── Pre-pass: per-class P99 wait across all drivers (for KDE y-axes) ────────
    _cls_waits = {g: [] for g in GROUP_LABELS}
    for _tag in valid_tags:
        for _jid, _submit_t in submit_maps[_tag].items():
            if _submit_t > common_submit_end:
                continue
            if (_jid in start_maps[_tag] and _jid in end_maps[_tag]
                    and _jid in procs_map):
                _g = assign_group(procs_map[_jid])
                if _g:
                    _cls_waits[_g].append(
                        (start_maps[_tag][_jid] - submit_maps[_tag][_jid]) / 3600.0)
    class_wait_p99 = {
        g: float(np.percentile(_cls_waits[g], 99)) if _cls_waits[g] else 48.0
        for g in GROUP_LABELS
    }
    print("Per-class P99 wait (KDE y-cap): "
          + ", ".join(f"{g}={class_wait_p99[g]:.2f}h" for g in GROUP_LABELS) + "\n")

    # ── Pass 2: build per-driver metrics within the analysis window ───────────
    # Wait/slowdown statistics use jobs submitted in the target calendar year,
    # then apply the common completed-job window when runs are partial.
    # Time-series plots are clipped to the calendar year for readability.
    per_driver_end_time_limit = analysis_end

    all_wait_by_group = {}   # tag -> {g -> [wait_hours, ...]}
    all_bsld_by_group = {}   # tag -> {g -> [bsld, ...]}
    all_wait = {}            # tag -> [wait_hours, ...]
    all_bsld = {}            # tag -> [bsld, ...]
    all_wait_large    = {}
    all_bsld_large    = {}
    large_job_info    = {}
    all_wait_small    = {}
    all_bsld_small    = {}
    small_job_info    = {}
    all_job_ids_map   = {}   # tag -> [job_ids] (for global scatter)
    all_wait_maps     = {}   # tag -> {jid: wait_seconds} (for global scatter)
    all_backfill_maps = {}   # tag -> set(backfill job ids) (for global scatter)

    for tag in valid_tags:
        print(f"Driver: {tag}{' (partial)' if partial else ''}")

        submit = submit_maps[tag]
        start  = start_maps[tag]
        end    = end_maps[tag]

        # Restrict to jobs submitted within the common submit-time window.
        # This ensures all drivers are compared on the same submitted job set
        # even when some drivers have fewer completed jobs (partial results).
        job_ids  = []
        wait_map = {}
        for jid, submit_time in submit.items():
            if submit_time > common_submit_end:
                continue
            if jid not in start or jid not in end:
                continue
            wait_map[jid] = start[jid] - submit[jid]
            job_ids.append(jid)

        n_windowed = len(job_ids)
        if partial:
            print(f"  {n_windowed} jobs within common submit window")

        events_path = os.path.join(output_dir, tag, "events.csv")
        backfill_ids = parse_backfill_jobs(events_path) if os.path.exists(events_path) else set()

        if not global_only:
            plot_cycletime_vs_time(
                tag, output_dir, plots_dir,
                time_start_limit=analysis_start,
                time_end_limit=analysis_end,
            )
            plot_queuelength_vs_time(
                tag, output_dir, plots_dir, maintenance=maintenance,
                time_start_limit=analysis_start,
                time_end_limit=analysis_end,
            )
            plot_throughput_jobs_per_day(
                tag, end, plots_dir, maintenance=maintenance,
                end_time_limit=per_driver_end_time_limit,
            )
            plot_throughput_proc_hours_per_day(
                tag, end, procs_map, runtimes_map, plots_dir,
                maintenance=maintenance,
                end_time_limit=per_driver_end_time_limit,
            )
        plot_per_driver_wait_vs_size_scatter(
            tag, job_ids, wait_map, procs_map, plots_dir,
            driver_cfg=tag_to_driver_cfg.get(tag, {}),
            backfill_ids=backfill_ids,
        )
        # plot_per_driver_bsld_vs_size_scatter(
        #     tag, job_ids, wait_map, procs_map, walltimes_map, plots_dir,
        #     driver_cfg=tag_to_driver_cfg.get(tag, {}),
        #     backfill_ids=backfill_ids,
        # )
        # plot_per_driver_size_wait_kde(
        #     tag, job_ids, wait_map, procs_map, plots_dir,
        #     driver_cfg=tag_to_driver_cfg.get(tag, {}),
        #     class_ymax=class_wait_p99,
        # )

        # Save raw maps for the global group scatter
        all_job_ids_map[tag]   = job_ids
        all_wait_maps[tag]     = wait_map
        all_backfill_maps[tag] = backfill_ids

        # Accumulate data for global plots (windowed)
        waits_by_g = {g: [] for g in GROUP_LABELS}
        bsld_by_g  = {g: [] for g in GROUP_LABELS}
        waits_all  = []
        bsld_all   = []
        for jid in job_ids:
            w_h = wait_map[jid] / 3600.0
            waits_all.append(w_h)
            walltime = max(walltimes_map.get(jid, 1), 1)
            bsld_all.append((wait_map[jid] + walltime) / walltime)

            if jid not in procs_map:
                continue
            g = assign_group(procs_map[jid])
            if g is None:
                continue
            waits_by_g[g].append(w_h)
            bsld_by_g[g].append((wait_map[jid] + walltime) / walltime)

        all_wait_by_group[tag] = waits_by_g
        all_bsld_by_group[tag] = bsld_by_g
        all_wait[tag] = waits_all
        all_bsld[tag] = bsld_all

        large_threshold  = S_SYS * 0.5
        wait_lg, bsld_lg = [], []
        wt_lg, rt_lg     = [], []
        wait_sm, bsld_sm = [], []
        wt_sm, rt_sm     = [], []
        for jid in job_ids:
            if jid not in procs_map:
                continue
            w_h      = wait_map[jid] / 3600.0
            walltime = max(walltimes_map.get(jid, 1), 1)
            bsld_val = (wait_map[jid] + walltime) / walltime
            wt_h     = walltimes_map.get(jid, 0) / 3600.0
            rt_h     = runtimes_map.get(jid, 0) / 3600.0
            if procs_map[jid] > large_threshold:
                wait_lg.append(w_h);  bsld_lg.append(bsld_val)
                wt_lg.append(wt_h);   rt_lg.append(rt_h)
            else:
                wait_sm.append(w_h);  bsld_sm.append(bsld_val)
                wt_sm.append(wt_h);   rt_sm.append(rt_h)
        all_wait_large[tag] = wait_lg
        all_bsld_large[tag] = bsld_lg
        large_job_info[tag] = (len(wait_lg),
                               float(np.mean(wt_lg)) if wt_lg else 0.0,
                               float(np.mean(rt_lg)) if rt_lg else 0.0)
        all_wait_small[tag] = wait_sm
        all_bsld_small[tag] = bsld_sm
        small_job_info[tag] = (len(wait_sm),
                               float(np.mean(wt_sm)) if wt_sm else 0.0,
                               float(np.mean(rt_sm)) if rt_sm else 0.0)
        print()

    # ── Global plots ──────────────────────────────────────────────────────────
    print("Global plots:")
    throughput_end_time_limit = min(t_window_end, analysis_end) if partial else analysis_end
    plot_global_throughput_jobs_per_day(
        valid_tags,
        end_maps,
        driver_colors,
        plots_dir,
        maintenance=maintenance,
        end_time_limit=throughput_end_time_limit,
    )
    plot_global_throughput_proc_hours_per_day(
        valid_tags,
        end_maps,
        driver_colors,
        procs_map,
        runtimes_map,
        plots_dir,
        maintenance=maintenance,
        end_time_limit=throughput_end_time_limit,
    )
    plot_global_cumulative_completed_jobs(
        valid_tags,
        end_maps,
        driver_colors,
        plots_dir,
        maintenance=maintenance,
        end_time_limit=throughput_end_time_limit,
    )
    plot_global_overall_wait_cdfs(valid_tags, all_wait, driver_colors, plots_dir)
    plot_global_overall_bsld_cdfs(valid_tags, all_bsld, driver_colors, plots_dir)
    print()

    # ── Global tables ─────────────────────────────────────────────────────────
    print("Global tables:")
    write_global_overall_wait_table(valid_tags, all_wait, plots_dir)
    write_global_overall_bsld_table(valid_tags, all_bsld, plots_dir)
    write_global_wait_tables(valid_tags, all_wait_by_group, plots_dir)
    write_global_bsld_tables(valid_tags, all_bsld_by_group, plots_dir)
    write_global_size_job_tables("large", valid_tags, all_wait_large, all_bsld_large,
                                 large_job_info, plots_dir)
    write_global_size_job_tables("small", valid_tags, all_wait_small, all_bsld_small,
                                 small_job_info, plots_dir)
    util_window_end = min(t_window_end, analysis_end) if partial else analysis_end
    write_global_util_table(
        valid_tags,
        raw_end_maps,
        raw_start_maps,
        raw_submit_maps,
        procs_map,
        plots_dir,
        window_start=analysis_start,
        window_end=util_window_end,
    )
    write_global_util_maintenance_table(
        valid_tags,
        raw_end_maps,
        raw_start_maps,
        output_dir,
        procs_map,
        plots_dir,
        CST,
        window_start=analysis_start,
        window_end=util_window_end,
    )
    write_global_tail_heatmap(valid_tags, all_job_ids_map, all_wait_maps,
                              procs_map, walltimes_map, plots_dir)
    print()

    print(f"All plots saved to: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
