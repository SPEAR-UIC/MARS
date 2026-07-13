#!/usr/bin/env python3
"""Combined plotting script for exp2a (Theta 2021) and exp2b (Polaris 2024).

Detects the system automatically from swf_path in the experiment JSON:
  - theta21cln  → Theta tiers T1–T5 (128–4096 nodes, 2021 trace)
  - polaris24cln → Polaris groups S/M/L (10–496 nodes, 2024 trace)

Usage:
    python3 scripts/exp2_plot.py experiments/exp2a.json
    python3 scripts/exp2_plot.py experiments/exp2b.json
    python3 scripts/exp2_plot.py experiments/exp2a.json --plots-dir results/custom
    python3 scripts/exp2_plot.py experiments/exp2b.json --global-only
"""

import sys
import os
import json
import csv
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches
from matplotlib.ticker import FuncFormatter
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

PERCENTILES      = [25, 50, 75, 99]
_SCATTER_PCTILES = [25, 50, 75, 90]

# Argonne / Chicago local time — matches both Theta and Polaris submit timestamps
CST = timezone(timedelta(hours=-6))

# Driver-comparison palette (tab10, stable assignment by driver index)
_DRIVER_PALETTE = list(plt.cm.tab10.colors)

# MARS family: uniform light yellow, with hatch patterns carrying the distinction
_MARS_COLORS = {
    '-CW': '#f6de6a',   # yellow (wait reward)
    '-CB': '#b8860b',   # dark goldenrod (bsld reward)
    '-CU': '#9467bd',   # medium purple (cumulative util)
    '-IU': '#c5b0d5',   # light purple (instant util)
}
_MARS_HATCHES  = ['///', '...', '---', 'xxx']

# Heuristic colors matching plot_avg_wait_sensitivity.py, with FCFS moved to neutral gray
_HEURISTIC_COLORS = {
    'FCFS':        '#6e6e6e',   # neutral gray
    'SJF':         '#1f77b4',   # tab20[0] — blue
    'WFP':        '#aec7e8',   # tab20[1] — light blue
    'F1':          '#ff7f0e',   # tab20[2] — orange
    'RLSCHEDULER': '#d62728',   # tab20[6] — red (not in sensitivity script; kept distinct)
    'RANDOM':      '#2ca02c',   # tab20[4] — green
    'LRF':         '#98df8a',   # tab20[5] — light green
}

# Shaped markers for percentiles on multi-line CDF plots
_PCTILE_MARKER = {25: '^', 50: 'D', 75: 's', 99: '*'}
_PCTILE_SIZE   = {25: 7,   50: 8,   75: 7,   99: 10}
_BOX_PCTILES = [25, 50, 75, 95, 99]
_BOX_PCTILE_MARKER = {25: '^', 50: 'D', 75: 's', 95: 'v', 99: '*'}
_BOX_PCTILE_SIZE   = {25: 7,   50: 7,   75: 7,   95: 7,   99: 10}
_BOX_MARKER_XOFFSETS = {25: -0.19, 50: -0.095, 75: 0.00, 95: 0.095, 99: 0.19}
_BOX_MEAN_XOFFSET = 0.28

_STAT_HEADERS = ["Min", "P25", "P50", "P75", "P85", "P95", "P99", "Max",
                 "Mean", "StdDev", "GeometricMean"]
_BAR_STAT_HEADERS = ["P50", "P75", "P95", "P99", "Mean"]

# ── System configuration ──────────────────────────────────────────────────────

_THETA_CFG = dict(
    GROUP_LABELS  = ['S', 'M', 'L', 'XL'],
    GROUP_DISPLAY = {
        'S':  'S (=128)',
        'M':  'M (129–256)',
        'L':  'L (257–1024)',
        'XL': 'XL (1025–4096)',
    },
    GROUP_COLORS = {
        'S':  'steelblue',
        'M':  'darkorange',
        'L':  'seagreen',
        'XL': 'crimson',
    },
    S_SYS         = 4096,
    ANALYSIS_YEAR = 2021,
    SCATTER_XLIM  = (100, 5000),
    _ZONE_X = {
        'S':  (126,  130),   # single value =128; slight width for visibility
        'M':  (129,  256),
        'L':  (257, 1024),
        'XL': (1025, 4096),
    },
    _ZONE_DARK = {
        'S':  '#1a527a',
        'M':  '#994d00',
        'L':  '#1a6640',
        'XL': '#8b0000',
    },
    _GROUP_XLIM = {
        'S':  (115,  140),
        'M':  (115,  280),
        'L':  (240, 1100),
        'XL': (950, 4500),
    },
    _KDE_CMAPS = {
        'S':  'plasma',
        'M':  'inferno',
        'L':  'magma',
        'XL': 'viridis',
    },
    _TAIL_NODE_EDGES  = [128, 129, 257, 1025, 4097],
    _TAIL_NODE_LABELS = ['S', 'M', 'L', 'XL'],
    _TAIL_WT_EDGES    = [0, 0.5, 1.0, 2.0, 3.0, 5.0, 6.0, 9.0, 24.01],
    _TAIL_WT_LABELS   = ['≤0.5h', '≤1h', '≤2h', '≤3h', '≤5h', '≤6h', '≤9h', '≤24h'],
)

_POLARIS_CFG = dict(
    GROUP_LABELS  = ['S', 'M', 'L', 'XL'],
    GROUP_DISPLAY = {
        'S':  'S (10–16)',
        'M':  'M (17–32)',
        'L':  'L (33–128)',
        'XL': 'XL (129–496)',
    },
    GROUP_COLORS = {
        'S':  'steelblue',
        'M':  'darkorange',
        'L':  'seagreen',
        'XL': 'crimson',
    },
    S_SYS         = 496,
    ANALYSIS_YEAR = 2024,
    SCATTER_XLIM  = (8, 512),
    _ZONE_X = {
        'S':  (10,   16),
        'M':  (17,   32),
        'L':  (33,  128),
        'XL': (129, 496),
    },
    _ZONE_DARK = {
        'S':  '#1a527a',
        'M':  '#994d00',
        'L':  '#1a6640',
        'XL': '#8b0000',
    },
    _GROUP_XLIM = {
        'S':  (8,   20),
        'M':  (14,  38),
        'L':  (28, 145),
        'XL': (115, 560),
    },
    _KDE_CMAPS = {
        'S':  'plasma',
        'M':  'inferno',
        'L':  'magma',
        'XL': 'viridis',
    },
    _TAIL_NODE_EDGES  = [10, 17, 33, 129, 497],
    _TAIL_NODE_LABELS = ['S', 'M', 'L', 'XL'],
    _TAIL_WT_EDGES    = [0, 0.5, 1.0, 2.0, 3.0, 5.0, 6.0, 9.0, 24.01],
    _TAIL_WT_LABELS   = ['≤0.5h', '≤1h', '≤2h', '≤3h', '≤5h', '≤6h', '≤9h', '≤24h'],
)

# Module-level globals — populated by _configure_system() in main()
SYSTEM        = 'polaris'
GROUP_LABELS  = _POLARIS_CFG['GROUP_LABELS']
GROUP_DISPLAY = _POLARIS_CFG['GROUP_DISPLAY']
GROUP_COLORS  = _POLARIS_CFG['GROUP_COLORS']
S_SYS         = _POLARIS_CFG['S_SYS']
ANALYSIS_YEAR = _POLARIS_CFG['ANALYSIS_YEAR']
SCATTER_XLIM  = _POLARIS_CFG['SCATTER_XLIM']
_ZONE_X       = _POLARIS_CFG['_ZONE_X']
_ZONE_DARK    = _POLARIS_CFG['_ZONE_DARK']
_GROUP_XLIM   = _POLARIS_CFG['_GROUP_XLIM']
_KDE_CMAPS    = _POLARIS_CFG['_KDE_CMAPS']
_TAIL_NODE_EDGES  = _POLARIS_CFG['_TAIL_NODE_EDGES']
_TAIL_NODE_LABELS = _POLARIS_CFG['_TAIL_NODE_LABELS']
_TAIL_WT_EDGES    = _POLARIS_CFG['_TAIL_WT_EDGES']
_TAIL_WT_LABELS   = _POLARIS_CFG['_TAIL_WT_LABELS']


def _configure_system(system):
    """Set all system-specific module globals for 'theta' or 'polaris'."""
    global SYSTEM, GROUP_LABELS, GROUP_DISPLAY, GROUP_COLORS, S_SYS, ANALYSIS_YEAR
    global SCATTER_XLIM, _ZONE_X, _ZONE_DARK, _GROUP_XLIM, _KDE_CMAPS
    global _TAIL_NODE_EDGES, _TAIL_NODE_LABELS, _TAIL_WT_EDGES, _TAIL_WT_LABELS
    SYSTEM = system
    cfg = _THETA_CFG if system == 'theta' else _POLARIS_CFG
    GROUP_LABELS       = cfg['GROUP_LABELS']
    GROUP_DISPLAY      = cfg['GROUP_DISPLAY']
    GROUP_COLORS       = cfg['GROUP_COLORS']
    S_SYS              = cfg['S_SYS']
    ANALYSIS_YEAR      = cfg['ANALYSIS_YEAR']
    SCATTER_XLIM       = cfg['SCATTER_XLIM']
    _ZONE_X            = cfg['_ZONE_X']
    _ZONE_DARK         = cfg['_ZONE_DARK']
    _GROUP_XLIM        = cfg['_GROUP_XLIM']
    _KDE_CMAPS         = cfg['_KDE_CMAPS']
    _TAIL_NODE_EDGES   = cfg['_TAIL_NODE_EDGES']
    _TAIL_NODE_LABELS  = cfg['_TAIL_NODE_LABELS']
    _TAIL_WT_EDGES     = cfg['_TAIL_WT_EDGES']
    _TAIL_WT_LABELS    = cfg['_TAIL_WT_LABELS']


# ── Maintenance window helpers ────────────────────────────────────────────────

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
    sma_times = {}
    sms_times = {}
    sme_times = {}
    ums_times = {}
    ume_times = {}

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
    for idx in sorted(sms_times):
        if idx not in sme_times:
            continue
        windows.append((int(sms_times[idx]), int(sme_times[idx]), 'S'))
    for idx in sorted(ums_times):
        if idx not in ume_times:
            continue
        windows.append((int(ums_times[idx]), int(ume_times[idx]), 'U'))

    return sorted(windows, key=lambda w: w[0])


def add_maintenance_shading(ax, maintenance, tz):
    if not maintenance:
        return False

    ylims = ax.get_ylim()
    label_y = ylims[1] * 0.97

    drawn = False
    for start_u, end_u, mtype in maintenance:
        start_dt = datetime.fromtimestamp(start_u, tz=tz)
        end_dt   = datetime.fromtimestamp(end_u,   tz=tz)
        style    = _MAINT_SCHED if mtype == 'S' else _MAINT_UNSCHED
        ax.axvspan(start_dt, end_dt, **style)

        mid_dt = start_dt + (end_dt - start_dt) / 2
        ax.text(mid_dt, label_y, mtype,
                ha='center', va='top', fontsize=8, fontweight='bold',
                color='#4477aa' if mtype == 'S' else '#cc4444',
                clip_on=True)
        drawn = True

    return drawn


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_backfill_jobs(events_csv_path):
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
    if SYSTEM == 'theta':
        if num_procs == 128:               return 'S'
        if 129  <= num_procs <= 256:       return 'M'
        if 257  <= num_procs <= 1024:      return 'L'
        if 1025 <= num_procs <= 4096:      return 'XL'
        return None
    else:  # polaris
        if 10  <= num_procs <= 16:  return 'S'
        if 17  <= num_procs <= 32:  return 'M'
        if 33  <= num_procs <= 128: return 'L'
        if 129 <= num_procs <= 496: return 'XL'
        return None


def apply_style(ax, xlabel=None, ylabel=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)


_FORCE_REGEN = False   # set to True via --force CLI flag
_EXISTING_PLOTS: set = set()  # populated once at startup by _scan_existing_plots()


def _scan_existing_plots(plots_dir: str) -> None:
    """Scan plots_dir once at startup; populate _EXISTING_PLOTS with abspaths."""
    global _EXISTING_PLOTS
    import glob
    _EXISTING_PLOTS = {
        os.path.abspath(p)
        for p in glob.glob(os.path.join(plots_dir, '**', '*.png'), recursive=True)
    }
    if _EXISTING_PLOTS and not _FORCE_REGEN:
        print(f"  Skipping {len(_EXISTING_PLOTS)} existing plots "
              f"(use --force to regenerate)")


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not _FORCE_REGEN and os.path.abspath(path) in _EXISTING_PLOTS:
        plt.close(fig)
        return
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


def driver_color(idx):
    return _DRIVER_PALETTE[idx % len(_DRIVER_PALETTE)]


def driver_hatch(tag):
    tag_up = tag.upper()
    for idx, suffix in enumerate(("-CW", "-CB", "-CU", "-IU")):
        if tag_up.endswith(suffix):
            return _MARS_HATCHES[idx]
    return None


def display_tag(tag):
    """Strip trailing window-size suffix (e.g. '-w256', '-w1024') for plot labels."""
    import re
    return re.sub(r'-w\d+$', '', tag, flags=re.IGNORECASE)


def tag_slug(tag):
    return display_tag(tag).strip().lower().replace('-', '_').replace(' ', '_')


def _compute_cdf(values):
    sv  = np.sort(values)
    cdf = np.arange(1, len(sv) + 1) / len(sv)
    return sv, cdf


def _pctile_legend_handles():
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
    if not values:
        return
    sv, cdf = _compute_cdf(values)
    ax.plot(sv, cdf, color=color, linewidth=LINE_WIDTH, label=display_tag(label))

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
            procs   = float(procs_map[jid])
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
                label=display_tag(tag), zorder=3)
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
                label=display_tag(tag), zorder=3)
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
                label=display_tag(tag), zorder=3)
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


# ── Per-driver: Stacked wait-time histogram by size group ────────────────────

def _wait_hist_bin_width(max_wait_h):
    if max_wait_h <= 24:
        return 1.0
    if max_wait_h <= 72:
        return 2.0
    if max_wait_h <= 168:
        return 4.0
    if max_wait_h <= 336:
        return 8.0
    return float(max(12.0, np.ceil(max_wait_h / 40.0)))


def _build_wait_distribution_by_group(job_ids, wait_map, procs_map, exclude_groups=None):
    exclude_groups = set(exclude_groups or [])
    waits_by_group = {g: [] for g in GROUP_LABELS}
    for jid in job_ids:
        if jid not in wait_map or jid not in procs_map:
            continue
        group = assign_group(procs_map[jid])
        if group is None or group in exclude_groups:
            continue
        waits_by_group[group].append(wait_map[jid] / 3600.0)

    plot_groups = [g for g in GROUP_LABELS if waits_by_group[g]]
    if not plot_groups:
        return None

    all_waits = np.array(
        [wait_h for group in plot_groups for wait_h in waits_by_group[group]],
        dtype=float,
    )
    positive_waits = all_waits[all_waits > 0]
    if positive_waits.size == 0:
        # Put zero-wait jobs in a tiny positive bucket so log-scale histograms render.
        x_lo = 1.0 / 60.0
        x_hi = 1.0
    else:
        x_lo = max((positive_waits.min() / 2.0), 1.0 / 60.0)
        x_hi = max(positive_waits.max(), x_lo * 10.0)

    n_bins = int(np.clip(np.ceil(np.log10(x_hi / x_lo)) * 8.0, 12, 48))
    bin_edges = np.geomspace(x_lo, x_hi, num=n_bins + 1)

    # Zero waits cannot be shown directly on a log x-axis, so place them in the first bin.
    waits_for_plot = []
    for group in plot_groups:
        arr = np.array(waits_by_group[group], dtype=float)
        arr = np.where(arr > 0, arr, x_lo)
        waits_for_plot.append(arr)

    counts_by_group = []
    for arr in waits_for_plot:
        counts, _ = np.histogram(arr, bins=bin_edges)
        counts_by_group.append(counts.astype(float))

    x_mid = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    return plot_groups, bin_edges, waits_for_plot, counts_by_group, x_mid


def _build_wait_distribution_shared_scale(distributions):
    dists = [dist for dist in distributions if dist is not None]
    if not dists:
        return None

    x_lo = min(float(dist[1][0]) for dist in dists)
    x_hi = max(float(dist[1][-1]) for dist in dists)
    y_hi = 1.0
    for dist in dists:
        counts_by_group = dist[3]
        if not counts_by_group:
            continue
        total_counts = np.sum(np.vstack(counts_by_group), axis=0)
        if total_counts.size:
            y_hi = max(y_hi, float(total_counts.max()))

    x_lo = 10 ** np.floor(np.log10(max(x_lo, 1e-6)))
    x_hi = 10 ** np.ceil(np.log10(max(x_hi, x_lo * 10.0)))
    y_hi = 10 ** np.ceil(np.log10(max(y_hi, 1.0)))
    return dict(x_lo=x_lo, x_hi=x_hi, y_hi=y_hi)


def _wait_kde_curve(waits_h, x_grid):
    arr = np.asarray(waits_h, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return np.zeros_like(x_grid, dtype=float)

    z = np.log10(arr)
    z_grid = np.log10(x_grid)

    if z.size == 1:
        bw = 0.12
    else:
        std = float(np.std(z, ddof=1))
        if not np.isfinite(std) or std <= 1e-9:
            bw = 0.12
        else:
            bw = max(1.06 * std * (z.size ** (-1.0 / 5.0)), 0.08)

    diffs = (z_grid[:, None] - z[None, :]) / bw
    dens = np.exp(-0.5 * diffs * diffs).sum(axis=1)
    dens /= (z.size * bw * np.sqrt(2.0 * np.pi))
    return dens


def _build_wait_kde_shared_scale(distributions, x_grid):
    dists = [dist for dist in distributions if dist is not None]
    if not dists:
        return None

    y_hi = 0.0
    for dist in dists:
        waits_for_plot = dist[2]
        for arr in waits_for_plot:
            dens = _wait_kde_curve(arr, x_grid)
            if dens.size:
                y_hi = max(y_hi, float(dens.max()))

    return dict(y_lo=0.0, y_hi=max(y_hi * 1.05, 1.0))


def plot_per_driver_stacked_wait_histogram(tag, job_ids, wait_map, procs_map, plots_dir,
                                           dist=None, shared_scale=None,
                                           filename="stacked_wait_time_histogram.png"):
    if dist is None:
        dist = _build_wait_distribution_by_group(job_ids, wait_map, procs_map)
    if dist is None:
        print(f"  [skip] {tag}: no wait-time data for stacked histogram")
        return

    plot_groups, bin_edges, waits_for_plot, counts_by_group, x_mid = dist

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.hist(
        waits_for_plot,
        bins=bin_edges,
        stacked=True,
        color=[GROUP_COLORS[g] for g in plot_groups],
        alpha=0.88,
        edgecolor='white',
        linewidth=0.45,
        label=plot_groups,
    )
    apply_style(ax, xlabel="Wait Time (hours)", ylabel="Job Count")
    ax.set_xscale('log')
    ax.set_yscale('log')
    if shared_scale is not None:
        ax.set_xlim(left=shared_scale['x_lo'], right=shared_scale['x_hi'])
        ax.set_ylim(bottom=1.0, top=shared_scale['y_hi'])
    else:
        ax.set_xlim(left=bin_edges[0], right=bin_edges[-1])
        ax.set_ylim(bottom=1.0)
    legend_handles = [
        matplotlib.patches.Patch(facecolor=GROUP_COLORS[g], edgecolor='none', label=g)
        for g in plot_groups
    ]
    ax.legend(handles=legend_handles, fontsize=LEGEND_SIZE - 1, loc='upper right', ncol=min(4, len(plot_groups)))
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, filename))


def plot_per_driver_wait_kde_by_size(tag, job_ids, wait_map, procs_map, plots_dir,
                                     dist=None, shared_scale=None, kde_scale=None,
                                     filename="wait_time_kde_by_size.png"):
    if dist is None:
        dist = _build_wait_distribution_by_group(job_ids, wait_map, procs_map)
    if dist is None:
        print(f"  [skip] {tag}: no wait-time data for KDE plot")
        return

    plot_groups, bin_edges, waits_for_plot, counts_by_group, x_mid = dist
    if shared_scale is not None:
        x_grid = np.geomspace(shared_scale['x_lo'], shared_scale['x_hi'], 400)
    else:
        x_grid = np.geomspace(bin_edges[0], bin_edges[-1], 400)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for group, arr in zip(plot_groups, waits_for_plot):
        dens = _wait_kde_curve(arr, x_grid)
        if not np.any(dens > 0):
            continue
        ax.plot(
            x_grid,
            dens,
            color=GROUP_COLORS[group],
            linewidth=LINE_WIDTH,
            label=group,
        )

    apply_style(ax, xlabel="Wait Time (hours)", ylabel="Density")
    ax.set_xscale('log')
    if shared_scale is not None:
        ax.set_xlim(left=shared_scale['x_lo'], right=shared_scale['x_hi'])
    else:
        ax.set_xlim(left=bin_edges[0], right=bin_edges[-1])
    if kde_scale is not None:
        ax.set_ylim(bottom=kde_scale['y_lo'], top=kde_scale['y_hi'])
    else:
        ax.set_ylim(bottom=0.0)
    legend_handles = [
        Line2D([0], [0], color=GROUP_COLORS[g], linewidth=LINE_WIDTH, label=g)
        for g in plot_groups
    ]
    ax.legend(handles=legend_handles, fontsize=LEGEND_SIZE - 1,
              loc='upper right', ncol=min(4, len(plot_groups)))
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, filename))


def plot_per_driver_stream_wait_area(tag, job_ids, wait_map, procs_map, plots_dir,
                                     dist=None, shared_scale=None,
                                     filename="stream_wait_time_area.png"):
    if dist is None:
        dist = _build_wait_distribution_by_group(job_ids, wait_map, procs_map)
    if dist is None:
        print(f"  [skip] {tag}: no wait-time data for stream area plot")
        return

    plot_groups, bin_edges, waits_for_plot, counts_by_group, x_mid = dist

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.stackplot(
        x_mid,
        *counts_by_group,
        labels=plot_groups,
        colors=[GROUP_COLORS[g] for g in plot_groups],
        alpha=0.88,
        baseline='wiggle',
        linewidth=0.4,
        edgecolor='white',
    )
    apply_style(ax, xlabel="Wait Time (hours)", ylabel="Stream Offset")
    ax.set_xscale('log')
    # A streamgraph uses a floating baseline, so negative y-values are visual offsets,
    # not negative job counts. The band thickness is the quantity.
    ax.set_yscale('symlog', linthresh=1.0)
    if shared_scale is not None:
        ax.set_xlim(left=shared_scale['x_lo'], right=shared_scale['x_hi'])
        ax.set_ylim(bottom=-shared_scale['y_hi'], top=shared_scale['y_hi'])
    else:
        ax.set_xlim(left=bin_edges[0], right=bin_edges[-1])
    ax.axhline(0.0, color='#888888', linewidth=0.8, alpha=0.8)
    legend_handles = [
        matplotlib.patches.Patch(facecolor=GROUP_COLORS[g], edgecolor='none', label=g)
        for g in plot_groups
    ]
    ax.legend(handles=legend_handles, fontsize=LEGEND_SIZE - 1, loc='upper right', ncol=min(4, len(plot_groups)))
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, filename))


def plot_per_driver_stacked_wait_area(tag, job_ids, wait_map, procs_map, plots_dir,
                                      dist=None, shared_scale=None,
                                      filename="stacked_wait_time_area.png"):
    if dist is None:
        dist = _build_wait_distribution_by_group(job_ids, wait_map, procs_map)
    if dist is None:
        print(f"  [skip] {tag}: no wait-time data for stacked area plot")
        return

    plot_groups, bin_edges, waits_for_plot, counts_by_group, x_mid = dist

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.stackplot(
        x_mid,
        *counts_by_group,
        labels=plot_groups,
        colors=[GROUP_COLORS[g] for g in plot_groups],
        alpha=0.88,
        baseline='zero',
        linewidth=0.4,
        edgecolor='white',
    )
    apply_style(ax, xlabel="Wait Time (hours)", ylabel="Job Count")
    ax.set_xscale('log')
    ax.set_yscale('log')
    if shared_scale is not None:
        ax.set_xlim(left=shared_scale['x_lo'], right=shared_scale['x_hi'])
        ax.set_ylim(bottom=1.0, top=shared_scale['y_hi'])
    else:
        ax.set_xlim(left=bin_edges[0], right=bin_edges[-1])
        ax.set_ylim(bottom=1.0)
    legend_handles = [
        matplotlib.patches.Patch(facecolor=GROUP_COLORS[g], edgecolor='none', label=g)
        for g in plot_groups
    ]
    ax.legend(handles=legend_handles, fontsize=LEGEND_SIZE - 1, loc='upper right', ncol=min(4, len(plot_groups)))
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, filename))


# ── Per-driver: Wait Time vs Size Scatter ─────────────────────────────────────

def plot_per_driver_wait_vs_size_scatter(tag, job_ids, wait_map, procs_map,
                                         plots_dir, driver_cfg=None,
                                         backfill_ids=None):
    if driver_cfg is None:
        driver_cfg = {}
    if backfill_ids is None:
        backfill_ids = set()

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

    # Reward=0.5 iso-line
    reward_type = driver_cfg.get('reward_type', '')
    A = driver_cfg.get('size_adj_A') or driver_cfg.get('reward_param_a')
    n = driver_cfg.get('size_adj_n') or driver_cfg.get('reward_param_n', 1.0)
    if reward_type in ('queuesnapshot', 'harmonic_wait_sized') and A is not None:
        x_min = min(v[0] for v in _ZONE_X.values())
        x_max = max(v[1] for v in _ZONE_X.values())
        B_values = np.linspace(x_min, x_max, 500)
        w_half = (A / 3600.0) * (B_values / S_SYS) ** n
        ax.plot(B_values, w_half, color='black', linestyle='-', linewidth=1.0, zorder=11)

    # Zone backgrounds
    for g in GROUP_LABELS:
        ax.axvspan(_ZONE_X[g][0], _ZONE_X[g][1], color=GROUP_COLORS[g], alpha=0.15, zorder=1)

    fixed_ymax = 48.0
    ax.set_ylim(bottom=0, top=fixed_ymax)
    label_y = fixed_ymax * 0.9

    _total_scatter = sum(len(v) for v in waits_by_group.values())
    for g in GROUP_LABELS:
        x_lo, x_hi = _ZONE_X[g]
        _pos = np.sqrt(x_lo * x_hi)
        _pct = len(waits_by_group[g]) / _total_scatter * 100 if _total_scatter else 0
        ax.text(_pos, label_y, f'{g}\n({_pct:.1f}%)', ha='center',
                fontweight='bold', alpha=0.6, fontsize=12, linespacing=1.4)

    for g, waits_g in waits_by_group.items():
        if not waits_g:
            continue
        arr = np.array(waits_g)
        x_lo, x_hi = _ZONE_X[g]
        for p in [50, 75, 90, 99]:
            pv = float(np.percentile(arr, p))
            if pv > fixed_ymax:
                continue
            ax.hlines(pv, x_lo, x_hi, colors=_ZONE_DARK[g], linestyles='--',
                      linewidth=1.3, zorder=8)

    apply_style(ax, xlabel="Node Size (log scale)", ylabel="Wait Time (hours)")
    ax.set_xscale('log')
    ax.set_xlim(*SCATTER_XLIM)
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, "scatter_wait_vs_size.png"))


# ── Per-driver: BSLD vs Size Scatter ─────────────────────────────────────────

def plot_per_driver_bsld_vs_size_scatter(tag, job_ids, wait_map, procs_map,
                                          walltimes_map, plots_dir,
                                          driver_cfg=None, backfill_ids=None):
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

    reward_type = driver_cfg.get('reward_type', '')
    A = driver_cfg.get('reward_param_a')
    n = driver_cfg.get('reward_param_n', 1.0)
    if A is not None:
        x_min = min(v[0] for v in _ZONE_X.values())
        x_max = max(v[1] for v in _ZONE_X.values())
        B_values = np.linspace(x_min, x_max, 500)
        if reward_type == 'harmonic_bsld_sized':
            bsld_line = A * (S_SYS / B_values) ** (1.0 / n)
            ax.plot(B_values, bsld_line, color='black', linestyle='-',
                    linewidth=1.0, zorder=11)
        elif reward_type == 'harmonic_wait_sized':
            wt_median_h = float(np.median(all_wt_h)) if all_wt_h else 1.0
            w_line_h = A * (S_SYS / B_values) ** (1.0 / n)
            bsld_line = 1.0 + w_line_h / wt_median_h
            ax.plot(B_values, bsld_line, color='black', linestyle='-',
                    linewidth=1.0, zorder=11)

    for g in GROUP_LABELS:
        ax.axvspan(_ZONE_X[g][0], _ZONE_X[g][1], color=GROUP_COLORS[g], alpha=0.15, zorder=1)

    fixed_ymax = 12.0
    ax.set_ylim(bottom=1.0, top=fixed_ymax)
    label_y = 1.0 + (fixed_ymax - 1.0) * 0.9

    _total = sum(len(v) for v in bsld_by_group.values())
    for g in GROUP_LABELS:
        x_lo, x_hi = _ZONE_X[g]
        _pos = np.sqrt(x_lo * x_hi)
        _pct = len(bsld_by_group[g]) / _total * 100 if _total else 0
        ax.text(_pos, label_y, f'{g}\n({_pct:.1f}%)', ha='center',
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
    ax.set_xlim(*SCATTER_XLIM)
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, "scatter_bsld_vs_size.png"))


# ── Per-driver: Job Size vs Wait Time — Kernel Density ───────────────────────

def plot_per_driver_size_wait_kde(tag, job_ids, wait_map, procs_map,
                                   plots_dir, driver_cfg=None,
                                   class_ymax=None):
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

    data = {g: {'ls': [], 'w': []} for g in GROUP_LABELS}
    for jid in job_ids:
        if jid not in wait_map or jid not in procs_map:
            continue
        g = assign_group(procs_map[jid])
        if g is None:
            continue
        data[g]['ls'].append(np.log10(procs_map[jid]))
        data[g]['w'].append(wait_map[jid] / 3600.0)

    n_groups = len(GROUP_LABELS)
    fig_w = FIG_W * (n_groups / 3.0) * 1.6
    fig, axes = plt.subplots(1, n_groups, figsize=(fig_w, FIG_H + 1))
    if n_groups == 1:
        axes = [axes]

    _GC = '#333333'
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
                              cmap=_KDE_CMAPS.get(g, 'plasma'),
                              shading='gouraud', zorder=3)

        if reward_type == 'harmonic_wait_sized' and A is not None:
            B_vals = np.linspace(x_lo, x_hi, 400)
            w_half  = A * (S_SYS / B_vals) ** (1.0 / n)
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

    fig.suptitle(f'Job Size vs Wait Time (KDE) — {display_tag(tag)}', fontsize=FONT_SIZE)
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, "size_wait_kde.png"))


# ── Per-driver: Normal-only and Backfill-only scatter helpers ────────────────

def _draw_scatter_subset(ax, sizes, waits, waits_by_group, driver_cfg, tag,
                         fixed_ymax=48.0, total_trace=None):
    if sizes:
        ax.scatter(sizes, waits, s=12, alpha=0.25, color='black',
                   marker='o', zorder=10, edgecolors='none')

    reward_type = driver_cfg.get('reward_type', '')
    A = driver_cfg.get('size_adj_A') or driver_cfg.get('reward_param_a')
    n = driver_cfg.get('size_adj_n') or driver_cfg.get('reward_param_n', 1.0)
    if reward_type in ('queuesnapshot', 'harmonic_wait_sized') and A is not None:
        x_min = min(v[0] for v in _ZONE_X.values())
        x_max = max(v[1] for v in _ZONE_X.values())
        B_values = np.linspace(x_min, x_max, 500)
        w_half = (A / 3600.0) * (B_values / S_SYS) ** n
        ax.plot(B_values, w_half, color='black', linestyle='-',
                linewidth=1.0, zorder=11)

    for g in GROUP_LABELS:
        ax.axvspan(_ZONE_X[g][0], _ZONE_X[g][1], color=GROUP_COLORS[g], alpha=0.15, zorder=1)

    ax.set_ylim(bottom=0, top=fixed_ymax)
    label_y = fixed_ymax * 0.9
    _denom = total_trace if total_trace else sum(len(v) for v in waits_by_group.values())
    for g in GROUP_LABELS:
        x_lo, x_hi = _ZONE_X[g]
        _pos = np.sqrt(x_lo * x_hi)
        _pct = len(waits_by_group[g]) / _denom * 100 if _denom else 0
        ax.text(_pos, label_y, f'{g}\n({_pct:.1f}%)', ha='center',
                fontweight='bold', alpha=0.6, fontsize=12, linespacing=1.4)

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
    ax.set_xlim(*SCATTER_XLIM)


def _draw_violin_row(axes_by_group, walltimes_by_group):
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
    if driver_cfg   is None: driver_cfg   = {}
    if backfill_ids is None: backfill_ids = set()
    if walltimes_map is None: walltimes_map = {}

    n_groups = len(GROUP_LABELS)
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

    fig_w = FIG_W + max(0, n_groups - 3) * 2
    fig = plt.figure(figsize=(fig_w, FIG_H + 3))
    gs  = fig.add_gridspec(2, n_groups, height_ratios=[1, 2.5], hspace=0.45, wspace=0.35)
    ax_vio = {g: fig.add_subplot(gs[0, i]) for i, g in enumerate(GROUP_LABELS)}
    ax_sct = fig.add_subplot(gs[1, :])

    _draw_violin_row(ax_vio, walltimes_by_group)
    _draw_scatter_subset(ax_sct, sizes, waits, waits_by_group, driver_cfg, tag,
                         total_trace=total_trace)
    save(fig, os.path.join(plots_dir, tag, "scatter_wait_vs_size_normal.png"))


def plot_per_driver_scatter_backfill(tag, job_ids, wait_map, procs_map,
                                     plots_dir, driver_cfg=None, backfill_ids=None,
                                     walltimes_map=None):
    if driver_cfg   is None: driver_cfg   = {}
    if backfill_ids is None: backfill_ids = set()
    if walltimes_map is None: walltimes_map = {}

    n_groups = len(GROUP_LABELS)
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

    fig_w = FIG_W + max(0, n_groups - 3) * 2
    fig = plt.figure(figsize=(fig_w, FIG_H + 3))
    gs  = fig.add_gridspec(2, n_groups, height_ratios=[1, 2.5], hspace=0.45, wspace=0.35)
    ax_vio = {g: fig.add_subplot(gs[0, i]) for i, g in enumerate(GROUP_LABELS)}
    ax_sct = fig.add_subplot(gs[1, :])

    _draw_violin_row(ax_vio, walltimes_by_group)
    _draw_scatter_subset(ax_sct, sizes, waits, waits_by_group, driver_cfg, tag,
                         total_trace=total_trace)
    save(fig, os.path.join(plots_dir, tag, "scatter_wait_vs_size_backfill.png"))


# ── Per-driver: Wait vs Submit Time ──────────────────────────────────────────

def plot_per_driver_wait_vs_submit_time(tag, job_ids, wait_map, submit_map,
                                        procs_map, plots_dir):
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


def _decision_csv_path(results_dir, tag):
    for name in ("descisions.csv", "decisions.csv"):
        path = os.path.join(results_dir, tag, name)
        if os.path.exists(path):
            return path
    return None


def _load_mcts_cycle_rows(results_dir, tag, time_start_limit=None, time_end_limit=None):
    perf_path = os.path.join(results_dir, tag, "performance.csv")
    decision_path = _decision_csv_path(results_dir, tag)
    if not os.path.exists(perf_path) or decision_path is None:
        return []

    perf_rows = {}
    with open(perf_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cycle = int(row["cycle"])
                sim_time = float(row["sim_time"])
                selected_policy = row["selected_policy"].strip().upper()
                mcts_iterations = int(float(row["mcts_iterations"]))
                jobs_run = int(float(row.get("jobs_run", 0) or 0))
            except (KeyError, ValueError, TypeError, AttributeError):
                continue
            if time_start_limit is not None and sim_time < time_start_limit:
                continue
            if time_end_limit is not None and sim_time >= time_end_limit:
                continue
            perf_rows[cycle] = (sim_time, selected_policy, mcts_iterations, jobs_run)

    samples = []
    with open(decision_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cycle = int(row["cycle"])
                root_branching = int(float(row["root_branching_factor"]))
                decision_policy = row["selected_policy"].strip().upper()
                possible_job_sets = row.get("possible_job_sets", "")
            except (KeyError, ValueError, TypeError):
                continue
            perf_row = perf_rows.get(cycle)
            if perf_row is None:
                continue
            sim_time, perf_policy, mcts_iterations, jobs_run = perf_row
            samples.append({
                "cycle": cycle,
                "sim_time": sim_time,
                "perf_policy": perf_policy,
                "decision_policy": decision_policy,
                "mcts_iterations": mcts_iterations,
                "root_branching": root_branching,
                "jobs_run": jobs_run,
                "possible_job_sets": possible_job_sets,
            })
    return samples


def _parse_mcts_cycle_metrics(results_dir, tag, time_start_limit=None, time_end_limit=None):
    rows = _load_mcts_cycle_rows(
        results_dir, tag,
        time_start_limit=time_start_limit,
        time_end_limit=time_end_limit,
    )
    samples = []
    for row in rows:
        selected_policy = row["decision_policy"]
        mcts_iterations = row["mcts_iterations"]
        root_branching = row["root_branching"]
        if selected_policy != "MCTS":
            continue
        if root_branching <= 2:
            continue
        samples.append((mcts_iterations, root_branching))
    return samples


def _count_drain_events(rows):
    """Count DRAIN episodes, collapsing back-to-back DRAIN cycles with no runs."""
    rows = sorted(rows, key=lambda row: (row["sim_time"], row["cycle"]))
    event_count = 0
    in_drain_streak = False

    for row in rows:
        jobs_run = int(row.get("jobs_run", 0) or 0)
        is_drain = row.get("decision_policy") == "DRAIN"

        if is_drain:
            if not in_drain_streak:
                event_count += 1
                in_drain_streak = True
            if jobs_run > 0:
                in_drain_streak = False
        else:
            in_drain_streak = False

    return event_count


def plot_per_driver_mcts_iterations_violin(tag, results_dir, plots_dir,
                                           driver_color_value=None,
                                           time_start_limit=None, time_end_limit=None):
    metric_pairs = _parse_mcts_cycle_metrics(
        results_dir, tag,
        time_start_limit=time_start_limit,
        time_end_limit=time_end_limit,
    )
    if not metric_pairs:
        print(f"  [skip] {tag}: no MCTS cycles with root branching factor > 2")
        return
    samples = [iters for iters, _ in metric_pairs]

    fig, ax = plt.subplots(figsize=(max(6.0, FIG_W * 0.7), FIG_H))
    parts = ax.violinplot([samples], positions=[1], showmedians=True,
                          showextrema=False, widths=0.8)

    violin_color = driver_color_value or 'seagreen'
    hatch = driver_hatch(tag)
    for body in parts.get('bodies', []):
        body.set_facecolor(violin_color)
        body.set_edgecolor('black')
        body.set_alpha(0.85)
        if hatch:
            body.set_hatch(hatch)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.4)

    ax.set_xticks([1])
    ax.set_xticklabels([display_tag(tag)], fontsize=TICK_SIZE)
    apply_style(ax, xlabel="Driver", ylabel="MCTS Iterations")
    ax.set_yscale('log')
    ax.set_ylim(bottom=max(1.0, min(samples) * 0.8))
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, "mcts_iterations_violin.png"))


def plot_per_driver_root_branching_violin(tag, results_dir, plots_dir,
                                          driver_color_value=None,
                                          time_start_limit=None, time_end_limit=None):
    metric_pairs = _parse_mcts_cycle_metrics(
        results_dir, tag,
        time_start_limit=time_start_limit,
        time_end_limit=time_end_limit,
    )
    if not metric_pairs:
        print(f"  [skip] {tag}: no MCTS cycles with root branching factor > 2")
        return
    samples = [root_branching for _, root_branching in metric_pairs]

    fig, ax = plt.subplots(figsize=(max(6.0, FIG_W * 0.7), FIG_H))
    parts = ax.violinplot([samples], positions=[1], showmedians=True,
                          showextrema=False, widths=0.8)

    violin_color = driver_color_value or 'seagreen'
    hatch = driver_hatch(tag)
    for body in parts.get('bodies', []):
        body.set_facecolor(violin_color)
        body.set_edgecolor('black')
        body.set_alpha(0.85)
        if hatch:
            body.set_hatch(hatch)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.4)

    ax.set_xticks([1])
    ax.set_xticklabels([display_tag(tag)], fontsize=TICK_SIZE)
    apply_style(ax, xlabel="Driver", ylabel="Root Branching Factor")
    ax.set_ylim(bottom=2)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, tag, "root_branching_violin.png"))


def plot_global_mcts_drain_rate(mcts_tags, results_dir, driver_colors, plots_dir,
                                time_start_limit=None, time_end_limit=None):
    if not mcts_tags:
        print("  [skip] mcts_drain_rate: no MCTS drivers")
        return

    labels = []
    values = []
    counts = []
    for tag in mcts_tags:
        rows = _load_mcts_cycle_rows(
            results_dir, tag,
            time_start_limit=time_start_limit,
            time_end_limit=time_end_limit,
        )
        eligible = [row for row in rows if row["root_branching"] >= 2]
        drain_event_count = _count_drain_events(eligible)
        pct = 100.0 * drain_event_count / len(eligible) if eligible else 0.0
        labels.append(display_tag(tag))
        values.append(pct)
        counts.append((drain_event_count, len(eligible)))

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    x = np.arange(len(mcts_tags))
    bars = []
    for idx, tag in enumerate(mcts_tags):
        hatch = driver_hatch(tag)
        bar = ax.bar(
            x[idx], values[idx], width=0.65,
            color=driver_colors.get(tag, 'gray'),
            alpha=0.85, hatch=hatch,
            edgecolor='black' if hatch else None,
        )
        bars.append(bar[0])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax.set_ylabel("Drain Events (%)", fontsize=FONT_SIZE)
    ax.set_xlabel("MCTS Strategy", fontsize=FONT_SIZE)
    ax.set_ylim(0, max(5.0, max(values, default=0.0) * 1.15 + 1.0))
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, pct, (n_drain, n_total) in zip(bars, values, counts):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y + 0.6,
            f"{pct:.2f}%\n({n_drain}/{n_total})",
            ha='center', va='bottom', fontsize=9,
        )

    fig.tight_layout()
    save(fig, os.path.join(plots_dir, "bar_chart_mcts_drain_pct.png"))


def _parse_possible_job_sets(raw):
    import re

    if not raw:
        return []
    job_ids = set()
    for token in re.findall(r'\|([^|]*)\|', raw):
        token = token.strip()
        if not token or token.upper() == "NONE":
            continue
        for part in token.split(','):
            part = part.strip()
            if not part or part.upper() == "NONE":
                continue
            try:
                job_ids.add(int(part))
            except ValueError:
                continue
    return sorted(job_ids)


def _run_job_ids_by_time(results_dir, tag):
    events_path = os.path.join(results_dir, tag, "events.csv")
    runs = defaultdict(list)
    if not os.path.exists(events_path):
        return runs
    with open(events_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                event = row["event"]
                sim_time = int(float(row["sim_time"]))
                job_id = int(row["id"])
            except (KeyError, ValueError, TypeError):
                continue
            if event == "Run":
                runs[sim_time].append(job_id)
    return runs


def _ordered_run_events(results_dir, tag, time_start_limit=None, time_end_limit=None):
    events_path = os.path.join(results_dir, tag, "events.csv")
    records = []
    if not os.path.exists(events_path):
        return records
    with open(events_path) as f:
        reader = csv.DictReader(f)
        seq = 0
        for row in reader:
            try:
                if row["event"] != "Run":
                    continue
                sim_time = float(row["sim_time"])
                job_id = int(row["id"])
            except (KeyError, ValueError, TypeError):
                continue
            if time_start_limit is not None and sim_time < time_start_limit:
                continue
            if time_end_limit is not None and sim_time >= time_end_limit:
                continue
            records.append((sim_time, seq, job_id))
            seq += 1
    records.sort(key=lambda item: (item[0], item[1]))
    return [(sim_time, job_id) for sim_time, _, job_id in records]


def _collapse_drain_rows_by_run_events(drain_rows, run_events):
    """Collapse repeated DRAIN cycles before the next run into one latest DRAIN event."""
    if not drain_rows:
        return []
    if not run_events:
        return list(drain_rows)

    run_times = [sim_time for sim_time, _ in run_events]
    collapsed = []
    current = drain_rows[0]

    for row in drain_rows[1:]:
        run_lo = bisect_right(run_times, current["sim_time"])
        run_hi = bisect_right(run_times, row["sim_time"])
        if run_hi > run_lo:
            collapsed.append(current)
            current = row
        else:
            current = row

    collapsed.append(current)
    return collapsed


def _summarize_job_collection(job_ids, procs_map, walltimes_map):
    group_counts = {g: 0 for g in GROUP_LABELS}
    group_nodes = {g: 0.0 for g in GROUP_LABELS}
    other_count = 0
    nodes = []
    wall_h = []
    total_nodes = 0.0
    seen = set()

    for jid in job_ids:
        seen.add(jid)
        procs = procs_map.get(jid)
        if procs is None:
            continue
        procs = float(procs)
        total_nodes += procs
        nodes.append(procs)
        walltime = walltimes_map.get(jid)
        if walltime is not None:
            wall_h.append(float(walltime) / 3600.0)
        group = assign_group(procs)
        if group is None:
            other_count += 1
            continue
        group_counts[group] += 1
        group_nodes[group] += procs

    total_jobs = sum(group_counts.values()) + other_count
    if total_jobs <= 0:
        total_jobs = len(job_ids)

    job_pct = {
        g: (100.0 * group_counts[g] / total_jobs) if total_jobs > 0 else 0.0
        for g in GROUP_LABELS
    }
    node_pct = {
        g: (100.0 * group_nodes[g] / total_nodes) if total_nodes > 0 else 0.0
        for g in GROUP_LABELS
    }

    return {
        "job_occurrences": len(job_ids),
        "unique_jobs": len(seen),
        "mean_nodes": float(np.mean(nodes)) if nodes else 0.0,
        "median_nodes": float(np.median(nodes)) if nodes else 0.0,
        "mean_walltime_h": float(np.mean(wall_h)) if wall_h else 0.0,
        "median_walltime_h": float(np.median(wall_h)) if wall_h else 0.0,
        "group_counts": group_counts,
        "group_job_pct": job_pct,
        "group_node_pct": node_pct,
        "other_count": other_count,
        "total_nodes": total_nodes,
    }


def write_driver_drain_cycle_table(tag, results_dir, plots_dir,
                                   submit_map, start_map, procs_map, walltimes_map,
                                   time_start_limit=None, time_end_limit=None):
    rows = _load_mcts_cycle_rows(
        results_dir, tag,
        time_start_limit=time_start_limit,
        time_end_limit=time_end_limit,
    )
    if not rows:
        print(f"  [skip] {tag}: no MCTS rows for drain analysis")
        return None

    rows = sorted(rows, key=lambda row: row["cycle"])
    run_ids_by_time = _run_job_ids_by_time(results_dir, tag)
    drain_rows = [
        row for row in rows
        if row["decision_policy"] == "DRAIN" and row["root_branching"] >= 2
    ]
    if not drain_rows:
        print(f"  [skip] {tag}: no DRAIN cycles with root branching >= 2")
        return None

    aggregated = {
        "Runnable options during DRAIN": [],
        "Started on next launch cycle": [],
    }
    matched_launch_count = 0
    cycle_records = []

    path = os.path.join(plots_dir, "table_mars_cw_drain_cycles.csv")
    os.makedirs(plots_dir, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Drain_Row_Index",
            "Drain_Cycle",
            "Drain_Time",
            "Next_Run_Cycle",
            "Next_Run_Time",
            "Next_Run_Cycle_Offset",
            "Next_Run_Time_Offset",
            "Root_Branching_Factor",
            "MCTS_Iterations",
            "Runnable_Option_Jobs",
            "Option_S", "Option_M", "Option_L", "Option_XL",
            "Started_On_Next_Launch_Jobs",
            "Started_S", "Started_M", "Started_L", "Started_XL",
        ])

        for idx, drain_row in enumerate(drain_rows, start=1):
            option_ids = _parse_possible_job_sets(drain_row.get("possible_job_sets", ""))

            next_run_row = None
            for row in rows:
                if row["cycle"] <= drain_row["cycle"]:
                    continue
                if row["jobs_run"] > 0:
                    next_run_row = row
                    break

            started_ids = []
            next_run_cycle = ""
            next_run_time = ""
            next_run_cycle_offset = ""
            next_run_time_offset = ""
            if next_run_row is not None:
                matched_launch_count += 1
                next_run_cycle = next_run_row["cycle"]
                next_run_time = int(next_run_row["sim_time"])
                next_run_cycle_offset = next_run_row["cycle"] - drain_row["cycle"]
                next_run_time_offset = int(next_run_row["sim_time"] - drain_row["sim_time"])
                started_ids = run_ids_by_time.get(int(next_run_row["sim_time"]), [])

            aggregated["Runnable options during DRAIN"].extend(option_ids)
            aggregated["Started on next launch cycle"].extend(started_ids)

            option_summary = _summarize_job_collection(option_ids, procs_map, walltimes_map)
            started_summary = _summarize_job_collection(started_ids, procs_map, walltimes_map)
            cycle_records.append({
                "drain_cycle": drain_row["cycle"],
                "drain_time": int(drain_row["sim_time"]),
                "next_run_cycle": next_run_cycle,
                "next_run_time": next_run_time,
                "has_next_launch": next_run_row is not None,
                "root_branching": drain_row["root_branching"],
                "mcts_iterations": drain_row["mcts_iterations"],
                "option_counts": dict(option_summary["group_counts"]),
                "started_counts": dict(started_summary["group_counts"]),
            })

            writer.writerow([
                idx,
                drain_row["cycle"],
                int(drain_row["sim_time"]),
                next_run_cycle,
                next_run_time,
                next_run_cycle_offset,
                next_run_time_offset,
                drain_row["root_branching"],
                drain_row["mcts_iterations"],
                option_summary["job_occurrences"],
                option_summary["group_counts"]["S"],
                option_summary["group_counts"]["M"],
                option_summary["group_counts"]["L"],
                option_summary["group_counts"]["XL"],
                started_summary["job_occurrences"],
                started_summary["group_counts"]["S"],
                started_summary["group_counts"]["M"],
                started_summary["group_counts"]["L"],
                started_summary["group_counts"]["XL"],
            ])
    print(f"  Saved: {path}")

    return {
        "drain_cycle_count": len(drain_rows),
        "matched_launch_count": matched_launch_count,
        "cycle_records": cycle_records,
        "aggregated": {
            label: _summarize_job_collection(job_ids, procs_map, walltimes_map)
            for label, job_ids in aggregated.items()
        },
    }


def write_driver_drain_summary_table(summary, plots_dir):
    if not summary:
        return
    path = os.path.join(plots_dir, "table_mars_cw_drain_summary.csv")
    os.makedirs(plots_dir, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Category",
            "Drain_Cycles",
            "Matched_Launch_Cycles",
            "Job_Occurrences",
            "Unique_Jobs",
            "Mean_Nodes",
            "Median_Nodes",
            "Mean_Walltime_h",
            "Median_Walltime_h",
            "S_Count", "M_Count", "L_Count", "XL_Count",
            "S_JobPct", "M_JobPct", "L_JobPct", "XL_JobPct",
            "S_NodePct", "M_NodePct", "L_NodePct", "XL_NodePct",
        ])
        for label, stats in summary["aggregated"].items():
            writer.writerow([
                label,
                summary["drain_cycle_count"],
                summary["matched_launch_count"],
                stats["job_occurrences"],
                stats["unique_jobs"],
                f"{stats['mean_nodes']:.2f}",
                f"{stats['median_nodes']:.2f}",
                f"{stats['mean_walltime_h']:.2f}",
                f"{stats['median_walltime_h']:.2f}",
                stats["group_counts"]["S"],
                stats["group_counts"]["M"],
                stats["group_counts"]["L"],
                stats["group_counts"]["XL"],
                f"{stats['group_job_pct']['S']:.2f}",
                f"{stats['group_job_pct']['M']:.2f}",
                f"{stats['group_job_pct']['L']:.2f}",
                f"{stats['group_job_pct']['XL']:.2f}",
                f"{stats['group_node_pct']['S']:.2f}",
                f"{stats['group_node_pct']['M']:.2f}",
                f"{stats['group_node_pct']['L']:.2f}",
                f"{stats['group_node_pct']['XL']:.2f}",
            ])
    print(f"  Saved: {path}")


def plot_driver_drain_group_shares(summary, plots_dir, suffix, value_key, ylabel):
    if not summary:
        return

    categories = [
        ("Runnable options during DRAIN", "#d28b00"),
        ("Started on next launch cycle", "#4c78a8"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(GROUP_LABELS))
    width = 0.34

    for idx, (label, color) in enumerate(categories):
        stats = summary["aggregated"].get(label, {})
        vals = [stats.get(value_key, {}).get(group, 0.0) for group in GROUP_LABELS]
        offsets = x + (idx - 0.5) * width
        ax.bar(offsets, vals, width=width * 0.95, color=color, alpha=0.85,
               edgecolor='black', label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS, fontsize=TICK_SIZE)
    ax.set_xlabel("Job Size Group", fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=LEGEND_SIZE - 1, loc='upper right')
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, suffix))


def write_driver_drain_transition_probability_table(summary, plots_dir):
    if not summary:
        return None

    records = [row for row in summary.get("cycle_records", []) if row.get("has_next_launch")]
    if not records:
        print("  [skip] drain transition probabilities: no matched launch cycles")
        return None

    eligible_counts = {opt_g: 0 for opt_g in GROUP_LABELS}
    joint_counts = {
        opt_g: {start_g: 0 for start_g in GROUP_LABELS}
        for opt_g in GROUP_LABELS
    }

    for row in records:
        option_present = {
            g: row["option_counts"].get(g, 0) > 0
            for g in GROUP_LABELS
        }
        started_present = {
            g: row["started_counts"].get(g, 0) > 0
            for g in GROUP_LABELS
        }
        for opt_g in GROUP_LABELS:
            if not option_present[opt_g]:
                continue
            eligible_counts[opt_g] += 1
            for start_g in GROUP_LABELS:
                if started_present[start_g]:
                    joint_counts[opt_g][start_g] += 1

    path = os.path.join(plots_dir, "table_mars_cw_drain_transition_probabilities.csv")
    os.makedirs(plots_dir, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Runnable_Group",
            "Eligible_Drain_Cycles",
            "Started_S_Count", "Started_S_ProbPct",
            "Started_M_Count", "Started_M_ProbPct",
            "Started_L_Count", "Started_L_ProbPct",
            "Started_XL_Count", "Started_XL_ProbPct",
        ])
        for opt_g in GROUP_LABELS:
            denom = eligible_counts[opt_g]
            row = [opt_g, denom]
            for start_g in GROUP_LABELS:
                count = joint_counts[opt_g][start_g]
                prob = 100.0 * count / denom if denom > 0 else 0.0
                row.extend([count, f"{prob:.2f}"])
            writer.writerow(row)
    print(f"  Saved: {path}")

    matrix = np.array([
        [
            (100.0 * joint_counts[opt_g][start_g] / eligible_counts[opt_g])
            if eligible_counts[opt_g] > 0 else 0.0
            for start_g in GROUP_LABELS
        ]
        for opt_g in GROUP_LABELS
    ], dtype=float)

    return {
        "matrix_pct": matrix,
        "eligible_counts": eligible_counts,
        "joint_counts": joint_counts,
        "record_count": len(records),
    }


def plot_driver_drain_transition_probability_heatmap(transition_stats, plots_dir):
    if not transition_stats:
        return

    matrix = transition_stats["matrix_pct"]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    im = ax.imshow(matrix, cmap="YlOrBr", vmin=0.0, vmax=max(5.0, float(matrix.max())))

    ax.set_xticks(np.arange(len(GROUP_LABELS)))
    ax.set_xticklabels(GROUP_LABELS, fontsize=TICK_SIZE, fontweight='bold')
    ax.set_yticks(np.arange(len(GROUP_LABELS)))
    ax.set_yticklabels(GROUP_LABELS, fontsize=TICK_SIZE, fontweight='bold')
    ax.set_xlabel("Started Group On Next Launch Cycle", fontsize=FONT_SIZE)
    ax.set_ylabel("Runnable Group During DRAIN", fontsize=FONT_SIZE)
    ax.tick_params(length=0)

    for row_idx, opt_g in enumerate(GROUP_LABELS):
        denom = transition_stats["eligible_counts"][opt_g]
        for col_idx, start_g in enumerate(GROUP_LABELS):
            pct = matrix[row_idx, col_idx]
            num = transition_stats["joint_counts"][opt_g][start_g]
            text_color = 'white' if pct >= 55.0 else 'black'
            ax.text(
                col_idx, row_idx,
                f"{pct:.1f}%\n({num}/{denom})",
                ha='center', va='center',
                fontsize=10,
                color=text_color,
                fontweight='bold',
                linespacing=1.2,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability (%)", fontsize=FONT_SIZE - 1)
    cbar.ax.tick_params(labelsize=TICK_SIZE - 1)

    fig.tight_layout()
    save(fig, os.path.join(plots_dir, "heatmap_mars_cw_drain_transition_probabilities.png"))


def write_driver_drain_run_horizon_probability_table(
        tag, results_dir, plots_dir, procs_map,
        horizons=(10, 20, 30, 40, 50),
        time_start_limit=None, time_end_limit=None):
    rows = _load_mcts_cycle_rows(
        results_dir, tag,
        time_start_limit=time_start_limit,
        time_end_limit=time_end_limit,
    )
    if not rows:
        print(f"  [skip] {tag}: no MCTS rows for drain horizon analysis")
        return None

    drain_rows = [
        row for row in sorted(rows, key=lambda row: (row["sim_time"], row["cycle"]))
        if row["decision_policy"] == "DRAIN" and row["root_branching"] >= 2
    ]
    if not drain_rows:
        print(f"  [skip] {tag}: no DRAIN cycles with root branching >= 2 for horizon analysis")
        return None

    run_events = _ordered_run_events(
        results_dir, tag,
        time_start_limit=time_start_limit,
        time_end_limit=time_end_limit,
    )
    if not run_events:
        print(f"  [skip] {tag}: no Run events for drain horizon analysis")
        return None

    run_times = [sim_time for sim_time, _ in run_events]
    eligible_counts = {int(h): 0 for h in horizons}
    presence_counts = {
        int(h): {group: 0 for group in GROUP_LABELS}
        for h in horizons
    }
    mean_event_counts = {
        int(h): {group: [] for group in GROUP_LABELS}
        for h in horizons
    }

    for drain_row in drain_rows:
        start_idx = bisect_right(run_times, drain_row["sim_time"])
        for horizon in horizons:
            horizon = int(horizon)
            horizon_events = run_events[start_idx:start_idx + horizon]
            if len(horizon_events) < horizon:
                continue
            eligible_counts[horizon] += 1
            event_counts = {group: 0 for group in GROUP_LABELS}
            for _, job_id in horizon_events:
                group = assign_group(procs_map.get(job_id))
                if group is None:
                    continue
                event_counts[group] += 1
            for group in GROUP_LABELS:
                if event_counts[group] > 0:
                    presence_counts[horizon][group] += 1
                mean_event_counts[horizon][group].append(event_counts[group])

    if not any(eligible_counts.values()):
        print(f"  [skip] {tag}: no DRAIN cycles had full post-DRAIN run horizons available")
        return None

    slug = tag_slug(tag)
    path = os.path.join(plots_dir, f"table_{slug}_drain_run_horizon_probabilities.csv")
    os.makedirs(plots_dir, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Run_Horizon",
            "Eligible_Drain_Cycles",
            "Has_S_Count", "Has_S_ProbPct", "Mean_S_Runs_In_Horizon",
            "Has_M_Count", "Has_M_ProbPct", "Mean_M_Runs_In_Horizon",
            "Has_L_Count", "Has_L_ProbPct", "Mean_L_Runs_In_Horizon",
            "Has_XL_Count", "Has_XL_ProbPct", "Mean_XL_Runs_In_Horizon",
        ])
        for horizon in horizons:
            horizon = int(horizon)
            denom = eligible_counts[horizon]
            row = [horizon, denom]
            for group in GROUP_LABELS:
                count = presence_counts[horizon][group]
                prob = 100.0 * count / denom if denom > 0 else 0.0
                mean_runs = (
                    float(np.mean(mean_event_counts[horizon][group]))
                    if mean_event_counts[horizon][group] else 0.0
                )
                row.extend([count, f"{prob:.2f}", f"{mean_runs:.2f}"])
            writer.writerow(row)
    print(f"  Saved: {path}")

    matrix = np.array([
        [
            (100.0 * presence_counts[int(horizon)][group] / eligible_counts[int(horizon)])
            if eligible_counts[int(horizon)] > 0 else 0.0
            for group in GROUP_LABELS
        ]
        for horizon in horizons
    ], dtype=float)

    return {
        "tag": tag,
        "slug": slug,
        "horizons": [int(h) for h in horizons],
        "eligible_counts": eligible_counts,
        "presence_counts": presence_counts,
        "matrix_pct": matrix,
    }


def plot_driver_drain_run_horizon_probability_heatmap(stats, plots_dir):
    if not stats:
        return

    matrix = stats["matrix_pct"]
    vmax = max(5.0, float(matrix.max())) if matrix.size else 5.0
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=vmax, aspect='auto')

    ax.set_xticks(np.arange(len(GROUP_LABELS)))
    ax.set_xticklabels(GROUP_LABELS, fontsize=TICK_SIZE, fontweight='bold')
    ax.set_yticks(np.arange(len(stats["horizons"])))
    ax.set_yticklabels(
        [f"Next {h}" for h in stats["horizons"]],
        fontsize=TICK_SIZE,
        fontweight='bold',
    )
    ax.set_xlabel("Started Job Group Present Within Horizon", fontsize=FONT_SIZE)
    ax.set_ylabel("Run-Event Horizon After DRAIN Cycle", fontsize=FONT_SIZE)
    ax.tick_params(length=0)

    for row_idx, horizon in enumerate(stats["horizons"]):
        denom = stats["eligible_counts"][int(horizon)]
        for col_idx, group in enumerate(GROUP_LABELS):
            pct = matrix[row_idx, col_idx]
            num = stats["presence_counts"][int(horizon)][group]
            text_color = 'white' if pct >= 55.0 else 'black'
            ax.text(
                col_idx, row_idx,
                f"{pct:.1f}%\n({num}/{denom})",
                ha='center', va='center',
                fontsize=10,
                color=text_color,
                fontweight='bold',
                linespacing=1.2,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability (%)", fontsize=FONT_SIZE - 1)
    cbar.ax.tick_params(labelsize=TICK_SIZE - 1)

    fig.tight_layout()
    save(fig, os.path.join(
        plots_dir,
        f"heatmap_{stats['slug']}_drain_run_horizon_probabilities.png",
    ))


def write_driver_drain_run_sequence_table(
        tag, results_dir, plots_dir, procs_map,
        total_group_counts=None,
        max_lookahead=3,
        time_start_limit=None, time_end_limit=None):
    run_events = _ordered_run_events(
        results_dir, tag,
        time_start_limit=time_start_limit,
        time_end_limit=time_end_limit,
    )
    if not run_events:
        print(f"  [skip] {tag}: no Run events for drain run-sequence analysis")
        return None

    rows = _load_mcts_cycle_rows(
        results_dir, tag,
        time_start_limit=time_start_limit,
        time_end_limit=time_end_limit,
    )
    if not rows:
        print(f"  [skip] {tag}: no MCTS rows for drain run-sequence analysis")
        return None

    raw_drain_rows = [
        row for row in sorted(rows, key=lambda row: (row["sim_time"], row["cycle"]))
        if row["decision_policy"] == "DRAIN" and row["root_branching"] >= 2
    ]
    if not raw_drain_rows:
        print(f"  [skip] {tag}: no DRAIN cycles with root branching >= 2 for run-sequence analysis")
        return None

    drain_rows = _collapse_drain_rows_by_run_events(raw_drain_rows, run_events)

    drain_times = [row["sim_time"] for row in drain_rows]
    assignments = [[] for _ in drain_rows]
    for run_time, job_id in run_events:
        group = assign_group(procs_map.get(job_id))
        if group is None:
            continue
        drain_idx = bisect_right(drain_times, run_time - 1e-9) - 1
        if drain_idx < 0:
            continue
        if len(assignments[drain_idx]) >= max_lookahead:
            continue
        assignments[drain_idx].append((run_time, job_id, group))

    lookahead_counts = {
        idx: {group: 0 for group in GROUP_LABELS}
        for idx in range(1, max_lookahead + 1)
    }
    eligible_by_lookahead = {idx: 0 for idx in range(1, max_lookahead + 1)}
    detail_rows = []

    for drain_row, assigned_runs in zip(drain_rows, assignments):
        detail = {
            "drain_cycle": drain_row["cycle"],
            "drain_time": int(drain_row["sim_time"]),
            "root_branching": drain_row["root_branching"],
            "mcts_iterations": drain_row["mcts_iterations"],
            "runs": [],
        }
        for idx, (run_time, job_id, group) in enumerate(assigned_runs, start=1):
            lookahead_counts[idx][group] += 1
            eligible_by_lookahead[idx] += 1
            detail["runs"].append({
                "lookahead": idx,
                "run_time": int(run_time),
                "job_id": job_id,
                "group": group,
            })
        detail_rows.append(detail)

    slug = tag_slug(tag)
    detail_path = os.path.join(plots_dir, f"table_{slug}_drain_run_sequence_details.csv")
    os.makedirs(plots_dir, exist_ok=True)
    with open(detail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Drain_Event_Index",
            "Drain_Cycle", "Drain_Time", "Root_Branching_Factor", "MCTS_Iterations",
            "First_Run_Time", "First_Run_Job", "First_Run_Group",
            "Second_Run_Time", "Second_Run_Job", "Second_Run_Group",
            "Third_Run_Time", "Third_Run_Job", "Third_Run_Group",
        ])
        for event_idx, row in enumerate(detail_rows, start=1):
            flat = [
                event_idx,
                row["drain_cycle"],
                row["drain_time"],
                row["root_branching"],
                row["mcts_iterations"],
            ]
            runs = {item["lookahead"]: item for item in row["runs"]}
            for lookahead in range(1, max_lookahead + 1):
                item = runs.get(lookahead)
                if item is None:
                    flat.extend(["", "", ""])
                else:
                    flat.extend([item["run_time"], item["job_id"], item["group"]])
            writer.writerow(flat)
    print(f"  Saved: {detail_path}")

    summary_path = os.path.join(plots_dir, f"table_{slug}_drain_run_sequence_counts.csv")
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Lookahead_Run",
            "Eligible_Drain_Cycles",
            "S_Count", "S_Total", "S_PctOfAllS",
            "M_Count", "M_Total", "M_PctOfAllM",
            "L_Count", "L_Total", "L_PctOfAllL",
            "XL_Count", "XL_Total", "XL_PctOfAllXL",
        ])
        for lookahead in range(1, max_lookahead + 1):
            row = [
                lookahead,
                eligible_by_lookahead[lookahead],
            ]
            for group in GROUP_LABELS:
                total = (total_group_counts or {}).get(group, 0)
                count = lookahead_counts[lookahead][group]
                pct = 100.0 * count / total if total > 0 else 0.0
                row.extend([count, total, f"{pct:.4f}"])
            writer.writerow([
                *row,
            ])
    print(f"  Saved: {summary_path}")

    return {
        "tag": tag,
        "slug": slug,
        "max_lookahead": max_lookahead,
        "drain_event_count": len(drain_rows),
        "eligible_by_lookahead": eligible_by_lookahead,
        "lookahead_counts": lookahead_counts,
        "total_group_counts": dict(total_group_counts or {}),
    }


def plot_driver_drain_run_sequence_counts(stats, plots_dir):
    if not stats:
        return

    lookahead_labels = {
        1: "1st Run",
        2: "2nd Run",
        3: "3rd Run",
    }
    lookahead_colors = {
        1: "#4c78a8",
        2: "#f58518",
        3: "#54a24b",
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    x = np.arange(len(GROUP_LABELS))
    width = 0.24
    ymax = 0.0

    for idx, lookahead in enumerate(range(1, stats["max_lookahead"] + 1)):
        pct_values = [
            100.0 * stats["lookahead_counts"][lookahead][group]
            / max(1, stats.get("total_group_counts", {}).get(group, 0))
            for group in GROUP_LABELS
        ]
        ymax = max(ymax, max(pct_values, default=0.0))
        offsets = x + (idx - 1) * width
        bars = ax.bar(
            offsets,
            pct_values,
            width=width * 0.95,
            color=lookahead_colors.get(lookahead, "#777777"),
            alpha=0.88,
            edgecolor='black',
            label=lookahead_labels.get(lookahead, f"Run {lookahead}"),
        )
        for bar in bars:
            if bar.get_height() <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(0.25, 0.015 * max(1.0, bar.get_height())),
                f"{bar.get_height():.1f}%",
                ha='center',
                va='bottom',
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS, fontsize=TICK_SIZE, fontweight='bold')
    ax.set_xlabel("Job Size Group", fontsize=FONT_SIZE)
    ax.set_ylabel("Jobs Started Relative to Total Size-Group Jobs (%)", fontsize=FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.set_ylim(0.0, max(5.0, ymax * 1.18))
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=LEGEND_SIZE - 1, loc='upper left')
    fig.tight_layout()
    save(fig, os.path.join(
        plots_dir,
        f"bar_chart_{stats['slug']}_drain_run_sequence_counts.png",
    ))


def write_driver_drain_sequence_combos(tag, plots_dir):
    """Compute (G1, G2, G3) post-drain sequence frequencies from the existing
    run-sequence details CSV.  Returns a stats dict for plotting."""
    slug = tag_slug(tag)
    details_path = os.path.join(plots_dir, f"table_{slug}_drain_run_sequence_details.csv")
    if not os.path.exists(details_path):
        print(f"  [skip] drain sequence combos: {details_path} not found")
        return None

    seq_counts   = {}   # {G1-G2-G3: int}
    pair_12      = {}   # {G1: {G2: int}}
    pair_23      = {}   # {G2: {G3: int}}
    total        = 0    # drain episodes with at least G1
    complete     = 0    # drain episodes with all three groups

    with open(details_path) as f:
        for row in csv.DictReader(f):
            g1 = row.get('First_Run_Group',  '').strip()
            g2 = row.get('Second_Run_Group', '').strip()
            g3 = row.get('Third_Run_Group',  '').strip()
            if not g1:
                continue
            total += 1

            if g1 and g2:
                pair_12.setdefault(g1, {})
                pair_12[g1][g2] = pair_12[g1].get(g2, 0) + 1
            if g2 and g3:
                pair_23.setdefault(g2, {})
                pair_23[g2][g3] = pair_23[g2].get(g3, 0) + 1
            if g1 and g2 and g3:
                complete += 1
                key = f"{g1}-{g2}-{g3}"
                seq_counts[key] = seq_counts.get(key, 0) + 1

    if total == 0:
        print(f"  [skip] drain sequence combos: no valid rows in {details_path}")
        return None

    sorted_seqs = sorted(seq_counts.items(), key=lambda x: x[1], reverse=True)

    combo_path = os.path.join(plots_dir, f"table_{slug}_drain_sequence_combos.csv")
    os.makedirs(plots_dir, exist_ok=True)
    with open(combo_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Sequence', 'Count',
            'Pct_Of_All_Drain_Episodes',
            'Pct_Of_Complete_Drain_Episodes',
        ])
        for seq, count in sorted_seqs:
            writer.writerow([
                seq, count,
                f"{100.0 * count / total:.2f}",
                f"{100.0 * count / complete:.2f}" if complete else "0.00",
            ])
    print(f"  Saved: {combo_path}")

    return {
        'tag':                    tag,
        'slug':                   slug,
        'total_drain_episodes':   total,
        'complete_drain_episodes': complete,
        'seq_counts':             dict(sorted_seqs),
        'pair_12':                pair_12,
        'pair_23':                pair_23,
    }


def _build_wildcard_sequence_entries(seq_counts, max_entries=15):
    """Return sorted [(label, count)] using a two-level wildcard scheme.

    Fixed rule:
      G1 in {L, XL}  → always aggregate as  G1-*-*
      G1 in {S, M}   → start at G1-G2-* level

    Then iteratively expand the highest-count G1-G2-* wildcard into its four
    G1-G2-G3 specifics until the list reaches max_entries or nothing can expand.
    """
    g1_totals  = {}
    g1g2_totals = {}
    for seq, cnt in seq_counts.items():
        parts = seq.split('-')
        if len(parts) != 3:
            continue
        g1, g2, _ = parts
        g1_totals[g1]        = g1_totals.get(g1, 0) + cnt
        g1g2_totals[(g1, g2)] = g1g2_totals.get((g1, g2), 0) + cnt

    entries = {}

    # Level-1 wildcards for L and XL
    for g1 in ('L', 'XL'):
        if g1_totals.get(g1, 0) > 0:
            entries[f'{g1}-*-*'] = g1_totals[g1]

    # Level-2 wildcards for S and M
    for g1 in ('S', 'M'):
        for g2 in GROUP_LABELS:
            cnt = g1g2_totals.get((g1, g2), 0)
            if cnt > 0:
                entries[f'{g1}-{g2}-*'] = cnt

    # Expand most common G1-G2-* wildcards until we reach max_entries
    while len(entries) < max_entries:
        expandable = sorted(
            [(k, v) for k, v in entries.items()
             if k.endswith('-*') and not k.startswith('L-') and not k.startswith('XL-')],
            key=lambda x: x[1], reverse=True,
        )
        if not expandable:
            break
        label, _ = expandable[0]
        del entries[label]
        g1, g2 = label.split('-')[:2]
        added = 0
        for g3 in GROUP_LABELS:
            cnt = seq_counts.get(f'{g1}-{g2}-{g3}', 0)
            if cnt > 0:
                entries[f'{g1}-{g2}-{g3}'] = cnt
                added += 1
        if added == 0:
            break  # nothing to expand

    return sorted(entries.items(), key=lambda x: x[1], reverse=True)[:max_entries]


def _load_drain_pct_for_tag(results_dir, tag,
                             time_start_limit=None, time_end_limit=None):
    """Return (drain_episode_count, eligible_total, pct) matching the bar chart."""
    rows = _load_mcts_cycle_rows(
        results_dir, tag,
        time_start_limit=time_start_limit,
        time_end_limit=time_end_limit,
    )
    eligible = [r for r in rows if r['root_branching'] >= 2]
    drain_count = _count_drain_events(eligible)
    total = len(eligible)
    pct = 100.0 * drain_count / total if total else 0.0
    return drain_count, total, pct


def _draw_drain_sequence_bar(ax, entries, total, show_ylabels=True,
                              title='', bar_color='#505050', hatch=None):
    """Draw a horizontal bar chart of wildcard sequence entries."""
    seqs  = [s for s, _ in entries]
    pcts  = [100.0 * c / total for _, c in entries]
    y_pos = np.arange(len(seqs))

    ax.barh(y_pos, pcts, color=bar_color, edgecolor='black',
            linewidth=0.5, height=0.55, hatch=hatch)
    x_max = max(pcts) * 1.30 if pcts else 1.0
    for y, p in zip(y_pos, pcts):
        ax.text(p + x_max * 0.015, y, f'{p:.1f}%',
                va='center', ha='left', fontsize=7.5, fontweight='bold')

    ax.set_yticks(y_pos)
    if show_ylabels:
        ax.set_yticklabels(seqs, fontsize=8, fontfamily='monospace',
                           fontweight='bold')
    else:
        ax.set_yticklabels(['' for _ in seqs])
    ax.invert_yaxis()
    ax.set_xlim(0, x_max)
    ax.set_xlabel('% of drain episodes', fontsize=8, fontweight='bold')
    ax.set_title(title, fontsize=8.5, fontweight='bold', pad=4)
    ax.tick_params(axis='x', labelsize=7.5)
    ax.tick_params(axis='y', length=0)
    ax.grid(True, axis='x', alpha=0.3)


def plot_driver_drain_sequence_combos(stats, plots_dir):
    """Single horizontal bar chart with wildcard-aggregated sequences."""
    if not stats:
        return

    entries = _build_wildcard_sequence_entries(stats['seq_counts'], max_entries=5)
    total   = stats['total_drain_episodes']

    n = len(entries)
    fig, ax = plt.subplots(figsize=(5.5, max(3.0, n * 0.38 + 1.2)))
    fig.subplots_adjust(left=0.22, right=0.92, top=0.88, bottom=0.12)

    _draw_drain_sequence_bar(ax, entries, total, show_ylabels=True,
                              title=f"{stats['tag']} — Post-Drain Run Sequences\n"
                                    f"(N={total:,} drain episodes)")
    save(fig, os.path.join(plots_dir,
                           f"drain_sequence_combos_{stats['slug']}.png"))


def plot_driver_drain_sequence_combos_combined(cw_stats, cu_stats,
                                               cw_drain_pct_tuple,
                                               cu_drain_pct_tuple,
                                               plots_dir):
    """Single-column combined figure: MARS-CW (left) and MARS-CU (right).

    Panels side by side; each has its own y-axis labels (sequences).
    Bars ordered most-common → least-common (top to bottom).
    cw_drain_pct_tuple / cu_drain_pct_tuple : (drain_episodes, eligible, pct)
    from _load_drain_pct_for_tag.
    """
    if not cw_stats and not cu_stats:
        return

    cw_entries = _build_wildcard_sequence_entries(
        cw_stats['seq_counts'] if cw_stats else {}, max_entries=5)
    cu_entries = _build_wildcard_sequence_entries(
        cu_stats['seq_counts'] if cu_stats else {}, max_entries=5)

    cw_map = dict(cw_entries)
    cu_map = dict(cu_entries)
    cw_total = cw_stats['total_drain_episodes'] if cw_stats else 1
    cu_total = cu_stats['total_drain_episodes'] if cu_stats else 1

    # Each panel uses its own ordering (most → least) independently
    def _title(label, pct_tuple):
        if pct_tuple:
            drain_ep, eligible, pct = pct_tuple
            return f'{label}\nN={drain_ep:,} ({pct:.1f}%)'
        return label

    n = max(len(cw_entries), len(cu_entries))
    fig_h = max(2.2, n * 0.44 + 1.0)
    fig, (ax_cw, ax_cu) = plt.subplots(1, 2, figsize=(5.2, fig_h))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.88, bottom=0.14,
                        wspace=0.30)

    _draw_drain_sequence_bar(ax_cw, cw_entries, cw_total, show_ylabels=True,
                              title=_title('MARS-CW', cw_drain_pct_tuple),
                              bar_color='#add8e6')
    _draw_drain_sequence_bar(ax_cu, cu_entries, cu_total, show_ylabels=True,
                              title=_title('MARS-CU', cu_drain_pct_tuple),
                              bar_color='#add8e6')

    save(fig, os.path.join(plots_dir,
                           'drain_sequence_combos_combined.png'))


def _find_tag_by_prefix(all_tags, prefix):
    """Return the first tag that starts with prefix (case-insensitive), or None."""
    up = prefix.upper()
    return next((t for t in all_tags if t.upper().startswith(up)), None)




def _collect_postdrain_waits(drain_details_by_tag, all_wait_maps):
    """Return {tag: {group: set_of_jids}}, {tag: {group: [wait_h]}} for post-drain jobs."""
    postdrain_ids  = {}   # {tag: {grp: set}}
    postdrain_wait = {}   # {tag: {grp: [wait_h]}}
    for tag, csv_path in drain_details_by_tag.items():
        if not os.path.exists(csv_path):
            continue
        wm    = all_wait_maps.get(tag, {})
        ids   = {'L': set(), 'XL': set()}
        waits = {'L': [], 'XL': []}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                for pos in ('First', 'Second', 'Third'):
                    jid_str = row.get(f'{pos}_Run_Job', '').strip()
                    grp     = row.get(f'{pos}_Run_Group', '').strip()
                    if not jid_str or grp not in ids:
                        continue
                    try:
                        jid = int(jid_str)
                    except ValueError:
                        continue
                    if jid in ids[grp]:
                        continue   # deduplicate
                    ids[grp].add(jid)
                    ws = wm.get(jid)
                    if ws is not None:
                        waits[grp].append(ws / 3600.0)
        postdrain_ids[tag]  = ids
        postdrain_wait[tag] = waits
    return postdrain_ids, postdrain_wait


def plot_drain_lxl_split_violin(
        drain_details_by_tag, all_wait_maps, procs_map, driver_colors,
        drain_dir,
        compare_prefixes=('WFP', 'MARS-CW', 'MARS-CU')):
    """Two vertically stacked split-violin panels (top=L, bottom=XL).

    For each scheduler the violin is split:
      left  half = post-drain jobs  (red)
      right half = non-drain jobs   (dark gray)
    A thin box plot over all jobs sits in the centre.
    Common legend at the bottom; no figure title.
    Sized for a single-column paper layout (~3.5 in wide).
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        print("  [skip] split violin: scipy not available")
        return

    all_tags = list(all_wait_maps.keys())
    sel_tags = [t for pfx in compare_prefixes
                for t in (_find_tag_by_prefix(all_tags, pfx),)
                if t is not None]
    if not sel_tags:
        return

    postdrain_ids, _ = _collect_postdrain_waits(drain_details_by_tag, all_wait_maps)

    # Per-tag, per-group: drain / non-drain wait lists
    tag_grp = {}
    for tag in sel_tags:
        wm   = all_wait_maps.get(tag, {})
        pd_g = postdrain_ids.get(tag, {})
        by_g = {'L':  {'d': [], 'n': []},
                'XL': {'d': [], 'n': []}}
        for jid, ws in wm.items():
            g = assign_group(procs_map.get(jid))
            if g not in by_g:
                continue
            w_h = max(ws / 3600.0, 1 / 60)   # floor at 1 min for log scale
            key = 'd' if jid in pd_g.get(g, set()) else 'n'
            by_g[g][key].append(w_h)
        tag_grp[tag] = by_g

    _DRAIN_COLOR    = '#d94f4f'
    _NONDRAIN_COLOR = '#4a4a4a'
    VW = 0.36   # max half-width of each violin half

    def _half_violin(ax, xc, data, side, color):
        if len(data) < 5:
            return
        log_d = np.log10(np.asarray(data, dtype=float))
        try:
            kde = gaussian_kde(log_d, bw_method='silverman')
        except Exception:
            return
        y_log = np.linspace(log_d.min() - 0.2, log_d.max() + 0.2, 300)
        dens  = kde(y_log)
        dens  = dens / dens.max() * VW
        y_act = 10 ** y_log
        if side == 'left':
            ax.fill_betweenx(y_act, xc - dens, xc,
                             color=color, alpha=0.68, linewidth=0, zorder=2)
            ax.plot(xc - dens, y_act, color=color, lw=0.7, alpha=0.90, zorder=3)
        else:
            ax.fill_betweenx(y_act, xc, xc + dens,
                             color=color, alpha=0.68, linewidth=0, zorder=2)
            ax.plot(xc + dens, y_act, color=color, lw=0.7, alpha=0.90, zorder=3)

    pos = np.arange(1, len(sel_tags) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.2))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.96, bottom=0.14,
                        hspace=0.26)

    for ax, grp in zip(axes, ('L', 'XL')):
        box_data = []
        for xi, tag in zip(pos, sel_tags):
            d = tag_grp[tag][grp]
            _half_violin(ax, xi, d['d'], 'left',  _DRAIN_COLOR)
            _half_violin(ax, xi, d['n'], 'right', _NONDRAIN_COLOR)
            combined = np.array(d['d'] + d['n'])
            box_data.append(combined if len(combined) else np.array([np.nan]))

        # Thin centre box plot over all jobs
        ax.boxplot(
            box_data,
            positions=pos,
            widths=0.055,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            boxprops=dict(linewidth=1.0, edgecolor='black', facecolor='white'),
            whiskerprops=dict(linewidth=0.9, color='black'),
            capprops=dict(linewidth=0.9, color='black'),
            medianprops=dict(linewidth=1.8, color='black'),
            zorder=5,
        )

        ax.set_xticks(pos)
        for lbl in ax.set_xticklabels([display_tag(t) for t in sel_tags],
                                       fontsize=8):
            lbl.set_fontweight('bold')
        ax.set_ylabel('Wait (h)', fontsize=8, fontweight='bold')
        ax.set_title(f'{grp} Jobs', fontsize=8.5, fontweight='bold', pad=2)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
        ax.tick_params(axis='y', labelsize=7.5)
        ax.grid(True, axis='y', alpha=0.20, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(0.45, len(sel_tags) + 0.55)

    # Common legend at the bottom of the figure
    legend_handles = [
        matplotlib.patches.Patch(facecolor=_DRAIN_COLOR,    alpha=0.70,
                                  label='Post-drain'),
        matplotlib.patches.Patch(facecolor=_NONDRAIN_COLOR, alpha=0.70,
                                  label='Non-drain'),
        Line2D([0], [0], color='black', linewidth=1.6, label='Median'),
    ]
    fig.legend(handles=legend_handles, fontsize=7.5,
               loc='lower center', bbox_to_anchor=(0.5, 0.00),
               ncol=3, framealpha=0.90)

    save(fig, os.path.join(drain_dir, 'violin_wait_lxl.png'))


def plot_drain_overview_combined(
        drain_tags, drain_pct_by_tag, drain_details_by_tag,
        total_group_counts_by_tag, driver_colors, drain_dir):
    """Single-column side-by-side figure.

    Left panel  – Drain % bar chart (one bar per MCTS tag).
    Right panel – Post-drain job share by size class (S/M/L/XL),
                  grouped bars (one per MCTS tag) expressed as % of all
                  jobs in that class for the same scheduler.

    Colors and hatches follow driver_colors / driver_hatch.
    """
    if not drain_tags:
        return

    # ── Collect data ─────────────────────────────────────────────────────────
    # Left: drain % per tag
    left_pcts    = []
    left_counts  = []      # (drain_ep, eligible)
    for tag in drain_tags:
        tup = drain_pct_by_tag.get(tag)
        if tup:
            drain_ep, eligible, pct = tup
            left_pcts.append(pct)
            left_counts.append((drain_ep, eligible))
        else:
            left_pcts.append(0.0)
            left_counts.append((0, 0))

    # Right: post-drain job counts per group per tag
    postdrain_group = {tag: {g: 0 for g in GROUP_LABELS} for tag in drain_tags}
    for tag in drain_tags:
        csv_path = drain_details_by_tag.get(tag, '')
        if not os.path.exists(csv_path):
            continue
        seen = set()
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                for pos in ('First', 'Second', 'Third'):
                    jid_str = row.get(f'{pos}_Run_Job', '').strip()
                    grp     = row.get(f'{pos}_Run_Group', '').strip()
                    if not jid_str or grp not in postdrain_group[tag]:
                        continue
                    try:
                        jid = int(jid_str)
                    except ValueError:
                        continue
                    if (jid, grp) in seen:
                        continue
                    seen.add((jid, grp))
                    postdrain_group[tag][grp] += 1

    # Right: convert to % of total class jobs for each tag
    right_pcts = {tag: {} for tag in drain_tags}
    for tag in drain_tags:
        totals = total_group_counts_by_tag.get(tag, {})
        for g in GROUP_LABELS:
            total = totals.get(g, 0)
            cnt   = postdrain_group[tag][g]
            right_pcts[tag][g] = 100.0 * cnt / total if total else 0.0

    # ── Layout ───────────────────────────────────────────────────────────────
    n_tags   = len(drain_tags)
    n_groups = len(GROUP_LABELS)

    FS_TICK  = 8.5    # tick label font size
    FS_AXIS  = 9.0    # axis label font size
    FS_TITLE = 9.5    # panel title font size
    FS_BAR   = 8.0    # in-bar annotation font size

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(5.5, 3.4),
        gridspec_kw={'width_ratios': [1, 2.4]},
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.91, bottom=0.22,
                        wspace=0.32)

    def _bold_ticks(ax):
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
            lbl.set_color('black')
        ax.tick_params(labelsize=FS_TICK, colors='black',
                       width=1.2, length=4)

    # ── Left panel ───────────────────────────────────────────────────────────
    bar_w_l = 0.38
    offs_l  = np.linspace(-(n_tags - 1) / 2, (n_tags - 1) / 2, n_tags) * bar_w_l
    x_l     = 0.5 + offs_l
    for i, tag in enumerate(drain_tags):
        ax_l.bar(
            x_l[i], left_pcts[i], width=bar_w_l,
            color=driver_colors.get(tag, 'gray'), alpha=0.88,
            hatch=driver_hatch(tag), edgecolor='black', linewidth=0.8,
        )
    y_pad = max(left_pcts, default=1) * 0.02
    for i, (pct, (n_drain, n_total)) in enumerate(zip(left_pcts, left_counts)):
        ax_l.text(x_l[i], pct + y_pad,
                  f'{pct:.1f}%  ({n_drain:,}/{n_total:,})',
                  ha='left', va='bottom', rotation=40,
                  fontsize=FS_BAR - 0.5, fontweight='bold', color='black')

    ax_l.set_xticks(x_l)
    lbls = ax_l.set_xticklabels([display_tag(t) for t in drain_tags],
                                 rotation=30, ha='right')
    for lbl in lbls:
        lbl.set_fontweight('bold'); lbl.set_color('black')
    ax_l.set_xlim(x_l[0] - bar_w_l * 1.4, x_l[-1] + bar_w_l * 1.4)
    ax_l.set_ylabel('Drain Episodes (%)', fontsize=FS_AXIS, fontweight='bold',
                    color='black')
    ax_l.set_ylim(0, max(left_pcts, default=1) * 1.75)
    ax_l.grid(True, axis='y', alpha=0.25)
    ax_l.set_axisbelow(True)
    ax_l.set_title('Drain Rate', fontsize=FS_TITLE, fontweight='bold', pad=3)
    _bold_ticks(ax_l)

    # ── Right panel ──────────────────────────────────────────────────────────
    bar_w_r = 0.32
    x_r     = np.arange(n_groups, dtype=float)
    offsets = np.linspace(-(n_tags - 1) / 2, (n_tags - 1) / 2, n_tags) * bar_w_r

    for i, tag in enumerate(drain_tags):
        vals = [right_pcts[tag][g] for g in GROUP_LABELS]
        ax_r.bar(
            x_r + offsets[i], vals, width=bar_w_r,
            color=driver_colors.get(tag, 'gray'), alpha=0.88,
            hatch=driver_hatch(tag), edgecolor='black', linewidth=0.8,
            label=display_tag(tag),
        )

    global_max_r = max(
        (right_pcts[t][g] for t in drain_tags for g in GROUP_LABELS),
        default=1,
    )
    y_pad_r = global_max_r * 0.02
    for i, tag in enumerate(drain_tags):
        for j, g in enumerate(GROUP_LABELS):
            v = right_pcts[tag][g]
            ax_r.text(x_r[j] + offsets[i], v + y_pad_r,
                      f'{v:.1f}%',
                      ha='left', va='bottom', rotation=40,
                      fontsize=FS_BAR - 0.5, fontweight='bold', color='black')

    ax_r.set_xticks(x_r)
    lbls = ax_r.set_xticklabels(GROUP_LABELS, rotation=30, ha='right')
    for lbl in lbls:
        lbl.set_fontweight('bold'); lbl.set_color('black')
    ax_r.set_xlim(x_r[0] - bar_w_r * 1.6, x_r[-1] + bar_w_r * 1.6)
    ax_r.set_ylabel('Post-drain Jobs (%)', fontsize=FS_AXIS, fontweight='bold',
                    color='black')
    ax_r.set_ylim(0, global_max_r * 1.75)
    ax_r.grid(True, axis='y', alpha=0.25)
    ax_r.set_axisbelow(True)
    ax_r.set_title('Post-Drain Share by Class', fontsize=FS_TITLE,
                   fontweight='bold', pad=3)
    ax_r.legend(fontsize=8.0, loc='upper right', framealpha=0.90,
                edgecolor='black')
    _bold_ticks(ax_r)

    save(fig, os.path.join(drain_dir, 'drain_overview_combined.png'))


def write_driver_drain_analysis(tag, results_dir, plots_dir,
                                submit_map, start_map, procs_map, walltimes_map,
                                time_start_limit=None, time_end_limit=None):
    stale_episode_table = os.path.join(plots_dir, "table_mars_cw_drain_episodes.csv")
    if os.path.exists(stale_episode_table):
        os.remove(stale_episode_table)
        print(f"  Removed stale: {stale_episode_table}")

    summary = write_driver_drain_cycle_table(
        tag, results_dir, plots_dir,
        submit_map, start_map, procs_map, walltimes_map,
        time_start_limit=time_start_limit,
        time_end_limit=time_end_limit,
    )
    if not summary:
        return
    write_driver_drain_summary_table(summary, plots_dir)
    plot_driver_drain_group_shares(
        summary, plots_dir,
        "bar_chart_mars_cw_drain_job_share.png",
        "group_job_pct",
        "Job Share (%)",
    )
    plot_driver_drain_group_shares(
        summary, plots_dir,
        "bar_chart_mars_cw_drain_node_share.png",
        "group_node_pct",
        "Requested Node Share (%)",
    )
    transition_stats = write_driver_drain_transition_probability_table(summary, plots_dir)
    plot_driver_drain_transition_probability_heatmap(transition_stats, plots_dir)


# ── Global: wait KDE per group, one file per group ────────────────────────────

def _detect_exp_axis(valid_tags, tag_to_driver_cfg):
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


def plot_global_wait_kde_by_group(driver_tags, all_wait_by_group, driver_colors,
                                   plots_dir):
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

            ax.set_title(display_tag(tag), fontsize=FONT_SIZE - 3)
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


# ── Global: overall CDFs ──────────────────────────────────────────────────────

def plot_global_overall_wait_cdfs(driver_tags, all_wait, driver_colors, plots_dir):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for tag in driver_tags:
        _draw_cdf_multi(ax, all_wait.get(tag, []), driver_colors[tag], tag)
    apply_style(ax, xlabel="Wait Time (hours)", ylabel="CDF")
    ax.set_ylim(0, 1.05)
    ax.set_xscale('log')
    handles, lbls = ax.get_legend_handles_labels()
    handles += _pctile_legend_handles()
    lbls    += [f'P{p}' for p in PERCENTILES] + ['Mean']
    ax.legend(handles, lbls, fontsize=LEGEND_SIZE, loc='lower right')
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, "cdf_wait_cdf.png"))


def plot_global_overall_bsld_cdfs(driver_tags, all_bsld, driver_colors, plots_dir):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for tag in driver_tags:
        _draw_cdf_multi(ax, all_bsld.get(tag, []), driver_colors[tag], tag)
    apply_style(ax, xlabel="Bounded Slowdown (Walltime)", ylabel="CDF")
    ax.set_ylim(0, 1.05)
    handles, lbls = ax.get_legend_handles_labels()
    handles += _pctile_legend_handles()
    lbls    += [f'P{p}' for p in PERCENTILES] + ['Mean']
    ax.legend(handles, lbls, fontsize=LEGEND_SIZE, loc='lower right')
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, "cdf_bsld.png"))


# ── Global: per-group CDFs ────────────────────────────────────────────────────

def plot_global_group_wait_cdfs(driver_tags, all_wait_by_group, driver_colors,
                                plots_dir):
    for g in GROUP_LABELS:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        for tag in driver_tags:
            _draw_cdf_multi(ax, all_wait_by_group[tag].get(g, []), driver_colors[tag], tag)
        apply_style(ax, xlabel="Wait Time (hours)", ylabel="CDF")
        ax.set_ylim(0, 1.05)
        ax.set_xscale('log')
        handles, lbls = ax.get_legend_handles_labels()
        handles += _pctile_legend_handles()
        lbls    += [f'P{p}' for p in PERCENTILES] + ['Mean']
        ax.legend(handles, lbls, fontsize=LEGEND_SIZE, loc='lower right')
        fig.tight_layout()
        save(fig, os.path.join(plots_dir, f"cdf_wait_{g}.png"))


def plot_global_group_bsld_cdfs(driver_tags, all_bsld_by_group, driver_colors,
                                plots_dir):
    for g in GROUP_LABELS:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        for tag in driver_tags:
            _draw_cdf_multi(ax, all_bsld_by_group[tag].get(g, []), driver_colors[tag], tag)
        apply_style(ax, xlabel="Bounded Slowdown (Walltime)", ylabel="CDF")
        ax.set_ylim(0, 1.05)
        handles, lbls = ax.get_legend_handles_labels()
        handles += _pctile_legend_handles()
        lbls    += [f'P{p}' for p in PERCENTILES] + ['Mean']
        ax.legend(handles, lbls, fontsize=LEGEND_SIZE, loc='lower right')
        fig.tight_layout()
        save(fig, os.path.join(plots_dir, f"cdf_bsld_{g}.png"))


# ── Table helpers ─────────────────────────────────────────────────────────────

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
    d = os.path.dirname(path)
    os.makedirs(d if d else ".", exist_ok=True)
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


def write_global_overall_wait_table(driver_tags, all_wait, plots_dir):
    _write_stat_table(os.path.join(plots_dir, "table_wait.csv"),
                      driver_tags, all_wait, multiplier=60.0)


def write_global_overall_bsld_table(driver_tags, all_bsld, plots_dir):
    _write_stat_table(os.path.join(plots_dir, "table_bsld.csv"),
                      driver_tags, all_bsld, multiplier=1.0)


def write_global_wait_tables(driver_tags, all_wait_by_group, plots_dir):
    for g in GROUP_LABELS:
        _write_stat_table(
            os.path.join(plots_dir, f"table_wait_{g}.csv"),
            driver_tags,
            {tag: all_wait_by_group[tag].get(g, []) for tag in driver_tags},
            multiplier=60.0)


def write_global_bsld_tables(driver_tags, all_bsld_by_group, plots_dir):
    for g in GROUP_LABELS:
        _write_stat_table(
            os.path.join(plots_dir, f"table_bsld_{g}.csv"),
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


def _maint_overlap_seconds(maintenance, t_start, t_end):
    """Total seconds of maintenance windows that overlap [t_start, t_end]."""
    total = 0.0
    for mstart, mend, _ in (maintenance or []):
        ov_s = max(mstart, t_start)
        ov_e = min(mend,   t_end)
        if ov_e > ov_s:
            total += ov_e - ov_s
    return total


def write_global_util_table(driver_tags, end_maps, start_maps, submit_maps,
                            procs_map, plots_dir, window_start=None, window_end=None,
                            maintenance=None):
    """Utilization table excluding any scheduled or unscheduled downtime."""
    path = os.path.join(plots_dir, "table_util.csv")
    os.makedirs(plots_dir, exist_ok=True)

    if window_start is not None and window_end is not None:
        maint_s  = _maint_overlap_seconds(maintenance, window_start, window_end)
        uptime_s = (window_end - window_start) - maint_s
    else:
        uptime_s = None

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Driver", "Utilization_Percent",
                         "Core_Hours_Used", "Core_Hours_Unused", "Total_Core_Hours"])
        for tag in driver_tags:
            end    = end_maps.get(tag, {})
            start  = start_maps.get(tag, {})
            submit = submit_maps.get(tag, {})
            if not end or not submit:
                writer.writerow([tag, "N/A", "N/A", "N/A", "N/A"])
                continue

            if window_start is not None and window_end is not None:
                # Used core-seconds in the full window
                used_full = sum_used_core_seconds(start, end, procs_map,
                                                  window_start, window_end)
                # Subtract core-seconds consumed during maintenance (machine offline)
                used_maint = 0.0
                for mstart, mend, _ in (maintenance or []):
                    ov_s = max(mstart, window_start)
                    ov_e = min(mend,   window_end)
                    if ov_e > ov_s:
                        used_maint += sum_used_core_seconds(start, end, procs_map,
                                                            ov_s, ov_e)
                used_core_s = used_full - used_maint
                eff_uptime  = uptime_s
            else:
                used_core_s = sum(
                    (end[jid] - start[jid]) * procs_map.get(jid, 0)
                    for jid in end if jid in start
                )
                eff_uptime = max(end.values()) - min(submit.values())

            total_core_s  = eff_uptime * S_SYS
            util_pct      = 100.0 * used_core_s / total_core_s if total_core_s > 0 else 0.0
            used_core_h   = used_core_s  / 3600.0
            total_core_h  = total_core_s / 3600.0
            unused_core_h = max(0.0, total_core_s - used_core_s) / 3600.0
            writer.writerow([tag,
                             f"{util_pct:.2f}",
                             f"{used_core_h:.2f}",
                             f"{unused_core_h:.2f}",
                             f"{total_core_h:.2f}"])
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

    event_order = []
    event_data  = {}
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
                u       = _compute_window_util(s_map, e_map, procs_map, ann_start, ann_end, S_SYS)
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


def write_drain_util_table(driver_tags, end_maps, start_maps, output_dir,
                           procs_map, plots_dir, tz, drain_days,
                           window_start=None, window_end=None):
    """Utilization during the N-day draining window before each scheduled maintenance."""
    date_fmt = "%Y-%m-%d %H:%M:%S"
    drain_s  = drain_days * 24 * 3600
    fname    = f"table_util_drain_{drain_days}day.csv"
    path     = os.path.join(plots_dir, fname)
    os.makedirs(plots_dir, exist_ok=True)

    # Collect scheduled maintenance windows (deduplicated across drivers)
    sched_windows = {}
    tags_seen     = []

    for tag in driver_tags:
        events_path = os.path.join(output_dir, tag, "events.csv")
        if not os.path.exists(events_path):
            continue
        tags_seen.append(tag)
        _, sms, sme, _, _ = _parse_maint_raw(events_path)
        for idx in sorted(sms):
            if idx not in sme:
                continue
            mstart, mend = sms[idx], sme[idx]
            if window_start is not None and mend   <= window_start:
                continue
            if window_end   is not None and mstart >= window_end:
                continue
            sched_windows[(mstart, mend)] = True

    if not sched_windows:
        print(f"  [skip] {fname}: no scheduled maintenance windows")
        return None

    drain_periods = []
    for (mstart, mend) in sorted(sched_windows):
        d_start = mstart - drain_s
        d_end   = mstart
        if window_start is not None:
            d_start = max(d_start, window_start)
        if window_end is not None:
            d_end = min(d_end, window_end)
        if d_end <= d_start:
            continue
        drain_periods.append((d_start, d_end))

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Draining_Period_Start", "Draining_Period_End"] + tags_seen)
        for d_start, d_end in drain_periods:
            row = [
                datetime.fromtimestamp(d_start, tz=tz).strftime(date_fmt),
                datetime.fromtimestamp(d_end,   tz=tz).strftime(date_fmt),
            ]
            for tag in tags_seen:
                u = _compute_window_util(
                    start_maps.get(tag, {}), end_maps.get(tag, {}),
                    procs_map, d_start, d_end, S_SYS)
                row.append(f"{u * 100:.2f}%")
            writer.writerow(row)
    print(f"  Saved: {path}")
    return path


def _parse_percent_cell(value):
    try:
        return float(str(value).strip().rstrip('%'))
    except (TypeError, ValueError):
        return None


def _load_drain_periods_from_table(table_path, tz=CST):
    if not table_path or not os.path.exists(table_path):
        return []

    periods = []
    with open(table_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_str = row.get("Draining_Period_Start", "")
            end_str = row.get("Draining_Period_End", "")
            try:
                start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
                end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
            except ValueError:
                continue
            periods.append((start_dt.timestamp(), end_dt.timestamp()))
    return periods


def _load_drain_period_labels_from_table(table_path):
    if not table_path or not os.path.exists(table_path):
        return []

    labels = []
    with open(table_path) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            start_str = row.get("Draining_Period_Start", "")
            end_str = row.get("Draining_Period_End", "")
            try:
                start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
                end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
                label = start_dt.strftime("%m/%d") + "\nto\n" + end_dt.strftime("%m/%d")
            except ValueError:
                label = f"W{idx}"
            labels.append(label)
    return labels


def _load_driver_columns_from_table(table_path, excluded_tags=None):
    if excluded_tags is None:
        excluded_tags = set()
    excluded_upper = {tag.upper() for tag in excluded_tags}

    with open(table_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    return [
        tag for tag in fieldnames[2:]
        if tag.upper() not in excluded_upper
    ]


def _load_queue_len_samples(results_dir, tag, periods):
    perf_path = os.path.join(results_dir, tag, "performance.csv")
    if not os.path.exists(perf_path) or not periods:
        return []

    periods = sorted(periods)
    samples = []
    period_idx = 0

    with open(perf_path) as f:
        reader = csv.DictReader(f)
        if "queue_len" not in (reader.fieldnames or []):
            return []
        for row in reader:
            try:
                sim_time = float(row["sim_time"])
                queue_len = float(row["queue_len"])
            except (ValueError, KeyError, TypeError):
                continue

            while period_idx < len(periods) and sim_time >= periods[period_idx][1]:
                period_idx += 1
            if period_idx >= len(periods):
                break
            if periods[period_idx][0] <= sim_time < periods[period_idx][1]:
                samples.append(queue_len)

    return samples


def _load_queue_len_samples_by_period(results_dir, tag, periods):
    perf_path = os.path.join(results_dir, tag, "performance.csv")
    if not os.path.exists(perf_path) or not periods:
        return [[] for _ in periods]

    periods = sorted(periods)
    samples = [[] for _ in periods]
    period_idx = 0

    with open(perf_path) as f:
        reader = csv.DictReader(f)
        if "queue_len" not in (reader.fieldnames or []):
            return samples
        for row in reader:
            try:
                sim_time = float(row["sim_time"])
                queue_len = float(row["queue_len"])
            except (ValueError, KeyError, TypeError):
                continue

            while period_idx < len(periods) and sim_time >= periods[period_idx][1]:
                period_idx += 1
            if period_idx >= len(periods):
                break
            if periods[period_idx][0] <= sim_time < periods[period_idx][1]:
                samples[period_idx].append(queue_len)

    return samples


_TRACE_MAPS_CACHE = {}


def _infer_trace_maps_for_results_dir(results_dir):
    key = os.path.abspath(os.path.normpath(results_dir))
    cached = _TRACE_MAPS_CACHE.get(key)
    if cached is not None:
        return cached

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.basename(key.rstrip(os.sep))
    exp_json = os.path.join(repo_root, "experiments", f"{base}.json")
    if not os.path.exists(exp_json):
        _TRACE_MAPS_CACHE[key] = ({}, {})
        return _TRACE_MAPS_CACHE[key]

    try:
        with open(exp_json) as f:
            cfg = json.load(f)
        swf_path = cfg.get("swf_path")
        if not swf_path or not os.path.exists(swf_path):
            _TRACE_MAPS_CACHE[key] = ({}, {})
            return _TRACE_MAPS_CACHE[key]
        procs_map, walltimes_map, _, _ = parse_swf(swf_path)
        _TRACE_MAPS_CACHE[key] = (procs_map, walltimes_map)
    except (OSError, ValueError, json.JSONDecodeError):
        _TRACE_MAPS_CACHE[key] = ({}, {})
    return _TRACE_MAPS_CACHE[key]


def _infer_walltimes_map_for_results_dir(results_dir):
    return _infer_trace_maps_for_results_dir(results_dir)[1]


def _infer_procs_map_for_results_dir(results_dir):
    return _infer_trace_maps_for_results_dir(results_dir)[0]


def _load_period_sample_times(perf_path, periods):
    if not os.path.exists(perf_path) or not periods:
        return [[] for _ in periods]

    periods = sorted(periods)
    samples = [[] for _ in periods]
    period_idx = 0

    with open(perf_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sim_time = float(row["sim_time"])
            except (ValueError, KeyError, TypeError):
                continue

            while period_idx < len(periods) and sim_time >= periods[period_idx][1]:
                period_idx += 1
            if period_idx >= len(periods):
                break
            if periods[period_idx][0] <= sim_time < periods[period_idx][1]:
                samples[period_idx].append(sim_time)

    return samples


def _build_announcement_queue_period_analysis(submit_map, start_map, walltimes_map,
                                              announce_time, maint_start_time):
    queue_before = 0
    submitted_total = 0
    submitted_fittable = 0
    submitted_unfittable = 0
    jobs_run = 0
    intervals = []
    inf = float("inf")

    for jid, submit_time in submit_map.items():
        start_time = start_map.get(jid, inf)
        walltime = walltimes_map.get(jid)

        if announce_time <= start_time < maint_start_time:
            jobs_run += 1

        if submit_time < announce_time:
            if start_time >= announce_time:
                queue_before += 1
                interval_end = min(start_time, maint_start_time)
                if interval_end > announce_time:
                    intervals.append((announce_time, interval_end))
            continue

        if not (announce_time <= submit_time < maint_start_time):
            continue

        submitted_total += 1
        if walltime is None:
            submitted_unfittable += 1
            continue

        if submit_time + walltime <= maint_start_time:
            submitted_fittable += 1
            interval_end = min(start_time, maint_start_time)
            if interval_end > submit_time:
                intervals.append((submit_time, interval_end))
        else:
            submitted_unfittable += 1

    return {
        "queue_before_announcement": queue_before,
        "submitted_after_announcement_total": submitted_total,
        "submitted_fittable_after_announcement": submitted_fittable,
        "submitted_unfittable_after_announcement": submitted_unfittable,
        "jobs_run_between_announcement_and_start": jobs_run,
        "eligible_queue_total": queue_before + submitted_fittable,
        "intervals": intervals,
    }


def _sample_counts_from_intervals(sample_times, intervals):
    if not sample_times or not intervals:
        return [0.0 for _ in sample_times]

    starts = sorted(start for start, end in intervals if end > start)
    ends = sorted(end for start, end in intervals if end > start)
    active = 0
    start_idx = 0
    end_idx = 0
    samples = []

    for sample_time in sample_times:
        while start_idx < len(starts) and starts[start_idx] <= sample_time:
            active += 1
            start_idx += 1
        while end_idx < len(ends) and ends[end_idx] <= sample_time:
            active -= 1
            end_idx += 1
        samples.append(float(max(active, 0)))
    return samples


def _load_announcement_queue_samples_by_period(results_dir, tag, periods, walltimes_map):
    if not periods:
        return [[] for _ in periods], []

    events_path = os.path.join(results_dir, tag, "events.csv")
    perf_path = os.path.join(results_dir, tag, "performance.csv")
    if not os.path.exists(events_path) or not os.path.exists(perf_path):
        return [[] for _ in periods], []

    submit_map, start_map, _ = parse_events(events_path)
    sample_times_by_period = _load_period_sample_times(perf_path, periods)
    samples_by_period = []
    analyses = []

    for (announce_time, maint_start_time), sample_times in zip(periods, sample_times_by_period):
        analysis = _build_announcement_queue_period_analysis(
            submit_map, start_map, walltimes_map,
            announce_time, maint_start_time,
        )
        analyses.append(analysis)
        samples_by_period.append(
            _sample_counts_from_intervals(sample_times, analysis["intervals"])
        )

    return samples_by_period, analyses


def _collect_drain_queue_samples_from_table(table_path, results_dir, excluded_tags=None):
    if not table_path or not os.path.exists(table_path):
        return [], {}

    periods = _load_drain_periods_from_table(table_path)
    driver_tags = _load_driver_columns_from_table(table_path, excluded_tags=excluded_tags)
    sample_map = {
        tag: _load_queue_len_samples(results_dir, tag, periods)
        for tag in driver_tags
    }
    ordered_tags = [tag for tag in driver_tags if sample_map.get(tag)]
    return ordered_tags, sample_map


def _collect_drain_queue_samples_by_period_from_table(table_path, results_dir,
                                                      excluded_tags=None,
                                                      walltimes_map=None):
    if not table_path or not os.path.exists(table_path):
        return [], [], {}

    periods = _load_drain_periods_from_table(table_path)
    labels = _load_drain_period_labels_from_table(table_path)
    driver_tags = _load_driver_columns_from_table(table_path, excluded_tags=excluded_tags)
    if walltimes_map is None:
        walltimes_map = _infer_walltimes_map_for_results_dir(results_dir)
    sample_map = {
        tag: _load_announcement_queue_samples_by_period(results_dir, tag, periods, walltimes_map)[0]
        for tag in driver_tags
    }
    ordered_tags = [
        tag for tag in driver_tags
        if any(len(period_samples) > 0 for period_samples in sample_map.get(tag, []))
    ]
    return ordered_tags, labels, sample_map


def write_drain_queue_announcement_table(table_path, results_dir, plots_dir,
                                         excluded_tags=None, walltimes_map=None, tz=CST,
                                         out_fname=None):
    _default_fname = out_fname or "table_queue_drain_2day.csv"
    if not table_path or not os.path.exists(table_path):
        print(f"  [skip] {_default_fname}: drain util table not found")
        return None

    periods = _load_drain_periods_from_table(table_path, tz=tz)
    driver_tags = _load_driver_columns_from_table(table_path, excluded_tags=excluded_tags)
    if walltimes_map is None:
        walltimes_map = _infer_walltimes_map_for_results_dir(results_dir)

    if not periods or not driver_tags:
        print(f"  [skip] {_default_fname}: no drain periods or drivers")
        return None

    path = os.path.join(plots_dir, _default_fname)
    os.makedirs(plots_dir, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Scheduler",
            "Draining_Period_Start",
            "Draining_Period_End",
            "Queue_Before_Announcement",
            "Submitted_After_Announcement_Total",
            "Submitted_Fittable_After_Announcement",
            "Submitted_Unfittable_After_Announcement",
            "Eligible_Queue_Total",
            "Jobs_Run_Between_Announcement_And_Start",
        ])
        for tag in driver_tags:
            _, analyses = _load_announcement_queue_samples_by_period(
                results_dir, tag, periods, walltimes_map)
            if not analyses:
                continue
            for (announce_time, maint_start_time), analysis in zip(periods, analyses):
                writer.writerow([
                    tag,
                    datetime.fromtimestamp(announce_time, tz=tz).strftime("%Y-%m-%d %H:%M:%S"),
                    datetime.fromtimestamp(maint_start_time, tz=tz).strftime("%Y-%m-%d %H:%M:%S"),
                    analysis["queue_before_announcement"],
                    analysis["submitted_after_announcement_total"],
                    analysis["submitted_fittable_after_announcement"],
                    analysis["submitted_unfittable_after_announcement"],
                    analysis["eligible_queue_total"],
                    analysis["jobs_run_between_announcement_and_start"],
                ])
    print(f"  Saved: {path}")
    return path


def write_drain_core_hours_table(table_path, results_dir, plots_dir,
                                 excluded_tags=None, procs_map=None,
                                 walltimes_map=None, tz=CST, out_fname=None):
    _default_fname = out_fname or "table_core_hours_drain_2day.csv"
    if not table_path or not os.path.exists(table_path):
        print(f"  [skip] {_default_fname}: drain util table not found")
        return None

    periods = _load_drain_periods_from_table(table_path, tz=tz)
    driver_tags = _load_driver_columns_from_table(table_path, excluded_tags=excluded_tags)
    if procs_map is None:
        procs_map = _infer_procs_map_for_results_dir(results_dir)
    if walltimes_map is None:
        walltimes_map = _infer_walltimes_map_for_results_dir(results_dir)

    if not periods or not driver_tags:
        print(f"  [skip] {_default_fname}: no drain periods or drivers")
        return None

    path = os.path.join(plots_dir, _default_fname)
    os.makedirs(plots_dir, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Scheduler",
            "Draining_Period_Start",
            "Draining_Period_End",
            "Submitted_Core_Hours",
            "Scheduled_Core_Hours",
        ])
        for tag in driver_tags:
            events_path = os.path.join(results_dir, tag, "events.csv")
            if not os.path.exists(events_path):
                continue
            submit_map, start_map, _ = parse_events(events_path)
            for d_start, d_end in periods:
                submitted_core_h = 0.0
                scheduled_core_h = 0.0
                for jid, submit_time in submit_map.items():
                    if d_start <= submit_time < d_end:
                        procs = procs_map.get(jid)
                        walltime = walltimes_map.get(jid)
                        if procs is None or walltime is None:
                            continue
                        submitted_core_h += procs * walltime / 3600.0
                for jid, start_time in start_map.items():
                    if d_start <= start_time < d_end:
                        procs = procs_map.get(jid)
                        walltime = walltimes_map.get(jid)
                        if procs is None or walltime is None:
                            continue
                        scheduled_core_h += procs * walltime / 3600.0
                writer.writerow([
                    tag,
                    datetime.fromtimestamp(d_start, tz=tz).strftime("%Y-%m-%d %H:%M:%S"),
                    datetime.fromtimestamp(d_end, tz=tz).strftime("%Y-%m-%d %H:%M:%S"),
                    f"{submitted_core_h:.2f}",
                    f"{scheduled_core_h:.2f}",
                ])
    print(f"  Saved: {path}")
    return path


def _queue_plot_legend_handles(ordered_tags, driver_colors):
    handles = []
    labels = []
    for tag in ordered_tags:
        patch = matplotlib.patches.Patch(
            facecolor=driver_colors.get(tag, 'gray'),
            edgecolor='black',
            hatch=driver_hatch(tag),
            label=display_tag(tag),
            alpha=0.82,
        )
        handles.append(patch)
        labels.append(display_tag(tag))
    return handles, labels


def _period_x_positions(n_windows, step=1.03):
    return np.arange(n_windows) * step


def _bold_log10_tick_formatter(value, _):
    if value not in (1, 10, 100, 1000):
        return ""
    exp = int(np.log10(value))
    return rf"$\mathbf{{10^{{{exp}}}}}$"


def _queue_before_ylim(top_y):
    return max(5.0, top_y * 1.15 + 2.0)


def _draw_period_backgrounds(ax, x_positions, group_width, highlight_indices=None,
                             shaded_color='#e0e0e0', shaded_alpha=0.95,
                             border_linewidth=2.1,
                             highlight_facecolor=None, highlight_alpha=0.20,
                             highlight_edgecolor='black'):
    if highlight_indices is None:
        highlight_indices = set()
    else:
        highlight_indices = set(highlight_indices)

    for i, xc in enumerate(x_positions):
        x_lo = xc - group_width * 0.57
        x_hi = xc + group_width * 0.57
        if i % 2 == 0:
            ax.axvspan(
                x_lo,
                x_hi,
                color=shaded_color,
                alpha=shaded_alpha,
                zorder=-2,
            )
        if i in highlight_indices:
            if highlight_facecolor is not None:
                ax.axvspan(
                    x_lo,
                    x_hi,
                    color=highlight_facecolor,
                    alpha=highlight_alpha,
                    zorder=-1,
                )
            if highlight_edgecolor is not None:
                ax.axvspan(
                    x_lo,
                    x_hi,
                    facecolor='none',
                    edgecolor=highlight_edgecolor,
                    linewidth=border_linewidth,
                    zorder=4,
                )


def _draw_drain_queue_boxplot(ax, ordered_tags, sample_map, driver_colors,
                              show_ylabel=True, tick_font=10):
    data = [
        np.clip(np.asarray(sample_map[tag], dtype=float), 1.0, 1000.0)
        for tag in ordered_tags
    ]
    positions = np.arange(1, len(ordered_tags) + 1)
    parts = ax.boxplot(
        data,
        positions=positions,
        widths=0.68,
        whis=(5, 95),
        showmeans=False,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(linewidth=1.8, edgecolor='black'),
        whiskerprops=dict(linewidth=1.1, color='black'),
        capprops=dict(linewidth=1.1, color='black'),
        medianprops=dict(linewidth=2.4, color='black'),
    )

    for box, tag in zip(parts.get('boxes', []), ordered_tags):
        box.set_facecolor(driver_colors.get(tag, 'gray'))
        box.set_alpha(0.82)
        hatch = driver_hatch(tag)
        if hatch:
            box.set_hatch(hatch)

    ax.set_xticks(positions)
    labels = ax.set_xticklabels(
        [display_tag(tag) for tag in ordered_tags],
        fontsize=tick_font,
        rotation=25,
        ha='right',
    )
    for lbl in labels:
        lbl.set_fontweight('bold')
    ax.tick_params(axis='y', labelsize=tick_font)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')
    if show_ylabel:
        ax.set_ylabel("Queue Size", fontsize=FONT_SIZE, fontweight='bold')
    ax.set_xlabel("Scheduler", fontsize=FONT_SIZE, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(1.0, 1000.0)
    ax.set_yticks([1, 10, 100, 1000])
    ax.yaxis.set_major_formatter(FuncFormatter(_bold_log10_tick_formatter))
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)


def _draw_grouped_drain_queue_boxplots(ax, ordered_tags, window_labels,
                                       sample_map_by_tag, driver_colors,
                                       show_ylabel=True, tick_font=10,
                                       show_xticklabels=True, show_xlabel=True,
                                       x_step=1.03, group_width=0.98,
                                       highlight_indices=None,
                                       background_kwargs=None):
    n_windows = len(window_labels)
    n_tags = len(ordered_tags)
    x = _period_x_positions(n_windows, step=x_step)
    width = group_width / max(n_tags, 1)

    if background_kwargs is None:
        background_kwargs = {}
    _draw_period_backgrounds(
        ax, x, group_width,
        highlight_indices=highlight_indices,
        **background_kwargs,
    )

    data = []
    positions = []
    tags_for_boxes = []
    for window_idx in range(n_windows):
        for tag_idx, tag in enumerate(ordered_tags):
            samples = sample_map_by_tag.get(tag, [])
            if window_idx >= len(samples):
                continue
            period_samples = samples[window_idx]
            if not period_samples:
                continue
            data.append(np.clip(np.asarray(period_samples, dtype=float), 1.0, 1000.0))
            positions.append(x[window_idx] + (tag_idx - (n_tags - 1) / 2.0) * width)
            tags_for_boxes.append(tag)

    if not data:
        return [], []

    parts = ax.boxplot(
        data,
        positions=positions,
        widths=width * 0.88,
        whis=(5, 95),
        showmeans=False,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
        boxprops=dict(linewidth=1.7, edgecolor='black'),
        whiskerprops=dict(linewidth=1.0, color='black'),
        capprops=dict(linewidth=1.0, color='black'),
        medianprops=dict(linewidth=2.2, color='black'),
    )

    for box, tag in zip(parts.get('boxes', []), tags_for_boxes):
        box.set_facecolor(driver_colors.get(tag, 'gray'))
        box.set_alpha(0.82)
        hatch = driver_hatch(tag)
        if hatch:
            box.set_hatch(hatch)

    ax.set_xticks(x)
    if show_xticklabels:
        labels = ax.set_xticklabels(window_labels, fontsize=tick_font - 1)
        for lbl in labels:
            lbl.set_fontweight('bold')
    else:
        ax.tick_params(axis='x', labelbottom=False, bottom=False)
    if show_xlabel:
        ax.set_xlabel("Drain Period", fontsize=FONT_SIZE, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel("Queue Size", fontsize=FONT_SIZE, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(1.0, 1000.0)
    ax.set_yticks([1, 10, 100, 1000])
    ax.yaxis.set_major_formatter(FuncFormatter(_bold_log10_tick_formatter))
    ax.margins(x=0.005)
    ax.set_xlim(x[0] - group_width * 0.54, x[-1] + group_width * 0.54)
    ax.tick_params(axis='y', labelsize=tick_font)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    return _queue_plot_legend_handles(ordered_tags, driver_colors)


def plot_drain_queue_boxplots_from_table(table_path, results_dir, driver_colors,
                                         plots_dir, fname, excluded_tags=None,
                                         walltimes_map=None):
    ordered_tags, window_labels, sample_map = _collect_drain_queue_samples_by_period_from_table(
        table_path, results_dir, excluded_tags=excluded_tags, walltimes_map=walltimes_map)
    if not ordered_tags or not window_labels:
        print(f"  [skip] {fname}: no queue samples to plot")
        return

    fig_w = max(8.5, 0.62 * len(window_labels) + 3.8)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    handles, labels = _draw_grouped_drain_queue_boxplots(
        ax, ordered_tags, window_labels, sample_map, driver_colors)
    _add_centered_two_row_legend(
        fig, handles, labels, fontsize=LEGEND_SIZE - 1,
        y_top=0.985, row_gap=0.046,
        columnspacing=0.85, handletextpad=0.35,
    )
    fig.subplots_adjust(top=0.82, bottom=0.25, left=0.08, right=0.995)
    save(fig, os.path.join(plots_dir, fname))


def plot_drain_util_bars_from_table(table_path, driver_colors, plots_dir, fname,
                                    excluded_tags=None):
    if not table_path or not os.path.exists(table_path):
        print(f"  [skip] {fname}: table not found")
        return

    if excluded_tags is None:
        excluded_tags = set()
    excluded_upper = {tag.upper() for tag in excluded_tags}

    with open(table_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        driver_tags = [
            tag for tag in fieldnames[2:]
            if tag.upper() not in excluded_upper
        ]

        window_labels = []
        values_by_window = []
        for idx, row in enumerate(reader, start=1):
            vals = []
            for tag in driver_tags:
                vals.append(_parse_percent_cell(row.get(tag)))
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

            window_labels.append(label)
            values_by_window.append(vals)

    if not driver_tags or not values_by_window:
        print(f"  [skip] {fname}: no data to plot")
        return

    n_windows = len(values_by_window)
    n_tags = len(driver_tags)
    x = np.arange(n_windows) * 1.03
    group_width = 0.98
    width = group_width / max(n_tags, 1)

    fig_w = max(8.5, 0.62 * n_windows + 3.8)
    fig, ax = plt.subplots(figsize=(fig_w, 4.9))

    for i, xc in enumerate(x):
        if i % 2 != 0:
            continue
        ax.axvspan(
            xc - group_width * 0.57,
            xc + group_width * 0.57,
            color='#e0e0e0',
            alpha=0.95,
            zorder=-2,
        )

    handles = []
    labels = []
    for idx, tag in enumerate(driver_tags):
        heights = [row[idx] if row[idx] is not None else 0.0 for row in values_by_window]
        offsets = x + (idx - (n_tags - 1) / 2.0) * width
        hatch = driver_hatch(tag)
        bars = ax.bar(
            offsets, heights,
            width=width * 0.95,
            color=driver_colors.get(tag, 'gray'),
            alpha=0.88,
            hatch=hatch,
            edgecolor='black' if hatch else None,
            label=display_tag(tag),
            zorder=3,
        )
        handles.append(bars[0])
        labels.append(display_tag(tag))

    star_handle = Line2D([0], [0], marker='*', linestyle='None',
                         color='black', markersize=10, label='Best')
    red_star_handle = Line2D([0], [0], marker='*', linestyle='None',
                             color='red', markersize=10, label='Best = MARS-CU/IU')
    handles.append(star_handle)
    labels.append('Best')
    handles.append(red_star_handle)
    labels.append('Best = MARS-CU/IU')

    top_y = max(
        max(v for v in row if v is not None)
        for row in values_by_window
    )
    for window_idx, row in enumerate(values_by_window):
        valid_vals = [v for v in row if v is not None]
        if not valid_vals:
            continue
        best = max(valid_vals)
        for driver_idx, value in enumerate(row):
            if value is None or abs(value - best) > 1e-9:
                continue
            tag = driver_tags[driver_idx]
            star_color = 'red' if tag.upper() in {'MARS-CU', 'MARS-IU'} else 'black'
            star_x = x[window_idx] + (driver_idx - (n_tags - 1) / 2.0) * width
            ax.text(
                star_x, value + 0.8, '★',
                ha='center', va='bottom',
                fontsize=10.5, fontweight='bold', color=star_color,
                zorder=5,
            )

    ax.set_xticks(x)
    xticklabels = ax.set_xticklabels(window_labels, fontsize=TICK_SIZE - 1)
    for lbl in xticklabels:
        lbl.set_fontweight('bold')
    ax.set_xlabel("Drain Period", fontsize=FONT_SIZE, fontweight='bold')
    ax.set_ylabel("Utilization (%)", fontsize=FONT_SIZE, fontweight='bold')
    ax.set_ylim(0.0, max(103.5, top_y + 3.5))
    ax.margins(x=0.005)
    ax.set_xlim(x[0] - group_width * 0.54, x[-1] + group_width * 0.54)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.grid(True, axis='y', alpha=0.3, zorder=0)

    _add_centered_two_row_legend(
        fig, handles, labels, fontsize=LEGEND_SIZE - 1,
        y_top=0.985, row_gap=0.046,
        columnspacing=0.85, handletextpad=0.35,
    )
    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.08, right=0.995)
    save(fig, os.path.join(plots_dir, fname))


def _load_queue_before_announcement_bar_data(table_path, excluded_tags=None):
    if not table_path or not os.path.exists(table_path):
        return [], [], []

    if excluded_tags is None:
        excluded_tags = set()
    excluded_upper = {tag.upper() for tag in excluded_tags}

    period_order = []
    period_labels = {}
    period_values = {}
    driver_tags = []
    seen_tags = set()

    with open(table_path) as f:
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
    values_by_window = [
        [period_values[key].get(tag) for tag in driver_tags]
        for key in period_order
    ]
    return driver_tags, labels, values_by_window


def plot_drain_queue_before_announcement_bars_from_table(table_path, driver_colors,
                                                         plots_dir, fname,
                                                         excluded_tags=None):
    driver_tags, window_labels, values_by_window = _load_queue_before_announcement_bar_data(
        table_path, excluded_tags=excluded_tags)
    if not driver_tags or not values_by_window:
        print(f"  [skip] {fname}: no queue-before-announcement data to plot")
        return

    n_windows = len(values_by_window)
    n_tags = len(driver_tags)
    x = np.arange(n_windows) * 1.03
    group_width = 0.98
    width = group_width / max(n_tags, 1)

    fig_w = max(8.5, 0.62 * n_windows + 3.8)
    fig, ax = plt.subplots(figsize=(fig_w, 4.9))

    for i, xc in enumerate(x):
        if i % 2 != 0:
            continue
        ax.axvspan(
            xc - group_width * 0.57,
            xc + group_width * 0.57,
            color='#e0e0e0',
            alpha=0.95,
            zorder=-2,
        )

    handles = []
    labels = []
    for idx, tag in enumerate(driver_tags):
        heights = [row[idx] if row[idx] is not None else 0.0 for row in values_by_window]
        offsets = x + (idx - (n_tags - 1) / 2.0) * width
        hatch = driver_hatch(tag)
        bars = ax.bar(
            offsets, heights,
            width=width * 0.95,
            color=driver_colors.get(tag, 'gray'),
            alpha=0.88,
            hatch=hatch,
            edgecolor='black' if hatch else None,
            label=display_tag(tag),
            zorder=3,
        )
        handles.append(bars[0])
        labels.append(display_tag(tag))

    top_y = max(
        max((v for v in row if v is not None), default=0.0)
        for row in values_by_window
    )

    ax.set_xticks(x)
    xticklabels = ax.set_xticklabels(window_labels, fontsize=TICK_SIZE - 1)
    for lbl in xticklabels:
        lbl.set_fontweight('bold')
    ax.set_xlabel("Drain Period", fontsize=FONT_SIZE, fontweight='bold')
    ax.set_ylabel("Queue Before Announcement", fontsize=FONT_SIZE, fontweight='bold')
    ax.set_ylim(0.0, _queue_before_ylim(top_y))
    ax.margins(x=0.005)
    ax.set_xlim(x[0] - group_width * 0.54, x[-1] + group_width * 0.54)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.grid(True, axis='y', alpha=0.3, zorder=0)

    _add_centered_two_row_legend(
        fig, handles, labels, fontsize=LEGEND_SIZE - 1,
        y_top=0.985, row_gap=0.046,
        columnspacing=0.85, handletextpad=0.35,
    )
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.08, right=0.995)
    save(fig, os.path.join(plots_dir, fname))


# ── Relative-improvement bar charts ──────────────────────────────────────────

def _compute_stat_row_numeric(values, multiplier=1.0):
    """Return list of floats for [Min, P25, P50, P75, P85, P95, P99, Max, Mean, StdDev, GeoMean]."""
    if not values:
        return None
    arr = np.array(values) * multiplier
    pos = arr[arr > 0]
    gmean = float(np.exp(np.mean(np.log(pos)))) if pos.size > 0 else 0.0
    return [
        float(arr.min()),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 75)),
        float(np.percentile(arr, 85)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
        float(arr.max()),
        float(arr.mean()),
        float(arr.std()),
        gmean,
    ]


def plot_relative_improvement_bars(driver_tags, data_map, driver_colors,
                                    plots_dir, fname, title,
                                    multiplier=1.0, baseline_tag=None):
    """Bar chart: % improvement of each driver over the WFP baseline per stat.

    Lower-is-better metric: improvement = (baseline - driver) / |baseline| * 100.
    Positive bars = driver beats WFP; negative = driver is worse.
    """
    if baseline_tag is None:
        baseline_tag = next((t for t in driver_tags if t.upper().startswith('WFP')), None)
    if baseline_tag is None or baseline_tag not in data_map:
        print(f"  [skip] {fname}: baseline '{baseline_tag}' not found")
        return

    other_tags, improvements = _compute_relative_improvements(
        driver_tags, data_map, multiplier=multiplier, baseline_tag=baseline_tag)
    if not other_tags:
        print(f"  [skip] {fname}: no non-baseline drivers to compare")
        return

    paper_col_w = 4.10
    paper_col_h = 3.95
    axis_font = 11
    tick_font = 10
    legend_font = 9.25
    metric_label = "Bounded Slowdown" if "bsld" in fname.lower() else "Wait Time"

    fig, ax = plt.subplots(figsize=(paper_col_w, paper_col_h))
    handles, labels = _plot_relative_improvement_bars_on_ax(
        ax, other_tags, improvements, driver_colors, group_width=0.88)
    _style_relative_improvement_axis(
        ax,
        metric_label=metric_label,
        axis_font=axis_font,
        tick_font=tick_font,
        show_xlabel=True,
        show_ylabel=True,
        guide_step=25,
    )

    _add_centered_two_row_legend(
        fig, handles, labels, fontsize=legend_font,
        y_top=0.915, row_gap=0.044,
        columnspacing=0.9, handletextpad=0.4,
    )
    fig.subplots_adjust(top=0.62, bottom=0.18, left=0.21, right=0.985)
    save(fig, os.path.join(plots_dir, fname))


def _compute_relative_improvements(driver_tags, data_map, multiplier=1.0, baseline_tag=None):
    if baseline_tag is None:
        baseline_tag = next((t for t in driver_tags if t.upper().startswith('WFP')), None)
    if baseline_tag is None or baseline_tag not in data_map:
        return [], {}

    base_stats = _compute_stat_row_numeric(data_map.get(baseline_tag, []), multiplier)
    if base_stats is None:
        return [], {}
    stat_indices = [_STAT_HEADERS.index(name) for name in _BAR_STAT_HEADERS]
    base_stats = [base_stats[i] for i in stat_indices]

    other_tags = [t for t in driver_tags if t != baseline_tag]
    if not other_tags:
        return [], {}

    improvements = {}
    for tag in other_tags:
        stats = _compute_stat_row_numeric(data_map.get(tag, []), multiplier)
        if stats is None:
            improvements[tag] = [0.0] * len(_BAR_STAT_HEADERS)
            continue
        stats = [stats[i] for i in stat_indices]
        row = []
        for b, s in zip(base_stats, stats):
            row.append(0.0 if b == 0 else (b - s) / abs(b) * 100.0)
        improvements[tag] = row
    return other_tags, improvements


def _plot_relative_improvement_bars_on_ax(ax, other_tags, improvements, driver_colors,
                                          group_width=0.8):
    n_stats  = len(_BAR_STAT_HEADERS)
    n_tags   = len(other_tags)
    x        = np.arange(n_stats) * 1.22
    width    = group_width / max(n_tags, 1)
    handles = []
    labels = []
    for i, tag in enumerate(other_tags):
        offsets = x + (i - (n_tags - 1) / 2.0) * width
        hatch = driver_hatch(tag)
        bar = ax.bar(offsets, improvements[tag], width=width * 0.97,
                     color=driver_colors.get(tag, 'gray'), label=display_tag(tag),
                     alpha=0.85, hatch=hatch, edgecolor='black' if hatch else None)
        handles.append(bar[0])
        labels.append(display_tag(tag))
    return handles, labels


def _metric_x_positions():
    return np.arange(len(_BAR_STAT_HEADERS)) * 1.22


def _add_metric_backgrounds(ax, x_positions):
    for i, x in enumerate(x_positions):
        if i % 2 == 0:
            ax.axvspan(x - 0.58, x + 0.58, color='#f3f3f3', zorder=-2)


def _add_horizontal_guides(ax, y_limits, y_ticks, step=25):
    lo, hi = y_limits
    start = int(np.ceil(lo / step) * step)
    stop = int(np.floor(hi / step) * step)
    for y in range(start, stop + 1, step):
        if y == 0:
            continue
        ax.axhline(y, color='#d0d0d0', linewidth=0.7, alpha=0.75, zorder=-1)
    for y in y_ticks:
        if y == 0 or y < lo or y > hi:
            continue
        ax.axhline(y, color='#9a9a9a', linewidth=1.0, alpha=0.9, zorder=0)


def _style_relative_improvement_axis(ax, metric_label,
                                     axis_font=11, tick_font=10,
                                     show_xlabel=True, show_ylabel=True,
                                     y_limits=(-100, 100), y_ticks=None,
                                     show_xticklabels=True,
                                     show_yticklabels=True,
                                     guide_step=25):
    ax.set_ylim(*y_limits)
    if y_ticks is None:
        y_ticks = [-100, -75, -25, 0, 25, 75, 100]
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.0f}")
    )
    x_positions = _metric_x_positions()
    _add_metric_backgrounds(ax, x_positions)
    _add_horizontal_guides(ax, y_limits, y_ticks, step=guide_step)
    ax.axhline(0, color='#d8d8d8', linewidth=0.9, zorder=0)
    ax.set_xticks(x_positions)
    ax.set_xlim(x_positions[0] - 0.72, x_positions[-1] + 0.72)
    if show_xticklabels:
        labels = ax.set_xticklabels(_BAR_STAT_HEADERS, fontsize=tick_font, rotation=25, ha='right')
        for lbl in labels:
            lbl.set_fontweight('bold')
    else:
        ax.tick_params(axis='x', labelbottom=False, bottom=False)
    if show_xlabel:
        ax.set_xlabel(metric_label, fontsize=axis_font, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel("% Improv. w.r.t WFP", fontsize=axis_font, labelpad=2, fontweight='bold')
    ax.tick_params(axis='y', labelsize=tick_font, labelleft=show_yticklabels)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.grid(False, axis='y')


def _draw_broken_y_axis_marks(ax_top, ax_bottom, d=0.015):
    kwargs_top = dict(transform=ax_top.transAxes, color='black', clip_on=False, linewidth=1.0)
    kwargs_bottom = dict(transform=ax_bottom.transAxes, color='black', clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), **kwargs_top)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs_top)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs_bottom)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_bottom)


def _add_centered_two_row_legend(fig, handles, labels, fontsize,
                                 y_top=0.965, row_gap=0.045,
                                 columnspacing=0.9, handletextpad=0.4):
    if not handles or not labels:
        return

    split = max(1, int(np.ceil(len(labels) / 2)))
    top_handles = handles[:split]
    top_labels = labels[:split]
    bottom_handles = handles[split:]
    bottom_labels = labels[split:]

    top_legend = fig.legend(
        top_handles,
        top_labels,
        fontsize=fontsize,
        loc='upper center',
        bbox_to_anchor=(0.5, y_top),
        ncol=len(top_labels),
        frameon=False,
        columnspacing=columnspacing,
        handletextpad=handletextpad,
    )
    if bottom_handles:
        fig.add_artist(top_legend)
        fig.legend(
            bottom_handles,
            bottom_labels,
            fontsize=fontsize,
            loc='upper center',
            bbox_to_anchor=(0.5, y_top - row_gap),
            ncol=len(bottom_labels),
            frameon=False,
            columnspacing=columnspacing,
            handletextpad=handletextpad,
        )


def _compute_wait_improvement_per_job(driver_tags, wait_maps, procs_map, baseline_tag=None):
    if baseline_tag is None:
        baseline_tag = next((t for t in driver_tags if t.upper().startswith('WFP')), None)
    if baseline_tag is None or baseline_tag not in wait_maps:
        return [], {}, {}

    baseline_wait = wait_maps.get(baseline_tag, {})
    other_tags = [t for t in driver_tags if t != baseline_tag]
    overall = {tag: [] for tag in other_tags}
    by_group = {tag: {g: [] for g in GROUP_LABELS} for tag in other_tags}

    for tag in other_tags:
        driver_wait = wait_maps.get(tag, {})
        common_job_ids = sorted(set(baseline_wait).intersection(driver_wait))
        for jid in common_job_ids:
            base_w = float(baseline_wait[jid])
            drv_w = float(driver_wait[jid])
            if base_w <= 0:
                improvement = 0.0 if drv_w <= 0 else -100.0
            else:
                improvement = (base_w - drv_w) / abs(base_w) * 100.0
            overall[tag].append(improvement)
            g = assign_group(procs_map.get(jid, -1))
            if g is not None:
                by_group[tag][g].append(improvement)
    return other_tags, overall, by_group


def _compute_metric_improvement_per_job(driver_tags, metric_maps, procs_map, baseline_tag=None):
    if baseline_tag is None:
        baseline_tag = next((t for t in driver_tags if t.upper().startswith('WFP')), None)
    if baseline_tag is None or baseline_tag not in metric_maps:
        return [], {}, {}

    baseline_metric = metric_maps.get(baseline_tag, {})
    other_tags = [t for t in driver_tags if t != baseline_tag]
    overall = {tag: [] for tag in other_tags}
    by_group = {tag: {g: [] for g in GROUP_LABELS} for tag in other_tags}

    for tag in other_tags:
        driver_metric = metric_maps.get(tag, {})
        common_job_ids = sorted(set(baseline_metric).intersection(driver_metric))
        for jid in common_job_ids:
            base_v = float(baseline_metric[jid])
            drv_v = float(driver_metric[jid])
            if base_v == 0.0:
                improvement = 0.0 if drv_v == 0.0 else -100.0
            else:
                improvement = (base_v - drv_v) / abs(base_v) * 100.0
            overall[tag].append(improvement)
            g = assign_group(procs_map.get(jid, -1))
            if g is not None:
                by_group[tag][g].append(improvement)
    return other_tags, overall, by_group


def _draw_wait_improvement_violins(ax, ordered_tags, value_map, driver_colors,
                                   y_limits=(-500, 500), tick_font=10):
    data = []
    for tag in ordered_tags:
        values = value_map.get(tag, [])
        if values:
            arr = np.clip(np.asarray(values, dtype=float), y_limits[0], y_limits[1])
            data.append(arr)
        else:
            data.append(np.asarray([0.0], dtype=float))

    positions = np.arange(1, len(ordered_tags) + 1)
    parts = ax.violinplot(
        data, positions=positions, showmedians=True,
        showextrema=False, widths=0.88
    )

    for body, tag in zip(parts.get('bodies', []), ordered_tags):
        hatch = driver_hatch(tag)
        body.set_facecolor(driver_colors.get(tag, 'gray'))
        body.set_edgecolor('black')
        body.set_alpha(0.85)
        if hatch:
            body.set_hatch(hatch)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.3)

    ax.set_ylim(*y_limits)
    y_ticks = [-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500]
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.0f}")
    )
    _add_horizontal_guides(ax, y_limits, y_ticks, step=50)
    ax.axhline(0, color='black', linewidth=1.0, zorder=1)
    ax.set_xticks(positions)
    labels = ax.set_xticklabels(
        [display_tag(tag) for tag in ordered_tags],
        fontsize=tick_font, rotation=30, ha='right'
    )
    for lbl in labels:
        lbl.set_fontweight('bold')
    ax.tick_params(axis='y', labelsize=tick_font)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.grid(False, axis='y')


def _box_improvement_legend_handles():
    return [
        Line2D([0], [0], color='black', linewidth=2.1, linestyle='--', label='Mean'),
        Line2D([0], [0], color='black', linewidth=3.0, label='P50'),
        matplotlib.patches.Patch(
            facecolor='#e8e8e8', edgecolor='black', label='Box = P25-P75'),
        Line2D([0], [0], color='black', linewidth=1.1, label='Whiskers = P5-P95'),
    ]


def _add_boxplot_horizontal_guides(ax, y_limits):
    lo, hi = y_limits
    start = int(np.ceil(lo / 25.0) * 25)
    stop = int(np.floor(hi / 25.0) * 25)
    for y in range(start, stop + 1, 25):
        if y == 0:
            continue
        if y % 50 == 0:
            ax.axhline(y, color='#b0b0b0', linewidth=0.95, alpha=0.9, zorder=-1)
        else:
            ax.axhline(y, color='#cdcdcd', linewidth=0.75, alpha=0.85,
                       linestyle=':', zorder=-1)


def _draw_improvement_boxplots(ax, ordered_tags, value_map, driver_colors,
                               y_limits=(-500, 500), tick_font=10,
                               y_ticks=None,
                               show_xticklabels=True,
                               show_yticklabels=True,
                               clip_limits=(-600, 150)):
    data = []
    has_data = []
    for tag in ordered_tags:
        values = value_map.get(tag, [])
        if values:
            arr = np.clip(np.asarray(values, dtype=float), clip_limits[0], clip_limits[1])
            data.append(arr)
            has_data.append(True)
        else:
            data.append(np.asarray([0.0], dtype=float))
            has_data.append(False)

    positions = np.arange(1, len(ordered_tags) + 1)
    parts = ax.boxplot(
        data,
        positions=positions,
        widths=0.68,
        whis=(5, 95),
        showmeans=True,
        meanline=True,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(linewidth=1.9, edgecolor='black'),
        whiskerprops=dict(linewidth=1.1, color='black'),
        capprops=dict(linewidth=1.1, color='black'),
        medianprops=dict(linewidth=3.0, color='black'),
        meanprops=dict(linewidth=2.1, color='red', linestyle=':'),
    )

    for box, tag, present in zip(parts.get('boxes', []), ordered_tags, has_data):
        box.set_facecolor(driver_colors.get(tag, 'gray'))
        box.set_alpha(0.78 if present else 0.18)
        hatch = driver_hatch(tag)
        if hatch:
            box.set_hatch(hatch)

    ax.set_ylim(*y_limits)
    if y_ticks is None:
        y_ticks = [-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500]
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.0f}")
    )
    _add_boxplot_horizontal_guides(ax, y_limits)
    ax.axhline(0, color='black', linewidth=1.0, zorder=1)
    ax.set_xticks(positions)
    if show_xticklabels:
        labels = ax.set_xticklabels(
            [display_tag(tag) for tag in ordered_tags],
            fontsize=tick_font, rotation=30, ha='right'
        )
        for lbl in labels:
            lbl.set_fontweight('bold')
    else:
        ax.tick_params(axis='x', labelbottom=False, bottom=False)
    ax.tick_params(axis='y', labelsize=tick_font, labelleft=show_yticklabels)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.grid(False, axis='y')


def _round_down_to_step(value, step):
    return int(step * np.floor(float(value) / float(step)))


def _round_up_to_step(value, step):
    return int(step * np.ceil(float(value) / float(step)))


def _ticks_within(lo, hi, step):
    start = int(np.ceil(float(lo) / float(step)) * step)
    stop = int(np.floor(float(hi) / float(step)) * step)
    if start > stop:
        mid = int(np.round(((float(lo) + float(hi)) / 2.0) / float(step)) * step)
        return [mid]
    return list(range(start, stop + step, step))


def _box_break_spec(ordered_tags, value_map, clip_lo=-500, clip_hi=500,
                    middle_tick_step=25, tail_tick_step=50):
    # Keep RLScheduler visible in the plot, but do not let its heavy tail
    # dominate the automatic broken-axis placement.
    tags_for_break = [
        tag for tag in ordered_tags
        if not tag.upper().startswith('RLSCHEDULER')
    ]
    if not tags_for_break:
        tags_for_break = list(ordered_tags)

    lower_whiskers = []
    upper_whiskers = []
    q1s = []
    q3s = []
    for tag in tags_for_break:
        values = value_map.get(tag, [])
        if not values:
            continue
        arr = np.clip(np.asarray(values, dtype=float), clip_lo, clip_hi)
        if arr.size == 0:
            continue
        lower_whiskers.append(float(np.percentile(arr, 5)))
        upper_whiskers.append(float(np.percentile(arr, 95)))
        q1s.append(float(np.percentile(arr, 25)))
        q3s.append(float(np.percentile(arr, 75)))

    if not lower_whiskers or not upper_whiskers or not q1s or not q3s:
        middle_limits = (-75.0, 100.0)
        lower_limits = (-500.0, -350.0)
        upper_limits = (125.0, 225.0)
        return {
            'upper_limits': upper_limits,
            'middle_limits': middle_limits,
            'lower_limits': lower_limits,
            'upper_ticks': _ticks_within(*upper_limits, step=tail_tick_step),
            'middle_ticks': _ticks_within(*middle_limits, step=middle_tick_step),
            'lower_ticks': _ticks_within(*lower_limits, step=tail_tick_step),
        }

    q1_min = min(q1s)
    q3_max = max(q3s)
    whislo_min = min(lower_whiskers)
    whislo_max = max(lower_whiskers)
    whishi_min = min(upper_whiskers)
    whishi_max = max(upper_whiskers)

    middle_lo = q1_min - 8.0
    middle_hi = q3_max + 8.0
    lower_lo = max(float(clip_lo), whislo_min - 8.0)
    lower_hi = whislo_max + 8.0
    upper_lo = whishi_min - 8.0
    upper_hi = min(float(clip_hi), whishi_max + 8.0)

    if lower_hi >= middle_lo - 6.0:
        split = (whislo_max + q1_min) / 2.0
        lower_hi = split - 3.0
        middle_lo = split + 3.0
    if upper_lo <= middle_hi + 6.0:
        split = (q3_max + whishi_min) / 2.0
        middle_hi = split - 3.0
        upper_lo = split + 3.0

    middle_lo = max(lower_hi + 6.0, middle_lo)
    upper_lo = max(middle_hi + 6.0, upper_lo)
    middle_hi = min(upper_lo - 6.0, middle_hi)

    if middle_hi <= middle_lo + 20.0:
        center = (q1_min + q3_max) / 2.0
        middle_lo = center - 35.0
        middle_hi = center + 35.0
        lower_hi = min(lower_hi, middle_lo - 6.0)
        upper_lo = max(upper_lo, middle_hi + 6.0)

    lower_limits = (lower_lo, max(lower_lo + 12.0, lower_hi))
    middle_limits = (middle_lo, max(middle_lo + 24.0, middle_hi))
    upper_lo = min(float(clip_hi) - 12.0, upper_lo)
    upper_hi = min(float(clip_hi), max(upper_lo + 12.0, upper_hi))
    upper_limits = (upper_lo, upper_hi)

    lower_ticks = _ticks_within(*lower_limits, step=tail_tick_step)
    middle_ticks = _ticks_within(*middle_limits, step=middle_tick_step)
    upper_ticks = _ticks_within(*upper_limits, step=tail_tick_step)

    return {
        'upper_limits': upper_limits,
        'middle_limits': middle_limits,
        'lower_limits': lower_limits,
        'upper_ticks': upper_ticks,
        'middle_ticks': middle_ticks,
        'lower_ticks': lower_ticks,
    }


def _draw_improvement_boxplots_broken(ax_main, ax_lower, ordered_tags,
                                      value_map, driver_colors, tick_font=10,
                                      main_limits=(-300, 150),
                                      lower_limits=(-600, -500),
                                      main_ticks=None,
                                      lower_ticks=None,
                                      show_yticklabels=True):
    if main_ticks is None:
        main_ticks = [-300, -250, -200, -150, -100, -50, 0, 50, 100, 150]
    if lower_ticks is None:
        lower_ticks = [-600, -550, -500]

    _draw_improvement_boxplots(
        ax_main, ordered_tags, value_map, driver_colors,
        y_limits=main_limits, tick_font=tick_font,
        y_ticks=main_ticks,
        show_xticklabels=False, show_yticklabels=show_yticklabels,
        clip_limits=(-600, 150),
    )
    _draw_improvement_boxplots(
        ax_lower, ordered_tags, value_map, driver_colors,
        y_limits=lower_limits, tick_font=max(8, tick_font - 1),
        y_ticks=lower_ticks,
        show_xticklabels=True, show_yticklabels=show_yticklabels,
        clip_limits=(-600, 150),
    )

    ax_main.spines['bottom'].set_visible(False)
    ax_lower.spines['top'].set_visible(False)
    ax_main.tick_params(axis='x', labelbottom=False, bottom=False)
    ax_lower.tick_params(axis='x', top=False)
    _draw_broken_y_axis_marks(ax_main, ax_lower, d=0.018)


def plot_wait_improvement_violin_overall(driver_tags, wait_improvement_map,
                                         driver_colors, plots_dir,
                                         baseline_tag=None):
    ordered_tags = [tag for tag in driver_tags
                    if baseline_tag is None or tag != baseline_tag]
    ordered_tags = [tag for tag in ordered_tags if wait_improvement_map.get(tag)]
    if not ordered_tags:
        print("  [skip] violin_wait_improvement.png: no non-baseline drivers")
        return

    fig, ax = plt.subplots(figsize=(6.3, 4.3))
    _draw_wait_improvement_violins(
        ax, ordered_tags, wait_improvement_map, driver_colors,
        y_limits=(-500, 500), tick_font=10
    )
    ax.set_ylabel("% Improv. w.r.t WFP", fontsize=12, fontweight='bold')
    ax.set_xlabel("Scheduler", fontsize=12, fontweight='bold')
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, "violin_wait_improvement.png"))


def plot_wait_improvement_box_overall(driver_tags, wait_improvement_map,
                                      driver_colors, plots_dir,
                                      baseline_tag=None):
    ordered_tags = [tag for tag in driver_tags
                    if baseline_tag is None or tag != baseline_tag]
    ordered_tags = [tag for tag in ordered_tags if wait_improvement_map.get(tag)]
    if not ordered_tags:
        print("  [skip] box_wait_improvement.png: no non-baseline drivers")
        return

    fig = plt.figure(figsize=(6.9, 5.6))
    inner = fig.add_gridspec(2, 1, height_ratios=[4.8, 1.05], hspace=0.16)
    ax_main = fig.add_subplot(inner[0])
    ax_lower = fig.add_subplot(inner[1], sharex=ax_main)
    _draw_improvement_boxplots_broken(
        ax_main, ax_lower, ordered_tags, wait_improvement_map, driver_colors,
        tick_font=10,
        main_limits=(-300, 150),
        lower_limits=(-600, -500),
        main_ticks=[-300, -250, -200, -150, -100, -50, 0, 50, 100, 150],
        lower_ticks=[-600, -550, -500],
        show_yticklabels=True,
    )
    ax_main.set_ylabel("% Improv. w.r.t WFP", fontsize=12, fontweight='bold')
    ax_lower.set_xlabel("Scheduler", fontsize=12, fontweight='bold')
    ax_lower.tick_params(axis='x', labelsize=10)
    _add_centered_two_row_legend(
        fig,
        _box_improvement_legend_handles(),
        [h.get_label() for h in _box_improvement_legend_handles()],
        fontsize=9,
        y_top=0.985,
        row_gap=0.044,
        columnspacing=0.85,
        handletextpad=0.35,
    )
    fig.subplots_adjust(top=0.82, bottom=0.17, left=0.11, right=0.98)
    save(fig, os.path.join(plots_dir, "box_wait_improvement.png"))


def plot_wait_improvement_violin_groups(driver_tags, wait_improvement_by_group,
                                        driver_colors, plots_dir,
                                        baseline_tag=None):
    ordered_tags = [tag for tag in driver_tags
                    if baseline_tag is None or tag != baseline_tag]
    if not ordered_tags:
        print("  [skip] violin_wait_improvement_groups.png: no non-baseline drivers")
        return

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), sharey=True)
    for idx, group in enumerate(GROUP_LABELS):
        row, col = divmod(idx, 2)
        ax = axes[row][col]
        value_map = {tag: wait_improvement_by_group.get(tag, {}).get(group, []) for tag in ordered_tags}
        _draw_wait_improvement_violins(
            ax, ordered_tags, value_map, driver_colors,
            y_limits=(-500, 500), tick_font=9
        )
        ax.set_title(group, fontsize=13, pad=5, fontweight='bold')
        if col == 0:
            ax.set_ylabel("% Improv. w.r.t WFP", fontsize=12, fontweight='bold')
        if row == 1:
            ax.set_xlabel("Scheduler", fontsize=11, fontweight='bold')
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, "violin_wait_improvement_groups.png"))


def plot_wait_improvement_box_groups(driver_tags, wait_improvement_by_group,
                                     driver_colors, plots_dir,
                                     baseline_tag=None):
    ordered_tags = [tag for tag in driver_tags
                    if baseline_tag is None or tag != baseline_tag]
    if not ordered_tags:
        print("  [skip] box_wait_improvement_groups.png: no non-baseline drivers")
        return

    fig = plt.figure(figsize=(9.7, 8.4))
    outer = fig.add_gridspec(2, 2)
    for idx, group in enumerate(GROUP_LABELS):
        row, col = divmod(idx, 2)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[4.8, 1.05], hspace=0.16)
        ax_main = fig.add_subplot(inner[0])
        ax_lower = fig.add_subplot(inner[1], sharex=ax_main)
        value_map = {tag: wait_improvement_by_group.get(tag, {}).get(group, []) for tag in ordered_tags}
        _draw_improvement_boxplots_broken(
            ax_main, ax_lower, ordered_tags, value_map, driver_colors,
            tick_font=9,
            main_limits=(-300, 150),
            lower_limits=(-600, -500),
            main_ticks=[-300, -250, -200, -150, -100, -50, 0, 50, 100, 150],
            lower_ticks=[-600, -550, -500],
            show_yticklabels=(col == 0),
        )
        ax_main.set_title(group, fontsize=13, pad=5, fontweight='bold')
        if col == 0:
            ax_main.set_ylabel("% Improv. w.r.t WFP", fontsize=12, fontweight='bold')
        if row == 1:
            ax_lower.set_xlabel("Scheduler", fontsize=11, fontweight='bold')

    _add_centered_two_row_legend(
        fig,
        _box_improvement_legend_handles(),
        [h.get_label() for h in _box_improvement_legend_handles()],
        fontsize=9,
        y_top=0.985,
        row_gap=0.044,
        columnspacing=0.85,
        handletextpad=0.35,
    )
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.09, right=0.99,
                        hspace=0.28, wspace=0.12)
    save(fig, os.path.join(plots_dir, "box_wait_improvement_groups.png"))


def _compute_wait_delta_per_job(driver_tags, wait_maps, procs_map, baseline_tag=None):
    """Per-job absolute wait delta: (driver_wait - baseline_wait) in hours.

    Returns (other_tags, overall_delta_map, by_group_delta_map).
    Negative = driver is faster; positive = driver made this job wait longer.
    """
    if baseline_tag is None:
        baseline_tag = next((t for t in driver_tags if t.upper().startswith('WFP')), None)
    if baseline_tag is None or baseline_tag not in wait_maps:
        return [], {}, {}

    baseline_wait = wait_maps[baseline_tag]
    other_tags = [t for t in driver_tags if t != baseline_tag]
    overall   = {tag: [] for tag in other_tags}
    by_group  = {tag: {g: [] for g in GROUP_LABELS} for tag in other_tags}

    for tag in other_tags:
        driver_wait = wait_maps.get(tag, {})
        for jid in sorted(set(baseline_wait).intersection(driver_wait)):
            delta_h = (float(driver_wait[jid]) - float(baseline_wait[jid])) / 3600.0
            overall[tag].append(delta_h)
            g = assign_group(procs_map.get(jid, -1))
            if g is not None:
                by_group[tag][g].append(delta_h)
    return other_tags, overall, by_group


def _draw_wait_delta_cdfs(ax, ordered_tags, delta_map, driver_colors):
    """Draw overlaid CDFs of per-job wait delta (hours) on ax."""
    for tag in ordered_tags:
        values = delta_map.get(tag, [])
        if not values:
            continue
        _draw_cdf_multi(ax, values, driver_colors.get(tag, 'gray'), tag)
    ax.axvline(0, color='black', linewidth=1.2, linestyle='--', zorder=2, label='No change')
    ax.set_xscale('symlog', linthresh=0.5)
    ax.set_xlabel("Wait time change vs WFP (hours, symlog)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Fraction of jobs", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)


def plot_wait_delta_cdf_overall(driver_tags, wait_delta_map, driver_colors, plots_dir,
                                baseline_tag=None):
    ordered_tags = [t for t in driver_tags
                    if (baseline_tag is None or t != baseline_tag)
                    and wait_delta_map.get(t)]
    if not ordered_tags:
        print("  [skip] cdf_wait_delta_overall.png: no non-baseline drivers")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _draw_wait_delta_cdfs(ax, ordered_tags, wait_delta_map, driver_colors)
    ax.set_title("Per-job wait change vs WFP — all jobs", fontsize=13, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    # separate driver lines from the 'No change' vline handle
    driver_handles = [h for h, l in zip(handles, labels) if l != 'No change']
    driver_labels  = [l for l in labels if l != 'No change']
    pctile_handles = _pctile_legend_handles()
    ax.legend(
        driver_handles + pctile_handles,
        driver_labels + [h.get_label() for h in pctile_handles],
        fontsize=9, loc='lower right', framealpha=0.85,
    )
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, "cdf_wait_delta_overall.png"))


def plot_wait_delta_cdf_groups(driver_tags, wait_delta_by_group, driver_colors, plots_dir,
                               baseline_tag=None):
    ordered_tags = [t for t in driver_tags
                    if baseline_tag is None or t != baseline_tag]
    if not ordered_tags:
        print("  [skip] cdf_wait_delta_groups.png: no non-baseline drivers")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    for idx, group in enumerate(GROUP_LABELS):
        row, col = divmod(idx, 2)
        ax = axes[row][col]
        delta_map = {t: wait_delta_by_group.get(t, {}).get(group, []) for t in ordered_tags}
        _draw_wait_delta_cdfs(ax, ordered_tags, delta_map, driver_colors)
        ax.set_title(group, fontsize=13, fontweight='bold', pad=4)
        if col != 0:
            ax.set_ylabel("")
        if row != 1:
            ax.set_xlabel("")

    # shared legend above
    handles, labels = axes[0][0].get_legend_handles_labels()
    driver_handles = [h for h, l in zip(handles, labels) if l != 'No change']
    driver_labels  = [l for l in labels if l != 'No change']
    pctile_handles = _pctile_legend_handles()
    _add_centered_two_row_legend(
        fig,
        driver_handles + pctile_handles,
        driver_labels + [h.get_label() for h in pctile_handles],
        fontsize=9,
        y_top=0.985,
        row_gap=0.040,
        columnspacing=0.85,
        handletextpad=0.40,
    )
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.09, right=0.99,
                        hspace=0.32, wspace=0.16)
    save(fig, os.path.join(plots_dir, "cdf_wait_delta_groups.png"))


def plot_bsld_improvement_violin_overall(driver_tags, bsld_improvement_map,
                                         driver_colors, plots_dir,
                                         baseline_tag=None):
    ordered_tags = [tag for tag in driver_tags
                    if baseline_tag is None or tag != baseline_tag]
    ordered_tags = [tag for tag in ordered_tags if bsld_improvement_map.get(tag)]
    if not ordered_tags:
        print("  [skip] violin_bsld_improvement.png: no non-baseline drivers")
        return

    fig, ax = plt.subplots(figsize=(6.3, 4.3))
    _draw_wait_improvement_violins(
        ax, ordered_tags, bsld_improvement_map, driver_colors,
        y_limits=(-500, 500), tick_font=10
    )
    ax.set_ylabel("% Improv. w.r.t WFP", fontsize=12, fontweight='bold')
    ax.set_xlabel("Scheduler", fontsize=12, fontweight='bold')
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, "violin_bsld_improvement.png"))


def plot_bsld_improvement_box_overall(driver_tags, bsld_improvement_map,
                                      driver_colors, plots_dir,
                                      baseline_tag=None):
    ordered_tags = [tag for tag in driver_tags
                    if baseline_tag is None or tag != baseline_tag]
    ordered_tags = [tag for tag in ordered_tags if bsld_improvement_map.get(tag)]
    if not ordered_tags:
        print("  [skip] box_bsld_improvement.png: no non-baseline drivers")
        return

    fig = plt.figure(figsize=(6.9, 5.6))
    inner = fig.add_gridspec(2, 1, height_ratios=[4.8, 1.05], hspace=0.16)
    ax_main = fig.add_subplot(inner[0])
    ax_lower = fig.add_subplot(inner[1], sharex=ax_main)
    _draw_improvement_boxplots_broken(
        ax_main, ax_lower, ordered_tags, bsld_improvement_map, driver_colors,
        tick_font=10,
        main_limits=(-300, 150),
        lower_limits=(-600, -500),
        main_ticks=[-300, -250, -200, -150, -100, -50, 0, 50, 100, 150],
        lower_ticks=[-600, -550, -500],
        show_yticklabels=True,
    )
    ax_main.set_ylabel("% Improv. w.r.t WFP", fontsize=12, fontweight='bold')
    ax_lower.set_xlabel("Scheduler", fontsize=12, fontweight='bold')
    ax_lower.tick_params(axis='x', labelsize=10)
    _add_centered_two_row_legend(
        fig,
        _box_improvement_legend_handles(),
        [h.get_label() for h in _box_improvement_legend_handles()],
        fontsize=9,
        y_top=0.985,
        row_gap=0.044,
        columnspacing=0.85,
        handletextpad=0.35,
    )
    fig.subplots_adjust(top=0.82, bottom=0.17, left=0.11, right=0.98)
    save(fig, os.path.join(plots_dir, "box_bsld_improvement.png"))


def plot_bsld_improvement_violin_groups(driver_tags, bsld_improvement_by_group,
                                        driver_colors, plots_dir,
                                        baseline_tag=None):
    ordered_tags = [tag for tag in driver_tags
                    if baseline_tag is None or tag != baseline_tag]
    if not ordered_tags:
        print("  [skip] violin_bsld_improvement_groups.png: no non-baseline drivers")
        return

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), sharey=True)
    for idx, group in enumerate(GROUP_LABELS):
        row, col = divmod(idx, 2)
        ax = axes[row][col]
        value_map = {tag: bsld_improvement_by_group.get(tag, {}).get(group, []) for tag in ordered_tags}
        _draw_wait_improvement_violins(
            ax, ordered_tags, value_map, driver_colors,
            y_limits=(-500, 500), tick_font=9
        )
        ax.set_title(group, fontsize=13, pad=5, fontweight='bold')
        if col == 0:
            ax.set_ylabel("% Improv. w.r.t WFP", fontsize=12, fontweight='bold')
        if row == 1:
            ax.set_xlabel("Scheduler", fontsize=11, fontweight='bold')
    fig.tight_layout()
    save(fig, os.path.join(plots_dir, "violin_bsld_improvement_groups.png"))


def plot_bsld_improvement_box_groups(driver_tags, bsld_improvement_by_group,
                                     driver_colors, plots_dir,
                                     baseline_tag=None):
    ordered_tags = [tag for tag in driver_tags
                    if baseline_tag is None or tag != baseline_tag]
    if not ordered_tags:
        print("  [skip] box_bsld_improvement_groups.png: no non-baseline drivers")
        return

    fig = plt.figure(figsize=(9.7, 8.4))
    outer = fig.add_gridspec(2, 2)
    for idx, group in enumerate(GROUP_LABELS):
        row, col = divmod(idx, 2)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[4.8, 1.05], hspace=0.16)
        ax_main = fig.add_subplot(inner[0])
        ax_lower = fig.add_subplot(inner[1], sharex=ax_main)
        value_map = {tag: bsld_improvement_by_group.get(tag, {}).get(group, []) for tag in ordered_tags}
        _draw_improvement_boxplots_broken(
            ax_main, ax_lower, ordered_tags, value_map, driver_colors,
            tick_font=9,
            main_limits=(-300, 150),
            lower_limits=(-600, -500),
            main_ticks=[-300, -250, -200, -150, -100, -50, 0, 50, 100, 150],
            lower_ticks=[-600, -550, -500],
            show_yticklabels=(col == 0),
        )
        ax_main.set_title(group, fontsize=13, pad=5, fontweight='bold')
        if col == 0:
            ax_main.set_ylabel("% Improv. w.r.t WFP", fontsize=12, fontweight='bold')
        if row == 1:
            ax_lower.set_xlabel("Scheduler", fontsize=11, fontweight='bold')

    _add_centered_two_row_legend(
        fig,
        _box_improvement_legend_handles(),
        [h.get_label() for h in _box_improvement_legend_handles()],
        fontsize=9,
        y_top=0.985,
        row_gap=0.044,
        columnspacing=0.85,
        handletextpad=0.35,
    )
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.09, right=0.99,
                        hspace=0.28, wspace=0.12)
    save(fig, os.path.join(plots_dir, "box_bsld_improvement_groups.png"))


def plot_relative_improvement_bars_2x2(driver_tags, data_by_group, driver_colors,
                                       plots_dir, fname, metric_label,
                                       multiplier=1.0, baseline_tag=None):
    if baseline_tag is None:
        baseline_tag = next((t for t in driver_tags if t.upper().startswith('WFP')), None)
    if baseline_tag is None:
        print(f"  [skip] {fname}: no WFP baseline found")
        return

    fig = plt.figure(figsize=(8.15, 6.7))
    outer = fig.add_gridspec(2, 2)
    legend_handles = None
    legend_labels = None
    top_row_share = None
    bottom_top_share = None
    bottom_bottom_share = None

    for idx, group in enumerate(GROUP_LABELS):
        row, col = divmod(idx, 2)
        group_map = {tag: data_by_group[tag].get(group, []) for tag in driver_tags}
        other_tags, improvements = _compute_relative_improvements(
            driver_tags, group_map, multiplier=multiplier, baseline_tag=baseline_tag)

        if group in ("L", "XL"):
            inner = outer[row, col].subgridspec(2, 1, height_ratios=[4.0, 1.1], hspace=0.14)
            ax_top = fig.add_subplot(inner[0], sharey=bottom_top_share)
            ax_bottom = fig.add_subplot(inner[1], sharex=ax_top, sharey=bottom_bottom_share)
            if bottom_top_share is None:
                bottom_top_share = ax_top
            if bottom_bottom_share is None:
                bottom_bottom_share = ax_bottom

            if other_tags:
                handles, labels = _plot_relative_improvement_bars_on_ax(
                    ax_top, other_tags, improvements, driver_colors, group_width=0.99)
                _plot_relative_improvement_bars_on_ax(
                    ax_bottom, other_tags, improvements, driver_colors, group_width=0.99)
                if legend_handles is None:
                    legend_handles, legend_labels = handles, labels

            _style_relative_improvement_axis(
                ax_top,
                metric_label=metric_label,
                axis_font=13,
                tick_font=11,
                show_xlabel=False,
                show_ylabel=False,
                y_limits=(-200, 100),
                y_ticks=[-200, -100, 0, 100],
                show_xticklabels=False,
                show_yticklabels=(col == 0),
                guide_step=50,
            )
            _style_relative_improvement_axis(
                ax_bottom,
                metric_label=metric_label,
                axis_font=11,
                tick_font=10,
                show_xlabel=True,
                show_ylabel=False,
                y_limits=(-500, -400),
                y_ticks=[-500, -400],
                show_xticklabels=True,
                show_yticklabels=(col == 0),
                guide_step=50,
            )
            ax_top.set_title(group, fontsize=13, pad=5, fontweight='bold')
            ax_top.tick_params(axis='y', pad=6)
            ax_bottom.tick_params(axis='y', pad=6)
            ax_top.spines['bottom'].set_visible(False)
            ax_bottom.spines['top'].set_visible(False)
            _draw_broken_y_axis_marks(ax_top, ax_bottom)
        else:
            ax = fig.add_subplot(outer[row, col], sharey=top_row_share)
            if top_row_share is None:
                top_row_share = ax
            if other_tags:
                handles, labels = _plot_relative_improvement_bars_on_ax(
                    ax, other_tags, improvements, driver_colors, group_width=0.99)
                if legend_handles is None:
                    legend_handles, legend_labels = handles, labels
            _style_relative_improvement_axis(
                ax,
                metric_label=metric_label,
                axis_font=13,
                tick_font=11,
                show_xlabel=False,
                show_ylabel=False,
                y_limits=(-150, 100),
                y_ticks=[-150, -100, -50, 0, 50, 100],
                show_xticklabels=True,
                show_yticklabels=(col == 0),
                guide_step=25,
            )
            ax.set_title(group, fontsize=13, pad=5, fontweight='bold')

    if legend_handles:
        _add_centered_two_row_legend(
            fig, legend_handles, legend_labels, fontsize=10.5,
            y_top=0.885, row_gap=0.046,
            columnspacing=1.0, handletextpad=0.4,
        )
    fig.text(0.024, 0.55, "% Improv. w.r.t WFP",
             rotation=90, va='center', ha='center', fontsize=13, fontweight='bold')
    fig.text(0.024, 0.22, "% Improv. w.r.t WFP",
             rotation=90, va='center', ha='center', fontsize=13, fontweight='bold')
    fig.subplots_adjust(top=0.71, bottom=0.12, left=0.10, right=0.988,
                        hspace=0.34, wspace=0.06)
    save(fig, os.path.join(plots_dir, fname))


# ── Tail-wait heatmap ────────────────────────────────────────────────────────

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
    if not driver_tags:
        return

    cell_w = 2.8
    cell_h = 1.8

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
            f'{display_tag(tag)}  ({n_tail} / {n_total},  {pct:.0f}%)',
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
        print(f"Usage: {sys.argv[0]} <experiment.json> [--plots-dir <dir>] [--global-only] [--force]")
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
        elif argv[i] == "--force":
            global _FORCE_REGEN
            _FORCE_REGEN = True
            i += 1
        else:
            i += 1

    with open(config_path) as f:
        config = json.load(f)

    swf_path   = config["swf_path"]
    output_dir = output_dir_override or config["output_dir"]
    plots_dir  = plots_dir_override or (output_dir.rstrip("/") + "_plots")

    _scan_existing_plots(plots_dir)

    # Detect system from swf_path
    swf_lower = swf_path.lower()
    if 'theta' in swf_lower:
        _configure_system('theta')
    elif 'polaris' in swf_lower:
        _configure_system('polaris')
    else:
        print(f"WARNING: cannot detect system from swf_path '{swf_path}', defaulting to polaris")
        _configure_system('polaris')

    print(f"System:    {SYSTEM.upper()}  ({S_SYS} nodes, {ANALYSIS_YEAR} trace, "
          f"groups: {GROUP_LABELS})")

    _IGNORE_TAGS = set()

    driver_tags = [
        d.get("tag", d["type"])
        for d in config.get("drivers", [])
        if d.get("plot", True) and d.get("tag", d["type"]) not in _IGNORE_TAGS
    ]
    tag_to_driver_cfg = {
        d.get("tag", d["type"]): d
        for d in config.get("drivers", [])
        if d.get("plot", True) and d.get("tag", d["type"]) not in _IGNORE_TAGS
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
    analysis_end_dt   = datetime.fromtimestamp(analysis_end,   tz=CST)
    print(
        "Analysis window: "
        f"{analysis_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} to "
        f"{analysis_end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} "
        "(jobs filtered by submit time)"
    )
    print()

    driver_colors = {}
    for i, tag in enumerate(driver_tags):
        tag_up = tag.upper()
        if tag_up.startswith('MARS'):
            mars_color = next(
                (c for suffix, c in _MARS_COLORS.items() if tag_up.endswith(suffix.upper())),
                '#f6de6a',
            )
            driver_colors[tag] = mars_color
        else:
            matched = next(
                (color for prefix, color in _HEURISTIC_COLORS.items()
                 if tag_up.startswith(prefix)),
                None
            )
            driver_colors[tag] = matched if matched is not None else driver_color(i)

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

    procs_map, walltimes_map, _, runtimes_map = parse_swf(swf_path)

    # ── Pass 1: load events ────────────────────────────────────────────────────
    valid_tags      = []
    raw_submit_maps = {}
    raw_start_maps  = {}
    raw_end_maps    = {}
    submit_maps     = {}
    start_maps      = {}
    end_maps        = {}

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

    # ── Pre-pass: per-class P99 wait (for KDE y-axes) ────────────────────────
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

    wait_plot_data = {}
    wait_plot_dists = {}
    wait_plot_dists_no_s = {}
    drain_total_group_counts = {}
    for tag in valid_tags:
        submit = submit_maps[tag]
        start = start_maps[tag]
        end = end_maps[tag]

        job_ids = []
        wait_map = {}
        for jid, submit_time in submit.items():
            if submit_time > common_submit_end:
                continue
            if jid not in start or jid not in end:
                continue
            wait_map[jid] = start[jid] - submit[jid]
            job_ids.append(jid)

        wait_plot_data[tag] = (job_ids, wait_map)
        wait_plot_dists[tag] = _build_wait_distribution_by_group(job_ids, wait_map, procs_map)
        wait_plot_dists_no_s[tag] = _build_wait_distribution_by_group(
            job_ids, wait_map, procs_map, exclude_groups={'S'}
        )
        totals = {group: 0 for group in GROUP_LABELS}
        for jid in job_ids:
            group = assign_group(procs_map.get(jid))
            if group is not None:
                totals[group] += 1
        drain_total_group_counts[tag] = totals

    wait_plot_shared_scale = _build_wait_distribution_shared_scale(wait_plot_dists.values())
    wait_plot_shared_scale_no_s = _build_wait_distribution_shared_scale(wait_plot_dists_no_s.values())
    wait_kde_shared_scale = None
    if wait_plot_shared_scale is not None:
        wait_kde_x_grid = np.geomspace(
            wait_plot_shared_scale['x_lo'],
            wait_plot_shared_scale['x_hi'],
            400,
        )
        wait_kde_shared_scale = _build_wait_kde_shared_scale(
            wait_plot_dists.values(),
            wait_kde_x_grid,
        )

    # ── Pass 2: build per-driver metrics ──────────────────────────────────────
    per_driver_end_time_limit = analysis_end

    all_wait_by_group = {}
    all_bsld_by_group = {}
    all_wait          = {}
    all_bsld          = {}
    all_job_ids_map   = {}
    all_wait_maps     = {}
    all_bsld_maps     = {}
    all_backfill_maps = {}

    for tag in valid_tags:
        print(f"Driver: {tag}{' (partial)' if partial else ''}")

        submit = submit_maps[tag]
        start  = start_maps[tag]
        end    = end_maps[tag]

        job_ids, wait_map = wait_plot_data[tag]
        wait_dist = wait_plot_dists.get(tag)
        wait_dist_no_s = wait_plot_dists_no_s.get(tag)

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
            if tag_to_driver_cfg.get(tag, {}).get("type") == "mcts":
                plot_per_driver_mcts_iterations_violin(
                    tag, output_dir, plots_dir,
                    driver_color_value=driver_colors.get(tag),
                    time_start_limit=analysis_start,
                    time_end_limit=analysis_end,
                )
                plot_per_driver_root_branching_violin(
                    tag, output_dir, plots_dir,
                    driver_color_value=driver_colors.get(tag),
                    time_start_limit=analysis_start,
                    time_end_limit=analysis_end,
                )
            plot_per_driver_stacked_wait_histogram(
                tag, job_ids, wait_map, procs_map, plots_dir,
                dist=wait_dist, shared_scale=wait_plot_shared_scale,
            )
            plot_per_driver_wait_kde_by_size(
                tag, job_ids, wait_map, procs_map, plots_dir,
                dist=wait_dist,
                shared_scale=wait_plot_shared_scale,
                kde_scale=wait_kde_shared_scale,
            )
            plot_per_driver_stacked_wait_histogram(
                tag, job_ids, wait_map, procs_map, plots_dir,
                dist=wait_dist_no_s, shared_scale=wait_plot_shared_scale_no_s,
                filename="stacked_wait_time_histogram_no_s.png",
            )
            plot_per_driver_stream_wait_area(
                tag, job_ids, wait_map, procs_map, plots_dir,
                dist=wait_dist, shared_scale=wait_plot_shared_scale,
            )
            plot_per_driver_stream_wait_area(
                tag, job_ids, wait_map, procs_map, plots_dir,
                dist=wait_dist_no_s, shared_scale=wait_plot_shared_scale_no_s,
                filename="stream_wait_time_area_no_s.png",
            )
            plot_per_driver_stacked_wait_area(
                tag, job_ids, wait_map, procs_map, plots_dir,
                dist=wait_dist, shared_scale=wait_plot_shared_scale,
            )
            plot_per_driver_stacked_wait_area(
                tag, job_ids, wait_map, procs_map, plots_dir,
                dist=wait_dist_no_s, shared_scale=wait_plot_shared_scale_no_s,
                filename="stacked_wait_time_area_no_s.png",
            )
        plot_per_driver_wait_vs_size_scatter(
            tag, job_ids, wait_map, procs_map, plots_dir,
            driver_cfg=tag_to_driver_cfg.get(tag, {}),
            backfill_ids=backfill_ids,
        )

        all_job_ids_map[tag]   = job_ids
        all_wait_maps[tag]     = wait_map
        all_backfill_maps[tag] = backfill_ids

        waits_by_g = {g: [] for g in GROUP_LABELS}
        bsld_by_g  = {g: [] for g in GROUP_LABELS}
        waits_all  = []
        bsld_all   = []
        bsld_map   = {}
        for jid in job_ids:
            w_h = wait_map[jid] / 3600.0
            waits_all.append(w_h)
            walltime = max(walltimes_map.get(jid, 1), 1)
            bsld_val = (wait_map[jid] + walltime) / walltime
            bsld_all.append(bsld_val)
            bsld_map[jid] = bsld_val

            if jid not in procs_map:
                continue
            g = assign_group(procs_map[jid])
            if g is None:
                continue
            waits_by_g[g].append(w_h)
            bsld_by_g[g].append(bsld_val)

        all_wait_by_group[tag] = waits_by_g
        all_bsld_by_group[tag] = bsld_by_g
        all_wait[tag] = waits_all
        all_bsld[tag] = bsld_all
        all_bsld_maps[tag] = bsld_map
        print()

    # ── Output subdirectories ─────────────────────────────────────────────────
    wait_dir = os.path.join(plots_dir, "wait")
    bsld_dir = os.path.join(plots_dir, "bsld")
    util_dir = os.path.join(plots_dir, "util")
    drain_dir = os.path.join(plots_dir, "drain")
    for _d in (wait_dir, bsld_dir, util_dir, drain_dir):
        os.makedirs(_d, exist_ok=True)

    baseline_tag = next((t for t in valid_tags if t.upper().startswith('WFP')), None)
    wait_impr_tags, wait_improvement_map, wait_improvement_by_group = (
        _compute_wait_improvement_per_job(
            valid_tags, all_wait_maps, procs_map, baseline_tag=baseline_tag
        )
    )
    _, bsld_improvement_map, bsld_improvement_by_group = (
        _compute_metric_improvement_per_job(
            valid_tags, all_bsld_maps, procs_map, baseline_tag=baseline_tag
        )
    )

    # ── Global plots ──────────────────────────────────────────────────────────
    print("Global plots:")
    throughput_end_time_limit = min(t_window_end, analysis_end) if partial else analysis_end
    plot_global_throughput_jobs_per_day(
        valid_tags, end_maps, driver_colors, util_dir,
        maintenance=maintenance,
        end_time_limit=throughput_end_time_limit,
    )
    plot_global_throughput_proc_hours_per_day(
        valid_tags, end_maps, driver_colors, procs_map, runtimes_map, util_dir,
        maintenance=maintenance,
        end_time_limit=throughput_end_time_limit,
    )
    plot_global_cumulative_completed_jobs(
        valid_tags, end_maps, driver_colors, util_dir,
        maintenance=maintenance,
        end_time_limit=throughput_end_time_limit,
    )
    preferred_drain_tags = [
        tag for tag in valid_tags
        if tag in ("MARS-CW", "MARS-CU", "MARS-CB")
    ]
    drain_rate_tags = preferred_drain_tags or [
        tag for tag in valid_tags
        if tag_to_driver_cfg.get(tag, {}).get("type") == "mcts"
    ]
    plot_global_mcts_drain_rate(
        drain_rate_tags,
        output_dir,
        driver_colors,
        drain_dir,
        time_start_limit=analysis_start,
        time_end_limit=analysis_end,
    )
    stale_drain_bar = os.path.join(util_dir, "bar_chart_mcts_drain_pct.png")
    if os.path.exists(stale_drain_bar):
        os.remove(stale_drain_bar)
        print(f"  Removed stale: {stale_drain_bar}")

    if "MARS-CW" in raw_submit_maps and "MARS-CW" in raw_start_maps:
        write_driver_drain_analysis(
            "MARS-CW",
            output_dir,
            drain_dir,
            raw_submit_maps["MARS-CW"],
            raw_start_maps["MARS-CW"],
            procs_map,
            walltimes_map,
            time_start_limit=analysis_start,
            time_end_limit=analysis_end,
        )
    _drain_combo_stats = {}
    _drain_pct_cache   = {}
    for drain_tag in ("MARS-CW", "MARS-CU", "MARS-CB"):
        if drain_tag not in valid_tags:
            continue
        slug = tag_slug(drain_tag)
        for stale_name in (
                f"table_{slug}_drain_run_horizon_probabilities.csv",
                f"heatmap_{slug}_drain_run_horizon_probabilities.png"):
            stale_path = os.path.join(drain_dir, stale_name)
            if os.path.exists(stale_path):
                os.remove(stale_path)
                print(f"  Removed stale: {stale_path}")
        drain_sequence_stats = write_driver_drain_run_sequence_table(
            drain_tag,
            output_dir,
            drain_dir,
            procs_map,
            total_group_counts=drain_total_group_counts.get(drain_tag),
            time_start_limit=analysis_start,
            time_end_limit=analysis_end,
        )
        plot_driver_drain_run_sequence_counts(
            drain_sequence_stats,
            drain_dir,
        )
        combo_stats = write_driver_drain_sequence_combos(drain_tag, drain_dir)
        plot_driver_drain_sequence_combos(combo_stats, drain_dir)
        _drain_combo_stats[drain_tag] = combo_stats
        _drain_pct_cache[drain_tag] = _load_drain_pct_for_tag(
            output_dir, drain_tag,
            time_start_limit=analysis_start,
            time_end_limit=analysis_end,
        )
    plot_driver_drain_sequence_combos_combined(
        _drain_combo_stats.get('MARS-CW'),
        _drain_combo_stats.get('MARS-CU'),
        _drain_pct_cache.get('MARS-CW'),
        _drain_pct_cache.get('MARS-CU'),
        drain_dir,
    )
    _drain_details_paths = {
        tag: os.path.join(drain_dir,
                          f"table_{tag_slug(tag)}_drain_run_sequence_details.csv")
        for tag in ('MARS-CW', 'MARS-CU', 'MARS-CB')
        if tag in valid_tags
    }
    plot_drain_lxl_split_violin(
        _drain_details_paths, all_wait_maps, procs_map, driver_colors, drain_dir,
    )
    _drain_overview_tags = [t for t in ('MARS-CW', 'MARS-CU', 'MARS-CB') if t in valid_tags]
    plot_drain_overview_combined(
        _drain_overview_tags,
        _drain_pct_cache,
        _drain_details_paths,
        drain_total_group_counts,
        driver_colors,
        drain_dir,
    )
    plot_global_overall_wait_cdfs(valid_tags, all_wait, driver_colors, wait_dir)
    plot_global_overall_bsld_cdfs(valid_tags, all_bsld, driver_colors, bsld_dir)
    print()

    # ── Global tables ─────────────────────────────────────────────────────────
    print("Global tables:")
    util_window_end = min(t_window_end, analysis_end) if partial else analysis_end

    # ── Wait time tables + bar charts ────────────────────────────────────────
    write_global_overall_wait_table(valid_tags, all_wait, wait_dir)
    plot_relative_improvement_bars(
        valid_tags, all_wait, driver_colors, wait_dir,
        "bar_chart_wait.png", "Wait Time: % improvement over WFP (all jobs)",
        multiplier=60.0)

    write_global_wait_tables(valid_tags, all_wait_by_group, wait_dir)
    plot_relative_improvement_bars_2x2(
        valid_tags, all_wait_by_group, driver_colors, wait_dir,
        "bar_chart_wait_groups.png", "Wait Time",
        multiplier=60.0)
    plot_wait_improvement_violin_overall(
        valid_tags, wait_improvement_map, driver_colors, wait_dir,
        baseline_tag=baseline_tag,
    )
    plot_wait_improvement_box_overall(
        valid_tags, wait_improvement_map, driver_colors, wait_dir,
        baseline_tag=baseline_tag,
    )
    plot_wait_improvement_violin_groups(
        valid_tags, wait_improvement_by_group, driver_colors, wait_dir,
        baseline_tag=baseline_tag,
    )
    plot_wait_improvement_box_groups(
        valid_tags, wait_improvement_by_group, driver_colors, wait_dir,
        baseline_tag=baseline_tag,
    )
    _, wait_delta_map, wait_delta_by_group = _compute_wait_delta_per_job(
        valid_tags, all_wait_maps, procs_map, baseline_tag=baseline_tag,
    )
    plot_wait_delta_cdf_overall(
        valid_tags, wait_delta_map, driver_colors, wait_dir,
        baseline_tag=baseline_tag,
    )
    plot_wait_delta_cdf_groups(
        valid_tags, wait_delta_by_group, driver_colors, wait_dir,
        baseline_tag=baseline_tag,
    )

    # ── BSLD tables + bar charts ─────────────────────────────────────────────
    write_global_overall_bsld_table(valid_tags, all_bsld, bsld_dir)
    plot_relative_improvement_bars(
        valid_tags, all_bsld, driver_colors, bsld_dir,
        "bar_chart_bsld.png", "BSLD: % improvement over WFP (all jobs)")

    write_global_bsld_tables(valid_tags, all_bsld_by_group, bsld_dir)
    plot_relative_improvement_bars_2x2(
        valid_tags, all_bsld_by_group, driver_colors, bsld_dir,
        "bar_chart_bsld_groups.png", "Bounded Slowdown")
    plot_bsld_improvement_violin_overall(
        valid_tags, bsld_improvement_map, driver_colors, bsld_dir,
        baseline_tag=baseline_tag,
    )
    plot_bsld_improvement_box_overall(
        valid_tags, bsld_improvement_map, driver_colors, bsld_dir,
        baseline_tag=baseline_tag,
    )
    plot_bsld_improvement_violin_groups(
        valid_tags, bsld_improvement_by_group, driver_colors, bsld_dir,
        baseline_tag=baseline_tag,
    )
    plot_bsld_improvement_box_groups(
        valid_tags, bsld_improvement_by_group, driver_colors, bsld_dir,
        baseline_tag=baseline_tag,
    )

    # ── Utilization tables ───────────────────────────────────────────────────
    write_global_util_table(
        valid_tags, raw_end_maps, raw_start_maps, raw_submit_maps,
        procs_map, util_dir,
        window_start=analysis_start,
        window_end=util_window_end,
        maintenance=maintenance,
    )
    drain_2day_table = write_drain_util_table(
        valid_tags, raw_end_maps, raw_start_maps, output_dir,
        procs_map, util_dir, CST, drain_days=2,
        window_start=analysis_start, window_end=util_window_end,
    )
    plot_drain_util_bars_from_table(
        drain_2day_table,
        driver_colors,
        util_dir,
        "bar_chart_util_drain_2day.png",
        excluded_tags={"MARS-CB", "MARS-CW", "MARS-IU", "Random"},
    )
    write_drain_queue_announcement_table(
        drain_2day_table,
        output_dir,
        util_dir,
        walltimes_map=walltimes_map,
    )
    write_drain_core_hours_table(
        drain_2day_table,
        output_dir,
        util_dir,
        procs_map=procs_map,
        walltimes_map=walltimes_map,
    )
    plot_drain_queue_before_announcement_bars_from_table(
        os.path.join(util_dir, "table_queue_drain_2day.csv"),
        driver_colors,
        util_dir,
        "bar_chart_queue_before_announcement_drain_2day.png",
        excluded_tags={"MARS-CB", "MARS-CW", "MARS-IU", "Random"},
    )
    plot_drain_queue_boxplots_from_table(
        drain_2day_table,
        output_dir,
        driver_colors,
        util_dir,
        "box_queue_size_drain_2day.png",
        excluded_tags={"MARS-CB", "MARS-CW", "MARS-IU", "Random"},
        walltimes_map=walltimes_map,
    )
    drain_1day_table = write_drain_util_table(
        valid_tags, raw_end_maps, raw_start_maps, output_dir,
        procs_map, util_dir, CST, drain_days=1,
        window_start=analysis_start, window_end=util_window_end,
    )
    plot_drain_util_bars_from_table(
        drain_1day_table,
        driver_colors,
        util_dir,
        "bar_chart_util_drain_1day.png",
        excluded_tags={"MARS-CB", "MARS-CW", "MARS-IU", "Random"},
    )
    write_drain_queue_announcement_table(
        drain_1day_table,
        output_dir,
        util_dir,
        walltimes_map=walltimes_map,
        out_fname="table_queue_drain_1day.csv",
    )
    write_drain_core_hours_table(
        drain_1day_table,
        output_dir,
        util_dir,
        procs_map=procs_map,
        walltimes_map=walltimes_map,
        out_fname="table_core_hours_drain_1day.csv",
    )
    plot_drain_queue_before_announcement_bars_from_table(
        os.path.join(util_dir, "table_queue_drain_1day.csv"),
        driver_colors,
        util_dir,
        "bar_chart_queue_before_announcement_drain_1day.png",
        excluded_tags={"MARS-CB", "MARS-CW", "MARS-IU", "Random"},
    )
    plot_drain_queue_boxplots_from_table(
        drain_1day_table,
        output_dir,
        driver_colors,
        util_dir,
        "box_queue_size_drain_1day.png",
        excluded_tags={"MARS-CB", "MARS-CW", "MARS-IU", "Random"},
        walltimes_map=walltimes_map,
    )
    write_global_util_maintenance_table(
        valid_tags, raw_end_maps, raw_start_maps, output_dir,
        procs_map, util_dir, CST,
        window_start=analysis_start, window_end=util_window_end,
    )

    write_global_tail_heatmap(valid_tags, all_job_ids_map, all_wait_maps,
                              procs_map, walltimes_map, plots_dir)
    print()

    print(f"All plots saved to: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
