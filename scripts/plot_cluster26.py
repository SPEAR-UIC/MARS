#!/usr/bin/env python3
"""plot_cluster26.py — Self-contained final plotting script for MARS paper.

Usage:
    python3 scripts/exp_final.py experiments/exp2a.json experiments/exp2b.json
    python3 scripts/exp_final.py experiments/exp2a.json experiments/exp2b.json \
        --output_dir results/plots_cluster26
"""

import argparse
import bisect
import csv
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────
# Tags excluded from the JSON config (never run)
EXCLUDE_TAGS = {"LRF", "LJF"}
# Tags present in results dirs but excluded from all plots
PLOT_EXCLUDED = {"MARS-CB", "MARS-IU"}

GROUPS = ["S", "M", "L", "XL"]
DPI = 150
CST = timezone(timedelta(hours=-6))
MCTS_NUM_CORES = 250

# Requested-walltime buckets (seconds, cumulative upper edge) for the job
# distribution heatmap.
WT_EDGES  = [1800, 3600, 7200, 10800, 18000, 21600, 32400, 86400]
WT_LABELS = ["≤0.5h", "≤1h", "≤2h", "≤3h",
             "≤5h", "≤6h", "≤9h", "≤24h"]

# Node-count ranges (inclusive) that back both assign_group() and the job
# distribution heatmap's row groups, keyed by system.
NODE_GROUP_RANGES = {
    "theta":   [("S", 128, 128), ("M", 129, 256), ("L", 257, 1024), ("XL", 1025, 4096)],
    "polaris": [("S", 10, 16),   ("M", 17, 32),   ("L", 33, 128),   ("XL", 129, 496)],
}

MARS_COLORS    = {"CW": "#f6de6a", "CB": "#b8860b", "CU": "#9467bd", "IU": "#c5b0d5"}
HEURISTIC_COLORS = {
    "FCFS": "#6e6e6e", "SJF": "#1f77b4", "WFP": "#aec7e8", "WFP3": "#aec7e8",
    "F1": "#ff7f0e", "RLSCHEDULER": "#d62728", "RANDOM": "#2ca02c",
    "LJF": "#17becf", "LRF": "#98df8a",
}
# Tags included in backfill fraction plot (heuristics only)
_BF_FRACTION_TAGS = {"FCFS", "WFP", "WFP3", "F1", "SJF", "LJF", "LRF"}
_BF_FRACTION_ORDER = ["FCFS", "WFP", "WFP3", "F1", "SJF", "LJF", "LRF"]
# Canonical display order (matches exp2ab_plot.py _order_all_tags)
_TAG_ORDER = ["RLSCHEDULER", "RANDOM", "WFP3", "WFP", "MARS-CW", "MARS-CU",
              "FCFS", "SJF", "F1"]

# ─── Tag helpers ──────────────────────────────────────────────────────────────
def display_tag(tag):
    return re.sub(r"-w\d+$", "", str(tag), flags=re.IGNORECASE)

def get_color(tag):
    u = tag.upper()
    if u.startswith("MARS"):
        for s in ("CW", "CB", "CU", "IU"):
            if u.endswith(s):
                return MARS_COLORS[s]
        return MARS_COLORS["CW"]
    for pfx, c in HEURISTIC_COLORS.items():
        if u.startswith(pfx):
            return c
    return "#999999"

def get_hatch(tag):
    u = tag.upper()
    if u.endswith("-CW"): return "///"
    if u.endswith("-CB"): return "..."
    if u.endswith("-CU"): return "---"
    if u.endswith("-IU"): return "xxx"
    return None

def is_plot_excluded(tag):
    u = tag.upper()
    return any(u.startswith(e.upper()) for e in PLOT_EXCLUDED)

def sort_tags(tags):
    base = [t.upper() for t in _TAG_ORDER]
    def key(t):
        d = display_tag(t).upper()
        try:
            return base.index(d)
        except ValueError:
            return len(base)
    return sorted(tags, key=key)

def build_driver_colors(tags):
    return {t: get_color(t) for t in tags}

# ─── Parsers ──────────────────────────────────────────────────────────────────
def parse_swf(path):
    procs_map, walltimes_map, submit_map, runtimes_map = {}, {}, {}, {}
    unix_start = 0
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(";"):
                if "UnixStartTime:" in line:
                    try:
                        unix_start = int(line.split("UnixStartTime:")[1].strip())
                    except (IndexError, ValueError):
                        pass
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                jid   = int(parts[0])
                sub_t = int(parts[1])
                run_t = int(parts[3])
                req_p = int(parts[7])
                req_w = int(parts[8])
            except (ValueError, IndexError):
                continue
            procs_map[jid]     = req_p
            walltimes_map[jid] = req_w
            submit_map[jid]    = unix_start + sub_t
            runtimes_map[jid]  = run_t
    return procs_map, walltimes_map, submit_map, runtimes_map

_MAINT_EVT = frozenset({
    "SMA", "SMS", "SME", "UMS", "UME",
    "MaintenanceAnnounced", "MaintenanceStart", "MaintenanceEnd",
    "ScheduledMaintenanceAnnounced", "ScheduledMaintenanceStart",
    "ScheduledMaintenanceEnd", "UnscheduledMaintenanceStart",
    "UnscheduledMaintenanceEnd",
})

def parse_events(path):
    submit, start, end = {}, {}, {}
    if not os.path.exists(path):
        return submit, start, end
    with open(path) as f:
        for row in csv.DictReader(f):
            event = row.get("event", "")
            if event in _MAINT_EVT:
                continue
            try:
                jid = int(row["id"])
                t   = float(row["sim_time"])
            except (KeyError, ValueError, TypeError):
                continue
            if event == "Submit":
                if jid not in submit:
                    submit[jid] = t
            elif event in ("Run", "Backfill"):
                start[jid] = t
            elif event == "End":
                end[jid] = t
    return submit, start, end

def parse_events_with_start_types(path):
    """Parse events.csv and preserve whether each job started via Run or Backfill."""
    submit, start, end, start_type = {}, {}, {}, {}
    if not os.path.exists(path):
        return submit, start, end, start_type
    with open(path) as f:
        for row in csv.DictReader(f):
            event = row.get("event", "")
            if event in _MAINT_EVT:
                continue
            try:
                jid = int(row["id"])
                t   = float(row["sim_time"])
            except (KeyError, ValueError, TypeError):
                continue
            if event == "Submit":
                if jid not in submit:
                    submit[jid] = t
            elif event == "Run":
                start[jid] = t
                start_type[jid] = "Run"
            elif event == "Backfill":
                start[jid] = t
                start_type[jid] = "Backfill"
            elif event == "End":
                end[jid] = t
    return submit, start, end, start_type

def parse_maintenance_from_events(path):
    """Extract maintenance windows (start, end, type) from SMA/SMS/SME events."""
    sms, sme, ums, ume = {}, {}, {}, {}
    if not os.path.exists(path):
        return []
    with open(path) as f:
        for row in csv.DictReader(f):
            event = row.get("event", "")
            if event not in ("SMS", "SME", "UMS", "UME"):
                continue
            try:
                idx = int(row["id"])
                t   = float(row["sim_time"])
            except (KeyError, ValueError, TypeError):
                continue
            if event == "SMS":   sms[idx] = t
            elif event == "SME": sme[idx] = t
            elif event == "UMS": ums[idx] = t
            elif event == "UME": ume[idx] = t
    windows = []
    for idx in sorted(sms):
        if idx in sme:
            windows.append((int(sms[idx]), int(sme[idx]), "S"))
    for idx in sorted(ums):
        if idx in ume:
            windows.append((int(ums[idx]), int(ume[idx]), "U"))
    return sorted(windows, key=lambda w: w[0])

def parse_rl_csv(path):
    submit, start, end = {}, {}, {}
    if not os.path.exists(path):
        return submit, start, end
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                jid   = int(row["job_id"])
                sub_t = float(row["submit_time"])
                sch_t = float(row["scheduled_time"])
                run_t = float(row["run_time"])
            except (KeyError, ValueError, TypeError):
                continue
            if jid not in submit:
                submit[jid] = sub_t
            start[jid] = sch_t
            end[jid]   = sch_t + run_t
    return submit, start, end

# ─── Group assignment ─────────────────────────────────────────────────────────
def assign_group(num_procs, system):
    if system == "theta":
        if num_procs == 128:          return "S"
        if 129 <= num_procs <= 256:   return "M"
        if 257 <= num_procs <= 1024:  return "L"
        if 1025 <= num_procs <= 4096: return "XL"
        return None
    else:
        if 10  <= num_procs <= 16:  return "S"
        if 17  <= num_procs <= 32:  return "M"
        if 33  <= num_procs <= 128: return "L"
        if 129 <= num_procs <= 496: return "XL"
        return None

# ─── Config ───────────────────────────────────────────────────────────────────
def load_config(json_path):
    with open(json_path) as f:
        cfg = json.load(f)
    drivers = [d for d in cfg["drivers"] if d.get("tag", "") not in EXCLUDE_TAGS]
    return cfg, drivers

def detect_system(cfg):
    swf = cfg.get("swf_path", "").lower()
    if "theta" in swf:
        return "theta", 4096, 2021
    elif "polaris" in swf:
        return "polaris", 496, 2024
    raise ValueError(f"Cannot detect system from swf_path: {cfg.get('swf_path')}")

def get_events_for_driver(driver, results_dir, base_dir):
    tag   = driver["tag"]
    dtype = driver.get("type", "")
    if dtype == "rlscheduler":
        csv_path = driver.get("csv_path", "")
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(base_dir, csv_path)
        return parse_rl_csv(csv_path)
    return parse_events(os.path.join(results_dir, tag, "events.csv"))

def first_events_csv(results_dir):
    """Return path to first available events.csv in any tag subdir."""
    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        p = os.path.join(results_dir, entry.name, "events.csv")
        if os.path.exists(p):
            return p
    return None

# ─── Maintenance/uptime helpers ───────────────────────────────────────────────
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

def compute_uptime_periods(maint_windows, year_start, year_end):
    """Return list of (start_ts, end_ts) uptime gaps between scheduled maintenances."""
    sched = [(ms, me) for ms, me, mt in maint_windows if mt == "S"]
    clipped = [(max(ms, year_start), min(me, year_end)) for ms, me in sched]
    clipped = sorted((ms, me) for ms, me in clipped if me > ms)
    merged = []
    for ms, me in clipped:
        if merged and ms <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], me)
        else:
            merged.append([ms, me])
    periods = []
    prev = year_start
    for ms, me in merged:
        if ms > prev:
            periods.append((prev, ms))
        prev = max(prev, me)
    if prev < year_end:
        periods.append((prev, year_end))
    return periods

# ─── Data loaders ────────────────────────────────────────────────────────────
def load_wait(submit, start, procs_map, system):
    """Return {group: [wait_hours > 0], 'all': [...]}"""
    data = {g: [] for g in GROUPS}
    data["all"] = []
    for jid in set(submit) & set(start):
        w = start[jid] - submit[jid]
        if w <= 0:
            continue
        data["all"].append(w / 3600.0)
        g = assign_group(procs_map.get(jid, -1), system)
        if g:
            data[g].append(w / 3600.0)
    return data

def load_bsld(submit, start, end, procs_map, system):
    """Return {group: [bsld > 1], 'all': [...]} using actual runtime from events."""
    data = {g: [] for g in GROUPS}
    data["all"] = []
    for jid in set(submit) & set(start) & set(end):
        w   = start[jid] - submit[jid]
        if w <= 0:
            continue
        run = max(end[jid] - start[jid], 1.0)
        b   = (w + run) / run
        data["all"].append(b)
        g = assign_group(procs_map.get(jid, -1), system)
        if g:
            data[g].append(b)
    return data

def _wt_bucket(req_wall_seconds):
    for i, edge in enumerate(WT_EDGES):
        if req_wall_seconds <= edge:
            return i
    return None

# Only the M size class gets split into a dominant-value row -- matches the
# reference job-distribution figure, where S/L/XL are always single rows
# even when a single node count also dominates within them (e.g. Polaris'
# S class is dominated by 10-node jobs but stays one "≤16" row).
_SPLITTABLE_GROUPS = {"M"}

def load_job_distribution(procs_map, submit_map, walltimes_map, system,
                           year_start, year_end, dominant_frac=0.5):
    """Bucket jobs submitted within [year_start, year_end) into node-count x
    requested-walltime cells for the job-distribution heatmap.

    Node-count rows follow NODE_GROUP_RANGES (the same S/M/L/XL boundaries as
    assign_group). Within the M group (see _SPLITTABLE_GROUPS), a single
    node-count value that accounts for more than `dominant_frac` of that
    group's jobs is split into its own "=N" row, leaving the remaining jobs
    in a "≤N" row -- this mirrors common HPC workloads where one popular job
    size (e.g. a full-node allocation) dominates a size class.

    Returns a dict with:
      "rows": ordered list of {"label", "group", "counts" (len-8 list),
              "total"} -- one entry per heatmap row, top (smallest) to
              bottom (largest).
      "total_jobs": total in-year job count (used for %-of-jobs coloring).
    """
    jids = [j for j in procs_map if year_start <= submit_map.get(j, -1) < year_end]
    total_jobs = len(jids)

    rows = []
    for glabel, lo, hi in NODE_GROUP_RANGES[system]:
        grp_jids = [j for j in jids if lo <= procs_map[j] <= hi]
        if not grp_jids:
            rows.append({"label": f"={lo}" if lo == hi else f"≤{hi}",
                         "group": glabel, "anchor": hi, "counts": [0] * 8, "total": 0})
            continue

        sub_groups = [(lo, hi, grp_jids)]
        if lo != hi and glabel in _SPLITTABLE_GROUPS:
            counts_by_value = Counter(procs_map[j] for j in grp_jids)
            dominant_value, dominant_count = counts_by_value.most_common(1)[0]
            if dominant_count / len(grp_jids) > dominant_frac:
                dominant_jids = [j for j in grp_jids if procs_map[j] == dominant_value]
                rest_jids = [j for j in grp_jids if procs_map[j] != dominant_value]
                rest_hi = dominant_value - 1 if dominant_value == hi else hi
                sub_groups = sorted(
                    [(dominant_value, dominant_value, dominant_jids),
                     (lo, rest_hi, rest_jids)],
                    key=lambda sg: sg[1],
                )

        for sub_lo, sub_hi, sub_jids in sub_groups:
            counts = [0] * 8
            for j in sub_jids:
                b = _wt_bucket(walltimes_map[j])
                if b is not None:
                    counts[b] += 1
            label = f"={sub_lo}" if sub_lo == sub_hi else f"≤{sub_hi}"
            rows.append({"label": label, "group": glabel, "anchor": sub_hi,
                        "counts": counts, "total": len(sub_jids)})

    return {"rows": rows, "total_jobs": total_jobs}

def load_uptime_util_per_period(results_dir, tag, procs_map, maint_windows,
                                 year_start, year_end, sys_size):
    """Return [util_pct per uptime period] computed from events.csv."""
    events_path = os.path.join(results_dir, tag, "events.csv")
    if not os.path.exists(events_path):
        return []
    _, start_map, end_map = parse_events(events_path)
    periods = compute_uptime_periods(maint_windows, year_start, year_end)
    utils = []
    for pstart, pend in periods:
        dur = pend - pstart
        if dur <= 0:
            continue
        used = sum_used_core_seconds(start_map, end_map, procs_map, pstart, pend)
        utils.append(100.0 * used / (dur * sys_size))
    return utils

def maintenance_overlap_seconds(maint_windows, t_start, t_end):
    total = 0.0
    for ms, me, _ in (maint_windows or []):
        ov_s = max(ms, t_start)
        ov_e = min(me, t_end)
        if ov_e > ov_s:
            total += ov_e - ov_s
    return total

def load_rl_overall_util(driver, base_dir, procs_map, maint_windows,
                         year_start, year_end, sys_size):
    """Return overall RL utilization % as a single value, excluding maintenance."""
    csv_path = driver.get("csv_path", "")
    if not csv_path:
        return None
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(base_dir, csv_path)
    _, start_map, end_map = parse_rl_csv(csv_path)
    if not start_map or not end_map:
        return None

    used_full = sum_used_core_seconds(start_map, end_map, procs_map, year_start, year_end)
    used_maint = 0.0
    for ms, me, _ in (maint_windows or []):
        ov_s = max(ms, year_start)
        ov_e = min(me, year_end)
        if ov_e > ov_s:
            used_maint += sum_used_core_seconds(start_map, end_map, procs_map, ov_s, ov_e)

    eff_uptime = (year_end - year_start) - maintenance_overlap_seconds(
        maint_windows, year_start, year_end)
    if eff_uptime <= 0:
        return None
    used_core_s = used_full - used_maint
    return 100.0 * used_core_s / (eff_uptime * sys_size)

def load_drain_period_utils(results_dir, tag, procs_map, maint_windows,
                              sys_size, days):
    """Return [util_pct per scheduled drain period] computed from events.csv."""
    events_path = os.path.join(results_dir, tag, "events.csv")
    if not os.path.exists(events_path):
        return []
    _, start_map, end_map = parse_events(events_path)
    sched = [(ms, me) for ms, me, mt in maint_windows if mt == "S"]
    wsec  = days * 86400
    utils = []
    for ms, _ in sched:
        pstart = ms - wsec
        pend   = ms
        dur    = pend - pstart
        if dur <= 0:
            continue
        used = sum_used_core_seconds(start_map, end_map, procs_map, pstart, pend)
        utils.append(100.0 * used / (dur * sys_size))
    return utils

def load_drain_queue_samples(results_dir, tag, maint_windows, days):
    """Return all queue_len samples during drain windows from performance.csv."""
    perf_path = os.path.join(results_dir, tag, "performance.csv")
    if not os.path.exists(perf_path):
        return []
    sched   = [(ms, me) for ms, me, mt in maint_windows if mt == "S"]
    wsec    = days * 86400
    periods = [(ms - wsec, ms) for ms, _ in sched]
    samples = []
    with open(perf_path) as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["sim_time"])
                q = float(row["queue_len"])
            except (KeyError, ValueError):
                continue
            for ps, pe in periods:
                if ps <= t < pe:
                    samples.append(q)
                    break
    return samples

def load_backfill_4grp(results_dir, tag, procs_map, system):
    events_path = os.path.join(results_dir, tag, "events.csv")
    submit, start, _, start_type = parse_events_with_start_types(events_path)
    data = {(k, g): [] for k in ("backfill", "primary") for g in GROUPS}
    for jid in set(submit) & set(start):
        w = start[jid] - submit[jid]
        if w < 0:
            continue
        g = assign_group(procs_map.get(jid, -1), system)
        if not g:
            continue
        kind = "backfill" if start_type.get(jid) == "Backfill" else "primary"
        data[(kind, g)].append(w / 3600.0)
    return data

def load_backfill_job_fraction(results_dir, tag):
    """Return the fraction of started jobs whose start event is Backfill."""
    events_path = os.path.join(results_dir, tag, "events.csv")
    _, start, _, start_type = parse_events_with_start_types(events_path)
    if not start:
        return None
    total = len(start)
    n_backfill = sum(1 for jid in start if start_type.get(jid) == "Backfill")
    return n_backfill / total if total else None

def load_backfill_overview(results_dir, tag, procs_map, system):
    """Return (backfill_pct_tuple, backfill_group, total_group) for one heuristic tag."""
    events_path = os.path.join(results_dir, tag, "events.csv")
    _, start, _, start_type = parse_events_with_start_types(events_path)

    total_group = {g: 0 for g in GROUPS}
    backfill_group = {g: 0 for g in GROUPS}
    for jid in start:
        g = assign_group(procs_map.get(jid, -1), system)
        if not g:
            continue
        total_group[g] += 1
        if start_type.get(jid) == "Backfill":
            backfill_group[g] += 1

    total_jobs = sum(total_group.values())
    n_backfill = sum(backfill_group.values())
    pct = 100.0 * n_backfill / total_jobs if total_jobs else 0.0
    return (n_backfill, total_jobs, pct), backfill_group, total_group

def _drain_decision_rows(results_dir, tag):
    """Per-cycle (policy, jobs_run, sim_time) rows used to detect DRAIN episodes.

    Only cycles with root_branching_factor >= 2 are considered "real" scheduling
    decisions (rbf < 2 means there was at most one feasible action, so the
    scheduler wasn't actually choosing to drain).
    """
    perf_p = os.path.join(results_dir, tag, "performance.csv")
    dec_p  = os.path.join(results_dir, tag, "descisions.csv")
    if not os.path.exists(perf_p) or not os.path.exists(dec_p):
        return []
    perf = {}
    with open(perf_p) as f:
        for row in csv.DictReader(f):
            try:
                c = int(row["cycle"])
                jobs_run = int(float(row.get("jobs_run", 0) or 0))
                sim_time = float(row["sim_time"])
            except (KeyError, ValueError, TypeError):
                continue
            perf[c] = (jobs_run, sim_time)
    rows = []
    with open(dec_p) as f:
        for row in csv.DictReader(f):
            try:
                c   = int(row["cycle"])
                rbf = int(float(row["root_branching_factor"]))
                pol = row["selected_policy"].strip().upper()
            except (KeyError, ValueError, TypeError):
                continue
            if rbf < 2 or c not in perf:
                continue
            jobs_run, sim_time = perf[c]
            rows.append({"cycle": c, "pol": pol, "jobs_run": jobs_run, "sim_time": sim_time})
    return rows

def _ordered_run_events(results_dir, tag):
    """(sim_time, job_id) for every Run event in events.csv, chronologically
    (ties broken by file order)."""
    events_path = os.path.join(results_dir, tag, "events.csv")
    if not os.path.exists(events_path):
        return []
    records = []
    with open(events_path) as f:
        seq = 0
        for row in csv.DictReader(f):
            if row.get("event") != "Run":
                continue
            try:
                sim_time = float(row["sim_time"])
                jid = int(row["id"])
            except (KeyError, ValueError):
                continue
            records.append((sim_time, seq, jid))
            seq += 1
    records.sort(key=lambda item: (item[0], item[1]))
    return [(t, jid) for t, _, jid in records]

def _collapse_drain_rows_by_run_events(drain_rows, run_events):
    """Collapse consecutive DRAIN decision cycles into one episode, unless a
    job actually started running somewhere in between them -- in which case
    they're distinct drain episodes rather than one continuous stall."""
    if not drain_rows:
        return []
    if not run_events:
        return list(drain_rows)
    run_times = [t for t, _ in run_events]
    collapsed = []
    current = drain_rows[0]
    for row in drain_rows[1:]:
        lo = bisect.bisect_right(run_times, current["sim_time"])
        hi = bisect.bisect_right(run_times, row["sim_time"])
        if hi > lo:
            collapsed.append(current)
        current = row
    collapsed.append(current)
    return collapsed

def compute_drain_episodes(results_dir, tag):
    """Detect MARS DRAIN episodes directly from descisions.csv + performance.csv.

    DRAIN cycles never run a job themselves (jobs_run is always 0 while the
    scheduler is holding the queue), so an episode can't be detected by
    watching for jobs_run > 0 on a DRAIN row. Instead, consecutive
    DRAIN-policy decision cycles (root_branching_factor >= 2) collapse into
    one episode unless an actual Run event happened somewhere between them --
    that Run event means the drain broke and a *new* stall is a separate
    episode. This mirrors the deleted scripts/exp2_plot.py methodology
    (write_driver_drain_run_sequence_table / _collapse_drain_rows_by_run_events).

    Returns (rows, episodes): `rows` is every rbf>=2 decision cycle (the
    denominator for drain rate), `episodes` is the list of collapsed DRAIN
    rows (each the anchor cycle/sim_time for one drain episode).
    """
    rows = _drain_decision_rows(results_dir, tag)
    raw_drain_rows = [r for r in rows if r["pol"] == "DRAIN"]
    if not raw_drain_rows:
        return rows, []
    run_events = _ordered_run_events(results_dir, tag)
    episodes = _collapse_drain_rows_by_run_events(raw_drain_rows, run_events)
    return rows, episodes

def load_drain_pct(results_dir, tag):
    rows, episodes = compute_drain_episodes(results_dir, tag)
    total = len(rows)
    drain_ep = len(episodes)
    pct = 100.0 * drain_ep / total if total else 0.0
    return drain_ep, total, pct

def load_drain_run_sequences(results_dir, tag, procs_map, system, max_jobs=3):
    """For each drain episode, the first `max_jobs` jobs that started running
    (chronologically) after it -- the jobs whose dispatch the drain was
    holding back, tagged with their size-class group.

    This is the direct replacement for the old precomputed
    table_*_drain_run_sequence_details.csv, built straight from events.csv
    Run events instead of a missing intermediate table. Returns
    (episode_jobs, total_episodes) where episode_jobs is a list of
    {"jobs": [(jid, group), ...]} for episodes with >=1 recognized job.
    """
    _, episodes = compute_drain_episodes(results_dir, tag)
    if not episodes:
        return [], 0
    run_events = _ordered_run_events(results_dir, tag)
    if not run_events:
        return [], len(episodes)

    episode_times = [e["sim_time"] for e in episodes]
    assignments = [[] for _ in episodes]
    for run_time, jid in run_events:
        idx = bisect.bisect_right(episode_times, run_time - 1e-9) - 1
        if idx < 0 or len(assignments[idx]) >= max_jobs:
            continue
        g = assign_group(procs_map.get(jid, -1), system)
        if g is None:
            continue
        assignments[idx].append((jid, g))

    out = [{"jobs": jobs} for jobs in assignments if jobs]
    return out, len(episodes)

def load_mars_drain_4grp(results_dir, tag, procs_map, system):
    """Return {('drain'|'nondrain', grp): [wait_hours]}.

    A job counts as 'drain' if it was one of the (up to 3) jobs dispatched at
    the cycle a DRAIN episode resolved (see load_drain_run_sequences).
    """
    submit, start, _ = parse_events(os.path.join(results_dir, tag, "events.csv"))
    sequences, _ = load_drain_run_sequences(results_dir, tag, procs_map, system)
    drain_ids = {jid for seq in sequences for jid, _ in seq["jobs"]}
    data = {(k, g): [] for k in ("drain", "nondrain") for g in GROUPS}
    for jid in set(submit) & set(start):
        w = start[jid] - submit[jid]
        if w < 0:
            continue
        g = assign_group(procs_map.get(jid, -1), system)
        if not g:
            continue
        kind = "drain" if jid in drain_ids else "nondrain"
        data[(kind, g)].append(w / 3600.0)
    return data

def load_drain_combo_stats(results_dir, tag, procs_map, system):
    """Sequence-combo stats built directly from resolved drain episodes.

    Only episodes where >=3 jobs were dispatched at the resolving cycle
    contribute a full "G1-G2-G3" combo (fewer than 3 can't form a triple);
    `total_drain_episodes` still counts every episode that resolved with at
    least one identified job, matching the old table's semantics.
    """
    sequences, _ = load_drain_run_sequences(results_dir, tag, procs_map, system)
    if not sequences:
        return None, None
    seq_counts = {}
    total = len(sequences)
    for seq in sequences:
        jobs = seq["jobs"]
        if len(jobs) < 3:
            continue
        label = "-".join(g for _, g in jobs)
        seq_counts[label] = seq_counts.get(label, 0) + 1
    if not seq_counts:
        return None, None
    stats = {"tag": tag, "seq_counts": seq_counts, "total_drain_episodes": total}
    return stats, load_drain_pct(results_dir, tag)

def load_drain_overview(results_dir, drain_tags, procs_map, system):
    drain_pct, total_grp = {}, {}
    for tag in drain_tags:
        drain_pct[tag] = load_drain_pct(results_dir, tag)
        ep = os.path.join(results_dir, tag, "events.csv")
        tg = {g: 0 for g in GROUPS}
        if os.path.exists(ep):
            _, st, _ = parse_events(ep)
            for jid in st:
                g = assign_group(procs_map.get(jid, -1), system)
                if g:
                    tg[g] += 1
        total_grp[tag] = tg
    postdrain = {tag: {g: 0 for g in GROUPS} for tag in drain_tags}
    for tag in drain_tags:
        sequences, _ = load_drain_run_sequences(results_dir, tag, procs_map, system)
        seen = set()
        for seq in sequences:
            for jid, grp in seq["jobs"]:
                if (jid, grp) not in seen:
                    seen.add((jid, grp))
                    postdrain[tag][grp] += 1
    return drain_pct, postdrain, total_grp

def load_mixed_backfill_drain_overview(results_dir, tags, procs_map, system):
    """Return overview data mixing backfill metrics and MARS drain metrics by tag."""
    tags = list(tags)
    drain_tags = [t for t in tags if display_tag(t).upper().startswith("MARS-C")]
    backfill_tags = [t for t in tags if t not in drain_tags]

    rate_by_tag = {}
    affected_by_tag = {}
    total_by_tag = {}

    if backfill_tags:
        for tag in backfill_tags:
            pct_tup, affected_grp, total_grp = load_backfill_overview(
                results_dir, tag, procs_map, system)
            rate_by_tag[tag] = pct_tup
            affected_by_tag[tag] = affected_grp
            total_by_tag[tag] = total_grp

    if drain_tags:
        drain_pct, postdrain, total_grp = load_drain_overview(
            results_dir, drain_tags, procs_map, system)
        for tag in drain_tags:
            rate_by_tag[tag] = drain_pct.get(tag, (0, 0, 0.0))
            affected_by_tag[tag] = postdrain.get(tag, {g: 0 for g in GROUPS})
            total_by_tag[tag] = total_grp.get(tag, {g: 0 for g in GROUPS})

    return rate_by_tag, affected_by_tag, total_by_tag

def load_drain_resume_times(results_dir, tag, maint_windows):
    """Compute drain→first-run gap directly from performance.csv + events.csv."""
    perf_path = os.path.join(results_dir, tag, "performance.csv")
    evt_path  = os.path.join(results_dir, tag, "events.csv")
    if not os.path.exists(perf_path) or not os.path.exists(evt_path):
        return []

    # Build set of maintenance intervals (to skip drains inside maintenance)
    maint_set = [(ms, me) for ms, me, _ in maint_windows]

    def in_maint(t):
        return any(ms <= t < me for ms, me in maint_set)

    perf_rows = {}
    with open(perf_path) as f:
        for row in csv.DictReader(f):
            try:
                c   = int(row["cycle"])
                pol = row["selected_policy"].strip().upper()
                t   = int(float(row["sim_time"]))
            except (KeyError, ValueError, TypeError):
                continue
            perf_rows[c] = (pol, t)

    run_events = []
    with open(evt_path) as f:
        for row in csv.DictReader(f):
            if row.get("event") == "Run":
                try:
                    run_events.append(int(float(row["sim_time"])))
                except (KeyError, ValueError):
                    pass
    run_events.sort()

    times = []
    in_drain = False
    first_drain_t = None
    for c in sorted(perf_rows):
        pol, sim_t = perf_rows[c]
        if pol == "DRAIN":
            if not in_drain:
                first_drain_t = sim_t
                in_drain = True
        elif pol in ("FCFS", "MCTS"):
            if in_drain and first_drain_t is not None:
                if not in_maint(first_drain_t):
                    idx = bisect.bisect_left(run_events, sim_t)
                    if idx < len(run_events):
                        gap = run_events[idx] - first_drain_t
                        times.append(gap / 60.0)
            in_drain = False
            first_drain_t = None
        else:
            in_drain = False
            first_drain_t = None
    return times

def load_mcts_perf_data(results_dir, tags=("MARS-CW", "MARS-CU")):
    """Return raw per-tag MCTS iterations and branching factors for rows with bf > 1."""
    data = {}
    for tag in tags:
        perf_p = os.path.join(results_dir, tag, "performance.csv")
        dec_p  = os.path.join(results_dir, tag, "descisions.csv")
        if not os.path.exists(perf_p):
            continue

        bf_map = {}
        if os.path.exists(dec_p):
            with open(dec_p) as f:
                for row in csv.DictReader(f):
                    try:
                        c = int(row["cycle"])
                        bf = int(float(row.get("root_branching_factor", 0) or 0))
                    except (KeyError, ValueError, TypeError):
                        continue
                    bf_map[c] = bf

        iterations = []
        branching = []
        with open(perf_p) as f:
            for row in csv.DictReader(f):
                try:
                    c = int(row["cycle"])
                    total = int(float(row.get("mcts_iterations", 0) or 0))
                except (KeyError, ValueError, TypeError):
                    continue
                bf = bf_map.get(c, 0)
                if total > 0 and bf > 1:
                    iterations.append(total)
                    branching.append(bf)
        data[tag] = {
            "iterations": iterations,
            "branching": branching,
        }
    return data

def find_wfp_baseline_tag(tags):
    """Prefer WFP3 when present, otherwise fall back to WFP."""
    tags = list(tags)
    for prefix in ("WFP3", "WFP"):
        for tag in tags:
            if tag.upper().startswith(prefix):
                return tag
    return None

def compute_overall_metric_stats(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    return {
        "Count": int(arr.size),
        "P25": float(np.percentile(arr, 25)),
        "P50": float(np.percentile(arr, 50)),
        "P75": float(np.percentile(arr, 75)),
        "Mean": float(arr.mean()),
    }

def compute_relative_improvement_pct(baseline_value, value, lower_is_better):
    if baseline_value == 0:
        return 0.0 if value == 0 else None
    if lower_is_better:
        return (baseline_value - value) / abs(baseline_value) * 100.0
    return (value - baseline_value) / abs(baseline_value) * 100.0

_WAIT_STAT_HEADERS = ["Min", "P25", "P50", "P75", "P85", "P95", "P99", "Max",
                      "Mean", "StdDev", "GeometricMean"]

def _wait_stats_row(values_minutes):
    arr = np.array(values_minutes)
    pos = arr[arr > 0]
    gmean = float(np.exp(np.mean(np.log(pos)))) if pos.size > 0 else 0.0
    return [
        arr.min(), np.percentile(arr, 25), np.percentile(arr, 50),
        np.percentile(arr, 75), np.percentile(arr, 85), np.percentile(arr, 95),
        np.percentile(arr, 99), arr.max(), arr.mean(), arr.std(), gmean,
    ]

def write_wait_stat_table(out_path, rows):
    """Wait-time distribution table (values in minutes), one row per
    (system, driver). `rows` is a list of (system_label, tag, wait_hours_list).
    Matches the original CLUSTER'26 table_wait.csv / table_wait_{group}.csv
    format from the (now-deleted) exp2_plot.py."""
    d = os.path.dirname(out_path)
    os.makedirs(d if d else ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["System", "Driver"] + _WAIT_STAT_HEADERS)
        for system_label, tag, wait_hours in rows:
            if not wait_hours:
                writer.writerow([system_label, tag] + ["N/A"] * len(_WAIT_STAT_HEADERS))
                continue
            values_minutes = [w * 60.0 for w in wait_hours]
            writer.writerow(
                [system_label, tag] +
                [f"{v:.4f}" for v in _wait_stats_row(values_minutes)]
            )
    print(f"  Saved: {out_path}")

def write_wait_pct_change_table(out_path, rows, baseline_tag, compare_tag):
    """% change in each wait-time stat for `compare_tag` vs `baseline_tag`,
    one row per (system, job class). `rows` is a list of
    (system_label, job_class, baseline_wait_hours, compare_wait_hours).

    Positive = compare_tag is faster (lower wait) than baseline_tag, matching
    compute_relative_improvement_pct's convention for a lower_is_better metric.
    """
    d = os.path.dirname(out_path)
    os.makedirs(d if d else ".", exist_ok=True)
    header = (["System", "Job_Class", f"N_{baseline_tag}", f"N_{compare_tag}"] +
              [f"{h}_Change_vs_{baseline_tag}_Pct" for h in _WAIT_STAT_HEADERS])
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for system_label, job_class, baseline_hours, compare_hours in rows:
            n_base, n_cmp = len(baseline_hours), len(compare_hours)
            if not baseline_hours or not compare_hours:
                writer.writerow(
                    [system_label, job_class, n_base, n_cmp] +
                    ["N/A"] * len(_WAIT_STAT_HEADERS)
                )
                continue
            base_stats = _wait_stats_row([w * 60.0 for w in baseline_hours])
            cmp_stats  = _wait_stats_row([w * 60.0 for w in compare_hours])
            pct = [
                compute_relative_improvement_pct(b, c, lower_is_better=True)
                for b, c in zip(base_stats, cmp_stats)
            ]
            writer.writerow(
                [system_label, job_class, n_base, n_cmp] +
                [f"{v:.2f}" if v is not None else "N/A" for v in pct]
            )
    print(f"  Saved: {out_path}")

def write_overall_metrics_quartiles_table(out_path, metric_rows):
    """Write quartiles and % change vs WFP/WFP3 baseline for overall metrics."""
    headers = [
        "System",
        "Metric",
        "Baseline_Tag",
        "Driver",
        "Raw_Tag",
        "Direction",
        "Count",
        "P25",
        "P50",
        "P75",
        "Mean",
        "P25_Change_vs_Baseline_Pct",
        "P50_Change_vs_Baseline_Pct",
        "P75_Change_vs_Baseline_Pct",
        "Mean_Change_vs_Baseline_Pct",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for metric_row in metric_rows:
            tags = metric_row["tags"]
            data_map = metric_row["data"]
            baseline_tag = find_wfp_baseline_tag(tags)
            baseline_stats = compute_overall_metric_stats(data_map.get(baseline_tag, [])) \
                if baseline_tag else None
            for tag in tags:
                stats = compute_overall_metric_stats(data_map.get(tag, []))
                row = {
                    "System": metric_row["system"],
                    "Metric": metric_row["metric"],
                    "Baseline_Tag": baseline_tag or "N/A",
                    "Driver": display_tag(tag),
                    "Raw_Tag": tag,
                    "Direction": "lower_is_better" if metric_row["lower_is_better"]
                                 else "higher_is_better",
                }
                if stats is None:
                    row.update({h: "N/A" for h in headers if h not in row})
                    writer.writerow(row)
                    continue
                row.update({
                    "Count": stats["Count"],
                    "P25": f"{stats['P25']:.4f}",
                    "P50": f"{stats['P50']:.4f}",
                    "P75": f"{stats['P75']:.4f}",
                    "Mean": f"{stats['Mean']:.4f}",
                })
                for stat_name, col_name in (
                        ("P25", "P25_Change_vs_Baseline_Pct"),
                        ("P50", "P50_Change_vs_Baseline_Pct"),
                        ("P75", "P75_Change_vs_Baseline_Pct"),
                        ("Mean", "Mean_Change_vs_Baseline_Pct"),
                ):
                    if baseline_stats is None:
                        row[col_name] = "N/A"
                    else:
                        delta = compute_relative_improvement_pct(
                            baseline_stats[stat_name], stats[stat_name],
                            metric_row["lower_is_better"],
                        )
                        row[col_name] = "N/A" if delta is None else f"{delta:.2f}"
                writer.writerow(row)
    print(f"  Saved: {out_path}")

def write_grouped_metrics_quartiles_table(out_path, metric_rows):
    """Write quartiles and % change vs WFP/WFP3 baseline for grouped metrics."""
    headers = [
        "System",
        "Metric",
        "Job_Class",
        "Baseline_Tag",
        "Driver",
        "Raw_Tag",
        "Direction",
        "Count",
        "P25",
        "P50",
        "P75",
        "Mean",
        "P25_Change_vs_Baseline_Pct",
        "P50_Change_vs_Baseline_Pct",
        "P75_Change_vs_Baseline_Pct",
        "Mean_Change_vs_Baseline_Pct",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for metric_row in metric_rows:
            tags = metric_row["tags"]
            data_map = metric_row["data"]
            baseline_tag = find_wfp_baseline_tag(tags)
            baseline_stats = compute_overall_metric_stats(data_map.get(baseline_tag, [])) \
                if baseline_tag else None
            for tag in tags:
                stats = compute_overall_metric_stats(data_map.get(tag, []))
                row = {
                    "System": metric_row["system"],
                    "Metric": metric_row["metric"],
                    "Job_Class": metric_row["job_class"],
                    "Baseline_Tag": baseline_tag or "N/A",
                    "Driver": display_tag(tag),
                    "Raw_Tag": tag,
                    "Direction": "lower_is_better" if metric_row["lower_is_better"]
                                 else "higher_is_better",
                }
                if stats is None:
                    row.update({h: "N/A" for h in headers if h not in row})
                    writer.writerow(row)
                    continue
                row.update({
                    "Count": stats["Count"],
                    "P25": f"{stats['P25']:.4f}",
                    "P50": f"{stats['P50']:.4f}",
                    "P75": f"{stats['P75']:.4f}",
                    "Mean": f"{stats['Mean']:.4f}",
                })
                for stat_name, col_name in (
                        ("P25", "P25_Change_vs_Baseline_Pct"),
                        ("P50", "P50_Change_vs_Baseline_Pct"),
                        ("P75", "P75_Change_vs_Baseline_Pct"),
                        ("Mean", "Mean_Change_vs_Baseline_Pct"),
                ):
                    if baseline_stats is None:
                        row[col_name] = "N/A"
                    else:
                        delta = compute_relative_improvement_pct(
                            baseline_stats[stat_name], stats[stat_name],
                            metric_row["lower_is_better"],
                        )
                        row[col_name] = "N/A" if delta is None else f"{delta:.2f}"
                writer.writerow(row)
    print(f"  Saved: {out_path}")

# ─── Wildcard sequence builder ────────────────────────────────────────────────
def build_wildcard_entries(seq_counts, max_entries=5):
    g1_tot, g1g2_tot = {}, {}
    for seq, cnt in seq_counts.items():
        p = seq.split("-")
        if len(p) != 3:
            continue
        g1, g2 = p[0], p[1]
        g1_tot[g1]          = g1_tot.get(g1, 0) + cnt
        g1g2_tot[(g1, g2)]  = g1g2_tot.get((g1, g2), 0) + cnt
    entries = {}
    for g1 in ("L", "XL"):
        if g1_tot.get(g1, 0) > 0:
            entries[f"{g1}-*-*"] = g1_tot[g1]
    for g1 in ("S", "M"):
        for g2 in GROUPS:
            cnt = g1g2_tot.get((g1, g2), 0)
            if cnt > 0:
                entries[f"{g1}-{g2}-*"] = cnt
    while len(entries) < max_entries:
        expandable = sorted(
            [(k, v) for k, v in entries.items()
             if k.endswith("-*") and not k.startswith("L-") and not k.startswith("XL-")],
            key=lambda x: x[1], reverse=True,
        )
        if not expandable:
            break
        label, _ = expandable[0]
        del entries[label]
        g1, g2 = label.split("-")[:2]
        added = 0
        for g3 in GROUPS:
            cnt = seq_counts.get(f"{g1}-{g2}-{g3}", 0)
            if cnt > 0:
                entries[f"{g1}-{g2}-{g3}"] = cnt
                added += 1
        if added == 0:
            break
    return sorted(entries.items(), key=lambda x: x[1], reverse=True)[:max_entries]

# ─── Plot helpers ─────────────────────────────────────────────────────────────
def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: {path}")

def _bold_ax(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.tick_params(labelsize=12)

def _draw_boxplot_panel(ax, tags, data_by_tag, driver_colors,
                         title, show_ylabel, log_scale=True,
                         ymin=None, ymax=None, whis=(5, 95),
                         show_xticklabels=True, xtick_fontsize=12,
                         xtick_rotation=25, title_fontsize=13,
                         box_width=0.58, use_hatches=True,
                         fixed_facecolor=None, box_positions=None,
                         xlim_padding=0.6):
    """Vertical boxplot panel (x=tags, y=metric)."""
    data       = [np.asarray(data_by_tag.get(t, [0.001]), dtype=float) for t in tags]
    if box_positions is None:
        positions = np.arange(1, len(tags) + 1, dtype=float)
    else:
        positions = np.asarray(box_positions, dtype=float)
    parts = ax.boxplot(
        data, positions=positions, widths=box_width,
        whis=whis, showmeans=True, meanline=True, showfliers=True,
        flierprops=dict(marker="o", markersize=2.0, linestyle="none",
                        alpha=0.35, markeredgewidth=0.0),
        patch_artist=True,
        boxprops=dict(linewidth=1.9, edgecolor="black"),
        whiskerprops=dict(linewidth=1.3, color="black"),
        capprops=dict(linewidth=1.3, color="black"),
        medianprops=dict(linewidth=2.8, color="black"),
        meanprops=dict(linewidth=2.0, color="red", linestyle=":"),
    )
    for box, flier, tag in zip(parts["boxes"], parts["fliers"], tags):
        color = fixed_facecolor if fixed_facecolor is not None else driver_colors.get(tag, "gray")
        box.set_facecolor(color)
        box.set_alpha(0.82)
        hatch = get_hatch(tag) if use_hatches else None
        if hatch:
            box.set_hatch(hatch)
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.1f}")
        )
    ax.tick_params(axis="y", labelsize=13, which="both")
    for lbl in ax.get_yticklabels(which="both"):
        lbl.set_fontweight("bold")
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)

    ax.set_xticks(positions)
    if show_xticklabels:
        xlbls = ax.set_xticklabels(
            [display_tag(t) for t in tags],
            fontsize=xtick_fontsize,
            rotation=xtick_rotation,
            ha="right",
        )
        for lbl in xlbls:
            lbl.set_fontweight("bold")
    else:
        ax.tick_params(axis="x", labelbottom=False)

    if ymin is not None or ymax is not None:
        ax.set_ylim(
            ymin if ymin is not None else ax.get_ylim()[0],
            ymax if ymax is not None else ax.get_ylim()[1],
        )
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=5)
    ax.grid(True, axis="y", which="major", alpha=0.3, zorder=0)
    ax.set_xlim(float(np.min(positions) - xlim_padding), float(np.max(positions) + xlim_padding))

def _draw_boxplot_panel_h(ax, tags, data_by_tag, driver_colors,
                          title, show_ylabels=True, log_scale=True,
                          xmin=None, xmax=None, whis=(5, 95)):
    """Horizontal boxplot panel (y=tags, x=metric)."""
    data      = [np.asarray(data_by_tag.get(t, [0.001]), dtype=float) for t in tags]
    positions = np.arange(1, len(tags) + 1)
    parts = ax.boxplot(
        data, positions=positions, widths=0.58, orientation="horizontal",
        whis=whis, showmeans=True, meanline=True, showfliers=True,
        flierprops=dict(marker="o", markersize=2.0, linestyle="none",
                        alpha=0.35, markeredgewidth=0.0),
        patch_artist=True,
        boxprops=dict(linewidth=1.9, edgecolor="black"),
        whiskerprops=dict(linewidth=1.3, color="black"),
        capprops=dict(linewidth=1.3, color="black"),
        medianprops=dict(linewidth=2.8, color="black"),
        meanprops=dict(linewidth=2.0, color="red", linestyle=":"),
    )
    for box, flier, tag in zip(parts["boxes"], parts["fliers"], tags):
        color = driver_colors.get(tag, "gray")
        box.set_facecolor(color)
        box.set_alpha(0.82)
        hatch = get_hatch(tag)
        if hatch:
            box.set_hatch(hatch)
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    if log_scale:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.1f}")
        )
    ax.tick_params(axis="x", labelsize=12, which="both")
    for lbl in ax.get_xticklabels(which="both"):
        lbl.set_fontweight("bold")

    ax.set_yticks(positions)
    if show_ylabels:
        ylbls = ax.set_yticklabels([display_tag(t) for t in tags], fontsize=12)
        for lbl in ylbls:
            lbl.set_fontweight("bold")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", length=0)

    if xmin is not None or xmax is not None:
        ax.set_xlim(
            xmin if xmin is not None else ax.get_xlim()[0],
            xmax if xmax is not None else ax.get_xlim()[1],
        )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=5)
    ax.grid(True, axis="x", which="major", alpha=0.3, zorder=0)
    ax.set_ylim(len(tags) + 0.6, 0.4)


# ─── wait/ figures ────────────────────────────────────────────────────────────
def plot_wait_boxplot(theta_data, polaris_data, theta_tags, polaris_tags, out_path):
    all_tags      = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = build_driver_colors(all_tags)

    n_tags = max(len(theta_tags), len(polaris_tags))
    fig_w  = max(7.0, n_tags * 0.72 + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 4.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.22, wspace=0.06)

    all_vals = [v for d in (theta_data, polaris_data) for vals in d.values() for v in vals]
    if all_vals:
        ymin = max(0.01, np.percentile(all_vals, 1) * 0.5)
        ymax = max(np.percentile(all_vals, 99) * 2.0, 1000.0)
    else:
        ymin, ymax = None, None

    _draw_boxplot_panel(axes[0], theta_tags,   theta_data,   driver_colors,
                        "Theta 2021",   show_ylabel=True,
                        ymin=ymin, ymax=ymax)
    _draw_boxplot_panel(axes[1], polaris_tags, polaris_data, driver_colors,
                        "Polaris 2024", show_ylabel=False,
                        ymin=ymin, ymax=ymax)

    axes[0].set_ylabel("Wait Time (hours)", fontsize=13, fontweight="bold", labelpad=2)

    save_fig(fig, out_path)

def plot_wait_by_group(theta_by_grp, polaris_by_grp, theta_tags, polaris_tags, out_path):
    all_tags      = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = build_driver_colors(all_tags)

    fig = plt.figure(figsize=(15.8, 7.2))
    gs  = fig.add_gridspec(
        2, 4, wspace=0.10, hspace=0.28,
        left=0.10, right=0.99, top=0.92, bottom=0.18,
    )

    system_data = [
        (theta_by_grp,   theta_tags,   "Theta 2021"),
        (polaris_by_grp, polaris_tags, "Polaris 2024"),
    ]
    first_ax = None
    for row_idx, (by_grp, tags, sys_name) in enumerate(system_data):
        for col_idx, grp in enumerate(GROUPS):
            ax = fig.add_subplot(gs[row_idx, col_idx],
                                 **({"sharey": first_ax} if first_ax else {}))
            if first_ax is None:
                first_ax = ax
            gd = {t: by_grp.get(t, {}).get(grp, []) for t in tags}
            _draw_boxplot_panel(
                ax, tags, gd, driver_colors,
                title=grp if row_idx == 0 else "",
                show_ylabel=(col_idx == 0),
                ymin=None, ymax=None,
                show_xticklabels=True,
                xtick_fontsize=9,
                xtick_rotation=28,
                title_fontsize=14,
            )
            if col_idx == 0:
                ax.set_ylabel(f"{sys_name}\nWait Time (hours)",
                              fontsize=13, fontweight="bold", labelpad=8)

    all_vals = [v for d in (theta_by_grp, polaris_by_grp)
                for gd in d.values() for vals in gd.values() for v in vals]
    if all_vals:
        ymin = max(0.01, np.percentile(all_vals, 1) * 0.5)
        ymax = max(np.percentile(all_vals, 99) * 2.0, 1000.0)
        for ax in fig.axes:
            ax.set_ylim(ymin, ymax)
    save_fig(fig, out_path)

def plot_wait_by_group_compact(theta_by_grp, polaris_by_grp, out_path):
    compact_tags = ["WFP", "MARS-CW", "MARS-CU"]
    driver_colors = build_driver_colors(compact_tags)

    fig = plt.figure(figsize=(10.8, 6.8))
    gs  = fig.add_gridspec(
        2, 4, wspace=0.12, hspace=0.26,
        left=0.11, right=0.99, top=0.92, bottom=0.14,
    )

    system_data = [
        (theta_by_grp,   "Theta 2021"),
        (polaris_by_grp, "Polaris 2024"),
    ]
    first_ax = None
    for row_idx, (by_grp, sys_name) in enumerate(system_data):
        for col_idx, grp in enumerate(GROUPS):
            ax = fig.add_subplot(gs[row_idx, col_idx],
                                 **({"sharey": first_ax} if first_ax else {}))
            if first_ax is None:
                first_ax = ax
            gd = {t: by_grp.get(t, {}).get(grp, []) for t in compact_tags}
            _draw_boxplot_panel(
                ax, compact_tags, gd, driver_colors,
                title=grp if row_idx == 0 else "",
                show_ylabel=(col_idx == 0),
                show_xticklabels=True,
                xtick_fontsize=10,
                xtick_rotation=25,
                title_fontsize=14,
            )
            if col_idx == 0:
                ax.set_ylabel(f"{sys_name}\nWait Time (hours)",
                              fontsize=13, fontweight="bold", labelpad=8)

    all_vals = [v for d in (theta_by_grp, polaris_by_grp)
                for tag_data in d.values() for vals in tag_data.values() for v in vals]
    if all_vals:
        ymin = max(0.01, np.percentile(all_vals, 1) * 0.5)
        ymax = max(np.percentile(all_vals, 99) * 2.0, 1000.0)
        for ax in fig.axes:
            ax.set_ylim(ymin, ymax)
    save_fig(fig, out_path)

# ─── bsld/ figures ────────────────────────────────────────────────────────────
def plot_bsld_boxplot(theta_data, polaris_data, theta_tags, polaris_tags, out_path):
    all_tags      = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = build_driver_colors(all_tags)

    n_tags = max(len(theta_tags), len(polaris_tags))
    fig_h  = max(4.5, n_tags * 0.55 + 1.2)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, fig_h), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.92, bottom=0.12, wspace=0.04)

    _draw_boxplot_panel_h(axes[0], theta_tags,   theta_data,   driver_colors,
                          "Theta 2021",   show_ylabels=True)
    _draw_boxplot_panel_h(axes[1], polaris_tags, polaris_data, driver_colors,
                          "Polaris 2024", show_ylabels=False)

    for ax in axes:
        ax.set_xlabel("Bounded Slowdown", fontsize=13, fontweight="bold", labelpad=3)

    all_vals = [v for d in (theta_data, polaris_data) for vals in d.values() for v in vals]
    if all_vals:
        xmin = max(1.0, np.percentile(all_vals, 1) * 0.9)
        xmax = max(np.percentile(all_vals, 99) * 2.0, 10.0)
        for ax in axes:
            ax.set_xlim(xmin, xmax)

    save_fig(fig, out_path)

def plot_bsld_by_group(theta_by_grp, polaris_by_grp, theta_tags, polaris_tags, out_path):
    all_tags      = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = build_driver_colors(all_tags)

    n_tags  = max(len(theta_tags), len(polaris_tags))
    cell_h  = max(2.5, n_tags * 0.42)
    fig_h   = cell_h * 2 + 2.5
    fig     = plt.figure(figsize=(14.0, fig_h))
    gs      = fig.add_gridspec(2, 4, wspace=0.04, hspace=0.35,
                                left=0.13, right=0.99, top=0.93, bottom=0.08)

    system_data = [
        (theta_by_grp,   theta_tags,   "Theta 2021"),
        (polaris_by_grp, polaris_tags, "Polaris 2024"),
    ]
    first_ax = None
    for row_idx, (by_grp, tags, sys_name) in enumerate(system_data):
        for col_idx, grp in enumerate(GROUPS):
            ax = fig.add_subplot(gs[row_idx, col_idx],
                                 **({"sharey": first_ax} if first_ax else {}))
            if first_ax is None:
                first_ax = ax
            gd = {t: by_grp.get(t, {}).get(grp, []) for t in tags}
            _draw_boxplot_panel_h(ax, tags, gd, driver_colors,
                                  title=grp if row_idx == 0 else "",
                                  show_ylabels=(col_idx == 0))
            if col_idx == 0:
                ax.set_ylabel(sys_name, fontsize=13, fontweight="bold", labelpad=4)

    all_vals = [v for d in (theta_by_grp, polaris_by_grp)
                for gd in d.values() for vals in gd.values() for v in vals]
    if all_vals:
        xmin = max(1.0, np.percentile(all_vals, 1) * 0.9)
        xmax = max(np.percentile(all_vals, 99) * 2.0, 10.0)
        for ax in fig.axes:
            ax.set_xlim(xmin, xmax)

    fig.text(0.5, 0.01, "Bounded Slowdown",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
    save_fig(fig, out_path)

# ─── util/ figures ────────────────────────────────────────────────────────────
def plot_util_boxplot(theta_data, polaris_data, theta_tags, polaris_tags,
                       out_path, ylabel="Utilization (%)"):
    """Generic vertical boxplot for utilization (uptime or drain period)."""
    all_tags      = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = build_driver_colors(all_tags)

    all_vals = [v for d in (theta_data, polaris_data) for vals in d.values() for v in vals]
    ymin = max(0, np.floor(min(all_vals) - 2)) if all_vals else 0

    n_tags = max(len(theta_tags), len(polaris_tags))
    fig_w  = max(7.0, n_tags * 0.72 + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 4.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.22, wspace=0.06)

    _draw_boxplot_panel(axes[0], theta_tags,   theta_data,   driver_colors,
                        "Theta 2021",   show_ylabel=True,
                        log_scale=False, ymin=ymin, ymax=100, whis=1.5)
    _draw_boxplot_panel(axes[1], polaris_tags, polaris_data, driver_colors,
                        "Polaris 2024", show_ylabel=False,
                        log_scale=False, ymin=ymin, ymax=100, whis=1.5)

    axes[0].set_ylabel(ylabel, fontsize=13, fontweight="bold", labelpad=2)
    save_fig(fig, out_path)

def plot_drain_queue_boxplot(theta_data, polaris_data, theta_tags, polaris_tags, out_path):
    """Vertical boxplot of queue sizes during drain window."""
    all_tags      = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = build_driver_colors(all_tags)

    n_tags = max(len(theta_tags), len(polaris_tags))
    fig_w  = max(7.0, n_tags * 0.72 + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 4.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.22, wspace=0.06)

    all_vals = [v for d in (theta_data, polaris_data) for vals in d.values() for v in vals]
    ymax = np.percentile(all_vals, 99) * 1.2 if all_vals else 100

    _draw_boxplot_panel(axes[0], theta_tags,   theta_data,   driver_colors,
                        "Theta 2021",   show_ylabel=True,
                        log_scale=False, ymin=0, ymax=ymax, whis=1.5)
    _draw_boxplot_panel(axes[1], polaris_tags, polaris_data, driver_colors,
                        "Polaris 2024", show_ylabel=False,
                        log_scale=False, ymin=0, ymax=ymax, whis=1.5)

    axes[0].set_ylabel("Queue Length (jobs)", fontsize=13, fontweight="bold", labelpad=2)
    save_fig(fig, out_path)

# ─── workload/ figures ─────────────────────────────────────────────────────────
def _cell_text_color(value, norm, cmap):
    r, g, b, _ = cmap(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"

# Job-distribution heatmap layout, all in inches -- axes are placed with
# fig.add_axes() at exactly-computed rectangles so cells come out perfectly
# square and the colorbar can be sized to match the heatmap width exactly,
# instead of fighting matplotlib's automatic (and here, unpredictable)
# gridspec/aspect layout.
_JD_CELL          = 0.72   # square cell edge length
_JD_ROWLABEL_W    = 1.60   # row label column ("<=1024\n(25.0%)")
_JD_YLABEL_W      = 0.13   # rotated "Nodes Used" column
_JD_RIGHTLABEL_W  = 0.68   # group letter + cumulative % column
_JD_TITLE_H       = 0.32   # panel title
_JD_XLABEL_H      = 0.40   # xtick labels + "Requested Walltime"
_JD_CBAR_LABEL_H  = 0.26   # "% of total jobs" label above colorbar
_JD_CBAR_H        = 0.20   # colorbar bar height
_JD_CBAR_GAP      = 0.26   # gap between colorbar ticks and first panel title
_JD_PANEL_GAP     = 0.10   # gap between one panel's xlabel and next panel's title
_JD_PAD           = 0.04   # outer figure padding

def _draw_job_distribution_panel(ax, job_dist, sys_size, title, norm, cmap):
    rows = job_dist["rows"]
    total_jobs = job_dist["total_jobs"]
    n_rows = len(rows)
    n_cols = len(WT_LABELS)
    row_trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    pct = np.array(
        [[100.0 * c / total_jobs if total_jobs else 0.0 for c in row["counts"]]
         for row in rows]
    )
    ax.imshow(pct, cmap=cmap, norm=norm, extent=(0, n_cols, n_rows, 0), aspect="auto")

    for r, row in enumerate(rows):
        for c in range(n_cols):
            color = _cell_text_color(pct[r, c], norm, cmap)
            ax.text(c + 0.5, r + 0.38, f"{row['counts'][c]:,}",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color=color)
            ax.text(c + 0.5, r + 0.68, f"{pct[r, c]:.1f}%",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color=color)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(WT_LABELS, fontsize=11, fontweight="bold")
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=6)

    # Row labels (left): "<label>\n(anchor/sys_size %)" -- the anchor% is
    # only meaningful for the row that actually defines a group boundary,
    # but matches the reference figure showing it on every row.
    for r, row in enumerate(rows):
        pct_of_sys = 100.0 * row["anchor"] / sys_size
        ax.text(-0.02, r + 0.5, f"{row['label']}\n({pct_of_sys:.1f}%)",
                ha="right", va="center", fontsize=11, fontweight="bold",
                transform=row_trans)

    # Group dividers, right-hand S/M/L/XL labels, and cumulative %-of-jobs.
    # Divider lines overshoot the heatmap on both sides for visual separation.
    cum = 0
    row_idx = 0
    while row_idx < n_rows:
        glabel = rows[row_idx]["group"]
        span_start = row_idx
        while row_idx < n_rows and rows[row_idx]["group"] == glabel:
            cum += rows[row_idx]["total"]
            row_idx += 1
        span_end = row_idx
        ax.text(1.07, (span_start + span_end) / 2.0, glabel,
                ha="left", va="center", fontsize=17, fontweight="bold",
                transform=row_trans)
        if row_idx < n_rows:
            ax.axhline(row_idx, color="black", linestyle="--", linewidth=2.0,
                       xmin=-0.04, xmax=1.03, clip_on=False)
            cum_pct = 100.0 * cum / total_jobs if total_jobs else 0.0
            ax.text(1.07, row_idx, f"{cum_pct:.1f}%",
                    ha="left", va="center", fontsize=11, fontweight="bold",
                    transform=row_trans)

def plot_job_distribution(panels, out_path):
    """Node-count x requested-walltime job-distribution heatmap.

    `panels` is an ordered list of {"label", "job_dist" (from
    load_job_distribution), "sys_size"} -- one entry per system, drawn top
    to bottom.
    """
    all_pct = [100.0 * c / p["job_dist"]["total_jobs"]
               for p in panels for row in p["job_dist"]["rows"] for c in row["counts"]
               if p["job_dist"]["total_jobs"]]
    global_max = max(all_pct, default=1.0)
    vmax = max(5.0, np.ceil(global_max / 5.0) * 5.0)
    # Not a true log scale: PowerNorm(gamma<1) keeps 0 mapped to 0 (unlike
    # LogNorm) while still stretching the low end of the range so small
    # percentages remain visually distinguishable -- matches the reference
    # figure's colorbar, which shows evenly-labeled 0/5/10/.../25 ticks but
    # gives the 0-5 band disproportionately more color range than 5-10 etc.
    norm = PowerNorm(gamma=0.4, vmin=0, vmax=vmax)
    cmap = plt.get_cmap("viridis_r")
    cbar_ticks = np.arange(0, vmax + 1, 5)

    n_cols = len(WT_LABELS)
    row_counts = [len(p["job_dist"]["rows"]) for p in panels]

    heatmap_w = n_cols * _JD_CELL
    fig_w = _JD_PAD + _JD_YLABEL_W + _JD_ROWLABEL_W + heatmap_w + _JD_RIGHTLABEL_W + _JD_PAD
    panel_block_h = [_JD_TITLE_H + n_rows * _JD_CELL + _JD_XLABEL_H for n_rows in row_counts]
    content_h = (_JD_CBAR_LABEL_H + _JD_CBAR_H + _JD_CBAR_GAP
                 + sum(panel_block_h) + _JD_PANEL_GAP * (len(panels) - 1))
    fig_h = _JD_PAD + content_h + _JD_PAD

    fig = plt.figure(figsize=(fig_w, fig_h))

    heatmap_x0 = _JD_PAD + _JD_YLABEL_W + _JD_ROWLABEL_W
    y_cursor = fig_h - _JD_PAD  # top-down cursor, in inches

    y_cursor -= _JD_CBAR_LABEL_H + _JD_CBAR_H
    cax = fig.add_axes([heatmap_x0 / fig_w, y_cursor / fig_h,
                        heatmap_w / fig_w, _JD_CBAR_H / fig_h])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=cbar_ticks)
    cbar.ax.set_xticklabels([f"{t:g}" for t in cbar_ticks])
    cbar.set_label("% of total jobs", fontsize=12, fontweight="bold", labelpad=4)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cax.tick_params(labelsize=10)
    y_cursor -= _JD_CBAR_GAP

    for i, panel in enumerate(panels):
        n_rows = row_counts[i]
        y_cursor -= _JD_TITLE_H + n_rows * _JD_CELL
        ax = fig.add_axes([heatmap_x0 / fig_w, y_cursor / fig_h,
                           heatmap_w / fig_w, (n_rows * _JD_CELL) / fig_h])
        _draw_job_distribution_panel(
            ax, panel["job_dist"], panel["sys_size"], panel["label"], norm, cmap)
        ax.set_xlabel("Requested Walltime", fontsize=12, fontweight="bold", labelpad=4)
        fig.text(_JD_PAD + _JD_YLABEL_W / 2, y_cursor / fig_h + (n_rows * _JD_CELL / fig_h) / 2,
                 "Nodes Used", ha="center", va="center", fontsize=12,
                 fontweight="bold", rotation=90)
        y_cursor -= _JD_XLABEL_H + _JD_PANEL_GAP

    save_fig(fig, out_path)

# ─── drain/ figures ───────────────────────────────────────────────────────────
def _build_backfill_drain_grid_panel(results_dir, procs_map, system, label):
    GREY = "#cccccc"
    RED  = "#e74c3c"
    BLUE = "#26edff"

    wfp_tag = next(
        (c for c in ("WFP", "WFP3-w256", "WFP3")
         if os.path.isdir(os.path.join(results_dir, c))), None
    )
    mars_tag = "MARS-CW" if os.path.isdir(os.path.join(results_dir, "MARS-CW")) else None

    rows_cfg = []
    if wfp_tag:
        rows_cfg.append((wfp_tag, "backfill", "primary", "WFP",     RED,  "backfill"))
    if mars_tag:
        rows_cfg.append((mars_tag, "drain",   "nondrain","MARS-CW", BLUE, "drain"))
    if not rows_cfg:
        return None

    all_data = {}
    for tag, hi_key, pri_key, rlabel, hc, kind in rows_cfg:
        if kind == "backfill":
            all_data[tag] = load_backfill_4grp(results_dir, tag, procs_map, system)
        else:
            all_data[tag] = load_mars_drain_4grp(results_dir, tag, procs_map, system)

    return {
        "label": label,
        "rows_cfg": rows_cfg,
        "all_data": all_data,
    }

def _plot_backfill_drain_grid_panels(panels, out_path,
                                     fig_width=None, fig_height=None,
                                     grid_wspace=0.08, font_scale=1.0,
                                     legend_fontsize=None,
                                     common_hist_y=False,
                                     row_label_fontsize=None,
                                     system_label_fontsize=None):
    GREY = "#cccccc"
    RED  = "#e74c3c"
    BLUE = "#26edff"
    bins = np.logspace(-1, 4, 41)

    panels = [p for p in panels if p]
    if not panels:
        return

    row_specs = []
    panel_ranges = []
    for panel in panels:
        row_start = len(row_specs)
        for row_cfg in panel["rows_cfg"]:
            row_specs.append((panel, *row_cfg))
        panel_ranges.append((row_start, len(row_specs) - 1, panel["label"]))

    n_rows = len(row_specs)
    n_panels = len(panels)
    is_combined = n_panels > 1

    fig_w = fig_width if fig_width is not None else (13.6 if not is_combined else 14.6)
    fig_h = fig_height if fig_height is not None else (n_rows * 2.45 + (0.70 if is_combined else 0.35))
    left_margin = 0.085 if not is_combined else 0.115
    bottom_margin = 0.10 if is_combined else 0.095
    ann_fs = 9.5 * font_scale
    group_fs = 17.0 * font_scale
    ylabel_fs = row_label_fontsize if row_label_fontsize is not None else 11.5 * font_scale
    ytick_fs = 8.5 * font_scale
    xlabel_fs = 10.5 * font_scale
    xtick_fs = 8.5 * font_scale
    legend_fs = legend_fontsize if legend_fontsize is not None else 9.5 * font_scale
    system_fs = system_label_fontsize if system_label_fontsize is not None else 13.0 * font_scale
    single_label_fs = 11.5 * font_scale
    major_xticks = [1, 10, 100, 1000, 10000]

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(
        n_rows * 2, 4, figure=fig,
        height_ratios=[5, 1] * n_rows,
        hspace=0.0, wspace=grid_wspace,
        left=left_margin, right=0.995, top=0.96, bottom=bottom_margin,
    )
    first_hist_ax = None
    hist_axes = []
    for r in range(n_rows):
        row_axes = []
        for c in range(4):
            share_kwargs = {"sharey": first_hist_ax} if first_hist_ax is not None else {}
            ax = fig.add_subplot(gs[r * 2, c], **share_kwargs)
            if first_hist_ax is None:
                first_hist_ax = ax
            row_axes.append(ax)
        hist_axes.append(row_axes)
    box_axes  = [[fig.add_subplot(gs[r * 2 + 1, c], sharex=hist_axes[r][c])
                  for c in range(4)] for r in range(n_rows)]

    row_ymaxes = []
    for panel, tag, hi_key, pri_key, *_ in row_specs:
        peak = 0.0
        for g in GROUPS:
            pri = np.array(panel["all_data"][tag].get((pri_key, g), []), dtype=float)
            hi  = np.array(panel["all_data"][tag].get((hi_key,  g), []), dtype=float)
            tot = max(len(pri) + len(hi), 1)
            st  = np.zeros(len(bins) - 1)
            for arr in (pri, hi):
                if len(arr):
                    hh, _ = np.histogram(arr, bins=bins)
                    st += hh
            peak = max(peak, float((st / tot * 100).max()))
        row_ymaxes.append(peak * 1.10)
    global_ymax = max(row_ymaxes, default=1.0)

    for r, (panel, tag, hi_key, pri_key, rlabel, hi_color, _) in enumerate(row_specs):
        ymax = global_ymax if common_hist_y else row_ymaxes[r]

        for c, grp in enumerate(GROUPS):
            ax_h = hist_axes[r][c]
            ax_b = box_axes[r][c]
            pri = np.array(panel["all_data"][tag].get((pri_key, grp), []), dtype=float)
            hi  = np.array(panel["all_data"][tag].get((hi_key,  grp), []), dtype=float)
            tot = max(len(pri) + len(hi), 1)
            h_pri, _ = np.histogram(pri, bins=bins)
            h_hi,  _ = np.histogram(hi,  bins=bins)
            pct_pri  = h_pri / tot * 100
            pct_hi   = h_hi  / tot * 100
            lefts    = bins[:-1]
            widths   = np.diff(bins)
            ax_h.bar(lefts, pct_pri, width=widths, bottom=0, align="edge",
                     color=GREY, zorder=3, edgecolor="none")
            ax_h.bar(lefts, pct_hi, width=widths, bottom=pct_pri, align="edge",
                     color=hi_color, zorder=3, edgecolor="none")
            ax_h.set_xscale("log")
            ax_h.set_xlim(1e-1, 1e4)
            ax_h.set_xticks(major_xticks)
            ax_h.set_ylim(0, ymax)
            ax_h.grid(True, alpha=0.25, which="both", zorder=0)
            ax_h.spines["bottom"].set_visible(False)
            ax_h.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            n_hi     = len(hi)
            pct_frac = 100.0 * n_hi / tot
            kind_lbl = "Drain" if hi_color == BLUE else "Backfill"
            ax_h.text(0.5, 0.97, f"{kind_lbl}: {n_hi:,}  ({pct_frac:.1f}%)",
                      transform=ax_h.transAxes, ha="center", va="top",
                      fontsize=ann_fs, fontweight="bold")
            if r == 0:
                ax_h.text(0.5, 1.08, grp, transform=ax_h.transAxes,
                          ha="center", va="bottom", fontsize=group_fs, fontweight="bold")
            ax_h.set_ylabel(f"{rlabel}\n% of Job Class" if c == 0 else "",
                            fontsize=ylabel_fs, fontweight="bold")
            if c == 0:
                ax_h.tick_params(axis="y", labelsize=ytick_fs)
                for lbl in ax_h.get_yticklabels():
                    lbl.set_fontweight("bold")
            else:
                ax_h.tick_params(axis="y", left=False, labelleft=False)

            all_vals = np.concatenate([pri, hi]) if len(pri) + len(hi) else np.array([])
            if len(all_vals) > 0:
                ax_b.boxplot(
                    all_vals, orientation="horizontal", widths=0.55,
                    patch_artist=True, whis=(5, 95),
                    showfliers=False, showmeans=True, meanline=True,
                    boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.8),
                    whiskerprops=dict(linewidth=0.8, color="black"),
                    capprops=dict(linewidth=0.8, color="black"),
                    medianprops=dict(linewidth=0.8, color="black"),
                    meanprops=dict(linewidth=1.3, color="red", linestyle=":"),
                )
            ax_b.set_xlim(1e-1, 1e4)
            ax_b.set_xscale("log")
            ax_b.set_xticks(major_xticks)
            ax_b.xaxis.set_major_formatter(LogFormatterMathtext())
            ax_b.spines["top"].set_visible(False)
            ax_b.spines["left"].set_visible(False)
            ax_b.spines["right"].set_visible(False)
            ax_b.set_yticks([])
            if r == n_rows - 1:
                ax_b.set_xlabel("Wait Time (hours)", fontsize=xlabel_fs, fontweight="bold")
                ax_b.tick_params(axis="x", labelsize=xtick_fs)
            else:
                ax_b.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            for lbl in ax_b.get_xticklabels():
                lbl.set_fontweight("bold")

    fig.legend(
        handles=[mpatches.Patch(facecolor=GREY, label="Primary"),
                 mpatches.Patch(facecolor=RED,  label="Backfill"),
                 mpatches.Patch(facecolor=BLUE, label="Drain")],
        loc="lower center", bbox_to_anchor=(0.5, 0.012),
        ncol=3, fontsize=legend_fs, frameon=False,
    )

    if is_combined:
        for row_start, row_end, label in panel_ranges:
            y_top = hist_axes[row_start][0].get_position().y1
            y_bot = box_axes[row_end][0].get_position().y0
            fig.text(0.018, 0.5 * (y_top + y_bot), label,
                     ha="center", va="center", rotation=90,
                     fontsize=system_fs, fontweight="bold")
    else:
        fig.text(0.5, 0.001, panels[0]["label"], ha="center", va="bottom",
                 fontsize=single_label_fs, fontweight="bold")

    save_fig(fig, out_path)

def plot_backfill_drain_grid(results_dir, procs_map, system, label, out_path):
    panel = _build_backfill_drain_grid_panel(results_dir, procs_map, system, label)
    _plot_backfill_drain_grid_panels([panel], out_path)

def plot_backfill_drain_grid_combined(theta_results_dir, theta_procs, theta_system,
                                      polaris_results_dir, polaris_procs, polaris_system,
                                      out_path):
    theta_panel = _build_backfill_drain_grid_panel(
        theta_results_dir, theta_procs, theta_system, "Theta 2021")
    polaris_panel = _build_backfill_drain_grid_panel(
        polaris_results_dir, polaris_procs, polaris_system, "Polaris 2024")
    _plot_backfill_drain_grid_panels(
        [theta_panel, polaris_panel], out_path,
        fig_width=12.2, fig_height=12.3,
        grid_wspace=0.045, font_scale=1.18,
        legend_fontsize=13.0,
        common_hist_y=True,
        row_label_fontsize=16.0,
        system_label_fontsize=18.0,
    )

def _overview_row_ymax(values, minimum=10.0):
    vmax = max(values, default=0.0)
    if vmax <= 0:
        return minimum
    padded = max(vmax * 1.35, vmax + 18.0)
    if padded <= 10:
        return 10.0
    if padded <= 25:
        return 25.0
    return float(np.ceil(padded / 10.0) * 10.0)

def plot_rate_share_overview(sys_data_list, out_path,
                             left_title, right_title, right_ylabel,
                             left_ymaxs=None, right_ymaxs=None,
                             figsize=(7.0, 4.2), bar_label_fontsize=7.0,
                             subplot_wspace=0.32, font_scale=1.0):
    """Overview figure with a left event-rate panel and right class-share panel."""
    FS_TICK = 8.5 * font_scale
    FS_AXIS = 9.0 * font_scale
    FS_BAR  = bar_label_fontsize
    FS_TITLE = 9.5 * font_scale
    FS_ROW = 10.0 * font_scale

    overview_tags = sys_data_list[0][1] if sys_data_list else []
    driver_colors = build_driver_colors(overview_tags)

    fig, axes = plt.subplots(
        2, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [1, 2.4]},
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.93, bottom=0.11,
                        wspace=subplot_wspace, hspace=0.33)

    for ri, (label, dtags, drain_pct_by_tag, postdrain_by_tag, total_by_tag) in enumerate(sys_data_list):
        ax_l = axes[ri][0]
        ax_r = axes[ri][1]

        # ── left: drain rate ──────────────────────────────────────────────────
        n      = len(dtags)
        bar_w  = min(0.30, 1.55 / max(n, 1))
        gap_l  = max(bar_w * 1.55, 0.26)
        offs_l = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * gap_l
        x_l    = 0.5 + offs_l
        left_pcts = [drain_pct_by_tag.get(t, (0, 0, 0.0))[2] for t in dtags]

        for i, tag in enumerate(dtags):
            ax_l.bar(x_l[i], left_pcts[i], width=bar_w,
                     color=driver_colors.get(tag, "gray"), alpha=0.88,
                     hatch=get_hatch(tag), edgecolor="black", linewidth=0.8)

        y_pad = max(left_pcts, default=1) * 0.02
        for i, pct in enumerate(left_pcts):
            if pct <= 0:
                continue
            ax_l.text(x_l[i], pct + y_pad, f"{pct:.1f}%",
                      ha="center", va="bottom", rotation=90,
                      fontsize=FS_BAR, fontweight="bold", color="black")

        ax_l.set_xticks(x_l)
        lbls = ax_l.set_xticklabels(
            [display_tag(t) for t in dtags],
            rotation=18 if n > 5 else 20,
            ha="right",
            fontsize=(7.8 if n > 5 else 8.5) * font_scale,
        )
        for lbl in lbls:
            lbl.set_fontweight("bold")
        ax_l.set_xlim(np.min(x_l) - bar_w * 1.1, np.max(x_l) + bar_w * 1.1)
        ax_l.set_ylabel(label, fontsize=FS_ROW, fontweight="bold", color="black", labelpad=1)
        if left_ymaxs is not None:
            ax_l.set_ylim(0, left_ymaxs[ri])
        else:
            ax_l.set_ylim(0, _overview_row_ymax(left_pcts))
        ax_l.grid(True, axis="y", alpha=0.25)
        ax_l.set_axisbelow(True)
        for lbl in ax_l.get_xticklabels() + ax_l.get_yticklabels():
            lbl.set_fontweight("bold")
        ax_l.tick_params(labelsize=FS_TICK)
        if ri == 0:
            ax_l.set_title(left_title, fontsize=FS_TITLE, fontweight="bold", pad=3)

        # ── right: post-drain class share ─────────────────────────────────────
        bw   = min(0.32, 0.82 / max(n, 1))
        x_r  = np.arange(len(GROUPS), dtype=float)
        offs = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * (bw * 1.08)

        right_pcts = {}
        for tag in dtags:
            totals = total_by_tag.get(tag, {})
            right_pcts[tag] = {
                g: 100.0 * postdrain_by_tag.get(tag, {}).get(g, 0) / totals.get(g, 1)
                if totals.get(g, 0) else 0.0
                for g in GROUPS
            }
            ax_r.bar(x_r + offs[dtags.index(tag)],
                     [right_pcts[tag][g] for g in GROUPS],
                     width=bw,
                     color=driver_colors.get(tag, "gray"), alpha=0.88,
                     hatch=get_hatch(tag), edgecolor="black", linewidth=0.8,
                     label=display_tag(tag))

        global_max = max(
            (right_pcts[t][g] for t in dtags for g in GROUPS), default=1)
        y_pad_r = global_max * 0.02
        for tag in dtags:
            xi = dtags.index(tag)
            for j, g in enumerate(GROUPS):
                v = right_pcts[tag][g]
                if v <= 0:
                    continue
                ax_r.text(x_r[j] + offs[xi], v + y_pad_r, f"{v:.1f}%",
                          ha="center", va="bottom", rotation=90,
                          fontsize=FS_BAR, fontweight="bold", color="black")

        ax_r.set_xticks(x_r)
        lbls = ax_r.set_xticklabels(GROUPS, rotation=30, ha="center")
        for lbl in lbls:
            lbl.set_fontweight("bold")
        ax_r.set_xlim(
            np.min(x_r + np.min(offs)) - bw * 0.7,
            np.max(x_r + np.max(offs)) + bw * 0.7,
        )
        ax_r.set_ylabel(right_ylabel, fontsize=FS_AXIS, fontweight="bold", color="black")
        if right_ymaxs is not None:
            ax_r.set_ylim(0, right_ymaxs[ri])
        else:
            ax_r.set_ylim(0, _overview_row_ymax(
                [right_pcts[t][g] for t in dtags for g in GROUPS],
                minimum=25.0,
            ))
        ax_r.grid(True, axis="y", alpha=0.25)
        ax_r.set_axisbelow(True)
        for lbl in ax_r.get_xticklabels() + ax_r.get_yticklabels():
            lbl.set_fontweight("bold")
        ax_r.tick_params(labelsize=FS_TICK)
        if ri == 0:
            ax_r.set_title(right_title, fontsize=FS_TITLE, fontweight="bold", pad=3)

    axes[1][1].set_xlabel("Size Class", fontsize=FS_TICK, fontweight="bold")
    save_fig(fig, out_path)

def plot_drain_overview(sys_data_list, out_path):
    """Match original plot_combined_drain_overview_ab formatting."""
    plot_rate_share_overview(
        sys_data_list, out_path,
        left_title="Drain Rate",
        right_title="Post-Drain Share by Class",
        right_ylabel="Post-drain Jobs (%)",
        left_ymaxs=[25, 25],
        right_ymaxs=[80, 70],
    )

def _wildcard_count(pattern, seq_counts):
    """Sum seq_counts for all sequences matching a wildcard pattern like 'S-S-*'."""
    parts = pattern.split("-")
    total = 0
    for seq, cnt in seq_counts.items():
        sp = seq.split("-")
        if len(sp) != 3:
            continue
        if all(p == "*" or p == s for p, s in zip(parts, sp)):
            total += cnt
    return total

def plot_drain_sequence_combos(sys_data_list, out_path):
    n_sys = len(sys_data_list)
    fig, axes = plt.subplots(n_sys, 2, figsize=(4.8, 2.5 * n_sys + 0.5))

    # Canonical y-ordering from the first row's CW subplot; applied to all subplots
    first_cw_stats = sys_data_list[0][1] if sys_data_list else None
    if first_cw_stats is not None:
        canonical_seqs = [s for s, _ in build_wildcard_entries(
            first_cw_stats["seq_counts"], max_entries=5)]
    else:
        canonical_seqs = []
    n_cats = len(canonical_seqs)

    for ri, (label, cw_stats, cu_stats, cw_pct, cu_pct) in enumerate(sys_data_list):
        axs = axes[ri] if n_sys > 1 else axes
        for ci, (stats, pct_tup, col_label) in enumerate([
                (cw_stats, cw_pct, "MARS-CW"), (cu_stats, cu_pct, "MARS-CU")]):
            ax = axs[ci]
            if stats is None:
                ax.axis("off")
                continue
            total  = stats["total_drain_episodes"]
            seqs   = canonical_seqs
            pcts_v = [100.0 * _wildcard_count(s, stats["seq_counts"]) / total
                      for s in seqs]
            x_max     = max(pcts_v) * 1.35 if any(pcts_v) else 1.0
            tag_color = get_color(col_label)
            tag_hatch = get_hatch(col_label)
            ax.barh(np.arange(n_cats), pcts_v, color=tag_color,
                    edgecolor="black", linewidth=0.5, height=0.55,
                    hatch=tag_hatch if tag_hatch else None)
            for y, p in enumerate(pcts_v):
                ax.text(p + x_max * 0.015, y, f"{p:.1f}%",
                        va="center", ha="left", fontsize=7.5, fontweight="bold")
            ax.set_yticks(range(n_cats))
            if ci == 0:
                ax.set_yticklabels(seqs, fontsize=8, fontfamily="monospace",
                                   fontweight="bold")
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False)
            ax.set_ylim(n_cats - 0.5, -0.5)
            ax.set_xlim(0, x_max)
            ax.set_xlabel("% of drain episodes", fontsize=8)
            if pct_tup:
                ep, _, pv = pct_tup
                title = f"{col_label}\n{label}\nN={ep:,} ({pv:.1f}%)"
            else:
                title = f"{col_label}\n{label}"
            ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
            ax.tick_params(axis="x", labelsize=7.5)
            ax.tick_params(axis="y", length=0)
            ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout(w_pad=0.3)
    save_fig(fig, out_path)

def plot_drain_resume_time(theta_data, polaris_data, out_path,
                            tags=("MARS-CW", "MARS-CU")):
    """Two-panel boxplot of drain→first-Run gap in minutes."""
    TAG_COLORS  = {t: get_color(t) for t in tags}
    TAG_HATCHES = {t: get_hatch(t) for t in tags}

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.06)

    for ax, (label, data_by_tag) in zip(axes, [
            ("Theta 2021",   theta_data),
            ("Polaris 2024", polaris_data),
    ]):
        present = [t for t in tags if data_by_tag.get(t)]
        data    = [data_by_tag[t] for t in present]
        colors  = [TAG_COLORS[t]  for t in present]
        hatches = [TAG_HATCHES[t] for t in present]

        bp = ax.boxplot(data, tick_labels=present, patch_artist=True,
                        showfliers=False, whis=1.5,
                        medianprops=dict(color="black", linewidth=2.8))
        for patch, color, hatch in zip(bp["boxes"], colors, hatches):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            if hatch:
                patch.set_hatch(hatch)
                patch.set_edgecolor("black")

        for i, vals in enumerate(data, start=1):
            if vals:
                mean_val = np.mean(vals)
                ax.plot([i - 0.4, i + 0.4], [mean_val, mean_val],
                        color="red", linestyle=":", linewidth=1.8)
                ax.text(i + 0.2, mean_val, f"{mean_val:.1f}",
                        color="red", fontsize=12, fontweight="bold",
                        va="bottom", ha="left")

        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", labelsize=12)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Minutes", fontsize=13, fontweight="bold")
    save_fig(fig, out_path)

def plot_backfill_fraction(theta_fracs, polaris_fracs, theta_tags, polaris_tags, out_path):
    all_tags      = list(dict.fromkeys(theta_tags + polaris_tags))
    driver_colors = build_driver_colors(all_tags)

    all_pcts = [v * 100.0 for d in (theta_fracs, polaris_fracs) for v in d.values()]
    ymax = min(100.0, max(all_pcts, default=1) * 1.25)

    fig, axes = plt.subplots(1, 2, figsize=(max(6, len(all_tags) * 0.7 + 2), 4.5),
                             sharey=True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.22, wspace=0.06)

    for ci, (ax, tags, fracs, label) in enumerate([
            (axes[0], theta_tags,   theta_fracs,   "Theta 2021"),
            (axes[1], polaris_tags, polaris_fracs, "Polaris 2024"),
    ]):
        x    = np.arange(len(tags))
        pcts = [fracs.get(t, 0) * 100.0 for t in tags]
        for i, (tag, pct) in enumerate(zip(tags, pcts)):
            ax.bar(x[i], pct, color=driver_colors.get(tag, "gray"), alpha=0.85,
                   hatch=get_hatch(tag), edgecolor="black", linewidth=0.8)
            text_y = pct + ymax * 0.01
            va = "bottom"
            if pct > 80.0 or text_y > ymax * 0.985:
                text_y = max(pct - ymax * 0.02, pct * 0.55)
                va = "top"
            ax.text(x[i], text_y, f"{pct:.1f}%",
                    ha="center", va=va, fontsize=9, fontweight="bold", rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([display_tag(t) for t in tags],
                           rotation=25, ha="right", fontsize=11, fontweight="bold")
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_ylim(0, ymax)
        ax.grid(True, axis="y", alpha=0.3)
        if ci == 0:
            ax.set_ylabel("Backfill Fraction (%)", fontsize=11, fontweight="bold")

    save_fig(fig, out_path)

def plot_mcts_iterations_branching_row(theta_data, polaris_data, out_path,
                                       tags=("MARS-CW", "MARS-CU"),
                                       iterations_divisor=250.0,
                                       iterations_ylabel="MCTS Iterations / 250 / Cycle"):
    """Single-row MCTS summary: iterations boxplot + branching boxplot per system."""
    driver_colors = build_driver_colors(tags)
    neutral_box = "#d9d9d9"

    iter_data = {}
    branch_data = {}
    for sys_label, sys_data in (("Theta 2021", theta_data), ("Polaris 2024", polaris_data)):
        iter_data[sys_label] = {
            tag: [v / iterations_divisor for v in sys_data.get(tag, {}).get("iterations", [])]
            for tag in tags
        }
        branch_data[sys_label] = {
            tag: sys_data.get(tag, {}).get("branching", [])
            for tag in tags
        }

    all_iter_vals = [v for sys_map in iter_data.values() for vals in sys_map.values() for v in vals]
    iter_ymin = max(0.01, np.percentile(all_iter_vals, 1) * 0.6) if all_iter_vals else None
    iter_ymax = max(np.percentile(all_iter_vals, 99) * 1.8, 100.0) if all_iter_vals else None

    all_branch_vals = [v for sys_map in branch_data.values() for vals in sys_map.values() for v in vals]
    branch_ymax = max(np.percentile(all_branch_vals, 99) * 1.10, 5.0) if all_branch_vals else None

    fig = plt.figure(figsize=(9.1, 3.0))
    outer = fig.add_gridspec(1, 2, left=0.075, right=0.995, top=0.80, bottom=0.22, wspace=0.15)
    iter_gs = outer[0].subgridspec(1, 2, wspace=0.04)
    branch_gs = outer[1].subgridspec(1, 2, wspace=0.04)
    axes = [
        fig.add_subplot(iter_gs[0, 0]),
        fig.add_subplot(iter_gs[0, 1]),
        fig.add_subplot(branch_gs[0, 0]),
        fig.add_subplot(branch_gs[0, 1]),
    ]
    axes[1].sharey(axes[0])
    axes[3].sharey(axes[2])
    box_positions = [1.00, 1.45]

    for ax, sys_label, plot_kind in (
            (axes[0], "Theta 2021", "iter"),
            (axes[1], "Polaris 2024", "iter"),
            (axes[2], "Theta 2021", "branch"),
            (axes[3], "Polaris 2024", "branch"),
    ):
        show_left_ticks = ax in (axes[0], axes[2])
        if plot_kind == "iter":
            _draw_boxplot_panel(
                ax, tags, iter_data[sys_label], driver_colors,
                title=sys_label,
                show_ylabel=show_left_ticks,
                log_scale=True,
                ymin=iter_ymin, ymax=iter_ymax,
                whis=(5, 95),
                xtick_fontsize=10,
                xtick_rotation=22,
                title_fontsize=11,
                box_width=0.28,
                use_hatches=False,
                fixed_facecolor=neutral_box,
                box_positions=box_positions,
                xlim_padding=0.40,
            )
            ax.set_ylabel("")
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
            if not show_left_ticks:
                ax.tick_params(axis="y", labelleft=False)
        else:
            _draw_boxplot_panel(
                ax, tags, branch_data[sys_label], driver_colors,
                title=sys_label,
                show_ylabel=show_left_ticks,
                log_scale=False,
                ymin=0, ymax=branch_ymax,
                whis=(5, 95),
                xtick_fontsize=10,
                xtick_rotation=22,
                title_fontsize=11,
                box_width=0.28,
                use_hatches=False,
                fixed_facecolor=neutral_box,
                box_positions=box_positions,
                xlim_padding=0.40,
            )
            ax.set_ylabel("")
            if not show_left_ticks:
                ax.tick_params(axis="y", labelleft=False)

    fig.text(0.27, 0.94, "Iterations / 250", ha="center", va="center",
             fontsize=13, fontweight="bold")
    fig.text(0.75, 0.94, "Branching Factor", ha="center", va="center",
             fontsize=13, fontweight="bold")
    save_fig(fig, out_path)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate final paper plots.")
    parser.add_argument("exp2a_json", help="Path to exp2a.json (Theta)")
    parser.add_argument("exp2b_json", help="Path to exp2b.json (Polaris)")
    parser.add_argument("--output_dir", default="CLUSTER26")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir  = args.output_dir

    cfg_a, drivers_a = load_config(args.exp2a_json)
    cfg_b, drivers_b = load_config(args.exp2b_json)

    sys_a, size_a, year_a = detect_system(cfg_a)
    sys_b, size_b, year_b = detect_system(cfg_b)

    def abs_path(p):
        return p if os.path.isabs(p) else os.path.join(base_dir, p)

    res_a  = abs_path(cfg_a.get("output_dir", "results/exp2a"))
    res_b  = abs_path(cfg_b.get("output_dir", "results/exp2b"))

    print("Parsing SWF files...")
    swf_a = abs_path(cfg_a.get("swf_path", ""))
    swf_b = abs_path(cfg_b.get("swf_path", ""))
    procs_a, walltimes_a, submit_a, _ = parse_swf(swf_a) if os.path.exists(swf_a) else ({}, {}, {}, {})
    procs_b, walltimes_b, submit_b, _ = parse_swf(swf_b) if os.path.exists(swf_b) else ({}, {}, {}, {})

    # Maintenance windows from events.csv (has SMA/SMS/SME events)
    print("Loading maintenance windows...")
    evt_a = first_events_csv(res_a)
    evt_b = first_events_csv(res_b)
    maint_a = parse_maintenance_from_events(evt_a) if evt_a else []
    maint_b = parse_maintenance_from_events(evt_b) if evt_b else []

    year_start_a = int(datetime(year_a,   1, 1, tzinfo=CST).timestamp())
    year_end_a   = int(datetime(year_a+1, 1, 1, tzinfo=CST).timestamp())
    year_start_b = int(datetime(year_b,   1, 1, tzinfo=CST).timestamp())
    year_end_b   = int(datetime(year_b+1, 1, 1, tzinfo=CST).timestamp())

    label_a = f"Theta {year_a}"
    label_b = f"Polaris {year_b}"

    # ── workload/ ────────────────────────────────────────────────────────────
    print("\nGenerating workload/ figures...")
    job_dist_a = load_job_distribution(procs_a, submit_a, walltimes_a, sys_a,
                                       year_start_a, year_end_a)
    job_dist_b = load_job_distribution(procs_b, submit_b, walltimes_b, sys_b,
                                       year_start_b, year_end_b)
    plot_job_distribution(
        [
            {"label": f"Polaris ({year_b})", "job_dist": job_dist_b, "sys_size": size_b},
            {"label": f"Theta ({year_a})",   "job_dist": job_dist_a, "sys_size": size_a},
        ],
        os.path.join(out_dir, "workload", "job_distribution.png"),
    )

    def _tags_for_plots(drivers, results_dir):
        """Ordered tags, skipping plot-excluded and those without data."""
        tags = [d["tag"] for d in drivers if not is_plot_excluded(d["tag"])]
        return sort_tags(tags)

    tags_a = _tags_for_plots(drivers_a, res_a)
    tags_b = _tags_for_plots(drivers_b, res_b)

    # ── wait/ ─────────────────────────────────────────────────────────────────
    print("\nGenerating wait/ figures...")
    wait_a, wait_a_grp = {}, {}
    for d in drivers_a:
        if is_plot_excluded(d["tag"]): continue
        sub, sta, _ = get_events_for_driver(d, res_a, base_dir)
        wd = load_wait(sub, sta, procs_a, sys_a)
        wait_a[d["tag"]] = wd["all"]
        wait_a_grp[d["tag"]] = {g: wd[g] for g in GROUPS}

    wait_b, wait_b_grp = {}, {}
    for d in drivers_b:
        if is_plot_excluded(d["tag"]): continue
        sub, sta, _ = get_events_for_driver(d, res_b, base_dir)
        wd = load_wait(sub, sta, procs_b, sys_b)
        wait_b[d["tag"]] = wd["all"]
        wait_b_grp[d["tag"]] = {g: wd[g] for g in GROUPS}

    plot_wait_boxplot(wait_a, wait_b, tags_a, tags_b,
                      os.path.join(out_dir, "wait", "wait_boxplot.png"))
    plot_wait_by_group(wait_a_grp, wait_b_grp, tags_a, tags_b,
                       os.path.join(out_dir, "wait", "wait_by_group.png"))
    plot_wait_by_group_compact(
        wait_a_grp, wait_b_grp,
        os.path.join(out_dir, "wait", "wait_by_group_compact.png"),
    )
    wait_group_rows = []
    for system_label, tags, grouped_data in (
            (label_a, tags_a, wait_a_grp),
            (label_b, tags_b, wait_b_grp),
    ):
        for grp in GROUPS:
            wait_group_rows.append({
                "system": system_label,
                "metric": "Wait_Time_hours",
                "job_class": grp,
                "tags": tags,
                "data": {tag: grouped_data.get(tag, {}).get(grp, []) for tag in tags},
                "lower_is_better": True,
            })
    write_grouped_metrics_quartiles_table(
        os.path.join(out_dir, "wait", "wait_quartiles_by_group_vs_wfp_baseline.csv"),
        wait_group_rows,
    )

    # Overall + per-size-class wait time distribution tables (minutes),
    # matching the original CLUSTER'26 table_wait.csv / table_wait_{group}.csv.
    write_wait_stat_table(
        os.path.join(out_dir, "wait", "table_wait.csv"),
        [(label_a, tag, wait_a.get(tag, [])) for tag in tags_a] +
        [(label_b, tag, wait_b.get(tag, [])) for tag in tags_b],
    )
    for grp in GROUPS:
        write_wait_stat_table(
            os.path.join(out_dir, "wait", f"table_wait_{grp}.csv"),
            [(label_a, tag, wait_a_grp.get(tag, {}).get(grp, [])) for tag in tags_a] +
            [(label_b, tag, wait_b_grp.get(tag, {}).get(grp, [])) for tag in tags_b],
        )

    # MARS-CW vs WFP wait-time % change, one row per (system, job class).
    wfp_tag_a_wait = find_wfp_baseline_tag(tags_a)
    wfp_tag_b_wait = find_wfp_baseline_tag(tags_b)
    pct_change_rows = []
    for system_label, wfp_tag, wait_all, wait_grp in (
            (label_a, wfp_tag_a_wait, wait_a, wait_a_grp),
            (label_b, wfp_tag_b_wait, wait_b, wait_b_grp),
    ):
        if not wfp_tag or "MARS-CW" not in wait_all:
            continue
        pct_change_rows.append(
            (system_label, "All", wait_all.get(wfp_tag, []), wait_all.get("MARS-CW", [])))
        for grp in GROUPS:
            pct_change_rows.append((
                system_label, grp,
                wait_grp.get(wfp_tag, {}).get(grp, []),
                wait_grp.get("MARS-CW", {}).get(grp, []),
            ))
    if pct_change_rows:
        write_wait_pct_change_table(
            os.path.join(out_dir, "wait", "table_wait_pct_change_mars_cw_vs_wfp.csv"),
            pct_change_rows, baseline_tag="WFP", compare_tag="MARS-CW",
        )

    # ── bsld/ ────────────────────────────────────────────────────────────────
    print("\nGenerating bsld/ figures...")
    bsld_a, bsld_a_grp = {}, {}
    for d in drivers_a:
        if is_plot_excluded(d["tag"]): continue
        tag = d["tag"]
        if d.get("type") == "rlscheduler":
            # RLScheduler: no end events, skip BSLD
            continue
        sub, sta, end = parse_events(os.path.join(res_a, tag, "events.csv"))
        bd = load_bsld(sub, sta, end, procs_a, sys_a)
        bsld_a[tag] = bd["all"]
        bsld_a_grp[tag] = {g: bd[g] for g in GROUPS}

    bsld_b, bsld_b_grp = {}, {}
    for d in drivers_b:
        if is_plot_excluded(d["tag"]): continue
        tag = d["tag"]
        if d.get("type") == "rlscheduler":
            continue
        sub, sta, end = parse_events(os.path.join(res_b, tag, "events.csv"))
        bd = load_bsld(sub, sta, end, procs_b, sys_b)
        bsld_b[tag] = bd["all"]
        bsld_b_grp[tag] = {g: bd[g] for g in GROUPS}

    bsld_tags_a = sort_tags(list(bsld_a.keys()))
    bsld_tags_b = sort_tags(list(bsld_b.keys()))
    plot_bsld_boxplot(bsld_a, bsld_b, bsld_tags_a, bsld_tags_b,
                      os.path.join(out_dir, "bsld", "bsld_boxplot.png"))
    plot_bsld_by_group(bsld_a_grp, bsld_b_grp, bsld_tags_a, bsld_tags_b,
                       os.path.join(out_dir, "bsld", "bsld_by_group.png"))

    # ── util/ ────────────────────────────────────────────────────────────────
    print("\nGenerating util/ figures...")
    def _has_events(results_dir, tag, dtype):
        return dtype != "rlscheduler" and os.path.exists(
            os.path.join(results_dir, tag, "events.csv"))

    util_a = {}
    for d in drivers_a:
        if is_plot_excluded(d["tag"]):
            continue
        if _has_events(res_a, d["tag"], d.get("type", "")):
            util_a[d["tag"]] = load_uptime_util_per_period(
                res_a, d["tag"], procs_a, maint_a, year_start_a, year_end_a, size_a)
        elif d.get("type") == "rlscheduler":
            rl_util = load_rl_overall_util(
                d, base_dir, procs_a, maint_a, year_start_a, year_end_a, size_a)
            if rl_util is not None:
                util_a[d["tag"]] = [rl_util]

    util_b = {}
    for d in drivers_b:
        if is_plot_excluded(d["tag"]):
            continue
        if _has_events(res_b, d["tag"], d.get("type", "")):
            util_b[d["tag"]] = load_uptime_util_per_period(
                res_b, d["tag"], procs_b, maint_b, year_start_b, year_end_b, size_b)
        elif d.get("type") == "rlscheduler":
            rl_util = load_rl_overall_util(
                d, base_dir, procs_b, maint_b, year_start_b, year_end_b, size_b)
            if rl_util is not None:
                util_b[d["tag"]] = [rl_util]
    util_tags_a = sort_tags(list(util_a.keys()))
    util_tags_b = sort_tags(list(util_b.keys()))
    plot_util_boxplot(util_a, util_b, util_tags_a, util_tags_b,
                      os.path.join(out_dir, "util", "util_uptime.png"),
                      ylabel="Utilization (%)")

    write_overall_metrics_quartiles_table(
        os.path.join(out_dir, "overall_metrics_quartiles_vs_wfp_baseline.csv"),
        [
            {
                "system": label_a,
                "metric": "Wait_Time_hours",
                "tags": tags_a,
                "data": wait_a,
                "lower_is_better": True,
            },
            {
                "system": label_b,
                "metric": "Wait_Time_hours",
                "tags": tags_b,
                "data": wait_b,
                "lower_is_better": True,
            },
            {
                "system": label_a,
                "metric": "Bounded_Slowdown",
                "tags": bsld_tags_a,
                "data": bsld_a,
                "lower_is_better": True,
            },
            {
                "system": label_b,
                "metric": "Bounded_Slowdown",
                "tags": bsld_tags_b,
                "data": bsld_b,
                "lower_is_better": True,
            },
            {
                "system": label_a,
                "metric": "Uptime_Utilization_pct",
                "tags": util_tags_a,
                "data": util_a,
                "lower_is_better": False,
            },
            {
                "system": label_b,
                "metric": "Uptime_Utilization_pct",
                "tags": util_tags_b,
                "data": util_b,
                "lower_is_better": False,
            },
        ],
    )

    for days in (1, 2):
        du_a = {
            d["tag"]: load_drain_period_utils(res_a, d["tag"], procs_a, maint_a, size_a, days)
            for d in drivers_a
            if not is_plot_excluded(d["tag"]) and _has_events(res_a, d["tag"], d.get("type",""))
        }
        du_b = {
            d["tag"]: load_drain_period_utils(res_b, d["tag"], procs_b, maint_b, size_b, days)
            for d in drivers_b
            if not is_plot_excluded(d["tag"]) and _has_events(res_b, d["tag"], d.get("type",""))
        }
        du_tags_a = sort_tags([t for t, v in du_a.items() if v])
        du_tags_b = sort_tags([t for t, v in du_b.items() if v])
        plot_util_boxplot(du_a, du_b, du_tags_a, du_tags_b,
                          os.path.join(out_dir, "util", f"util_drain_{days}day.png"),
                          ylabel=f"Utilization (%) — {days}d before maintenance")

        dq_a = {
            d["tag"]: load_drain_queue_samples(res_a, d["tag"], maint_a, days)
            for d in drivers_a
            if not is_plot_excluded(d["tag"]) and _has_events(res_a, d["tag"], d.get("type",""))
        }
        dq_b = {
            d["tag"]: load_drain_queue_samples(res_b, d["tag"], maint_b, days)
            for d in drivers_b
            if not is_plot_excluded(d["tag"]) and _has_events(res_b, d["tag"], d.get("type",""))
        }
        dq_tags_a = sort_tags([t for t, v in dq_a.items() if v])
        dq_tags_b = sort_tags([t for t, v in dq_b.items() if v])
        plot_drain_queue_boxplot(
            dq_a, dq_b, dq_tags_a, dq_tags_b,
            os.path.join(out_dir, "util", f"drain_queue_{days}day.png"),
        )

    # ── drain/ ───────────────────────────────────────────────────────────────
    print("\nGenerating drain/ figures...")
    plot_backfill_drain_grid(res_a, procs_a, sys_a, label_a,
                             os.path.join(out_dir, "drain", "backfill_drain_grid_theta.png"))
    plot_backfill_drain_grid(res_b, procs_b, sys_b, label_b,
                             os.path.join(out_dir, "drain", "backfill_drain_grid_polaris.png"))
    plot_backfill_drain_grid_combined(
        res_a, procs_a, sys_a,
        res_b, procs_b, sys_b,
        os.path.join(out_dir, "drain", "backfill_drain_grid_combined.png"),
    )

    # Backfill fraction — heuristics only (FCFS, WFP, F1, SJF, LJF, LRF)
    # Metric: fraction of started jobs whose start event is Backfill.
    _bf_tag_up = {t.upper() for t in _BF_FRACTION_TAGS}
    bf_a, bf_b = {}, {}
    for all_drivers, res_dir, bf_dict in [
            (cfg_a["drivers"], res_a, bf_a),
            (cfg_b["drivers"], res_b, bf_b),
    ]:
        for d in all_drivers:
            tag = d["tag"]
            if tag.upper() not in _bf_tag_up:
                continue
            frac = load_backfill_job_fraction(res_dir, tag)
            if frac is not None:
                bf_dict[tag] = frac

    def _bf_sort_key(t):
        u = t.upper()
        for i, p in enumerate(_BF_FRACTION_ORDER):
            if u.startswith(p):
                return i
        return len(_BF_FRACTION_ORDER)

    bf_tags_a = sorted(list(bf_a.keys()), key=_bf_sort_key)
    bf_tags_b = sorted(list(bf_b.keys()), key=_bf_sort_key)
    plot_backfill_fraction(bf_a, bf_b, bf_tags_a, bf_tags_b,
                           os.path.join(out_dir, "drain", "backfill_fraction.png"))

    # Drain overview
    drain_tags = ["MARS-CW", "MARS-CU"]
    pct_a, post_a, tot_a = load_drain_overview(res_a, drain_tags, procs_a, sys_a)
    pct_b, post_b, tot_b = load_drain_overview(res_b, drain_tags, procs_b, sys_b)
    plot_drain_overview(
        [(label_a, drain_tags, pct_a, post_a, tot_a),
         (label_b, drain_tags, pct_b, post_b, tot_b)],
        os.path.join(out_dir, "drain", "drain_overview.png"),
    )

    wfp_tag_a = find_wfp_baseline_tag(tags_a)
    wfp_tag_b = find_wfp_baseline_tag(tags_b)
    if wfp_tag_a and wfp_tag_b:
        wfp_pct_a, wfp_backfill_a, wfp_total_a = load_backfill_overview(
            res_a, wfp_tag_a, procs_a, sys_a)
        wfp_pct_b, wfp_backfill_b, wfp_total_b = load_backfill_overview(
            res_b, wfp_tag_b, procs_b, sys_b)

        plot_rate_share_overview(
            [(label_a, [wfp_tag_a], {wfp_tag_a: wfp_pct_a},
              {wfp_tag_a: wfp_backfill_a}, {wfp_tag_a: wfp_total_a}),
             (label_b, [wfp_tag_b], {wfp_tag_b: wfp_pct_b},
              {wfp_tag_b: wfp_backfill_b}, {wfp_tag_b: wfp_total_b})],
            os.path.join(out_dir, "drain", "backfill_overview_wfp.png"),
            left_title="Backfill Rate",
            right_title="Backfill Share by Class",
            right_ylabel="Backfilled Jobs (%)",
        )

        all_tags_a = [wfp_tag_a] + drain_tags
        all_tags_b = [wfp_tag_b] + drain_tags
        plot_rate_share_overview(
            [(label_a, all_tags_a,
              {wfp_tag_a: wfp_pct_a, **pct_a},
              {wfp_tag_a: wfp_backfill_a, **post_a},
              {wfp_tag_a: wfp_total_a, **tot_a}),
             (label_b, all_tags_b,
              {wfp_tag_b: wfp_pct_b, **pct_b},
              {wfp_tag_b: wfp_backfill_b, **post_b},
              {wfp_tag_b: wfp_total_b, **tot_b})],
            os.path.join(out_dir, "drain", "drain_overview_all.png"),
            left_title="Backfill / Drain Rate",
            right_title="Backfill / Post-Drain Share by Class",
            right_ylabel="Affected Jobs (%)",
            figsize=(8.4, 4.9),
            bar_label_fontsize=9.0,
        )

        all_strategy_tags_a = [
            t for t in tags_a
            if display_tag(t).upper() not in {"RLSCHEDULER", "RANDOM"}
        ]
        all_strategy_tags_b = [
            t for t in tags_b
            if display_tag(t).upper() not in {"RLSCHEDULER", "RANDOM"}
        ]
        all_strategy_order = {
            "FCFS": 0,
            "SJF": 1,
            "F1": 2,
            "WFP": 3,
            "MARS-CW": 4,
            "MARS-CU": 5,
        }
        all_strategy_tags_a = sorted(
            all_strategy_tags_a,
            key=lambda t: all_strategy_order.get(display_tag(t).upper(), 999),
        )
        all_strategy_tags_b = sorted(
            all_strategy_tags_b,
            key=lambda t: all_strategy_order.get(display_tag(t).upper(), 999),
        )
        mix_pct_a, mix_aff_a, mix_tot_a = load_mixed_backfill_drain_overview(
            res_a, all_strategy_tags_a, procs_a, sys_a)
        mix_pct_b, mix_aff_b, mix_tot_b = load_mixed_backfill_drain_overview(
            res_b, all_strategy_tags_b, procs_b, sys_b)
        plot_rate_share_overview(
            [(label_a, all_strategy_tags_a, mix_pct_a, mix_aff_a, mix_tot_a),
             (label_b, all_strategy_tags_b, mix_pct_b, mix_aff_b, mix_tot_b)],
            os.path.join(out_dir, "drain", "drain_overview_all_except_rlscheduler.png"),
            left_title="Backfill / Drain Rate",
            right_title="Backfill / Post-Drain Share by Class",
            right_ylabel="Affected Jobs (%)",
            figsize=(14.2, 5.9),
            bar_label_fontsize=12.2,
            subplot_wspace=0.16,
            font_scale=1.36,
        )

    # Drain sequence combos
    combo_rows = []
    for label, res_dir, procs, system in [
            (label_a, res_a, procs_a, sys_a), (label_b, res_b, procs_b, sys_b)]:
        cw_stats, cw_pct = load_drain_combo_stats(res_dir, "MARS-CW", procs, system)
        cu_stats, cu_pct = load_drain_combo_stats(res_dir, "MARS-CU", procs, system)
        combo_rows.append((label, cw_stats, cu_stats, cw_pct, cu_pct))
    plot_drain_sequence_combos(combo_rows,
                               os.path.join(out_dir, "drain", "drain_sequence_combos.png"))

    # Drain resume time (computed directly from performance.csv)
    rt_a = {tag: load_drain_resume_times(res_a, tag, maint_a) for tag in drain_tags}
    rt_b = {tag: load_drain_resume_times(res_b, tag, maint_b) for tag in drain_tags}
    plot_drain_resume_time(rt_a, rt_b,
                           os.path.join(out_dir, "drain", "drain_resume_time.png"))

    # ── mcts/ ────────────────────────────────────────────────────────────────
    print("\nGenerating mcts/ figures...")
    theta_mcts   = load_mcts_perf_data(res_a)
    polaris_mcts = load_mcts_perf_data(res_b)
    plot_mcts_iterations_branching_row(
        theta_mcts, polaris_mcts,
        os.path.join(out_dir, "mcts", "mcts_iterations_branching_div250.png"),
        iterations_divisor=250.0,
        iterations_ylabel="MCTS Iterations / 250 / Cycle",
    )

    print(f"\nDone. Outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
