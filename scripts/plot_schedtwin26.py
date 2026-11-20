#!/usr/bin/env python3
"""plot_schedtwin26.py — CLUSTER26-style plots for the four SchedTwin traces.

Reuses the helpers in plot_cluster26.py, but with four systems
(Frontier, Aurora, Polaris, Theta) instead of two, and with every metric
restricted to the 10,000 target jobs named in each SWF header (warm-up and
cool-down jobs are simulated but not counted). The traces have no
maintenance windows, so the maintenance-specific CLUSTER26 figures
(uptime-period util, drain-before-maintenance util/queue, drain resume time)
are replaced by per-day utilization over the target window.

Usage:
    python3 scripts/plot_schedtwin26.py [--output_dir SchedTwin26]
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_cluster26 as pc
from slurm_dist import read_swf, target_ids

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS = pc.GROUPS

SYSTEMS = [  # (key, display label, experiment json)
    ("frontier", "Frontier 2025", "experiments/schedtwin-frontier.json"),
    ("aurora",   "Aurora 2026",   "experiments/schedtwin-aurora.json"),
    ("polaris",  "Polaris 2024",  "experiments/schedtwin-polaris.json"),
    ("theta",    "Theta 2022",    "experiments/schedtwin-theta.json"),
]

# Size classes (nodes, inclusive) chosen so every target job falls in a class.
# The CLUSTER26 classes start at 128 nodes (Theta) / 10 nodes (Polaris) and
# would drop roughly half of the SchedTwin target jobs, which include
# debug-sized work.
SIZE_CLASSES = {
    "frontier": [("S", 1, 1), ("M", 2, 16), ("L", 17, 256),  ("XL", 257, 9408)],
    "aurora":   [("S", 1, 1), ("M", 2, 16), ("L", 17, 256),  ("XL", 257, 10624)],
    "polaris":  [("S", 1, 1), ("M", 2, 8),  ("L", 9, 32),    ("XL", 33, 550)],
    "theta":    [("S", 1, 8), ("M", 9, 128), ("L", 129, 512), ("XL", 513, 4392)],
}

# Same policy set as CLUSTER26 (LJF/LRF excluded there too).
# MARS-SS (StartShare reward) is plotted when its results exist.
PLOT_TAGS = ["Random", "WFP", "MARS-CW", "MARS-CU", "MARS-SS", "FCFS", "SJF", "F1"]
HEURISTIC_TAGS = ["FCFS", "SJF", "F1", "WFP"]
MARS_TAGS = ["MARS-CW", "MARS-CU", "MARS-SS"]


def assign_group(num_procs, system):
    for g, lo, hi in SIZE_CLASSES.get(system, []):
        if lo <= num_procs <= hi:
            return g
    return None


# Route every plot_cluster26 helper through the SchedTwin size classes.
pc.assign_group = assign_group
pc.NODE_GROUP_RANGES.update(SIZE_CLASSES)

# MARS-SS gets its own color/hatch and sits after MARS-CU.
_base_color, _base_hatch = pc.get_color, pc.get_hatch
pc.get_color = lambda tag: "#2ca6a4" if tag.upper().endswith("-SS") else _base_color(tag)
pc.get_hatch = lambda tag: "\\\\" if tag.upper().endswith("-SS") else _base_hatch(tag)
if "MARS-SS" not in pc._TAG_ORDER:
    pc._TAG_ORDER.insert(pc._TAG_ORDER.index("MARS-CU") + 1, "MARS-SS")


# ─── Loading ──────────────────────────────────────────────────────────────────
def load_system(key, label, cfg_path):
    with open(os.path.join(BASE_DIR, cfg_path)) as f:
        cfg = json.load(f)
    swf = os.path.join(BASE_DIR, cfg["swf_path"])
    res = os.path.join(BASE_DIR, cfg["output_dir"])
    info, jobs = read_swf(swf)
    tids = set(target_ids(info, jobs))
    procs_all, walls, submits, _ = pc.parse_swf(swf)
    procs_t = {j: p for j, p in procs_all.items() if j in tids}
    tags = [t for t in PLOT_TAGS if os.path.exists(os.path.join(res, t, "events.csv"))]
    tags = pc.sort_tags(tags)

    s = {"key": key, "label": label, "res": res, "tids": tids, "tags": tags,
         "max_procs": info["max_procs"], "procs_all": procs_all,
         "procs_t": procs_t, "walls": walls, "submits": submits,
         "events": {}}
    for tag in tags:
        sub, st, en, stype = pc.parse_events_with_start_types(
            os.path.join(res, tag, "events.csv"))
        s["events"][tag] = (sub, st, en, stype)
    # Target window = first target submit .. last target end (per run).
    s["window"] = {}
    for tag, (sub, st, en, _) in s["events"].items():
        t0 = min(sub.get(j, submits[j]) for j in tids)
        t1 = max(en[j] for j in tids if j in en)
        s["window"][tag] = (t0, t1)
    return s


def target_waits(s, tag):
    """All target-job waits in seconds (zeros included)."""
    sub, st, _, _ = s["events"][tag]
    return np.array([max(0.0, st[j] - sub[j]) for j in s["tids"] if j in st and j in sub])


def grouped(s, tag, kind):
    """{group: [values], 'all': [...]} over target jobs with wait > 0,
    matching CLUSTER26's load_wait / load_bsld conventions."""
    sub, st, en, _ = s["events"][tag]
    sub = {j: v for j, v in sub.items() if j in s["tids"]}
    if kind == "wait":
        return pc.load_wait(sub, st, s["procs_t"], s["key"])
    return pc.load_bsld(sub, st, en, s["procs_t"], s["key"])


def util_target_window(s, tag):
    _, st, en, _ = s["events"][tag]
    t0, t1 = s["window"][tag]
    used = pc.sum_used_core_seconds(st, en, s["procs_all"], t0, t1)
    return used / (s["max_procs"] * (t1 - t0))


def daily_util(s, tag):
    """Utilization (%) of each full 24 h slice of the target window."""
    _, st, en, _ = s["events"][tag]
    t0, t1 = s["window"][tag]
    out = []
    t = t0
    while t + 86400 <= t1:
        used = pc.sum_used_core_seconds(st, en, s["procs_all"], t, t + 86400)
        out.append(100.0 * used / (s["max_procs"] * 86400))
        t += 86400
    return out


def backfill_fraction(s, tag):
    _, st, _, stype = s["events"][tag]
    started = [j for j in s["tids"] if j in st]
    return sum(stype.get(j) == "Backfill" for j in started) / len(started)


def drain_pct_in_window(s, tag):
    """Drain episodes / decision cycles, both inside the target window."""
    rows, episodes = pc.compute_drain_episodes(s["res"], tag)
    t0, t1 = s["window"][tag]
    rows = [r for r in rows if t0 <= r["sim_time"] <= t1]
    episodes = [e for e in episodes if t0 <= e["sim_time"] <= t1]
    pct = 100.0 * len(episodes) / len(rows) if rows else 0.0
    return len(episodes), len(rows), pct


# ─── Generic 4-system figures ─────────────────────────────────────────────────
def _colors(systems):
    return pc.build_driver_colors(list(dict.fromkeys(t for s in systems for t in s["tags"])))


def plot_boxplot_row(systems, data, out_path, ylabel, log_scale=True,
                     ylim=None, whis=(5, 95), tags_override=None):
    colors = _colors(systems)
    n = max(len(s["tags"]) for s in systems)
    fig, axes = plt.subplots(1, len(systems), figsize=(max(12.0, 4 * (n * 0.5 + 1)), 4.6))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.22, wspace=0.06)
    for i, (ax, s) in enumerate(zip(axes, systems)):
        tags = [t for t in (tags_override or s["tags"]) if t in s["tags"]]
        pc._draw_boxplot_panel(ax, tags, data[s["key"]], colors, s["label"],
                               show_ylabel=(i == 0), log_scale=log_scale,
                               ymin=ylim[0] if ylim else None,
                               ymax=ylim[1] if ylim else None, whis=whis)
    axes[0].set_ylabel(ylabel, fontsize=13, fontweight="bold", labelpad=2)
    pc.save_fig(fig, out_path)


def plot_by_group(systems, data_grp, out_path, ylabel, tags_override=None,
                  figsize=(15.8, 13.0), xtick_fontsize=9):
    colors = _colors(systems)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(len(systems), 4, wspace=0.10, hspace=0.32,
                          left=0.09, right=0.99, top=0.96, bottom=0.07)
    first = None
    all_vals = []
    for r, s in enumerate(systems):
        tags = [t for t in (tags_override or s["tags"]) if t in s["tags"]]
        for c, grp in enumerate(GROUPS):
            ax = fig.add_subplot(gs[r, c], **({"sharey": first} if first else {}))
            first = first or ax
            gd = {t: data_grp[s["key"]].get(t, {}).get(grp, []) for t in tags}
            all_vals += [v for vals in gd.values() for v in vals]
            lo, hi = SIZE_CLASSES[s["key"]][c][1:]
            rng = f"{lo}" if lo == hi else f"{lo}–{hi}"
            pc._draw_boxplot_panel(ax, tags, gd, colors,
                                   title=f"{grp} ({rng} nodes)",
                                   show_ylabel=(c == 0),
                                   xtick_fontsize=xtick_fontsize,
                                   xtick_rotation=28, title_fontsize=12)
            if c == 0:
                ax.set_ylabel(f"{s['label']}\n{ylabel}", fontsize=12,
                              fontweight="bold", labelpad=6)
    if all_vals:
        lo = max(0.01, np.percentile(all_vals, 1) * 0.5)
        hi = max(np.percentile(all_vals, 99) * 2.0, 1000.0)
        for ax in fig.axes:
            ax.set_ylim(lo, hi)
    pc.save_fig(fig, out_path)


def plot_bars(systems, values, out_path, ylabel, tags_by_sys, fmt="{:.1f}%"):
    colors = _colors(systems)
    all_v = [v for d in values.values() for v in d.values()]
    ymax = min(100.0, max(all_v, default=1) * 1.25)
    fig, axes = plt.subplots(1, len(systems), figsize=(13.0, 4.4), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.22, wspace=0.06)
    for i, (ax, s) in enumerate(zip(axes, systems)):
        tags = tags_by_sys[s["key"]]
        x = np.arange(len(tags))
        for j, t in enumerate(tags):
            v = values[s["key"]].get(t, 0.0)
            ax.bar(x[j], v, color=colors.get(t, "gray"), alpha=0.85,
                   hatch=pc.get_hatch(t), edgecolor="black", linewidth=0.8)
            ax.text(x[j], v + ymax * 0.01, fmt.format(v), ha="center",
                    va="bottom", fontsize=9, fontweight="bold", rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([pc.display_tag(t) for t in tags], rotation=25,
                           ha="right", fontsize=11, fontweight="bold")
        ax.set_title(s["label"], fontsize=13, fontweight="bold")
        ax.set_ylim(0, ymax)
        ax.grid(True, axis="y", alpha=0.3)
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    pc.save_fig(fig, out_path)


def plot_mcts(systems, out_path, divisor=48.0):
    """Iterations (top) and branching factor (bottom) per system."""
    tags = [t for t in MARS_TAGS if all(t in s["tags"] for s in systems)]
    mcts = {s["key"]: pc.load_mcts_perf_data(s["res"], tags) for s in systems}
    colors = pc.build_driver_colors(tags)
    fig, axes = plt.subplots(2, len(systems), figsize=(11.0, 5.6))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.10,
                        wspace=0.08, hspace=0.35)
    it_vals = [v / divisor for d in mcts.values() for t in d.values() for v in t["iterations"]]
    br_vals = [v for d in mcts.values() for t in d.values() for v in t["branching"]]
    it_lim = (max(0.01, np.percentile(it_vals, 1) * 0.6), np.percentile(it_vals, 99) * 1.8)
    br_lim = (0, max(np.percentile(br_vals, 99) * 1.1, 5.0))
    for c, s in enumerate(systems):
        d = mcts[s["key"]]
        for r, (kind, lim, log) in enumerate((("iterations", it_lim, True),
                                               ("branching", br_lim, False))):
            vals = {t: ([v / divisor for v in d[t][kind]] if kind == "iterations"
                        else d[t][kind]) for t in tags if t in d}
            ax = axes[r][c]
            pc._draw_boxplot_panel(ax, tags, vals, colors,
                                   title=s["label"] if r == 0 else "",
                                   show_ylabel=(c == 0), log_scale=log,
                                   ymin=lim[0], ymax=lim[1], xtick_fontsize=10,
                                   xtick_rotation=22, title_fontsize=12,
                                   box_width=0.28, use_hatches=False,
                                   fixed_facecolor="#d9d9d9",
                                   box_positions=[1.0 + 0.45 * i for i in range(len(tags))],
                                   xlim_padding=0.40)
            if log:
                ax.yaxis.set_major_formatter(pc.LogFormatterMathtext())
    axes[0][0].set_ylabel(f"Iterations / {divisor:g}\n/ cycle", fontsize=12, fontweight="bold")
    axes[1][0].set_ylabel("Branching factor", fontsize=12, fontweight="bold")
    pc.save_fig(fig, out_path)


# ─── Main results table ───────────────────────────────────────────────────────
def write_main_results(systems, out_dir):
    """Per-trace wt_Mean / wt_Geo / wt_P99 (s, zeros included) and Util.
    wt_Geo = exp(mean(ln(1 + wait))), matching the paper table."""
    rows = []
    for s in systems:
        for tag in s["tags"]:
            w = target_waits(s, tag)
            rows.append({"System": s["label"], "Policy": tag, "Jobs": len(w),
                         "wt_Mean_s": w.mean(),
                         "wt_Geo_s": float(np.exp(np.log1p(w).mean())),
                         "wt_P99_s": float(np.percentile(w, 99)),
                         "Util": util_target_window(s, tag)})
    path = os.path.join(out_dir, "main_results.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output_dir", default="SchedTwin26")
    out = ap.parse_args().output_dir

    print("Loading traces and runs...")
    systems = [load_system(*s) for s in SYSTEMS]
    for s in systems:
        print(f"  {s['label']}: {len(s['tids'])} target jobs, tags {s['tags']}")

    print("\nmain_results.csv")
    write_main_results(systems, out)

    # workload/
    print("\nworkload/")
    panels = []
    for s in systems:
        jd = pc.load_job_distribution(s["procs_t"], s["submits"], s["walls"],
                                      s["key"], -np.inf, np.inf)
        panels.append({"label": s["label"], "job_dist": jd, "sys_size": s["max_procs"]})
    pc.plot_job_distribution(panels, os.path.join(out, "workload", "job_distribution.png"))
    pc.write_job_distribution_table(os.path.join(out, "workload", "job_distribution.csv"), panels)

    # wait/ and bsld/
    print("\nwait/ and bsld/")
    wait = {s["key"]: {t: grouped(s, t, "wait") for t in s["tags"]} for s in systems}
    bsld = {s["key"]: {t: grouped(s, t, "bsld") for t in s["tags"]} for s in systems}
    wait_all = {k: {t: v["all"] for t, v in d.items()} for k, d in wait.items()}
    bsld_all = {k: {t: v["all"] for t, v in d.items()} for k, d in bsld.items()}

    vals = [v for d in wait_all.values() for l in d.values() for v in l]
    plot_boxplot_row(systems, wait_all, os.path.join(out, "wait", "wait_boxplot.png"),
                     "Wait Time (hours)",
                     ylim=(max(0.01, np.percentile(vals, 1) * 0.5),
                           max(np.percentile(vals, 99) * 2.0, 1000.0)))
    plot_by_group(systems, wait, os.path.join(out, "wait", "wait_by_group.png"),
                  "Wait Time (hours)")
    plot_by_group(systems, wait, os.path.join(out, "wait", "wait_by_group_compact.png"),
                  "Wait Time (hours)", tags_override=["WFP", "MARS-CW", "MARS-CU", "MARS-SS"],
                  figsize=(10.8, 12.0), xtick_fontsize=10)

    pc.write_wait_stat_table(
        os.path.join(out, "wait", "table_wait.csv"),
        [(s["label"], t, wait_all[s["key"]][t]) for s in systems for t in s["tags"]])
    for grp in GROUPS:
        pc.write_wait_stat_table(
            os.path.join(out, "wait", f"table_wait_{grp}.csv"),
            [(s["label"], t, wait[s["key"]][t][grp])
             for s in systems for t in s["tags"]])
    for cmp_tag in MARS_TAGS:
        rows = []
        for s in systems:
            if cmp_tag not in s["tags"]:
                continue
            rows.append((s["label"], "All", wait_all[s["key"]]["WFP"], wait_all[s["key"]][cmp_tag]))
            for grp in GROUPS:
                rows.append((s["label"], grp, wait[s["key"]]["WFP"][grp], wait[s["key"]][cmp_tag][grp]))
        if not rows:
            continue
        pc.write_wait_pct_change_table(
            os.path.join(out, "wait", f"table_wait_pct_change_{cmp_tag.lower().replace('-', '_')}_vs_wfp.csv"),
            rows, baseline_tag="WFP", compare_tag=cmp_tag)
    pc.write_grouped_metrics_quartiles_table(
        os.path.join(out, "wait", "wait_quartiles_by_group_vs_wfp_baseline.csv"),
        [{"system": s["label"], "metric": "Wait_Time_hours", "job_class": g,
          "tags": s["tags"], "data": {t: wait[s["key"]][t][g] for t in s["tags"]},
          "lower_is_better": True} for s in systems for g in GROUPS])

    vals = [v for d in bsld_all.values() for l in d.values() for v in l]
    plot_boxplot_row(systems, bsld_all, os.path.join(out, "bsld", "bsld_boxplot.png"),
                     "Bounded Slowdown",
                     ylim=(max(1.0, np.percentile(vals, 1) * 0.9),
                           max(np.percentile(vals, 99) * 2.0, 10.0)))
    plot_by_group(systems, bsld, os.path.join(out, "bsld", "bsld_by_group.png"),
                  "Bounded Slowdown")

    # util/
    print("\nutil/")
    util = {s["key"]: {t: daily_util(s, t) for t in s["tags"]} for s in systems}
    vals = [v for d in util.values() for l in d.values() for v in l]
    plot_boxplot_row(systems, util, os.path.join(out, "util", "util_daily.png"),
                     "Daily Utilization (%)", log_scale=False,
                     ylim=(max(0, np.floor(min(vals) - 2)), 100), whis=1.5)

    pc.write_overall_metrics_quartiles_table(
        os.path.join(out, "overall_metrics_quartiles_vs_wfp_baseline.csv"),
        [{"system": s["label"], "metric": m, "tags": s["tags"], "data": d[s["key"]],
          "lower_is_better": lb}
         for m, d, lb in (("Wait_Time_hours", wait_all, True),
                          ("Bounded_Slowdown", bsld_all, True),
                          ("Daily_Utilization_pct", util, False))
         for s in systems])

    # drain/
    print("\ndrain/")
    bf_tags = {s["key"]: [t for t in HEURISTIC_TAGS if t in s["tags"]] for s in systems}
    bf = {s["key"]: {t: 100.0 * backfill_fraction(s, t) for t in bf_tags[s["key"]]} for s in systems}
    plot_bars(systems, bf, os.path.join(out, "drain", "backfill_fraction.png"),
              "Backfill Fraction (%)", bf_tags)

    mars_by_sys = {s["key"]: [t for t in MARS_TAGS if t in s["tags"]] for s in systems}
    dr = {s["key"]: {t: drain_pct_in_window(s, t)[2] for t in mars_by_sys[s["key"]]} for s in systems}
    plot_bars(systems, dr, os.path.join(out, "drain", "drain_rate.png"),
              "Drain episodes / decision cycles (%)",
              mars_by_sys, fmt="{:.2f}%")

    order = ["FCFS", "SJF", "F1", "WFP", "MARS-CW", "MARS-CU"]
    mixed = []
    for s in systems:
        tags = [t for t in order if t in s["tags"]]
        pct, aff, tot = pc.load_mixed_backfill_drain_overview(s["res"], tags, s["procs_t"], s["key"])
        mixed.append((s["label"], tags, pct, aff, tot))
    for pair, name in ((mixed[:2], "frontier_aurora"), (mixed[2:], "polaris_theta")):
        pc.plot_rate_share_overview(
            pair, os.path.join(out, "drain", f"drain_overview_all_{name}.png"),
            left_title="Backfill / Drain Rate",
            right_title="Backfill / Post-Drain Share by Class",
            right_ylabel="Affected Jobs (%)", figsize=(14.2, 5.9),
            bar_label_fontsize=12.2, subplot_wspace=0.16, font_scale=1.36)

    combos = []
    for s in systems:
        cw, cwp = pc.load_drain_combo_stats(s["res"], "MARS-CW", s["procs_t"], s["key"])
        cu, cup = pc.load_drain_combo_stats(s["res"], "MARS-CU", s["procs_t"], s["key"])
        combos.append((s["label"], cw, cu, cwp, cup))
    pc.plot_drain_sequence_combos(combos, os.path.join(out, "drain", "drain_sequence_combos.png"))

    # mcts/
    print("\nmcts/")
    plot_mcts(systems, os.path.join(out, "mcts", "mcts_iterations_branching.png"))

    print(f"\nDone. Outputs in: {out}/")


if __name__ == "__main__":
    main()
