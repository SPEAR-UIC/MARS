#!/usr/bin/env python3
"""compare_cpp_python.py — Verify the C++ cqsim reimplementation against the
reference Python cqsim (cqsim-python) by comparing per-job and aggregate
scheduling metrics on the same input trace.

Data sources
------------
Python side  : cqsim-python/data/Results/<trace>.rst
               Semicolon-separated per finished job, in *completion* order:
               id;reqProc;reqProc;reqTime;run;wait;submit;start;end
               (see cqsim-python/src/IOModule/Output_log.py:print_result)

C++ side     : results/<exp>/<driver_tag>/events.csv
               sim_time,event,id  with event in {Submit, Run, Backfill, End, ...}
               A job's start time is marked by either "Run" (started in its
               primary scheduling slot) or "Backfill" (started via backfilling);
               "End" marks its finish time.

Ground truth : the shared .swf trace (e.g. data/theta21cln.swf) gives
               reqProc/reqTime/original run time per job id, used only as a
               sanity cross-check that both sides refer to the same job.

Job ids are the raw SWF first-column ids and line up 1:1 between the two
simulators, EXCEPT that some jobs finish in one output but not the other
(e.g. never scheduled by simulation end). Those are reported as coverage
diagnostics and excluded from the deviation statistics, which are computed
only on the intersection of completed job ids.

For each matched job we compare, using each simulator's own numbers:
    wait  = start - submit
    run   = end   - start
    bsld  = max(1, (wait + run) / max(run, 10))     # bounded slowdown, tau=10s

Usage
-----
    python3 scripts/compare_cpp_python.py \\
        --cpp-dir results/cqsimpy-test1 --cpp-tag FCFS \\
        --py-rst cqsim-python/data/Results/theta21cln_regulate.rst \\
        --swf data/theta21cln.swf \\
        --label theta21cln --output-dir results/cpp_vs_py/theta21cln

    # or just run the two built-in experiments this repo ships with:
    python3 scripts/compare_cpp_python.py --preset
"""

import argparse
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 150
BSLD_TAU = 10.0


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


def parse_rst(path):
    """Return {id: {'reqProc', 'reqTime', 'run', 'wait', 'submit', 'start', 'end'}}."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split(";")
            if len(fields) < 9:
                continue
            jid = int(fields[0])
            out[jid] = {
                "reqProc": int(float(fields[1])),
                "reqTime": float(fields[3]),
                "run": float(fields[4]),
                "wait": float(fields[5]),
                "submit": float(fields[6]),
                "start": float(fields[7]),
                "end": float(fields[8]),
            }
    return out


def parse_cpp_events(path):
    """Return {id: {'submit', 'start', 'end'}} built from events.csv."""
    submit_t, start_t, end_t = {}, {}, {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            jid = int(row["id"])
            t = int(row["sim_time"])
            ev = row["event"]
            if ev == "Submit":
                submit_t.setdefault(jid, t)
            elif ev == "Run" or ev == "Backfill":
                # A job starts running via either its primary scheduling slot
                # ("Run") or a backfill slot ("Backfill") -- both mark start time.
                start_t.setdefault(jid, t)
            elif ev == "End":
                end_t.setdefault(jid, t)

    out = {}
    for jid, sub in submit_t.items():
        if jid in start_t and jid in end_t:
            out[jid] = {"submit": sub, "start": start_t[jid], "end": end_t[jid]}
    return out


# ─── Metrics ──────────────────────────────────────────────────────────────────

def bsld(wait, run):
    return max(1.0, (wait + run) / max(run, BSLD_TAU))


def build_job_metrics(rec):
    """rec has 'submit','start','end' (cpp) or additionally 'wait','run' (py).
    Always recompute wait/run/turnaround/bsld from submit/start/end so both
    sides are derived identically."""
    wait = rec["start"] - rec["submit"]
    run = rec["end"] - rec["start"]
    turnaround = rec["end"] - rec["submit"]
    return {
        "wait": wait,
        "run": run,
        "turnaround": turnaround,
        "bsld": bsld(wait, run),
    }


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


# ─── Core comparison ──────────────────────────────────────────────────────────

def compare(label, swf_path, rst_path, events_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    ground_truth = parse_swf(swf_path)
    py = parse_rst(rst_path)
    cpp = parse_cpp_events(events_path)

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

    # Sanity cross-check: reqProc from python rst vs ground-truth swf,
    # on the matched set, to make sure job ids really line up.
    mismatches = 0
    for jid in matched:
        if jid in ground_truth and py[jid]["reqProc"] != ground_truth[jid]["reqProc"]:
            mismatches += 1
    out(f"reqProc mismatches (py vs swf, sanity check): {mismatches}/{len(matched)}")

    if not matched:
        out("No overlapping completed jobs -- cannot compute deviation stats.")
        return None

    py_metrics = {jid: build_job_metrics(py[jid]) for jid in matched}
    cpp_metrics = {jid: build_job_metrics(cpp[jid]) for jid in matched}

    fields = ["wait", "run", "turnaround", "bsld"]
    diffs = {f: np.array([cpp_metrics[j][f] - py_metrics[j][f] for j in matched]) for f in fields}
    py_arr = {f: np.array([py_metrics[j][f] for j in matched]) for f in fields}
    cpp_arr = {f: np.array([cpp_metrics[j][f] for j in matched]) for f in fields}

    out("\n--- per-job deviation (cpp - python), over matched jobs ---")
    for f in fields:
        d = diffs[f]
        mae = float(np.mean(np.abs(d)))
        rmse = float(np.sqrt(np.mean(d ** 2)))
        corr = float(np.corrcoef(py_arr[f], cpp_arr[f])[0, 1]) if np.std(py_arr[f]) > 0 and np.std(cpp_arr[f]) > 0 else float("nan")
        out(f"  {f:10s}  MAE={mae:12.3f}  RMSE={rmse:12.3f}  mean_diff={float(np.mean(d)):12.3f}  "
            f"max_abs_diff={float(np.max(np.abs(d))):12.3f}  corr={corr:.4f}")

    out("\n--- aggregate metrics computed identically on both sides ---")
    out(f"{'metric':12s} {'python':>14s} {'cpp':>14s} {'abs diff':>12s} {'rel diff %':>10s}")
    for f in fields:
        p_mean = float(np.mean(py_arr[f]))
        c_mean = float(np.mean(cpp_arr[f]))
        rel = 100.0 * (c_mean - p_mean) / p_mean if p_mean != 0 else float("nan")
        out(f"{f:12s} {p_mean:14.3f} {c_mean:14.3f} {abs(c_mean - p_mean):12.3f} {rel:10.2f}")

    # ─── Plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f"cqsim C++ vs Python -- {label} (n={len(matched)} matched jobs)")

    def scatter_ax(ax, f, unit=""):
        x, y = py_arr[f], cpp_arr[f]
        ax.scatter(x, y, s=4, alpha=0.25, color="#1f77b4", linewidths=0)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="#d62728", lw=1, ls="--", label="y = x")
        ax.set_xlabel(f"python {f} {unit}")
        ax.set_ylabel(f"cpp {f} {unit}")
        ax.set_title(f.capitalize())
        ax.legend(fontsize=8)

    scatter_ax(axes[0, 0], "wait", "(s)")
    scatter_ax(axes[0, 1], "run", "(s)")
    scatter_ax(axes[1, 0], "bsld")

    ax = axes[1, 1]
    ax.hist(diffs["wait"], bins=60, color="#2ca02c", alpha=0.8)
    ax.axvline(0, color="#d62728", lw=1, ls="--")
    ax.set_xlabel("wait_cpp - wait_py (s)")
    ax.set_ylabel("job count")
    ax.set_title("Wait-time deviation")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(output_dir, f"{label}_deviation.png")
    fig.savefig(fig_path, dpi=DPI)
    plt.close(fig)
    out(f"\nwrote {fig_path}")

    # CDF overlay of wait time
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    for arr, name, color in [(py_arr["wait"], "python", "#1f77b4"), (cpp_arr["wait"], "cpp", "#ff7f0e")]:
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax2.plot(xs, ys, label=name, color=color)
    ax2.set_xscale("symlog")
    ax2.set_xlabel("wait time (s)")
    ax2.set_ylabel("CDF")
    ax2.set_title(f"Wait-time CDF -- {label}")
    ax2.legend()
    fig2.tight_layout()
    fig2_path = os.path.join(output_dir, f"{label}_wait_cdf.png")
    fig2.savefig(fig2_path, dpi=DPI)
    plt.close(fig2)
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


PRESETS = [
    dict(
        label="theta21cln",
        swf_path="data/theta21cln.swf",
        rst_path="cqsim-python/data/Results/theta21cln_regulate.rst",
        events_path="results/cqsimpy-test1/FCFS/events.csv",
    ),
    dict(
        label="polaris24cln",
        swf_path="data/polaris24cln.swf",
        rst_path="cqsim-python/data/Results/polaris24cln_regulate.rst",
        events_path="results/cqsimpt-test2/FCFS/events.csv",
    ),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", action="store_true",
                     help="run both built-in experiments (cqsimpy-test1 / cqsimpy-test2) shipped in this repo")
    ap.add_argument("--cpp-dir", help="cpp experiment output_dir, e.g. results/cqsimpy-test1")
    ap.add_argument("--cpp-tag", help="driver tag subfolder under --cpp-dir, e.g. FCFS")
    ap.add_argument("--py-rst", help="path to the python .rst result file")
    ap.add_argument("--swf", help="path to the shared .swf input trace")
    ap.add_argument("--label", default="comparison", help="name used for output files/titles")
    ap.add_argument("--output-dir", default="results/cpp_vs_py", help="where to write plots/report")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    jobs = []
    if args.preset or not (args.cpp_dir and args.py_rst and args.swf):
        if not args.preset:
            print("Missing --cpp-dir/--py-rst/--swf; falling back to --preset experiments.\n")
        for p in PRESETS:
            jobs.append(dict(p, output_dir=os.path.join(args.output_dir, p["label"])))
    else:
        jobs.append(dict(
            label=args.label,
            swf_path=args.swf,
            rst_path=args.py_rst,
            events_path=os.path.join(args.cpp_dir, args.cpp_tag, "events.csv"),
            output_dir=os.path.join(args.output_dir, args.label),
        ))

    results = []
    for j in jobs:
        for key in ("swf_path", "rst_path", "events_path"):
            if not os.path.exists(j[key]):
                print(f"[skip {j['label']}] missing file: {j[key]}", file=sys.stderr)
                break
        else:
            r = compare(j["label"], j["swf_path"], j["rst_path"], j["events_path"], j["output_dir"])
            if r:
                results.append(r)
            print()

    if len(results) > 1:
        print("=== Overall summary ===")
        for r in results:
            wait_mae = float(np.mean(np.abs(r["diffs"]["wait"])))
            bsld_mae = float(np.mean(np.abs(r["diffs"]["bsld"])))
            print(f"{r['label']:15s} matched={r['matched']:6d}  "
                  f"only_py={r['only_py']:5d}  only_cpp={r['only_cpp']:5d}  "
                  f"wait_MAE={wait_mae:10.2f}s  bsld_MAE={bsld_mae:8.4f}")


if __name__ == "__main__":
    main()
