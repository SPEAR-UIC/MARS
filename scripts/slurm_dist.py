#!/usr/bin/env python3
"""slurm_dist.py: run MARS experiments as per-driver Slurm jobs.

Each experiment JSON is split into one single-driver config per driver. Every
driver runs as its own process, writes a heartbeat status file while it runs,
and the results are merged and summarised once all drivers have finished.

Subcommands
-----------
  prepare    [--smoke] [--merge] [--run-id ID] <exp> [<exp> ...]
             Create results/slurm_runs/<RUN_ID>/ with per-driver configs, a
             manifest, initial status files and heavy.list / light.list.
             Prints the run directory on stdout (all other output -> stderr).
             --merge marks the run as adding drivers to an output_dir that
             already holds results (see aggregate).

  run-driver <run_dir> <key> [--interval SECONDS]
             Run one driver (called from the sbatch scripts). Streams the
             simulator output to drivers/<key>/run.log and rewrites
             status/<key>.json every --interval seconds.

  status     [<run_dir>|latest] [--watch SECONDS] [--stale SECONDS]
             Per-driver progress table (state, %, elapsed, ETA, node, job).

  aggregate  [<run_dir>|latest] [--merge]
             Merge per-driver outputs into each experiment's output_dir and
             write target-job metrics (with --merge, or for a run prepared
             with --merge, rows for tags this run did not produce are kept
             and experiment.log is appended to instead of replaced):
               <output_dir>/summary.csv            C++ all-jobs metrics
               <output_dir>/summary_target.{csv,json}
               <run_dir>/aggregate/summary.{csv,json,md}   all experiments

<exp> is a path to an experiment JSON or a bare name resolved to
experiments/<exp>.json. Only the Python standard library is used.
"""

import argparse
import csv
import datetime
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(REPO_DIR, "results", "slurm_runs")
BINARY = os.environ.get("MARS_BINARY", os.path.join(REPO_DIR, "cqsimcpp"))
HEAVY_TYPES = {"mcts", "adaptive_mcts"}
SKIP_TYPES = {"rlscheduler"}  # not simulated by cqsimcpp

PROGRESS_RE = re.compile(r"(\d+)%\s+\((\d+)/(\d+)\)")
TARGET_RE = re.compile(r"target\s*=.*?(\d+)\s+jobs.*?ids\s+(\d+)\s*-\s*(\d+)")

SMOKE_LEAD_JOBS = 100
SMOKE_MAX_SUBMITS = 400
SMOKE_TIME_LIMIT_MS = 200
SMOKE_NUM_CORES = 4


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def write_json_atomic(path, obj):
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (IOError, OSError, ValueError):
        return default


def resolve_exp_json(arg):
    if os.path.isfile(arg):
        return os.path.abspath(arg)
    cand = os.path.join(REPO_DIR, "experiments", arg + ".json")
    if os.path.isfile(cand):
        return cand
    raise SystemExit("Cannot find experiment JSON for: %s" % arg)


def resolve_run_dir(arg):
    if arg in (None, "", "latest"):
        path = os.path.join(RUNS_DIR, "latest")
    elif os.path.isdir(arg):
        path = arg
    elif os.path.isdir(os.path.join(RUNS_DIR, arg)):
        path = os.path.join(RUNS_DIR, arg)
    else:
        raise SystemExit("Run directory not found: %s" % arg)
    path = os.path.realpath(path)
    if not os.path.isfile(os.path.join(path, "manifest.json")):
        raise SystemExit("No manifest.json in %s" % path)
    return path


def repo_path(p):
    return p if os.path.isabs(p) else os.path.join(REPO_DIR, p)


# ── SWF helpers ───────────────────────────────────────────────────────────────

def read_swf(path):
    """Return (header_info, jobs) where jobs maps id -> dict(submit, run, procs, walltime)."""
    info = {"max_procs": None, "target_lo": None, "target_hi": None, "target_count": None}
    jobs = {}
    with open(path) as f:
        for line in f:
            if line.startswith(";"):
                body = line[1:].strip()
                if body.startswith("MaxProcs:"):
                    info["max_procs"] = int(body.split(":", 1)[1])
                m = TARGET_RE.search(body)
                if m:
                    info["target_count"] = int(m.group(1))
                    info["target_lo"] = int(m.group(2))
                    info["target_hi"] = int(m.group(3))
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                jid = int(parts[0])
                submit = int(parts[1])
                run = int(parts[3])
                alloc = int(parts[4])
                req = int(parts[7])
                wall = int(parts[8])
            except ValueError:
                continue
            jobs[jid] = {"submit": submit, "run": run,
                         "procs": req if req > 0 else alloc, "walltime": wall}
    return info, jobs


def target_ids(info, jobs):
    lo, hi = info["target_lo"], info["target_hi"]
    if lo is None:
        return sorted(jobs)
    return sorted(j for j in jobs if lo <= j <= hi)


# ── prepare ───────────────────────────────────────────────────────────────────

def smoke_start_index(swf_path):
    info, jobs = read_swf(swf_path)
    order = sorted(jobs, key=lambda j: (jobs[j]["submit"], j))
    tids = set(target_ids(info, jobs))
    for idx, jid in enumerate(order):
        if jid in tids:
            return max(0, idx - SMOKE_LEAD_JOBS)
    return 0


def cmd_prepare(args):
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.smoke:
        run_id += "-smoke"
    run_dir = os.path.join(RUNS_DIR, run_id)
    if os.path.exists(run_dir):
        raise SystemExit("Run directory already exists: %s" % run_dir)
    for sub in ("configs", "status", "drivers", "logs"):
        os.makedirs(os.path.join(run_dir, sub))

    manifest = {"run_id": run_id, "run_dir": run_dir, "repo_dir": REPO_DIR,
                "created": time.time(), "smoke": bool(args.smoke),
                "merge": bool(args.merge) and not args.smoke,
                "experiments": {}, "drivers": []}
    heavy, light = [], []

    for arg in args.experiments:
        json_path = resolve_exp_json(arg)
        exp = os.path.splitext(os.path.basename(json_path))[0]
        if exp in manifest["experiments"]:
            raise SystemExit("Experiment listed twice: %s" % exp)
        with open(json_path) as f:
            cfg = json.load(f)

        swf = repo_path(cfg["swf_path"])
        if not os.path.isfile(swf):
            raise SystemExit("%s: swf_path not found: %s" % (exp, swf))
        base = dict(cfg)
        base["swf_path"] = swf
        if "maintenance_csv" in base:
            base["maintenance_csv"] = repo_path(base["maintenance_csv"])
        if args.smoke:
            base["start_job_index"] = smoke_start_index(swf)
            base["max_submits"] = SMOKE_MAX_SUBMITS
            out_dir = os.path.join(run_dir, "aggregate", exp)
        else:
            out_dir = repo_path(cfg.get("output_dir", os.path.join("results", exp)))

        manifest["experiments"][exp] = {"json_path": json_path, "swf_path": swf,
                                        "output_dir": out_dir}

        for i, drv in enumerate(cfg["drivers"]):
            dtype = drv.get("type", "")
            tag = drv.get("tag", dtype)
            if dtype in SKIP_TYPES:
                log("  %s: skipping driver %d (%s): not simulated by cqsimcpp" % (exp, i, tag))
                continue
            drv = dict(drv)
            kind = "heavy" if dtype in HEAVY_TYPES else "light"
            if args.smoke and kind == "heavy":
                drv["time_limit_ms"] = SMOKE_TIME_LIMIT_MS
                drv["num_cores"] = min(int(drv.get("num_cores", 1)), SMOKE_NUM_CORES)
            cores = int(drv.get("num_cores", 1)) if kind == "heavy" else 1

            key = "%s_d%d" % (exp, i)
            outdir = os.path.join(run_dir, "drivers", key)
            single = dict(base)
            single["drivers"] = [drv]
            single["output_dir"] = outdir
            cfg_path = os.path.join(run_dir, "configs", key + ".json")
            write_json_atomic(cfg_path, single)

            entry = {"key": key, "exp": exp, "index": i, "tag": tag, "type": dtype,
                     "kind": kind, "cores": cores, "config": cfg_path, "outdir": outdir}
            manifest["drivers"].append(entry)
            (heavy if kind == "heavy" else light).append(key)

            status = dict(entry)
            status.update({"state": "pending", "updated": time.time()})
            write_json_atomic(os.path.join(run_dir, "status", key + ".json"), status)

    write_json_atomic(os.path.join(run_dir, "manifest.json"), manifest)
    for name, keys in (("heavy.list", heavy), ("light.list", light)):
        with open(os.path.join(run_dir, name), "w") as f:
            f.write("".join(k + "\n" for k in keys))

    latest = os.path.join(RUNS_DIR, "latest")
    if os.path.islink(latest) or not os.path.exists(latest):
        if os.path.islink(latest):
            os.remove(latest)
        os.symlink(run_id, latest)

    log("Prepared %s: %d heavy + %d light drivers across %d experiment(s)%s"
        % (run_dir, len(heavy), len(light), len(manifest["experiments"]),
           " [SMOKE]" if args.smoke else ""))
    print(run_dir)


# ── run-driver ────────────────────────────────────────────────────────────────

def read_progress(log_path):
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", "replace")
    except (IOError, OSError):
        return None
    matches = PROGRESS_RE.findall(tail.replace("\r", "\n"))
    if not matches:
        return None
    pct, done, total = matches[-1]
    return int(pct), int(done), int(total)


def slurm_job_label():
    arr, task = os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    if arr and task:
        return "%s_%s" % (arr, task)
    return os.environ.get("SLURM_JOB_ID", "local")


def cmd_run_driver(args):
    run_dir = resolve_run_dir(args.run_dir)
    manifest = load_json(os.path.join(run_dir, "manifest.json"))
    drv = next((d for d in manifest["drivers"] if d["key"] == args.key), None)
    if drv is None:
        raise SystemExit("Driver %s not in manifest" % args.key)

    status_path = os.path.join(run_dir, "status", args.key + ".json")
    os.makedirs(drv["outdir"], exist_ok=True)
    log_path = os.path.join(drv["outdir"], "run.log")

    env = dict(os.environ)
    if drv["kind"] == "heavy":
        env.update({"OMP_NUM_THREADS": str(drv["cores"]),
                    "OMP_PROC_BIND": "spread", "OMP_PLACES": "cores"})

    status = dict(drv)
    status.update({"state": "running", "node": socket.gethostname().split(".")[0],
                   "slurm_job": slurm_job_label(), "started": time.time(),
                   "ended": None, "exit_code": None, "pct": 0, "done": 0,
                   "total": None, "log": log_path})

    def refresh(state=None):
        now = time.time()
        prog = read_progress(log_path)
        if prog:
            status["pct"], status["done"], status["total"] = prog
        if state:
            status["state"] = state
        status["updated"] = now
        status["elapsed_s"] = round((status["ended"] or now) - status["started"], 1)
        write_json_atomic(status_path, status)

    cancelled = {"flag": False}
    logf = open(log_path, "w")
    child = subprocess.Popen([BINARY, "run", "config", drv["config"]],
                             cwd=REPO_DIR, env=env, stdout=logf, stderr=subprocess.STDOUT)
    status["pid"] = child.pid

    def on_signal(signum, _frame):
        cancelled["flag"] = True
        try:
            child.send_signal(signal.SIGTERM)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    refresh()
    log("[%s] started %s (%s) pid=%d on %s" % (args.key, drv["tag"], drv["kind"],
                                               child.pid, status["node"]))
    next_beat = time.time() + args.interval
    while child.poll() is None:
        time.sleep(1)
        if time.time() >= next_beat:
            refresh()
            next_beat = time.time() + args.interval

    rc = child.returncode
    logf.close()
    status["ended"] = time.time()
    status["exit_code"] = rc
    final = "cancelled" if cancelled["flag"] else ("done" if rc == 0 else "failed")
    refresh(final)
    log("[%s] %s (exit %s) after %.0fs" % (args.key, final, rc, status["elapsed_s"]))
    return 0 if rc == 0 else (rc if rc > 0 else 1)


# ── status ────────────────────────────────────────────────────────────────────

def fmt_dur(sec):
    if sec is None or sec < 0:
        return "-"
    sec = int(sec)
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    if d:
        return "%dd%02dh" % (d, h)
    if h:
        return "%dh%02dm" % (h, m)
    return "%dm%02ds" % (m, sec % 60)


def squeue_jobs():
    """Set of job labels currently known to Slurm, or None if squeue is unavailable."""
    if not shutil.which("squeue"):
        return None
    user = os.environ.get("USER", "")
    try:
        out = subprocess.check_output(["squeue", "-h", "-r", "-u", user, "-o", "%i"],
                                      stderr=subprocess.DEVNULL, universal_newlines=True)
    except (subprocess.CalledProcessError, OSError):
        return None
    return set(line.strip() for line in out.splitlines() if line.strip())


COLORS = {"running": "\033[0;36m", "done": "\033[0;32m", "failed": "\033[0;31m",
          "lost": "\033[0;31m", "cancelled": "\033[1;33m", "pending": "\033[0;37m",
          "stale?": "\033[1;33m"}


def render_status(run_dir, stale_after, color):
    manifest = load_json(os.path.join(run_dir, "manifest.json"))
    jobs = load_json(os.path.join(run_dir, "slurm_jobs.json"), {}) or {}
    queued = squeue_jobs()
    now = time.time()
    lines = []
    title = "MARS run %s%s   (%s)" % (manifest["run_id"], " [SMOKE]" if manifest.get("smoke") else "",
                                     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append(title)
    if jobs:
        lines.append("Slurm jobs: " + "  ".join("%s=%s" % (k, v) for k, v in sorted(jobs.items())))
    counts = {}
    hdr = "  %-10s %-10s %-24s %6s  %-15s %9s %9s  %-22s %s" % (
        "Tag", "State", "Progress", "", "Jobs done", "Elapsed", "ETA", "Node", "Slurm job")

    by_exp = {}
    for d in manifest["drivers"]:
        by_exp.setdefault(d["exp"], []).append(d)

    for exp, drivers in by_exp.items():
        lines.append("")
        lines.append("== %s" % exp)
        lines.append(hdr)
        for d in drivers:
            st = load_json(os.path.join(run_dir, "status", d["key"] + ".json"), {}) or {}
            state = st.get("state", "pending")
            age = now - st.get("updated", now)
            if state == "running" and age > stale_after:
                if queued is None:
                    state = "stale?"
                elif st.get("slurm_job") not in queued:
                    state = "lost"
            counts[state] = counts.get(state, 0) + 1

            done, total = st.get("done") or 0, st.get("total")
            frac = (float(done) / total) if total else 0.0
            if state == "done":
                frac = 1.0 if not total else frac
            bar_n = int(round(frac * 20))
            bar = "[" + "#" * bar_n + "." * (20 - bar_n) + "]"
            elapsed = None
            if st.get("started"):
                elapsed = (st.get("ended") or now) - st["started"]
            eta = None
            if state == "running" and elapsed and 0 < frac < 1:
                eta = elapsed * (1 - frac) / frac
            jobs_txt = "%d/%s" % (done, total) if total else "-"
            state_txt = "%-10s" % state
            if color:
                state_txt = COLORS.get(state, "") + state_txt + "\033[0m"
            lines.append("  %-10s %s %-24s %5.1f%%  %-15s %9s %9s  %-22s %s" % (
                d["tag"], state_txt, bar, frac * 100, jobs_txt, fmt_dur(elapsed),
                fmt_dur(eta) if eta is not None else "-", st.get("node", "-"),
                st.get("slurm_job", "-")))

    lines.append("")
    lines.append("Totals: " + "  ".join("%s=%d" % (k, v) for k, v in sorted(counts.items()))
                 + "  (of %d drivers)" % len(manifest["drivers"]))
    agg = os.path.join(run_dir, "aggregate", "summary.md")
    if os.path.isfile(agg):
        lines.append("Aggregated summary: %s" % agg)
    return "\n".join(lines)


def cmd_status(args):
    run_dir = resolve_run_dir(args.run_dir)
    color = sys.stdout.isatty() and not args.no_color
    if not args.watch:
        print(render_status(run_dir, args.stale, color))
        return 0
    try:
        while True:
            out = render_status(run_dir, args.stale, color)
            sys.stdout.write("\033[2J\033[H" + out + "\n\n(refreshing every %ds, Ctrl-C to exit)\n"
                             % args.watch)
            sys.stdout.flush()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


# ── aggregate ─────────────────────────────────────────────────────────────────

def parse_events(path):
    submit, start, end = {}, {}, {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ev = row.get("event", "")
            if ev not in ("Submit", "Run", "Backfill", "End"):
                continue
            try:
                jid = int(row["id"])
                t = float(row["sim_time"])
            except (KeyError, TypeError, ValueError):
                continue
            if ev == "Submit":
                submit.setdefault(jid, t)
            elif ev == "End":
                end[jid] = t
            else:
                start[jid] = t
    return submit, start, end


def percentile(sorted_vals, q):
    """Linear-interpolation percentile (numpy's default) of an already sorted list."""
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def mean(vals):
    return sum(vals) / len(vals) if vals else None


METRIC_COLS = ["wait_avg_h", "wait_p50_h", "wait_p90_h", "wait_p99_h", "wait_max_h",
               "bsld_avg", "bsld_p50", "bsld_p99", "util_target_window", "target_span_h"]
LOWER_BETTER = {"wait_avg_h", "wait_p50_h", "wait_p90_h", "wait_p99_h", "wait_max_h",
                "bsld_avg", "bsld_p50", "bsld_p99", "target_span_h"}


def target_metrics(events_path, info, jobs, tids):
    submit, start, end = parse_events(events_path)
    waits, bslds, t_submits, t_ends = [], [], [], []
    for jid in tids:
        sub = submit.get(jid, jobs[jid]["submit"])
        t_submits.append(sub)
        if jid not in start or jid not in end:
            continue
        w = max(0.0, start[jid] - sub)
        run = end[jid] - start[jid]
        waits.append(w / 3600.0)
        bslds.append(max(1.0, (w + run) / max(run, 10.0)))
        t_ends.append(end[jid])

    res = {"target_jobs": len(tids), "completed": len(waits)}
    waits.sort()
    bslds.sort()
    res.update({"wait_avg_h": mean(waits), "wait_p50_h": percentile(waits, 50),
                "wait_p90_h": percentile(waits, 90), "wait_p99_h": percentile(waits, 99),
                "wait_max_h": waits[-1] if waits else None,
                "bsld_avg": mean(bslds), "bsld_p50": percentile(bslds, 50),
                "bsld_p99": percentile(bslds, 99)})

    util, span = None, None
    if t_submits and t_ends and info["max_procs"]:
        t0, t1 = min(t_submits), max(t_ends)
        if t1 > t0:
            used = 0.0
            for jid, s in start.items():
                e = end.get(jid)
                if e is None or jid not in jobs:
                    continue
                overlap = min(e, t1) - max(s, t0)
                if overlap > 0:
                    used += overlap * jobs[jid]["procs"]
            util = used / (info["max_procs"] * (t1 - t0))
            span = (t1 - t0) / 3600.0
    res["util_target_window"] = util
    res["target_span_h"] = span
    return res


def fmt_num(v, nd=3):
    if v is None:
        return ""
    return ("%." + str(nd) + "f") % v


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: (fmt_num(r[c], 4) if isinstance(r.get(c), float) else r.get(c, ""))
                        for c in cols})


def markdown_table(exp, rows, target_note):
    cols = [("Driver", "tag"), ("State", "state"), ("Done", "completed"),
            ("Avg wait (h)", "wait_avg_h"), ("P50 wait (h)", "wait_p50_h"),
            ("P99 wait (h)", "wait_p99_h"), ("Max wait (h)", "wait_max_h"),
            ("Avg BSLD", "bsld_avg"), ("P99 BSLD", "bsld_p99"),
            ("Util", "util_target_window")]
    best = {}
    for _, key in cols[3:]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            best[key] = min(vals) if key in LOWER_BETTER else max(vals)
    out = ["### %s" % exp, "", target_note, "",
           "| " + " | ".join(c for c, _ in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for _, key in cols:
            v = r.get(key)
            if key == "completed":
                cells.append("%s/%s" % (v, r.get("target_jobs")))
            elif isinstance(v, float):
                s = "%.1f%%" % (v * 100) if key == "util_target_window" else "%.2f" % v
                cells.append("**%s**" % s if best.get(key) == v else s)
            else:
                cells.append(str(v if v is not None else "-"))
        out.append("| " + " | ".join(cells) + " |")

    baselines = [r for r in rows if r["type"] not in HEAVY_TYPES and r.get("wait_avg_h") is not None]
    mars = [r for r in rows if r["type"] in HEAVY_TYPES and r.get("wait_avg_h") is not None]
    if baselines and mars:
        out.append("")
        for key, label in (("wait_avg_h", "avg wait"), ("wait_p99_h", "P99 wait"),
                           ("bsld_avg", "avg BSLD"), ("util_target_window", "util")):
            lower = key in LOWER_BETTER
            bvals = [b for b in baselines if b.get(key) is not None]
            if not bvals:
                continue
            ref = (min if lower else max)(bvals, key=lambda b: b[key])
            parts = []
            for m in mars:
                if m.get(key) is None or not ref[key]:
                    continue
                delta = (m[key] - ref[key]) / ref[key] * 100.0
                parts.append("%s %+.1f%%" % (m["tag"], delta))
            if parts:
                out.append("- %s vs best baseline (%s): %s" % (label, ref["tag"], ", ".join(parts)))
    out.append("")
    return out


def cmd_aggregate(args):
    run_dir = resolve_run_dir(args.run_dir)
    manifest = load_json(os.path.join(run_dir, "manifest.json"))
    agg_dir = os.path.join(run_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    merge = bool(getattr(args, "merge", False) or manifest.get("merge"))
    if merge:
        log("Merge mode: keeping existing rows for tags not in this run")
    all_rows = []
    md = ["# MARS results: run %s%s" % (manifest["run_id"], " [SMOKE]" if manifest.get("smoke") else ""), ""]
    target_cols = ["exp", "tag", "type", "state", "complete", "target_jobs", "completed"] + METRIC_COLS

    for exp, meta in manifest["experiments"].items():
        dest = meta["output_dir"]
        os.makedirs(dest, exist_ok=True)
        info, jobs = read_swf(meta["swf_path"])
        tids = target_ids(info, jobs)
        if info["target_lo"] is not None:
            note = "Target jobs: ids %d-%d (%d jobs in trace, header says %d)" % (
                info["target_lo"], info["target_hi"], len(tids), info["target_count"])
            if not manifest.get("smoke") and info["target_count"] and len(tids) != info["target_count"]:
                log("WARNING %s: found %d target jobs, header says %d" % (exp, len(tids), info["target_count"]))
        else:
            note = "No target range in SWF header: metrics cover all %d jobs" % len(tids)
        log("== %s: %s" % (exp, note))

        exp_log = os.path.join(dest, "experiment.log")
        run_tags = {d["tag"] for d in manifest["drivers"] if d["exp"] == exp}
        kept_rows, kept_summary, kept_header = [], [], None
        if merge:
            old = load_json(os.path.join(dest, "summary_target.json"), {}) or {}
            kept_rows = [r for r in old.get("drivers", []) if r.get("tag") not in run_tags]
            old_sum = os.path.join(dest, "summary.csv")
            if os.path.isfile(old_sum):
                with open(old_sum) as f:
                    r = list(csv.reader(f))
                if r:
                    kept_header = r[0]
                    kept_summary = [x for x in r[1:] if x and x[0] not in run_tags]
            log("  keeping %d existing driver(s): %s" % (
                len(kept_rows), ", ".join(r.get("tag", "?") for r in kept_rows) or "none"))
        else:
            with open(exp_log, "w"):
                pass
        summary_rows, summary_header = [], None
        rows = []
        for d in (d for d in manifest["drivers"] if d["exp"] == exp):
            st = load_json(os.path.join(run_dir, "status", d["key"] + ".json"), {}) or {}
            src = os.path.join(d["outdir"], d["tag"])
            row = {"exp": exp, "tag": d["tag"], "type": d["type"],
                   "state": st.get("state", "pending")}

            if os.path.isdir(src):
                tgt = os.path.join(dest, d["tag"])
                if os.path.realpath(tgt) != os.path.realpath(src):
                    if os.path.isdir(tgt):
                        shutil.rmtree(tgt)
                    shutil.copytree(src, tgt)
            dlog = os.path.join(d["outdir"], "experiment.log")
            if os.path.isfile(dlog):
                with open(exp_log, "a") as out, open(dlog) as inp:
                    out.write("# -- driver %d: %s --\n" % (d["index"], d["tag"]))
                    out.write(inp.read())
            dsum = os.path.join(d["outdir"], "summary.csv")
            if os.path.isfile(dsum):
                with open(dsum) as f:
                    r = list(csv.reader(f))
                if r:
                    summary_header = r[0]
                    summary_rows.extend(r[1:])

            events = os.path.join(src, "events.csv")
            if os.path.isfile(events) and tids:
                row.update(target_metrics(events, info, jobs, tids))
            else:
                row.update({"target_jobs": len(tids), "completed": 0})
                log("  %-10s missing events.csv (state=%s)" % (d["tag"], row["state"]))
            row["complete"] = (row["state"] == "done" and row["completed"] == row["target_jobs"]
                               and row["target_jobs"] > 0)
            rows.append(row)
            log("  %-10s %-9s %5d/%d target jobs  avg wait %s h" % (
                d["tag"], row["state"], row["completed"], row["target_jobs"],
                fmt_num(row.get("wait_avg_h"), 2) or "-"))

        summary_header = summary_header or kept_header
        summary_rows = kept_summary + summary_rows
        rows = kept_rows + rows
        if summary_header:
            with open(os.path.join(dest, "summary.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(summary_header)
                w.writerows(summary_rows)
        write_csv(os.path.join(dest, "summary_target.csv"), rows, target_cols)
        write_json_atomic(os.path.join(dest, "summary_target.json"),
                          {"exp": exp, "swf": meta["swf_path"], "note": note,
                           "max_procs": info["max_procs"], "drivers": rows})
        md.extend(markdown_table(exp, rows, note))
        all_rows.extend(rows)
        log("  merged into %s" % dest)

    md.extend(["Metric definitions: wait = start - submit over all target jobs (zeros included);",
               "BSLD = max(1, (wait + run) / max(run, 10 s)); Util = core-seconds of all jobs",
               "running inside [first target submit, last target end] / (MaxProcs x window).",
               "Bold = best value per column. Percentages compare each MARS driver",
               "with the best baseline for that metric (negative = lower).", ""])
    write_csv(os.path.join(agg_dir, "summary.csv"), all_rows, target_cols)
    write_json_atomic(os.path.join(agg_dir, "summary.json"),
                      {"run_id": manifest["run_id"], "generated": time.time(), "drivers": all_rows})
    with open(os.path.join(agg_dir, "summary.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    incomplete = [r for r in all_rows if not r["complete"]]
    log("Wrote %s/summary.{csv,json,md}; %d/%d drivers complete"
        % (agg_dir, len(all_rows) - len(incomplete), len(all_rows)))
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("prepare", help="create a run directory with per-driver configs")
    sp.add_argument("experiments", nargs="+")
    sp.add_argument("--smoke", action="store_true",
                    help="tiny run: ~%d jobs around the first target job, short MCTS budget"
                         % SMOKE_MAX_SUBMITS)
    sp.add_argument("--merge", action="store_true",
                    help="add these drivers to existing results in output_dir")
    sp.add_argument("--run-id")
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("run-driver", help="run one driver with heartbeat status updates")
    sp.add_argument("run_dir")
    sp.add_argument("key")
    sp.add_argument("--interval", type=int, default=30, help="heartbeat seconds (default 30)")
    sp.set_defaults(func=cmd_run_driver)

    sp = sub.add_parser("status", help="show per-driver progress")
    sp.add_argument("run_dir", nargs="?", default="latest")
    sp.add_argument("--watch", type=int, default=0, metavar="SECONDS")
    sp.add_argument("--stale", type=int, default=300, metavar="SECONDS",
                    help="heartbeat age after which a running driver is checked against squeue")
    sp.add_argument("--no-color", action="store_true")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("aggregate", help="merge outputs and compute target-job metrics")
    sp.add_argument("run_dir", nargs="?", default="latest")
    sp.add_argument("--merge", action="store_true",
                    help="keep existing rows for tags not in this run")
    sp.set_defaults(func=cmd_aggregate)

    args = p.parse_args()
    if not getattr(args, "func", None):
        p.print_help()
        return 1
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
