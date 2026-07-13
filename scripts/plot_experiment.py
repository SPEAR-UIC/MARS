#!/usr/bin/env python3
"""Dispatch to the correct cleaned-trace plotter for an experiment config.

Single config:
    python3 scripts/plot_experiment.py experiments/exp2b.json

Multiple configs (merged plot — configs must share the same trace/SWF):
    python3 scripts/plot_experiment.py experiments/exp2b.json experiments/exp1b.json
    python3 scripts/plot_experiment.py experiments/exp2b.json experiments/exp1b.json --plots-dir results/custom
    python3 scripts/plot_experiment.py experiments/exp2b.json experiments/exp1b.json --merge-dir results/my_merge

Single experiment, globals only
    python3 scripts/plot_experiment.py experiments/exp2b.json --global-only

Merged experiments, globals only
    python3 scripts/plot_experiment.py experiments/exp2b.json experiments/exp1b.json --global-only
"""

import argparse
import csv
import importlib
import json
import logging
import math
import os
import re
import shutil
import sys
from pathlib import Path

_LOG = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from plot_results import infer_trace_family, load_experiment_config, parse_events, parse_swf

REPO_DIR = Path(__file__).resolve().parent.parent

_PLOTTER_MODULES = {
    "theta": "thetacln_plot",
    "polaris": "polariscln_plot",
}

_COMPARABLE_FIELDS = [
    "swf_path", "start_job_index", "max_submits", "mode", "maintenance_csv",
]

_STAT_COLUMNS = ["Min", "P25", "P50", "P75", "P85", "P95", "P99", "Max"]
_RELATIVE_BASELINES = ("fcfs", "wfp3")
_MAINT_META_COLUMNS = {
    "Maintenance_Start",
    "Maintenance_End",
    "Total_Maintenance_Hours",
    "Type",
}
_IGNORED_TABLE_DRIVERS = {"MARS-IU"}

# Single-column figure styling, matched to plot_avg_wait_vs_window*.py.
_FIG_W = 3.45
_STAT_FIG_H = 2.65
_RADAR_FIG_H = 3.15
_MAINT_FIG_H = 3.55
_UTIL_FIG_H = 2.45
_FIG_DPI = 300
_AXIS_FONT = 8.7
_TICK_FONT = 6.1
_RADAR_RADIAL_FONT = 4.4
_LEGEND_FONT = 6.3
_LINE_WIDTH = 0.75
_MARKER_SIZE = 2.4
_TABLE_ARTIFACT_DIR = "tables"

_HEURISTIC_POLICY_COLORS = {
    "FCFS": "#000000",
    "SJF": "#1f77b4",
    "WFP3": "#aec7e8",
    "F1": "#ff7f0e",
    "UNICEP": "#ffbb78",
    "FAT": "#2ca02c",
    "LRF": "#98df8a",
    "LJF": "#d62728",
}
_MARS_VARIANT_COLORS = {
    "CW": "#9467bd",
    "CB": "#e377c2",
    "CU": "#17becf",
    "IU": "#8c564b",
    "T1": "#9467bd",
    "T2.5": "#8c564b",
    "T5": "#e377c2",
    "T15": "#008b8b",
    "T30": "#bcbd22",
    "T60": "#17becf",
}
_KNOWN_DRIVER_COLORS = {
    "RLScheduler": "#6a6a6a",
}
_FALLBACK_COLORS = (
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#6699CC",
    "#661100",
    "#AA4466",
    "#4477AA",
    "#228833",
    "#EE6677",
    "#AA3377",
    "#BBBBBB",
)
_TABLE_LABEL_ORDER = {
    "overall": 0,
    "XS": 10,
    "S": 11,
    "M": 12,
    "L": 13,
    "T1": 20,
    "T2": 21,
    "T3": 22,
    "T4": 23,
    "T5": 24,
    "small": 90,
    "large": 91,
}
_ODD_PERCENT_GUIDES = tuple(value / 100.0 for value in range(-90, 100, 20) if value != 0)
_SUMMARY_STAT_COLUMNS = ("P25", "P50", "P75", "P85", "P95", "P99", "Mean")
_WINDOWED_DRIVER_RE = re.compile(r"^(?P<policy>.+)-w(?P<window>\d+)$")
_GROUP_STEP = 1.0
_GROUP_WIDTH = 0.76
_BAR_WIDTH_SCALE = 1.0
_SINGLE_BAR_WIDTH = 0.58
_HEATMAP_GROUP_BINS = {
    "polaris": (
        ("XS", 10, 24),
        ("S", 25, 32),
        ("M", 33, 128),
        ("L", 129, 496),
    ),
    "theta": (
        ("XS", 128, 128),
        ("S", 129, 256),
        ("M", 257, 1024),
        ("L", 1025, 4096),
    ),
}
_OBSOLETE_GROUP_LABELS = {
    "polaris": ("S", "M", "L", "XL", "small", "large"),
    "theta": ("S", "M", "L", "XL", "T1", "T2", "T3", "T4", "T5", "small", "large"),
}
_STAT_HEADERS = ["Min", "P25", "P50", "P75", "P85", "P95", "P99", "Max", "Mean", "StdDev", "GeometricMean"]
_HEATMAP_WALLTIME_BINS = [
    ("<=0.5h", None, 0.5),
    ("<=1h", 0.5, 1.0),
    ("<=2h", 1.0, 2.0),
    ("<=3h", 2.0, 3.0),
    ("<=5h", 3.0, 5.0),
    ("<=6h", 5.0, 6.0),
    ("<=9h", 6.0, 9.0),
    ("<=24h", 9.0, 24.0),
]
_SYSTEM_HEATMAP_LAYOUTS = {
    "polaris": {
        "node_bins": [
            ("=10", 10, 10, 10),
            ("<=16", 11, 16, 16),
            ("<=24", 17, 24, 24),
            ("=25", 25, 25, 25),
            ("<=32", 26, 32, 32),
            ("<=128", 33, 128, 128),
            ("<=496", 129, 496, 496),
        ],
        "dividers": [2, 4, 5],
        "size_labels": ["XS", "S", "M", "L"],
        "total_nodes": 496,
    },
    "theta": {
        "node_bins": [
            ("=128", 128, 128, 128),
            ("<=255", 129, 255, 255),
            ("=256", 256, 256, 256),
            ("<=1024", 257, 1024, 1024),
            ("<=2048", 1025, 2048, 2048),
            ("<=4096", 2049, 4096, 4096),
        ],
        "dividers": [0, 2, 3],
        "size_labels": ["XS", "S", "M", "L"],
        "total_nodes": 4096,
    },
}
_RELATIVE_HEATMAP_METRICS = (
    ("wait", "Mean Wait Improvement vs FCFS", "heatmap_wait_relative_fcfs.png"),
    ("bsld", "Mean Bounded Slowdown Improvement vs FCFS", "heatmap_bsld_relative_fcfs.png"),
)
_RELATIVE_HEATMAP_COLOR_LIMIT = 100.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Auto-select the right plotter for a cleaned-trace experiment JSON. "
                    "Pass multiple configs to merge them before plotting."
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="One or more experiment JSON configs. Multiple configs are merged before plotting.",
    )
    parser.add_argument(
        "--plots-dir",
        dest="plots_dir",
        help="Optional output directory override for generated plots",
    )
    parser.add_argument(
        "--merge-dir",
        dest="merge_dir",
        help="Directory to store the merged results symlinks and config. "
             "Defaults to results/plotmerge_<exp1>_<exp2>...",
    )
    parser.add_argument(
        "--global-only",
        dest="global_only",
        action="store_true",
        help="Skip per-driver plots; generate only global comparison plots and tables",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Override the output_dir from the experiment config JSON",
    )
    parser.add_argument(
        "--print-plotter",
        action="store_true",
        help="Print the selected plotter module and exit without plotting",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging",
    )
    return parser.parse_args(argv)


def _check_compatible(base_cfg, other_cfg, base_name, other_name):
    base_family = infer_trace_family(base_cfg)
    other_family = infer_trace_family(other_cfg)
    if base_family != other_family:
        raise SystemExit(
            f"cannot merge '{other_name}' into '{base_name}': "
            f"trace families differ ({base_family} vs {other_family})"
        )
    mismatches = [f for f in _COMPARABLE_FIELDS if base_cfg.get(f) != other_cfg.get(f)]
    if mismatches:
        raise SystemExit(
            f"cannot merge '{other_name}' into '{base_name}': "
            f"config mismatch in {', '.join(mismatches)}"
        )


def _remove(path: Path):
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _build_merged(config_paths: list[Path], merge_dir: Path):
    """Create merge_dir with per-driver symlinks and a plotmerge.json config.

    Returns (merged_config_dict, merged_config_path).
    """
    configs = [load_experiment_config(str(p)) for p in config_paths]
    exp_names = [p.stem for p in config_paths]

    base_cfg = configs[0]
    for other_cfg, other_name in zip(configs[1:], exp_names[1:]):
        _check_compatible(base_cfg, other_cfg, exp_names[0], other_name)

    merge_dir.mkdir(parents=True, exist_ok=True)

    merged_cfg = dict(base_cfg)
    merged_cfg["output_dir"] = os.path.relpath(merge_dir, REPO_DIR)
    merged_cfg["plot_merge_sources"] = [str(p) for p in config_paths]

    merged_drivers = []
    seen_tags: dict[str, str] = {}

    for cfg, exp_name in zip(configs, exp_names):
        src_output_dir = Path(cfg["output_dir"])
        if not src_output_dir.is_absolute():
            src_output_dir = REPO_DIR / src_output_dir

        for driver in cfg.get("drivers", []):
            tag = driver.get("tag", driver["type"])
            if tag in seen_tags:
                raise SystemExit(
                    f"duplicate driver tag '{tag}' found in both '{seen_tags[tag]}' "
                    f"and '{exp_name}'; rename one before merging"
                )
            seen_tags[tag] = exp_name
            merged_drivers.append(driver)

            src_driver_dir = src_output_dir / tag
            dest_driver_dir = merge_dir / tag
            if not src_driver_dir.exists():
                _LOG.warning("Missing results for %s:%s -> %s", exp_name, tag, src_driver_dir)
                continue
            _remove(dest_driver_dir)
            dest_driver_dir.symlink_to(src_driver_dir.resolve())

    merged_cfg["drivers"] = merged_drivers

    merged_config_path = merge_dir / "plotmerge.json"
    with open(merged_config_path, "w") as f:
        json.dump(merged_cfg, f, indent=2)
        f.write("\n")

    return merged_cfg, merged_config_path


def _load_plotting_libs():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    return plt, mticker, np


def _parse_numeric(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def _driver_order_from_config(config):
    order = []
    seen = set()
    for driver in config.get("drivers", []):
        if not driver.get("plot", True):
            continue
        tag = driver.get("tag", driver.get("type"))
        if tag and tag not in _IGNORED_TABLE_DRIVERS and tag not in seen:
            seen.add(tag)
            order.append(tag)
    return order


def _append_unseen_driver(order, seen, driver):
    if driver and driver not in _IGNORED_TABLE_DRIVERS and driver not in seen:
        seen.add(driver)
        order.append(driver)


def _collect_table_drivers(plots_dir: Path, base_order):
    order = list(base_order)
    seen = set(order)

    for path in _discover_stat_tables(plots_dir) + [
        _table_path_for(plots_dir, "table_util_overall.csv"),
        _table_path_for(plots_dir, "table_util_announcement_to_start.csv"),
    ]:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            if "Driver" in reader.fieldnames:
                for row in reader:
                    _append_unseen_driver(order, seen, row.get("Driver", "").strip())
            else:
                for field in reader.fieldnames:
                    if field not in _MAINT_META_COLUMNS:
                        _append_unseen_driver(order, seen, field)
    return order


def _split_windowed_driver(driver: str):
    match = _WINDOWED_DRIVER_RE.match(driver)
    if not match:
        return driver, None
    return match.group("policy"), int(match.group("window"))


def _driver_policy_label(driver: str) -> str:
    policy, _ = _split_windowed_driver(driver)
    return policy


def _driver_legend_label(driver: str) -> str:
    return _driver_policy_label(driver)


def _mars_variant(driver: str):
    if driver.startswith("MARS-") or driver.startswith("MCTS-"):
        return driver.split("-", 1)[1]
    return None


def _driver_color(driver: str):
    policy = _driver_policy_label(driver)
    if policy in _HEURISTIC_POLICY_COLORS:
        return _HEURISTIC_POLICY_COLORS[policy]

    variant = _mars_variant(driver)
    if variant in _MARS_VARIANT_COLORS:
        return _MARS_VARIANT_COLORS[variant]

    return _KNOWN_DRIVER_COLORS.get(driver)


def _build_color_map(driver_order):
    colors = {}
    fallback_idx = 0
    for driver in driver_order:
        known_color = _driver_color(driver)
        if known_color is not None:
            colors[driver] = known_color
            continue
        colors[driver] = _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
        fallback_idx += 1
    return colors


def _plots_dir_for_run(config, args) -> Path:
    if args.plots_dir:
        return Path(args.plots_dir).resolve()
    output_dir = str(args.output_dir or config["output_dir"])
    return Path(output_dir.rstrip("/") + "_plots").resolve()


def _resolve_repo_path(path_value) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_DIR / path


def _heatmap_group_bins(family: str):
    return _HEATMAP_GROUP_BINS.get(family, ())


def _assign_heatmap_group(nodes_used: int, family: str) -> str | None:
    for label, lo, hi in _heatmap_group_bins(family):
        if lo <= nodes_used <= hi:
            return label
    return None


def _analysis_window_for_module(module):
    return None, None


def _filter_jobs_for_analysis(module, submit, start, end, analysis_start, analysis_end):
    if analysis_start is None or analysis_end is None:
        return submit, start, end
    if hasattr(module, "filter_jobs_by_submit_window"):
        return module.filter_jobs_by_submit_window(submit, start, end, analysis_start, analysis_end)

    allowed = {
        job_id
        for job_id, submit_time in submit.items()
        if analysis_start <= submit_time < analysis_end
    }
    return (
        {job_id: submit_time for job_id, submit_time in submit.items() if job_id in allowed},
        {job_id: start_time for job_id, start_time in start.items() if job_id in allowed},
        {job_id: end_time for job_id, end_time in end.items() if job_id in allowed},
    )


def _system_heatmap_layout(family: str):
    return _SYSTEM_HEATMAP_LAYOUTS.get(family)


def _heatmap_node_labels(node_bins, total_nodes):
    labels = []
    for label, _, _, upper in node_bins:
        frac = 100.0 * upper / total_nodes
        labels.append(f"{label}\n({frac:.1f}%)")
    return labels


def _heatmap_row_index(nodes_used: int, node_bins) -> int | None:
    for idx, (_, lo, hi, _) in enumerate(node_bins):
        if lo <= nodes_used <= hi:
            return idx
    return None


def _heatmap_col_index(walltime_h: float) -> int | None:
    for idx, (_, lo, hi) in enumerate(_HEATMAP_WALLTIME_BINS):
        if lo is None:
            if walltime_h <= hi:
                return idx
        elif lo < walltime_h <= hi:
            return idx
    return None


def _load_driver_timings(driver_cfg, results_dir: Path, module):
    driver_type = driver_cfg.get("type")
    tag = driver_cfg.get("tag", driver_type)
    if driver_type == "rlscheduler" or "csv_path" in driver_cfg:
        csv_path = _resolve_repo_path(driver_cfg.get("csv_path", ""))
        if not csv_path.exists():
            return None
        return module.parse_rlscheduler_csv(str(csv_path))

    events_path = results_dir / tag / "events.csv"
    if not events_path.exists():
        return None
    return parse_events(str(events_path))


def _collect_windowed_driver_jobs(config, results_dir: Path, module, *, skip_ignored: bool):
    swf_path = _resolve_repo_path(config["swf_path"])
    procs_map, walltimes_map, _, _ = parse_swf(str(swf_path))
    analysis_start, analysis_end = _analysis_window_for_module(module)

    valid_tags = []
    submit_maps = {}
    start_maps = {}
    end_maps = {}

    for driver_cfg in config.get("drivers", []):
        if not driver_cfg.get("plot", True):
            continue
        tag = driver_cfg.get("tag", driver_cfg.get("type"))
        if not tag:
            continue
        if skip_ignored and tag in _IGNORED_TABLE_DRIVERS:
            continue

        timings = _load_driver_timings(driver_cfg, results_dir, module)
        if timings is None:
            continue
        submit, start, end = timings
        if not end:
            continue

        submit_f, start_f, end_f = _filter_jobs_for_analysis(
            module,
            submit,
            start,
            end,
            analysis_start,
            analysis_end,
        )
        if not end_f and (analysis_start is not None or analysis_end is not None):
            _LOG.warning("  %s: no jobs in analysis window, using all available data", tag)
            submit_f, start_f, end_f = submit, start, end
        if not end_f:
            continue

        valid_tags.append(tag)
        submit_maps[tag] = submit_f
        start_maps[tag] = start_f
        end_maps[tag] = end_f

    if not valid_tags:
        return None

    # Use the latest submit time of a completed job per driver, then take the
    # minimum across all drivers as the common window end.  This ensures every
    # driver is compared on exactly the same set of submitted jobs even when
    # some drivers have fewer completed jobs (partial results).
    driver_max_submit = {tag: max(submit_maps[tag].values()) for tag in valid_tags}
    common_submit_end = min(driver_max_submit.values())

    job_counts = {tag: len(submit_maps[tag]) for tag in valid_tags}
    if len(set(job_counts.values())) > 1:
        _LOG.warning(
            "Partial results detected — restricting to jobs submitted up to t=%.0f "
            "(earliest max-submit time across all drivers).",
            common_submit_end,
        )
        for tag in valid_tags:
            _LOG.info("  %s: %d completed jobs, max submit = %.0f",
                      tag, job_counts[tag], driver_max_submit[tag])

    jobs_by_driver = {}
    for tag in valid_tags:
        driver_jobs = []
        submit = submit_maps[tag]
        start = start_maps[tag]
        end = end_maps[tag]
        for job_id, submit_time in submit.items():
            if submit_time > common_submit_end:
                continue
            if job_id not in start or job_id not in end:
                continue
            nodes_used = procs_map.get(job_id)
            walltime_seconds = walltimes_map.get(job_id)
            if nodes_used is None or walltime_seconds is None or walltime_seconds <= 0:
                continue
            wait_seconds = start[job_id] - submit[job_id]
            wait_hours = wait_seconds / 3600.0
            walltime_h = walltime_seconds / 3600.0
            bsld = (wait_seconds + walltime_seconds) / walltime_seconds
            driver_jobs.append((nodes_used, walltime_h, wait_hours, bsld))
        jobs_by_driver[tag] = driver_jobs
        _LOG.debug("  %s: %d jobs in common submit window", tag, len(driver_jobs))

    return {
        "valid_tags": valid_tags,
        "jobs_by_driver": jobs_by_driver,
        "common_submit_end": common_submit_end,
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
    }


def _compute_stats_row(values):
    import numpy as np

    arr = np.array(values, dtype=float)
    pos = arr[arr > 0]
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


def _write_stat_table(path: Path, driver_tags, data_map, *, multiplier: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Driver"] + _STAT_HEADERS)
        for tag in driver_tags:
            values = data_map.get(tag, [])
            if not values:
                writer.writerow([tag] + ["N/A"] * len(_STAT_HEADERS))
                continue
            scaled = [value * multiplier for value in values]
            writer.writerow([tag] + _compute_stats_row(scaled))


def _remove_obsolete_group_outputs(plots_dir: Path, family: str) -> None:
    for label in _OBSOLETE_GROUP_LABELS.get(family, ()):
        for metric in ("wait", "bsld"):
            root_table = plots_dir / f"table_{label}_{metric}.csv"
            artifact_dir = plots_dir / _TABLE_ARTIFACT_DIR / f"{label}_{metric}"
            _remove(root_table)
            _remove(artifact_dir)


def _rewrite_group_stat_tables(config, plots_dir: Path, results_dir: Path, family: str, module) -> None:
    group_bins = _heatmap_group_bins(family)
    if not group_bins:
        return

    _remove_obsolete_group_outputs(plots_dir, family)

    bundle = _collect_windowed_driver_jobs(config, results_dir, module, skip_ignored=True)
    if bundle is None:
        return

    valid_tags = bundle["valid_tags"]
    jobs_by_driver = bundle["jobs_by_driver"]
    group_labels = [label for label, _, _ in group_bins]
    all_wait_by_group = {tag: {label: [] for label in group_labels} for tag in valid_tags}
    all_bsld_by_group = {tag: {label: [] for label in group_labels} for tag in valid_tags}

    for tag in valid_tags:
        for nodes_used, _walltime_h, wait_hours, bsld in jobs_by_driver[tag]:
            label = _assign_heatmap_group(nodes_used, family)
            if label is None:
                continue
            all_wait_by_group[tag][label].append(wait_hours)
            all_bsld_by_group[tag][label].append(bsld)

    for label in group_labels:
        wait_path = plots_dir / f"table_{label}_wait.csv"
        bsld_path = plots_dir / f"table_{label}_bsld.csv"
        _write_stat_table(
            wait_path,
            valid_tags,
            {tag: all_wait_by_group[tag][label] for tag in valid_tags},
            multiplier=60.0,
        )
        _write_stat_table(
            bsld_path,
            valid_tags,
            {tag: all_bsld_by_group[tag][label] for tag in valid_tags},
            multiplier=1.0,
        )


def _aggregate_driver_heatmap_metrics(np, jobs, node_bins):
    n_rows = len(node_bins)
    n_cols = len(_HEATMAP_WALLTIME_BINS)
    counts = np.zeros((n_rows, n_cols), dtype=int)
    wait_sum = np.zeros((n_rows, n_cols), dtype=float)
    bsld_sum = np.zeros((n_rows, n_cols), dtype=float)

    for nodes_used, walltime_h, wait_hours, bsld in jobs:
        row_idx = _heatmap_row_index(nodes_used, node_bins)
        col_idx = _heatmap_col_index(walltime_h)
        if row_idx is None or col_idx is None:
            continue
        counts[row_idx, col_idx] += 1
        wait_sum[row_idx, col_idx] += wait_hours
        bsld_sum[row_idx, col_idx] += bsld

    wait_mean = np.full((n_rows, n_cols), np.nan, dtype=float)
    bsld_mean = np.full((n_rows, n_cols), np.nan, dtype=float)
    nonzero = counts > 0
    wait_mean[nonzero] = wait_sum[nonzero] / counts[nonzero]
    bsld_mean[nonzero] = bsld_sum[nonzero] / counts[nonzero]

    return {
        "counts": counts,
        "wait": wait_mean,
        "bsld": bsld_mean,
    }


def _relative_improvement_grid(np, driver_stats, baseline_stats):
    relative = np.full(driver_stats.shape, np.nan, dtype=float)
    for row_idx in range(driver_stats.shape[0]):
        for col_idx in range(driver_stats.shape[1]):
            value = driver_stats[row_idx, col_idx]
            baseline = baseline_stats[row_idx, col_idx]
            if not (math.isfinite(value) and math.isfinite(baseline)):
                continue
            fallback = abs(value - baseline)
            relative[row_idx, col_idx] = 100.0 * _relative_scalar_to_baseline(
                value,
                baseline,
                fallback,
                lower_is_better=True,
            )
    return relative


def _heatmap_text_color(cmap, norm, value):
    rgba = cmap(norm(value))
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if luminance < 0.52 else "black"


def _save_relative_driver_heatmap(driver, metric_title, output_path: Path, layout, counts, relative_grid):
    plt, _mticker, np = _load_plotting_libs()
    from matplotlib import colors as mcolors

    masked_grid = np.ma.masked_invalid(relative_grid)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#f3f3f3")
    norm = mcolors.TwoSlopeNorm(
        vmin=-_RELATIVE_HEATMAP_COLOR_LIMIT,
        vcenter=0.0,
        vmax=_RELATIVE_HEATMAP_COLOR_LIMIT,
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.35), dpi=_FIG_DPI)
    im = ax.imshow(masked_grid, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    for row_idx in range(counts.shape[0]):
        for col_idx in range(counts.shape[1]):
            count = int(counts[row_idx, col_idx])
            value = relative_grid[row_idx, col_idx]
            if math.isfinite(value):
                text = f"{count:,}\n{value:+.1f}%"
                color = _heatmap_text_color(cmap, norm, value)
            elif count > 0:
                text = f"{count:,}\nN/A"
                color = "#303030"
            else:
                text = "0"
                color = "#666666"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=7.2,
                color=color,
                linespacing=1.25,
                fontweight="bold",
            )

    wt_labels = [label for label, _, _ in _HEATMAP_WALLTIME_BINS]
    node_bins = layout["node_bins"]
    ax.set_xticks(range(len(_HEATMAP_WALLTIME_BINS)))
    ax.set_xticklabels(wt_labels, fontsize=7.6, fontweight="bold")
    ax.set_yticks(range(len(node_bins)))
    ax.set_yticklabels(
        _heatmap_node_labels(node_bins, layout["total_nodes"]),
        fontsize=7.1,
        fontweight="bold",
    )
    ax.set_title(f"{driver} - {metric_title}", fontsize=10.2, pad=6, fontweight="bold")
    ax.set_xlabel("Requested Walltime", fontsize=9.0, fontweight="bold")
    ax.set_ylabel("Nodes Used", fontsize=9.0, fontweight="bold")
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(len(_HEATMAP_WALLTIME_BINS)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(node_bins)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    n_cols = len(_HEATMAP_WALLTIME_BINS)
    for divider in layout.get("dividers", []):
        y_div = divider + 0.5
        ax.plot(
            [-0.5, n_cols - 0.5 + 0.28],
            [y_div, y_div],
            color="black",
            linestyle="--",
            linewidth=1.1,
            clip_on=False,
            zorder=10,
        )

    boundaries = [-0.5] + [divider + 0.5 for divider in sorted(layout.get("dividers", []))] + [len(node_bins) - 0.5]
    x_label = n_cols - 0.5 + 0.38
    for size_label, lo, hi in zip(layout["size_labels"], boundaries[:-1], boundaries[1:]):
        ax.text(
            x_label,
            (lo + hi) / 2,
            size_label,
            ha="left",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            clip_on=False,
            color="black",
        )

    ticks = np.linspace(-_RELATIVE_HEATMAP_COLOR_LIMIT, _RELATIVE_HEATMAP_COLOR_LIMIT, 5)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.11, fraction=0.07, ticks=ticks)
    cbar.set_label("% Improvement vs FCFS", fontsize=8.2, labelpad=2)
    cbar.ax.set_xticklabels([f"{tick:.0f}%" for tick in ticks])
    cbar.ax.tick_params(labelsize=7.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_FIG_DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return output_path


def _create_relative_driver_heatmaps(config, plots_dir: Path, results_dir: Path, family: str, module):
    layout = _system_heatmap_layout(family)
    if layout is None:
        return []

    bundle = _collect_windowed_driver_jobs(config, results_dir, module, skip_ignored=False)
    if bundle is None:
        return []

    valid_tags = bundle["valid_tags"]
    if "FCFS" not in valid_tags:
        _LOG.warning("Driver heatmaps skipped: FCFS baseline not available")
        return []

    _plt, _mticker, np = _load_plotting_libs()
    node_bins = layout["node_bins"]
    aggregated = {
        tag: _aggregate_driver_heatmap_metrics(np, bundle["jobs_by_driver"][tag], node_bins)
        for tag in valid_tags
    }
    baseline = aggregated["FCFS"]

    relative_grids = {tag: {} for tag in valid_tags}
    for metric, _title, _filename in _RELATIVE_HEATMAP_METRICS:
        for tag in valid_tags:
            grid = _relative_improvement_grid(np, aggregated[tag][metric], baseline[metric])
            relative_grids[tag][metric] = grid

    outputs = []
    for tag in valid_tags:
        driver_dir = plots_dir / tag
        counts = aggregated[tag]["counts"]
        for metric, metric_title, filename in _RELATIVE_HEATMAP_METRICS:
            outputs.append(
                _save_relative_driver_heatmap(
                    tag,
                    metric_title,
                    driver_dir / filename,
                    layout,
                    counts,
                    relative_grids[tag][metric],
                )
            )
    return outputs


def _stat_table_label(path: Path, metric: str) -> str:
    stem = path.stem
    if stem == f"table_{metric}_overall":
        return "overall"
    prefix = "table_"
    suffix = f"_{metric}"
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix):-len(suffix)]
    return stem


def _stat_table_sort_key(path: Path):
    metric = "wait" if path.stem.endswith("_wait") or path.stem == "table_wait_overall" else "bsld"
    label = _stat_table_label(path, metric)
    return _TABLE_LABEL_ORDER.get(label, 1000), label


def _discover_stat_tables(plots_dir: Path):
    candidates = []
    for metric in ("wait", "bsld"):
        overall = plots_dir / f"table_{metric}_overall.csv"
        if overall.exists():
            candidates.append(overall)
        candidates.extend(sorted(plots_dir.glob(f"table_*_{metric}.csv"), key=_stat_table_sort_key))
        candidates.extend(
            sorted(
                (plots_dir / _TABLE_ARTIFACT_DIR).glob(f"*/table_*_{metric}.csv"),
                key=_stat_table_sort_key,
            )
        )

    paths = []
    seen = set()
    for path in candidates:
        slug = _table_slug(path)
        if slug not in seen:
            seen.add(slug)
            paths.append(path)
    return paths


def _table_slug(path: Path) -> str:
    stem = path.stem
    return stem[len("table_"):] if stem.startswith("table_") else stem


def _table_artifact_path(table_path: Path) -> Path:
    """Return the artifact directory path without creating it."""
    if table_path.parent.parent.name == _TABLE_ARTIFACT_DIR:
        return table_path.parent
    return table_path.parent / _TABLE_ARTIFACT_DIR / _table_slug(table_path)


def _table_artifact_dir(table_path: Path) -> Path:
    """Return the artifact directory path, creating it if it does not exist."""
    directory = _table_artifact_path(table_path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _relative_artifact_dir(table_path: Path, baseline_key: str) -> Path:
    directory = _table_artifact_dir(table_path) / f"relative_{baseline_key}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _clean_table_artifacts(table_path: Path) -> None:
    directory = _table_artifact_path(table_path)
    if not directory.exists():
        return
    for filename in (
        "parallel.png",
        "bar.png",
        "bar_summary.png",
        "bar_min_to_p75.png",
        "bar_p85_to_max.png",
        "radar.png",
    ):
        path = directory / filename
        if path.exists():
            path.unlink()
    for baseline_key in _RELATIVE_BASELINES:
        baseline_dir = directory / f"relative_{baseline_key}"
        if baseline_dir.exists():
            shutil.rmtree(baseline_dir)


def _clean_legacy_table_artifacts(plots_dir: Path) -> None:
    for path in plots_dir.glob("paper_*.png"):
        if path.is_file():
            path.unlink()
    old_table_dir = plots_dir / "paper_tables"
    if old_table_dir.exists():
        shutil.rmtree(old_table_dir)


def _table_path_for(plots_dir: Path, filename: str) -> Path:
    root_path = plots_dir / filename
    if root_path.exists():
        return root_path
    grouped_path = plots_dir / _TABLE_ARTIFACT_DIR / _table_slug(Path(filename)) / filename
    return grouped_path


def _move_table_to_artifact_dir(table_path: Path) -> Path:
    output_path = _table_artifact_dir(table_path) / table_path.name
    if table_path.resolve() == output_path.resolve():
        return output_path
    if output_path.exists():
        output_path.unlink()
    shutil.move(str(table_path), str(output_path))
    return output_path


def _move_root_tables_to_artifact_dirs(plots_dir: Path):
    moved = []
    for table_path in sorted(plots_dir.glob("table_*.csv")):
        moved.append(_move_table_to_artifact_dir(table_path))
    return moved


def _group_x_positions(np, count):
    return np.arange(count, dtype=float) * _GROUP_STEP


def _set_group_x_limits(ax, x_positions):
    if len(x_positions) == 0:
        return
    if len(x_positions) == 1:
        half_step = 0.5
    else:
        half_step = (x_positions[1] - x_positions[0]) * 0.5
    ax.set_xlim(x_positions[0] - half_step, x_positions[-1] + half_step)


def _rows_without_driver(rows, driver_to_skip: str):
    return [(driver, values) for driver, values in rows if driver != driver_to_skip]


def _metric_for_stat_table(path: Path) -> str:
    stem = path.stem
    if stem == "table_wait_overall" or stem.endswith("_wait"):
        return "wait"
    return "bsld"


def _load_stat_table(path: Path, driver_order):
    metric = _metric_for_stat_table(path)
    transform = (lambda value: value / 60.0) if metric == "wait" else (lambda value: value)
    rows_by_driver = {}
    stats_by_driver = {}

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Driver" not in reader.fieldnames:
            return [], metric, []
        if any(column not in reader.fieldnames for column in _STAT_COLUMNS):
            return [], metric, []
        for row in reader:
            driver = row.get("Driver", "").strip()
            if not driver or driver in _IGNORED_TABLE_DRIVERS:
                continue
            values = []
            for column in _STAT_COLUMNS:
                value = _parse_numeric(row.get(column))
                if value is None:
                    values = []
                    break
                values.append(transform(value))
            if values:
                rows_by_driver[driver] = values
                
                stats_dict = {}
                for col in [*_SUMMARY_STAT_COLUMNS, "StdDev"]:
                    val = _parse_numeric(row.get(col))
                    stats_dict[col] = transform(val) if val is not None else 0.0
                stats_by_driver[driver] = stats_dict

    ordered = [
        (driver, rows_by_driver[driver])
        for driver in driver_order
        if driver in rows_by_driver
    ]
    ordered.extend(
        (driver, rows_by_driver[driver])
        for driver in rows_by_driver
        if driver not in driver_order
    )

    ordered_stats = [
        (driver, stats_by_driver[driver])
        for driver in driver_order
        if driver in stats_by_driver
    ]
    ordered_stats.extend(
        (driver, stats_by_driver[driver])
        for driver in stats_by_driver
        if driver not in driver_order
    )
    return ordered, metric, ordered_stats


def _log_e_formatter(value: float, _pos: int) -> str:
    if value <= 0:
        return ""
    exponent = int(math.floor(math.log10(value)))
    scale = 10 ** exponent
    mantissa = value / scale
    rounded = round(mantissa)
    if not math.isclose(mantissa, rounded, rel_tol=0.0, abs_tol=1e-9):
        return ""
    if rounded not in (1, 2):
        return ""
    return f"{rounded}e{exponent}"


def _apply_compact_log_ticks(ax, mticker) -> None:
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_log_e_formatter))
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def _clip_rows_for_log(rows):
    positives = [value for _, values in rows for value in values if value > 0]
    floor = min(positives) * 0.5 if positives else 1e-6
    floor = max(floor, 1e-12)
    clipped = [
        (driver, [value if value > 0 else floor for value in values])
        for driver, values in rows
    ]
    return clipped, floor


def _set_log_limits(ax, rows, floor):
    values = [value for _, row_values in rows for value in row_values]
    if not values:
        return
    upper = max(values) * 1.35
    if upper <= floor:
        upper = floor * 2.0
    ax.set_ylim(floor * 0.75, upper)


def _baseline_driver_name(driver_names, baseline_key: str):
    if baseline_key == "fcfs":
        return "FCFS" if "FCFS" in driver_names else None
    if baseline_key == "wfp3":
        if "WFP3" in driver_names:
            return "WFP3"
        if "WFP3-w256" in driver_names:
            return "WFP3-w256"
        for driver in driver_names:
            if driver.startswith("WFP3"):
                return driver
    return None


def _available_relative_baselines(driver_names):
    baselines = []
    for baseline_key in _RELATIVE_BASELINES:
        driver = _baseline_driver_name(driver_names, baseline_key)
        if driver is not None:
            baselines.append((baseline_key, driver))
    return baselines


def _relative_metric_label(metric: str) -> str:
    if metric == "wait":
        return "Wait (h)"
    if metric == "bsld":
        return "Slowdown"
    if metric == "util":
        return "Utilization"
    return metric


def _relative_ylabel(baseline_driver: str, metric: str | None = None) -> str:
    label = f"% Improvement vs {_driver_legend_label(baseline_driver)}"
    if metric is not None:
        label += f" ({_relative_metric_label(metric)})"
    return label


def _relative_rows_to_baseline(rows, baseline_driver: str, *, lower_is_better: bool):
    baseline_values = None
    for driver, values in rows:
        if driver == baseline_driver:
            baseline_values = values
            break
    if baseline_values is None:
        return []

    fallback_denoms = []
    for column_idx, baseline in enumerate(baseline_values):
        deltas = []
        for _, values in rows:
            value = values[column_idx]
            delta = baseline - value if lower_is_better else value - baseline
            deltas.append(abs(delta))
        fallback_denoms.append(max(deltas) if deltas else 0.0)

    relative_rows = []
    for driver, values in rows:
        relative_values = []
        for column_idx, value in enumerate(values):
            baseline = baseline_values[column_idx]
            delta = baseline - value if lower_is_better else value - baseline
            if not math.isclose(baseline, 0.0, rel_tol=0.0, abs_tol=1e-12):
                relative_values.append(delta / abs(baseline))
            elif fallback_denoms[column_idx] > 0:
                relative_values.append(delta / fallback_denoms[column_idx])
            else:
                relative_values.append(0.0)
        relative_rows.append((driver, relative_values))
    return relative_rows


def _relative_scalar_to_baseline(value, baseline, fallback_denom, *, lower_is_better: bool = False):
    delta = baseline - value if lower_is_better else value - baseline
    if not math.isclose(baseline, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return delta / abs(baseline)
    if fallback_denom > 0:
        return delta / fallback_denom
    return 0.0


def _set_relative_y_limits(ax, values):
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        ax.set_ylim(-0.05, 0.05)
        return
    lo = min(finite_values + [0.0])
    hi = max(finite_values + [0.0])
    span = hi - lo
    pad = max(0.04, 0.12 * span) if span > 0 else 0.05
    ax.set_ylim(lo - pad, hi + pad)


def _relative_limits(values):
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return -0.05, 0.05
    lo = min(finite_values + [0.0])
    hi = max(finite_values + [0.0])
    span = hi - lo
    pad = max(0.04, 0.12 * span) if span > 0 else 0.05
    return lo - pad, hi + pad


def _apply_odd_percent_guides(ax, mticker, *, axis="y"):
    lo, hi = ax.get_ylim()
    major_ticks = list(ax.get_yticks())
    guide_ticks = [
        value
        for value in _ODD_PERCENT_GUIDES
        if lo <= value <= hi
        and not any(math.isclose(value, tick, rel_tol=0.0, abs_tol=1e-9) for tick in major_ticks)
    ]
    if not guide_ticks:
        return
    ax.yaxis.set_minor_locator(mticker.FixedLocator(guide_ticks))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    if axis == "both":
        ax.grid(True, which="minor", color="#ececec", linewidth=0.34)
    else:
        ax.grid(True, which="minor", axis=axis, color="#ececec", linewidth=0.34)


def _reorder_handles_for_row_major(handles, labels, ncol):
    rows = [
        list(zip(handles[i:i + ncol], labels[i:i + ncol]))
        for i in range(0, len(handles), ncol)
    ]
    ordered = []
    max_cols = max((len(row) for row in rows), default=0)
    for col_idx in range(max_cols):
        for row in rows:
            if col_idx < len(row):
                ordered.append(row[col_idx])
    return [handle for handle, _ in ordered], [label for _, label in ordered]


def _prepare_legend_entries(handles, labels, *, omit_labels=()):
    omitted = {_driver_legend_label(label) for label in omit_labels}
    legend_map = {}
    ordered_labels = []

    for handle, label in zip(handles, labels):
        display_label = _driver_legend_label(label)
        if display_label in omitted or display_label in legend_map:
            continue
        legend_map[display_label] = handle
        ordered_labels.append(display_label)

    return [legend_map[label] for label in ordered_labels], ordered_labels


def _add_top_legend(fig, handles, labels, *, omit_labels=()):
    handles, labels = _prepare_legend_entries(handles, labels, omit_labels=omit_labels)
    if not handles:
        return 0.92
    ncol = min(4, len(handles))
    rows = max(1, math.ceil(len(handles) / ncol))
    handles, labels = _reorder_handles_for_row_major(handles, labels, ncol)
    font_size = max(4.8, _LEGEND_FONT - max(0, rows - 2) * 0.35)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=ncol,
        fontsize=font_size,
        frameon=False,
        handlelength=1.45,
        handletextpad=0.35,
        columnspacing=0.68,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    return max(0.50, 0.91 - rows * 0.065)


def _apply_plot_axes(ax, ylabel: str, *, log_y: bool = False, mticker=None):
    ax.set_ylabel(ylabel, fontsize=_AXIS_FONT, labelpad=2)
    ax.tick_params(axis="both", labelsize=_TICK_FONT)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", color="#d0d0d0", linewidth=0.55)
    if log_y:
        ax.set_yscale("log")
        _apply_compact_log_ticks(ax, mticker)
        ax.grid(True, which="minor", axis="y", color="#ebebeb", linewidth=0.40)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _stat_ylabel(metric: str) -> str:
    return "Wait Time (h)" if metric == "wait" else "Bounded Slowdown"


def _plot_parallel_stat_table(table_path: Path, rows, metric: str, color_map):
    plt, mticker, np = _load_plotting_libs()
    clipped_rows, floor = _clip_rows_for_log(rows)
    fig, ax = plt.subplots(figsize=(_FIG_W, _STAT_FIG_H), dpi=_FIG_DPI)
    x_positions = np.arange(len(_STAT_COLUMNS))

    for driver, values in clipped_rows:
        color = color_map.get(driver, "#333333")
        ax.plot(
            x_positions,
            values,
            label=driver,
            color=color,
            linestyle="--",
            linewidth=_LINE_WIDTH,
            marker="o",
            markersize=_MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.0,
            alpha=0.96,
        )

    _apply_plot_axes(ax, _stat_ylabel(metric), log_y=True, mticker=mticker)
    _set_log_limits(ax, clipped_rows, floor)
    ax.set_xlim(-0.2, len(_STAT_COLUMNS) - 0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(_STAT_COLUMNS, rotation=32, ha="right")
    ax.set_xlabel("Statistic", fontsize=_AXIS_FONT, labelpad=1)
    ax.grid(True, which="major", axis="x", color="#f0f0f0", linewidth=0.40)

    handles, labels = ax.get_legend_handles_labels()
    top = _add_top_legend(fig, handles, labels)
    fig.subplots_adjust(left=0.16, right=0.98, top=top, bottom=0.24)

    output_path = _table_artifact_dir(table_path) / "parallel.png"
    fig.savefig(output_path, dpi=_FIG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def _plot_bar_stat_table(table_path: Path, rows, metric: str, color_map):
    plt, mticker, np = _load_plotting_libs()
    outputs = []
    baselines = _available_relative_baselines([driver for driver, _ in rows])
    if not baselines:
        return _plot_raw_bar_stat_table(table_path, rows, metric, color_map)

    for baseline_key, baseline_driver in baselines:
        relative_rows = _relative_rows_to_baseline(rows, baseline_driver, lower_is_better=True)
        plot_rows = _rows_without_driver(relative_rows, baseline_driver)
        if not plot_rows:
            continue

        csv_path = _relative_artifact_dir(table_path, baseline_key) / table_path.name
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Driver"] + _STAT_COLUMNS)
            for driver, values in relative_rows:
                writer.writerow([driver] + [f"{(v*100):.4f}" for v in values])
        outputs.append(csv_path)

        fig, ax = plt.subplots(figsize=(_FIG_W, _STAT_FIG_H), dpi=_FIG_DPI)
        x_positions = _group_x_positions(np, len(_STAT_COLUMNS))
        n_drivers = max(1, len(plot_rows))
        group_width = _GROUP_WIDTH
        width = group_width / n_drivers

        for idx, (driver, values) in enumerate(plot_rows):
            color = color_map.get(driver, "#333333")
            offsets = x_positions - group_width / 2 + width * (idx + 0.5)
            ax.bar(
                offsets,
                values,
                width=width * _BAR_WIDTH_SCALE,
                label=driver,
                color=color,
                edgecolor="none",
                linewidth=0.0,
            )

        _apply_plot_axes(ax, _relative_ylabel(baseline_driver, metric))
        ax.axhline(0.0, color="#9c9c9c", linewidth=0.65, zorder=1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        _set_relative_y_limits(ax, [value for _, values in plot_rows for value in values])
        _apply_odd_percent_guides(ax, mticker)
        _set_group_x_limits(ax, x_positions)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(_STAT_COLUMNS, rotation=32, ha="right")
        ax.set_xlabel("Statistic", fontsize=_AXIS_FONT, labelpad=1)

        handles, labels = ax.get_legend_handles_labels()
        top = _add_top_legend(fig, handles, labels, omit_labels=(baseline_driver,))
        fig.subplots_adjust(left=0.16, right=0.98, top=top, bottom=0.24)

        output_path = _relative_artifact_dir(table_path, baseline_key) / "bar.png"
        fig.savefig(output_path, dpi=_FIG_DPI, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        outputs.append(output_path)

    return outputs


def _plot_raw_bar_stat_table(table_path: Path, rows, metric: str, color_map):
    plt, mticker, np = _load_plotting_libs()
    clipped_rows, floor = _clip_rows_for_log(rows)
    fig, ax = plt.subplots(figsize=(_FIG_W, _STAT_FIG_H), dpi=_FIG_DPI)
    x_positions = _group_x_positions(np, len(_STAT_COLUMNS))
    n_drivers = max(1, len(clipped_rows))
    group_width = _GROUP_WIDTH
    width = group_width / n_drivers
    bottom = floor * 0.75

    for idx, (driver, values) in enumerate(clipped_rows):
        color = color_map.get(driver, "#333333")
        offsets = x_positions - group_width / 2 + width * (idx + 0.5)
        heights = [max(value - bottom, bottom * 0.03) for value in values]
        ax.bar(
            offsets,
            heights,
            width=width * _BAR_WIDTH_SCALE,
            bottom=bottom,
            label=driver,
            color=color,
            edgecolor="none",
            linewidth=0.0,
        )

    _apply_plot_axes(ax, _stat_ylabel(metric), log_y=True, mticker=mticker)
    _set_log_limits(ax, clipped_rows, floor)
    _set_group_x_limits(ax, x_positions)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(_STAT_COLUMNS, rotation=32, ha="right")
    ax.set_xlabel("Statistic", fontsize=_AXIS_FONT, labelpad=1)

    handles, labels = ax.get_legend_handles_labels()
    top = _add_top_legend(fig, handles, labels)
    fig.subplots_adjust(left=0.16, right=0.98, top=top, bottom=0.24)

    output_path = _table_artifact_dir(table_path) / "bar.png"
    fig.savefig(output_path, dpi=_FIG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return [output_path]


def _plot_summary_bar_stat_table(table_path: Path, stats_rows, metric: str, color_map):
    plt, mticker, np = _load_plotting_libs()
    outputs = []
    baselines = _available_relative_baselines([driver for driver, _ in stats_rows])
    plot_cols = _SUMMARY_STAT_COLUMNS

    if not baselines:
        return _plot_raw_summary_bar_stat_table(table_path, stats_rows, metric, color_map)

    for baseline_key, baseline_driver in baselines:
        baseline_dict = None
        for driver, s_dict in stats_rows:
            if driver == baseline_driver:
                baseline_dict = s_dict
                break
        if baseline_dict is None:
            continue

        fallback_denoms = {}
        for col in plot_cols:
            baseline_val = baseline_dict[col]
            deltas = []
            for _, s_dict in stats_rows:
                val = s_dict[col]
                delta = baseline_val - val  # lower is better
                deltas.append(abs(delta))
            fallback_denoms[col] = max(deltas) if deltas else 0.0

        relative_rows = []
        for driver, s_dict in stats_rows:
            rel_dict = {}
            for col in plot_cols:
                baseline_val = baseline_dict[col]
                val = s_dict[col]
                delta = baseline_val - val
                denom = abs(baseline_val) if not math.isclose(baseline_val, 0.0, abs_tol=1e-12) else fallback_denoms[col]
                rel_dict[col] = delta / denom if denom > 0 else 0.0

            relative_rows.append((driver, rel_dict))
        plot_rows = _rows_without_driver(relative_rows, baseline_driver)
        if not plot_rows:
            continue

        fig, ax = plt.subplots(figsize=(_FIG_W, _STAT_FIG_H), dpi=_FIG_DPI)
        x_positions = _group_x_positions(np, len(plot_cols))
        n_drivers = max(1, len(plot_rows))
        group_width = _GROUP_WIDTH
        width = group_width / n_drivers

        all_vals = []
        for idx, (driver, rel_dict) in enumerate(plot_rows):
            color = color_map.get(driver, "#333333")
            offsets = x_positions - group_width / 2 + width * (idx + 0.5)
            heights = []
            for col in plot_cols:
                heights.append(rel_dict[col])
                all_vals.append(rel_dict[col])

            ax.bar(
                offsets,
                heights,
                width=width * _BAR_WIDTH_SCALE,
                label=driver,
                color=color,
                edgecolor="none",
                linewidth=0.0,
            )

        _apply_plot_axes(ax, _relative_ylabel(baseline_driver, metric))
        ax.axhline(0.0, color="#9c9c9c", linewidth=0.65, zorder=1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(-0.5, 1.0)
        _apply_odd_percent_guides(ax, mticker)
        _set_group_x_limits(ax, x_positions)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(plot_cols, rotation=32, ha="right")
        ax.set_xlabel("Statistic", fontsize=_AXIS_FONT, labelpad=1)

        handles, labels = ax.get_legend_handles_labels()
        top = _add_top_legend(fig, handles, labels, omit_labels=(baseline_driver,))
        fig.subplots_adjust(left=0.16, right=0.98, top=top, bottom=0.24)

        output_path = _relative_artifact_dir(table_path, baseline_key) / "bar_summary.png"
        fig.savefig(output_path, dpi=_FIG_DPI, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        outputs.append(output_path)

    outputs.extend(_plot_raw_summary_bar_stat_table(table_path, stats_rows, metric, color_map))
    return outputs


def _plot_raw_summary_bar_stat_table(table_path: Path, stats_rows, metric: str, color_map):
    plt, mticker, np = _load_plotting_libs()
    plot_cols = _SUMMARY_STAT_COLUMNS
    
    fig, ax = plt.subplots(figsize=(_FIG_W, _STAT_FIG_H), dpi=_FIG_DPI)
    x_positions = _group_x_positions(np, len(plot_cols))
    n_drivers = max(1, len(stats_rows))
    group_width = _GROUP_WIDTH
    width = group_width / n_drivers
    
    all_vals = []
    for _, s_dict in stats_rows:
        for c in plot_cols:
            all_vals.append(s_dict[c])
    
    positives = [v for v in all_vals if v > 0]
    floor = min(positives) * 0.5 if positives else 1e-6
    floor = max(floor, 1e-12)
    bottom = floor * 0.75
    
    for idx, (driver, s_dict) in enumerate(stats_rows):
        color = color_map.get(driver, "#333333")
        offsets = x_positions - group_width / 2 + width * (idx + 0.5)
        heights = []
        for c in plot_cols:
            val = s_dict[c]
            heights.append(max(val - bottom, bottom * 0.03))
        
        ax.bar(
            offsets,
            heights,
            width=width * _BAR_WIDTH_SCALE,
            bottom=bottom,
            label=driver,
            color=color,
            edgecolor="none",
            linewidth=0.0,
        )

    _apply_plot_axes(ax, _stat_ylabel(metric), log_y=True, mticker=mticker)
    
    upper = max(all_vals) * 1.35 if all_vals else floor * 2.0
    if upper <= floor:
        upper = floor * 2.0
    ax.set_ylim(bottom, upper)
    
    _set_group_x_limits(ax, x_positions)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_cols, rotation=32, ha="right")
    ax.set_xlabel("Statistic", fontsize=_AXIS_FONT, labelpad=1)

    handles, labels = ax.get_legend_handles_labels()
    top = _add_top_legend(fig, handles, labels)
    fig.subplots_adjust(left=0.16, right=0.98, top=top, bottom=0.24)

    output_path = _table_artifact_dir(table_path) / "bar_summary.png"
    fig.savefig(output_path, dpi=_FIG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return [output_path]


def _normalize_rows_to_best(rows):
    normalized = [(driver, []) for driver, _ in rows]
    for column_idx in range(len(_STAT_COLUMNS)):
        values = [values[column_idx] for _, values in rows]
        best = min(values)
        worst = max(values)
        if math.isclose(best, worst, rel_tol=0.0, abs_tol=1e-12):
            scores = [1.0] * len(values)
        else:
            span = worst - best
            scores = [(worst - value) / span for value in values]
        for idx, score in enumerate(scores):
            normalized[idx][1].append(max(0.0, min(1.0, score)))
    return normalized


def _plot_radar_stat_table(table_path: Path, rows, color_map):
    baselines = _available_relative_baselines([driver for driver, _ in rows])
    if not baselines:
        return [_plot_one_radar_stat_table(table_path, _normalize_rows_to_best(rows), color_map)]

    outputs = []
    for baseline_key, baseline_driver in baselines:
        radar_rows = _relative_rows_to_baseline(rows, baseline_driver, lower_is_better=True)
        outputs.append(
            _plot_one_radar_stat_table(
                table_path,
                radar_rows,
                color_map,
                baseline_key=baseline_key,
                baseline_driver=baseline_driver,
            )
        )
    return outputs


def _plot_one_radar_stat_table(table_path: Path, radar_rows, color_map, *, baseline_key=None, baseline_driver=None):
    plt, mticker, np = _load_plotting_libs()
    use_relative = baseline_key is not None
    angles = np.linspace(0, 2 * np.pi, len(_STAT_COLUMNS), endpoint=False).tolist()
    closed_angles = angles + angles[:1]
    all_values = [value for _, values in radar_rows for value in values]
    y_min, y_max = _relative_limits(all_values) if use_relative else (0.0, 1.0)

    fig, ax = plt.subplots(
        figsize=(_FIG_W, _RADAR_FIG_H),
        dpi=_FIG_DPI,
        subplot_kw={"projection": "polar"},
    )
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for driver, scores in radar_rows:
        color = color_map.get(driver, "#333333")
        closed_scores = scores + scores[:1]
        ax.plot(
            closed_angles,
            closed_scores,
            label=driver,
            color=color,
            linestyle="-",
            linewidth=1.35,
            marker="o",
            markersize=2.1,
            markerfacecolor=color,
            markeredgewidth=0.0,
            alpha=0.96,
        )
        ax.fill(closed_angles, closed_scores, color=color, alpha=0.065)

    ax.set_ylim(y_min, y_max)
    ax.set_xticks(angles)
    ax.set_xticklabels(_STAT_COLUMNS, fontsize=_TICK_FONT)
    if use_relative:
        locator = mticker.MaxNLocator(nbins=5)
        ticks = [tick for tick in locator.tick_values(y_min, y_max) if y_min <= tick <= y_max]
        if y_min < 0.0 < y_max and not any(math.isclose(tick, 0.0, abs_tol=1e-9) for tick in ticks):
            ticks.append(0.0)
            ticks.sort()
    else:
        ticks = [0.25, 0.50, 0.75, 1.00]
    ax.set_yticks(ticks)
    if use_relative:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    else:
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1"])
    ax.set_rlabel_position(90)
    ax.tick_params(axis="x", pad=1, labelsize=_TICK_FONT)
    ax.tick_params(axis="y", labelsize=_RADAR_RADIAL_FONT, pad=0)
    ax.grid(True, color="#d9d9d9", linewidth=0.55)
    _apply_odd_percent_guides(ax, mticker, axis="both")
    ax.spines["polar"].set_color("#bdbdbd")
    ax.spines["polar"].set_linewidth(0.65)
    if use_relative:
        ax.plot(closed_angles, [0.0] * len(closed_angles), color="#9c9c9c", linewidth=0.65)

    handles, labels = ax.get_legend_handles_labels()
    top = _add_top_legend(fig, handles, labels, omit_labels=(baseline_driver,) if use_relative else ())
    fig.subplots_adjust(left=0.05, right=0.95, top=top, bottom=0.04)

    output_dir = _relative_artifact_dir(table_path, baseline_key) if use_relative else _table_artifact_dir(table_path)
    output_path = output_dir / "radar.png"
    fig.savefig(output_path, dpi=_FIG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def _maintenance_window_label(row, index):
    prefix = "S" if str(row.get("Type", "")).startswith("Scheduled") else "U"
    return f"{prefix}{index}"


def _load_maintenance_announcement_table(path: Path, driver_order):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return [], []
        table_drivers = [
            field for field in reader.fieldnames
            if field not in _MAINT_META_COLUMNS and field not in _IGNORED_TABLE_DRIVERS
        ]
        ordered_drivers = [driver for driver in driver_order if driver in table_drivers]
        ordered_drivers.extend(driver for driver in table_drivers if driver not in ordered_drivers)

        windows = []
        for idx, row in enumerate(reader, start=1):
            values = []
            has_value = False
            for driver in ordered_drivers:
                value = _parse_numeric(row.get(driver))
                if value is None:
                    values.append(None)
                else:
                    has_value = True
                    values.append(value)
            if has_value:
                windows.append((_maintenance_window_label(row, idx), values))

    return ordered_drivers, windows


def _plot_maintenance_announcement_bars(plots_dir: Path, driver_order, color_map):
    table_path = _table_path_for(plots_dir, "table_util_announcement_to_start.csv")
    if not table_path.exists():
        return []

    drivers, windows = _load_maintenance_announcement_table(table_path, driver_order)
    if not drivers or not windows:
        return []

    outputs = []
    baselines = _available_relative_baselines(drivers)
    if baselines:
        for baseline_key, baseline_driver in baselines:
            baseline_idx = drivers.index(baseline_driver)
            plot_windows = []
            for label, values in windows:
                baseline = values[baseline_idx]
                if baseline is None:
                    plot_windows.append((label, [None] * len(values)))
                    continue
                fallback_denom = max(abs(value - baseline) for value in values if value is not None)
                relative_values = [
                    None if value is None else _relative_scalar_to_baseline(value, baseline, fallback_denom)
                    for value in values
                ]
                plot_windows.append((label, relative_values))

            csv_path = _relative_artifact_dir(table_path, baseline_key) / table_path.name
            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Window"] + drivers)
                for label, values in plot_windows:
                    writer.writerow([label] + [f"{v:.4f}" if v is not None else "N/A" for v in values])
            outputs.append(csv_path)

            outputs.append(
                _save_maintenance_bar(
                    table_path,
                    drivers,
                    windows,
                    plot_windows,
                    color_map,
                    baseline_key=baseline_key,
                    baseline_driver=baseline_driver,
                )
            )
        return outputs

    outputs.append(_save_maintenance_bar(table_path, drivers, windows, windows, color_map))
    return outputs


def _save_maintenance_bar(
    table_path: Path,
    drivers,
    windows,
    plot_windows,
    color_map,
    *,
    baseline_key=None,
    baseline_driver=None,
):
    plt, mticker, np = _load_plotting_libs()
    fig_h = _MAINT_FIG_H + max(0, len(windows) - 18) * 0.035
    fig, ax = plt.subplots(figsize=(_FIG_W, fig_h), dpi=_FIG_DPI)
    x_positions = _group_x_positions(np, len(windows))
    use_relative = baseline_key is not None
    plot_drivers = [driver for driver in drivers if not (use_relative and driver == baseline_driver)]
    driver_indices = [drivers.index(driver) for driver in plot_drivers]
    n_drivers = max(1, len(plot_drivers))
    group_width = _GROUP_WIDTH
    width = group_width / n_drivers
    all_plot_values = [
        values[driver_idx]
        for _, values in plot_windows
        for driver_idx in driver_indices
        if values[driver_idx] is not None
    ]

    for idx, (driver, driver_idx) in enumerate(zip(plot_drivers, driver_indices)):
        offsets = x_positions - group_width / 2 + width * (idx + 0.5)
        xs = []
        ys = []
        for x_pos, (_, values) in zip(offsets, plot_windows):
            value = values[driver_idx]
            if value is not None:
                xs.append(x_pos)
                ys.append(value)
        ax.bar(
            xs,
            ys,
            width=width * _BAR_WIDTH_SCALE,
            label=driver,
            color=color_map.get(driver, "#333333"),
            edgecolor="none",
            linewidth=0.0,
        )

    if use_relative:
        _apply_plot_axes(ax, _relative_ylabel(baseline_driver, "util"))
        ax.axhline(0.0, color="#9c9c9c", linewidth=0.65, zorder=1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        _set_relative_y_limits(ax, all_plot_values)
        _apply_odd_percent_guides(ax, mticker)
    else:
        _apply_plot_axes(ax, "Utilization (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.set_ylim(0, min(100.0, max(10.0, max(all_plot_values, default=0.0) * 1.10)))
    _set_group_x_limits(ax, x_positions)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for label, _ in windows], rotation=90, ha="center")
    ax.set_xlabel("Maintenance Window", fontsize=_AXIS_FONT, labelpad=1)

    handles, labels = ax.get_legend_handles_labels()
    top = _add_top_legend(fig, handles, labels, omit_labels=(baseline_driver,) if use_relative else ())
    fig.subplots_adjust(left=0.16, right=0.98, top=top, bottom=0.23)

    output_dir = _relative_artifact_dir(table_path, baseline_key) if use_relative else _table_artifact_dir(table_path)
    output_path = output_dir / "bar.png"
    fig.savefig(output_path, dpi=_FIG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def _load_util_overall_table(path: Path, driver_order):
    rows_by_driver = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Driver" not in reader.fieldnames or "Utilization_Percent" not in reader.fieldnames:
            return []
        for row in reader:
            driver = row.get("Driver", "").strip()
            if not driver or driver in _IGNORED_TABLE_DRIVERS:
                continue
            value = _parse_numeric(row.get("Utilization_Percent"))
            if value is not None:
                rows_by_driver[driver] = value

    rows = [(driver, rows_by_driver[driver]) for driver in driver_order if driver in rows_by_driver]
    rows.extend(
        (driver, rows_by_driver[driver])
        for driver in rows_by_driver
        if driver not in driver_order
    )
    return rows


def _plot_overall_util_bars(plots_dir: Path, driver_order, color_map):
    table_path = _table_path_for(plots_dir, "table_util_overall.csv")
    if not table_path.exists():
        return []
    rows = _load_util_overall_table(table_path, driver_order)
    if not rows:
        return []

    outputs = []
    baselines = _available_relative_baselines([driver for driver, _ in rows])
    if baselines:
        row_map = dict(rows)
        for baseline_key, baseline_driver in baselines:
            baseline = row_map[baseline_driver]
            fallback_denom = max(abs(value - baseline) for _, value in rows)
            relative_rows = [
                (driver, _relative_scalar_to_baseline(value, baseline, fallback_denom))
                for driver, value in rows
            ]
            plot_rows = _rows_without_driver(relative_rows, baseline_driver)
            if not plot_rows:
                continue

            csv_path = _relative_artifact_dir(table_path, baseline_key) / table_path.name
            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Driver", "Utilization_Percent"])
                for driver, value in relative_rows:
                    writer.writerow([driver, f"{value:.4f}"])
            outputs.append(csv_path)

            outputs.append(
                _save_overall_util_bar(
                    table_path,
                    plot_rows,
                    color_map,
                    baseline_key=baseline_key,
                    baseline_driver=baseline_driver,
                )
            )
        return outputs

    outputs.append(_save_overall_util_bar(table_path, rows, color_map))
    return outputs


def _save_overall_util_bar(table_path: Path, plot_rows, color_map, *, baseline_key=None, baseline_driver=None):
    plt, mticker, np = _load_plotting_libs()
    fig, ax = plt.subplots(figsize=(_FIG_W, _UTIL_FIG_H), dpi=_FIG_DPI)
    x_positions = _group_x_positions(np, len(plot_rows))
    use_relative = baseline_key is not None

    for idx, (driver, value) in enumerate(plot_rows):
        ax.bar(
            x_positions[idx],
            value,
            width=_SINGLE_BAR_WIDTH,
            label=driver,
            color=color_map.get(driver, "#333333"),
            edgecolor="none",
            linewidth=0.0,
        )

    if use_relative:
        _apply_plot_axes(ax, _relative_ylabel(baseline_driver, "util"))
        ax.axhline(0.0, color="#9c9c9c", linewidth=0.65, zorder=1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        _set_relative_y_limits(ax, [value for _, value in plot_rows])
        _apply_odd_percent_guides(ax, mticker)
    else:
        _apply_plot_axes(ax, "Utilization (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.set_ylim(0, min(100.0, max(10.0, max(value for _, value in plot_rows) * 1.10)))
    _set_group_x_limits(ax, x_positions)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([driver for driver, _ in plot_rows], rotation=36, ha="right")
    ax.set_xlabel("Scheduling Strategy", fontsize=_AXIS_FONT, labelpad=1)

    handles, labels = ax.get_legend_handles_labels()
    top = _add_top_legend(fig, handles, labels, omit_labels=(baseline_driver,) if use_relative else ())
    fig.subplots_adjust(left=0.16, right=0.98, top=top, bottom=0.31)

    output_dir = _relative_artifact_dir(table_path, baseline_key) if use_relative else _table_artifact_dir(table_path)
    output_path = output_dir / "bar.png"
    fig.savefig(output_path, dpi=_FIG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def _create_table_plots(plots_dir: Path, config):
    if not plots_dir.exists():
        _LOG.warning("Table plots skipped: plots dir not found: %s", plots_dir)
        return []

    base_order = _driver_order_from_config(config)
    driver_order = _collect_table_drivers(plots_dir, base_order)
    color_map = _build_color_map(driver_order)

    _clean_legacy_table_artifacts(plots_dir)

    created = []
    for table_path in _discover_stat_tables(plots_dir):
        rows, metric, stats_rows = _load_stat_table(table_path, driver_order)
        if not rows:
            continue
        _clean_table_artifacts(table_path)
        artifact_table_path = _move_table_to_artifact_dir(table_path)
        created.append(artifact_table_path)
        created.append(_plot_parallel_stat_table(table_path, rows, metric, color_map))
        created.extend(_plot_bar_stat_table(table_path, rows, metric, color_map))
        created.extend(_plot_summary_bar_stat_table(table_path, stats_rows, metric, color_map))
        created.extend(_plot_radar_stat_table(table_path, rows, color_map))

    maintenance_table_path = _table_path_for(plots_dir, "table_util_announcement_to_start.csv")
    if maintenance_table_path.exists():
        _clean_table_artifacts(maintenance_table_path)
        created.extend(_plot_maintenance_announcement_bars(plots_dir, driver_order, color_map))
        created.append(_move_table_to_artifact_dir(maintenance_table_path))

    util_table_path = _table_path_for(plots_dir, "table_util_overall.csv")
    if util_table_path.exists():
        _clean_table_artifacts(util_table_path)
        created.extend(_plot_overall_util_bars(plots_dir, driver_order, color_map))
        created.append(_move_table_to_artifact_dir(util_table_path))

    created.extend(_move_root_tables_to_artifact_dirs(plots_dir))
    return created


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    if len(args.configs) == 1:
        config_path = args.configs[0]
        config = load_experiment_config(config_path)
    else:
        config_paths = [Path(p).resolve() for p in args.configs]
        exp_names = [p.stem for p in config_paths]
        merge_dir = (
            Path(args.merge_dir).resolve()
            if args.merge_dir
            else REPO_DIR / "results" / ("plotmerge_" + "_".join(exp_names))
        )
        config, merged_config_path = _build_merged(config_paths, merge_dir)
        config_path = str(merged_config_path)
        _LOG.info("Merged %d experiments -> %s", len(args.configs), config_path)

    family = infer_trace_family(config)
    module_name = _PLOTTER_MODULES[family]

    if args.print_plotter:
        print(module_name)
        return 0

    module = importlib.import_module(module_name)
    child_argv = [config_path]
    if args.plots_dir:
        child_argv.extend(["--plots-dir", args.plots_dir])
    if args.output_dir:
        child_argv.extend(["--output-dir", args.output_dir])
    if args.global_only:
        child_argv.append("--global-only")
    result = module.main(child_argv)
    exit_code = 0 if result is None else result
    if exit_code != 0:
        return exit_code

    plots_dir = _plots_dir_for_run(config, args)
    results_dir = _resolve_repo_path(args.output_dir or config["output_dir"])
    _rewrite_group_stat_tables(config, plots_dir, results_dir, family, module)
    created = []
    created.extend(_create_relative_driver_heatmaps(config, plots_dir, results_dir, family, module))
    created.extend(_create_table_plots(plots_dir, config))
    for path in created:
        _LOG.info("Saved: %s", path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
