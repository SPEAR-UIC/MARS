#!/usr/bin/env python3
"""Tail-wait heatmap analysis.

For each driver, identifies jobs above the Pth-percentile wait time and plots
a 2-D heatmap over (node-tier × requested-walltime) bins.

Cell colour  = log(count + 1) of tail jobs in that bin.
Cell text    = top line: count  /  bottom line: mean wait (hours).

The percentile threshold is computed PER DRIVER so the comparison is always
between each scheduler's own worst N% of jobs.

Usage:
    python3 scripts/tail_heatmap.py experiments/exp2a0.json
    python3 scripts/tail_heatmap.py experiments/exp2b0.json --pctile 80
    python3 scripts/tail_heatmap.py experiments/exp2a0.json --plots-dir results/custom
"""

import sys
import os
import json
import argparse
from datetime import timezone, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(__file__))
from plot_results import parse_swf, parse_events, infer_trace_family

# ── Theta bin definitions ─────────────────────────────────────────────────────
_THETA_NODE_EDGES  = [128, 256, 384, 640, 802, 4097]
_THETA_NODE_LABELS = ['T1\n128–255', 'T2\n256–383', 'T3\n384–639',
                      'T4\n640–801', 'T5\n802–4096']

_THETA_WT_EDGES  = [0, 1, 2, 3, 6, 12, 24.01]   # hours
_THETA_WT_LABELS = ['0–1 h', '1–2 h', '2–3 h', '3–6 h', '6–12 h', '12–24 h']

# ── Polaris bin definitions ───────────────────────────────────────────────────
_POLARIS_NODE_EDGES  = [10, 25, 100, 497]
_POLARIS_NODE_LABELS = ['S\n10–24', 'M\n25–99', 'L\n100–496']

_POLARIS_WT_EDGES  = [0, 2, 6, 12, 24, 48.01]   # hours
_POLARIS_WT_LABELS = ['0–2 h', '2–6 h', '6–12 h', '12–24 h', '24–48 h']

DPI       = 150
CELL_W    = 2.8   # inches per walltime bin column
CELL_H    = 1.8   # inches per node-tier row


def _bin(value, edges):
    """Return 0-based bin index for value (None if out of range)."""
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return None


def _build_grids(job_ids, wait_map, procs_map, walltimes_map,
                 node_edges, wt_edges):
    """Return (count_grid, sum_wait_grid) arrays of shape (n_node_bins, n_wt_bins)."""
    n_node = len(node_edges) - 1
    n_wt   = len(wt_edges)   - 1
    count    = np.zeros((n_node, n_wt), dtype=float)
    sum_wait = np.zeros((n_node, n_wt), dtype=float)

    for jid in job_ids:
        if jid not in procs_map or jid not in walltimes_map:
            continue
        ri = _bin(procs_map[jid], node_edges)
        ci = _bin(walltimes_map[jid] / 3600.0, wt_edges)
        if ri is None or ci is None:
            continue
        count[ri, ci]    += 1
        sum_wait[ri, ci] += wait_map[jid] / 3600.0   # hours

    return count, sum_wait


def _draw_heatmap(ax, count_grid, sum_wait_grid,
                  row_labels, col_labels, title, pctile_threshold_h,
                  shared_vmax):
    n_rows, n_cols = count_grid.shape
    img = np.log1p(count_grid)

    im = ax.imshow(img, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=shared_vmax, interpolation='nearest')

    for r in range(n_rows):
        for c in range(n_cols):
            cnt = int(count_grid[r, c])
            if cnt == 0:
                ax.text(c, r, '0', ha='center', va='center',
                        fontsize=8, color='#bbbbbb')
                continue
            avg_w    = sum_wait_grid[r, c] / cnt
            cell_val = img[r, c]
            txt_col  = 'white' if cell_val > shared_vmax * 0.55 else 'black'
            ax.text(c, r, f'{cnt}\n{avg_w:.1f}h',
                    ha='center', va='center', fontsize=10,
                    color=txt_col, fontweight='bold', linespacing=1.4)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(f'{title}\n(tail > {pctile_threshold_h:.1f} h)', fontsize=10, pad=5)
    ax.set_xlabel("Requested Walltime", fontsize=9)
    ax.set_ylabel("Node Count", fontsize=9)
    return im


def run(config_path, plots_dir_override=None, pctile=80, output_dir_override=None):
    with open(config_path) as f:
        config = json.load(f)

    family     = infer_trace_family(config)
    output_dir = output_dir_override or config["output_dir"]
    swf_path   = config["swf_path"]
    plots_dir  = plots_dir_override or (output_dir.rstrip("/") + "_plots")
    os.makedirs(plots_dir, exist_ok=True)

    if family == "theta":
        node_edges  = _THETA_NODE_EDGES
        node_labels = _THETA_NODE_LABELS
        wt_edges    = _THETA_WT_EDGES
        wt_labels   = _THETA_WT_LABELS
    else:
        node_edges  = _POLARIS_NODE_EDGES
        node_labels = _POLARIS_NODE_LABELS
        wt_edges    = _POLARIS_WT_EDGES
        wt_labels   = _POLARIS_WT_LABELS

    procs_map, walltimes_map, _, _ = parse_swf(swf_path)

    driver_tags = [d.get("tag", d["type"]) for d in config.get("drivers", [])
                   if d.get("plot", True)]

    # ── Load wait times per driver ────────────────────────────────────────────
    drivers = []   # list of (tag, all_wait_map)
    for tag in driver_tags:
        events_path = os.path.join(output_dir, tag, "events.csv")
        if not os.path.exists(events_path):
            print(f"  [skip] {tag}: events.csv not found")
            continue
        submit, start, end = parse_events(events_path)
        if not end:
            continue
        wait_map = {jid: start[jid] - submit[jid]
                    for jid in end if jid in submit and jid in start}
        drivers.append((tag, wait_map))

    if not drivers:
        print("No driver data found.")
        return 1

    n_drivers  = len(drivers)
    ncols      = min(n_drivers, 2)
    nrows      = (n_drivers + ncols - 1) // ncols
    subplot_w  = CELL_W * len(wt_labels)
    subplot_h  = CELL_H * len(node_labels)
    fig, axes  = plt.subplots(nrows, ncols,
                              figsize=(subplot_w * ncols + 1.2,
                                       subplot_h * nrows + 1.0),
                              squeeze=False)

    # ── Build grids, find shared vmax for consistent color scale ─────────────
    all_grids = []
    for tag, wait_map in drivers:
        waits = np.array(list(wait_map.values())) / 3600.0   # hours
        threshold_h = float(np.percentile(waits, pctile))
        tail_ids    = [jid for jid, w in wait_map.items()
                       if w / 3600.0 > threshold_h]
        tail_wait   = {jid: wait_map[jid] for jid in tail_ids}
        count_g, sum_w_g = _build_grids(tail_ids, tail_wait,
                                        procs_map, walltimes_map,
                                        node_edges, wt_edges)
        all_grids.append((tag, count_g, sum_w_g, threshold_h,
                          len(tail_ids), len(waits)))

    shared_vmax = max(float(np.log1p(g[1].max())) for g in all_grids)
    shared_vmax = max(shared_vmax, 1.0)

    # ── Draw ──────────────────────────────────────────────────────────────────
    last_im = None
    for idx, (tag, count_g, sum_w_g, thr_h, n_tail, n_total) in enumerate(all_grids):
        row, col = divmod(idx, ncols)
        ax  = axes[row][col]
        pct_tail = 100.0 * n_tail / n_total if n_total else 0
        title = f'{tag}  ({n_tail} / {n_total},  {pct_tail:.0f}%)'
        last_im = _draw_heatmap(ax, count_g, sum_w_g,
                                node_labels, wt_labels,
                                title, thr_h, shared_vmax)

    # Hide unused axes
    for idx in range(n_drivers, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Shared colorbar
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(),
                            orientation='vertical', fraction=0.015, pad=0.02)
        cbar.set_label('log(count + 1)', fontsize=8)
        tick_vals = np.linspace(0, shared_vmax, 5)
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([f'{int(np.expm1(v))}' for v in tick_vals], fontsize=7)

    pctile_label = f'P{pctile}'
    fig.suptitle(
        f'Tail-wait job characteristics — above {pctile_label} per driver\n'
        f'Cell: count (top)  /  mean wait in hours (bottom)',
        fontsize=11, y=1.01
    )
    fig.tight_layout(rect=[0, 0, 0.97, 1.0])

    out_path = os.path.join(plots_dir, f'tail_heatmap_p{int(pctile)}.png')
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Experiment JSON config file")
    parser.add_argument("--plots-dir", dest="plots_dir",
                        help="Output directory (default: <output_dir>_plots)")
    parser.add_argument("--pctile", type=float, default=80,
                        help="Percentile threshold for tail definition (default: 80)")
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Override the output_dir from the config JSON")
    args = parser.parse_args(argv)
    return run(args.config, args.plots_dir, args.pctile, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
