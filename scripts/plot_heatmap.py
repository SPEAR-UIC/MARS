"""
Heatmap: percentage of total jobs per (node-size bin × walltime bin).
Polaris (left) and Theta (right) side by side, common 0–100% color scale.
Each cell annotated with job count and percentage.
Output: figures/heatmap.{png,pdf}

Usage: python scripts/plot_heatmap.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import PowerNorm
from matplotlib.transforms import blended_transform_factory

os.makedirs("figures", exist_ok=True)

WALLTIME_BINS = [
    ("≤0.5h",  None, 0.5),
    ("≤1h",    0.5,  1.0),
    ("≤2h",    1.0,  2.0),
    ("≤3h",    2.0,  3.0),
    ("≤5h",    3.0,  5.0),
    ("≤6h",    5.0,  6.0),
    ("≤9h",    6.0,  9.0),
    ("≤24h",   9.0,  24.0),
]

# (label, lo_inclusive, hi_inclusive, upper_for_fraction)
POLARIS_NODE_BINS = [
    ("=10",    10,   10,   10),
    ("≤16",    11,   16,   16),
    ("≤24",    17,   24,   24),
    ("=25",    25,   25,   25),
    ("≤32",    26,   32,   32),
    ("≤128",   33,  128,  128),
    ("≤496",  129,  496,  496),
]

THETA_NODE_BINS = [
    ("=128",   128,  128,  128),
    ("≤255",   129,  255,  255),
    ("=256",   256,  256,  256),
    ("≤1024",  257, 1024, 1024),
    ("≤2048", 1025, 2048, 2048),
    ("≤4096", 2049, 4096, 4096),
]

SYSTEMS = [
    {
        "name":        "Polaris (2024)",
        "swf":         "extracted_data/polaris24-clean.swf",
        "min_nodes":   1,
        "total_nodes": 496,
        "node_bins":   POLARIS_NODE_BINS,
        "dividers":    [1, 4, 5],   # after ≤16, after ≤32, after ≤128
    },
    {
        "name":        "Theta (2021)",
        "swf":         "extracted_data/theta21-clean.swf",
        "min_nodes":   128,
        "total_nodes": 4096,
        "node_bins":   THETA_NODE_BINS,
        "dividers":    [0, 2, 3],   # after =128, between =256/≤1024, ≤1024/≤2048
    },
]

FS   = 8
FS_K = 9
FS_C = 7.5   # cell annotation font


def load_df(s):
    df = pd.read_csv(
        s["swf"],
        sep=r"\s+", comment=";", header=None,
        usecols=[1, 2, 3, 4, 7, 8],
        names=["submit", "wait", "runtime", "nodes_used", "nodes_req", "walltime_req"],
        engine="python",
    )
    df = df[
        (df["nodes_used"] >= s["min_nodes"]) &
        (df["walltime_req"] > 0) &
        (df["runtime"] > 0)
    ].copy()
    df["walltime_h"] = df["walltime_req"] / 3600.0
    print(f"  {s['name']}: {len(df):,} jobs")
    return df


def compute_heatmap(df, node_bins, total_jobs):
    n_n  = len(node_bins)
    n_wt = len(WALLTIME_BINS)
    counts = np.zeros((n_n, n_wt), dtype=int)
    pcts   = np.zeros((n_n, n_wt), dtype=float)
    n  = df["nodes_used"]
    wt = df["walltime_h"]
    for i, (_, nlo, nhi, _) in enumerate(node_bins):
        for j, (_, wtlo, wthi) in enumerate(WALLTIME_BINS):
            node_mask = (n >= nlo) & (n <= nhi)
            wt_mask   = (wt <= wthi) if wtlo is None else ((wt > wtlo) & (wt <= wthi))
            c = (node_mask & wt_mask).sum()
            counts[i, j] = c
            pcts[i, j]   = 100.0 * c / total_jobs
    return counts, pcts


def make_node_labels(node_bins, total_nodes):
    labels = []
    for label, _, _, upper in node_bins:
        frac = 100.0 * upper / total_nodes
        labels.append(f"{label}\n({frac:.1f}%)")
    return labels


print("Loading data...")
loaded = [(s, load_df(s)) for s in SYSTEMS]

# Compute all percentages first to find the global max
all_pcts = []
all_counts = []
for s, df in loaded:
    counts, pcts = compute_heatmap(df, s["node_bins"], len(df))
    all_counts.append(counts)
    all_pcts.append(pcts)

vmax = np.ceil(max(p.max() for p in all_pcts))   # round up to nearest integer

wt_labels = [b[0] for b in WALLTIME_BINS]
GAMMA = 0.4   # < 1 stretches low values, compresses high ones
norm = PowerNorm(gamma=GAMMA, vmin=0, vmax=vmax)
cmap = plt.cm.YlOrRd

print(f"Color scale: 0 – {vmax:.0f}%")

# ── Figure layout: stacked vertically, horizontal colorbar at top ─────────────
fig = plt.figure(figsize=(6.0, 7.5))
gs = gridspec.GridSpec(3, 1, height_ratios=[0.05, 1, 1],
                       hspace=0.38, left=0.12, right=0.82,
                       top=0.94, bottom=0.06)
ax_cb  = fig.add_subplot(gs[0, 0])
ax_pol = fig.add_subplot(gs[1, 0])
ax_the = fig.add_subplot(gs[2, 0])

im_ref = None

for ax, (s, df), counts, pcts in zip([ax_pol, ax_the], loaded, all_counts, all_pcts):
    node_bins = s["node_bins"]

    im = ax.imshow(pcts, cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")
    im_ref = im

    # ── Cell text: count on top, pct below ───────────────────────────────────
    for i in range(len(node_bins)):
        for j in range(len(WALLTIME_BINS)):
            c   = counts[i, j]
            pct = pcts[i, j]
            # With PowerNorm(gamma), midpoint of colormap is at vmax * 0.5^(1/gamma)
            fg  = "white" if pct > vmax * (0.5 ** (1 / GAMMA)) else "black"
            ax.text(j, i, f"{c:,}\n{pct:.1f}%",
                    ha="center", va="center", fontsize=FS_C,
                    color=fg, linespacing=1.3, fontweight="bold")

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xticks(range(len(WALLTIME_BINS)))
    ax.set_xticklabels(wt_labels, fontsize=FS_K, fontweight="bold")
    ax.set_yticks(range(len(node_bins)))
    ax.set_yticklabels(make_node_labels(node_bins, s["total_nodes"]),
                       fontsize=FS_K - 0.5, fontweight="bold")
    ax.set_title(s["name"], fontsize=FS + 4, pad=6, fontweight="bold")
    ax.set_xlabel("Requested Walltime", fontsize=FS + 2, fontweight="bold")
    ax.tick_params(length=0)
    # Draw cell borders
    ax.set_xticks(np.arange(len(WALLTIME_BINS)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(node_bins))     - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Vertical walltime divider lines ──────────────────────────────────
    n_rows = len(node_bins)
    for d_col in s.get("wt_dividers", []):
        x_div      = d_col + 0.5
        jobs_left  = int(counts[:, :d_col + 1].sum())
        pct_left   = 100.0 * jobs_left / len(df)
        ax.plot([x_div, x_div], [-0.5, n_rows - 0.5 + 0.35],
                color="black", linestyle="--", linewidth=1.4,
                clip_on=False, zorder=10)
        ax.text(x_div, n_rows - 0.5 + 0.45, f"{pct_left:.1f}%",
                ha="center", va="top",
                fontsize=FS_K - 1.5, fontweight="bold",
                clip_on=False, color="black")

    # ── Horizontal node divider lines + annotations ───────────────────────
    n_cols = len(WALLTIME_BINS)
    for d_row in s.get("dividers", []):
        y_div     = d_row + 0.5
        jobs_above = int(counts[:d_row + 1, :].sum())
        pct_above  = 100.0 * jobs_above / len(df)

        ax.plot([-0.5, n_cols - 0.5 + 0.35], [y_div, y_div],
                color="black", linestyle="--", linewidth=1.4,
                clip_on=False, zorder=10)
        ax.text(n_cols - 0.5 + 0.45, y_div,
                f"{pct_above:.1f}%",
                ha="left", va="center",
                fontsize=FS_K - 1.5, fontweight="bold",
                clip_on=False, color="black")

    # ── S/M/L/XL section labels ───────────────────────────────────────────
    size_labels = ["S", "M", "L", "XL"]
    dividers_sorted = sorted(s.get("dividers", []))
    n_rows = len(node_bins)
    boundaries = [-0.5] + [d + 0.5 for d in dividers_sorted] + [n_rows - 0.5]
    x_label = n_cols - 0.5 + 0.45
    for k, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if k >= len(size_labels):
            break
        center = (lo + hi) / 2
        ax.text(x_label, center, size_labels[k],
                ha="left", va="center",
                fontsize=14, fontweight="bold", clip_on=False, color="black")

ax_pol.set_ylabel("Nodes Used", fontsize=FS + 2, fontweight="bold")
ax_the.set_ylabel("Nodes Used", fontsize=FS + 2, fontweight="bold")

# ── Horizontal colorbar at top ────────────────────────────────────────────────
cb = plt.colorbar(im_ref, cax=ax_cb, orientation="horizontal")
cb.set_label("% of total jobs", fontsize=FS, labelpad=2)
cb.ax.tick_params(labelsize=FS_K)
ax_cb.xaxis.set_label_position("top")
ax_cb.xaxis.tick_top()

for ext in ("png", "pdf"):
    out = f"figures/heatmap.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")

plt.close(fig)
