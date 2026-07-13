#!/usr/bin/env python3
"""Plot the CSV produced by `cqsimcpp mcts-convergence`."""

import argparse
import csv
import os
from collections import OrderedDict

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required: python3 -m pip install matplotlib") from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot selected MCTS root choice, confidence, and value over iterations."
    )
    parser.add_argument("csv_path", help="Path to mcts-convergence CSV output")
    parser.add_argument(
        "--output",
        help="PNG path. Defaults to <csv_path without extension>.png",
    )
    return parser.parse_args()


def read_rows(path):
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "iteration": int(row["iteration"]),
                    "choice": row["selected_choice"],
                    "visit_pct": float(row["selected_visit_pct"]),
                    "avg_reward": float(row["selected_avg_reward"]),
                    "ties": int(row.get("best_visit_ties", "0")),
                    "margin": float(row.get("visit_margin", "0")),
                    "changed": row["changed"] == "1",
                }
            )
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    return rows


def main():
    args = parse_args()
    rows = read_rows(args.csv_path)
    output = args.output or os.path.splitext(args.csv_path)[0] + ".png"

    choice_codes = OrderedDict()
    codes = []
    for row in rows:
        choice_codes.setdefault(row["choice"], len(choice_codes) + 1)
        codes.append(choice_codes[row["choice"]])

    x = [row["iteration"] for row in rows]
    stabilization = max((row["iteration"] for row in rows if row["changed"]), default=x[-1])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("MCTS Root Choice Convergence", fontsize=14, fontweight="bold")

    axes[0].step(x, codes, where="post", color="#1b4965", linewidth=2)
    axes[0].axvline(stabilization, color="#ca6702", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Choice code")
    axes[0].set_title(f"Selected choice over time; last change at iteration {stabilization}")
    if len(choice_codes) <= 15:
        axes[0].set_yticks(list(choice_codes.values()))
        axes[0].set_yticklabels(list(choice_codes.keys()), fontsize=8)

    axes[1].plot(x, [row["visit_pct"] for row in rows], color="#005f73", label="visit pct")
    reward_axis = axes[1].twinx()
    reward_axis.plot(x, [row["avg_reward"] for row in rows], color="#9b2226", label="avg reward")
    axes[1].set_ylabel("Selected visit %")
    reward_axis.set_ylabel("Selected avg reward")
    axes[1].set_title("Selected branch share and value")

    axes[2].plot(x, [row["ties"] for row in rows], color="#0a9396", label="best visit ties")
    margin_axis = axes[2].twinx()
    margin_axis.plot(x, [row["margin"] for row in rows], color="#bb3e03", label="visit margin")
    axes[2].set_ylabel("Ties for best visits")
    margin_axis.set_ylabel("Visit margin")
    axes[2].set_xlabel("MCTS iterations")
    axes[2].set_title("Confidence diagnostics")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.axvline(stabilization, color="#ca6702", linestyle="--", linewidth=1)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
