#!/usr/bin/env bash
# slurm/submit_ss.sh: run the MARS-SS driver (StartShare reward) on the four
# SchedTwin traces and add it to the existing results in results/schedtwin-*.
#
#   slurm/submit_ss.sh [--no-build] [--smoke] [--dry-run] [exp ...]
#
# Default experiments: schedtwin-{aurora,frontier,polaris,theta}-ss, one
# MARS-SS driver each (4 exclusive nodes, no baseline job). Each config
# writes to the same output_dir as the full experiment, and the aggregate job
# runs in merge mode, so summary.csv and summary_target.{csv,json} keep the
# existing drivers (FCFS, WFP, MARS-CW, MARS-CU, ...) and gain a MARS-SS row.
# An earlier MARS-SS result for the same trace is replaced.
#
# Reward: w_count * N_start/N_queue + w_age * A_start/A_queue, set in
# experiments/schedtwin-*-ss.json (w_count = 0.75, w_age = 0.25).
#
# All other options and environment variables are those of slurm/submit.sh.
# Use --smoke first: it runs ~400 jobs in a few minutes and writes only to
# the run directory, leaving results/ untouched.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ARGS=()
EXPS=()
for a in "$@"; do
    case "$a" in
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
        -*)        ARGS+=("$a") ;;
        *)         EXPS+=("$a") ;;
    esac
done
if [[ ${#EXPS[@]} -eq 0 ]]; then
    EXPS=(schedtwin-aurora-ss schedtwin-frontier-ss schedtwin-polaris-ss schedtwin-theta-ss)
fi

exec "$REPO_DIR/slurm/submit.sh" --merge ${ARGS[@]+"${ARGS[@]}"} "${EXPS[@]}"
