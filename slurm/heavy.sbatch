#!/bin/bash
# slurm/heavy.sbatch: one MARS (mcts) driver per array task, on a whole node.
# Submitted by slurm/submit.sh, which sets --array, --partition and --output.
#   Usage: sbatch --array=0-<N-1> slurm/heavy.sbatch <run_dir>
#SBATCH --job-name=mars-heavy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=0
set -euo pipefail

RUN_DIR="$1"
cd "${MARS_REPO_DIR:-$SLURM_SUBMIT_DIR}"

KEY="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUN_DIR/heavy.list")"
if [[ -z "$KEY" ]]; then
    echo "No heavy driver listed for array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

echo "[$(date '+%F %T')] ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname): ${KEY}"
exec python3 scripts/slurm_dist.py run-driver "$RUN_DIR" "$KEY"
