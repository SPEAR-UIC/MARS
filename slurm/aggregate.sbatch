#!/bin/bash
# slurm/aggregate.sbatch: merge per-driver results once all drivers have ended.
# Submitted by slurm/submit.sh with --dependency=afterany:<heavy>:<light>, so it
# also runs (and reports partial results) if some drivers failed.
#   Usage: sbatch slurm/aggregate.sbatch <run_dir>
#SBATCH --job-name=mars-aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
set -euo pipefail

RUN_DIR="$1"
cd "${MARS_REPO_DIR:-$SLURM_SUBMIT_DIR}"

python3 scripts/slurm_dist.py status "$RUN_DIR" --no-color || true
echo
python3 scripts/slurm_dist.py aggregate "$RUN_DIR"
