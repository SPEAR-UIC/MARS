#!/bin/bash
# slurm/light.sbatch: every non-MCTS baseline driver, concurrently on one node.
# The baselines are single-threaded, so they share one node instead of each
# taking a whole node (the partition may allocate whole nodes per job).
#   Usage: sbatch slurm/light.sbatch <run_dir>
#SBATCH --job-name=mars-light
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=0
set -uo pipefail

RUN_DIR="$1"
cd "${MARS_REPO_DIR:-$SLURM_SUBMIT_DIR}"
MAX_PAR="${MARS_LIGHT_PARALLEL:-${SLURM_CPUS_PER_TASK:-48}}"

# Forward scancel's SIGTERM to the driver wrappers so they record "cancelled".
trap 'kill -TERM $(jobs -p) 2>/dev/null' TERM INT

echo "[$(date '+%F %T')] ${SLURM_JOB_ID} on $(hostname): up to ${MAX_PAR} drivers at once"
while read -r key; do
    [[ -n "$key" ]] || continue
    while (( $(jobs -rp | wc -l) >= MAX_PAR )); do sleep 5; done   # portable throttle (no wait -n)
    python3 scripts/slurm_dist.py run-driver "$RUN_DIR" "$key" < /dev/null &
done < "$RUN_DIR/light.list"

while (( $(jobs -rp | wc -l) > 0 )); do wait || true; done

total=$(grep -c . "$RUN_DIR/light.list")
ok=0
while read -r key; do
    [[ -n "$key" ]] || continue
    grep -q '"state": "done"' "$RUN_DIR/status/${key}.json" && ok=$((ok + 1))
done < "$RUN_DIR/light.list"
echo "[$(date '+%F %T')] light drivers done: ${ok}/${total}"
[[ "$ok" -eq "$total" ]]
