#!/usr/bin/env bash
# slurm/submit.sh: launch MARS experiments on Slurm as per-driver jobs.
#
#   slurm/submit.sh [--no-build] [--smoke] [--dry-run] [exp ...]
#
# Default experiments: schedtwin-aurora schedtwin-frontier schedtwin-polaris
# schedtwin-theta. Run from a login node; the repo must be on a filesystem
# shared with the compute nodes. Load your compiler / cmake / python modules
# first: sbatch exports the current environment to the jobs.
#
# What it does:
#   1. builds cqsimcpp in Release mode (build.sh if cmake is available, otherwise
#      a direct g++ -O3 -fopenmp compile) and checks that it links OpenMP
#   2. scripts/slurm_dist.py prepare: results/slurm_runs/<RUN_ID>/ (+ "latest")
#   3. sbatch heavy.sbatch     one exclusive node per MARS driver (job array)
#   4. sbatch light.sbatch     all baseline drivers together on one node
#   5. sbatch aggregate.sbatch afterany:heavy:light, merges + summarises
#
# Options:
#   --no-build   reuse the existing ./cqsimcpp
#   --smoke      tiny end-to-end test (~400 jobs, 200 ms MCTS budget, 4 cores)
#   --dry-run    prepare the run directory but only print the sbatch commands
#
# Environment:
#   MARS_PARTITION       Slurm partition (default: compute)
#   MARS_SBATCH_ARGS     extra arguments for every sbatch call (e.g. --account=X)
#   MARS_LIGHT_PARALLEL  max concurrent baseline drivers (default: CPUs on node)
#   MARS_BINARY          simulator binary (default: ./cqsimcpp)
#   MARS_DIRECT_BUILD=1  compile without CMake even if cmake is available
#   CXX                  compiler for the direct build (default: g++)
#
# Monitor:    python3 scripts/slurm_dist.py status latest --watch 60
# Aggregate:  python3 scripts/slurm_dist.py aggregate latest   (also automatic)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PARTITION="${MARS_PARTITION:-compute}"
BUILD=1
DRY=0
PREP_ARGS=()
EXPS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build) BUILD=0 ;;
        --smoke)    PREP_ARGS+=(--smoke) ;;
        --dry-run)  DRY=1 ;;
        -h|--help)  sed -n '2,32p' "$0"; exit 0 ;;
        -*)         echo "Unknown option: $1" >&2; exit 1 ;;
        *)          EXPS+=("$1") ;;
    esac
    shift
done
if [[ ${#EXPS[@]} -eq 0 ]]; then
    EXPS=(schedtwin-aurora schedtwin-frontier schedtwin-polaris schedtwin-theta)
fi

die() { echo "[submit] ERROR: $*" >&2; exit 1; }
say() { echo "[submit] $*" >&2; }

python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 6) else 1)' \
    || die "python3 >= 3.6 is required"
if [[ $DRY -eq 0 ]]; then
    command -v sbatch >/dev/null || die "sbatch not found (are you on the login node?)"
fi

# 1. Build (Release: the default Debug build gets far fewer MCTS iterations per budget)
build_direct() {
    # Same sources and Release flags as the cqsimcpp target in CMakeLists.txt,
    # without CMake or the GoogleTest download.
    local cxx="${CXX:-g++}"
    command -v "$cxx" >/dev/null || die "cmake and $cxx are both missing; load a compiler module"
    local srcs
    srcs=$(sed -n '/set(HPCSCHED_SOURCES/,/)/p' CMakeLists.txt | grep -oE 'src/[^ )]+\.cpp' | tr '\n' ' ')
    [[ -n "$srcs" ]] || die "could not read HPCSCHED_SOURCES from CMakeLists.txt"
    local flags=(-std=c++17 -O3 -DNDEBUG -fopenmp -Isrc -Ithird_party)
    say "compiling cqsimcpp directly with $cxx ($("$cxx" -dumpversion 2>/dev/null || echo '?'))"
    # shellcheck disable=SC2086
    if ! "$cxx" "${flags[@]}" $srcs -o cqsimcpp.tmp 2>/dev/null; then
        # GCC 8 needs libstdc++fs for <filesystem>; rerun with it and show any errors.
        # shellcheck disable=SC2086
        "$cxx" "${flags[@]}" $srcs -o cqsimcpp.tmp -lstdc++fs \
            || die "compile failed: needs a C++17 compiler with OpenMP (GCC 8 or newer)"
    fi
    mv cqsimcpp.tmp cqsimcpp
}

if [[ $BUILD -eq 1 ]]; then
    if [[ "${MARS_DIRECT_BUILD:-0}" != "1" ]] && command -v cmake >/dev/null; then
        say "building cqsimcpp with CMake (Release)"
        bash build.sh --release
    else
        [[ "${MARS_DIRECT_BUILD:-0}" == "1" ]] || say "cmake not found, building without it"
        build_direct
    fi
fi
BIN="${MARS_BINARY:-./cqsimcpp}"
[[ -x "$BIN" ]] || die "$BIN not found; build it or drop --no-build"
if command -v ldd >/dev/null; then
    ldd "$BIN" | grep -qE 'libgomp|libomp|libiomp' \
        || die "cqsimcpp is not linked against OpenMP: MARS would run single-threaded. Load a compiler with OpenMP and rebuild."
else
    say "WARNING: ldd not available, cannot verify OpenMP linkage"
fi

# 2. Prepare the run directory
RUN_DIR="$(python3 scripts/slurm_dist.py prepare ${PREP_ARGS[@]+"${PREP_ARGS[@]}"} "${EXPS[@]}")"
N_HEAVY=$(grep -c . "$RUN_DIR/heavy.list" || true)
N_LIGHT=$(grep -c . "$RUN_DIR/light.list" || true)

# shellcheck disable=SC2206
EXTRA=(${MARS_SBATCH_ARGS:-})
# Pass exactly --export=ALL. Any other value (even "ALL,VAR=x") makes sbatch set
# SLURM_GET_USER_ENV=1, so each compute node re-creates the login environment
# with "su - $USER"; where that fails, Slurm requeues and holds the job
# ("user env retrieval failed requeued held").
export MARS_REPO_DIR="$REPO_DIR"
unset SLURM_GET_USER_ENV SBATCH_GET_USER_ENV
COMMON=(--parsable --partition="$PARTITION" --chdir="$REPO_DIR"
        --export=ALL ${EXTRA[@]+"${EXTRA[@]}"})

submit() {  # prints the job id
    if [[ $DRY -eq 1 ]]; then
        echo "sbatch $*" >&2
        echo "DRYRUN$RANDOM"
    else
        sbatch "$@" | cut -d';' -f1
    fi
}

# 3-5. Submit
deps=()
HEAVY_ID="" LIGHT_ID=""
if [[ $N_HEAVY -gt 0 ]]; then
    HEAVY_ID=$(submit "${COMMON[@]}" --array="0-$((N_HEAVY - 1))" \
        --output="$RUN_DIR/logs/heavy_%A_%a.out" slurm/heavy.sbatch "$RUN_DIR")
    deps+=("$HEAVY_ID")
    say "heavy (MARS) array: job $HEAVY_ID  ($N_HEAVY drivers, one node each)"
fi
if [[ $N_LIGHT -gt 0 ]]; then
    LIGHT_ID=$(submit "${COMMON[@]}" \
        --output="$RUN_DIR/logs/light_%j.out" slurm/light.sbatch "$RUN_DIR")
    deps+=("$LIGHT_ID")
    say "light (baselines):  job $LIGHT_ID  ($N_LIGHT drivers on one node)"
fi
[[ ${#deps[@]} -gt 0 ]] || die "no drivers to run"
AGG_ID=$(submit "${COMMON[@]}" --dependency="afterany:$(IFS=:; echo "${deps[*]}")" \
    --output="$RUN_DIR/logs/aggregate_%j.out" slurm/aggregate.sbatch "$RUN_DIR")
say "aggregate:          job $AGG_ID  (runs after both finish)"

python3 - "$RUN_DIR/slurm_jobs.json" "$HEAVY_ID" "$LIGHT_ID" "$AGG_ID" <<'PY'
import json, sys
path, heavy, light, agg = sys.argv[1:]
jobs = {k: v for k, v in (("heavy", heavy), ("light", light), ("aggregate", agg)) if v}
with open(path, "w") as f:
    json.dump(jobs, f, indent=2)
PY

cat >&2 <<MSG

Run directory: $RUN_DIR
  Monitor:     python3 scripts/slurm_dist.py status $(basename "$RUN_DIR") --watch 60
  Driver log:  tail -f $RUN_DIR/drivers/<key>/run.log
  Cancel all:  scancel ${deps[*]} $AGG_ID
  Results:     $RUN_DIR/aggregate/summary.md  (written by the aggregate job)
MSG
