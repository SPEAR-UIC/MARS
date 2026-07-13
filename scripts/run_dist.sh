#!/usr/bin/env bash
# scripts/run_dist.sh
#
# General-purpose distributed runner.  Splits any experiment's drivers across
# nodes, one driver per tmux session, then aggregates results for plotting.
#
# ── Node assignment strategy ───────────────────────────────────────────────────
#   Drivers are assigned by core capacity across the available nodes.
#   By default the node list comes from the repo-root "nodefile" (one hostname
#   per line, duplicates allowed and automatically collapsed; include "local"
#   there to represent this machine).
#   Each node is treated as having DIST_NODE_CORES cores (default: 256).
#   Driver core reservations are inferred from each experiment JSON's
#   ".drivers[].num_cores" when present.
#   "Heavy" drivers (MCTS by default) fall back to DIST_HEAVY_DRIVER_CORES
#   cores each (default: 64) only if num_cores is omitted.
#
# Override nodes via environment variables:
#   DIST_NODES="ccred2 ccred3 ccred4"    space-separated explicit node list
#   DIST_NODEFILE="/path/to/nodefile"    alternate nodefile path
#   DIST_INCLUDE_LOCAL=1                 prepend "local" to the node list
#   DIST_REMOTE_DIR="~/MyProject"        remote path (default: ~/CQSimPrivate)
#   DIST_HEAVY_TYPES="mcts adaptive_mcts custom_heavy"  space-separated types
#   DIST_NODE_CORES=256                  cores available per node
#   DIST_HEAVY_DRIVER_CORES=64           fallback cores per heavy driver
#   DIST_LIGHT_DRIVER_CORES=1            fallback cores per light driver
#   DIST_ALLOW_OVERSUBSCRIBE=1           allow assignments past per-node capacity
#
# ── Commands ───────────────────────────────────────────────────────────────────
#   ./scripts/run_dist.sh dist-launch    <exp> [exp2 ...]   build+sync+launch
#   ./scripts/run_dist.sh dist-status    <exp> [exp2 ...]   per-driver state
#   ./scripts/run_dist.sh dist-fetch     <exp> [exp2 ...]   pull from nodes
#   ./scripts/run_dist.sh dist-aggregate <exp> [exp2 ...]   merge for plotting
#   ./scripts/run_dist.sh dist-stop      <exp> [exp2 ...]   kill sessions
#   ./scripts/run_dist.sh dist-nuke      <exp> [exp2 ...]   stop + delete
#   ./scripts/run_dist.sh dist-plan      <exp> [exp2 ...]   preview assignment
#   ./scripts/run_dist.sh help
#
# <exp> is either a bare experiment name ("exp2a"), a unique
# prefix, both resolved to experiments/<exp>.json, or a path to a
# JSON file.
#
# Requirements: jq, tmux, SSH aliases for remote nodes in ~/.ssh/config

set -euo pipefail

# ── CLI pre-processing ─────────────────────────────────────────────────────────
# Parse --nodefile <path> (or --nodefile=<path>) before the configuration block
# so it overrides DIST_NODEFILE. All other arguments are passed through unchanged.
_new_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodefile)
            [[ $# -ge 2 ]] || { echo "run_dist.sh: --nodefile requires a path argument" >&2; exit 1; }
            DIST_NODEFILE="$2"; shift 2 ;;
        --nodefile=*)
            DIST_NODEFILE="${1#--nodefile=}"; shift ;;
        *)
            _new_args+=("$1"); shift ;;
    esac
done
set -- "${_new_args[@]+"${_new_args[@]}"}"

# ── Configuration ──────────────────────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Nodes. Priority:
#   1. DIST_NODES environment override
#   2. repo-root nodefile (may include "local")
#   3. fallback to local-only
NODEFILE_PATH="${DIST_NODEFILE:-${REPO_DIR}/nodefile}"
if [[ -n "${DIST_NODES:-}" ]]; then
    read -ra NODES <<< "$DIST_NODES"
    NODES_SOURCE="DIST_NODES"
elif [[ -f "$NODEFILE_PATH" ]]; then
    mapfile -t NODES < <(
        awk 'NF && $1 !~ /^#/ { print $1 }' "$NODEFILE_PATH" | awk '!seen[$0]++'
    )
    NODES_SOURCE="$NODEFILE_PATH"
else
    NODES=("local")
    NODES_SOURCE="fallback(local)"
fi

if [[ "${DIST_INCLUDE_LOCAL:-0}" == "1" ]]; then
    _has_local=0
    for _n in "${NODES[@]}"; do
        if [[ "$_n" == "local" ]]; then
            _has_local=1
            break
        fi
    done
    if [[ "$_has_local" -eq 0 ]]; then
        NODES=("local" "${NODES[@]}")
    fi
fi

[[ "${#NODES[@]}" -gt 0 ]] || error "No nodes configured"

# Remote directory path on each remote host.
REMOTE_DIR="${DIST_REMOTE_DIR:-~/CQSimPrivate}"

# Driver types considered "heavy" (get dedicated remote nodes).
if [[ -n "${DIST_HEAVY_TYPES:-}" ]]; then
    read -ra HEAVY_TYPES <<< "$DIST_HEAVY_TYPES"
else
    HEAVY_TYPES=("mcts" "adaptive_mcts")
fi

NODE_CORES="${DIST_NODE_CORES:-256}"
HEAVY_DRIVER_CORES="${DIST_HEAVY_DRIVER_CORES:-64}"
LIGHT_DRIVER_CORES="${DIST_LIGHT_DRIVER_CORES:-1}"
ALLOW_OVERSUBSCRIBE="${DIST_ALLOW_OVERSUBSCRIBE:-0}"

[[ "$NODE_CORES" =~ ^[0-9]+$ && "$NODE_CORES" -gt 0 ]] \
    || error "DIST_NODE_CORES must be a positive integer"
[[ "$HEAVY_DRIVER_CORES" =~ ^[0-9]+$ && "$HEAVY_DRIVER_CORES" -gt 0 ]] \
    || error "DIST_HEAVY_DRIVER_CORES must be a positive integer"
[[ "$LIGHT_DRIVER_CORES" =~ ^[0-9]+$ && "$LIGHT_DRIVER_CORES" -gt 0 ]] \
    || error "DIST_LIGHT_DRIVER_CORES must be a positive integer"

# Derive REMOTE_NODES as all NODES except "local".
REMOTE_NODES=()
for _n in "${NODES[@]}"; do
    [[ "$_n" != "local" ]] && REMOTE_NODES+=("$_n")
done

# ── Colour helpers ─────────────────────────────────────────────────────────────

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[dist]${NC} $*"; }
success() { echo -e "${GREEN}[dist]${NC} $*"; }
warn()    { echo -e "${YELLOW}[dist]${NC} $*"; }
error()   { echo -e "${RED}[dist]${NC} $*" >&2; exit 1; }
header()  { echo -e "\n${BOLD}$*${NC}"; }

# ── Helpers ────────────────────────────────────────────────────────────────────

run_on() {
    local node="$1"; shift
    if [ "$node" = "local" ]; then bash -c "$*"
    else ssh -o ConnectTimeout=10 -q "$node" "$@"
    fi
}

run_on_quiet() {
    local node="$1"; shift
    if [ "$node" = "local" ]; then bash -c "$*" 2>/dev/null
    else ssh -o ConnectTimeout=10 -q "$node" "$@" 2>/dev/null
    fi
}

node_dir() {
    local node="$1"
    [ "$node" = "local" ] && echo "$REPO_DIR" || echo "$REMOTE_DIR"
}

_is_generated_driver_json_basename() {
    local name="$1"
    [[ "$name" =~ _d[0-9]+$ ]]
}

# Resolve "<exp>" → path to JSON (accepts file path, bare name, or unique prefix).
resolve_json() {
    local arg="$1"
    if [[ -f "$arg" ]]; then
        echo "$arg"
    elif [[ -f "${REPO_DIR}/experiments/${arg}.json" ]]; then
        echo "${REPO_DIR}/experiments/${arg}.json"
    else
        local matches=()
        local primary_matches=()
        local match
        shopt -s nullglob
        for match in "${REPO_DIR}/experiments/${arg}"*.json; do
            matches+=("$match")
            local base
            base="$(basename "${match%.json}")"
            if ! _is_generated_driver_json_basename "$base"; then
                primary_matches+=("$match")
            fi
        done
        shopt -u nullglob

        if [[ "${#primary_matches[@]}" -eq 1 ]]; then
            echo "${primary_matches[0]}"
            return 0
        fi

        if [[ "${#primary_matches[@]}" -gt 1 ]]; then
            local names=()
            for match in "${primary_matches[@]}"; do
                names+=("$(basename "${match%.json}")")
            done
            error "Ambiguous experiment name '$arg' — matches: ${names[*]}"
        fi

        if [[ "${#matches[@]}" -gt 0 ]]; then
            error "Cannot resolve '$arg' to a primary experiment JSON (only generated per-driver configs matched)"
        fi

        error "Cannot find experiment JSON for: $arg"
    fi
}

# Derive experiment name from JSON path (basename without .json).
exp_name() { basename "${1%.json}"; }

# Manifest / outdir / session naming.
_manifest()   { echo "${REPO_DIR}/results/${1}/.dist_manifest.json"; }
_outdir()     { echo "results/${1}_d${2}"; }        # exp, idx
_session()    { echo "cqsim-${1}-d${2}"; }          # exp, idx
_driver_key() { echo "${1}::${2}"; }                # exp, idx

_resolve_exp_for_manifest_commands() {
    local arg="$1"
    if [[ -f "$(_manifest "$arg")" ]]; then
        echo "$arg"
        return 0
    fi

    local json
    if ! json="$(resolve_json "$arg")"; then
        return 1
    fi
    exp_name "$json"
}

_init_capacity_state() {
    declare -gA GLOBAL_NODE_USED_CORES GLOBAL_NODE_DRIVER_COUNTS
    declare -gA GLOBAL_EXP_NODE_USED_CORES PLAN_JSON_PATHS PLAN_NUM_DRIVERS
    declare -gA PLAN_DRIVER_NODES PLAN_DRIVER_CORES PLAN_DRIVER_CORE_START PLAN_DRIVER_CORE_END
    declare -ga PLANNED_EXPS
    GLOBAL_NODE_USED_CORES=()
    GLOBAL_NODE_DRIVER_COUNTS=()
    GLOBAL_EXP_NODE_USED_CORES=()
    PLAN_JSON_PATHS=()
    PLAN_NUM_DRIVERS=()
    PLAN_DRIVER_NODES=()
    PLAN_DRIVER_CORES=()
    PLAN_DRIVER_CORE_START=()
    PLAN_DRIVER_CORE_END=()
    PLANNED_EXPS=()
    for nd in "${NODES[@]}"; do
        GLOBAL_NODE_USED_CORES["$nd"]=0
        GLOBAL_NODE_DRIVER_COUNTS["$nd"]=0
    done
    declare -g GLOBAL_NODE_PICK_INDEX=0
}

# Check whether a driver type is heavy.
_is_heavy() {
    local dtype="$1"
    for ht in "${HEAVY_TYPES[@]}"; do
        [[ "$dtype" == "$ht" ]] && return 0
    done
    return 1
}

# Configured num_cores for a driver, if present and valid.
_configured_driver_cores() {
    local json="$1"
    local idx="$2"
    local configured
    configured=$(jq -r ".drivers[$idx].num_cores // empty" "$json")
    if [[ "$configured" =~ ^[0-9]+$ && "$configured" -gt 0 ]]; then
        echo "$configured"
        return 0
    fi
    return 1
}

# Reserved core count for a driver.
# Light (non-heavy) drivers return 0 — they run unpinned and cost no capacity.
_driver_cores() {
    local json="$1"
    local idx="$2"
    local dtype
    dtype=$(jq -r ".drivers[$idx].type" "$json")

    if ! _is_heavy "$dtype"; then
        echo "0"
        return 0
    fi

    if _configured_driver_cores "$json" "$idx"; then
        return 0
    fi

    echo "$HEAVY_DRIVER_CORES"
}

_select_node_for_cores() {
    local exp="$1"
    local req="$2"
    local num_nodes="${#NODES[@]}"
    local best=""
    local best_idx=-1
    local best_used=""
    local best_driver_count=""
    local best_exp_used=""
    local allow_oversub=0

    [[ "$ALLOW_OVERSUBSCRIBE" == "1" ]] && allow_oversub=1

    for (( offset=0; offset<num_nodes; offset++ )); do
        local idx=$(( (GLOBAL_NODE_PICK_INDEX + offset) % num_nodes ))
        local nd="${NODES[$idx]}"
        local used="${GLOBAL_NODE_USED_CORES[$nd]:-0}"
        local projected=$(( used + req ))
        local driver_count="${GLOBAL_NODE_DRIVER_COUNTS[$nd]:-0}"
        local exp_used="${GLOBAL_EXP_NODE_USED_CORES["$exp|$nd"]:-0}"

        if (( projected > NODE_CORES && allow_oversub == 0 )); then
            continue
        fi

        if [[ -z "$best" ]]; then
            best="$nd"
            best_idx=$idx
            best_used="$used"
            best_driver_count="$driver_count"
            best_exp_used="$exp_used"
            continue
        fi

        if (( used < best_used )); then
            best="$nd"
            best_idx=$idx
            best_used="$used"
            best_driver_count="$driver_count"
            best_exp_used="$exp_used"
            continue
        fi

        if (( used == best_used && driver_count < best_driver_count )); then
            best="$nd"
            best_idx=$idx
            best_used="$used"
            best_driver_count="$driver_count"
            best_exp_used="$exp_used"
            continue
        fi

        if (( used == best_used && driver_count == best_driver_count && exp_used < best_exp_used )); then
            best="$nd"
            best_idx=$idx
            best_used="$used"
            best_driver_count="$driver_count"
            best_exp_used="$exp_used"
            continue
        fi
    done

    if (( best_idx >= 0 )); then
        SELECTED_NODE="$best"
        SELECTED_NODE_INDEX="$best_idx"
        return 0
    fi

    return 1
}

_format_core_range() {
    local start="$1"
    local end="$2"
    if (( start == end )); then
        echo "$start"
    else
        echo "${start}-${end}"
    fi
}

# Print the current assignment held in the global plan arrays.
_print_assignment() {
    local json="$1"
    local exp="$2"
    local num_drivers="$3"

    header "PLAN — $exp ($num_drivers drivers)"
    printf "  %-12s %s\n" "Node source" "$NODES_SOURCE"
    printf "  %-12s %s\n" "Node cores" "$NODE_CORES"
    printf "  %-12s %s\n" "Heavy def." "$HEAVY_DRIVER_CORES"
    echo ""
    printf "  %-4s  %-10s  %-5s  %-11s  %-30s  %s\n" "D#" "Node" "Cores" "CoreRange" "Tag" "Type"
    printf "  %-4s  %-10s  %-5s  %-11s  %-30s  %s\n" "----" "----------" "-----" "-----------" "------------------------------" "-------"
    for (( i=0; i<num_drivers; i++ )); do
        local key tag dtype range
        key="$(_driver_key "$exp" "$i")"
        tag=$(jq -r   ".drivers[$i].tag"  "$json")
        dtype=$(jq -r ".drivers[$i].type" "$json")
        [[ "$dtype" == "rlscheduler" ]] && continue
        range="$(_format_core_range "${PLAN_DRIVER_CORE_START[$key]}" "${PLAN_DRIVER_CORE_END[$key]}")"
        printf "  %-4s  %-10s  %-5s  %-11s  %-30s  %s\n" \
            "$i" "${PLAN_DRIVER_NODES[$key]}" "${PLAN_DRIVER_CORES[$key]}" "$range" "$tag" "$dtype"
    done

    echo ""
    printf "  %-10s  %-17s  %-17s  %s\n" "Node" "ExpUsed/Total" "GlobalUsed/Total" "Drivers(E/G)"
    printf "  %-10s  %-17s  %-17s  %s\n" "----------" "-----------------" "-----------------" "------------"
    for nd in "${NODES[@]}"; do
        local exp_count=0
        local exp_used=0
        for (( i=0; i<num_drivers; i++ )); do
            local key
            key="$(_driver_key "$exp" "$i")"
            [[ -v "PLAN_DRIVER_NODES[$key]" ]] || continue
            if [[ "${PLAN_DRIVER_NODES[$key]}" == "$nd" ]]; then
                exp_count=$(( exp_count + 1 ))
                exp_used=$(( exp_used + PLAN_DRIVER_CORES[$key] ))
            fi
        done
        local global_used="${GLOBAL_NODE_USED_CORES[$nd]:-0}"
        local global_count="${GLOBAL_NODE_DRIVER_COUNTS[$nd]:-0}"
        printf "  %-10s  %4s/%-11s  %4s/%-11s  %s/%s\n" \
            "$nd" "$exp_used" "$NODE_CORES" "$global_used" "$NODE_CORES" "$exp_count" "$global_count"
        if (( global_used > NODE_CORES )); then
            warn "$nd is oversubscribed in the global plan (${global_used}/${NODE_CORES} cores)"
        fi
    done
    echo ""
}

_assign_driver() {
    local exp="$1"
    local json="$2"
    local idx="$3"
    local req start end key node
    req="$(_driver_cores "$json" "$idx")"

    if ! _select_node_for_cores "$exp" "$req"; then
        error "Insufficient node capacity for driver $idx in $(basename "$json") — need ${req} cores, available nodes=${NODES[*]}"
    fi

    node="$SELECTED_NODE"
    key="$(_driver_key "$exp" "$idx")"

    PLAN_DRIVER_NODES["$key"]="$node"
    PLAN_DRIVER_CORES["$key"]="$req"
    PLAN_DRIVER_CORE_START["$key"]="0"
    PLAN_DRIVER_CORE_END["$key"]="0"

    if (( req > 0 )); then
        local start
        start="${GLOBAL_NODE_USED_CORES[$node]:-0}"
        PLAN_DRIVER_CORE_START["$key"]="$start"
        PLAN_DRIVER_CORE_END["$key"]=$(( start + req - 1 ))
        GLOBAL_NODE_USED_CORES["$node"]=$(( start + req ))
        GLOBAL_EXP_NODE_USED_CORES["$exp|$node"]=$(( ${GLOBAL_EXP_NODE_USED_CORES["$exp|$node"]:-0} + req ))
    fi
    GLOBAL_NODE_DRIVER_COUNTS["$node"]=$(( ${GLOBAL_NODE_DRIVER_COUNTS[$node]:-0} + 1 ))
    GLOBAL_NODE_PICK_INDEX=$(( (SELECTED_NODE_INDEX + 1) % ${#NODES[@]} ))
}

_plan_assignments() {
    local args=("$@")
    local task_lines=()
    local exp_order=0
    declare -A seen_exps

    _init_capacity_state

    for arg in "${args[@]}"; do
        local json exp num_drivers heavy_round light_round
        if ! json="$(resolve_json "$arg")"; then
            return 1
        fi
        exp="$(exp_name "$json")"

        if [[ -n "${seen_exps[$exp]+x}" ]]; then
            error "Duplicate experiment specified: $exp"
        fi
        seen_exps["$exp"]=1

        PLANNED_EXPS+=("$exp")
        PLAN_JSON_PATHS["$exp"]="$json"
        num_drivers=$(jq '.drivers | length' "$json")
        PLAN_NUM_DRIVERS["$exp"]="$num_drivers"
        heavy_round=0
        light_round=0

        for (( i=0; i<num_drivers; i++ )); do
            local dtype phase round cores neg_cores
            dtype=$(jq -r ".drivers[$i].type" "$json")
            [[ "$dtype" == "rlscheduler" ]] && continue
            cores="$(_driver_cores "$json" "$i")"
            neg_cores=$(( -cores ))
            if _is_heavy "$dtype"; then
                phase=0
                round="$heavy_round"
                heavy_round=$(( heavy_round + 1 ))
            else
                phase=1
                round="$light_round"
                light_round=$(( light_round + 1 ))
            fi
            task_lines+=("$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s' "$phase" "$neg_cores" "$round" "$exp_order" "$exp" "$json" "$i")")
        done

        exp_order=$(( exp_order + 1 ))
    done

    if (( ${#task_lines[@]} == 0 )); then
        return 0
    fi

    while IFS=$'\t' read -r _phase _neg_cores _round _exp_order exp json idx; do
        _assign_driver "$exp" "$json" "$idx"
    done < <(printf '%s\n' "${task_lines[@]}" | sort -t$'\t' -k1,1n -k2,2n -k3,3n -k4,4n)
}

_print_global_assignment_summary() {
    local total_experiments="${#PLANNED_EXPS[@]}"
    local total_drivers=0
    local exp

    for exp in "${PLANNED_EXPS[@]}"; do
        total_drivers=$(( total_drivers + ${PLAN_NUM_DRIVERS[$exp]:-0} ))
    done

    header "GLOBAL PLAN — ${total_experiments} experiment(s), ${total_drivers} driver(s)"
    printf "  %-12s %s\n" "Node source" "$NODES_SOURCE"
    printf "  %-12s %s\n" "Node cores" "$NODE_CORES"
    printf "  %-12s %s\n" "Heavy def." "$HEAVY_DRIVER_CORES"
    echo ""
    printf "  %-10s  %-13s  %-7s  %s\n" "Node" "Used/Total" "Drivers" "Experiments"
    printf "  %-10s  %-13s  %-7s  %s\n" "----------" "-------------" "-------" "-----------"
    for nd in "${NODES[@]}"; do
        local used="${GLOBAL_NODE_USED_CORES[$nd]:-0}"
        local drivers="${GLOBAL_NODE_DRIVER_COUNTS[$nd]:-0}"
        local exp_count=0
        for exp in "${PLANNED_EXPS[@]}"; do
            if (( ${GLOBAL_EXP_NODE_USED_CORES["$exp|$nd"]:-0} > 0 )); then
                exp_count=$(( exp_count + 1 ))
            fi
        done
        printf "  %-10s  %4s/%-8s  %-7s  %s\n" "$nd" "$used" "$NODE_CORES" "$drivers" "$exp_count"
        if (( used > NODE_CORES )); then
            warn "$nd is oversubscribed in the global plan (${used}/${NODE_CORES} cores)"
        fi
    done
    echo ""
}

# ── dist-plan (dry-run preview) ────────────────────────────────────────────────

_plan_one() {
    local json exp num_drivers
    if ! json="$(resolve_json "$1")"; then
        return 1
    fi
    exp="$(exp_name "$json")"
    num_drivers="${PLAN_NUM_DRIVERS[$exp]:-0}"
    _print_assignment "$json" "$exp" "$num_drivers"
}

# ── dist-launch ────────────────────────────────────────────────────────────────

_launch_one() {
    local json
    if ! json="$(resolve_json "$1")"; then
        return 1
    fi
    local exp
    exp="$(exp_name "$json")"
    local num_drivers
    num_drivers="${PLAN_NUM_DRIVERS[$exp]:-0}"

    header "DIST-LAUNCH — $exp ($num_drivers drivers)"
    _print_assignment "$json" "$exp" "$num_drivers"

    local manifest
    manifest="$(_manifest "$exp")"
    mkdir -p "$(dirname "$manifest")"
    jq -n --arg exp "$exp" --arg json_path "$json" \
        '{exp:$exp, json_path:$json_path, drivers:[]}' > "$manifest"

    for (( i=0; i<num_drivers; i++ )); do
        local key node
        key="$(_driver_key "$exp" "$i")"
        local tag dtype
        tag=$(jq -r ".drivers[$i].tag" "$json")
        dtype=$(jq -r ".drivers[$i].type" "$json")
        [[ "$dtype" == "rlscheduler" ]] && continue
        node="${PLAN_DRIVER_NODES[$key]}"
        local dir
        dir="$(node_dir "$node")"
        local session
        session="$(_session "$exp" "$i")"
        local cfg_name="${exp}_d${i}.json"
        local tmp_json="/tmp/${cfg_name}"
        local outdir
        outdir="$(_outdir "$exp" "$i")"
        local cores="${PLAN_DRIVER_CORES[$key]}"
        local core_range
        core_range="$(_format_core_range "${PLAN_DRIVER_CORE_START[$key]}" "${PLAN_DRIVER_CORE_END[$key]}")"

        # Single-driver config with its own output_dir. Preserve configured
        # num_cores, and only inject the planned core count as a fallback.
        jq --argjson idx "$i" \
           --arg outdir "$outdir" \
           --argjson cores "$cores" \
           '
           .drivers = [.drivers[$idx]]
           | .output_dir = $outdir
           | .drivers[0].num_cores =
               (if ((.drivers[0].num_cores // 0) | tonumber? // 0) > 0
                then .drivers[0].num_cores
                else $cores
                end)
           ' \
           "$json" > "$tmp_json"

        if [ "$node" = "local" ]; then
            cp "$tmp_json" "${REPO_DIR}/experiments/${cfg_name}"
        else
            rsync -az "$tmp_json" "${node}:${REMOTE_DIR}/experiments/${cfg_name}"
        fi

        run_on_quiet "$node" "tmux kill-session -t ${session} 2>/dev/null" || true

        local inner_cmd tmux_cmd tmux_cmd_quoted is_heavy_driver
        _is_heavy "$dtype" && is_heavy_driver=1 || is_heavy_driver=0
        if (( is_heavy_driver )); then
        inner_cmd=$(
            cat <<EOF
set -o pipefail
cd ${dir}
mkdir -p ${outdir}
export OMP_NUM_THREADS=${cores} OMP_PROC_BIND=spread OMP_PLACES=cores
if command -v taskset >/dev/null 2>&1; then
    taskset -c ${core_range} ./cqsimcpp run config experiments/${cfg_name}
else
    ./cqsimcpp run config experiments/${cfg_name}
fi 2>&1 | tee -a ${outdir}/run.log
_ec=\$?
if [ "\$_ec" -eq 0 ]; then
    echo "[done] ${exp} d${i} (${tag}) finished" | tee -a ${outdir}/run.log
else
    echo "[crashed] ${exp} d${i} (${tag}) exited with code \$_ec" | tee -a ${outdir}/run.log
fi
exec bash
EOF
        )
        else
        inner_cmd=$(
            cat <<EOF
set -o pipefail
cd ${dir}
mkdir -p ${outdir}
./cqsimcpp run config experiments/${cfg_name} 2>&1 | tee -a ${outdir}/run.log
_ec=\$?
if [ "\$_ec" -eq 0 ]; then
    echo "[done] ${exp} d${i} (${tag}) finished" | tee -a ${outdir}/run.log
else
    echo "[crashed] ${exp} d${i} (${tag}) exited with code \$_ec" | tee -a ${outdir}/run.log
fi
exec bash
EOF
        )
        fi
        tmux_cmd="bash -lc $(printf '%q' "$inner_cmd")"
        tmux_cmd_quoted="$(printf '%q' "$tmux_cmd")"

        run_on_quiet "$node" "tmux new-session -d -s ${session} -x 220 -y 50 ${tmux_cmd_quoted}"
        if (( is_heavy_driver )); then
            success "$node: launched driver $i  ($tag, ${cores} cores, ${core_range})  →  session $session"
        else
            success "$node: launched driver $i  ($tag, unpinned)  →  session $session"
        fi

        local tmp_manifest="/tmp/${exp}_manifest_tmp.json"
        jq --arg  node    "$node"    \
           --arg  tag     "$tag"     \
           --arg  session "$session" \
           --arg  outdir  "$outdir"  \
           --arg  dtype   "$dtype"   \
           --arg  core_range "$core_range" \
           --argjson cores "$cores" \
           --argjson idx  "$i"       \
           '.drivers += [{"index":$idx,"tag":$tag,"type":$dtype,"node":$node,"session":$session,"outdir":$outdir,"cores":$cores,"core_range":$core_range}]' \
           "$manifest" > "$tmp_manifest" && mv "$tmp_manifest" "$manifest"
    done

    success "$exp: all $num_drivers drivers launched."
    echo "  Manifest: $manifest"
    echo ""
    echo "  Monitor:   $(basename "$0") dist-status $exp"
    echo "  Fetch:     $(basename "$0") dist-fetch $exp"
    echo "  Aggregate: $(basename "$0") dist-aggregate $exp"
    echo ""
    echo "  Attach to a session:"
    for (( i=0; i<num_drivers; i++ )); do
        local key node session tag dtype range
        key="$(_driver_key "$exp" "$i")"
        tag=$(jq -r   ".drivers[$i].tag"  "$json")
        dtype=$(jq -r ".drivers[$i].type" "$json")
        [[ "$dtype" == "rlscheduler" ]] && continue
        node="${PLAN_DRIVER_NODES[$key]}"
        session="$(_session "$exp" "$i")"
        range="$(_format_core_range "${PLAN_DRIVER_CORE_START[$key]}" "${PLAN_DRIVER_CORE_END[$key]}")"
        if [ "$node" = "local" ]; then
            printf "    %-32s  tmux attach -t %s    # cores %s\n" "$tag" "$session" "$range"
        else
            printf "    %-32s  ssh %s -t 'tmux attach -t %s'    # cores %s\n" "$tag" "$node" "$session" "$range"
        fi
    done
    echo ""
}

# ── dist-status ────────────────────────────────────────────────────────────────

_status_one() {
    local exp="$1"
    local manifest
    manifest="$(_manifest "$exp")"
    [[ -f "$manifest" ]] || error "No manifest for $exp — run dist-launch first"

    local num_drivers
    num_drivers=$(jq '.drivers | length' "$manifest")

    header "DIST-STATUS — $exp ($num_drivers drivers)"
    printf "  %-4s  %-10s  %-32s  %-5s  %-11s  %-22s  %-10s  %s\n" \
        "D#" "Node" "Tag" "Core" "CoreRange" "Session" "State" "Progress"
    printf "  %-4s  %-10s  %-32s  %-5s  %-11s  %-22s  %-10s  %s\n" \
        "----" "----------" "--------------------------------" "-----" "-----------" "----------------------" "----------" "--------"

    for (( i=0; i<num_drivers; i++ )); do
        local node tag session outdir cores core_range
        node=$(jq -r    ".drivers[$i].node"    "$manifest")
        tag=$(jq -r     ".drivers[$i].tag"     "$manifest")
        session=$(jq -r ".drivers[$i].session" "$manifest")
        outdir=$(jq -r  ".drivers[$i].outdir"  "$manifest")
        cores=$(jq -r   ".drivers[$i].cores // \"?\"" "$manifest")
        core_range=$(jq -r ".drivers[$i].core_range // \"-\"" "$manifest")
        local dir log
        dir="$(node_dir "$node")"
        log="${dir}/${outdir}/run.log"

        local info_line state
        info_line=$(run_on_quiet "$node" "
            if tmux has-session -t ${session} 2>/dev/null; then
                s=running
            elif grep -qF '[crashed]' ${log} 2>/dev/null; then
                s=crashed
            elif grep -qF '[done]' ${log} 2>/dev/null; then
                s=done
            else
                s=stopped
            fi
            p=\$(cat ${log} 2>/dev/null | tr '\r' '\n' | grep -oP '\d+%\s+\(\d+/\d+\)' | tail -1)
            printf '%s\t%s\n' \"\$s\" \"\$p\"
        " || true)

        local raw_state="${info_line%%$'\t'*}"
        local raw_progress="${info_line#*$'\t'}"

        # Detect premature finish: done/crashed but progress < 100%
        local pct
        pct=$(echo "$raw_progress" | grep -oP '^\d+' || true)

        case "$raw_state" in
            running)  state="${GREEN}running${NC}" ;;
            done)
                if [[ -n "$pct" && "$pct" -lt 100 ]]; then
                    state="${YELLOW}done-premature${NC}"
                else
                    state="${GREEN}done${NC}"
                fi
                ;;
            crashed)  state="${RED}crashed${NC}" ;;
            *)        state="${YELLOW}stopped${NC}" ;;
        esac

        printf "  %-4s  %-10s  %-32s  %-5s  %-11s  %-22s  " \
            "$i" "$node" "$tag" "$cores" "$core_range" "$session"
        echo -e "${state}  ${raw_progress:-—}"
    done
    echo ""
}

# ── dist-fetch ─────────────────────────────────────────────────────────────────

_fetch_one() {
    local exp="$1"
    local manifest
    manifest="$(_manifest "$exp")"
    [[ -f "$manifest" ]] || error "No manifest for $exp — run dist-launch first"

    local num_drivers
    num_drivers=$(jq '.drivers | length' "$manifest")

    header "DIST-FETCH — $exp: pulling from remote nodes"

    declare -A fetched_nodes
    for (( i=0; i<num_drivers; i++ )); do
        local node
        node=$(jq -r ".drivers[$i].node" "$manifest")
        [[ "$node" == "local" ]] && continue
        [[ -n "${fetched_nodes[$node]+x}" ]] && continue
        fetched_nodes[$node]=1

        # Fetch every outdir assigned to this node in one pass
        for (( j=0; j<num_drivers; j++ )); do
            local jnode joutdir
            jnode=$(jq -r   ".drivers[$j].node"   "$manifest")
            joutdir=$(jq -r ".drivers[$j].outdir" "$manifest")
            [[ "$jnode" != "$node" ]] && continue

            info "← $node  pulling ${joutdir}/"
            mkdir -p "${REPO_DIR}/${joutdir}"
            rsync -az --progress \
                "${node}:${REMOTE_DIR}/${joutdir}/" \
                "${REPO_DIR}/${joutdir}/" \
                || warn "$node: fetch of $joutdir failed (node may be offline)"
            success "$node: ${joutdir} synced"
        done
    done

    info "Local drivers written directly — no fetch needed."
}

# ── dist-aggregate ─────────────────────────────────────────────────────────────

_aggregate_one() {
    local exp="$1"
    local manifest
    manifest="$(_manifest "$exp")"
    [[ -f "$manifest" ]] || error "No manifest for $exp — run dist-launch first"

    local num_drivers
    num_drivers=$(jq '.drivers | length' "$manifest")
    local dest="${REPO_DIR}/results/${exp}"
    mkdir -p "$dest"

    header "DIST-AGGREGATE — merging $exp into results/$exp"

    for (( i=0; i<num_drivers; i++ )); do
        local tag outdir
        tag=$(jq -r    ".drivers[$i].tag"    "$manifest")
        outdir=$(jq -r ".drivers[$i].outdir" "$manifest")
        local src="${REPO_DIR}/${outdir}"

        if [ -d "${src}/${tag}" ]; then
            cp -r "${src}/${tag}" "${dest}/"
            success "Merged: $tag"
        else
            warn "Missing: ${src}/${tag}  (driver $i not finished yet?)"
        fi

        if [ -f "${src}/experiment.log" ]; then
            echo "# ── driver $i: $tag ──" >> "${dest}/experiment.log"
            cat "${src}/experiment.log"    >> "${dest}/experiment.log"
        fi
    done

    success "Done — results in ${dest}/"
    local json_path
    json_path=$(jq -r '.json_path // empty' "$manifest")
    if [[ -n "$json_path" ]]; then
        echo "  Plot: python3 scripts/plot_experiment.py ${json_path}"
    else
        echo "  Plot: python3 scripts/plot_experiment.py experiments/${exp}.json"
    fi
    echo ""
}

# ── dist-stop ──────────────────────────────────────────────────────────────────

_stop_one() {
    local exp="$1"
    local manifest
    manifest="$(_manifest "$exp")"
    [[ -f "$manifest" ]] || error "No manifest for $exp — run dist-launch first"

    local num_drivers
    num_drivers=$(jq '.drivers | length' "$manifest")

    header "DIST-STOP — killing sessions for $exp"

    for (( i=0; i<num_drivers; i++ )); do
        local node session
        node=$(jq -r    ".drivers[$i].node"    "$manifest")
        session=$(jq -r ".drivers[$i].session" "$manifest")

        if run_on_quiet "$node" "tmux has-session -t ${session} 2>/dev/null"; then
            run_on_quiet "$node" "tmux kill-session -t ${session}"
            warn "$node: killed $session"
        else
            info "$node: $session not running"
        fi
    done

    success "$exp: all sessions stopped."
}

# ── dist-nuke ──────────────────────────────────────────────────────────────────

_nuke_one() {
    local exp="$1"
    local manifest
    manifest="$(_manifest "$exp")"

    header "DIST-NUKE — stop + clean $exp"

    if [[ -f "$manifest" ]]; then
        _stop_one "$exp"

        local num_drivers
        num_drivers=$(jq '.drivers | length' "$manifest")

        for (( i=0; i<num_drivers; i++ )); do
            local node outdir dir remote_path
            node=$(jq -r   ".drivers[$i].node"   "$manifest")
            outdir=$(jq -r ".drivers[$i].outdir" "$manifest")
            dir="$(node_dir "$node")"
            remote_path="${dir}/${outdir}"

            if run_on_quiet "$node" "[ -d ${remote_path} ]"; then
                run_on_quiet "$node" "rm -rf ${remote_path}"
                warn "$node: removed ${remote_path}"
            fi

            if [[ "$node" != "local" ]]; then
                local local_copy="${REPO_DIR}/${outdir}"
                if [[ -d "$local_copy" ]]; then
                    rm -rf "$local_copy"
                    warn "local: removed fetched copy $local_copy"
                fi
            fi
        done
    else
        warn "$exp: no manifest found — skipping session stop"
    fi

    local agg_dir="${REPO_DIR}/results/${exp}"
    if [[ -d "$agg_dir" ]]; then
        rm -rf "$agg_dir"
        warn "local: removed aggregated results $agg_dir"
    fi

    success "$exp: nuked."
}

# ── Build + sync (shared by dist-launch) ──────────────────────────────────────

_build_and_sync() {
    info "Building cqsimcpp..."
    cmake --build "$REPO_DIR/build" --target cqsimcpp
    rm -f "$REPO_DIR/cqsimcpp"
    cp "$REPO_DIR/build/cqsimcpp" "$REPO_DIR/cqsimcpp"
    success "Build complete"
    info "Node source: ${NODES_SOURCE}"
    info "Nodes: ${NODES[*]}"

    for node in "${REMOTE_NODES[@]}"; do
        info "→ syncing repo to $node"
        rsync -az --progress \
            --exclude='.git/' --exclude='build/' --exclude='.venv/' \
            --exclude='results/' --exclude='*.pyc' --exclude='__pycache__/' \
            "$REPO_DIR/" "${node}:${REMOTE_DIR}/"
        run_on_quiet "$node" "mkdir -p ${REMOTE_DIR}/results"
        success "$node: sync complete"
    done
}

# ── Public commands ────────────────────────────────────────────────────────────

cmd_dist_plan() {
    [[ $# -gt 0 ]] || error "Usage: dist-plan <exp> [exp2 ...]"
    command -v jq &>/dev/null || error "'jq' is required"
    _plan_assignments "$@"
    _print_global_assignment_summary
    for arg in "$@"; do _plan_one "$arg"; done
}

cmd_dist_launch() {
    [[ $# -gt 0 ]] || error "Usage: dist-launch <exp> [exp2 ...]"
    command -v jq &>/dev/null || error "'jq' is required"
    _plan_assignments "$@"
    _print_global_assignment_summary
    _build_and_sync
    for arg in "$@"; do _launch_one "$arg"; done
}

cmd_dist_status() {
    [[ $# -gt 0 ]] || error "Usage: dist-status <exp> [exp2 ...]"
    command -v jq &>/dev/null || error "'jq' is required"
    local exp
    for arg in "$@"; do
        if ! exp="$(_resolve_exp_for_manifest_commands "$arg")"; then
            return 1
        fi
        _status_one "$exp"
    done
}

cmd_dist_fetch() {
    [[ $# -gt 0 ]] || error "Usage: dist-fetch <exp> [exp2 ...]"
    command -v jq &>/dev/null || error "'jq' is required"
    local exp
    for arg in "$@"; do
        if ! exp="$(_resolve_exp_for_manifest_commands "$arg")"; then
            return 1
        fi
        _fetch_one "$exp"
    done
    echo ""
    success "All fetches complete."
}

cmd_dist_aggregate() {
    [[ $# -gt 0 ]] || error "Usage: dist-aggregate <exp> [exp2 ...]"
    command -v jq &>/dev/null || error "'jq' is required"
    local exp
    for arg in "$@"; do
        if ! exp="$(_resolve_exp_for_manifest_commands "$arg")"; then
            return 1
        fi
        _aggregate_one "$exp"
    done
}

cmd_dist_stop() {
    [[ $# -gt 0 ]] || error "Usage: dist-stop <exp> [exp2 ...]"
    command -v jq &>/dev/null || error "'jq' is required"
    local exp
    for arg in "$@"; do
        if ! exp="$(_resolve_exp_for_manifest_commands "$arg")"; then
            return 1
        fi
        _stop_one "$exp"
    done
}

cmd_dist_nuke() {
    [[ $# -gt 0 ]] || error "Usage: dist-nuke <exp> [exp2 ...]"
    command -v jq &>/dev/null || error "'jq' is required"
    local exp
    for arg in "$@"; do
        if ! exp="$(_resolve_exp_for_manifest_commands "$arg")"; then
            return 1
        fi
        _nuke_one "$exp"
    done
}

cmd_help() {
    cat <<EOF

${BOLD}run_dist.sh${NC} — general-purpose distributed experiment runner

Splits any experiment's drivers across nodes, one driver per tmux session,
then aggregates results into a single directory for plotting.

${BOLD}Usage:${NC}
  ./scripts/run_dist.sh [--nodefile <path>] <command> <exp> [exp2 ...]

  <exp> is a bare experiment name ("exp2a"), a unique prefix, both resolved
  to experiments/<exp>.json, or a direct path to a JSON file
  ("experiments/exp2a.json").

${BOLD}Commands:${NC}
  ${CYAN}dist-plan      <exp> [exp2 ...]${NC}   preview node assignment (dry-run, no changes)
  ${CYAN}dist-launch    <exp> [exp2 ...]${NC}   build + sync + launch all drivers
  ${CYAN}dist-status    <exp> [exp2 ...]${NC}   per-driver state (running / done / stopped)
  ${CYAN}dist-fetch     <exp> [exp2 ...]${NC}   rsync per-driver result dirs from remote nodes
  ${CYAN}dist-aggregate <exp> [exp2 ...]${NC}   merge into results/<exp>/ for plotting
  ${CYAN}dist-stop      <exp> [exp2 ...]${NC}   kill all driver tmux sessions
  ${CYAN}dist-nuke      <exp> [exp2 ...]${NC}   stop + delete all result directories

${BOLD}Node assignment:${NC}
  Drivers are packed by core capacity across the available nodes.
  Multiple experiments are planned together in one pass, so resources are
  balanced across experiments instead of filling one experiment at a time.
  Default node source: repo-root nodefile (or DIST_NODES if set).
  Default capacity: 256 cores per node.
  Driver cores are taken from each JSON driver when specified.
  Default heavy-driver fallback: 64 cores per driver.

${BOLD}Configuration via environment variables:${NC}
  DIST_NODES="ccred2 ccred3 ccred4"          override node list
  DIST_NODEFILE="/path/to/nodefile"          alternate nodefile path (env var)
  --nodefile <path>                          same as DIST_NODEFILE (CLI flag)
                                              (may include "local")
  DIST_INCLUDE_LOCAL=1                       prepend local host to node list
  DIST_REMOTE_DIR="~/CQSimPrivate"           remote working directory
  DIST_HEAVY_TYPES="mcts adaptive_mcts"      space-separated heavy driver types
  DIST_NODE_CORES=256                        per-node core capacity
  DIST_HEAVY_DRIVER_CORES=64                 fallback cores per heavy driver
  DIST_LIGHT_DRIVER_CORES=1                  fallback cores per light driver
  DIST_ALLOW_OVERSUBSCRIBE=1                 allow plans past per-node capacity

${BOLD}Typical workflow:${NC}
  1.  ./scripts/run_dist.sh dist-plan    exp2a
  2.  ./scripts/run_dist.sh dist-launch  exp2a
  3.  ./scripts/run_dist.sh dist-status  exp2a
  4.  ./scripts/run_dist.sh dist-fetch   exp2a
  5.  ./scripts/run_dist.sh dist-aggregate exp2a
  6.  python3 scripts/plot_experiment.py experiments/exp7a.json
  7.  ./scripts/mcts_eval_examples.sh

${BOLD}Requirements:${NC}  jq, tmux, SSH aliases for remote nodes in ~/.ssh/config

EOF
}

# ── Dispatch ───────────────────────────────────────────────────────────────────

CMD="${1:-help}"
shift || true

case "$CMD" in
    dist-plan)      cmd_dist_plan      "$@" ;;
    dist-launch)    cmd_dist_launch    "$@" ;;
    dist-status)    cmd_dist_status    "$@" ;;
    dist-fetch)     cmd_dist_fetch     "$@" ;;
    dist-aggregate) cmd_dist_aggregate "$@" ;;
    dist-stop)      cmd_dist_stop      "$@" ;;
    dist-nuke)      cmd_dist_nuke      "$@" ;;
    help|--help|-h) cmd_help ;;
    *) error "Unknown command: $CMD  (try: help)" ;;
esac
