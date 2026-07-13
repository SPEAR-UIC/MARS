#!/usr/bin/env bash
# scripts/remote_experiments.sh
#
# Deploy, launch, monitor, and fetch results for exp5a/b/c on remote nodes.
#
# Node assignment:
#   ccred2  →  exp5a  (reward comparison,    8 MCTS + WFP3)
#   ccred3  →  exp5b  (window comparison,    5 MCTS + WFP3)
#   ccred4  →  exp5c  (search time,          4 MCTS + WFP3)
#
# Usage:
#   ./scripts/remote_experiments.sh deploy    # sync repo to all nodes
#   ./scripts/remote_experiments.sh launch    # start experiments in tmux on each node
#   ./scripts/remote_experiments.sh monitor   # open local tmux with live log tails
#   ./scripts/remote_experiments.sh status    # show driver completion counts
#   ./scripts/remote_experiments.sh fetch     # pull results/ back to this machine
#   ./scripts/remote_experiments.sh all       # deploy + launch  (then monitor/fetch separately)
#   ./scripts/remote_experiments.sh stop      # kill tmux sessions on all nodes
#
# Requirements on remote nodes:
#   - SSH alias configured (ccred2/ccred3/ccred4 in ~/.ssh/config or /etc/hosts)
#   - Same OS and CPU architecture as this machine (binary is rsynced directly)
#   - tmux installed

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="~/CQSimPrivate"            # destination path on each remote node
TMUX_SESSION="cqsim"                   # tmux session name used on remote nodes
MONITOR_SESSION="cqsim-monitor"        # local tmux session for watching logs

# Node → experiment mapping
declare -A NODE_EXP
NODE_EXP["ccred2"]="exp5a"
NODE_EXP["ccred3"]="exp5b"
NODE_EXP["ccred4"]="exp5c"

NODES=("ccred2" "ccred3" "ccred4")

# Driver counts per experiment (WFP3 + MCTS variants)
declare -A EXP_DRIVER_COUNT
EXP_DRIVER_COUNT["exp5a"]=9    # WFP3 + 8 reward variants
EXP_DRIVER_COUNT["exp5b"]=6    # WFP3 + 5 window sizes
EXP_DRIVER_COUNT["exp5c"]=5    # WFP3 + 4 search times

# ── Colour helpers ─────────────────────────────────────────────────────────────

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[remote]${NC} $*"; }
success() { echo -e "${GREEN}[remote]${NC} $*"; }
warn()    { echo -e "${YELLOW}[remote]${NC} $*"; }
error()   { echo -e "${RED}[remote]${NC} $*" >&2; exit 1; }
header()  { echo -e "\n${BOLD}$*${NC}"; }

# ── SSH helper: run a command on a node, print output prefixed with node name ──

ssh_run() {
    local node="$1"; shift
    ssh -o ConnectTimeout=10 "$node" "$@"
}

ssh_run_quiet() {
    local node="$1"; shift
    ssh -o ConnectTimeout=10 -q "$node" "$@"
}

# ── deploy ─────────────────────────────────────────────────────────────────────
# rsync the repo to every node.  Excludes:
#   - build/   (compiled objects — only the final binary is synced)
#   - results/ (we pull results back, not push them)
#   - .git/    (not needed on remote)
#   - .venv/   (Python environment — not needed for C++ runs)

cmd_deploy() {
    header "DEPLOY — syncing repo to all nodes"

    # Ensure the binary is up-to-date before syncing
    if [ ! -f "$REPO_DIR/cqsimcpp" ]; then
        warn "Binary $REPO_DIR/cqsimcpp not found — building first..."
        cmake --build "$REPO_DIR/build" --target cqsimcpp
        cp "$REPO_DIR/build/cqsimcpp" "$REPO_DIR/cqsimcpp"
    fi

    for node in "${NODES[@]}"; do
        info "→ $node  (${NODE_EXP[$node]})"
        rsync -az --progress \
            --exclude='.git/' \
            --exclude='build/' \
            --exclude='.venv/' \
            --exclude='results/' \
            --exclude='*.pyc' \
            --exclude='__pycache__/' \
            "$REPO_DIR/" \
            "${node}:${REMOTE_DIR}/"

        # Create the results directory on the remote so the experiment can write there
        ssh_run_quiet "$node" "mkdir -p ${REMOTE_DIR}/results"
        success "$node: sync complete"
    done
}

# ── launch ─────────────────────────────────────────────────────────────────────
# Start each experiment inside a named tmux session on its node.
# The session stays alive after you disconnect from SSH.

cmd_launch() {
    header "LAUNCH — starting experiments on remote nodes"

    for node in "${NODES[@]}"; do
        exp="${NODE_EXP[$node]}"
        info "→ $node  running $exp"

        # Check if a session already exists
        if ssh_run_quiet "$node" "tmux has-session -t ${TMUX_SESSION} 2>/dev/null"; then
            warn "$node: tmux session '${TMUX_SESSION}' already exists — skipping"
            warn "  To restart: run './scripts/remote_experiments.sh stop' first"
            continue
        fi

        # Launch: create detached tmux session, run the experiment, log stdout+stderr
        ssh_run_quiet "$node" "
            tmux new-session -d -s ${TMUX_SESSION} -x 220 -y 50 \
                \"cd ${REMOTE_DIR} && ./cqsimcpp run config experiments/${exp}.json \
                  2>&1 | tee -a results/${exp}/run.log; \
                  echo '[done] experiment ${exp} finished'; \
                  exec bash\"
        "
        success "$node: launched '$exp' in tmux session '${TMUX_SESSION}'"
    done

    echo ""
    info "To attach to a node's session interactively:"
    for node in "${NODES[@]}"; do
        echo "    ssh $node -t 'tmux attach -t ${TMUX_SESSION}'"
    done
    echo ""
    info "To watch logs from this machine:"
    echo "    ./scripts/remote_experiments.sh monitor"
}

# ── status ─────────────────────────────────────────────────────────────────────
# SSH to each node and report driver completion counts.

cmd_status() {
    header "STATUS — driver completion on each node"
    printf "  %-10s  %-10s  %-10s  %-10s\n" "Node" "Experiment" "Done" "Total"
    printf "  %-10s  %-10s  %-10s  %-10s\n" "----------" "----------" "----" "-----"

    for node in "${NODES[@]}"; do
        exp="${NODE_EXP[$node]}"
        total="${EXP_DRIVER_COUNT[$exp]}"
        log_path="${REMOTE_DIR}/results/${exp}/experiment.log"

        # Count "finished" lines in the experiment log
        done_count=$(ssh_run_quiet "$node" \
            "grep -c '\[run\].*finished in' ${log_path} 2>/dev/null || echo 0" 2>/dev/null || echo "?")

        # Check if tmux session is still running
        if ssh_run_quiet "$node" "tmux has-session -t ${TMUX_SESSION} 2>/dev/null"; then
            state="${GREEN}running${NC}"
        else
            if [ "$done_count" = "$total" ] 2>/dev/null; then
                state="${GREEN}complete${NC}"
            else
                state="${YELLOW}stopped${NC}"
            fi
        fi

        printf "  %-10s  %-10s  " "$node" "$exp"
        echo -e "${done_count}/${total}         ${state}"
    done
    echo ""

    # Show last few lines of each log
    for node in "${NODES[@]}"; do
        exp="${NODE_EXP[$node]}"
        log_path="${REMOTE_DIR}/results/${exp}/experiment.log"
        echo -e "${CYAN}── $node ($exp) — last 3 log lines:${NC}"
        ssh_run_quiet "$node" \
            "tail -3 ${log_path} 2>/dev/null || echo '  (no log yet)'" \
            | sed 's/^/  /'
        echo ""
    done
}

# ── monitor ────────────────────────────────────────────────────────────────────
# Open a local tmux session with 3 panes, each tailing a remote experiment log.
# Requires tmux to be installed locally.

cmd_monitor() {
    header "MONITOR — opening local tmux session '${MONITOR_SESSION}'"

    if ! command -v tmux &>/dev/null; then
        error "tmux not found locally. Install with: sudo apt-get install tmux"
    fi

    # Kill existing monitor session if present
    tmux kill-session -t "${MONITOR_SESSION}" 2>/dev/null || true

    # Build the tail command for a node (retries until the log appears)
    tail_cmd() {
        local node="$1" exp="$2"
        local log="${REMOTE_DIR}/results/${exp}/experiment.log"
        # Loop: wait for log to appear, then tail it
        echo "ssh -o ConnectTimeout=10 -q ${node} \
            'while [ ! -f ${log} ]; do \
               echo \"[waiting for ${exp}/experiment.log...]\"; sleep 5; \
             done; \
             echo \"--- ${node}: ${exp} ---\"; \
             tail -n 50 -f ${log}'"
    }

    # Create session with first node
    node0="${NODES[0]}"; exp0="${NODE_EXP[$node0]}"
    tmux new-session -d -s "${MONITOR_SESSION}" -x 220 -y 60 \
        -n "logs" \
        "$(tail_cmd "$node0" "$exp0"); exec bash"

    # Split vertically for second node
    node1="${NODES[1]}"; exp1="${NODE_EXP[$node1]}"
    tmux split-window -t "${MONITOR_SESSION}:logs" -v \
        "$(tail_cmd "$node1" "$exp1"); exec bash"

    # Split the bottom pane horizontally for third node
    node2="${NODES[2]}"; exp2="${NODE_EXP[$node2]}"
    tmux split-window -t "${MONITOR_SESSION}:logs.1" -h \
        "$(tail_cmd "$node2" "$exp2"); exec bash"

    # Even out pane sizes
    tmux select-layout -t "${MONITOR_SESSION}:logs" even-vertical

    # Add a status pane in a second window that refreshes every 30s
    tmux new-window -t "${MONITOR_SESSION}" -n "status" \
        "watch -n 30 '$(realpath "${BASH_SOURCE[0]}") status'; exec bash"

    tmux select-window -t "${MONITOR_SESSION}:logs"
    tmux select-pane -t "${MONITOR_SESSION}:logs.0"

    success "Monitor session ready. Attaching..."
    echo ""
    echo "  Pane layout:"
    printf "  %-8s  %-10s  %s\n" "Node" "Experiment" "Pane"
    printf "  %-8s  %-10s  %s\n" "------" "----------" "----"
    printf "  %-8s  %-10s  %s\n" "$node0" "$exp0" "top"
    printf "  %-8s  %-10s  %s\n" "$node1" "$exp1" "bottom-left"
    printf "  %-8s  %-10s  %s\n" "$node2" "$exp2" "bottom-right"
    echo ""
    echo "  Window 'logs'   — live experiment.log tails"
    echo "  Window 'status' — driver completion (refreshes every 30s)"
    echo ""
    echo "  Detach with Ctrl-b d   |   Kill with: tmux kill-session -t ${MONITOR_SESSION}"
    echo ""

    tmux attach-session -t "${MONITOR_SESSION}"
}

# ── fetch ──────────────────────────────────────────────────────────────────────
# rsync results/ back from each remote node into the local results/ directory.
# Safe to run multiple times — only changed files are transferred.

cmd_fetch() {
    header "FETCH — pulling results back to this machine"

    for node in "${NODES[@]}"; do
        exp="${NODE_EXP[$node]}"
        info "← $node  pulling results/${exp}/"
        rsync -az --progress \
            "${node}:${REMOTE_DIR}/results/${exp}/" \
            "$REPO_DIR/results/${exp}/"
        success "$node: results/${exp}/ synced"
    done

    echo ""
    success "All results available locally under $REPO_DIR/results/"
    echo ""
    info "Run plots with:"
    echo "  python3 scripts/plot_experiment.py experiments/exp7a.json"
    echo "  python3 scripts/plot_experiment.py experiments/exp7b.json"
}

# ── stop ───────────────────────────────────────────────────────────────────────
# Kill the tmux session on each remote node.

cmd_stop() {
    header "STOP — killing tmux sessions on remote nodes"

    for node in "${NODES[@]}"; do
        if ssh_run_quiet "$node" "tmux has-session -t ${TMUX_SESSION} 2>/dev/null"; then
            ssh_run_quiet "$node" "tmux kill-session -t ${TMUX_SESSION}"
            warn "$node: session '${TMUX_SESSION}' killed"
        else
            info "$node: no active session '${TMUX_SESSION}'"
        fi
    done
}

# ── all ────────────────────────────────────────────────────────────────────────

cmd_all() {
    cmd_deploy
    cmd_launch
    echo ""
    success "Deploy and launch complete."
    echo ""
    info "Next steps:"
    echo "  Watch progress:  ./scripts/remote_experiments.sh monitor"
    echo "  Quick status:    ./scripts/remote_experiments.sh status"
    echo "  Pull results:    ./scripts/remote_experiments.sh fetch"
}

# ── help ───────────────────────────────────────────────────────────────────────

cmd_help() {
    cat <<EOF

${BOLD}remote_experiments.sh${NC} — run exp5a/b/c on ccred2/3/4

Node assignment:
  ccred2  →  exp5a  (reward comparison,  8 MCTS + WFP3)
  ccred3  →  exp5b  (window comparison,  5 MCTS + WFP3)
  ccred4  →  exp5c  (search time,        4 MCTS + WFP3)

Commands:
  ${CYAN}deploy${NC}    rsync repo + binary to all nodes (skips unchanged files)
  ${CYAN}launch${NC}    start each experiment in a detached tmux session on its node
  ${CYAN}monitor${NC}   open a local tmux session with live log tails from all nodes
  ${CYAN}status${NC}    SSH to each node and show driver completion counts
  ${CYAN}fetch${NC}     rsync results/ back from all nodes to this machine
  ${CYAN}stop${NC}      kill tmux sessions on all nodes
  ${CYAN}all${NC}       deploy + launch  (convenience shortcut)

Typical workflow:
  1.  ./scripts/remote_experiments.sh all       # deploy and start
  2.  ./scripts/remote_experiments.sh monitor   # watch live logs
  3.  ./scripts/remote_experiments.sh fetch     # pull results when done
  4.  python3 scripts/plot_experiment.py experiments/exp7a.json

Attaching to a node manually:
  ssh ccred2 -t 'tmux attach -t ${TMUX_SESSION}'

EOF
}

# ── Entry point ────────────────────────────────────────────────────────────────

CMD="${1:-help}"
case "$CMD" in
    deploy)  cmd_deploy  ;;
    launch)  cmd_launch  ;;
    monitor) cmd_monitor ;;
    status)  cmd_status  ;;
    fetch)   cmd_fetch   ;;
    stop)    cmd_stop    ;;
    all)     cmd_all     ;;
    help|--help|-h) cmd_help ;;
    *) error "Unknown command: $CMD  (run with 'help' to see usage)" ;;
esac
