#!/usr/bin/env bash
# Kill every tmux session on all nodes (local + remotes).
#
# Usage:
#   ./scripts/kill_all_tmux.sh                        # hardcoded default list
#   ./scripts/kill_all_tmux.sh nodefile1 nodefile2    # nodes from nodefiles (1st column)
#   DIST_NODES="local ccred2" ./scripts/kill_all_tmux.sh  # explicit override

set -euo pipefail

if [[ $# -gt 0 ]]; then
    # Read node names from the first column of each nodefile argument
    NODES=()
    for nodefile in "$@"; do
        while IFS= read -r line || [[ -n "$line" ]]; do
            [[ -z "$line" || "$line" == \#* ]] && continue
            node=$(awk '{print $1}' <<< "$line")
            NODES+=("$node")
        done < "$nodefile"
    done
elif [[ -n "${DIST_NODES:-}" ]]; then
    read -ra NODES <<< "$DIST_NODES"
else
    NODES=("local" "ccred2" "ccred3" "ccred4" "ccred5" "ccred6" "ccred7" "ccred8" "ccred9" "ccred10" "ccred11" "ccred12" "ccred13" "ccred14" "ccred15")
fi

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

for node in "${NODES[@]}"; do
    if [[ "$node" == "local" ]]; then
        sessions=$(tmux list-sessions -F '#S' 2>/dev/null || true)
        if [[ -z "$sessions" ]]; then
            echo -e "${YELLOW}[local]${NC} no sessions"
        else
            while IFS= read -r s; do
                tmux kill-session -t "$s" 2>/dev/null && \
                    echo -e "${GREEN}[local]${NC} killed: $s"
            done <<< "$sessions"
        fi
    else
        sessions=$(ssh -o ConnectTimeout=8 -q "$node" \
            "tmux list-sessions -F '#S' 2>/dev/null || true" 2>/dev/null || true)
        if [[ -z "$sessions" ]]; then
            echo -e "${YELLOW}[$node]${NC} no sessions (or unreachable)"
        else
            while IFS= read -r s; do
                ssh -o ConnectTimeout=8 -q "$node" \
                    "tmux kill-session -t '$s' 2>/dev/null" 2>/dev/null && \
                    echo -e "${GREEN}[$node]${NC} killed: $s"
            done <<< "$sessions"
        fi
    fi
done

echo -e "\n${CYAN}Done.${NC}"
