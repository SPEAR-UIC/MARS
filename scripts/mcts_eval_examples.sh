#!/usr/bin/env bash
# Print ready-to-run command examples for the short MCTS experiment names.

set -euo pipefail

cat <<'EOF'
# Single distributed run example
./scripts/run_dist.sh dist-plan exp2a
./scripts/run_dist.sh dist-launch exp2a
./scripts/run_dist.sh dist-status exp2a
./scripts/run_dist.sh dist-fetch exp2a
./scripts/run_dist.sh dist-aggregate exp2a
python3 scripts/plot_experiment.py experiments/exp2a.json

# Run the whole Theta suite locally
./cqsimcpp experiments/exp1a.json
./cqsimcpp experiments/exp2a.json
./cqsimcpp experiments/exp3a.json
./cqsimcpp experiments/exp4a.json
./cqsimcpp experiments/exp5a.json
./cqsimcpp experiments/exp6a.json
./cqsimcpp experiments/exp7a.json

# Run the whole Polaris suite locally
./cqsimcpp experiments/exp1b.json
./cqsimcpp experiments/exp2b.json
./cqsimcpp experiments/exp3b.json
./cqsimcpp experiments/exp4b.json
./cqsimcpp experiments/exp5b.json
./cqsimcpp experiments/exp6b.json
./cqsimcpp experiments/exp7b.json

# Distributed helpers for the whole Theta suite
for exp in exp1a exp2a exp3a exp4a exp5a exp6a exp7a; do ./scripts/run_dist.sh dist-plan "$exp"; done
for exp in exp1a exp2a exp3a exp4a exp5a exp6a exp7a; do ./scripts/run_dist.sh dist-launch "$exp"; done
for exp in exp1a exp2a exp3a exp4a exp5a exp6a exp7a; do ./scripts/run_dist.sh dist-status "$exp"; done
for exp in exp1a exp2a exp3a exp4a exp5a exp6a exp7a; do ./scripts/run_dist.sh dist-fetch "$exp"; done
for exp in exp1a exp2a exp3a exp4a exp5a exp6a exp7a; do ./scripts/run_dist.sh dist-aggregate "$exp"; done
for exp in exp1a exp2a exp3a exp4a exp5a exp6a exp7a; do ./scripts/run_dist.sh dist-nuke "$exp"; done

# Distributed helpers for the whole Polaris suite
for exp in exp1b exp2b exp3b exp4b exp5b exp6b exp7b; do ./scripts/run_dist.sh dist-plan "$exp"; done
for exp in exp1b exp2b exp3b exp4b exp5b exp6b exp7b; do ./scripts/run_dist.sh dist-launch "$exp"; done
for exp in exp1b exp2b exp3b exp4b exp5b exp6b exp7b; do ./scripts/run_dist.sh dist-status "$exp"; done
for exp in exp1b exp2b exp3b exp4b exp5b exp6b exp7b; do ./scripts/run_dist.sh dist-fetch "$exp"; done
for exp in exp1b exp2b exp3b exp4b exp5b exp6b exp7b; do ./scripts/run_dist.sh dist-aggregate "$exp"; done
for exp in exp1b exp2b exp3b exp4b exp5b exp6b exp7b; do ./scripts/run_dist.sh dist-nuke "$exp"; done
# Plot the final showdown configs
python3 scripts/plot_experiment.py experiments/exp7a.json
python3 scripts/plot_experiment.py experiments/exp7b.json
EOF
