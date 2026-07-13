#!/bin/bash


# ./scripts/run_dist.sh dist-plan --nodefile ./nodefile1 exp2a exp2b  
# ./scripts/run_dist.sh dist-launch --nodefile ./nodefile1 exp2a exp2b 
# ./scripts/run_dist.sh dist-status exp2a exp2b 
# ./scripts/run_dist.sh dist-fetch exp2a exp2b 
# ./scripts/run_dist.sh dist-aggregate exp2a exp2b 
# python3 scripts/plot_experiment.py experiments/exp2a.json
# python3 scripts/plot_experiment.py experiments/exp2b.json
# ./scripts/run_dist.sh dist-nuke exp2a exp2b 

# python3 scripts/exp2_plot.py experiments/exp2a.json   # Theta 2021
# python3 scripts/exp2_plot.py experiments/exp2b.json   # Polaris 2024
# python3 scripts/exp2ab_plot.py
python3 scripts/exp_final.py experiments/exp2a.json experiments/exp2b.json