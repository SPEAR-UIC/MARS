# ./scripts/run_dist.sh dist-plan  --nodefile ./nodefile1 exp2a exp2b
# ./scripts/run_dist.sh dist-launch  --nodefile ./nodefile1 exp2a exp2b
# ./scripts/run_dist.sh dist-status exp2a exp2b exp3a exp3b exp4a exp4b 
./scripts/run_dist.sh dist-fetch exp2a exp2b  
./scripts/run_dist.sh dist-aggregate exp2a exp2b
python3 scripts/plot_experiment.py experiments/exp2a.json
python3 scripts/plot_experiment.py experiments/exp2b.json