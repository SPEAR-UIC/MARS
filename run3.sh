# ./scripts/run_dist.sh dist-plan  --nodefile ./nodefile2 exp7b
# ./scripts/run_dist.sh dist-launch  --nodefile ./nodefile2 exp7b
# ./scripts/run_dist.sh dist-status exp7b
./scripts/run_dist.sh dist-fetch exp7b  
./scripts/run_dist.sh dist-aggregate exp7b
python3 scripts/plot_experiment.py experiments/exp7b.json