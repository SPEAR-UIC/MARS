#!/bin/bash

# Run the experiments
./cqsimcpp ./experiments/exp2a.json
./cqsimcpp ./experiments/exp2b.json

# Make the plots from CLUSTER'26 submission
python3 scripts/plot_cluster26.py experiments/exp2a.json experiments/exp2b.json