# CLUSTER 2026


## Evaluation Environment

Evaluations for MARS were conducted in a simulated environment. We have carefully separated our codebase to modularize the simulated environment and the scheduler logic. More details on the archtecture of MARS and the evaulation framework can be found in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

For any reproducibilty attempts we document below our exact setup used for our evaluations:

### Hardware Setup

Our evaluations were conducted on [Chamelon Cloud](https://chameleoncloud.org) on x86_64 bare metal nodes. We specicifally leased out the a single bare metal node from TACC equipped with a 2 CPUS of AMD EPYC 7763 64-Core Processor. In total the node had 256 threads and 256 GiB of RAM.

### Software Setup

The bare metal node was installed with Ubuntu 24.04 LTS, with the the following software dependencies and versions:

- g++ 13.3.0
- OpenMP 4.5
- Cmake 4.2.3
- Google Tests 1.15.2
- Python 3.12.3 inside a virtual environment

All dependencies can be installed using `setup.sh`, which also builds the binary for MARS.

## Unit Testing and Simulator verification

To validate and test our simulator we include a comprehensive test suite under `tests` using google tests. These tests can be run using the `gtests.sh` script.

TODO: Add tests for simulator verification against python version.

## Running Simulated Experiments

The `setup.sh` script outputs a binary called `cqsimcpp` at the root of the project. This binary is used to run the experiments reported in our submission. It takes input a experiment configuration files that lists the various polices and the log file that needs to be used for experiments. We have provided two simple samples: `sample-1.json` and `sample-2.json` under the `experiments` directory.

To run an experiment you can use this config file as:
```
./cqsimcpp ./experiments/sample-1.json 
```
The first sample is compares three heuristic scheduling policies on a year long production log from the Theta Supercomputer. The second compares the same but includes the simulation of maintenance windows.


## Experiments from submission
All the logs used in our evaluations can be found under the `data` directory. To run the results for the Theta 2021 log use:
```
./cqsimcpp ./experiments/exp2a.json
```
For the Polaris 2024 log, use:
```
./cqsimcpp ./experiments/exp2b.json
```

Each experiment runs multiple simulations of the same log with the different polices one after the other. The simulations for `MCTS-CU` and `MCTS-CW` take the longest, as each scheduling cycle can last upto `15 seconds`.

If the reporoducer has access to multiple compute nodes over `ssh`, the script `/scripts/run_dist.sh` may help to run the longer running simulations in parallel. More details on its usage can be found the in the scripts header.
