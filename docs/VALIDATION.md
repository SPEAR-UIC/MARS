# Simulator Validation

MARS is built using a rebuilt C++ version of [CQSIM](https://github.com/SPEAR-UIC/CQSim), originally written in Python. To validate the C++ version, we tested the output of the C++ against the known and previously validated Python version.

The output of the Python version can be found under [cqsim-python/data/Results](cqsim-python/data/Results). Here we have extracted the scheduling events in csv files after repalying the logs under various heursitc scehduling polices. Similarly, for the C++ version the [results](results/) folder contains the csv outputs for the C++ version.

Here we provide the steps to recreate the output files and run the tests using the provided scripts.

# Recreating CQSIM Python outputs

The python version of cqsim lives under `cqsim-python/src` in the root of this repository. It must be run from this directory for all the paths to work.

The following commands will generate outputs for FCFS with EASY backfill on both the Polaris 2024 and Theta 2021 logs:
```
python3 cqsim.py -C fcfs_polaris24.set -e polaris24_events.csv -x
```
```
python3 cqsim.py -C fcfs_theta21.set -e theta21_events.csv -x
```
The `csv` output files can be observed inside `cqsim-python/data/Results/`.

# Recreating CQSIM C++ outputs

From the root of the repository, run the following commands:
```
./cqsimcpp ./experiments/cqsimpy-test1.json 
./cqsimcpp ./experiments/cqsimpy-test2.json 
```
Each run writes `<output_dir>/<driver_tag>/events.csv` (e.g.
`results/cqsimpy-test1/FCFS/events.csv`). This raw file can be compared
directly -- `scripts/validation.py` normalizes the C++-only event kinds
itself, so no separate conversion step is needed.

# Validation Script

`scripts/validation.py` compares a C++ `events.csv` against a python
`events.csv` for the same trace: it matches jobs by id, reports coverage
(jobs completed on only one side) and per-field deviation (submit/start/end),
finds the first point of divergence in submit order, and writes plots plus a
text report to `results/cpp_vs_py/<label>/`.

Only the Theta 2021 trace has a finished C++ run right now, so the defaults
point at it -- running with no arguments compares `theta21cln`:
```
python3 scripts/validation.py
```
Once the Polaris 2024 C++ run finishes, point the flags at it:
```
python3 scripts/validation.py \
    --cpp-events results/cqsimpt-test2/FCFS/events.csv \
    --py-events cqsim-python/data/Results/polaris24_events.csv \
    --swf data/polaris24cln.swf \
    --label polaris24cln
```
