<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/MARS-Card.png" />
  <source media="(prefers-color-scheme: light)" srcset="docs/images/MARS-Card.png" />
  <img src="docs/images/MARS-Card.png" alt="MARS Card" />
</picture>

---

# MARS: A Monte Carlo Tree Search-based Adaptive and Responsive Scheduler

Modern High Performance Computing (HPC) systems depend on static
heuristics and manual administration for job scheduling and reservation
management. Deep Reinforcement Learning (DRL) has shown promising
scheduling performance but requires historical training data and fixes
the optimization goal at training time, forcing operators to retrain
whenever priorities shift. We introduce **MARS**
(**M**onte Carlo Tree Search-based
**A**daptive and **R**esponsive
**S**cheduler), a training-free HPC scheduler whose
optimization goal is configurable through a reward function rather than
baked into a learned model. MARS uses a lightweight discrete-event
simulator to explore the future consequences of scheduling decisions
within a strict time budget, adapting to the configured reward at each
scheduling cycle. If you use MARS in your work, please cite the following papers:

Yash Kurkure, Yihe Zhang, Zhiling Lan, Michael E. Papka, “MARS: A Monte Carlo Tree Search based Adaptive and Responsive Scheduler”, Proc. of IEEE Cluster, 2026.

## Running on a Slurm cluster

`slurm/submit.sh` runs each driver of an experiment as its own Slurm job,
tracks every driver's progress, and merges the results once all drivers have
finished. With no arguments it runs the four SchedTwin experiments,
`experiments/schedtwin-{aurora,frontier,polaris,theta}.json`. Each of them
compares MARS-CU and MARS-CW against the LRF, LJF, Random, FCFS, SJF, F1 and
WFP baselines.

### Requirements

- The repository is on a filesystem shared by the login and compute nodes.
- The login node has a C++17 compiler with OpenMP, CMake 3.20 or newer, and
  Python 3.6 or newer. The first build downloads GoogleTest, so the login node
  also needs internet access.
- Compute nodes have 48 CPUs. For other node sizes, change `--cpus-per-task`
  in `slurm/*.sbatch` and `num_cores` in the experiment files.
- The partition allows multi-day jobs. On the full traces a MARS driver runs
  for days, and a driver that is killed has to start over.

### Quick start

Run these on the login node from the repository root. The jobs inherit the
environment you submit from, so load your modules first.

```bash
module load gcc cmake python         # module names depend on your cluster
slurm/submit.sh --smoke              # end-to-end test on ~400 jobs, a few minutes
slurm/submit.sh                      # full run
python3 scripts/slurm_dist.py status latest --watch 60
```

`submit.sh` builds `cqsimcpp` in Release mode and stops if the binary is not
linked against OpenMP, since MARS would otherwise run on a single thread. It
then submits three jobs:

| Job | Resources | What it runs |
|---|---|---|
| `mars-heavy` (job array) | one exclusive node per task | one MARS driver per task |
| `mars-light` | one exclusive node | every baseline driver, concurrently |
| `mars-aggregate` | one CPU, starts after both jobs end (`afterany`) | merges outputs and writes summaries |

For the four SchedTwin experiments this is 8 MARS drivers and 28 baselines,
so the run uses at most 9 nodes.

### Options

```
slurm/submit.sh [--no-build] [--smoke] [--dry-run] [experiment ...]
```

| Option | Effect |
|---|---|
| `--no-build` | Reuse the existing `./cqsimcpp`. |
| `--smoke` | Run about 400 jobs around the first target job, with a 200 ms MCTS budget on 4 cores. |
| `--dry-run` | Create the run directory and print the `sbatch` commands without submitting them. |
| `experiment` | A name from `experiments/` or a path to a JSON file, for example `slurm/submit.sh exp2a exp2b`. |

| Environment variable | Default | Purpose |
|---|---|---|
| `MARS_PARTITION` | `compute` | Partition for every job. |
| `MARS_SBATCH_ARGS` | none | Extra `sbatch` arguments, for example `--account=myproject`. |
| `MARS_LIGHT_PARALLEL` | CPUs on the node | Maximum number of baseline drivers running at once. |

### Monitoring

```bash
python3 scripts/slurm_dist.py status latest              # one snapshot
python3 scripts/slurm_dist.py status latest --watch 60   # refresh every 60 s
```

Every driver rewrites a status file every 30 seconds. The status command
prints one table per experiment:

```
== schedtwin-theta
  Tag        State      Progress                         Jobs done         Elapsed       ETA  Node                   Slurm job
  FCFS       done       [####################]   100.0%  11762/11762         0m03s         -  compute-03             4812
  MARS-CW    running    [#######.............]    35.2%  4140/11762         14h02m     1d01h  compute-07             4811_6
```

| State | Meaning |
|---|---|
| `pending` | The driver has not started yet. |
| `running` | The driver is running and its status file is being updated. |
| `done` | The simulator exited normally. |
| `failed` | The simulator exited with an error. Check the driver's `run.log`. |
| `cancelled` | The job was cancelled, for example with `scancel`. |
| `lost` | No update for 5 minutes and Slurm no longer lists the job, for example after a node failure. |

To follow a single driver's raw output, run
`tail -f results/slurm_runs/latest/drivers/<experiment>_d<index>/run.log`.
`submit.sh` prints the `scancel` command for the run. The job IDs are also
stored in `slurm_jobs.json` in the run directory.

### Results

The aggregation job writes:

| Path | Contents |
|---|---|
| `results/<experiment>/<tag>/` | Each driver's `events.csv`, `performance.csv` and `descisions.csv`, in the same layout as a local run. |
| `results/<experiment>/summary.csv` | The simulator's metrics over all jobs. |
| `results/<experiment>/summary_target.{csv,json}` | Wait time, bounded slowdown and utilization over the target jobs only. |
| `results/slurm_runs/<RUN_ID>/aggregate/summary.md` | One table per experiment, comparing each MARS driver with the best baseline. |

The SchedTwin traces pad the 10,000 target jobs with warm-up and cool-down
jobs, and their SWF headers give the target ID range. For traces without that
header line, the target metrics cover all jobs.

Aggregation also runs when some drivers failed, and marks those drivers as
incomplete. To rerun a failed MARS driver and then refresh the summaries, use
the following commands. Here `<i>` is the driver's 0-based line number in
`heavy.list`:

```bash
RUN=results/slurm_runs/latest
sbatch --partition=compute --array=<i> slurm/heavy.sbatch "$(realpath $RUN)"
python3 scripts/slurm_dist.py aggregate latest   # after the job finishes
```

Each run keeps its working files in `results/slurm_runs/<RUN_ID>/`, and
`latest` points to the most recent run:

| Path | Contents |
|---|---|
| `manifest.json` | Every driver in the run. |
| `configs/` | The single-driver configuration files. |
| `status/` | The per-driver status files. |
| `drivers/<key>/` | The raw outputs and `run.log` for each driver. |
| `logs/` | Slurm's output files. |
