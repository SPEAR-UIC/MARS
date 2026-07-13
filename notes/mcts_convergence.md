# MCTS Convergence Analysis

Use the synthetic convergence command to study how many iterations MCTS needs before
its current root choice stops changing.

```bash
./build/cqsimcpp mcts-convergence \
  --jobs 12 \
  --capacity 6 \
  --iterations 5000 \
  --sample-every 100 \
  --output results/mcts_convergence.csv
```

The default instance is intentionally high branching:

- All jobs arrive at time 0.
- Each job needs one processor.
- The cluster has fewer processors than jobs.
- `BranchingMode::Permutation` is used, so the root alternatives are distinct sets
  of jobs that can start in the first scheduling cycle.
- With `--jobs 12 --capacity 6`, the root branching factor is 924.

The command prints:

- `root_branching_factor`: number of root choices MCTS has to compare.
- `stabilization_iteration`: the last iteration where the selected root choice
  changed within the requested budget.
- `final_choice`: sorted job IDs actually started by the final selected root branch.
- `final_best_visit_ties`: number of root branches tied for the highest visit count.
- `final_visit_margin`: visit-count lead over the next-best branch; this is `0` when tied.

The CSV includes one row for every choice change plus periodic samples. A stable
choice with many `best_visit_ties` is not strong convergence; rerun with a larger
`--iterations` budget and look for `best_visit_ties` to approach `1` and
`visit_margin` to grow.

Optional plot:

```bash
python3 scripts/plot_mcts_convergence.py results/mcts_convergence.csv
```

## Real Trace Scan

Use `mcts-trace-convergence` to replay real SWF traces with FCFS, detect
high-branching MCTS decision points, and then run convergence traces only at
those snapshots.

```bash
./build/cqsimcpp mcts-trace-convergence \
  --trace polaris=data/polaris24cln.swf \
  --trace theta=data/theta21cln.swf \
  --branching comprehensive_heuristic \
  --window 1024 \
  --min-queue 64 \
  --branch-threshold 30 \
  --top-k 16 \
  --iterations 5000 \
  --sample-every 100 \
  --num-workers 32 \
  --output-dir results/real_trace_mcts_convergence
```

This writes `summary.csv` plus one convergence CSV per selected snapshot. The
summary records the trace, FCFS cycle, simulation time, queue length, root
branching factor, stabilization iteration, and final tie diagnostics.
