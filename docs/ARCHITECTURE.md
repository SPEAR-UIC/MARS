# `src` overview

MARS is a scheduling policy (including an MCTS-based scheduler) embedded inside a
CQSim-derived event-driven simulator, used to experiment with and evaluate scheduling
decisions against HPC job traces. The code is organized into six directories:

```
src/
├── core/          data structures shared by every other layer
├── adapters/      the event-driven simulation engine (CQSim C++ port)
│   └── simulation/
├── policies/      heuristic SchedulerPolicy implementations (FIFO, SJF, WFP, ...)
├── mcts/          the MCTS scheduler, built on top of adapters/simulation
├── experiment/    orchestrates one or more Drivers and reports metrics
└── io/            trace/config file readers (SWF job traces, maintenance windows)
```

## `core`

Defines the plain data structures used across the codebase: `Job`/`SimJob`
(`job.h`), `NodeInfo` (`resources.h`, `node.h`), `Decision` (`decision.h`), and
`State` (`state.h`) — a snapshot of pending/running jobs and node state handed
to a policy at schedule time. `policy.h` defines the `SchedulerPolicy`
interface (`schedule(State) -> vector<Decision>`) that every scheduler —
heuristic or MCTS — implements. Nothing in `core` depends on the simulator
itself; `State::sim` is an optional pointer back to the `Simulator`, set only
when a policy is running inside a simulation.

## `adapters/simulation`

The event-driven CQSim adaptation, written in C++. Two classes matter here:

- **`Simulator`** — owns all simulation state: the event queue, job queue,
  node/resource occupancy, and running/completed job bookkeeping. It exposes
  callbacks (`set_scheduling_callback`) invoked on each `SchedulingCycle`
  event, plus helpers (`try_start_job`, `available_procs`, `step`, ...) that a
  policy or driver uses to move the simulation forward. A `Simulator` can run
  in `SimMode::Real` (jobs end at their true runtime) or
  `SimMode::Prediction` (jobs end at their predicted/estimated walltime) —
  this mode distinction is what separates the `Driver`'s ground-truth
  execution from MCTS's speculative look-ahead (see below).
- **`Driver`** — represents a single simulation timeline. It owns one
  `Simulator` plus one `SchedulerPolicy`, and repeatedly calls `simulator.step()`
  to advance simulated time, invoking the policy's callback at each
  scheduling cycle and logging events/metrics as it goes. The `Driver`
  always runs its `Simulator` in `SimMode::Real`, i.e. against actual job
  end times, since it produces the numbers used for evaluation.

## `policies`

One header per heuristic `SchedulerPolicy` (FIFO, SJF, LJF, SRF, LRF, SCF,
LCF, WFP/WFP1, F1–F4, FAT, UNICEP, FCSJ, LCFS...). Most share a common
comparator-based skeleton from `sort_policy.h` and differ only in how they
order the pending queue before dispatch.

## `mcts`

The MCTS-based scheduler. `Mcts::search` takes a `Simulator` snapshot handed
to it by the `Driver` (via `Driver::simulator()`) and repeatedly clones it
(`Simulator::get_copy`) to build a search tree of future states, at each
expansion applying an action and then fast-forwarding time — critically,
using each job's *predicted* end time (`SimMode::Prediction`) rather than its
real one, since the true end time isn't knowable during planning. `MctsPolicy`
wraps `Mcts` as a `SchedulerPolicy`, so it drops into a `Driver` exactly like
any heuristic policy in `policies/`. This is the key Driver/MCTS contrast:
the `Driver`'s own `Simulator` always advances on real end times for
evaluation; the copies MCTS explores internally advance on predicted end
times for planning.

## `experiment`

`Experiment` holds and runs multiple `Driver`s (`register_driver`), one per
scheduling policy under test (e.g. FIFO vs. WFP vs. MCTS on the same job
trace), so they can be evaluated side by side. `JsonExperiment` builds an
`Experiment` from a JSON config file. `metrics.h` defines the metrics
(average wait time, makespan, etc.) written out after all drivers complete.

## `io`

Input readers decoupled from simulation logic: `swf_reader` parses Standard
Workload Format (SWF) job traces into `SimJob`s, and `maintenance_reader`
parses maintenance-window schedules consumed by `Simulator::add_maintenance_windows`.