# Backfill Divergence Investigation: C++ vs Python cqsim (FCFS)

This document records an investigation into why the C++ reimplementation
(`cqsimcpp`) and the reference Python simulator (`cqsim-python`) produce
different job start/end times under FCFS + EASY backfill, using
`scripts/compare_cpp_python.py` against the `theta21cln` and `polaris24cln`
traces (experiments `cqsimpy-test1` / `cqsimpy-test2`).

No source changes came out of this investigation — see `changes.md` in this
directory. This file exists so the analysis isn't lost.

## Background

`scripts/compare_cpp_python.py` compares each side's `events.csv`
(`sim_time,event,id`, with `Submit` / `Run` / `Backfill` / `End` rows) job by
job, matched on job id, and reports the first job (in submit-time order)
where the two sides' `start`/`end` timestamps disagree.

## Divergence 1: job 49 (theta21cln) — stale build, not a real bug

The first run flagged job 49 as backfilling immediately in C++ while Python
delayed it by ~525,000s. Tracing the C++ `schedule_with_backfill_reservation`
logic (`src/core/backfill.h`) by hand — and confirming with temporary
`fprintf` instrumentation — showed job 49 *should* have been rejected by the
current source (both admission conditions evaluate false against the
4096-proc head job, job 24, which needs the entire cluster).

The `results/cqsimpy-test1` directory on disk was stale, built from an older
`cqsimcpp` binary. Rebuilding from current source (`cmake --build build
--target cqsimcpp`) and rerunning `experiments/cqsimpy-test1.json` /
`cqsimpy-test2.json` made this divergence disappear entirely: job 24 (the
full-machine job) now starts at the identical timestamp in both simulators.

**Lesson**: always rebuild before trusting `results/` output when
investigating a divergence — an on-disk `events.csv` is only as fresh as the
binary that produced it.

## Divergence 2: job 75 (theta21cln), job 19 (polaris24cln) — real, reproducible

After the rebuild, per-job correlation between the two simulators jumped
from ~0.89 to ~0.98–0.99, but a smaller, genuine divergence remained. In
`theta21cln`, job 75 (256 procs, 3600s) starts at `sim_time` 1607451661 in
C++ but not until 1607462461 in Python — a ~3-hour delay, exactly the moment
job 66 (1792 procs) itself finishes.

### Root cause: event granularity, not the backfill math

Both simulators implement the same EASY backfill idea (reserve the head
job's shadow start time, then admit any later-queued job that either
finishes before that shadow time or leaves enough spare capacity at the
shadow instant). The difference is *when* each simulator re-evaluates the
queue:

- **C++** (`src/adapters/simulation/driver.cpp`,
  `src/adapters/simulation/simulator.cpp`) has an explicit `SchedulingCycle`
  event type. When several jobs finish at the exact same `sim_time`, each
  `End` event is processed (releasing that job's procs) and each pushes its
  own `SchedulingCycle` event — but because all the `End` events at that
  timestamp are drained from the event queue before their `SchedulingCycle`
  events run, the scheduling decision that follows sees the *combined*
  freed capacity.

- **Python** (`cqsim-python/src/CqSim/Cqsim_sim.py`) has no separate
  scheduling-cycle event. `event_job()` calls `start_scan()` /
  `backfill()` synchronously, immediately after *each individual* `Submit`
  or `Finish` — one scheduling decision per event, not per timestamp.

### Concrete trace (theta21cln, `sim_time` 1607451661 / internal `t=638332`)

Jobs 61, 67, and 68 (128 procs each) all finish at this instant, freeing 384
procs. Job 66 (1792 procs) is the head/reservation job; its shadow start is
`sim_time` 1607458861 (confirmed identical in both simulators). Candidates
74 (128 procs, 3600s) and 75 (256 procs, 3600s) both finish well before that
shadow time, so admitting both is safe and doesn't delay job 66.

- **C++**: all three `End` events are applied first (real free capacity =
  384), then one `SchedulingCycle` admits 74 *and* 75 together
  (128 + 256 = 384, exact fit) — confirmed via temporary tracing added to
  `schedule_with_backfill_reservation` (reverted; see `changes.md`).
- **Python**: three separate `backfill()` calls fire, one per `End` event,
  each seeing only that event's own 128-proc increment:
  1. `avail=128` → admits job 74 (128, exact fit for that increment).
  2. `avail=128` (the *next* increment) → 75 doesn't fit (needs 256), a
     later-queued 128-proc job (86) is admitted instead.
  3. `avail=128` again → another later-queued 128-proc job (98) is admitted.

  Job 75 never sees a single moment with ≥256 free procs, because the 384
  arrives in three separate 128-proc installments and each is consumed
  before the next arrives. It ends up delayed until job 66 itself
  completes and the whole 1792+ procs free up at once.

This was confirmed empirically with temporary instrumentation in
`Cqsim_sim.backfill()`, `Node_struc_SWF.node_allocate()`, and
`Node_struc_SWF.pre_reset()` (all reverted): real `self.avail` at the start
of each of the three `backfill()` calls was 128, 128, 128 — never 384 — even
though all three releases had already happened by the time the *first* of
the three calls ran.

### A tempting but wrong local fix

`Node_struc_SWF.pre_reset()` (and the base `Node_struc.pre_reset()`) contain:

```python
if (self.predict_node[j]['time']!=self.job_list[i]['end'] or i == 0):
```

The `or i == 0` looks redundant/buggy — dropping it initially appeared to
fix job 75's case (predicted timeline no longer showed a spurious duplicate
breakpoint at the "now" timestamp). It is not a bug. It exists to keep
`predict_node[0]` (the "right now" snapshot) equal to the *real* currently
free capacity, rather than being pre-inflated by `job_list` entries whose
`end` time happens to coincide with "now" but haven't actually been
processed yet by this specific event (they're still separate, pending `End`
events in the queue). Removing it makes the predictive reservation layer
assume capacity is free before it really is, and the real allocation layer
(`node_allocate`) then silently fails for the over-admitted job — silently,
because `Cqsim_sim.start()` never checks `node_allocate()`'s return value.
That job is still marked "running" and gets an `End` event scheduled for it
anyway; when that `End` event eventually fires, `node_release()` can't find
it in `job_list` and raises `IndexError: pop index out of range`. This
turned two `1000`-job runs into `57`/`28`-job crashes.

### The correct fix (not implemented)

The real fix is to batch same-timestamp `End`/`Submit` events in
`Cqsim_sim.py`'s event loop (`scan_event()` / `event_job()`) before running
a single combined `start_scan()`/`backfill()` pass — mirroring the C++
`SchedulingCycle` model, where all events sharing a timestamp are drained
before the scheduling decision runs. This is a change to the core event
loop, not to `Node_struc.py`, and is materially larger in scope than the
`Node_struc.py` bug that was originally suspected. It was not implemented —
see `changes.md`.

## Summary

| # | Symptom | Root cause | Status |
|---|---|---|---|
| 1 | job 49 (theta21) admitted early in C++, delayed in Python | stale `results/cqsimpy-test1` build artifacts | resolved by rebuilding; no source change needed |
| 2 | job 75 (theta21) / job 19 (polaris24) delayed in Python relative to C++ | Python's event loop makes one backfill decision per individual `Submit`/`Finish` event instead of batching same-timestamp events like C++'s `SchedulingCycle` does | root-caused; fix not implemented (see `changes.md`) |
