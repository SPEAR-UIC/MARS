# MCTS Integration With the Simulator

This note explains how MCTS fits into the simulator as it exists now.

The important point is that the MCTS path is not "reorder the queue and let a later FIFO pass decide what actually starts." When MCTS is active, it uses the simulator callback path to choose the exact jobs to start at the current `SchedulingCycle`.

## Event Ordering

The simulator event types and priorities are unchanged.

At the same timestamp, events are processed in this order:

1. `End`
2. maintenance end
3. maintenance announce / maintenance start
4. `Submit` / `Resubmit`
5. `SchedulingCycle`
6. `Run` / `Backfill`

This is implemented in [src/adapters/simulation/event.h](../src/adapters/simulation/event.h).

## Callback Path vs Default Path

There are two scheduling modes in the simulator.

### Default simulator path

If no scheduling callback is installed, the simulator uses its built-in scheduler:

- strict FIFO dispatch
- optional EASY backfilling

### Policy-driven path

If a callback is installed, the callback decides what starts during the scheduling cycle.

That is the path used by MCTS. The driver:

1. builds a `State`
2. calls `policy_->schedule(state)`
3. gets back `DecisionType::Start` actions
4. applies them directly to the simulator
5. removes started jobs from the queue

So once MCTS is active, the simulator is not doing a second FIFO scheduling pass underneath it.

## What the MCTS Return Value Means

The code still uses names like `window` and `ordered_window`, but in the MCTS path the returned vector is really the ordered set of jobs to start in the current cycle.

That means the current action is best understood as:

- "which jobs should start now, and in what order?"

not:

- "how should I permanently reorder the queue head?"

## How the Queue Window Still Matters

The queue-head window still matters, but it matters during candidate construction.

MCTS still begins from the first `w` jobs in the queue. Depending on the branching mode, it then:

- permutes that prefix
- sorts that prefix with a heuristic
- or applies heuristic-windowed variants to that prefix

That reordered prefix is then combined with the rest of the queue to form an ordered scan. MCTS walks that scan and builds the dispatch sequence for the current cycle.

So the window still controls:

- which jobs are considered early
- which jobs consume currently free processors first
- which actions are available to the search

What changed from the older mental model is that the prefix is not committed back as a persistent queue order.

## Backfilling Under MCTS

Classic simulator EASY backfilling only runs in the default simulator path.

Under MCTS:

- there is no second EASY pass after the policy returns
- MCTS itself decides which jobs to start in the cycle
- the chosen jobs are started directly through the callback path

So backfilling behavior under MCTS is whatever the current dispatch-construction logic allows from the ordered scan, not a separate simulator feature layered afterward.

## Tree and Rollout Consistency

The same dispatch semantics are used in three places:

- child generation
- rollout simulation
- real policy dispatch

So when the search explores "start these jobs now," that is also what the rollout and real execution path do.

## Summary

MCTS fits into the simulator by reusing the existing `SchedulingCycle` callback mechanism.

- no new event types are added
- the simulator event loop is unchanged
- MCTS still uses queue-head windows to shape candidate actions
- the returned action is the current cycle's dispatch sequence, not a persistent queue reorder
