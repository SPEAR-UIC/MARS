#pragma once

#include <cstddef>
#include <vector>

#include "decision.h"
#include "job.h"
#include "node.h"
#include "state.h"
#include "../adapters/simulation/simulator.h"

// Shared EASY-backfill dispatch used by every classical SchedulerPolicy
// (FifoPolicy, SortPolicy and its subclasses).
//
// `ordered_jobs` must already be in the policy's desired priority order.
// Jobs are started in that order for as long as each one currently fits.
// The first job that doesn't fit becomes the single reservation target:
// its earliest possible start time is computed from the jobs actually
// running (via Simulator::earliest_start_time, which reasons in walltime),
// and no later job is allowed to start early if doing so would push that
// reservation back — mirroring Simulator::default_scheduling_cycle().
//
// The whole batch of decisions built here is only applied to the Simulator
// (running_jobs_/end_times_) by the caller *after* this function returns —
// so Simulator::earliest_start_time / calculate_free_procs_at, queried
// mid-function, can't see any admission this same call has already made
// (neither the primary-loop starts nor earlier backfills in this pass).
// `committed_this_cycle` tracks that provisional total ourselves and feeds
// it in as `reserved_procs` so the shadow-time and per-candidate safety
// checks stay accurate as the batch builds up, instead of every candidate
// being checked independently against the same stale pre-cycle snapshot
// (which let multiple backfills collectively overrun the reservation they
// were each individually "safe" against).
inline std::vector<Decision> schedule_with_backfill_reservation(
        const std::vector<const Job*>& ordered_jobs,
        const State& s)
{
    std::vector<NodeInfo> nodes = s.nodes;
    std::vector<Decision> decisions;
    int committed_this_cycle = 0;

    std::size_t i = 0;
    for (; i < ordered_jobs.size(); ++i) {
        const Job* job = ordered_jobs[i];
        bool scheduled = false;
        for (NodeInfo& node : nodes) {
            if (node.can_fit(job->requested)) {
                node.allocate(job->requested);
                decisions.push_back({DecisionType::Start, job->id, {node.id}, false});
                scheduled = true;
                committed_this_cycle += job->requested.procs;
                break;
            }
        }
        if (!scheduled) break;
    }

    // Everything fit, or there's no simulator context to reserve against
    // (e.g. a minimal State built for comparator-only use).
    if (i >= ordered_jobs.size() || !s.sim) return decisions;

    const Job* head_job   = ordered_jobs[i];
    int        head_procs = head_job->requested.procs;
    long       shadow_time = s.sim->earliest_start_time(head_procs, committed_this_cycle);

    for (std::size_t k = i + 1; k < ordered_jobs.size(); ++k) {
        const Job* job = ordered_jobs[k];

        NodeInfo* target = nullptr;
        for (NodeInfo& node : nodes) {
            if (node.can_fit(job->requested)) { target = &node; break; }
        }
        if (!target) continue;

        long duration = s.sim->job_duration_for(job->id);
        bool safe = (s.current_time + duration <= shadow_time) ||
                    (s.sim->calculate_free_procs_at(shadow_time, committed_this_cycle)
                         - job->requested.procs >= head_procs);
        if (!safe) continue;

        target->allocate(job->requested);
        decisions.push_back({DecisionType::Start, job->id, {target->id}, true});
        committed_this_cycle += job->requested.procs;
    }

    return decisions;
}
