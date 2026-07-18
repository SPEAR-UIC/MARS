#pragma once

#include <algorithm>
#include <climits>
#include "../core/policy.h"
#include "../core/backfill.h"

// Base for policies that sort a window of pending jobs then dispatch greedily.
// Subclasses implement compare() to define priority ordering.
class SortPolicy : public SchedulerPolicy {
public:
    explicit SortPolicy(int window_size = INT_MAX) : window_size_(window_size) {}

    std::vector<Decision> schedule(const State& s) override {
        std::vector<const Job*> jobs = s.pending;

        int w = std::min(window_size_, static_cast<int>(jobs.size()));
        std::stable_sort(jobs.begin(), jobs.begin() + w,
                         [this, &s](const Job* a, const Job* b) {
                             return compare(a, b, s);
                         });

        last_head_job_id_ = jobs.empty() ? -1 : jobs[0]->id;

        return schedule_with_backfill_reservation(jobs, s);
    }

    int last_cycle_root_branching() const override { return 1; }
    std::string last_cycle_possible_policies() const override {
        return last_head_job_id_ >= 0
            ? "|" + std::to_string(last_head_job_id_) + "|"
            : "";
    }

    // Return true if a has higher priority than b.
    // Public so MCTS can reuse comparators without going through schedule().
    virtual bool compare(const Job* a, const Job* b, const State& s) const = 0;

protected:
    int window_size_;
    int last_head_job_id_ = -1;
};
