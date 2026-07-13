#pragma once
#include "sort_policy.h"

// Largest Core Hours First — walltime * procs descending
class LcfPolicy : public SortPolicy {
public:
    explicit LcfPolicy(int window_size = INT_MAX) : SortPolicy(window_size) {}
    std::string name() const override { return "lcf"; }
protected:
    bool compare(const Job* a, const Job* b, const State&) const override {
        double ca = static_cast<double>(a->walltime) * a->requested.procs;
        double cb = static_cast<double>(b->walltime) * b->requested.procs;
        if (ca != cb) return ca > cb;
        return a->submit_time < b->submit_time;
    }
};
