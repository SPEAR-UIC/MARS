#pragma once
#include "sort_policy.h"

// Longest Job First — longest walltime gets highest priority
class LjfPolicy : public SortPolicy {
public:
    explicit LjfPolicy(int window_size = INT_MAX) : SortPolicy(window_size) {}
    std::string name() const override { return "ljf"; }
protected:
    bool compare(const Job* a, const Job* b, const State&) const override {
        if (a->walltime != b->walltime) return a->walltime > b->walltime;
        return a->submit_time < b->submit_time;
    }
};
