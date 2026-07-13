#pragma once
#include "sort_policy.h"

// FCSJ: score = -(wait/walltime) — higher wait-to-walltime ratio = higher priority
class FcsjPolicy : public SortPolicy {
public:
    explicit FcsjPolicy(int window_size = INT_MAX) : SortPolicy(window_size) {}
    std::string name() const override { return "fcsj"; }
protected:
    bool compare(const Job* a, const Job* b, const State& s) const override {
        double sa = score(a, s), sb = score(b, s);
        if (sa != sb) return sa < sb;
        return a->submit_time < b->submit_time;
    }
private:
    static double score(const Job* j, const State& s) {
        double wt = j->walltime > 0 ? static_cast<double>(j->walltime) : 1.0;
        double wait = static_cast<double>(s.current_time - j->submit_time);
        return -(wait / wt);
    }
};
