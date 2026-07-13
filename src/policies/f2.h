#pragma once
#include <cmath>
#include "sort_policy.h"

// F2: score = sqrt(walltime) * procs + 2.56e4 * log10(submit_time)
class F2Policy : public SortPolicy {
public:
    explicit F2Policy(int window_size = INT_MAX) : SortPolicy(window_size) {}
    std::string name() const override { return "f2"; }
protected:
    bool compare(const Job* a, const Job* b, const State&) const override {
        double sa = score(a), sb = score(b);
        if (sa != sb) return sa < sb;
        return a->submit_time < b->submit_time;
    }
private:
    static double score(const Job* j) {
        double wt  = j->walltime > 0 ? static_cast<double>(j->walltime) : 1.0;
        double sub = j->submit_time > 0 ? static_cast<double>(j->submit_time) : 1.0;
        return std::sqrt(wt) * j->requested.procs + 2.56e4 * std::log10(sub);
    }
};
