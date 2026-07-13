#pragma once
#include <cmath>
#include "sort_policy.h"

// F4: score = walltime * sqrt(procs) + 5.30e5 * log10(submit_time)
class F4Policy : public SortPolicy {
public:
    explicit F4Policy(int window_size = INT_MAX) : SortPolicy(window_size) {}
    std::string name() const override { return "f4"; }
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
        return wt * std::sqrt(static_cast<double>(j->requested.procs)) + 5.30e5 * std::log10(sub);
    }
};
