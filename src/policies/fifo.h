#pragma once

#include <string>
#include <vector>
#include "../core/policy.h"
#include "../core/backfill.h"

class FifoPolicy : public SchedulerPolicy {
public:
    std::vector<Decision> schedule(const State& s) override {
        return schedule_with_backfill_reservation(s.pending, s);
    }

    std::string name() const override { return "fifo"; }
};
