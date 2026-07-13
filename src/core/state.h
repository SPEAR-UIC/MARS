#pragma once

#include <vector>
#include <unordered_map>
#include "job.h"
#include "node.h"

class Simulator;  // forward declaration — only MctsPolicy uses this field

struct State {
    std::vector<NodeInfo>                     nodes;
    std::vector<const Job*>                   pending;
    std::vector<const Job*>                   running;
    std::unordered_map<int, std::vector<int>> job_to_nodes;
    long                                      current_time = 0;
    const Simulator*                          sim = nullptr;  // set in simulation context; null elsewhere
};
