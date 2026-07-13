#pragma once

#include <vector>
#include <memory>
#include <optional>
#include <string>
#include <random>
#include <functional>
#include <unordered_set>
#include <climits>
#include <chrono>

#include "../adapters/simulation/simulator.h"
#include "../core/policy.h"

// ── Enums ─────────────────────────────────────────────────────────────────────

enum class BranchingMode {
    Permutation            = 0,  // DFS over unique schedulable subsets
    Heuristic              = 1,  // one branch per heuristic policy
    HeuristicWindowed      = 2,  // one branch per (heuristic, window_size) spec
    ComprehensiveHeuristic = 3   // 16 heuristics x standard window sweep + FCFS
};


enum class HeuristicPolicy {
    FCFS   = 0,
    SJF    = 1,
    WFP    = 2,
    F1     = 3,
    UNICEP = 4,
    LCFS   = 5,
    LJF    = 6,
    SRF    = 7,
    LRF    = 8,
    SCF    = 9,
    LCF    = 10,
    F2     = 11,
    F3     = 12,
    F4     = 13,
    FAT    = 14,
    WFP1   = 15,
    FCSJ   = 16
};

static constexpr int NUM_ALL_HEURISTICS = 17;

enum class RewardType {
    InvCumWait    = 0,  // 1 / (1 + avg_wait_hours), cumulative from search root
    InvCumBSLD    = 1,  // 1 / avg_bsld, cumulative from search root
    InstantUtil   = 2,  // used_procs / total_procs at the evaluated state
    CumUtil       = 3,  // useful_proc_seconds / (total_procs * elapsed_seconds)
    InvCumP99Wait    = 4,  // (1/(1+avg_wait_h))^avg_lambda * (1/(1+p99_wait_h))^p99_lambda
    NegCumWait       = 5,  // -avg_wait_hours (raw hours, unbounded negative)
    NegP99Wait       = 6,  // -p99_wait_hours (raw P99 hours, unbounded negative)
    Constant         = 7,  // always 1.0; use reward_scale to set the effective value
    InvCumP99MaxWait = 8,  // (1/(1+avg_wait_h))^avg_lambda * (1/(1+p99_wait_h))^p99_lambda * (1/(1+max_wait_h))^max_lambda
    ExpWait          = 9   // exp(-exp_scale*(avg_lambda*avg_wait_h + p99_lambda*p99_wait_h + max_lambda*max_wait_h))
};

// ── Helpers ───────────────────────────────────────────────────────────────────

inline HeuristicPolicy parse_heuristic_policy(const std::string& s) {
    if (s == "fcfs")   return HeuristicPolicy::FCFS;
    if (s == "sjf")    return HeuristicPolicy::SJF;
    if (s == "wfp" || s == "wfp3") return HeuristicPolicy::WFP;
    if (s == "f1")     return HeuristicPolicy::F1;
    if (s == "unicep" || s == "unicef") return HeuristicPolicy::UNICEP;
    if (s == "lcfs")   return HeuristicPolicy::LCFS;
    if (s == "ljf")    return HeuristicPolicy::LJF;
    if (s == "srf")    return HeuristicPolicy::SRF;
    if (s == "lrf")    return HeuristicPolicy::LRF;
    if (s == "scf")    return HeuristicPolicy::SCF;
    if (s == "lcf")    return HeuristicPolicy::LCF;
    if (s == "f2")     return HeuristicPolicy::F2;
    if (s == "f3")     return HeuristicPolicy::F3;
    if (s == "f4")     return HeuristicPolicy::F4;
    if (s == "fat")    return HeuristicPolicy::FAT;
    if (s == "wfp1")   return HeuristicPolicy::WFP1;
    if (s == "fcsj")   return HeuristicPolicy::FCSJ;
    return HeuristicPolicy::FCFS;
}

struct HeuristicWindowSpec {
    HeuristicPolicy policy;
    int             window_size;
};

inline std::string heuristic_policy_tag(HeuristicPolicy hp) {
    switch (hp) {
        case HeuristicPolicy::FCFS:   return "FCFS";
        case HeuristicPolicy::SJF:    return "SJF";
        case HeuristicPolicy::WFP:    return "WFP";
        case HeuristicPolicy::F1:     return "F1";
        case HeuristicPolicy::UNICEP: return "UNICEP";
        case HeuristicPolicy::LCFS:   return "LCFS";
        case HeuristicPolicy::LJF:    return "LJF";
        case HeuristicPolicy::SRF:    return "SRF";
        case HeuristicPolicy::LRF:    return "LRF";
        case HeuristicPolicy::SCF:    return "SCF";
        case HeuristicPolicy::LCF:    return "LCF";
        case HeuristicPolicy::F2:     return "F2";
        case HeuristicPolicy::F3:     return "F3";
        case HeuristicPolicy::F4:     return "F4";
        case HeuristicPolicy::FAT:    return "FAT";
        case HeuristicPolicy::WFP1:   return "WFP1";
        case HeuristicPolicy::FCSJ:   return "FCSJ";
    }
    return "FCFS";
}

inline std::string heuristic_window_tag(HeuristicPolicy hp, int window_size) {
    if (hp == HeuristicPolicy::FCFS) return "FCFS";
    return heuristic_policy_tag(hp) + "-w" + std::to_string(window_size);
}

inline std::vector<HeuristicPolicy> all_heuristic_policies() {
    return {
        HeuristicPolicy::FCFS,  HeuristicPolicy::SJF,  HeuristicPolicy::WFP,
        HeuristicPolicy::F1,    HeuristicPolicy::UNICEP,HeuristicPolicy::LCFS,
        HeuristicPolicy::LJF,   HeuristicPolicy::SRF,  HeuristicPolicy::LRF,
        HeuristicPolicy::SCF,   HeuristicPolicy::LCF,  HeuristicPolicy::F2,
        HeuristicPolicy::F3,    HeuristicPolicy::F4,   HeuristicPolicy::FAT,
        HeuristicPolicy::WFP1,  HeuristicPolicy::FCSJ
    };
}

inline std::vector<HeuristicPolicy> all_non_fcfs_heuristic_policies() {
    return {
        HeuristicPolicy::SJF,    HeuristicPolicy::WFP,  HeuristicPolicy::F1,
        HeuristicPolicy::UNICEP, HeuristicPolicy::LCFS, HeuristicPolicy::LJF,
        HeuristicPolicy::SRF,    HeuristicPolicy::LRF,  HeuristicPolicy::SCF,
        HeuristicPolicy::LCF,    HeuristicPolicy::F2,   HeuristicPolicy::F3,
        HeuristicPolicy::F4,     HeuristicPolicy::FAT,  HeuristicPolicy::WFP1,
        HeuristicPolicy::FCSJ
    };
}

inline std::vector<int> standard_window_sweep_sizes() {
    return {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
}

inline std::vector<HeuristicWindowSpec> comprehensive_heuristic_window_specs() {
    std::vector<HeuristicWindowSpec> specs;
    specs.reserve(161);

    for (auto hp : all_non_fcfs_heuristic_policies()) {
        for (int w : standard_window_sweep_sizes())
            specs.push_back({hp, w});
    }

    // FCFS is included exactly once because its window size does not affect
    // the ordering.
    specs.push_back({HeuristicPolicy::FCFS, 1});
    return specs;
}

// Polaris-tuned subset — top 7 by composite wait metric at w=64
inline std::vector<HeuristicPolicy> polaris_heuristic_policies() {
    return {
        HeuristicPolicy::WFP,   // strong L-job wait; default rollout policy
        HeuristicPolicy::F1,    // strong S/M-job wait
        HeuristicPolicy::SCF,
        HeuristicPolicy::F3,
        HeuristicPolicy::SJF,
        HeuristicPolicy::FCSJ,
        HeuristicPolicy::FCFS,  // FIFO baseline for ordering diversity
    };
}

// ── MctsConfig ────────────────────────────────────────────────────────────────
// Bundles all MCTS parameters into one struct (cleaner than a 15-arg constructor).

struct MctsConfig {
    long         time_limit_ms     = 100;
    int          max_depth         = 1000;
    int          window_size       = 5;
    RewardType   reward_type       = RewardType::InvCumWait;
    double       exploration       = 10000.0;
    double       discount          = 1.0;
    std::string  log_file;
    unsigned int seed              = 0;
    BranchingMode branching        = BranchingMode::Heuristic;
    std::vector<HeuristicPolicy>     heuristics        = polaris_heuristic_policies();
    std::vector<HeuristicWindowSpec> heuristic_windows = {};
    HeuristicPolicy rollout_policy = HeuristicPolicy::WFP;
    bool         dynamic_exploration = false;
    int          num_cores         = 1;
    double       avg_lambda        = 1.0;  // exponent on the avg-wait term for InvCumP99Wait / InvCumP99MaxWait / ExpWait
    double       p99_lambda        = 1.0;  // exponent on the P99 term for InvCumP99Wait / InvCumP99MaxWait / ExpWait
    double       max_lambda        = 1.0;  // exponent on the max-wait term for InvCumP99MaxWait / ExpWait
    double       exp_scale         = 1.0;  // overall scale on the exponent for ExpWait
    double       reward_scale      = 1.0;  // multiplier applied to every reward value
};

// ── MctsNode ──────────────────────────────────────────────────────────────────

struct MctsNode {
    std::optional<Simulator> state;       // released after full expansion to save memory
    int                      action_job_id;
    std::vector<int>         ordered_window;
    std::string              source_policy_label = "";
    MctsNode*                parent;
    std::vector<std::unique_ptr<MctsNode>> children;

    struct PotentialChild {
        int              action_job_id;
        std::vector<int> window;          // reordered queue-head window to apply this cycle
        std::vector<int> outcome;         // job IDs that left the queue after this action
        std::string      source_policy = "";   // heuristic/window spec that produced this ordering
        std::vector<std::string> source_policies; // all policies that map to this branch
    };
    std::vector<PotentialChild> untried_children;

    using OutcomeGenerator = std::function<std::vector<PotentialChild>(const Simulator&, int)>;

    double visits     = 0;
    double score      = 0;
    double score_sq   = 0;
    double immediate_reward = 0.0;
    std::vector<int> outcome;
    int    depth      = 0;

    // Transition tracking from the search root only; past trace history is not
    // folded into these accumulators.
    double cum_wait_s      = 0.0;
    double cum_bsld        = 0.0;
    double cum_used_proc_s = 0.0;
    double cum_elapsed_s   = 0.0;
    int    cum_count       = 0;
    std::vector<double> cum_wait_sorted;  // sorted wait times, for P99 computation

    // Metrics accrued while fast-forwarding deterministic states after the
    // chosen action has already been applied.
    std::vector<int> transition_outcome;
    double transition_wait_s      = 0.0;
    double transition_bsld        = 0.0;
    double transition_used_proc_s = 0.0;
    double transition_elapsed_s   = 0.0;
    int    transition_count       = 0;

    bool solved          = false;
    int  solved_children = 0;
    bool terminal_       = false;

    MctsNode(Simulator s, int action, MctsNode* p,
             int window_size, int depth, int max_depth,
             std::vector<int> ordered_window = {},
             std::mt19937* rng = nullptr,
             OutcomeGenerator gen = nullptr,
             std::string source_policy_label = "");

    bool      is_terminal()      const;
    bool      is_fully_expanded() const;
    MctsNode* best_child(double exploration);
    void      release_state();
};

// ── MctsChildResult ───────────────────────────────────────────────────────────

struct MctsChildResult {
    std::vector<int> ordered_window;
    double           visits;
    double           score;
    std::vector<int> outcome;
    std::string      source_policy;
};

// ── MCTS convergence tracing ─────────────────────────────────────────────────

struct MctsConvergenceSample {
    int              iteration          = 0;
    int              root_branching     = 0;
    int              expanded_children  = 0;
    std::vector<int> selected_window;
    std::vector<int> selected_outcome;
    std::string      selected_policy;
    double           selected_visits    = 0.0;
    double           total_root_visits  = 0.0;
    double           selected_visit_pct = 0.0;
    double           selected_avg_reward= 0.0;
    int              best_visit_ties    = 0;
    double           visit_margin       = 0.0;
    bool             changed            = false;
};

struct MctsConvergenceTrace {
    int              max_iterations          = 0;
    int              final_iteration         = 0;
    int              root_branching          = 0;
    int              stabilization_iteration = -1;
    std::vector<int> final_selected_window;
    std::vector<int> final_selected_outcome;
    std::string      final_selected_policy;
    std::vector<MctsConvergenceSample> samples;
};

struct MctsRootAnalysis {
    int         branching       = 0;
    int         queue_len       = 0;
    int         available_procs = 0;
    int         running_jobs    = 0;
    long        sim_time        = 0;
    std::string possible_job_sets;
};

// ── Outcome generators (defined in mcts.cpp) ──────────────────────────────────

std::vector<MctsNode::PotentialChild>
generate_permutation_outcomes(const Simulator& state, int window_size);

std::vector<MctsNode::PotentialChild>
generate_heuristic_outcomes(const Simulator& state, int window_size,
                             const std::vector<HeuristicPolicy>& policies);

std::vector<MctsNode::PotentialChild>
generate_heuristic_windowed_outcomes(const Simulator& state, int window_size,
                                     const std::vector<HeuristicWindowSpec>& specs);

// ── Mcts ──────────────────────────────────────────────────────────────────────

class Mcts {
public:
    explicit Mcts(MctsConfig config);

    std::vector<int> search(const Simulator& root_state);
    MctsRootAnalysis inspect_root(const Simulator& root_state);
    MctsConvergenceTrace analyze_convergence(const Simulator& root_state,
                                             int max_iterations,
                                             int sample_every = 1);

    int    last_iterations()       const { return last_iterations_; }
    int    last_unique_nodes()     const { return last_unique_nodes_; }
    int    last_max_depth()        const { return last_max_depth_; }
    int    last_root_branching()   const { return last_root_branching_; }
    double last_chosen_visit_pct() const { return last_chosen_visit_pct_; }
    double last_chosen_avg_reward()const { return last_chosen_avg_reward_; }
    const std::string& last_selected_policy() const { return last_selected_policy_; }
    const std::string& last_root_possible_policies() const {
        return last_root_possible_policies_;
    }

    static std::vector<int> parallel_search(
        const Simulator& root_state, const MctsConfig& config,
        int*    total_iterations   = nullptr,
        int*    unique_nodes_out   = nullptr,
        int*    max_depth_out      = nullptr,
        int*    root_branching_out = nullptr,
        double* chosen_visit_pct_out   = nullptr,
        double* chosen_avg_reward_out  = nullptr,
        std::string* chosen_policy_out = nullptr,
        std::string* root_possible_policies_out = nullptr);

    // ── Members listed in declaration order to avoid -Wreorder ──────────────
    MctsConfig config_;                               // must come first (rng_ reads it)
    MctsNode::OutcomeGenerator outcome_gen_;

    int    last_iterations_       = 0;
    int    last_unique_nodes_     = 0;
    int    last_max_depth_        = 0;
    int    last_root_branching_   = 0;
    double last_chosen_visit_pct_ = 0.0;
    double last_chosen_avg_reward_= 0.0;
    std::string last_selected_policy_;
    std::string last_root_possible_policies_;
    // Exposed for parallel workers (each worker holds its own Mcts instance)
    std::mt19937                             rng_;
    double global_min_score_;
    double global_max_score_;
    std::chrono::steady_clock::time_point    deadline_;
    MctsNode* tree_policy(MctsNode* node);
    double    rollout(Simulator state, int depth,
                      double cum_wait_s = 0.0, int cum_count = 0,
                      double cum_bsld = 0.0, double cum_used_proc_s = 0.0,
                      double cum_elapsed_s = 0.0,
                      std::vector<double> cum_wait_sorted = {});
    void      backpropagate(MctsNode* node, double reward);

    // Sort a window of job IDs using the named heuristic's SortPolicy comparator.
    void sort_window(HeuristicPolicy hp,
                     std::vector<int>& window,
                     const Simulator& sim) const;

private:
    MctsNode* expand(MctsNode* node);
    void      mark_solved(MctsNode* node);
};

// ── MctsPolicy ────────────────────────────────────────────────────────────────
// Wraps Mcts as a SchedulerPolicy so it can be dropped into Driver like any
// other policy.  Requires State::sim to be non-null (simulation context).

class MctsPolicy : public SchedulerPolicy {
public:
    explicit MctsPolicy(MctsConfig config);

    std::vector<Decision> schedule(const State& s) override;
    std::string           name()                   const override { return "mcts"; }
    std::string           last_cycle_policy()      const override { return last_cycle_policy_; }
    int                   last_cycle_root_branching() const override {
        return last_cycle_root_branching_;
    }
    std::string           last_cycle_possible_policies() const override {
        return last_cycle_possible_policies_;
    }
    int                   last_cycle_iterations() const override {
        return last_cycle_iterations_;
    }

private:
    MctsConfig config_;
    Mcts       mcts_;
    std::string last_cycle_policy_;
    int         last_cycle_root_branching_ = 0;
    std::string last_cycle_possible_policies_;
    int         last_cycle_iterations_ = 0;

    // Apply FIFO + optional EASY backfill after reordering the queue by the
    // MCTS-determined window.
    std::vector<Decision> dispatch(const std::vector<int>& window,
                                   const State& s) const;
};

// ── RandomHeuristicPolicy ─────────────────────────────────────────────────────
// At each scheduling cycle, enumerates all unique outcomes reachable from the
// comprehensive heuristic pool (17 heuristics × standard window sweep) and
// picks one uniformly at random from those that start at least one job.
// Useful as a baseline: shows how much of MCTS quality comes from
// discrimination vs. the quality of the heuristic pool itself.

class RandomHeuristicPolicy : public SchedulerPolicy {
public:
    RandomHeuristicPolicy(int window_size, unsigned seed = 0);

    std::vector<Decision> schedule(const State& s) override;
    std::string           name()              const override { return "random_heuristic"; }
    std::string           last_cycle_policy() const override { return last_cycle_policy_; }

private:
    int          window_size_;
    std::mt19937 rng_;
    Mcts         mcts_;        // holds the comprehensive heuristic outcome_gen_
    std::string  last_cycle_policy_;

    std::vector<Decision> dispatch(const std::vector<int>& window,
                                   const State& s) const;
};
