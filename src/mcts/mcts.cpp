#include "mcts.h"

#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <fstream>

// Policy headers — MCTS reuses existing SortPolicy comparators
#include "../policies/sort_policy.h"
#include "../policies/fifo.h"
#include "../policies/lcfs.h"
#include "../policies/sjf.h"
#include "../policies/ljf.h"
#include "../policies/srf.h"
#include "../policies/lrf.h"
#include "../policies/scf.h"
#include "../policies/lcf.h"
#include "../policies/wfp.h"
#include "../policies/wfp1.h"
#include "../policies/fcsj.h"
#include "../policies/fat.h"
#include "../policies/unicep.h"
#include "../policies/f1.h"
#include "../policies/f2.h"
#include "../policies/f3.h"
#include "../policies/f4.h"

// ── Mcts::sort_window ─────────────────────────────────────────────────────────
// Sorts a vector of job IDs in-place using the named heuristic's comparator.
// Builds a minimal State from the simulator snapshot; policies that need
// total_procs (FAT, UNICEP) get a full node copy, others just current_time.

void Mcts::sort_window(HeuristicPolicy hp, std::vector<int>& window,
                        const Simulator& sim) const {
    if (hp == HeuristicPolicy::FCFS || window.size() <= 1) return;

    State s;
    s.current_time = sim.current_time();
    // FAT and UNICEP need total_procs via s.nodes
    if (hp == HeuristicPolicy::FAT || hp == HeuristicPolicy::UNICEP) {
        s.nodes = sim.nodes();
    }

    auto do_sort = [&](const SortPolicy& p) {
        std::stable_sort(window.begin(), window.end(),
            [&](int a, int b) {
                return p.compare(&sim.get_job(a), &sim.get_job(b), s);
            });
    };

    switch (hp) {
        case HeuristicPolicy::SJF:    { SjfPolicy   p; do_sort(p); break; }
        case HeuristicPolicy::LJF:    { LjfPolicy   p; do_sort(p); break; }
        case HeuristicPolicy::LCFS:   { LcfsPolicy  p; do_sort(p); break; }
        case HeuristicPolicy::SRF:    { SrfPolicy   p; do_sort(p); break; }
        case HeuristicPolicy::LRF:    { LrfPolicy   p; do_sort(p); break; }
        case HeuristicPolicy::SCF:    { ScfPolicy   p; do_sort(p); break; }
        case HeuristicPolicy::LCF:    { LcfPolicy   p; do_sort(p); break; }
        case HeuristicPolicy::WFP:    { WfpPolicy   p; do_sort(p); break; }
        case HeuristicPolicy::WFP1:   { Wfp1Policy  p; do_sort(p); break; }
        case HeuristicPolicy::FCSJ:   { FcsjPolicy  p; do_sort(p); break; }
        case HeuristicPolicy::FAT:    { FatPolicy   p; do_sort(p); break; }
        case HeuristicPolicy::UNICEP: { UnicepPolicy p; do_sort(p); break; }
        case HeuristicPolicy::F1:     { F1Policy    p; do_sort(p); break; }
        case HeuristicPolicy::F2:     { F2Policy    p; do_sort(p); break; }
        case HeuristicPolicy::F3:     { F3Policy    p; do_sort(p); break; }
        case HeuristicPolicy::F4:     { F4Policy    p; do_sort(p); break; }
        default: break;
    }
}

// ── Outcome generation helpers ────────────────────────────────────────────────

static std::vector<int> get_outcome(const Simulator& before, const Simulator& after) {
    const auto& qb = before.get_job_queue();
    const auto& qa = after.get_job_queue();
    if (qb.size() == qa.size()) return {};

    std::unordered_set<int> after_set(qa.begin(), qa.end());
    std::vector<int> out;
    for (int id : qb)
        if (!after_set.count(id)) out.push_back(id);
    return out;
}

static std::deque<int>
build_reordered_queue(const std::deque<int>& queue, const std::vector<int>& window) {
    std::deque<int> ordered;
    std::unordered_set<int> seen;

    auto add = [&](int job_id) {
        if (seen.insert(job_id).second) ordered.push_back(job_id);
    };

    for (int job_id : window) add(job_id);
    for (int job_id : queue) add(job_id);

    return ordered;
}

static Simulator
apply_window_ordering(const Simulator& parent,
                      const std::vector<int>& window,
                      std::vector<int>* started_jobs_out = nullptr) {
    Simulator sim = parent.get_copy(false);
    auto next = sim.peek();
    if (!next) return sim;

    if (next->type != EventType::SchedulingCycle) {
        sim.step();
        return sim;
    }

    sim.sort_job_queue(build_reordered_queue(parent.get_job_queue(), window));
    sim.step();

    if (started_jobs_out) *started_jobs_out = get_outcome(parent, sim);
    return sim;
}

// Simulate a window ordering and record the actual jobs started after the
// simulator applies FIFO + optional EASY backfill to that reordered queue.
// Deduplicates by the sorted set of started job IDs.
static void try_add_window_outcome(const Simulator& state,
                                   const std::vector<int>& window,
                                   const std::string& source_policy,
                                   std::vector<MctsNode::PotentialChild>& outcomes,
                                   std::vector<std::vector<int>>& seen_outcomes) {
    std::vector<int> started_jobs;
    apply_window_ordering(state, window, &started_jobs);

    std::vector<int> outcome_sorted = started_jobs;
    std::sort(outcome_sorted.begin(), outcome_sorted.end());
    for (size_t i = 0; i < seen_outcomes.size(); ++i) {
        if (seen_outcomes[i] != outcome_sorted) continue;
        auto& policies = outcomes[i].source_policies;
        if (std::find(policies.begin(), policies.end(), source_policy) == policies.end())
            policies.push_back(source_policy);
        return;
    }

    seen_outcomes.push_back(std::move(outcome_sorted));
    int action_job_id = window.empty() ? -1 : window.front();
    MctsNode::PotentialChild child;
    child.action_job_id = action_job_id;
    child.window = window;
    child.outcome = std::move(started_jobs);
    child.source_policy = source_policy;
    child.source_policies.push_back(source_policy);
    outcomes.push_back(std::move(child));
}

// Regenerate a child simulator by applying a window ordering to the parent state.
static Simulator regenerate_child_sim(const Simulator& parent,
                                      const std::vector<int>& window,
                                      int /*window_size*/) {
    return apply_window_ordering(parent, window);
}

// ── Permutation outcome generation ───────────────────────────────────────────
// DFS over unique greedy scheduling sequences — avoids simulating all W! orderings.

std::vector<MctsNode::PotentialChild>
generate_permutation_outcomes(const Simulator& state, int window_size) {
    const auto& queue = state.get_job_queue();
    if (queue.empty()) return {};

    int w = std::min(static_cast<int>(queue.size()), window_size);
    std::vector<int> window(queue.begin(), queue.begin() + w);
    std::deque<int>  rest(queue.begin() + w, queue.end());
    int avail = state.available_procs();

    std::vector<int> procs(w);
    for (int i = 0; i < w; ++i)
        procs[i] = state.get_job(window[i]).requested.procs;

    std::vector<std::vector<int>> unique_sets;
    std::vector<bool> used(w, false);

    std::function<void(int, std::vector<int>&)> dfs =
        [&](int remaining, std::vector<int>& sched) {
            bool is_new = true;
            for (const auto& s : unique_sets)
                if (s == sched) { is_new = false; break; }
            if (is_new) unique_sets.push_back(sched);

            for (int i = 0; i < w; ++i) {
                if (used[i] || procs[i] > remaining) continue;
                used[i] = true;
                auto pos = std::lower_bound(sched.begin(), sched.end(), i);
                sched.insert(pos, i);
                dfs(remaining - procs[i], sched);
                sched.erase(std::lower_bound(sched.begin(), sched.end(), i));
                used[i] = false;
            }
        };

    std::vector<int> init;
    dfs(avail, init);

    std::vector<MctsNode::PotentialChild> outcomes;
    std::vector<std::vector<int>> seen;

    for (const auto& sched_indices : unique_sets) {
        std::unordered_set<int> sched_set(sched_indices.begin(), sched_indices.end());
        std::vector<int> ordered;
        for (int idx : sched_indices) ordered.push_back(window[idx]);
        for (int i = 0; i < w; ++i)
            if (!sched_set.count(i)) ordered.push_back(window[i]);
        for (int id : rest) ordered.push_back(id);
        try_add_window_outcome(state, ordered, "PERMUTATION", outcomes, seen);
    }
    return outcomes;
}

// ── Heuristic outcome generation ─────────────────────────────────────────────

// sort_window_by_heuristic_free: free-function wrapper used by generate_* functions.
// Takes an Mcts* only for its sort_window method — null = skip sort (FCFS).
static void sort_window_free(HeuristicPolicy hp, std::vector<int>& window,
                              const Simulator& sim, const Mcts* mcts) {
    if (mcts) mcts->sort_window(hp, window, sim);
    // if mcts==null, FCFS (no-op)
}

std::vector<MctsNode::PotentialChild>
generate_heuristic_outcomes(const Simulator& state, int window_size,
                             const std::vector<HeuristicPolicy>& policies) {
    // Note: this free function is used as an OutcomeGenerator lambda that captures
    // the Mcts* via the closure in Mcts constructor.
    // When called without an Mcts context (e.g. pre-check in parallel_search),
    // we fall back to a temporary Mcts created with default config.
    // In normal use the lambda always provides the Mcts*.
    const auto& queue = state.get_job_queue();
    if (queue.empty()) return {};

    int w = std::min(static_cast<int>(queue.size()), window_size);
    std::vector<int> window(queue.begin(), queue.begin() + w);
    std::vector<MctsNode::PotentialChild> outcomes;
    std::vector<std::vector<int>> seen;

    // This overload is used by the pre-check path which passes a temporary Mcts.
    // The Mcts lambda path passes a different closure — see Mcts constructor.
    Mcts tmp(MctsConfig{});  // temporary just for sort_window access
    for (auto hp : policies) {
        std::vector<int> sorted = window;
        sort_window_free(hp, sorted, state, &tmp);
        try_add_window_outcome(state, sorted, heuristic_window_tag(hp, window_size), outcomes, seen);
    }
    return outcomes;
}

std::vector<MctsNode::PotentialChild>
generate_heuristic_windowed_outcomes(const Simulator& state, int /*window_size*/,
                                      const std::vector<HeuristicWindowSpec>& specs) {
    const auto& queue = state.get_job_queue();
    if (queue.empty()) return {};

    std::vector<MctsNode::PotentialChild> outcomes;
    std::vector<std::vector<int>> seen;

    Mcts tmp(MctsConfig{});
    for (const auto& spec : specs) {
        int hw = std::min(static_cast<int>(queue.size()), spec.window_size);
        std::vector<int> window(queue.begin(), queue.begin() + hw);
        sort_window_free(spec.policy, window, state, &tmp);
        try_add_window_outcome(state, window, heuristic_window_tag(spec.policy, spec.window_size),
                               outcomes, seen);
    }
    return outcomes;
}

// ── Tree stats helpers ────────────────────────────────────────────────────────

static std::string window_key(const std::vector<int>& w) {
    std::string key;
    for (int id : w) key += std::to_string(id) + ",";
    return key;
}

static std::string format_root_possible_job_sets(
    const std::vector<MctsNode::PotentialChild>& outcomes) {
    std::vector<std::vector<int>> groups;
    groups.reserve(outcomes.size());

    for (const auto& child : outcomes) {
        std::vector<int> jobs = child.outcome;
        std::sort(jobs.begin(), jobs.end());
        groups.push_back(std::move(jobs));
    }

    std::sort(groups.begin(), groups.end());

    std::string formatted;
    for (size_t i = 0; i < groups.size(); ++i) {
        if (i > 0) formatted += ",";
        if (groups[i].empty()) {
            formatted += "|NONE|";
        } else {
            formatted += "|";
            for (size_t j = 0; j < groups[i].size(); ++j) {
                if (j > 0) formatted += ",";
                formatted += std::to_string(groups[i][j]);
            }
            formatted += "|";
        }
    }

    return formatted;
}

static void collect_tree_stats(const MctsNode* node, const std::string& parent_key,
                                std::unordered_set<std::string>& keys, int& max_depth) {
    for (const auto& child : node->children) {
        std::string k = parent_key + "|" + window_key(child->ordered_window);
        keys.insert(k);
        if (child->depth > max_depth) max_depth = child->depth;
        collect_tree_stats(child.get(), k, keys, max_depth);
    }
}

static std::string sorted_outcome_key(const std::vector<int>& outcome) {
    std::vector<int> sorted = outcome;
    std::sort(sorted.begin(), sorted.end());
    if (sorted.empty()) return "NONE";

    std::string key;
    for (int id : sorted) {
        if (!key.empty()) key += ",";
        key += std::to_string(id);
    }
    return key;
}

static const MctsNode* best_root_child_by_visits(const MctsNode& root) {
    const MctsNode* best = nullptr;
    double max_visits = -1.0;

    for (const auto& child : root.children) {
        if (child->visits > max_visits) {
            max_visits = child->visits;
            best = child.get();
        }
    }

    return best;
}

static std::pair<int, double> root_visit_leader_stats(const MctsNode& root,
                                                      const MctsNode* best) {
    if (!best) return {0, 0.0};

    int ties = 0;
    double second_best = -1.0;
    for (const auto& child : root.children) {
        if (child.get() == best) continue;
        if (child->visits == best->visits) ties++;
        else if (child->visits > second_best) second_best = child->visits;
    }

    ties += 1; // include the selected child itself
    double margin = ties > 1 ? 0.0
        : (second_best >= 0.0 ? best->visits - second_best : best->visits);
    return {ties, margin};
}

static MctsConvergenceSample make_convergence_sample(
    const MctsNode& root, int iteration, int root_branching, bool changed) {

    MctsConvergenceSample sample;
    sample.iteration         = iteration;
    sample.root_branching    = root_branching;
    sample.expanded_children = static_cast<int>(root.children.size());
    sample.total_root_visits = root.visits;
    sample.changed           = changed;

    const MctsNode* best = best_root_child_by_visits(root);
    if (!best) return sample;

    sample.selected_window     = best->ordered_window;
    sample.selected_outcome    = best->outcome;
    sample.selected_policy     = best->source_policy_label;
    sample.selected_visits     = best->visits;
    sample.selected_visit_pct  = root.visits > 0.0 ? 100.0 * best->visits / root.visits : 0.0;
    sample.selected_avg_reward = best->visits > 0.0 ? best->score / best->visits : 0.0;
    auto [ties, margin] = root_visit_leader_stats(root, best);
    sample.best_visit_ties = ties;
    sample.visit_margin = margin;
    return sample;
}

static MctsConvergenceSample make_convergence_sample_from_unsearched_choice(
    const MctsNode::PotentialChild& choice, int root_branching) {

    MctsConvergenceSample sample;
    sample.iteration        = 0;
    sample.root_branching   = root_branching;
    sample.selected_window  = choice.window;
    sample.selected_outcome = choice.outcome;
    sample.selected_policy  = choice.source_policy;
    sample.best_visit_ties = 1;
    sample.changed          = true;
    return sample;
}

struct AdvanceResult {
    std::vector<MctsNode::PotentialChild> untried_children;
    std::vector<int> transition_outcome;
    double transition_wait_s      = 0.0;
    double transition_bsld        = 0.0;
    double transition_used_proc_s = 0.0;
    double transition_elapsed_s   = 0.0;
    int    transition_count       = 0;
    bool   solved                 = false;
    bool   terminal               = false;
};

static AdvanceResult advance_to_decision_point(Simulator& state, MctsNode* parent,
                                               int window_size, int depth,
                                               int max_depth,
                                               const MctsNode::OutcomeGenerator& gen);

// ── MctsNode implementation ───────────────────────────────────────────────────

MctsNode::MctsNode(Simulator s, int action, MctsNode* p,
                   int window_size, int depth_in, int max_depth,
                   std::vector<int> ordered_window_in,
                   std::mt19937* rng, OutcomeGenerator gen,
                   std::string source_policy_label_in)
    : state(std::move(s)), action_job_id(action),
      ordered_window(std::move(ordered_window_in)),
      source_policy_label(std::move(source_policy_label_in)),
      parent(p), depth(depth_in) {

    if (!gen) gen = generate_permutation_outcomes;
    AdvanceResult advanced = advance_to_decision_point(*state, parent, window_size,
                                                       depth, max_depth, gen);
    untried_children       = std::move(advanced.untried_children);
    transition_outcome     = std::move(advanced.transition_outcome);
    transition_wait_s      = advanced.transition_wait_s;
    transition_bsld        = advanced.transition_bsld;
    transition_used_proc_s = advanced.transition_used_proc_s;
    transition_elapsed_s   = advanced.transition_elapsed_s;
    transition_count       = advanced.transition_count;
    terminal_              = advanced.terminal;
    solved                 = advanced.solved;

    if (solved) return;

    if (rng) {
        std::shuffle(untried_children.begin(), untried_children.end(), *rng);
    } else {
        std::mt19937 local_rng(std::random_device{}());
        std::shuffle(untried_children.begin(), untried_children.end(), local_rng);
    }
}

bool MctsNode::is_terminal()       const { return terminal_; }
bool MctsNode::is_fully_expanded() const { return untried_children.empty(); }
void MctsNode::release_state()           { state.reset(); }

MctsNode* MctsNode::best_child(double exploration) {
    MctsNode* best     = nullptr;
    double    best_val = -std::numeric_limits<double>::infinity();

    for (const auto& child : children) {
        double val;
        if (child->visits < 1.0) {
            val = std::numeric_limits<double>::infinity();
        } else {
            double mean = child->score / child->visits;
            val = mean + exploration * std::sqrt(2.0 * std::log(visits) / child->visits);
        }
        if (val > best_val) { best_val = val; best = child.get(); }
    }
    return best;
}

// ── Reward helpers ────────────────────────────────────────────────────────────

static double get_wait_s(const Simulator& sim, int id) {
    return static_cast<double>(sim.current_time() - sim.get_job(id).submit_time);
}

static double get_wall_s(const Simulator& sim, int id) {
    // MCTS always runs in Prediction mode — walltime is the job duration used
    return static_cast<double>(sim.get_job(id).walltime);
}

static double get_bsld(const Simulator& sim, int id) {
    double wait = get_wait_s(sim, id);
    double wall = std::max(get_wall_s(sim, id), 100.0);
    return (wall + wait) / wall;
}

static double current_utilization(const Simulator& sim) {
    double total = static_cast<double>(sim.total_procs());
    if (total <= 0.0) return 0.0;
    double used = total - static_cast<double>(sim.available_procs());
    return std::clamp(used / total, 0.0, 1.0);
}

static void accumulate_utilization_interval(const Simulator& sim, long next_time,
                                            double& used_proc_s,
                                            double& elapsed_s) {
    long dt = next_time - sim.current_time();
    if (dt <= 0) return;

    double total = static_cast<double>(sim.total_procs());
    double used  = total - static_cast<double>(sim.available_procs());
    used_proc_s += used * static_cast<double>(dt);
    elapsed_s   += static_cast<double>(dt);
}

static void insert_sorted(std::vector<double>& v, double val) {
    v.insert(std::lower_bound(v.begin(), v.end(), val), val);
}

static double p99_from_sorted(const std::vector<double>& sorted) {
    if (sorted.empty()) return 0.0;
    size_t idx = static_cast<size_t>(std::ceil(0.99 * static_cast<double>(sorted.size())));
    if (idx > 0) idx--;
    return sorted[std::min(idx, sorted.size() - 1)];
}

static double calc_reward(RewardType type, const Simulator& sim,
                          double cum_wait_s, int cum_count, double cum_bsld,
                          double cum_used_proc_s, double cum_elapsed_s,
                          double p99_lambda = 1.0,
                          const std::vector<double>* cum_wait_sorted = nullptr,
                          double max_lambda = 1.0,
                          double avg_lambda = 1.0,
                          double exp_scale  = 1.0) {
    switch (type) {
        case RewardType::InvCumWait: {
            double avg_wait_h = cum_count > 0
                                ? (cum_wait_s / static_cast<double>(cum_count)) / 3600.0
                                : 0.0;
            return 1.0 / (1.0 + avg_wait_h);
        }
        case RewardType::InvCumBSLD: {
            double avg_bsld = cum_count > 0
                              ? cum_bsld / static_cast<double>(cum_count)
                              : 1.0;
            return 1.0 / avg_bsld;
        }
        case RewardType::InstantUtil:
            return current_utilization(sim);
        case RewardType::CumUtil: {
            double total = static_cast<double>(sim.total_procs());
            if (total <= 0.0) return 0.0;
            if (cum_elapsed_s <= 0.0) return current_utilization(sim);
            return std::clamp(cum_used_proc_s / (total * cum_elapsed_s), 0.0, 1.0);
        }
        case RewardType::InvCumP99Wait: {
            double avg_wait_h = cum_count > 0
                                ? (cum_wait_s / static_cast<double>(cum_count)) / 3600.0
                                : 0.0;
            double p99_h = (cum_wait_sorted && !cum_wait_sorted->empty())
                           ? p99_from_sorted(*cum_wait_sorted) / 3600.0
                           : avg_wait_h;
            double inv_avg = 1.0 / (1.0 + avg_wait_h);
            double inv_p99 = 1.0 / (1.0 + p99_h);
            return std::pow(inv_avg, avg_lambda) * std::pow(inv_p99, p99_lambda);
        }
        case RewardType::NegCumWait: {
            double avg_wait_h = cum_count > 0
                                ? (cum_wait_s / static_cast<double>(cum_count)) / 3600.0
                                : 0.0;
            return -avg_wait_h;
        }
        case RewardType::NegP99Wait: {
            double p99_h = (cum_wait_sorted && !cum_wait_sorted->empty())
                           ? p99_from_sorted(*cum_wait_sorted) / 3600.0
                           : 0.0;
            return -p99_h;
        }
        case RewardType::Constant:
            return 1.0;
        case RewardType::InvCumP99MaxWait: {
            double avg_wait_h = cum_count > 0
                                ? (cum_wait_s / static_cast<double>(cum_count)) / 3600.0
                                : 0.0;
            double p99_h = (cum_wait_sorted && !cum_wait_sorted->empty())
                           ? p99_from_sorted(*cum_wait_sorted) / 3600.0
                           : avg_wait_h;
            double max_h = (cum_wait_sorted && !cum_wait_sorted->empty())
                           ? cum_wait_sorted->back() / 3600.0
                           : avg_wait_h;
            double inv_avg = 1.0 / (1.0 + avg_wait_h);
            double inv_p99 = 1.0 / (1.0 + p99_h);
            double inv_max = 1.0 / (1.0 + max_h);
            return std::pow(inv_avg, avg_lambda) * std::pow(inv_p99, p99_lambda) * std::pow(inv_max, max_lambda);
        }
        case RewardType::ExpWait: {
            double avg_wait_h = cum_count > 0
                                ? (cum_wait_s / static_cast<double>(cum_count)) / 3600.0
                                : 0.0;
            double p99_h = (cum_wait_sorted && !cum_wait_sorted->empty())
                           ? p99_from_sorted(*cum_wait_sorted) / 3600.0
                           : avg_wait_h;
            double max_h = (cum_wait_sorted && !cum_wait_sorted->empty())
                           ? cum_wait_sorted->back() / 3600.0
                           : avg_wait_h;
            return std::exp(-exp_scale * (avg_lambda * avg_wait_h + p99_lambda * p99_h + max_lambda * max_h));
        }
    }

    return 0.0;
}

static AdvanceResult advance_to_decision_point(Simulator& state, MctsNode* parent,
                                               int window_size, int depth,
                                               int max_depth,
                                               const MctsNode::OutcomeGenerator& gen) {
    AdvanceResult result;

    while (true) {
        if (state.is_done() || depth >= max_depth) {
            result.terminal = state.is_done();
            result.solved   = true;
            return result;
        }

        auto next_event = state.peek();
        if (!next_event) {
            result.terminal = state.is_done();
            result.solved   = true;
            return result;
        }

        if (next_event->type != EventType::SchedulingCycle) {
            accumulate_utilization_interval(state, next_event->time,
                                            result.transition_used_proc_s,
                                            result.transition_elapsed_s);
            state.step();
            continue;
        }

        result.untried_children = gen(state, window_size);

        // Fast-forward through single-outcome states (non-root only)
        if (parent != nullptr && result.untried_children.size() == 1) {
            const auto& choice = result.untried_children[0];
            Simulator sim = regenerate_child_sim(state, choice.window, window_size);
            if (sim.get_job_queue().size() < state.get_job_queue().size()) {
                for (int id : choice.outcome) {
                    result.transition_outcome.push_back(id);
                    result.transition_wait_s += get_wait_s(state, id);
                    result.transition_bsld   += get_bsld(state, id);
                    result.transition_count  += 1;
                }
                state = std::move(sim);
                result.untried_children.clear();
                continue;
            }
        }

        // Fast-forward if only one action leads to actual scheduling
        if (parent != nullptr && result.untried_children.size() > 1) {
            int ne_count = 0, ne_idx = -1;
            for (int i = 0; i < static_cast<int>(result.untried_children.size()); ++i) {
                if (!result.untried_children[i].outcome.empty()) {
                    ne_count++;
                    ne_idx = i;
                }
            }
            if (ne_count == 0) {
                int pick = (ne_count == 1) ? ne_idx : 0;
                const auto& choice = result.untried_children[pick];
                Simulator sim = regenerate_child_sim(state, choice.window, window_size);
                if (sim.get_job_queue().size() < state.get_job_queue().size() || ne_count == 0) {
                    for (int id : choice.outcome) {
                        result.transition_outcome.push_back(id);
                        result.transition_wait_s += get_wait_s(state, id);
                        result.transition_bsld   += get_bsld(state, id);
                        result.transition_count  += 1;
                    }
                    state = std::move(sim);
                    result.untried_children.clear();
                    continue;
                }
            }
        }

        break;
    }

    result.terminal = state.is_done();
    return result;
}

// ── Mcts implementation ───────────────────────────────────────────────────────

Mcts::Mcts(MctsConfig config)
    : config_(std::move(config)),
      rng_(config_.seed == 0 ? std::random_device{}() : config_.seed),
      global_min_score_( std::numeric_limits<double>::infinity()),
      global_max_score_(-std::numeric_limits<double>::infinity()) {

    if (config_.branching == BranchingMode::ComprehensiveHeuristic) {
        outcome_gen_ = [this](const Simulator& s, int /*ws*/) {
            const auto specs = comprehensive_heuristic_window_specs();
            const auto& queue = s.get_job_queue();
            if (queue.empty()) return std::vector<MctsNode::PotentialChild>{};
            std::vector<MctsNode::PotentialChild> outcomes;
            std::vector<std::vector<int>> seen;
            for (const auto& spec : specs) {
                int hw = std::min(static_cast<int>(queue.size()), spec.window_size);
                std::vector<int> prefix(queue.begin(), queue.begin() + hw);
                sort_window(spec.policy, prefix, s);
                try_add_window_outcome(s, prefix,
                                       heuristic_window_tag(spec.policy, spec.window_size),
                                       outcomes, seen);
            }
            return outcomes;
        };
    } else if (config_.branching == BranchingMode::HeuristicWindowed) {
        outcome_gen_ = [this](const Simulator& s, int ws) {
            const auto& specs = config_.heuristic_windows;
            const auto& queue = s.get_job_queue();
            if (queue.empty()) return std::vector<MctsNode::PotentialChild>{};
            std::vector<MctsNode::PotentialChild> outcomes;
            std::vector<std::vector<int>> seen;
            for (const auto& spec : specs) {
                int hw = std::min(static_cast<int>(queue.size()), spec.window_size);
                std::vector<int> prefix(queue.begin(), queue.begin() + hw);
                sort_window(spec.policy, prefix, s);
                try_add_window_outcome(s, prefix,
                                       heuristic_window_tag(spec.policy, spec.window_size),
                                       outcomes, seen);
            }
            return outcomes;
        };
    } else if (config_.branching == BranchingMode::Heuristic) {
        outcome_gen_ = [this](const Simulator& s, int ws) {
            const auto& queue = s.get_job_queue();
            if (queue.empty()) return std::vector<MctsNode::PotentialChild>{};
            int w = std::min(static_cast<int>(queue.size()), ws);
            std::vector<MctsNode::PotentialChild> outcomes;
            std::vector<std::vector<int>> seen;
            for (auto hp : config_.heuristics) {
                std::vector<int> prefix(queue.begin(), queue.begin() + w);
                sort_window(hp, prefix, s);
                try_add_window_outcome(s, prefix,
                                       heuristic_window_tag(hp, config_.window_size),
                                       outcomes, seen);
            }
            return outcomes;
        };
    } else {
        outcome_gen_ = generate_permutation_outcomes;
    }
}

std::vector<int> Mcts::search(const Simulator& root_state) {
    auto log = [this](const std::string& msg) {
        if (!config_.log_file.empty()) {
            std::ofstream f(config_.log_file, std::ios::app);
            f << msg << "\n";
        }
    };

    Simulator mcts_sim = root_state.get_copy(false);
    mcts_sim.set_mode(SimMode::Prediction);

    auto root = std::make_unique<MctsNode>(
        std::move(mcts_sim), -1, nullptr,
        config_.window_size, 0, config_.max_depth,
        std::vector<int>{}, &rng_, outcome_gen_, "");

    last_root_branching_ = static_cast<int>(root->untried_children.size());
    last_selected_policy_.clear();
    last_root_possible_policies_ = format_root_possible_job_sets(root->untried_children);

    if (root->untried_children.empty() && root->children.empty()) return {};

    int non_empty_count = 0;
    for (const auto& uc : root->untried_children)
        if (!uc.outcome.empty()) non_empty_count++;
    if (non_empty_count == 0) {
        last_selected_policy_ = "FCFS";
        return root->untried_children[0].window;
    }
    if (root->untried_children.size() == 1) {
        last_selected_policy_ = "FCFS";
        return root->untried_children[0].window;
    }
    // Empty-outcome branches are meaningful (drain: start nothing to free resources for a
    // large job later). Only short-circuit when there are zero non-empty branches; if even
    // one branch starts jobs AND others don't, MCTS must compare them.

    auto start   = std::chrono::steady_clock::now();
    deadline_    = start + std::chrono::milliseconds(config_.time_limit_ms);
    last_iterations_ = 0;

    while (!root->solved) {
        if (std::chrono::steady_clock::now() >= deadline_) break;
        MctsNode* leaf   = tree_policy(root.get());
        double reward    = rollout(leaf->state->get_copy(false), leaf->depth,
                                   leaf->cum_wait_s, leaf->cum_count,
                                   leaf->cum_bsld, leaf->cum_used_proc_s,
                                   leaf->cum_elapsed_s, leaf->cum_wait_sorted);
        backpropagate(leaf, reward);
        last_iterations_++;
    }

    MctsNode* best = nullptr;
    double    max_visits = -1;
    for (const auto& child : root->children) {
        if (child->visits > max_visits) { max_visits = child->visits; best = child.get(); }
    }

    std::unordered_set<std::string> node_keys;
    last_max_depth_ = 0;
    collect_tree_stats(root.get(), "", node_keys, last_max_depth_);
    last_unique_nodes_ = static_cast<int>(node_keys.size());

    if (best && root->visits > 0) {
        last_chosen_visit_pct_  = 100.0 * best->visits / root->visits;
        last_chosen_avg_reward_ = best->visits > 0 ? best->score / best->visits : 0.0;
    }
    if (best) last_selected_policy_ = best->source_policy_label;

    log("=== MCTS | t=" + std::to_string(root_state.current_time())
        + " q=" + std::to_string(root_state.get_job_queue().size())
        + " iters=" + std::to_string(last_iterations_)
        + " nodes=" + std::to_string(last_unique_nodes_)
        + " depth=" + std::to_string(last_max_depth_)
        + " best=" + std::to_string(best ? best->action_job_id : -1)
        + " policy=" + last_selected_policy_
        + " visit%=" + std::to_string(last_chosen_visit_pct_));

    return best ? best->ordered_window : std::vector<int>{};
}

MctsRootAnalysis Mcts::inspect_root(const Simulator& root_state) {
    Simulator mcts_sim = root_state.get_copy(false);
    mcts_sim.set_mode(SimMode::Prediction);

    auto root = std::make_unique<MctsNode>(
        std::move(mcts_sim), -1, nullptr,
        config_.window_size, 0, config_.max_depth,
        std::vector<int>{}, &rng_, outcome_gen_, "");

    MctsRootAnalysis analysis;
    analysis.branching = static_cast<int>(root->untried_children.size());
    analysis.queue_len = static_cast<int>(root_state.get_job_queue().size());
    analysis.available_procs = root_state.available_procs();
    analysis.running_jobs = static_cast<int>(root_state.running_jobs().size());
    analysis.sim_time = root_state.current_time();
    analysis.possible_job_sets = format_root_possible_job_sets(root->untried_children);

    last_root_branching_ = analysis.branching;
    last_root_possible_policies_ = analysis.possible_job_sets;
    return analysis;
}

MctsConvergenceTrace Mcts::analyze_convergence(const Simulator& root_state,
                                               int max_iterations,
                                               int sample_every) {
    MctsConvergenceTrace trace;
    trace.max_iterations = std::max(0, max_iterations);
    if (max_iterations <= 0) return trace;
    if (sample_every <= 0) sample_every = 1;

    Simulator mcts_sim = root_state.get_copy(false);
    mcts_sim.set_mode(SimMode::Prediction);

    auto root = std::make_unique<MctsNode>(
        std::move(mcts_sim), -1, nullptr,
        config_.window_size, 0, config_.max_depth,
        std::vector<int>{}, &rng_, outcome_gen_, "");

    const int root_branching = static_cast<int>(root->untried_children.size());
    trace.root_branching = root_branching;
    last_root_branching_ = root_branching;
    last_iterations_ = 0;
    last_selected_policy_.clear();
    last_root_possible_policies_ = format_root_possible_job_sets(root->untried_children);

    if (root->untried_children.empty() && root->children.empty()) {
        trace.stabilization_iteration = 0;
        return trace;
    }

    int non_empty_count = 0;
    for (const auto& uc : root->untried_children)
        if (!uc.outcome.empty()) non_empty_count++;

    if (non_empty_count == 0 || root->untried_children.size() == 1) {
        const auto& choice = root->untried_children.front();
        trace.final_iteration = 0;
        trace.stabilization_iteration = 0;
        trace.final_selected_window = choice.window;
        trace.final_selected_outcome = choice.outcome;
        trace.final_selected_policy = choice.source_policy.empty() ? "FCFS" : choice.source_policy;
        trace.samples.push_back(
            make_convergence_sample_from_unsearched_choice(choice, root_branching));
        last_selected_policy_ = trace.final_selected_policy;
        return trace;
    }

    deadline_ = std::chrono::steady_clock::time_point::max();

    std::string last_choice_key;
    int last_change_iteration = -1;

    for (int iter = 1; iter <= max_iterations && !root->solved; ++iter) {
        MctsNode* leaf = tree_policy(root.get());
        double reward = rollout(leaf->state->get_copy(false), leaf->depth,
                                leaf->cum_wait_s, leaf->cum_count,
                                leaf->cum_bsld, leaf->cum_used_proc_s,
                                leaf->cum_elapsed_s, leaf->cum_wait_sorted);
        backpropagate(leaf, reward);
        last_iterations_ = iter;

        const MctsNode* best = best_root_child_by_visits(*root);
        std::string choice_key = best ? sorted_outcome_key(best->outcome) : "";
        bool changed = choice_key != last_choice_key;
        if (changed) {
            last_choice_key = choice_key;
            last_change_iteration = iter;
        }

        if (changed || iter % sample_every == 0 || iter == max_iterations) {
            trace.samples.push_back(
                make_convergence_sample(*root, iter, root_branching, changed));
        }
    }

    trace.final_iteration = last_iterations_;
    trace.stabilization_iteration =
        last_change_iteration >= 0 ? last_change_iteration : trace.final_iteration;

    const MctsNode* best = best_root_child_by_visits(*root);
    if (best) {
        trace.final_selected_window  = best->ordered_window;
        trace.final_selected_outcome = best->outcome;
        trace.final_selected_policy  = best->source_policy_label;
        last_selected_policy_        = best->source_policy_label;
        if (root->visits > 0.0) {
            last_chosen_visit_pct_ = 100.0 * best->visits / root->visits;
            last_chosen_avg_reward_ = best->visits > 0.0 ? best->score / best->visits : 0.0;
        }
    }

    if (trace.samples.empty() ||
        trace.samples.back().iteration != trace.final_iteration) {
        trace.samples.push_back(
            make_convergence_sample(*root, trace.final_iteration,
                                    root_branching, false));
    }

    std::unordered_set<std::string> node_keys;
    last_max_depth_ = 0;
    collect_tree_stats(root.get(), "", node_keys, last_max_depth_);
    last_unique_nodes_ = static_cast<int>(node_keys.size());

    return trace;
}

MctsNode* Mcts::tree_policy(MctsNode* node) {
    while (!node->is_terminal()) {
        if (node->depth >= config_.max_depth) return node;
        if (!node->is_fully_expanded()) {
            if (std::chrono::steady_clock::now() >= deadline_) return node;
            return expand(node);
        }
        if (node->children.empty()) return node;
        double c_eff = config_.exploration;
        if (config_.dynamic_exploration && global_max_score_ > global_min_score_)
            c_eff *= (global_max_score_ - global_min_score_);
        node = node->best_child(c_eff);
    }
    return node;
}

MctsNode* Mcts::expand(MctsNode* node) {
    auto potential = std::move(node->untried_children.back());
    node->untried_children.pop_back();

    Simulator child_sim = regenerate_child_sim(*node->state, potential.window, config_.window_size);
    auto child = std::make_unique<MctsNode>(
        std::move(child_sim), potential.action_job_id, node,
        config_.window_size, node->depth + 1, config_.max_depth,
        potential.window, &rng_, outcome_gen_, potential.source_policy);

    child->outcome         = potential.outcome;
    child->cum_wait_s      = node->cum_wait_s;
    child->cum_bsld        = node->cum_bsld;
    child->cum_used_proc_s = node->cum_used_proc_s + child->transition_used_proc_s;
    child->cum_elapsed_s   = node->cum_elapsed_s + child->transition_elapsed_s;
    child->cum_count       = node->cum_count;
    child->cum_wait_sorted = node->cum_wait_sorted;

    for (int id : potential.outcome) {
        double w = get_wait_s(*node->state, id);
        child->cum_wait_s += w;
        child->cum_bsld   += get_bsld(*node->state, id);
        child->cum_count  += 1;
        insert_sorted(child->cum_wait_sorted, w);
    }

    child->outcome.insert(child->outcome.end(),
                          child->transition_outcome.begin(),
                          child->transition_outcome.end());
    child->cum_wait_s += child->transition_wait_s;
    child->cum_bsld   += child->transition_bsld;
    child->cum_count  += child->transition_count;

    for (int id : child->transition_outcome) {
        double w = static_cast<double>(child->state->get_job(id).start_time
                                       - child->state->get_job(id).submit_time);
        insert_sorted(child->cum_wait_sorted, w);
    }

    child->immediate_reward = config_.reward_scale *
                              calc_reward(config_.reward_type, *child->state,
                                          child->cum_wait_s, child->cum_count,
                                          child->cum_bsld, child->cum_used_proc_s,
                                          child->cum_elapsed_s,
                                          config_.p99_lambda,
                                          &child->cum_wait_sorted,
                                          config_.max_lambda,
                                          config_.avg_lambda,
                                          config_.exp_scale);

    if (child->solved) mark_solved(child.get());
    node->children.push_back(std::move(child));

    if (node->is_fully_expanded() &&
        node->solved_children == static_cast<int>(node->children.size()))
        mark_solved(node);
    if (node->is_fully_expanded()) node->release_state();

    return node->children.back().get();
}

double Mcts::rollout(Simulator state, int current_depth,
                     double cum_wait_s, int cum_count, double cum_bsld,
                     double cum_used_proc_s, double cum_elapsed_s,
                     std::vector<double> cum_wait_sorted) {
    double cumulative = 0.0;
    double gamma_pow  = 1.0;

    while (!state.is_done() && current_depth < config_.max_depth) {
        if (std::chrono::steady_clock::now() >= deadline_) break;
        auto next_e = state.peek();
        if (next_e && next_e->type == EventType::SchedulingCycle) {
            Simulator before = state.get_copy(false);
            const auto& q    = state.get_job_queue();
            if (!q.empty()) {
                int w = std::min(static_cast<int>(q.size()), config_.window_size);
                std::vector<int> window(q.begin(), q.begin() + w);

                if (config_.branching == BranchingMode::ComprehensiveHeuristic) {
                    const auto& specs = comprehensive_heuristic_window_specs();
                    const auto& spec = specs[rng_() % specs.size()];
                    int hw = std::min(static_cast<int>(q.size()), spec.window_size);
                    window.assign(q.begin(), q.begin() + hw);
                    sort_window(spec.policy, window, state);
                } else if (config_.branching == BranchingMode::HeuristicWindowed) {
                    const auto& spec = config_.heuristic_windows[rng_() % config_.heuristic_windows.size()];
                    int hw = std::min(static_cast<int>(q.size()), spec.window_size);
                    window.assign(q.begin(), q.begin() + hw);
                    sort_window(spec.policy, window, state);
                } else if (config_.branching == BranchingMode::Heuristic) {
                    sort_window(config_.rollout_policy, window, state);
                } else {
                    std::shuffle(window.begin(), window.end(), rng_);
                }

                state = apply_window_ordering(state, window);
            } else {
                state.step();
            }

            std::vector<int> outcome = get_outcome(before, state);
            for (int id : outcome) {
                double w = get_wait_s(before, id);
                cum_wait_s += w;
                cum_bsld   += get_bsld(before, id);
                cum_count++;
                insert_sorted(cum_wait_sorted, w);
            }

            // Drain cycle: queue had jobs but nothing started — don't consume depth budget.
            // Mirrors advance_to_decision_point which doesn't create tree nodes for these.
            if (!before.get_job_queue().empty() && outcome.empty()) continue;

            double step_reward = config_.reward_scale *
                                 calc_reward(config_.reward_type, state,
                                             cum_wait_s, cum_count, cum_bsld,
                                             cum_used_proc_s, cum_elapsed_s,
                                             config_.p99_lambda, &cum_wait_sorted,
                                             config_.max_lambda, config_.avg_lambda,
                                             config_.exp_scale);
            cumulative += gamma_pow * step_reward;
            gamma_pow  *= config_.discount;
            current_depth++;
        } else {
            if (next_e) {
                accumulate_utilization_interval(state, next_e->time,
                                                cum_used_proc_s, cum_elapsed_s);
            }
            state.step();
        }
    }
    return cumulative * (1.0 - config_.discount);
}

void Mcts::backpropagate(MctsNode* node, double reward) {
    double traj = reward;
    while (node) {
        traj = (1.0 - config_.discount) * node->immediate_reward + config_.discount * traj;
        node->visits   += 1;
        node->score    += traj;
        node->score_sq += traj * traj;
        if (config_.dynamic_exploration) {
            global_min_score_ = std::min(global_min_score_, traj);
            global_max_score_ = std::max(global_max_score_, traj);
        }
        node = node->parent;
    }
}

void Mcts::mark_solved(MctsNode* node) {
    if (node->solved) return;
    node->solved = true;
    if (node->parent) {
        node->parent->solved_children++;
        if (node->parent->is_fully_expanded() &&
            node->parent->solved_children == static_cast<int>(node->parent->children.size()))
            mark_solved(node->parent);
    }
}

// ── Parallel MCTS ─────────────────────────────────────────────────────────────

std::vector<int> Mcts::parallel_search(const Simulator& root_state,
                                       const MctsConfig& config,
                                       int* total_iterations,
                                       int* unique_nodes_out,
                                       int* max_depth_out,
                                       int* root_branching_out,
                                       double* chosen_visit_pct_out,
                                       double* chosen_avg_reward_out,
                                       std::string* chosen_policy_out,
                                       std::string* root_possible_policies_out) {
    auto log = [&config](const std::string& msg) {
        if (!config.log_file.empty()) {
            std::ofstream f(config.log_file, std::ios::app);
            f << msg << "\n";
        }
    };

    if (config.num_cores <= 1) {
        Mcts mcts(config);
        auto result = mcts.search(root_state);
        if (total_iterations)   *total_iterations   = mcts.last_iterations();
        if (unique_nodes_out)   *unique_nodes_out   = mcts.last_unique_nodes();
        if (max_depth_out)      *max_depth_out      = mcts.last_max_depth();
        if (root_branching_out) *root_branching_out = mcts.last_root_branching();
        if (chosen_visit_pct_out)  *chosen_visit_pct_out  = mcts.last_chosen_visit_pct();
        if (chosen_avg_reward_out) *chosen_avg_reward_out = mcts.last_chosen_avg_reward();
        if (chosen_policy_out)     *chosen_policy_out     = mcts.last_selected_policy();
        return result;
    }

    // Pre-check: skip MCTS if only one meaningful action exists
    Mcts probe(config);
    auto root_outcomes = probe.outcome_gen_(root_state, config.window_size);
    if (root_branching_out) *root_branching_out = static_cast<int>(root_outcomes.size());
    if (root_possible_policies_out)
        *root_possible_policies_out = format_root_possible_job_sets(root_outcomes);

    if (root_outcomes.empty()) return {};

    int ne_count = 0;
    for (const auto& ro : root_outcomes)
        if (!ro.outcome.empty()) ne_count++;
    if (ne_count == 0) {
        if (chosen_policy_out) *chosen_policy_out = root_outcomes[0].source_policy;
        return root_outcomes[0].window;
    }
    if (root_outcomes.size() == 1) {
        if (chosen_policy_out) *chosen_policy_out = root_outcomes[0].source_policy;
        return root_outcomes[0].window;
    }
    // Empty-outcome branches are meaningful (drain). Only skip search when there is
    // truly one branch total (handled above) or all branches are blocked (ne_count==0).

    int num_cores = config.num_cores;
    std::vector<std::vector<MctsChildResult>> all_results(num_cores);
    std::vector<int> worker_iters(num_cores, 0);
    std::vector<std::unordered_set<std::string>> worker_keys(num_cores);
    std::vector<int> worker_max_depth(num_cores, 0);

    std::mt19937 seed_gen(static_cast<unsigned>(config.seed));
    std::vector<unsigned int> seeds(num_cores);
    for (int i = 0; i < num_cores; ++i) seeds[i] = seed_gen();

    #pragma omp parallel for num_threads(num_cores) schedule(static)
    for (int i = 0; i < num_cores; ++i) {
        MctsConfig worker_cfg = config;
        worker_cfg.seed = seeds[i];
        worker_cfg.log_file = "";

        Mcts mcts(worker_cfg);
        Simulator sim_copy = root_state.get_copy(false);
        sim_copy.set_mode(SimMode::Prediction);

        auto root = std::make_unique<MctsNode>(
            std::move(sim_copy), -1, nullptr,
            config.window_size, 0, config.max_depth,
            std::vector<int>{}, &mcts.rng_, mcts.outcome_gen_, "");

        if (!root->untried_children.empty() || !root->children.empty()) {
            auto start_t  = std::chrono::steady_clock::now();
            mcts.deadline_ = start_t + std::chrono::milliseconds(config.time_limit_ms);
            int iters = 0;
            while (!root->solved) {
                if (std::chrono::steady_clock::now() >= mcts.deadline_) break;
                MctsNode* leaf = mcts.tree_policy(root.get());
                double reward  = mcts.rollout(leaf->state->get_copy(false), leaf->depth,
                                              leaf->cum_wait_s, leaf->cum_count,
                                              leaf->cum_bsld, leaf->cum_used_proc_s,
                                              leaf->cum_elapsed_s, leaf->cum_wait_sorted);
                mcts.backpropagate(leaf, reward);
                iters++;
            }
            worker_iters[i] = iters;
            for (const auto& child : root->children)
                all_results[i].push_back({child->ordered_window, child->visits, child->score,
                                          child->outcome, child->source_policy_label});
        }
        collect_tree_stats(root.get(), "", worker_keys[i], worker_max_depth[i]);
    }

    // Aggregate stats
    int total_iters = 0;
    for (int n : worker_iters) total_iters += n;
    if (total_iterations) *total_iterations = total_iters;

    std::unordered_set<std::string> all_keys;
    int g_max_depth = 0;
    for (int i = 0; i < num_cores; ++i) {
        all_keys.insert(worker_keys[i].begin(), worker_keys[i].end());
        g_max_depth = std::max(g_max_depth, worker_max_depth[i]);
    }
    if (unique_nodes_out) *unique_nodes_out = static_cast<int>(all_keys.size());
    if (max_depth_out)    *max_depth_out    = g_max_depth;

    // Merge results by outcome key (sorted scheduled job set)
    auto outcome_key = [](const std::vector<int>& out) {
        std::vector<int> s = out;
        std::sort(s.begin(), s.end());
        std::string k;
        for (int id : s) k += std::to_string(id) + ",";
        return k.empty() ? std::string("none") : k;
    };

    std::unordered_map<std::string, std::pair<double,double>> agg;  // visits, score
    std::unordered_map<std::string, std::vector<int>> key_to_window;
    std::unordered_map<std::string, std::string> key_to_policy;

    for (int w = 0; w < num_cores; ++w) {
        for (const auto& r : all_results[w]) {
            std::string k = outcome_key(r.outcome);
            agg[k].first  += r.visits;
            agg[k].second += r.score;
            if (!key_to_window.count(k)) key_to_window[k] = r.ordered_window;
            if (!key_to_policy.count(k)) key_to_policy[k] = r.source_policy;
        }
    }

    std::string best_key;
    double max_v = -1;
    for (const auto& [k, stats] : agg) {
        if (stats.first > max_v) { max_v = stats.first; best_key = k; }
    }

    if (chosen_visit_pct_out || chosen_avg_reward_out) {
        double total_v = 0;
        for (const auto& [k, stats] : agg) total_v += stats.first;
        if (chosen_visit_pct_out)
            *chosen_visit_pct_out = total_v > 0 ? 100.0 * agg[best_key].first / total_v : 0.0;
        if (chosen_avg_reward_out)
            *chosen_avg_reward_out = agg[best_key].first > 0
                                     ? agg[best_key].second / agg[best_key].first : 0.0;
    }
    if (chosen_policy_out && !best_key.empty())
        *chosen_policy_out = key_to_policy[best_key];

    log("=== Parallel MCTS | t=" + std::to_string(root_state.current_time())
        + " cores=" + std::to_string(num_cores)
        + " total_iters=" + std::to_string(total_iters)
        + " best=" + best_key);

    if (best_key.empty()) return {};
    return key_to_window[best_key];
}

// ── MctsPolicy ────────────────────────────────────────────────────────────────

MctsPolicy::MctsPolicy(MctsConfig config)
    : config_(config), mcts_(config) {}

std::vector<Decision> MctsPolicy::schedule(const State& s) {
    last_cycle_policy_.clear();
    last_cycle_root_branching_ = 0;
    last_cycle_possible_policies_.clear();
    if (!s.sim || s.pending.empty()) return {};

    const auto& queue = s.sim->get_job_queue();
    if (queue.size() == 1) {
        last_cycle_policy_ = "FCFS";
        last_cycle_root_branching_ = 1;
        last_cycle_possible_policies_ = "|" + std::to_string(queue.front()) + "|";
        return dispatch({queue.front()}, s);
    }

    // When called from a Driver scheduling callback, the SchedulingCycle event
    // has already been consumed. MCTS needs to fire a SchedulingCycle to evaluate
    // outcomes. Inject one into a copy so the original sim is unaffected.
    Simulator search_sim = s.sim->get_copy(false);
    auto next = search_sim.peek();
    if (!search_sim.get_job_queue().empty() &&
        (!next || next->type != EventType::SchedulingCycle)) {
        search_sim.inject_scheduling_cycle();
    }

    std::vector<int> best_window;
    if (config_.num_cores > 1) {
        best_window = Mcts::parallel_search(search_sim, config_, &last_cycle_iterations_, nullptr, nullptr,
                                            &last_cycle_root_branching_, nullptr, nullptr,
                                            &last_cycle_policy_, &last_cycle_possible_policies_);
    } else {
        best_window = mcts_.search(search_sim);
        last_cycle_policy_ = mcts_.last_selected_policy();
        last_cycle_root_branching_ = mcts_.last_root_branching();
        last_cycle_possible_policies_ = mcts_.last_root_possible_policies();
        last_cycle_iterations_ = mcts_.last_iterations();
    }

    if (best_window.empty()) return {};
    return dispatch(best_window, s);
}

std::vector<Decision> MctsPolicy::dispatch(const std::vector<int>& window,
                                            const State& s) const {
    if (!s.sim || s.pending.empty()) return {};

    Simulator apply_sim = s.sim->get_copy(false);
    auto next = apply_sim.peek();
    if (!apply_sim.get_job_queue().empty() &&
        (!next || next->type != EventType::SchedulingCycle)) {
        apply_sim.inject_scheduling_cycle();
    }

    std::vector<int> started_jobs;
    apply_window_ordering(apply_sim, window, &started_jobs);

    std::unordered_map<int, const Job*> pending_map;
    for (const Job* j : s.pending) pending_map[j->id] = j;

    std::vector<NodeInfo> nodes = s.nodes;
    std::vector<Decision> decisions;
    for (int id : started_jobs) {
        auto it = pending_map.find(id);
        if (it == pending_map.end()) continue;
        for (NodeInfo& node : nodes) {
            if (node.can_fit(it->second->requested)) {
                node.allocate(it->second->requested);
                decisions.push_back({DecisionType::Start, id, {node.id}});
                break;
            }
        }
    }
    return decisions;
}

// ── RandomHeuristicPolicy ─────────────────────────────────────────────────────

RandomHeuristicPolicy::RandomHeuristicPolicy(int window_size, unsigned seed)
    : window_size_(window_size),
      rng_(seed == 0 ? std::random_device{}() : seed),
      mcts_([seed] {
          MctsConfig mc;
          mc.branching = BranchingMode::ComprehensiveHeuristic;
          mc.seed      = seed;
          return mc;
      }()) {}

std::vector<Decision> RandomHeuristicPolicy::schedule(const State& s) {
    last_cycle_policy_.clear();
    if (!s.sim || s.pending.empty()) return {};

    Simulator search_sim = s.sim->get_copy(false);
    auto next = search_sim.peek();
    if (!search_sim.get_job_queue().empty() &&
        (!next || next->type != EventType::SchedulingCycle)) {
        search_sim.inject_scheduling_cycle();
    }

    // Enumerate all unique scheduling outcomes from the comprehensive pool.
    auto outcomes = mcts_.outcome_gen_(search_sim, window_size_);
    if (outcomes.empty()) return {};

    // Prefer outcomes that actually start at least one job.
    std::vector<int> candidates;
    for (int i = 0; i < static_cast<int>(outcomes.size()); ++i)
        if (!outcomes[i].outcome.empty()) candidates.push_back(i);
    if (candidates.empty()) candidates.push_back(0);

    std::uniform_int_distribution<int> dist(0, static_cast<int>(candidates.size()) - 1);
    int chosen = candidates[dist(rng_)];
    last_cycle_policy_ = outcomes[chosen].source_policy;
    return dispatch(outcomes[chosen].window, s);
}

std::vector<Decision> RandomHeuristicPolicy::dispatch(const std::vector<int>& window,
                                                       const State& s) const {
    if (!s.sim || s.pending.empty()) return {};

    Simulator apply_sim = s.sim->get_copy(false);
    auto next = apply_sim.peek();
    if (!apply_sim.get_job_queue().empty() &&
        (!next || next->type != EventType::SchedulingCycle)) {
        apply_sim.inject_scheduling_cycle();
    }

    std::vector<int> started_jobs;
    apply_window_ordering(apply_sim, window, &started_jobs);

    std::unordered_map<int, const Job*> pending_map;
    for (const Job* j : s.pending) pending_map[j->id] = j;

    std::vector<NodeInfo> nodes = s.nodes;
    std::vector<Decision> decisions;
    for (int id : started_jobs) {
        auto it = pending_map.find(id);
        if (it == pending_map.end()) continue;
        for (NodeInfo& node : nodes) {
            if (node.can_fit(it->second->requested)) {
                node.allocate(it->second->requested);
                decisions.push_back({DecisionType::Start, id, {node.id}});
                break;
            }
        }
    }
    return decisions;
}
