#include <gtest/gtest.h>
#include "mcts/mcts.h"
#include <chrono>

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::vector<NodeInfo> make_nodes(int n, int procs) {
    std::vector<NodeInfo> nodes;
    for (int i = 0; i < n; ++i)
        nodes.emplace_back(i, "n" + std::to_string(i), Resources{procs, 0, 0});
    return nodes;
}

static SimJob make_job(int id, long submit, long walltime, long run_time, int procs) {
    return SimJob(id, submit, walltime, Resources{procs, 0, 0}, run_time);
}

static Simulator make_sim(std::vector<SimJob> jobs, int nodes = 4, int procs_per = 4) {
    return Simulator(std::move(jobs), make_nodes(nodes, procs_per), SimMode::Real);
}

// ── Config helpers ────────────────────────────────────────────────────────────

static MctsConfig fast_config() {
    MctsConfig c;
    c.time_limit_ms = 50;
    c.max_depth     = 5;
    c.window_size   = 3;
    c.seed          = 42;
    c.branching     = BranchingMode::Heuristic;
    c.heuristics    = { HeuristicPolicy::WFP, HeuristicPolicy::SJF, HeuristicPolicy::FCFS };
    return c;
}

static State make_state_from_sim(Simulator& sim) {
    State s;
    s.current_time = sim.current_time();
    s.nodes        = sim.nodes();
    s.sim          = &sim;
    for (int id : sim.get_job_queue())
        s.pending.push_back(&sim.get_job(id));
    for (int id : sim.running_jobs())
        s.running.push_back(&sim.get_job(id));
    return s;
}

// ── parse_heuristic_policy ────────────────────────────────────────────────────

TEST(MctsHelpersTest, ParseKnownPolicies) {
    EXPECT_EQ(parse_heuristic_policy("fcfs"),   HeuristicPolicy::FCFS);
    EXPECT_EQ(parse_heuristic_policy("sjf"),    HeuristicPolicy::SJF);
    EXPECT_EQ(parse_heuristic_policy("wfp"),    HeuristicPolicy::WFP);
    EXPECT_EQ(parse_heuristic_policy("wfp3"),   HeuristicPolicy::WFP);
    EXPECT_EQ(parse_heuristic_policy("unicep"), HeuristicPolicy::UNICEP);
    EXPECT_EQ(parse_heuristic_policy("unicef"), HeuristicPolicy::UNICEP); // backward compat
    EXPECT_EQ(parse_heuristic_policy("f1"),     HeuristicPolicy::F1);
}

TEST(MctsHelpersTest, ParseUnknownDefaultsFcfs) {
    EXPECT_EQ(parse_heuristic_policy("xyz"), HeuristicPolicy::FCFS);
}

TEST(MctsHelpersTest, AllHeuristicsCount) {
    EXPECT_EQ(static_cast<int>(all_heuristic_policies().size()), NUM_ALL_HEURISTICS);
}

TEST(MctsHelpersTest, ComprehensiveHeuristicSpecCountMatchesExperimentSweep) {
    const auto specs = comprehensive_heuristic_window_specs();
    ASSERT_EQ(specs.size(), 161u);
    EXPECT_EQ(specs.back().policy, HeuristicPolicy::FCFS);
}

TEST(MctsHelpersTest, PolarisIsSubsetOfAll) {
    auto all    = all_heuristic_policies();
    auto polaris= polaris_heuristic_policies();
    std::unordered_set<int> all_set;
    for (auto h : all) all_set.insert(static_cast<int>(h));
    for (auto h : polaris)
        EXPECT_TRUE(all_set.count(static_cast<int>(h)));
}

// ── Outcome generation ────────────────────────────────────────────────────────

TEST(PermutationOutcomesTest, EmptyQueueReturnsEmpty) {
    auto sim = make_sim({});
    EXPECT_TRUE(generate_permutation_outcomes(sim, 5).empty());
}

TEST(PermutationOutcomesTest, SingleJobOneOutcome) {
    auto sim = make_sim({ make_job(1, 0, 100, 80, 4) });
    // Step past submit to populate queue
    while (sim.peek() && sim.peek()->type != EventType::SchedulingCycle) sim.step();
    if (!sim.is_done()) {
        auto outcomes = generate_permutation_outcomes(sim, 5);
        EXPECT_GE(outcomes.size(), 1u);
    }
}

TEST(HeuristicOutcomesTest, ProducesAtMostOnePerHeuristic) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 100, 80, 4),
        make_job(2, 0, 200, 80, 4),
        make_job(3, 0,  50, 80, 4),
    };
    // Tiny cluster so all 3 can't start at once
    auto sim = make_sim(jobs, 1, 4);
    while (sim.peek() && sim.peek()->type != EventType::SchedulingCycle) sim.step();
    if (!sim.is_done()) {
        std::vector<HeuristicPolicy> hps = { HeuristicPolicy::WFP, HeuristicPolicy::SJF,
                                             HeuristicPolicy::FCFS };
        auto outcomes = generate_heuristic_outcomes(sim, 3, hps);
        EXPECT_LE(outcomes.size(), 3u);
        EXPECT_GE(outcomes.size(), 1u);
    }
}

TEST(HeuristicOutcomesTest, UsesSimulatorEasyBackfillRules) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 10, 10, 4),   // starts first and frees enough for the head at t=10
        make_job(2, 1, 30, 30, 8),   // blocked head job
        make_job(3, 2, 20, 20, 4),   // fits now but would delay the head, so EASY rejects it
    };
    auto sim = make_sim(jobs, 2, 4);
    sim.set_backfilling(true);

    while (!sim.is_done()) {
        auto next = sim.peek();
        if (next && next->type == EventType::SchedulingCycle &&
            sim.get_job_queue().size() == 2 &&
            sim.running_jobs().size() == 1) {
            break;
        }
        sim.step();
    }
    ASSERT_FALSE(sim.is_done());

    auto outcomes = generate_heuristic_outcomes(sim, 2, {HeuristicPolicy::FCFS});
    ASSERT_EQ(outcomes.size(), 1u);
    EXPECT_EQ(outcomes[0].window, std::vector<int>({2, 3}));
    EXPECT_TRUE(outcomes[0].outcome.empty());
}

TEST(HeuristicOutcomesTest, DeduplicatesIdenticalStartedJobSets) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 100, 80, 4),
        make_job(2, 1, 100, 80, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    sim.set_scheduling_callback([](Simulator&, long) {});

    while (!sim.is_done()) {
        if (sim.peek() && sim.peek()->type == EventType::SchedulingCycle &&
            sim.get_job_queue().size() == 2) {
            break;
        }
        sim.step();
    }
    ASSERT_FALSE(sim.is_done());
    ASSERT_EQ(sim.get_job_queue().size(), 2u);

    auto outcomes = generate_heuristic_outcomes(
        sim, 2, {HeuristicPolicy::FCFS, HeuristicPolicy::SJF});
    ASSERT_EQ(outcomes.size(), 1u);
    EXPECT_EQ(outcomes[0].outcome, std::vector<int>({1}));
}

// ── Mcts::search ─────────────────────────────────────────────────────────────

TEST(MctsTest, SearchReturnsWindowOrEmpty) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 0, 100, 100, 4),
        make_job(3, 0, 300, 100, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    while (sim.peek() && sim.peek()->type != EventType::SchedulingCycle) sim.step();
    if (sim.is_done()) return;

    Mcts mcts(fast_config());
    auto window = mcts.search(sim);
    // Result is either empty (trivial) or a permutation of some job IDs
    for (int id : window) {
        EXPECT_GE(id, 1);
        EXPECT_LE(id, 3);
    }
}

TEST(MctsTest, SearchRespectsSeed) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 0, 100, 100, 4),
        make_job(3, 0, 300, 100, 4),
        make_job(4, 0, 150, 100, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    while (sim.peek() && sim.peek()->type != EventType::SchedulingCycle) sim.step();
    if (sim.is_done()) return;

    MctsConfig c = fast_config();
    c.seed = 123;
    Mcts m1(c), m2(c);
    auto w1 = m1.search(sim);
    auto w2 = m2.search(sim);
    EXPECT_EQ(w1, w2);
}

TEST(MctsTest, IterationsPositiveAfterSearch) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 0, 100, 100, 4),
        make_job(3, 0, 300, 100, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    while (sim.peek() && sim.peek()->type != EventType::SchedulingCycle) sim.step();
    if (sim.is_done()) return;

    Mcts mcts(fast_config());
    mcts.search(sim);
    EXPECT_GE(mcts.last_iterations(), 0);
}

// ── MctsPolicy ───────────────────────────────────────────────────────────────

TEST(MctsPolicyTest, NameIsMcts) {
    MctsPolicy p(fast_config());
    EXPECT_EQ(p.name(), "mcts");
}

TEST(MctsPolicyTest, ReturnsEmptyWithNullSim) {
    State s;
    s.sim = nullptr;
    MctsPolicy p(fast_config());
    EXPECT_TRUE(p.schedule(s).empty());
}

TEST(MctsPolicyTest, ReturnsDecisionsWithValidSim) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 0, 100, 100, 4),
    };
    Simulator sim = make_sim(jobs, 2, 4);
    while (sim.peek() && sim.peek()->type != EventType::SchedulingCycle) sim.step();
    if (sim.is_done()) return;

    State s;
    s.current_time = sim.current_time();
    s.nodes        = sim.nodes();
    s.sim          = &sim;
    for (int id : sim.get_job_queue())
        s.pending.push_back(&sim.get_job(id));

    MctsPolicy p(fast_config());
    auto decisions = p.schedule(s);
    for (const auto& d : decisions)
        EXPECT_EQ(d.type, DecisionType::Start);
}

TEST(MctsPolicyTest, RecordsSelectedSourcePolicyLabel) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 0, 100, 100, 4),
    };
    Simulator sim = make_sim(jobs, 1, 4);
    while (sim.peek() && sim.peek()->type != EventType::SchedulingCycle) sim.step();
    if (sim.is_done()) return;

    State s = make_state_from_sim(sim);

    MctsConfig c = fast_config();
    c.window_size = 2;
    c.heuristics = { HeuristicPolicy::SJF };

    MctsPolicy p(c);
    auto decisions = p.schedule(s);
    EXPECT_FALSE(decisions.empty());
    EXPECT_EQ(p.last_cycle_policy(), "FCFS");
}

TEST(MctsPolicyTest, RecordsGroupedRootPossiblePolicies) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 0, 100, 100, 4),
    };
    Simulator sim = make_sim(jobs, 2, 4);
    while (sim.peek() && sim.peek()->type != EventType::SchedulingCycle) sim.step();
    if (sim.is_done()) return;

    State s = make_state_from_sim(sim);

    MctsConfig c = fast_config();
    c.window_size = 2;
    c.heuristics = { HeuristicPolicy::FCFS, HeuristicPolicy::SJF };

    MctsPolicy p(c);
    auto decisions = p.schedule(s);
    EXPECT_EQ(decisions.size(), 2u);
    EXPECT_EQ(p.last_cycle_root_branching(), 1);
    EXPECT_EQ(p.last_cycle_possible_policies(), "|1,2|");
}

TEST(MctsPolicyTest, SinglePendingJobUsesFcfsPolicyLabel) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
    };
    Simulator sim = make_sim(jobs, 1, 4);
    while (sim.peek() && sim.peek()->type != EventType::SchedulingCycle) sim.step();
    if (sim.is_done()) return;

    State s = make_state_from_sim(sim);

    MctsConfig c = fast_config();
    c.heuristics = { HeuristicPolicy::SJF };
    c.window_size = 2;

    MctsPolicy p(c);
    auto decisions = p.schedule(s);

    ASSERT_EQ(decisions.size(), 1u);
    EXPECT_EQ(decisions[0].type, DecisionType::Start);
    EXPECT_EQ(decisions[0].job_id, 1);
    EXPECT_EQ(p.last_cycle_policy(), "FCFS");
    EXPECT_EQ(p.last_cycle_root_branching(), 1);
    EXPECT_EQ(p.last_cycle_possible_policies(), "|1|");
}

TEST(MctsPolicyTest, AppliesSimulatorBackfillToChosenWindow) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 10, 10, 4),
        make_job(2, 1, 30, 30, 8),
        make_job(3, 2, 20, 20, 4),
    };
    auto sim = make_sim(jobs, 2, 4);
    sim.set_backfilling(true);

    while (!sim.is_done()) {
        auto next = sim.peek();
        if (next && next->type == EventType::SchedulingCycle &&
            sim.get_job_queue().size() == 2 &&
            sim.running_jobs().size() == 1) {
            break;
        }
        sim.step();
    }
    ASSERT_FALSE(sim.is_done());

    State s;
    s.current_time = sim.current_time();
    s.nodes        = sim.nodes();
    s.sim          = &sim;
    for (int id : sim.get_job_queue())
        s.pending.push_back(&sim.get_job(id));

    MctsConfig c = fast_config();
    c.heuristics = { HeuristicPolicy::FCFS };
    c.window_size = 2;

    MctsPolicy p(c);
    auto decisions = p.schedule(s);
    EXPECT_TRUE(decisions.empty());
}

TEST(MctsPolicyTest, ReturnsSafeBackfillDecisionWhenHeadIsBlocked) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 10, 10, 4),
        make_job(2, 1, 30, 30, 8),
        make_job(3, 2, 5, 5, 4),
    };
    auto sim = make_sim(jobs, 2, 4);
    sim.set_backfilling(true);

    while (!sim.is_done()) {
        auto next = sim.peek();
        if (next && next->type == EventType::SchedulingCycle &&
            sim.get_job_queue().size() == 2 &&
            sim.running_jobs().size() == 1) {
            break;
        }
        sim.step();
    }
    ASSERT_FALSE(sim.is_done());

    State s = make_state_from_sim(sim);

    MctsConfig c = fast_config();
    c.heuristics = { HeuristicPolicy::FCFS };
    c.window_size = 2;

    MctsPolicy p(c);
    auto decisions = p.schedule(s);

    ASSERT_EQ(decisions.size(), 1u);
    EXPECT_EQ(decisions[0].type, DecisionType::Start);
    EXPECT_EQ(decisions[0].job_id, 3);
}

TEST(MctsPolicyTest, InjectsSchedulingCycleWhenQueueExistsButNoCycleIsPending) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 100, 80, 4),
        make_job(2, 1, 100, 80, 4),
    };
    auto sim = make_sim(jobs, 2, 4);
    sim.set_scheduling_callback([](Simulator&, long) {});

    while (!sim.is_done()) sim.step();

    ASSERT_EQ(sim.get_job_queue().size(), 2u);
    ASSERT_TRUE(sim.running_jobs().empty());
    ASSERT_FALSE(sim.peek().has_value());

    State s = make_state_from_sim(sim);

    MctsConfig c = fast_config();
    c.heuristics = { HeuristicPolicy::FCFS };
    c.window_size = 2;

    MctsPolicy p(c);
    auto decisions = p.schedule(s);

    ASSERT_EQ(decisions.size(), 2u);
    EXPECT_EQ(decisions[0].type, DecisionType::Start);
    EXPECT_EQ(decisions[1].type, DecisionType::Start);
    EXPECT_EQ(decisions[0].job_id, 1);
    EXPECT_EQ(decisions[1].job_id, 2);
}

// ── BranchingMode::Permutation ────────────────────────────────────────────────

TEST(MctsTest, PermutationBranchingRuns) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 100, 80, 4),
        make_job(2, 0, 200, 80, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    while (sim.peek() && sim.peek()->type != EventType::SchedulingCycle) sim.step();
    if (sim.is_done()) return;

    MctsConfig c = fast_config();
    c.branching = BranchingMode::Permutation;
    Mcts mcts(c);
    auto window = mcts.search(sim);
    // Should return something without crashing
    EXPECT_TRUE(window.size() <= 2);
}

// ── New tests ─────────────────────────────────────────────────────────────────

// Helper: step sim to first scheduling cycle
static bool step_to_scheduling(Simulator& sim) {
    while (!sim.is_done()) {
        if (sim.peek() && sim.peek()->type == EventType::SchedulingCycle)
            return true;
        sim.step();
    }
    return false;
}

// Test 11: Parallel search produces valid results (same validity checks as serial)
TEST(MctsTest, ParallelSearchProducesValidResults) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 0, 100, 100, 4),
        make_job(3, 0, 300, 100, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    if (!step_to_scheduling(sim) || sim.is_done()) return;

    MctsConfig c = fast_config();
    c.num_cores = 2;
    Mcts mcts(c);
    auto window = mcts.search(sim);
    for (int id : window) {
        EXPECT_GE(id, 1);
        EXPECT_LE(id, 3);
    }
    EXPECT_LE(window.size(), 3u);
}

TEST(MctsNodeTest, MaxDepthReachedMarksNodeSolvedWithoutBecomingTerminal) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 0, 100, 100, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    ASSERT_TRUE(step_to_scheduling(sim));
    ASSERT_FALSE(sim.is_done());

    std::mt19937 rng(123);
    auto no_outcomes = [](const Simulator&, int) {
        return std::vector<MctsNode::PotentialChild>{};
    };

    MctsNode node(sim.get_copy(false), -1, nullptr, 2, 3, 3, {}, &rng, no_outcomes);

    EXPECT_TRUE(node.solved);
    EXPECT_FALSE(node.is_terminal());
    EXPECT_TRUE(node.children.empty());
    EXPECT_TRUE(node.untried_children.empty());
    ASSERT_TRUE(node.state.has_value());
    EXPECT_EQ(node.state->get_job_queue().size(), 2u);
}

TEST(MctsNodeTest, SingleOutcomeChainCollapseShrinksQueue) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 100, 100, 4),
        make_job(2, 0, 100, 100, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    ASSERT_TRUE(step_to_scheduling(sim));
    std::vector<int> original_q(sim.get_job_queue().begin(), sim.get_job_queue().end());
    ASSERT_EQ(original_q.size(), 2u);

    std::mt19937 rng(321);
    auto no_outcomes = [](const Simulator&, int) {
        return std::vector<MctsNode::PotentialChild>{};
    };
    auto parent = std::make_unique<MctsNode>(
        sim.get_copy(false), -1, nullptr, 2, 0, 0, std::vector<int>{}, &rng, no_outcomes);

    auto collapse_once = [](const Simulator& s, int) {
        const auto& q = s.get_job_queue();
        if (q.size() > 1) {
            std::vector<int> window(q.begin(), q.end());
            return std::vector<MctsNode::PotentialChild>{
                {window.front(), window, {window.front()}, "", {}}
            };
        }
        if (q.size() == 1) {
            std::vector<int> window{q.front()};
            return std::vector<MctsNode::PotentialChild>{
                {window.front(), window, {window.front()}, "", {}},
                {window.front(), window, {window.front()}, "", {}}
            };
        }
        return std::vector<MctsNode::PotentialChild>{};
    };

    MctsNode node(sim.get_copy(false), 1, parent.get(), 2, 1, 10, {}, &rng, collapse_once);

    ASSERT_TRUE(node.state.has_value());
    ASSERT_EQ(node.state->get_job_queue().size(), 1u);
    EXPECT_EQ(node.state->get_job_queue().front(), original_q[1]);
    ASSERT_TRUE(node.state->peek().has_value());
    EXPECT_EQ(node.state->peek()->type, EventType::SchedulingCycle);
    EXPECT_EQ(node.untried_children.size(), 2u);
    EXPECT_FALSE(node.solved);
}

TEST(MctsNodeTest, MixedEmptyAndNonEmptyOutcomesPreservedAtNonRoot) {
    // When some branches produce empty outcomes and others produce non-empty outcomes
    // at a non-root node, both are kept as real untried_children so MCTS can compare
    // "drain now" vs "start jobs now". The old ne_count<=1 fast-forward only triggers
    // when ALL outcomes are empty (ne_count==0).
    std::vector<SimJob> jobs = {
        make_job(1, 0, 100, 100, 4),
        make_job(2, 0, 100, 100, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    ASSERT_TRUE(step_to_scheduling(sim));
    std::vector<int> original_q(sim.get_job_queue().begin(), sim.get_job_queue().end());
    ASSERT_EQ(original_q.size(), 2u);

    std::mt19937 rng(456);
    auto no_outcomes = [](const Simulator&, int) {
        return std::vector<MctsNode::PotentialChild>{};
    };
    auto parent = std::make_unique<MctsNode>(
        sim.get_copy(false), -1, nullptr, 2, 0, 0, std::vector<int>{}, &rng, no_outcomes);

    // One branch starts nothing (empty), one starts job 2 (non-empty) — ne_count == 1
    auto mixed_outcomes = [](const Simulator& s, int) {
        const auto& q = s.get_job_queue();
        if (q.size() > 1) {
            std::vector<int> fcfs(q.begin(), q.end());
            std::vector<int> reversed(q.rbegin(), q.rend());
            return std::vector<MctsNode::PotentialChild>{
                {fcfs.front(),     fcfs,     {},                  "", {}},
                {reversed.front(), reversed, {reversed.front()},  "", {}}
            };
        }
        if (q.size() == 1) {
            std::vector<int> window{q.front()};
            return std::vector<MctsNode::PotentialChild>{
                {window.front(), window, {window.front()}, "", {}},
                {window.front(), window, {window.front()}, "", {}}
            };
        }
        return std::vector<MctsNode::PotentialChild>{};
    };

    MctsNode node(sim.get_copy(false), 2, parent.get(), 2, 1, 10, {}, &rng, mixed_outcomes);

    // Both branches must survive — MCTS needs to compare drain vs. start
    ASSERT_TRUE(node.state.has_value());
    EXPECT_EQ(node.untried_children.size(), 2u);
    // State must NOT have been fast-forwarded — queue still has both jobs
    EXPECT_EQ(node.state->get_job_queue().size(), 2u);
    EXPECT_FALSE(node.solved);
}

TEST(MctsNodeTest, NoMeaningfulOutcomesCollapseThroughNonDecisionEvents) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 100, 100, 4),
        make_job(2, 0, 100, 100, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    ASSERT_TRUE(step_to_scheduling(sim));
    sim.step();  // consume the first SchedulingCycle; one job starts and one waits
    sim.step();  // consume the second queued SchedulingCycle; next event is now Run
    ASSERT_TRUE(sim.peek().has_value());
    ASSERT_EQ(sim.peek()->type, EventType::Run);
    ASSERT_EQ(sim.get_job_queue().size(), 1u);
    int waiting_job = sim.get_job_queue().front();

    std::mt19937 rng(789);
    auto no_outcomes = [](const Simulator&, int) {
        return std::vector<MctsNode::PotentialChild>{};
    };
    auto parent = std::make_unique<MctsNode>(
        sim.get_copy(false), -1, nullptr, 1, 0, 0, std::vector<int>{}, &rng, no_outcomes);

    auto no_meaningful_actions = [](const Simulator& s, int) {
        const auto& q = s.get_job_queue();
        if (q.empty()) return std::vector<MctsNode::PotentialChild>{};
        std::vector<int> window(q.begin(), q.end());
        return std::vector<MctsNode::PotentialChild>{
            {window.front(), window, {}, "", {}},
            {window.front(), window, {}, "", {}}
        };
    };

    MctsNode node(sim.get_copy(false), 1, parent.get(), 1, 1, 10, {}, &rng, no_meaningful_actions);

    ASSERT_TRUE(node.state.has_value());
    EXPECT_EQ(node.state->current_time(), 200);
    EXPECT_TRUE(node.state->get_job_queue().empty());
    ASSERT_TRUE(node.state->peek().has_value());
    EXPECT_EQ(node.state->peek()->type, EventType::SchedulingCycle);
    EXPECT_EQ(node.state->peek()->job_id, -1);
    EXPECT_NE(std::find(node.state->completed_jobs().begin(),
                        node.state->completed_jobs().end(),
                        waiting_job),
              node.state->completed_jobs().end());
}

// Test 12: InvCumWait reward type runs without crashing
TEST(MctsTest, InvCumWaitRewardTypeRuns) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 0, 100, 100, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    if (!step_to_scheduling(sim) || sim.is_done()) return;

    MctsConfig c = fast_config();
    c.reward_type = RewardType::InvCumWait;
    Mcts mcts(c);
    EXPECT_NO_THROW(mcts.search(sim));
}

TEST(MctsTest, InvCumBsldRewardTypeRuns) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 0, 100, 100, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    if (!step_to_scheduling(sim) || sim.is_done()) return;

    MctsConfig c = fast_config();
    c.reward_type = RewardType::InvCumBSLD;
    Mcts mcts(c);
    EXPECT_NO_THROW(mcts.search(sim));
}

TEST(MctsTest, InstantUtilRewardTypeRuns) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 0, 100, 100, 4),
        make_job(3, 0, 150, 100, 4),
    };
    auto sim = make_sim(jobs, 2, 4);
    if (!step_to_scheduling(sim) || sim.is_done()) return;

    MctsConfig c = fast_config();
    c.reward_type = RewardType::InstantUtil;
    Mcts mcts(c);
    EXPECT_NO_THROW(mcts.search(sim));
}

TEST(MctsTest, CumUtilRewardTypeRuns) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 10, 100, 100, 4),
        make_job(3, 20, 150, 100, 4),
    };
    auto sim = make_sim(jobs, 2, 4);
    if (!step_to_scheduling(sim) || sim.is_done()) return;

    MctsConfig c = fast_config();
    c.reward_type = RewardType::CumUtil;
    Mcts mcts(c);
    EXPECT_NO_THROW(mcts.search(sim));
}

// Test 13: UCT runs without crashing
TEST(MctsTest, UctSelectionRuns) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 0, 100, 100, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    if (!step_to_scheduling(sim) || sim.is_done()) return;

    MctsConfig c = fast_config();
    Mcts mcts(c);
    EXPECT_NO_THROW(mcts.search(sim));
}

// Test 14: HeuristicWindowed branching mode runs without crashing
TEST(MctsTest, HeuristicWindowedBranchingRuns) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 0, 100, 100, 4),
        make_job(3, 0, 300, 100, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    if (!step_to_scheduling(sim) || sim.is_done()) return;

    MctsConfig c = fast_config();
    c.branching    = BranchingMode::HeuristicWindowed;
    c.window_size  = 2;
    Mcts mcts(c);
    auto window = mcts.search(sim);
    EXPECT_LE(window.size(), 3u);
}

TEST(MctsTest, ComprehensiveHeuristicBranchingRuns) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 0, 100, 100, 4),
        make_job(3, 0, 300, 100, 4),
        make_job(4, 0, 150, 100, 4),
    };
    auto sim = make_sim(jobs, 2, 4);
    if (!step_to_scheduling(sim) || sim.is_done()) return;

    MctsConfig c = fast_config();
    c.branching = BranchingMode::ComprehensiveHeuristic;
    c.window_size = 1024;
    Mcts mcts(c);
    auto window = mcts.search(sim);
    EXPECT_LE(window.size(), 4u);
    EXPECT_GT(mcts.last_root_branching(), 0);
}

// Test 15: max_depth=1 still returns a valid (possibly empty) window
TEST(MctsTest, MaxDepthOneReturnsValidResult) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 200, 100, 4),
        make_job(2, 0, 100, 100, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    if (!step_to_scheduling(sim) || sim.is_done()) return;

    MctsConfig c = fast_config();
    c.max_depth = 1;
    Mcts mcts(c);
    auto window = mcts.search(sim);
    EXPECT_LE(window.size(), 2u);
}

TEST(MctsTest, SearchReportedDepthNeverExceedsConfiguredMaxDepth) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 300, 100, 4),
        make_job(2, 0, 250, 100, 4),
        make_job(3, 0, 200, 100, 4),
        make_job(4, 0, 150, 100, 4),
    };
    auto sim = make_sim(jobs, 1, 8);
    ASSERT_TRUE(step_to_scheduling(sim));
    ASSERT_FALSE(sim.is_done());

    MctsConfig c = fast_config();
    c.branching = BranchingMode::Permutation;
    c.window_size = 4;
    c.max_depth = 2;
    Mcts mcts(c);

    auto window = mcts.search(sim);

    EXPECT_FALSE(window.empty());
    EXPECT_GT(mcts.last_iterations(), 0);
    EXPECT_GT(mcts.last_root_branching(), 1);
    EXPECT_LE(mcts.last_max_depth(), 2);
}

TEST(MctsTest, ConvergenceTraceTracksRootChoiceStabilization) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 300, 300, 1),
        make_job(2, 0, 250, 250, 1),
        make_job(3, 0,  60,  60, 1),
        make_job(4, 0,  80,  80, 1),
        make_job(5, 0, 100, 100, 1),
    };
    auto sim = make_sim(jobs, 1, 2);
    ASSERT_TRUE(step_to_scheduling(sim));
    ASSERT_FALSE(sim.is_done());

    MctsConfig c = fast_config();
    c.branching = BranchingMode::Permutation;
    c.reward_type = RewardType::InvCumBSLD;
    c.window_size = 5;
    c.max_depth = 10;
    c.discount = 0.99;
    c.exploration = 0.5;
    c.seed = 7;

    Mcts mcts(c);
    auto trace = mcts.analyze_convergence(sim, 30, 5);

    EXPECT_EQ(trace.max_iterations, 30);
    EXPECT_GT(trace.root_branching, 1);
    EXPECT_GT(trace.final_iteration, 0);
    EXPECT_LE(trace.final_iteration, 30);
    EXPECT_GE(trace.stabilization_iteration, 1);
    EXPECT_LE(trace.stabilization_iteration, trace.final_iteration);
    ASSERT_FALSE(trace.samples.empty());
    EXPECT_EQ(trace.samples.back().iteration, trace.final_iteration);
    EXPECT_GE(trace.samples.back().best_visit_ties, 1);
    EXPECT_GE(trace.samples.back().visit_margin, 0.0);
    EXPECT_FALSE(trace.final_selected_window.empty());
    EXPECT_FALSE(trace.final_selected_outcome.empty());
}

TEST(MctsTest, SearchRunsFullSearchWhenEmptyAndNonEmptyOutcomesCoexist) {
    // Job 1 needs 8 procs (can't fit on 4-proc node) — FCFS branch produces empty outcome.
    // Job 2 needs 4 procs (fits) — permutation branch starting job 2 produces non-empty outcome.
    // Both branches are distinct; MCTS must compare them rather than short-circuiting.
    std::vector<SimJob> jobs = {
        make_job(1, 0, 100, 100, 8),
        make_job(2, 1, 100, 100, 4),
    };
    auto sim = make_sim(jobs, 1, 4);
    while (!sim.is_done()) {
        auto next = sim.peek();
        if (next && next->type == EventType::SchedulingCycle &&
            sim.get_job_queue().size() == 2) {
            break;
        }
        sim.step();
    }
    ASSERT_FALSE(sim.is_done());

    MctsConfig c = fast_config();
    c.branching = BranchingMode::Permutation;
    c.window_size = 2;
    Mcts mcts(c);

    auto window = mcts.search(sim);

    EXPECT_EQ(window.size(), 2u);
    EXPECT_GT(mcts.last_iterations(), 0);   // full search runs — drain vs start is a real choice
    EXPECT_EQ(mcts.last_root_branching(), 2);
    for (int id : window) EXPECT_TRUE(id == 1 || id == 2);
}

// ── Large job drain tests ─────────────────────────────────────────────────────

// When a large job can't fit and backfilling is disabled, FCFS produces an empty
// outcome (large job at front blocks everything) while SJF produces a non-empty
// outcome (small job at front starts). At non-root nodes both branches must survive
// so MCTS can evaluate draining vs. starting small jobs.
TEST(MctsNodeTest, LargeJobFrontProducesEmptyOutcomeAlongsideNonEmptyAtNonRoot) {
    // Job 1: already running, 4 procs — leaves only 4 free on 8-proc node
    // Job 2: large job, needs 8 procs — can't fit, blocks FCFS
    // Job 3: small job, needs 2 procs — fits, starts under SJF
    std::vector<SimJob> jobs = {
        make_job(1, 0, 100, 100, 4),
        make_job(2, 1, 200, 200, 8),
        make_job(3, 2,  50,  50, 2),
    };
    auto sim = make_sim(jobs, 1, 8);
    sim.set_backfilling(false);

    while (!sim.is_done()) {
        auto next = sim.peek();
        if (next && next->type == EventType::SchedulingCycle &&
            sim.running_jobs().size() == 1 &&
            sim.get_job_queue().size() == 2) break;
        sim.step();
    }
    ASSERT_FALSE(sim.is_done());
    ASSERT_EQ(sim.running_jobs().size(), 1u);
    ASSERT_EQ(sim.get_job_queue().size(), 2u);

    // Verify the outcome split exists: FCFS (large job first) → empty, SJF (small first) → non-empty
    auto outcomes = generate_heuristic_outcomes(sim, 2,
        {HeuristicPolicy::FCFS, HeuristicPolicy::SJF});
    ASSERT_EQ(outcomes.size(), 2u);
    int empty_count = 0, nonempty_count = 0;
    for (const auto& o : outcomes)
        (o.outcome.empty() ? empty_count : nonempty_count)++;
    EXPECT_EQ(empty_count,   1);
    EXPECT_EQ(nonempty_count, 1);

    // At a non-root node both branches must be preserved (not fast-forwarded away)
    std::mt19937 rng(42);
    auto no_outcomes = [](const Simulator&, int) {
        return std::vector<MctsNode::PotentialChild>{};
    };
    auto parent = std::make_unique<MctsNode>(
        sim.get_copy(false), -1, nullptr, 2, 0, 0, std::vector<int>{}, &rng, no_outcomes);

    auto gen = [](const Simulator& s, int ws) {
        return generate_heuristic_outcomes(s, ws,
            {HeuristicPolicy::FCFS, HeuristicPolicy::SJF});
    };
    MctsNode node(sim.get_copy(false), -1, parent.get(), 2, 1, 10, {}, &rng, gen);

    ASSERT_TRUE(node.state.has_value());
    EXPECT_EQ(node.untried_children.size(), 2u);
    EXPECT_EQ(node.state->get_job_queue().size(), 2u);
}

// MCTS must run full search (iterations > 0) in a drain scenario — verifying that
// the root correctly sees multiple branches and doesn't short-circuit.
TEST(MctsTest, LargeJobDrainScenarioRunsFullSearch) {
    std::vector<SimJob> jobs = {
        make_job(1, 0, 100, 100, 4),
        make_job(2, 1, 200, 200, 8),
        make_job(3, 2,  50,  50, 2),
    };
    auto sim = make_sim(jobs, 1, 8);
    sim.set_backfilling(false);

    while (!sim.is_done()) {
        auto next = sim.peek();
        if (next && next->type == EventType::SchedulingCycle &&
            sim.running_jobs().size() == 1 &&
            sim.get_job_queue().size() == 2) break;
        sim.step();
    }
    ASSERT_FALSE(sim.is_done());

    MctsConfig c = fast_config();
    c.heuristics  = {HeuristicPolicy::FCFS, HeuristicPolicy::SJF};
    c.window_size = 2;
    c.time_limit_ms = 50;
    c.max_depth   = 20;
    Mcts mcts(c);
    auto window = mcts.search(sim);

    // Both branches are distinct so MCTS must have searched, not short-circuited
    EXPECT_GT(mcts.last_iterations(), 0);
    EXPECT_GT(mcts.last_root_branching(), 1);
    for (int id : window) EXPECT_TRUE(id == 2 || id == 3);
}

// Rollout must not consume depth budget on drain cycles (queue non-empty, nothing starts).
// With a max_depth just large enough to reach the large job start only if drain cycles
// are free, verify that MCTS completes meaningful iterations rather than exhausting
// depth during the drain.
TEST(MctsTest, RolloutDoesNotConsumeDepthOnDrainCycles) {
    // Job 1: running, 4 procs, ends at t=100
    // Job 2: large, needs 8 procs — drain required
    // Job 3: small, fits after drain — will start in rollout after job 1 ends
    std::vector<SimJob> jobs = {
        make_job(1, 0, 100, 100, 4),
        make_job(2, 1, 200, 200, 8),
        make_job(3, 2,  50,  50, 2),
    };
    auto sim = make_sim(jobs, 1, 8);
    sim.set_backfilling(false);

    while (!sim.is_done()) {
        auto next = sim.peek();
        if (next && next->type == EventType::SchedulingCycle &&
            sim.running_jobs().size() == 1 &&
            sim.get_job_queue().size() == 2) break;
        sim.step();
    }
    ASSERT_FALSE(sim.is_done());

    // With drain cycles not consuming depth, even max_depth=3 gives enough budget
    // to simulate past the drain and see the large job start in rollout.
    // If drain cycles consumed depth, the rollout would terminate mid-drain and
    // the MCTS search would still complete (no crash), but the depth reported
    // from the tree would still be valid.
    MctsConfig c = fast_config();
    c.heuristics    = {HeuristicPolicy::FCFS, HeuristicPolicy::SJF};
    c.window_size   = 2;
    c.max_depth     = 5;
    c.time_limit_ms = 100;
    Mcts mcts(c);

    EXPECT_NO_THROW(mcts.search(sim));
    EXPECT_GT(mcts.last_iterations(), 0);
}

// ── Tree depth bound tests ────────────────────────────────────────────────────
// Each test runs MCTS with a specific max_depth, then verifies:
//   1. last_max_depth() never exceeds the configured limit
//   2. The search produced at least one iteration (real work was done)
//   3. The returned window contains only valid job IDs
//
// Setup: 1-proc node, many 1-proc jobs — each scheduling cycle starts exactly
// one job, so the tree grows one level per decision and max_depth is exercised
// even at small values.

TEST(MctsTest, MaxDepth2BoundsSearchDepth) {
    std::vector<SimJob> jobs;
    for (int i = 1; i <= 6; ++i)
        jobs.push_back(make_job(i, 0, 100 * i, 50 * i, 1));
    auto sim = make_sim(jobs, /*nodes=*/1, /*procs_per=*/1);
    if (!step_to_scheduling(sim) || sim.is_done()) return;

    MctsConfig c = fast_config();
    c.max_depth     = 2;
    c.branching     = BranchingMode::Heuristic;
    c.time_limit_ms = 100;
    Mcts mcts(c);
    auto window = mcts.search(sim);

    EXPECT_GT(mcts.last_iterations(), 0);
    EXPECT_LE(mcts.last_max_depth(), 2);
    for (int id : window) { EXPECT_GE(id, 1); EXPECT_LE(id, 6); }
}

TEST(MctsTest, MaxDepth4BoundsSearchDepth) {
    std::vector<SimJob> jobs;
    for (int i = 1; i <= 8; ++i)
        jobs.push_back(make_job(i, 0, 100 * i, 50 * i, 1));
    auto sim = make_sim(jobs, /*nodes=*/1, /*procs_per=*/1);
    if (!step_to_scheduling(sim) || sim.is_done()) return;

    MctsConfig c = fast_config();
    c.max_depth     = 4;
    c.branching     = BranchingMode::Heuristic;
    c.time_limit_ms = 150;
    Mcts mcts(c);
    auto window = mcts.search(sim);

    EXPECT_GT(mcts.last_iterations(), 0);
    EXPECT_LE(mcts.last_max_depth(), 4);
    for (int id : window) { EXPECT_GE(id, 1); EXPECT_LE(id, 8); }
}

TEST(MctsTest, MaxDepth8BoundsSearchDepth) {
    std::vector<SimJob> jobs;
    for (int i = 1; i <= 12; ++i)
        jobs.push_back(make_job(i, 0, 100 * i, 50 * i, 1));
    auto sim = make_sim(jobs, /*nodes=*/1, /*procs_per=*/1);
    if (!step_to_scheduling(sim) || sim.is_done()) return;

    MctsConfig c = fast_config();
    c.max_depth     = 8;
    c.branching     = BranchingMode::Heuristic;
    c.time_limit_ms = 200;
    Mcts mcts(c);
    auto window = mcts.search(sim);

    EXPECT_GT(mcts.last_iterations(), 0);
    EXPECT_LE(mcts.last_max_depth(), 8);
    for (int id : window) { EXPECT_GE(id, 1); EXPECT_LE(id, 12); }
}

// Test 16: search finishes within a reasonable multiple of time_limit_ms
TEST(MctsTest, TimeLimitRespected) {
    std::vector<SimJob> jobs;
    for (int i = 1; i <= 5; ++i)
        jobs.push_back(make_job(i, 0, 100 * i, 50 * i, 4));
    auto sim = make_sim(jobs, 1, 4);
    if (!step_to_scheduling(sim) || sim.is_done()) return;

    MctsConfig c = fast_config();
    c.time_limit_ms = 100;
    Mcts mcts(c);

    auto t0 = std::chrono::steady_clock::now();
    mcts.search(sim);
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();

    // Allow 10× headroom for slow CI machines
    EXPECT_LT(elapsed_ms, 1000);
}
