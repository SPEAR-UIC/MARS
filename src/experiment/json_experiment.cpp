#include "json_experiment.h"
#include "../io/swf_reader.h"
#include "../io/maintenance_reader.h"
#include "../mcts/mcts.h"
#include "../policies/fifo.h"
#include "../policies/sjf.h"
#include "../policies/ljf.h"
#include "../policies/srf.h"
#include "../policies/lrf.h"
#include "../policies/scf.h"
#include "../policies/lcf.h"
#include "../policies/lcfs.h"
#include "../policies/wfp.h"
#include "../policies/wfp1.h"
#include "../policies/fcsj.h"
#include "../policies/fat.h"
#include "../policies/unicep.h"
#include "../policies/f1.h"
#include "../policies/f2.h"
#include "../policies/f3.h"
#include "../policies/f4.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>

using json = nlohmann::json;

// ── Helpers ───────────────────────────────────────────────────────────────────

static RewardType parse_reward_type(const std::string& s) {
    if (s == "invcumwait")                 return RewardType::InvCumWait;
    if (s == "invcumbsld")
        return RewardType::InvCumBSLD;
    if (s == "instantutil" || s == "instutil" || s == "instant_util")
        return RewardType::InstantUtil;
    if (s == "cumutil" || s == "cumulativeutil" || s == "cumulative_util")
        return RewardType::CumUtil;
    if (s == "invcump99wait" || s == "p99wait")
        return RewardType::InvCumP99Wait;
    if (s == "negcumwait" || s == "neg_cum_wait")
        return RewardType::NegCumWait;
    if (s == "negp99wait" || s == "neg_p99_wait")
        return RewardType::NegP99Wait;
    if (s == "constant" || s == "const")
        return RewardType::Constant;
    if (s == "invcump99maxwait" || s == "p99maxwait")
        return RewardType::InvCumP99MaxWait;
    if (s == "expwait" || s == "exp_wait")
        return RewardType::ExpWait;
    std::cerr << "Warning: unknown reward_type '" << s
              << "', defaulting to invcumwait\n";
    return RewardType::InvCumWait;
}

static std::unique_ptr<SchedulerPolicy>
make_heuristic_policy(const std::string& type, int window) {
    if (type == "fcfs")                          return std::make_unique<FifoPolicy>();
    if (type == "sjf")                           return std::make_unique<SjfPolicy>(window);
    if (type == "ljf")                           return std::make_unique<LjfPolicy>(window);
    if (type == "srf")                           return std::make_unique<SrfPolicy>(window);
    if (type == "lrf")                           return std::make_unique<LrfPolicy>(window);
    if (type == "scf")                           return std::make_unique<ScfPolicy>(window);
    if (type == "lcf")                           return std::make_unique<LcfPolicy>(window);
    if (type == "lcfs")                          return std::make_unique<LcfsPolicy>(window);
    if (type == "wfp" || type == "wfp3")         return std::make_unique<WfpPolicy>(window);
    if (type == "wfp1")                          return std::make_unique<Wfp1Policy>(window);
    if (type == "fcsj")                          return std::make_unique<FcsjPolicy>(window);
    if (type == "fat")                           return std::make_unique<FatPolicy>(window);
    if (type == "unicep" || type == "unicef")    return std::make_unique<UnicepPolicy>(window);
    if (type == "f1")                            return std::make_unique<F1Policy>(window);
    if (type == "f2")                            return std::make_unique<F2Policy>(window);
    if (type == "f3")                            return std::make_unique<F3Policy>(window);
    if (type == "f4")                            return std::make_unique<F4Policy>(window);
    return nullptr;
}

static std::vector<NodeInfo> make_nodes_for_procs(int total_procs) {
    return { NodeInfo(0, "cluster", Resources{total_procs, 0, 0}) };
}

// ── JsonExperiment ────────────────────────────────────────────────────────────

JsonExperiment::JsonExperiment(const std::string& json_path)
    : Experiment("JSON experiment: " + json_path),
      json_path_(json_path)
{
    // Read output_dir eagerly so it's available before run()
    std::ifstream f(json_path_);
    if (f.is_open()) {
        auto cfg = json::parse(f);
        if (cfg.contains("output_dir"))
            output_dir_ = cfg["output_dir"].get<std::string>();
    }
}

void JsonExperiment::input_initialization() {
    std::ifstream f(json_path_);
    if (!f.is_open())
        throw std::runtime_error("Cannot open config: " + json_path_);
    auto cfg = json::parse(f);

    std::string swf_path = cfg.at("swf_path").get<std::string>();

    if (cfg.contains("output_dir"))
        output_dir_ = cfg["output_dir"].get<std::string>();
    if (cfg.contains("max_submits"))
        max_submits_ = cfg["max_submits"].get<int>();
    if (cfg.contains("start_job_index"))
        start_job_index_ = cfg["start_job_index"].get<int>();
    if (cfg.contains("mode"))
        mode_ = (cfg["mode"].get<std::string>() == "prediction")
                  ? SimMode::Prediction : SimMode::Real;
    if (cfg.contains("size_bins"))
        for (const auto& v : cfg["size_bins"])
            size_bins_.push_back(v.get<double>());
    if (cfg.contains("walltime_bins"))
        for (const auto& v : cfg["walltime_bins"])
            walltime_bins_.push_back(v.get<double>());

    SwfReader reader(swf_path);
    jobs_        = reader.read();
    total_procs_ = reader.max_procs();

    if (cfg.contains("maintenance_csv")) {
        std::string mcsv     = cfg["maintenance_csv"].get<std::string>();
        double notice_hours  = cfg.value("maintenance_notice_hours", 48.0);
        MaintenanceReader mr(mcsv, notice_hours);
        maintenance_ = mr.read();
        log("loaded " + std::to_string(maintenance_.size())
            + " maintenance windows from " + mcsv);
    }

    log("swf: " + swf_path + "  jobs=" + std::to_string(jobs_.size())
        + "  total_procs=" + std::to_string(total_procs_));
}

void JsonExperiment::driver_initialization() {
    std::ifstream f(json_path_);
    auto cfg = json::parse(f);

    int max_sub = std::min(max_submits_, static_cast<int>(jobs_.size()));

    for (const auto& d : cfg.at("drivers")) {
        std::string type    = d.at("type").get<std::string>();
        std::string tag     = d.value("tag", type);
        bool backfill       = d.value("backfilling", false);
        int window          = d.value("window_size", std::numeric_limits<int>::max());
        bool parallel_exec  = d.value("parallel", false);

        DriverConfig dc;
        dc.tag         = tag;
        dc.output_dir  = output_dir_;
        dc.max_jobs    = max_sub;
        dc.start_job   = start_job_index_;
        dc.mode        = mode_;
        dc.backfill    = backfill;
        dc.maintenance = maintenance_;

        auto nodes = make_nodes_for_procs(total_procs_);

        if (type == "rlscheduler") continue;

        std::unique_ptr<SchedulerPolicy> policy;
        bool is_mcts   = (type == "mcts");
        bool is_random = (type == "random" || type == "random_heuristic");

        if (!is_mcts) {
            if (is_random) {
                unsigned seed = d.value("seed", 0u);
                policy = std::make_unique<RandomHeuristicPolicy>(window, seed);
            } else {
                policy = make_heuristic_policy(type, window);
                if (!policy) {
                    std::cerr << "Warning: unknown driver type '" << type << "', skipping\n";
                    continue;
                }
            }
        } else {
            MctsConfig mc;
            mc.time_limit_ms     = d.value("time_limit_ms",  15000L);
            mc.max_depth         = d.value("max_depth",       1000);
            mc.window_size       = window;
            mc.reward_type       = parse_reward_type(d.value("reward_type", "invcumwait"));
            mc.exploration       = d.value("exploration",     10000.0);
            mc.discount          = d.value("discount",        1.0);
            mc.num_cores         = d.value("num_cores",       1);
            mc.seed              = d.value("seed",            0u);
            mc.dynamic_exploration = d.value("dynamic_exploration", false);
            mc.avg_lambda        = d.value("avg_lambda",      1.0);
            mc.p99_lambda        = d.value("p99_lambda",      1.0);
            mc.max_lambda        = d.value("max_lambda",      1.0);
            mc.exp_scale         = d.value("exp_scale",       1.0);
            mc.reward_scale      = d.value("reward_scale",    1.0);

            std::string branch_str = d.value("branching", "heuristic");
            if (branch_str == "permutation")
                mc.branching = BranchingMode::Permutation;
            else if (branch_str == "comprehensive_heuristic")
                mc.branching = BranchingMode::ComprehensiveHeuristic;
            else if (branch_str == "heuristic_windowed")
                mc.branching = BranchingMode::HeuristicWindowed;
            else
                mc.branching = BranchingMode::Heuristic;

            mc.rollout_policy = parse_heuristic_policy(
                d.value("rollout_policy", "wfp"));

            if (d.contains("heuristics")) {
                mc.heuristics.clear();
                for (const auto& h : d["heuristics"])
                    mc.heuristics.push_back(parse_heuristic_policy(h.get<std::string>()));
            }
            if (d.contains("heuristic_windows")) {
                const auto& hw = d["heuristic_windows"];
                if (hw.is_array()) {
                    for (const auto& pair : hw)
                        mc.heuristic_windows.push_back(
                            {parse_heuristic_policy(pair[0].get<std::string>()),
                             pair[1].get<int>()});
                } else {
                    for (auto& [key, val] : hw.items())
                        mc.heuristic_windows.push_back(
                            {parse_heuristic_policy(key), val.get<int>()});
                }
            }

            if (d.value("log_file", false)) {
                mc.log_file = output_dir_ + "/" + tag + "_mcts.log";
            }

            policy = std::make_unique<MctsPolicy>(mc);

            if (parallel_exec && mc.num_cores > 1) {
                // Wrap with parallel_search variant via a custom policy class.
                // For now, MctsPolicy internally dispatches to parallel_search
                // when num_cores > 1 — see mcts.cpp MctsPolicy::schedule().
            }
        }

        // policies_ owns the policy objects; drivers reference them by pointer
        policies_.push_back(std::move(policy));
        auto driver = std::make_unique<Driver>(jobs_, nodes, policies_.back().get(), dc);
        log("created driver: " + tag + " (type=" + type + ")");
        register_driver(std::move(driver));
    }
}

void JsonExperiment::output_analytics() {
    write_metrics(total_procs_, jobs_);

    std::cout << "\n=== Experiment Results ===\n";
    std::cout << std::left;
    std::cout << "  " << std::setw(20) << "Driver"
              << std::setw(15) << "Completed"
              << std::setw(15) << "Makespan(s)"
              << std::setw(12) << "AvgWait(s)" << "\n";
    std::cout << "  " << std::string(62, '-') << "\n";
    for (const auto& dp : drivers()) {
        std::cout << "  " << std::setw(20) << dp->tag()
                  << std::setw(15) << dp->jobs_completed()
                  << std::setw(15) << dp->makespan()
                  << std::setw(12) << std::fixed << std::setprecision(1)
                  << dp->avg_wait_time() << "\n";
    }
    std::cout << "\nSummary written to " << output_dir_ << "/summary.csv\n";
}
