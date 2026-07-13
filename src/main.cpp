#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "io/swf_reader.h"
#include "mcts/mcts.h"
#include "experiment/json_experiment.h"

struct ConvergenceCliOptions {
    int jobs = 12;
    int capacity = 6;
    int window = 12;
    int iterations = 5000;
    int sample_every = 100;
    int max_depth = 50;
    double exploration = 0.8;
    double discount = 0.99;
    unsigned int seed = 42;
    std::string output = "results/mcts_convergence.csv";
    RewardType reward = RewardType::InvCumBSLD;
};

static std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static RewardType parse_convergence_reward(const std::string& s) {
    std::string v = lower_copy(s);
    if (v == "invcumwait") return RewardType::InvCumWait;
    if (v == "invcumbsld") return RewardType::InvCumBSLD;
    if (v == "instantutil" || v == "instutil" || v == "instant_util")
        return RewardType::InstantUtil;
    if (v == "cumutil" || v == "cumulativeutil" || v == "cumulative_util")
        return RewardType::CumUtil;
    throw std::runtime_error("unknown reward: " + s);
}

static std::string csv_escape(const std::string& value) {
    std::string escaped = "\"";
    for (char c : value) {
        if (c == '"') escaped += "\"\"";
        else escaped += c;
    }
    escaped += "\"";
    return escaped;
}

static std::string join_ids(std::vector<int> ids, bool sort_ids = false) {
    if (ids.empty()) return "NONE";
    if (sort_ids) std::sort(ids.begin(), ids.end());

    std::string out;
    for (int id : ids) {
        if (!out.empty()) out += "|";
        out += std::to_string(id);
    }
    return out;
}

static void print_convergence_usage() {
    std::cerr
        << "Usage:\n"
        << "  cqsimcpp mcts-convergence [options]\n\n"
        << "Options:\n"
        << "  --jobs N             Number of same-time synthetic jobs (default: 12)\n"
        << "  --capacity N         Processors available; each job uses 1 (default: 6)\n"
        << "  --window N           MCTS permutation window (default: jobs)\n"
        << "  --iterations N       Rollout iterations to trace (default: 5000)\n"
        << "  --sample-every N     CSV sampling interval; changes are always recorded (default: 100)\n"
        << "  --seed N             RNG seed (default: 42)\n"
        << "  --max-depth N        MCTS rollout depth (default: 50)\n"
        << "  --exploration X      UCT exploration parameter (default: 0.8)\n"
        << "  --discount X         Reward discount (default: 0.99)\n"
        << "  --reward NAME        invcumwait|invcumbsld|instantutil|cumutil (default: invcumbsld)\n"
        << "  --output PATH        CSV output path (default: results/mcts_convergence.csv)\n";
}

static ConvergenceCliOptions parse_convergence_options(int argc, char* argv[]) {
    ConvergenceCliOptions opts;
    opts.window = opts.jobs;
    bool window_explicit = false;

    auto require_value = [&](int& i, const std::string& arg) -> std::string {
        if (i + 1 >= argc)
            throw std::runtime_error("missing value for " + arg);
        return argv[++i];
    };

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_convergence_usage();
            std::exit(0);
        } else if (arg == "--jobs") {
            opts.jobs = std::stoi(require_value(i, arg));
            if (!window_explicit) opts.window = opts.jobs;
        } else if (arg == "--capacity") {
            opts.capacity = std::stoi(require_value(i, arg));
        } else if (arg == "--window") {
            opts.window = std::stoi(require_value(i, arg));
            window_explicit = true;
        } else if (arg == "--iterations") {
            opts.iterations = std::stoi(require_value(i, arg));
        } else if (arg == "--sample-every") {
            opts.sample_every = std::stoi(require_value(i, arg));
        } else if (arg == "--seed") {
            opts.seed = static_cast<unsigned int>(std::stoul(require_value(i, arg)));
        } else if (arg == "--max-depth") {
            opts.max_depth = std::stoi(require_value(i, arg));
        } else if (arg == "--exploration") {
            opts.exploration = std::stod(require_value(i, arg));
        } else if (arg == "--discount") {
            opts.discount = std::stod(require_value(i, arg));
        } else if (arg == "--reward") {
            opts.reward = parse_convergence_reward(require_value(i, arg));
        } else if (arg == "--output") {
            opts.output = require_value(i, arg);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (opts.jobs <= 1) throw std::runtime_error("--jobs must be > 1");
    if (opts.capacity <= 0) throw std::runtime_error("--capacity must be > 0");
    if (opts.capacity >= opts.jobs)
        throw std::runtime_error("--capacity must be smaller than --jobs for a high-branching decision");
    if (opts.window <= 0) opts.window = opts.jobs;
    if (opts.iterations <= 0) throw std::runtime_error("--iterations must be > 0");
    if (opts.sample_every <= 0) throw std::runtime_error("--sample-every must be > 0");
    if (opts.max_depth <= 0) throw std::runtime_error("--max-depth must be > 0");

    return opts;
}

static Simulator make_high_branching_sim(const ConvergenceCliOptions& opts) {
    std::vector<SimJob> jobs;
    jobs.reserve(opts.jobs);

    for (int i = 1; i <= opts.jobs; ++i) {
        long walltime = i <= opts.capacity
            ? 900 + 45L * i       // long jobs at the front make FCFS intentionally weak
            : 60 + 15L * (i - opts.capacity);
        jobs.emplace_back(i, 0, walltime, Resources{1, 0, 0}, walltime);
    }

    std::vector<NodeInfo> nodes;
    nodes.emplace_back(0, "synthetic", Resources{opts.capacity, 0, 0});

    Simulator sim(std::move(jobs), std::move(nodes), SimMode::Prediction);
    while (!sim.is_done()) {
        auto next = sim.peek();
        if (next && next->type == EventType::SchedulingCycle) break;
        sim.step();
    }
    return sim;
}

static void write_convergence_csv(const std::string& path,
                                  const MctsConvergenceTrace& trace) {
    namespace fs = std::filesystem;
    fs::path out(path);
    if (!out.parent_path().empty())
        fs::create_directories(out.parent_path());

    std::ofstream csv(path, std::ios::trunc);
    if (!csv.is_open())
        throw std::runtime_error("cannot write convergence CSV: " + path);

    csv << "iteration,root_branching_factor,expanded_children,"
        << "selected_choice,selected_window,selected_policy,"
        << "selected_visits,total_root_visits,selected_visit_pct,"
        << "selected_avg_reward,best_visit_ties,visit_margin,changed\n";

    for (const auto& s : trace.samples) {
        csv << s.iteration << ","
            << s.root_branching << ","
            << s.expanded_children << ","
            << csv_escape(join_ids(s.selected_outcome, true)) << ","
            << csv_escape(join_ids(s.selected_window, false)) << ","
            << csv_escape(s.selected_policy) << ","
            << std::fixed << std::setprecision(6)
            << s.selected_visits << ","
            << s.total_root_visits << ","
            << s.selected_visit_pct << ","
            << s.selected_avg_reward << ","
            << s.best_visit_ties << ","
            << s.visit_margin << ","
            << (s.changed ? "1" : "0") << "\n";
    }
}

static int run_mcts_convergence(int argc, char* argv[]) {
    ConvergenceCliOptions opts = parse_convergence_options(argc, argv);
    Simulator sim = make_high_branching_sim(opts);

    MctsConfig cfg;
    cfg.branching = BranchingMode::Permutation;
    cfg.reward_type = opts.reward;
    cfg.window_size = opts.window;
    cfg.max_depth = opts.max_depth;
    cfg.exploration = opts.exploration;
    cfg.discount = opts.discount;
    cfg.seed = opts.seed;

    Mcts mcts(cfg);
    MctsConvergenceTrace trace =
        mcts.analyze_convergence(sim, opts.iterations, opts.sample_every);

    write_convergence_csv(opts.output, trace);
    const MctsConvergenceSample* final_sample =
        trace.samples.empty() ? nullptr : &trace.samples.back();

    std::cout << "Synthetic MCTS convergence trace\n"
              << "  jobs=" << opts.jobs
              << " capacity=" << opts.capacity
              << " window=" << opts.window
              << " iterations=" << opts.iterations << "\n"
              << "  root_branching_factor=" << trace.root_branching << "\n"
              << "  final_iteration=" << trace.final_iteration << "\n"
              << "  stabilization_iteration=" << trace.stabilization_iteration << "\n"
              << "  final_choice=" << join_ids(trace.final_selected_outcome, true) << "\n"
              << "  final_window=" << join_ids(trace.final_selected_window, false) << "\n"
              << "  final_best_visit_ties="
              << (final_sample ? final_sample->best_visit_ties : 0) << "\n"
              << "  final_visit_margin="
              << (final_sample ? final_sample->visit_margin : 0.0) << "\n"
              << "  samples_written=" << trace.samples.size() << "\n"
              << "  csv=" << opts.output << "\n"
              << "\nNote: stabilization is measured within this iteration budget; "
              << "rerun with a larger --iterations value to test longer horizons.\n";

    return 0;
}

struct TraceSpec {
    std::string label;
    std::string path;
};

struct RealTraceConvergenceOptions {
    std::vector<TraceSpec> traces;
    std::string output_dir = "results/real_trace_mcts_convergence";
    int max_jobs = std::numeric_limits<int>::max();
    int start_job_index = 0;
    int min_queue = 16;
    int top_k = 16;
    int branch_threshold = 100;
    int iterations = 5000;
    int sample_every = 100;
    int window_size = 1024;
    int max_depth = 50;
    int num_workers = 1;
    double exploration = 10000.0;
    double discount = 0.99;
    unsigned int seed = 17;
    bool backfilling = false;
    RewardType reward = RewardType::InvCumWait;
    BranchingMode branching = BranchingMode::ComprehensiveHeuristic;
};

struct RealTraceCandidate {
    std::string trace_label;
    std::string trace_path;
    int cycle = 0;
    long sim_time = 0;
    int queue_len = 0;
    int available_procs = 0;
    int running_jobs = 0;
    int root_branching = 0;
    Simulator snapshot;

    RealTraceCandidate(std::string label, std::string path, int cycle_in,
                       long sim_time_in, int queue_len_in, int available_procs_in,
                       int running_jobs_in, int root_branching_in, Simulator snapshot_in)
        : trace_label(std::move(label)),
          trace_path(std::move(path)),
          cycle(cycle_in),
          sim_time(sim_time_in),
          queue_len(queue_len_in),
          available_procs(available_procs_in),
          running_jobs(running_jobs_in),
          root_branching(root_branching_in),
          snapshot(std::move(snapshot_in)) {}
};

struct RealTraceRunSummary {
    std::string trace_label;
    int rank = 0;
    int cycle = 0;
    long sim_time = 0;
    int queue_len = 0;
    int available_procs = 0;
    int running_jobs = 0;
    int root_branching = 0;
    int final_iteration = 0;
    int stabilization_iteration = -1;
    std::string final_choice;
    int final_best_visit_ties = 0;
    double final_visit_margin = 0.0;
    bool converged_unique = false;
    std::string csv_path;
};

static BranchingMode parse_branching_mode(const std::string& s) {
    std::string v = lower_copy(s);
    if (v == "permutation") return BranchingMode::Permutation;
    if (v == "heuristic") return BranchingMode::Heuristic;
    if (v == "heuristic_windowed") return BranchingMode::HeuristicWindowed;
    if (v == "comprehensive" || v == "comprehensive_heuristic")
        return BranchingMode::ComprehensiveHeuristic;
    throw std::runtime_error("unknown branching mode: " + s);
}

static std::string branching_mode_tag(BranchingMode mode) {
    switch (mode) {
        case BranchingMode::Permutation: return "permutation";
        case BranchingMode::Heuristic: return "heuristic";
        case BranchingMode::HeuristicWindowed: return "heuristic_windowed";
        case BranchingMode::ComprehensiveHeuristic: return "comprehensive_heuristic";
    }
    return "unknown";
}

static std::string path_stem_label(const std::string& path) {
    std::filesystem::path p(path);
    std::string stem = p.stem().string();
    return stem.empty() ? "trace" : stem;
}

static std::string sanitize_filename(std::string value) {
    for (char& c : value) {
        bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
        if (!ok) c = '_';
    }
    return value.empty() ? "trace" : value;
}

static TraceSpec parse_trace_spec(const std::string& value) {
    size_t eq = value.find('=');
    if (eq == std::string::npos) {
        size_t colon = value.find(':');
        if (colon != std::string::npos) eq = colon;
    }
    if (eq == std::string::npos)
        return {path_stem_label(value), value};

    std::string label = value.substr(0, eq);
    std::string path = value.substr(eq + 1);
    if (label.empty() || path.empty())
        throw std::runtime_error("trace must be label=path or path: " + value);
    return {label, path};
}

static void print_real_trace_convergence_usage() {
    std::cerr
        << "Usage:\n"
        << "  cqsimcpp mcts-trace-convergence [options]\n\n"
        << "Defaults scan data/polaris24cln.swf and data/theta21cln.swf with FCFS,\n"
        << "rank high-branching MCTS snapshots, then run convergence traces.\n\n"
        << "Options:\n"
        << "  --trace LABEL=PATH       Add a SWF trace; may be repeated\n"
        << "  --output-dir DIR         Output directory (default: results/real_trace_mcts_convergence)\n"
        << "  --max-jobs N             Limit each trace after sorting by submit time\n"
        << "  --start-job-index N      Start offset in sorted trace (default: 0)\n"
        << "  --min-queue N            Only inspect cycles with at least N queued jobs (default: 16)\n"
        << "  --top-k N                Keep N high-branching points per trace (default: 16)\n"
        << "  --branch-threshold N     Minimum root branching factor to keep (default: 100)\n"
        << "  --iterations N           Convergence iterations per selected point (default: 5000)\n"
        << "  --sample-every N         CSV sampling interval; changes are always recorded (default: 100)\n"
        << "  --num-workers N          Parallel convergence workers (default: 1)\n"
        << "  --window N               MCTS window size (default: 1024)\n"
        << "  --max-depth N            MCTS depth (default: 50)\n"
        << "  --exploration X          MCTS exploration parameter (default: 10000)\n"
        << "  --discount X             MCTS discount (default: 0.99)\n"
        << "  --seed N                 Base seed (default: 17)\n"
        << "  --reward NAME            invcumwait|invcumbsld|instantutil|cumutil (default: invcumwait)\n"
        << "  --branching NAME         comprehensive_heuristic|heuristic|permutation (default: comprehensive_heuristic)\n"
        << "  --backfilling            Replay FCFS with EASY backfilling while scanning\n"
        << "  --no-backfilling         Replay strict FCFS while scanning (default)\n";
}

static RealTraceConvergenceOptions parse_real_trace_convergence_options(
    int argc, char* argv[]) {

    RealTraceConvergenceOptions opts;
    opts.traces = {
        {"polaris", "data/polaris24cln.swf"},
        {"theta",   "data/theta21cln.swf"}
    };
    bool traces_explicit = false;

    auto require_value = [&](int& i, const std::string& arg) -> std::string {
        if (i + 1 >= argc)
            throw std::runtime_error("missing value for " + arg);
        return argv[++i];
    };

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_real_trace_convergence_usage();
            std::exit(0);
        } else if (arg == "--trace") {
            if (!traces_explicit) {
                opts.traces.clear();
                traces_explicit = true;
            }
            opts.traces.push_back(parse_trace_spec(require_value(i, arg)));
        } else if (arg == "--output-dir") {
            opts.output_dir = require_value(i, arg);
        } else if (arg == "--max-jobs") {
            opts.max_jobs = std::stoi(require_value(i, arg));
        } else if (arg == "--start-job-index") {
            opts.start_job_index = std::stoi(require_value(i, arg));
        } else if (arg == "--min-queue") {
            opts.min_queue = std::stoi(require_value(i, arg));
        } else if (arg == "--top-k") {
            opts.top_k = std::stoi(require_value(i, arg));
        } else if (arg == "--branch-threshold") {
            opts.branch_threshold = std::stoi(require_value(i, arg));
        } else if (arg == "--iterations") {
            opts.iterations = std::stoi(require_value(i, arg));
        } else if (arg == "--sample-every") {
            opts.sample_every = std::stoi(require_value(i, arg));
        } else if (arg == "--num-workers") {
            opts.num_workers = std::stoi(require_value(i, arg));
        } else if (arg == "--window") {
            opts.window_size = std::stoi(require_value(i, arg));
        } else if (arg == "--max-depth") {
            opts.max_depth = std::stoi(require_value(i, arg));
        } else if (arg == "--exploration") {
            opts.exploration = std::stod(require_value(i, arg));
        } else if (arg == "--discount") {
            opts.discount = std::stod(require_value(i, arg));
        } else if (arg == "--seed") {
            opts.seed = static_cast<unsigned int>(std::stoul(require_value(i, arg)));
        } else if (arg == "--reward") {
            opts.reward = parse_convergence_reward(require_value(i, arg));
        } else if (arg == "--branching") {
            opts.branching = parse_branching_mode(require_value(i, arg));
        } else if (arg == "--backfilling") {
            opts.backfilling = true;
        } else if (arg == "--no-backfilling") {
            opts.backfilling = false;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (opts.traces.empty()) throw std::runtime_error("at least one --trace is required");
    if (opts.max_jobs <= 0) throw std::runtime_error("--max-jobs must be > 0");
    if (opts.start_job_index < 0) throw std::runtime_error("--start-job-index must be >= 0");
    if (opts.min_queue < 0) throw std::runtime_error("--min-queue must be >= 0");
    if (opts.top_k <= 0) throw std::runtime_error("--top-k must be > 0");
    if (opts.branch_threshold < 0) throw std::runtime_error("--branch-threshold must be >= 0");
    if (opts.iterations <= 0) throw std::runtime_error("--iterations must be > 0");
    if (opts.sample_every <= 0) throw std::runtime_error("--sample-every must be > 0");
    if (opts.num_workers <= 0) throw std::runtime_error("--num-workers must be > 0");
    if (opts.window_size <= 0) throw std::runtime_error("--window must be > 0");
    if (opts.max_depth <= 0) throw std::runtime_error("--max-depth must be > 0");

    return opts;
}

static MctsConfig make_real_trace_mcts_config(const RealTraceConvergenceOptions& opts,
                                              unsigned int seed) {
    MctsConfig cfg;
    cfg.branching = opts.branching;
    cfg.reward_type = opts.reward;
    cfg.window_size = opts.window_size;
    cfg.max_depth = opts.max_depth;
    cfg.exploration = opts.exploration;
    cfg.discount = opts.discount;
    cfg.seed = seed;
    cfg.dynamic_exploration = true;
    return cfg;
}

static std::vector<SimJob> select_trace_jobs(std::vector<SimJob> jobs,
                                             int start_job_index,
                                             int max_jobs) {
    std::sort(jobs.begin(), jobs.end(), [](const SimJob& a, const SimJob& b) {
        return a.submit_time != b.submit_time ? a.submit_time < b.submit_time : a.id < b.id;
    });

    int start = std::min(start_job_index, static_cast<int>(jobs.size()));
    int end = std::min(static_cast<int>(jobs.size()), start + max_jobs);
    return std::vector<SimJob>(jobs.begin() + start, jobs.begin() + end);
}

static int infer_total_procs(int reader_max_procs, const std::vector<SimJob>& jobs) {
    if (reader_max_procs > 0) return reader_max_procs;
    int max_seen = 1;
    for (const auto& job : jobs)
        max_seen = std::max(max_seen, job.requested.procs);
    return max_seen;
}

static void keep_high_branching_candidate(std::vector<RealTraceCandidate>& candidates,
                                          RealTraceCandidate candidate,
                                          int top_k) {
    if (static_cast<int>(candidates.size()) < top_k) {
        candidates.push_back(std::move(candidate));
        return;
    }

    auto weakest = std::min_element(
        candidates.begin(), candidates.end(),
        [](const RealTraceCandidate& a, const RealTraceCandidate& b) {
            if (a.root_branching != b.root_branching)
                return a.root_branching < b.root_branching;
            return a.queue_len < b.queue_len;
        });

    if (weakest != candidates.end() &&
        (candidate.root_branching > weakest->root_branching ||
         (candidate.root_branching == weakest->root_branching &&
          candidate.queue_len > weakest->queue_len))) {
        *weakest = std::move(candidate);
    }
}

static std::vector<RealTraceCandidate> scan_trace_for_high_branching(
    const TraceSpec& trace,
    const RealTraceConvergenceOptions& opts) {

    SwfReader reader(trace.path);
    std::vector<SimJob> jobs = reader.read();
    std::vector<SimJob> selected_jobs =
        select_trace_jobs(std::move(jobs), opts.start_job_index, opts.max_jobs);
    int total_procs = infer_total_procs(reader.max_procs(), selected_jobs);

    std::vector<NodeInfo> nodes;
    nodes.emplace_back(0, trace.label + "_cluster", Resources{total_procs, 0, 0});

    Simulator sim(std::move(selected_jobs), std::move(nodes), SimMode::Real);
    sim.set_backfilling(opts.backfilling);

    MctsConfig probe_cfg = make_real_trace_mcts_config(opts, opts.seed);
    std::vector<RealTraceCandidate> candidates;
    int cycle = 0;
    int inspected = 0;

    while (!sim.is_done()) {
        auto next = sim.peek();
        if (!next) break;

        if (next->type == EventType::SchedulingCycle) {
            cycle++;
            int q_len = static_cast<int>(sim.get_job_queue().size());

            if (q_len >= opts.min_queue) {
                Mcts probe(probe_cfg);
                MctsRootAnalysis root = probe.inspect_root(sim);
                inspected++;

                if (root.branching >= opts.branch_threshold) {
                    keep_high_branching_candidate(
                        candidates,
                        RealTraceCandidate(
                            trace.label, trace.path, cycle, next->time, q_len,
                            sim.available_procs(),
                            static_cast<int>(sim.running_jobs().size()),
                            root.branching, sim.get_copy(false)),
                        opts.top_k);
                }
            }
        }

        sim.step();
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const RealTraceCandidate& a, const RealTraceCandidate& b) {
                  if (a.root_branching != b.root_branching)
                      return a.root_branching > b.root_branching;
                  if (a.queue_len != b.queue_len)
                      return a.queue_len > b.queue_len;
                  return a.sim_time < b.sim_time;
              });

    std::cout << "Scanned " << trace.label
              << " cycles=" << cycle
              << " inspected=" << inspected
              << " kept=" << candidates.size()
              << " total_procs=" << total_procs << "\n";

    return candidates;
}

static void write_real_trace_summary(
    const std::string& path,
    const std::vector<RealTraceRunSummary>& summaries) {

    std::ofstream csv(path, std::ios::trunc);
    if (!csv.is_open())
        throw std::runtime_error("cannot write summary CSV: " + path);

    csv << "trace,rank,cycle,sim_time,queue_len,available_procs,running_jobs,"
        << "root_branching_factor,final_iteration,stabilization_iteration,"
        << "final_choice,final_best_visit_ties,final_visit_margin,"
        << "converged_unique,csv_path\n";

    for (const auto& s : summaries) {
        csv << csv_escape(s.trace_label) << ","
            << s.rank << ","
            << s.cycle << ","
            << s.sim_time << ","
            << s.queue_len << ","
            << s.available_procs << ","
            << s.running_jobs << ","
            << s.root_branching << ","
            << s.final_iteration << ","
            << s.stabilization_iteration << ","
            << csv_escape(s.final_choice) << ","
            << s.final_best_visit_ties << ","
            << std::fixed << std::setprecision(6) << s.final_visit_margin << ","
            << (s.converged_unique ? "1" : "0") << ","
            << csv_escape(s.csv_path) << "\n";
    }
}

static RealTraceRunSummary run_candidate_convergence(
    const RealTraceCandidate& candidate,
    const RealTraceConvergenceOptions& opts,
    int rank,
    int global_index) {

    MctsConfig cfg = make_real_trace_mcts_config(
        opts, opts.seed + static_cast<unsigned int>(global_index + 1));
    Mcts mcts(cfg);
    MctsConvergenceTrace trace =
        mcts.analyze_convergence(candidate.snapshot, opts.iterations, opts.sample_every);

    std::string label = sanitize_filename(candidate.trace_label);
    std::string filename = label
        + "_rank" + std::to_string(rank)
        + "_cycle" + std::to_string(candidate.cycle)
        + "_b" + std::to_string(candidate.root_branching)
        + ".csv";
    std::string csv_path = opts.output_dir + "/" + filename;
    write_convergence_csv(csv_path, trace);

    const MctsConvergenceSample* final_sample =
        trace.samples.empty() ? nullptr : &trace.samples.back();

    RealTraceRunSummary summary;
    summary.trace_label = candidate.trace_label;
    summary.rank = rank;
    summary.cycle = candidate.cycle;
    summary.sim_time = candidate.sim_time;
    summary.queue_len = candidate.queue_len;
    summary.available_procs = candidate.available_procs;
    summary.running_jobs = candidate.running_jobs;
    summary.root_branching = candidate.root_branching;
    summary.final_iteration = trace.final_iteration;
    summary.stabilization_iteration = trace.stabilization_iteration;
    summary.final_choice = join_ids(trace.final_selected_outcome, true);
    summary.final_best_visit_ties = final_sample ? final_sample->best_visit_ties : 0;
    summary.final_visit_margin = final_sample ? final_sample->visit_margin : 0.0;
    summary.converged_unique = summary.final_best_visit_ties == 1;
    summary.csv_path = csv_path;
    return summary;
}

static int run_mcts_trace_convergence(int argc, char* argv[]) {
    RealTraceConvergenceOptions opts =
        parse_real_trace_convergence_options(argc, argv);

    std::filesystem::create_directories(opts.output_dir);

    std::cout << "Real-trace MCTS convergence scan\n"
              << "  branching=" << branching_mode_tag(opts.branching)
              << " window=" << opts.window_size
              << " threshold=" << opts.branch_threshold
              << " top_k_per_trace=" << opts.top_k
              << " iterations=" << opts.iterations
              << " workers=" << opts.num_workers
              << " fcfs_backfilling=" << (opts.backfilling ? "true" : "false")
              << "\n";

    std::vector<RealTraceCandidate> all_candidates;
    for (const auto& trace : opts.traces) {
        auto candidates = scan_trace_for_high_branching(trace, opts);
        for (auto& candidate : candidates)
            all_candidates.push_back(std::move(candidate));
    }

    if (all_candidates.empty()) {
        std::cout << "No high-branching points met the threshold.\n";
        return 0;
    }

    std::vector<int> ranks(all_candidates.size(), 1);
    for (size_t i = 0; i < all_candidates.size(); ++i) {
        int rank = 1;
        for (size_t j = 0; j < i; ++j)
            if (all_candidates[j].trace_label == all_candidates[i].trace_label)
                rank++;
        ranks[i] = rank;
    }

    std::vector<RealTraceRunSummary> summaries(all_candidates.size());

    #ifdef _OPENMP
    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(opts.num_workers) schedule(dynamic)
    #endif
    for (int i = 0; i < static_cast<int>(all_candidates.size()); ++i) {
        summaries[i] = run_candidate_convergence(
            all_candidates[i], opts, ranks[i], i);
        #ifdef _OPENMP
        #pragma omp critical
        #endif
        {
            std::cout << "Finished " << summaries[i].trace_label
                      << " rank=" << summaries[i].rank
                      << " branching=" << summaries[i].root_branching
                      << " stabilization=" << summaries[i].stabilization_iteration
                      << " ties=" << summaries[i].final_best_visit_ties << "\n";
        }
    }

    std::string summary_path = opts.output_dir + "/summary.csv";
    write_real_trace_summary(summary_path, summaries);

    std::cout << "Wrote " << summary_path << "\n";
    return 0;
}

int main(int argc, char* argv[]) {
    auto print_usage = []() {
        std::cerr << "Usage:\n"
                  << "  cqsimcpp <config.json>\n"
                  << "  cqsimcpp run config <config.json>\n"
                  << "  cqsimcpp list\n"
                  << "  cqsimcpp mcts-convergence [options]\n"
                  << "  cqsimcpp mcts-trace-convergence [options]\n";
    };

    if (argc < 2) {
        print_usage();
        return 1;
    }

    if (argc == 2 && std::string(argv[1]) == "list") {
        namespace fs = std::filesystem;
        fs::path experiments_dir("experiments");
        std::vector<std::string> names;

        if (fs::exists(experiments_dir) && fs::is_directory(experiments_dir)) {
            for (const auto& entry : fs::directory_iterator(experiments_dir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".json")
                    names.push_back(entry.path().stem().string());
            }
        }

        std::sort(names.begin(), names.end());
        for (const auto& name : names)
            std::cout << name << "\n";
        return 0;
    }

    if (argc >= 2 && std::string(argv[1]) == "mcts-convergence") {
        try {
            return run_mcts_convergence(argc, argv);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n\n";
            print_convergence_usage();
            return 1;
        }
    }

    if (argc >= 2 && std::string(argv[1]) == "mcts-trace-convergence") {
        try {
            return run_mcts_trace_convergence(argc, argv);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n\n";
            print_real_trace_convergence_usage();
            return 1;
        }
    }

    std::string config_path;
    if (argc == 2) {
        config_path = argv[1];
    } else if (argc == 4 &&
               std::string(argv[1]) == "run" &&
               std::string(argv[2]) == "config") {
        config_path = argv[3];
    } else {
        print_usage();
        return 1;
    }

    try {
        JsonExperiment exp(config_path);
        exp.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
