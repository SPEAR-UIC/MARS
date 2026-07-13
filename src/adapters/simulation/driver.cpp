#include "driver.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <limits>
#include <unordered_set>

static std::string csv_escape(const std::string& value) {
    std::string escaped = "\"";
    for (char c : value) {
        if (c == '"') escaped += "\"\"";
        else escaped += c;
    }
    escaped += "\"";
    return escaped;
}

// ── Constructor ───────────────────────────────────────────────────────────────

Driver::Driver(std::vector<SimJob>   jobs,
               std::vector<NodeInfo> nodes,
               SchedulerPolicy*      policy,
               DriverConfig          config)
    : config_(std::move(config)),
      run_dir_(config_.output_dir + "/" + config_.tag),
      sim_(std::vector<SimJob>{}, std::move(nodes), config_.mode, run_dir_),
      policy_(policy)
{
    // Sort jobs by submit_time for deterministic JIT insertion
    jobs_sorted_ = std::move(jobs);
    std::sort(jobs_sorted_.begin(), jobs_sorted_.end(),
              [](const SimJob& a, const SimJob& b) {
                  return a.submit_time != b.submit_time
                       ? a.submit_time < b.submit_time
                       : a.id < b.id;
              });

    int start  = std::max(0, std::min(config_.start_job,
                                      static_cast<int>(jobs_sorted_.size())));
    int remain = static_cast<int>(jobs_sorted_.size()) - start;
    submit_cursor_ = start;
    submit_limit_  = start + std::min(config_.max_jobs, remain);
    total_jobs_    = submit_limit_ - start;

    sim_.set_mode(config_.mode);
    sim_.set_backfilling(config_.backfill);

    if (!config_.maintenance.empty())
        sim_.add_maintenance_windows(config_.maintenance);

    // Wire the policy as the scheduling callback
    if (policy_) {
        sim_.set_scheduling_callback([this](Simulator& s, long t) {
            State state;
            state.current_time = t;
            state.nodes        = s.nodes();
            state.sim          = &s;

            for (int id : s.get_job_queue()) {
                state.pending.push_back(&s.get_job(id));
            }
            for (int id : s.running_jobs()) {
                state.running.push_back(&s.get_job(id));
            }

            auto decisions = policy_->schedule(state);

            std::unordered_set<int> actually_started;
            cycle_primary_started_  = 0;
            cycle_backfill_started_ = 0;
            for (const auto& d : decisions) {
                if (d.type == DecisionType::Start && s.try_start_job(d.job_id, t, d.is_backfill)) {
                    actually_started.insert(d.job_id);
                    if (d.is_backfill) cycle_backfill_started_++;
                    else               cycle_primary_started_++;
                }
            }

            // Remove only jobs that were actually started from wait queue
            auto q = s.get_job_queue();
            std::deque<int> remaining;
            for (int id : q) {
                if (!actually_started.count(id))
                    remaining.push_back(id);
            }
            s.sort_job_queue(remaining);
        });
    }
}

// ── run ───────────────────────────────────────────────────────────────────────

void Driver::run() {
    std::filesystem::create_directories(run_dir_);
    open_output_files();

    auto run_start = std::chrono::steady_clock::now();

    // Seed first submit
    auto pending = next_submit_event();
    if (pending) {
        sim_.create_submit_event(pending->first, pending->second);
        Event e = sim_.step();
        log_event(e);
        pending = next_submit_event();
    }

    int jobs_run_this_cycle = 0;

    while (true) {
        auto next_ev = sim_.peek();

        insert_pending_submits(next_ev, pending);

        if (!has_pending_ && !sim_.has_events()) break;

        next_ev = sim_.peek();
        if (!next_ev) break;

        if (next_ev->type == EventType::SchedulingCycle) {
            int free_before    = sim_.available_procs();
            int q_len          = static_cast<int>(sim_.get_job_queue().size());
            int running_before = static_cast<int>(sim_.running_jobs().size());
            int finished_before = static_cast<int>(sim_.completed_jobs().size());
            int total_p        = sim_.total_procs();
            double util        = total_p > 0
                ? static_cast<double>(free_before) / total_p
                : 0.0;
            long sim_time      = sim_.current_time();
            auto t0            = std::chrono::steady_clock::now();

            Event e = sim_.step();  // fires the callback
            log_event(e);

            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();

            int q_after = static_cast<int>(sim_.get_job_queue().size());
            int jobs_scheduled = q_len - q_after;
            int root_branching = policy_ ? policy_->last_cycle_root_branching() : 0;
            int mcts_iterations = policy_ ? policy_->last_cycle_iterations() : 0;
            std::string possible_policies =
                policy_ ? policy_->last_cycle_possible_policies() : "";

            bool is_mcts = policy_ && policy_->name() == "mcts";
            std::string selected_policy;
            if (is_mcts) {
                if (q_len == 0)               selected_policy = "NONE";
                else if (q_len == 1)          selected_policy = "FCFS";
                else if (jobs_scheduled == 0) selected_policy = "DRAIN";
                else                          selected_policy = "MCTS";
                if (jobs_scheduled == 0 && q_len > 1) {
                    bool had_runnable = std::any_of(possible_policies.begin(),
                                                    possible_policies.end(), ::isdigit);
                    selected_policy = had_runnable ? "DRAIN" : "NONE";
                }
            } else {
                std::string pname = policy_ ? policy_->name() : "unknown";
                std::transform(pname.begin(), pname.end(), pname.begin(), ::toupper);
                if (q_len == 0) {
                    selected_policy = "NONE";
                } else if (jobs_scheduled == 0) {
                    selected_policy = "DRAIN";
                } else if (cycle_primary_started_ > 0 && cycle_backfill_started_ > 0) {
                    selected_policy = pname + "+BACKFILL";
                } else if (cycle_primary_started_ == 0) {
                    selected_policy = "BACKFILL";
                } else {
                    selected_policy = pname;
                }
            }
            log_cycle(++cycle_number_, sim_time,
                      finished_before, q_len, util,
                      running_before, ms,
                      selected_policy, jobs_scheduled,
                      root_branching, possible_policies, mcts_iterations);
            jobs_run_this_cycle = 0;

        } else {
            Event e = sim_.step();
            log_event(e);
            if (e.type == EventType::Run || e.type == EventType::Backfill)
                jobs_run_this_cycle++;
            if (e.type == EventType::End)
                print_progress();
        }
    }

    // Final progress line
    {
        double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - run_start).count();
        print_progress();
        std::cout << " done (" << std::fixed << std::setprecision(2) << secs << "s)\n";
    }

    compute_stats();

    perf_file_.flush();
    event_file_.flush();
    if (descision_file_.is_open()) descision_file_.flush();
}

// ── Stats ─────────────────────────────────────────────────────────────────────

int Driver::jobs_completed() const {
    return static_cast<int>(sim_.completed_jobs().size());
}

void Driver::compute_stats() {
    const auto& completed = sim_.completed_jobs();
    if (completed.empty()) return;

    long total_wait     = 0;
    long earliest_sub   = std::numeric_limits<long>::max();

    for (int id : completed) {
        const SimJob& j = sim_.get_job(id);
        long start = sim_.job_start_time(id);
        long wait = (start >= 0) ? start - j.submit_time : 0;
        total_wait   += wait;
        earliest_sub  = std::min(earliest_sub, j.submit_time);
    }

    makespan_       = sim_.current_time() - earliest_sub;
    avg_wait_time_  = static_cast<double>(total_wait) / completed.size();
}

// ── JIT submit insertion ──────────────────────────────────────────────────────

std::optional<std::pair<long,int>> Driver::next_submit_event() {
    if (submit_cursor_ >= submit_limit_) {
        has_pending_ = false;
        return std::nullopt;
    }
    const SimJob& j = jobs_sorted_[submit_cursor_++];
    // Register job data in the Simulator before we create the Submit event;
    // the scheduling callback calls get_job() which needs it in jobs_ map.
    sim_.register_job(j);
    return std::make_pair(j.submit_time, j.id);
}

void Driver::insert_pending_submits(std::optional<Event>                  next_ev,
                                    std::optional<std::pair<long,int>>&   pending)
{
    if (!has_pending_ || !pending) return;

    if (!next_ev) {
        sim_.create_submit_event(pending->first, pending->second);
        pending = next_submit_event();
        log_event(sim_.step());
        return;
    }

    while (pending && pending->first <= next_ev->time) {
        sim_.create_submit_event(pending->first, pending->second);
        pending = next_submit_event();
        log_event(sim_.step());
        next_ev = sim_.peek();
        if (!next_ev) break;
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

void Driver::open_output_files() {
    perf_file_.open(run_dir_ + "/performance.csv", std::ios::trunc);
    perf_file_ << "cycle,sim_time,jobs_finished,queue_len,util,"
                  "jobs_running,cycle_ms,selected_policy,jobs_run,mcts_iterations\n";

    event_file_.open(run_dir_ + "/events.csv", std::ios::trunc);
    event_file_ << "sim_time,event,id\n";

    if (policy_) {
        descision_file_.open(run_dir_ + "/descisions.csv", std::ios::trunc);
        descision_file_ << "cycle,root_branching_factor,selected_policy,possible_job_sets\n";
    }
}

void Driver::log_event(const Event& e) {
    if (!event_file_.is_open()) return;
    event_file_ << e.time << "," << e.type_str() << ",";
    if (e.type == EventType::SchedulingCycle)
        event_file_ << cycle_number_;
    else
        event_file_ << e.job_id;
    event_file_ << "\n";
}

void Driver::log_cycle(int cycle, long sim_time,
                        int jobs_finished, int q_len, double util,
                        int jobs_running, double ms,
                        const std::string& selected_policy,
                        int jobs_run,
                        int root_branching_factor,
                        const std::string& possible_policies,
                        int mcts_iterations) {
    if (!perf_file_.is_open()) return;
    perf_file_ << cycle << ","
               << sim_time << ","
               << jobs_finished << ","
               << q_len << ","
               << std::fixed << std::setprecision(4) << util << ","
               << jobs_running << ","
               << ms << ","
               << selected_policy << ","
               << jobs_run << ","
               << mcts_iterations << "\n";

    if (!descision_file_.is_open()) return;
    descision_file_ << cycle << ","
                    << root_branching_factor << ","
                    << selected_policy << ","
                    << csv_escape(possible_policies) << "\n";
}

void Driver::print_progress() {
    int done  = jobs_completed();
    int pct   = total_jobs_ > 0 ? (done * 100 / total_jobs_) : 0;
    int filled = 30 * done / std::max(total_jobs_, 1);

    std::cout << "\r" << std::left << std::setw(14) << config_.tag << " [";
    for (int i = 0; i < 30; ++i)
        std::cout << (i < filled ? '#' : '.');
    std::cout << "] " << std::right << std::setw(3) << pct
              << "% (" << done << "/" << total_jobs_ << ")" << std::flush;
}
