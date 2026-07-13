#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <climits>

#include "simulator.h"
#include "../../core/policy.h"

struct DriverConfig {
    std::string tag;
    std::string output_dir;
    int         max_jobs  = INT_MAX;   // cap on how many jobs to submit
    int         start_job = 0;         // index into sorted job list to start from
    SimMode     mode      = SimMode::Real;
    bool        backfill  = false;
    std::vector<MaintenanceWindow> maintenance;
};

class Driver {
public:
    Driver(std::vector<SimJob>    jobs,
           std::vector<NodeInfo>  nodes,
           SchedulerPolicy*       policy,
           DriverConfig           config);

    // Run the simulation to completion
    void run();

    // Results (valid after run())
    double avg_wait_time()  const { return avg_wait_time_; }
    long   makespan()       const { return makespan_; }
    int    jobs_completed() const;

    // Access the underlying simulator (e.g. for MCTS to snapshot state)
    Simulator&       simulator()       { return sim_; }
    const Simulator& simulator() const { return sim_; }

    const std::string&   tag()    const { return config_.tag; }
    const DriverConfig& config() const { return config_; }

private:
    // ── Just-in-time submit insertion ────────────────────────────────────────
    // Keeps the event queue small by inserting Submit events one at a time
    // rather than pre-loading all of them at construction.
    std::optional<std::pair<long,int>> next_submit_event();
    void insert_pending_submits(std::optional<Event> next_sim_event,
                                std::optional<std::pair<long,int>>& pending);

    // ── Output ───────────────────────────────────────────────────────────────
    void open_output_files();
    void log_event(const Event& e);
    void log_cycle(int cycle, long sim_time,
                   int jobs_finished, int queue_len, double util,
                   int jobs_running, double cycle_ms,
                   const std::string& selected_policy,
                   int jobs_run,
                   int root_branching_factor,
                   const std::string& possible_policies,
                   int mcts_iterations = 0);
    void print_progress();

    // ── Post-run ─────────────────────────────────────────────────────────────
    void compute_stats();

    // ── State ────────────────────────────────────────────────────────────────
    DriverConfig     config_;
    std::string      run_dir_;       // output_dir/tag/

    std::vector<SimJob> jobs_sorted_;  // sorted by submit_time
    int  submit_cursor_ = 0;           // next index to submit
    int  submit_limit_  = 0;           // absolute index limit
    bool has_pending_   = true;        // false once all jobs submitted

    Simulator        sim_;
    SchedulerPolicy* policy_;          // not owned — caller keeps it alive

    int  cycle_number_          = 0;
    int  total_jobs_            = 0;
    int  cycle_primary_started_ = 0;
    int  cycle_backfill_started_= 0;

    // Output files
    std::ofstream perf_file_;
    std::ofstream event_file_;
    std::ofstream descision_file_;

    // Post-run stats
    double avg_wait_time_ = 0.0;
    long   makespan_      = 0;
};
