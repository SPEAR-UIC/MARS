#pragma once

#include <vector>
#include <deque>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <optional>
#include <memory>
#include <stdexcept>
#include <string>

#include "event.h"
#include "maintenance.h"
#include "../../core/job.h"
#include "../../core/node.h"

enum class SimMode {
    Real,        // jobs end at run_time  (ground truth)
    Prediction   // jobs end at walltime  (scheduler's view)
};

class Simulator {
public:
    // ── Construction ─────────────────────────────────────────────────────────
    Simulator(std::vector<SimJob> jobs,
              std::vector<NodeInfo> nodes,
              SimMode mode,
              const std::string& log_dir = "");

    // ── Copy / Move ──────────────────────────────────────────────────────────
    // get_copy: creates a snapshot for MCTS tree nodes.
    // enable_logging=false for rollout copies (cheaper, no file I/O).
    Simulator get_copy(bool enable_logging = true) const;
    Simulator(Simulator&&) noexcept = default;
    Simulator& operator=(Simulator&&) noexcept = default;

    // ── Event injection ──────────────────────────────────────────────────────
    void create_submit_event(long time, int job_id);

    // Register job data without creating a Submit event.
    // Used by Driver for JIT submission: job metadata must be available before
    // the corresponding Submit event fires and the scheduling callback runs.
    void register_job(const SimJob& job);

    // Inject a SchedulingCycle event at the current time.
    // Used by MctsPolicy when called from inside a scheduling callback where
    // the SchedulingCycle has already been consumed from the event queue.
    void inject_scheduling_cycle();

    // ── Event stepping ───────────────────────────────────────────────────────
    Event                  step();
    std::optional<Event>   peek()          const;
    std::vector<Event>     kpeek(int k)    const;
    bool                   has_events()    const;
    bool                   is_done()       const;

    // ── Job queue access / reordering ────────────────────────────────────────
    // MCTS reads and rewrites the queue directly before calling step().
    const std::deque<int>& get_job_queue()                          const;
    void                   sort_job_queue(const std::deque<int>& new_order);

    // ── Scheduling callback ──────────────────────────────────────────────────
    // Called on every SchedulingCycle event.
    // Signature: void fn(Simulator& sim, long current_time)
    // Leave unset for MCTS rollout copies — MCTS drives the queue directly.
    using SchedulingCycleFn = std::function<void(Simulator&, long)>;
    void set_scheduling_callback(SchedulingCycleFn fn);

    // ── Simulator configuration ──────────────────────────────────────────────
    void set_mode(SimMode mode);
    void set_backfilling(bool enabled);
    void set_logging(bool enabled);
    bool is_logging_enabled() const;

    // ── Maintenance ──────────────────────────────────────────────────────────
    void add_maintenance_windows(const std::vector<MaintenanceWindow>& windows);
    long next_maintenance_start() const;
    bool maintenance_active()     const { return maintenance_active_; }

    // ── Helpers exposed for scheduling callbacks and MCTS ────────────────────
    bool            try_start_job(int job_id, long current_time, bool is_backfill = false);
    int             available_procs()            const;
    int             total_procs()                const;
    const SimJob&   get_job(int job_id)          const;
    long            job_start_time(int job_id)   const;
    long            current_time()               const;
    SimMode         mode()                       const;
    int             sim_id()                     const;
    long            job_duration_for(int job_id) const;

    // ── Backfill helpers ─────────────────────────────────────────────────────
    // `reserved_procs` lets a caller in the middle of building up a batch of
    // decisions (e.g. schedule_with_backfill_reservation) account for procs
    // it has already provisionally committed this cycle but that aren't yet
    // reflected in running_jobs_/end_times_ (those only update once the
    // batch is applied). Defaults to 0 for callers that just want the true
    // committed state.
    long earliest_start_time(int procs_needed, int reserved_procs = 0) const;
    long get_job_end_time(int job_id)             const;
    int  calculate_free_procs_at(long time, int reserved_procs = 0) const;

    // ── Stats ────────────────────────────────────────────────────────────────
    const std::vector<int>& completed_jobs() const;
    const std::vector<int>& running_jobs()   const;

    // ── Node access ──────────────────────────────────────────────────────────
    const std::vector<NodeInfo>& nodes() const;

private:
    // ── Event handlers ───────────────────────────────────────────────────────
    void handle_submit(const Event& e);
    void handle_resubmit(const Event& e);
    void handle_run(const Event& e);
    void handle_end(const Event& e);
    void handle_scheduling_cycle(const Event& e);
    void handle_sma(const Event& e);
    void handle_sms(const Event& e);
    void handle_sme(const Event& e);
    void handle_ums(const Event& e);
    void handle_ume(const Event& e);

    // ── Default FIFO scheduling ──────────────────────────────────────────────
    void default_scheduling_cycle(long current_time);

    // ── Internal helpers ─────────────────────────────────────────────────────
    void push_event(Event e);
    long job_duration(const SimJob& job) const;

    // ── Private copy constructor (used by get_copy) ──────────────────────────
    Simulator(const Simulator& other, bool enable_logging);

    // ── State ────────────────────────────────────────────────────────────────
    SimMode mode_;
    long    current_time_ = 0;

    // Jobs: shared across copies — read-only after construction (MCTS safe)
    std::shared_ptr<const std::unordered_map<int, SimJob>> jobs_;

    // Nodes: copied per instance (each copy tracks its own resource state)
    std::vector<NodeInfo> nodes_;

    // Event queue (min-heap: earliest time first, ties broken by priority)
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> event_queue_;

    // Wait queue (FIFO by default; can be reordered by MXTS or heuristic function)
    std::deque<int> job_queue_;

    // Tracking
    std::vector<int>              running_jobs_;
    std::vector<int>              completed_jobs_;
    std::unordered_map<int, long> first_start_times_; // job_id -> first dispatch time
    std::unordered_map<int, long> end_times_;      // job_id → scheduled end time
    std::unordered_map<int, int>  job_to_node_;    // job_id → node index in nodes_

    // Scheduling
    SchedulingCycleFn callback_;
    bool              backfilling_enabled_ = false;

    // Logging
    std::string log_dir_;
    int         sim_id_ = -1;
    bool        logging_ = false;

    // Maintenance
    std::vector<MaintenanceWindow>   maintenance_windows_;
    std::optional<MaintenanceWindow> announced_maintenance_;
    bool                             maintenance_active_ = false;
    std::unordered_set<int>          cancelled_end_events_;
};
