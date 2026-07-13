#pragma once

#include "experiment.h"
#include "../core/job.h"
#include "../core/policy.h"
#include "../adapters/simulation/simulator.h"
#include "../adapters/simulation/maintenance.h"
#include <string>
#include <vector>
#include <limits>

class JsonExperiment : public Experiment {
public:
    explicit JsonExperiment(const std::string& json_path);

protected:
    void input_initialization()  override;
    void driver_initialization() override;
    void output_analytics()      override;

private:
    std::string json_path_;

    std::vector<SimJob>            jobs_;
    std::vector<MaintenanceWindow> maintenance_;
    int    total_procs_      = 0;
    int    max_submits_      = std::numeric_limits<int>::max();
    int    start_job_index_  = 0;
    SimMode mode_            = SimMode::Real;
    std::vector<double> size_bins_;
    std::vector<double> walltime_bins_;

    // Owns the policy objects referenced by drivers
    std::vector<std::unique_ptr<SchedulerPolicy>> policies_;
};
