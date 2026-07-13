
#pragma once

#include "resources.h"

enum class JobState{
    Queued,
    Running,
    Held, // Currently not being considered for scheduler
    Completed, // Exit code 0
    Cancelled, // User Cancellation
    Failed
};

struct Job {

    int id;
    JobState state;
    Resources requested;
    long submit_time;
    long walltime;
    int priority;
    int queue_id = -1;
    long start_time = -1;
    long end_time = -1;

    Job(int id, long submit_time, long walltime, Resources requested, int priority = 0)
        : id(id),
        state(JobState::Queued),
        requested(requested),
        submit_time(submit_time),
        walltime(walltime),
        priority(priority)
    {}
};

struct SimJob : Job {
    long run_time = -1;

    SimJob(int id, long submit_time, long walltime, Resources requested, long run_time, int priority = 0)
        : Job(id, submit_time, walltime, requested, priority),
          run_time(run_time)
    {}
};