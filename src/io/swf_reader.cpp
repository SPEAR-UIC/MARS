#include "swf_reader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

SwfReader::SwfReader(const std::string& filepath) : filepath_(filepath) {}

std::vector<SimJob> SwfReader::read() {
    std::ifstream file(filepath_);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open SWF file: " + filepath_);
    }

    std::vector<SimJob> jobs;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == ';') {
            parse_header(line);
            continue;
        }

        std::istringstream iss(line);

        long wait_time, used_memory, requested_memory, think_time;
        double avg_cpu_time;
        int num_allocated_procs, status, project_id, allocation_id,
            executable_number, partition_number, preceding_job_number;

        int  id, queue_id, num_requested_procs;
        long submit_time, run_time, walltime;

        iss >> id >> submit_time >> wait_time >> run_time
            >> num_allocated_procs >> avg_cpu_time >> used_memory
            >> num_requested_procs >> walltime >> requested_memory
            >> status >> project_id >> allocation_id >> executable_number
            >> queue_id >> partition_number >> preceding_job_number >> think_time;

        if (iss.fail()) continue;

        Resources res;
        res.procs     = num_requested_procs > 0 ? num_requested_procs : num_allocated_procs;
        res.memory_mb = requested_memory > 0 ? requested_memory : 0;

        jobs.push_back(SimJob(id, submit_time, walltime, res, run_time));
    }

    return jobs;
}

void SwfReader::parse_header(const std::string& line) {
    if (line.size() < 2 || line[0] != ';') return;

    std::string content = line.substr(1);
    size_t start = content.find_first_not_of(" \t");
    if (start == std::string::npos) return;
    content = content.substr(start);

    size_t colon = content.find(':');
    if (colon == std::string::npos) return;

    std::string key   = content.substr(0, colon);
    std::string value = content.substr(colon + 1);
    start = value.find_first_not_of(" \t");
    if (start != std::string::npos) value = value.substr(start);

    if      (key == "MaxNodes")      max_nodes_       = std::stoi(value);
    else if (key == "MaxProcs")      max_procs_       = std::stoi(value);
    else if (key == "UnixStartTime") unix_start_time_ = std::stol(value);
}

int  SwfReader::max_procs()       const { return max_procs_; }
int  SwfReader::max_nodes()       const { return max_nodes_; }
long SwfReader::unix_start_time() const { return unix_start_time_; }
