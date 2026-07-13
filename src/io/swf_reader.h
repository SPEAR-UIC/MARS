#pragma once

#include <string>
#include <vector>

#include "../core/job.h"

class SwfReader {

public:
    explicit SwfReader(const std::string& filepath);
    std::vector<SimJob> read();
    int max_procs()     const;
    int max_nodes()     const;
    long unix_start_time()  const;

private:
    void parse_header(const std::string& line);
    std::string filepath_;
    int max_procs_          = 0;
    int max_nodes_          = 0;
    long unix_start_time_   = 0; 
};