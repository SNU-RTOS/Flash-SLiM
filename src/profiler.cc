#include "profiler.h"

namespace
{
    // Helper function to convert timeval to seconds
    double __timeval_to_sec(const struct timeval &tv)
    {
        return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
    }

    // Helper function to convert timespec to seconds
    double __timespec_to_sec(const struct timespec &ts)
    {
        return ts.tv_sec + (ts.tv_nsec / 1.0e9);
    }

    double __timeval_to_ms(const struct timeval &tv)
    {
        return (double)tv.tv_sec * 1000.0 + ((double)tv.tv_usec * 1e6) / 1e9;
    }

    double __timespec_to_ms(const struct timespec &ts)
    {
        return ts.tv_sec * 1e3 + (ts.tv_nsec * 1e3 / 1.0e9);
    }

    // Function to get CPU time for a specific core (if possible)
    // Note: This uses /proc/stat to get per-CPU statistics
    std::pair<double, double> __get_core_cpu_time(int core_id)
    {
        std::ifstream stat_file("/proc/stat");
        if (!stat_file.is_open())
        {
            return std::make_pair(0.0, 0.0);
        }

        std::string line;
        std::string cpu_prefix = "cpu" + std::to_string(core_id);

        while (std::getline(stat_file, line))
        {
            if (line.find(cpu_prefix) == 0)
            {
                std::istringstream iss(line);
                std::string cpu_label;
                unsigned long user, nice, system, idle, iowait, irq, softirq, steal;

                iss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

                // user + nice = user time, system + irq + softirq = system time
                double user_time = user + nice;
                double system_time = system + irq + softirq;

                return std::make_pair(user_time, system_time);
            }
        }

        return std::make_pair(0.0, 0.0);
    }

    // etc
    // inline double GetParallelEfficiency(const  &stats)
    // {
    //     if (stats.wall_time_ms <= 0)
    //         return 0.0;
    //     return (stats.cpu_time_sec * 1000.0) / stats.wall_time_ms;
    // }
}

namespace ai_edge_torch::custom::profiler
{
    // Function to detect which cores the process is actually running on
    void detect_active_cores(std::vector<int> &cores)
    {
        // Try to read process affinity mask
        cpu_set_t mask;
        CPU_ZERO(&mask);

        if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
        {
            for (int i = 0; i < CPU_SETSIZE; i++)
            {
                if (CPU_ISSET(i, &mask))
                {
                    cores.push_back(i);
                }
            }
        }

        std::cout << "[INFO] Process is running on cores: ";
        for (int core : cores)
        {
            std::cout << core << " ";
        }
        std::cout << std::endl;
    }

    IOStats get_io_stats()
    {
        IOStats stats = {0, 0, 0, 0};
        std::ifstream io_stat("/proc/self/io");
        if (!io_stat.is_open())
        {
            return stats;
        }

        std::string line;
        while (std::getline(io_stat, line))
        {
            if (line.find("read_bytes:") != std::string::npos)
            {
                stats.bytes_read = std::stoull(line.substr(line.find(":") + 1));
            }
            else if (line.find("write_bytes:") != std::string::npos)
            {
                stats.bytes_written = std::stoull(line.substr(line.find(":") + 1));
            }
            else if (line.find("syscr:") != std::string::npos)
            {
                stats.read_ops = std::stoull(line.substr(line.find(":") + 1));
            }
            else if (line.find("syscw:") != std::string::npos)
            {
                stats.write_ops = std::stoull(line.substr(line.find(":") + 1));
            }
        }

        return stats;
    }

    void print_rusage(rusage usage_start, rusage usage_end, double wall_time_ms, const std::string phase_name)
    {
        double user_time_start = __timeval_to_ms(usage_start.ru_utime);
        double user_time_end = __timeval_to_ms(usage_end.ru_utime);
        double sys_time_start = __timeval_to_ms(usage_start.ru_stime);
        double sys_time_end = __timeval_to_ms(usage_end.ru_stime);
        double cpu_time_ms = (user_time_end - user_time_start) + (sys_time_end - sys_time_start);
        double user_time_ms = (user_time_end - user_time_start);
        double sys_time_ms = (sys_time_end - sys_time_start);

        std::cout << "[INFO] " << phase_name << "\n"
                  << "Time Statistics\n"
                  << "  " << wall_time_ms << " [ms] Wall Clock Time\n"
                  << "  - " << cpu_time_ms << " [ms] Total CPU Time (user + system time across all threads)\n"
                  << "      - " << user_time_ms << " [ms] User Time (time spent in user mode)\n"
                  << "      - " << sys_time_ms << " [ms] System Time (time spent in kernel mode)\n";

        double idle_time_ms = wall_time_ms - cpu_time_ms;
        if (idle_time_ms > 0.0)
        {
            std::cout << "  - " << idle_time_ms << " [ms] Idle or Waiting time (CPU inactive)\n";
        }

        std::cout << "Memory Usage\n"
                  << "  " << usage_end.ru_maxrss << " [kB] Maximum Resident Set Size (peak memory usage)\n";

        std::cout << "Page Faults\n"
                  << "  " << (usage_end.ru_minflt - usage_start.ru_minflt) << " Minor Page Faults\n"
                  << "  " << (usage_end.ru_majflt - usage_start.ru_majflt) << " Major Page Faults\n";

        std::cout << "Swap Activity\n"
                  << "  " << (usage_end.ru_nswap - usage_start.ru_nswap) << " Swaps (process swapped out of RAM)\n";

        std::cout << "Disk I/O\n"
                  << "  " << (usage_end.ru_inblock - usage_start.ru_inblock) << " Block Input Operations (disk reads)\n"
                  << "  " << (usage_end.ru_oublock - usage_start.ru_oublock) << " Block Output Operations (disk writes)\n";

        std::cout << "Context Switches\n"
                  << "  " << (usage_end.ru_nvcsw - usage_start.ru_nvcsw) << " Voluntary Context Switches (process yielded CPU)\n"
                  << "  " << (usage_end.ru_nivcsw - usage_start.ru_nivcsw) << " Involuntary Context Switches (preempted by OS)\n";

        std::cout << "\n";
    }

    void print_rusage_records(const std::vector<RUsageRecord> &records, const std::string &phase_name_prefix)
    {
        if (records.empty())
        {
            std::cout << "[RUsage] No records to print.\n";
            return;
        }

        double total_user_time = 0.0;
        double total_sys_time = 0.0;
        double total_cpu_time = 0.0;
        double total_wall_time = 0.0;

        size_t count = records.size();
        size_t valid_count = (count > 1) ? count - 1 : 0;

        for (size_t i = 1; i < records.size(); ++i) // i=1부터 합산: Decode_0 제외
        {
            const auto &rec = records[i];

            double user_start_ms = __timeval_to_ms(rec.start_.ru_utime);
            double user_end_ms = __timeval_to_ms(rec.end_.ru_utime);
            double sys_start_ms = __timeval_to_ms(rec.start_.ru_stime);
            double sys_end_ms = __timeval_to_ms(rec.end_.ru_stime);

            double user_time = user_end_ms - user_start_ms;
            double sys_time = sys_end_ms - sys_start_ms;
            double cpu_time = user_time + sys_time;

            total_user_time += user_time;
            total_sys_time += sys_time;
            total_cpu_time += cpu_time;
            total_wall_time += rec.wall_time_ms;
        }

        double avg_user_time = (valid_count > 0) ? (total_user_time / valid_count) : 0.0;
        double avg_sys_time = (valid_count > 0) ? (total_sys_time / valid_count) : 0.0;
        double avg_cpu_time = (valid_count > 0) ? (total_cpu_time / valid_count) : 0.0;
        double avg_wall_time = (valid_count > 0) ? (total_wall_time / valid_count) : 0.0;

        std::cout << "\n"
                  << COLOR_GREEN << "RUsage Records Report (" << phase_name_prefix << ")" << COLOR_RESET << "\n";

        if (valid_count > 0)
        {
            // 1) Total 출력
            std::cout << "\n[INFO] " << phase_name_prefix << " (total, excluding first) took\n"
                      << "  " << total_wall_time << " [ms] Wall Clock Time\n"
                      << "  - " << total_cpu_time << " [ms] CPU Time\n"
                      << "      - " << total_user_time << " [ms] User Time\n"
                      << "      - " << total_sys_time << " [ms] System Time\n";
            double total_idle_time_ms = total_wall_time - total_cpu_time;
            if (total_idle_time_ms > 0.0)
            {
                std::cout << "  - " << total_idle_time_ms << " [ms] Idle or Waiting time (CPU inactive)\n";
            }

            // 2) Average 출력
            std::cout << "\n[INFO] " << phase_name_prefix << " (avg, excluding first) took\n"
                      << "  " << avg_wall_time << " ms Wall Clock Time\n"
                      << "  - " << avg_cpu_time << " [ms] CPU Time\n"
                      << "      - " << avg_user_time << " [ms] User Time\n"
                      << "      - " << avg_sys_time << " [ms] System Time\n";
            double avg_idle_time_ms = avg_wall_time - avg_cpu_time;
            if (avg_idle_time_ms > 0.0)
            {
                std::cout << "  - " << avg_idle_time_ms << " [ms] Idle or Waiting time (CPU inactive)\n";
            }
        }
        else
        {
            std::cout << "\n[INFO] Not enough records to calculate average excluding the first.\n";
        }

        std::cout << "\n";

        // 0) 첫 번째 record (First decoding) 별도로 출력
        const auto &first = records[0];
        print_rusage(first.start_, first.end_, first.wall_time_ms, phase_name_prefix + "_0");

        // 3) 나머지 개별 단계 출력
        for (size_t i = 1; i < records.size(); ++i)
        {
            const auto &rec = records[i];
            std::string phase_name = phase_name_prefix + "_" + std::to_string(i);
            print_rusage(rec.start_, rec.end_, rec.wall_time_ms, phase_name);
        }
    }

    void upload_tensors_for_all_subgraphs(tflite::Interpreter *interpreter)
    {
        if (!interpreter)
        {
            std::cerr << "Invalid interpreter pointer\n";
            return;
        }

        // Get the number of subgraphs
        size_t num_subgraphs = interpreter->subgraphs_size();
        std::cout << "Processing " << num_subgraphs << " subgraphs\n";

        // Keep track of total tensors touched across all subgraphs
        size_t total_tensors_touched = 0;

        // Process each subgraph
        for (size_t subgraph_idx = 0; subgraph_idx < num_subgraphs; ++subgraph_idx)
        {
            const tflite::Subgraph &subgraph = (subgraph_idx == 0) ? interpreter->primary_subgraph() : *interpreter->subgraph(subgraph_idx);

            const std::vector<int> &execution_plan = subgraph.execution_plan();
            std::unordered_set<int> seen_tensors;

            std::cout << "Touching tensors for subgraph " << subgraph_idx << "\n";

            // Process each node in the execution plan
            for (int node_idx : execution_plan)
            {
                const auto *node_and_reg = subgraph.node_and_registration(node_idx);
                const TfLiteNode *node = &node_and_reg->first;

                // Helper lambda to process tensors
                auto processTensors = [&](const TfLiteIntArray *tensor_array)
                {
                    if (!tensor_array)
                        return;
                    for (int i = 0; i < tensor_array->size; ++i)
                    {
                        int tensor_idx = tensor_array->data[i];
                        if (tensor_idx < 0 || seen_tensors.count(tensor_idx))
                            continue;

                        TfLiteTensor *tensor = interpreter->tensor(tensor_idx);
                        if (tensor && tensor->data.raw)
                        {
                            size_t size = tensor->bytes;
                            for (size_t offset = 0; offset < size; offset += 4096)
                            {
                                volatile char dummy = *reinterpret_cast<char *>(tensor->data.raw + std::min(offset, size - 1));
                                (void)dummy;
                            }
                        }
                        seen_tensors.insert(tensor_idx);
                    }
                };

                // Process all tensor types
                processTensors(node->inputs);
                processTensors(node->outputs);
                processTensors(node->temporaries);
            }

            total_tensors_touched += seen_tensors.size();
            std::cout << "Touched " << seen_tensors.size() << " tensors in subgraph " << subgraph_idx << "\n";
        }

        std::cout << "Total tensors touched across all subgraphs: " << total_tensors_touched << "\n";
    }

    //////////////////////////////////////////////////////////////////////////
    /* class PerformanceMonitor: Performance monitoring class with improved measurements */
    // pubilc

    // Start monitoring a phase
    void PerformanceMonitor::start_phase(const std::string &phase_name)
    {
        // Record wall clock start time
        phase_start_times[phase_name] = std::chrono::steady_clock::now();

        // Record CPU time via rusage
        rusage start_rusage;
        getrusage(RUSAGE_SELF, &start_rusage);
        phase_start_rusage[phase_name] = start_rusage;

        // Record I/O stats
        phase_start_io[phase_name] = get_io_stats();

        // Add timespec measurements for CPU time
        struct timespec process_ts;
        if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &process_ts) == 0)
        {
            phase_start_process_time[phase_name] = process_ts;
        }
        else
        {
            std::cerr << "Warning: Failed to get CLOCK_PROCESS_CPUTIME_ID" << std::endl;
        }

        // Record per-core CPU times from /proc/stat
        std::vector<std::pair<double, double>> core_start_times;
        for (int core_id : monitored_cores)
        {
            core_start_times.push_back(__get_core_cpu_time(core_id));
        }
        phase_start_core_times[phase_name] = core_start_times;

        // NEW: Setup per-core timespec measurements
        CoreTimespec core_timespec;
        core_timespec.start_times.resize(monitored_cores.size());

        // For per-core CPU time measurement, we'll use perf_event_open with CPU counters
        // This is more reliable for per-core measurements than clock_gettime
        for (size_t i = 0; i < monitored_cores.size(); ++i)
        {
            // Initialize with zeros in case we can't get actual measurements
            struct timespec ts = {0, 0};
            core_timespec.start_times[i] = ts;

            // We'll use the start time from perf_event counters later
            // This timespec is primarily for backup and compatibility
            clock_gettime(CLOCK_MONOTONIC, &ts);
            core_timespec.start_times[i] = ts;
        }
        phase_core_timespec[phase_name] = core_timespec;

        // Setup per-core monitoring
        CoreEventFds core_fds;
        core_fds.user_time_fds.resize(monitored_cores.size(), -1);
        core_fds.system_time_fds.resize(monitored_cores.size(), -1);
        core_fds.io_wait_fds.resize(monitored_cores.size(), -1);
        core_fds.cpu_cycles_fds.resize(monitored_cores.size(), -1);
        core_fds.cpu_instructions_fds.resize(monitored_cores.size(), -1);
        core_fds.cpu_ref_cycles_fds.resize(monitored_cores.size(), -1);

        for (size_t i = 0; i < monitored_cores.size(); ++i)
        {
            int core_id = monitored_cores[i];

            // Setup user time counters
            core_fds.user_time_fds[i] = setup_user_time_counter(core_id);
            if (core_fds.user_time_fds[i] != -1)
            {
                ioctl(core_fds.user_time_fds[i], PERF_EVENT_IOC_RESET, 0);
                ioctl(core_fds.user_time_fds[i], PERF_EVENT_IOC_ENABLE, 0);
            }

            // Setup system time counters
            core_fds.system_time_fds[i] = setup_system_time_counter(core_id);
            if (core_fds.system_time_fds[i] != -1)
            {
                ioctl(core_fds.system_time_fds[i], PERF_EVENT_IOC_RESET, 0);
                ioctl(core_fds.system_time_fds[i], PERF_EVENT_IOC_ENABLE, 0);
            }

            // Setup I/O wait counters
            core_fds.io_wait_fds[i] = setup_io_wait_counter(core_id);
            if (core_fds.io_wait_fds[i] != -1)
            {
                ioctl(core_fds.io_wait_fds[i], PERF_EVENT_IOC_RESET, 0);
                ioctl(core_fds.io_wait_fds[i], PERF_EVENT_IOC_ENABLE, 0);
            }

            // Setup CPU cycles counter
            // core_fds.cpu_cycles_fds[i] = setup_cpu_cycles_counter(core_id);
            // if (core_fds.cpu_cycles_fds[i] != -1) {
            //     ioctl(core_fds.cpu_cycles_fds[i], PERF_EVENT_IOC_RESET, 0);
            //     ioctl(core_fds.cpu_cycles_fds[i], PERF_EVENT_IOC_ENABLE, 0);
            // }

            // Setup CPU instructions counter
            core_fds.cpu_instructions_fds[i] = setup_cpu_instructions_counter(core_id);
            if (core_fds.cpu_instructions_fds[i] != -1)
            {
                ioctl(core_fds.cpu_instructions_fds[i], PERF_EVENT_IOC_RESET, 0);
                ioctl(core_fds.cpu_instructions_fds[i], PERF_EVENT_IOC_ENABLE, 0);
            }

            // Setup CPU reference cycles counter
            // core_fds.cpu_ref_cycles_fds[i] = setup_cpu_ref_cycles_counter(core_id);
            // if (core_fds.cpu_ref_cycles_fds[i] != -1) {
            //     ioctl(core_fds.cpu_ref_cycles_fds[i], PERF_EVENT_IOC_RESET, 0);
            //     ioctl(core_fds.cpu_ref_cycles_fds[i], PERF_EVENT_IOC_ENABLE, 0);
            // }
        }

        phase_core_fds[phase_name] = core_fds;
    }

    // End monitoring a phase and return statistics
    void PerformanceMonitor::end_phase(const std::string &phase_name, PerfStats &stats)
    {
        // Check if the phase exists in all required maps
        auto time_it = phase_start_times.find(phase_name);
        auto rusage_it = phase_start_rusage.find(phase_name);
        auto io_it = phase_start_io.find(phase_name);
        auto core_fds_it = phase_core_fds.find(phase_name);
        auto core_times_it = phase_start_core_times.find(phase_name);
        auto core_timespec_it = phase_core_timespec.find(phase_name);
        auto process_time_it = phase_start_process_time.find(phase_name);

        // Handle missing phase records gracefully
        if (time_it == phase_start_times.end())
        {
            std::cerr << "Warning: Phase '" << phase_name << "' not found in time records. Skipping wall clock time measurement." << std::endl;
        }
        else
        {
            // Calculate wall clock time
            auto end_time = std::chrono::steady_clock::now();
            stats.wall_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     end_time - time_it->second)
                                     .count();

            // Clean up
            phase_start_times.erase(time_it);
        }

        if (rusage_it == phase_start_rusage.end())
        {
            std::cerr << "Warning: Phase '" << phase_name << "' not found in rusage records. Skipping CPU time measurement." << std::endl;
        }
        else
        {
            // Calculate CPU time from rusage
            rusage end_rusage;
            getrusage(RUSAGE_SELF, &end_rusage);

            stats.user_time_sec = __timeval_to_sec(end_rusage.ru_utime) -
                                  __timeval_to_sec(rusage_it->second.ru_utime);

            stats.system_time_sec = __timeval_to_sec(end_rusage.ru_stime) -
                                    __timeval_to_sec(rusage_it->second.ru_stime);

            stats.cpu_time_sec = stats.user_time_sec + stats.system_time_sec;

            // Clean up
            phase_start_rusage.erase(rusage_it);
        }

        if (io_it == phase_start_io.end())
        {
            std::cerr << "Warning: Phase '" << phase_name << "' not found in I/O records. Skipping I/O measurement." << std::endl;
        }
        else
        {
            // Calculate I/O stats from /proc/self/io
            IOStats end_io = get_io_stats();
            stats.io_bytes_read = end_io.bytes_read - io_it->second.bytes_read;
            stats.io_bytes_written = end_io.bytes_written - io_it->second.bytes_written;

            // Estimate I/O wait time based on I/O volume and throughput
            uint64_t total_io_bytes = stats.io_bytes_read + stats.io_bytes_written;
            uint64_t total_io_ops = (end_io.read_ops - io_it->second.read_ops) +
                                    (end_io.write_ops - io_it->second.write_ops);

            // A simple heuristic: if there was I/O activity, estimate wait time
            if (total_io_bytes > 0)
            {
                double io_throughput = 100.0 * 1024 * 1024; // Assume 100 MB/s
                stats.io_wait_time_ms = (total_io_bytes / io_throughput) * 1000.0;

                // Don't let I/O wait exceed wall time
                if (stats.wall_time_ms > 0)
                {
                    stats.io_wait_time_ms = std::min(stats.io_wait_time_ms, stats.wall_time_ms * 0.9);
                }
            }

            // Clean up
            phase_start_io.erase(io_it);
        }

        if (process_time_it != phase_start_process_time.end())
        {
            struct timespec end_process_ts;
            if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_process_ts) == 0)
            {
                stats.process_cpu_time_sec = __timespec_to_sec(end_process_ts) -
                                             __timespec_to_sec(process_time_it->second);
            }
            // Cleanup
            phase_start_process_time.erase(process_time_it);
        }

        // Handle per-core metrics
        if (core_times_it == phase_start_core_times.end())
        {
            std::cerr << "Warning: Phase '" << phase_name << "' not found in core times records. Skipping per-core measurements." << std::endl;
        }
        else
        {
            // Get per-core CPU times from /proc/stat for comparison
            std::vector<std::pair<double, double>> core_end_times;
            for (int core_id : monitored_cores)
            {
                core_end_times.push_back(__get_core_cpu_time(core_id));
            }

            // Calculate per-core CPU utilization from /proc/stat
            for (size_t i = 0; i < monitored_cores.size(); ++i)
            {
                double user_delta = core_end_times[i].first - core_times_it->second[i].first;
                double system_delta = core_end_times[i].second - core_times_it->second[i].second;

                // Store in our stats structure (these are in jiffies, need to convert to seconds)
                // On most Linux systems, there are 100 jiffies per second
                const double JIFFIES_PER_SEC = 100.0;
                stats.core_user_times.push_back(user_delta / JIFFIES_PER_SEC);
                stats.core_system_times.push_back(system_delta / JIFFIES_PER_SEC);
            }

            // Clean up
            phase_start_core_times.erase(core_times_it);
        }

        // Initialize per-core CPU times
        stats.core_cpu_times.resize(monitored_cores.size(), 0.0);

        if (core_fds_it == phase_core_fds.end())
        {
            std::cerr << "Warning: Phase '" << phase_name << "' not found in core fds records. Skipping perf event measurements." << std::endl;
        }
        else
        {
            // Read per-core metrics from perf events
            auto &core_fds = core_fds_it->second;

            // Structure to read perf event results
            struct read_format
            {
                uint64_t value;
                uint64_t time_enabled;
                uint64_t time_running;
            };

            for (size_t i = 0; i < monitored_cores.size(); ++i)
            {
                double core_user_time = 0.0;
                double core_system_time = 0.0;

                // Read and disable user time counter
                if (core_fds.user_time_fds[i] != -1)
                {
                    struct read_format rf;
                    if (read(core_fds.user_time_fds[i], &rf, sizeof(rf)) == sizeof(rf))
                    {
                        ioctl(core_fds.user_time_fds[i], PERF_EVENT_IOC_DISABLE, 0);
                        // Convert from ns to sec
                        core_user_time = rf.value / 1000000000.0;

                        // Update our stats with the more accurate perf event data
                        if (i < stats.core_user_times.size())
                        {
                            stats.core_user_times[i] = core_user_time;
                        }
                    }
                    close(core_fds.user_time_fds[i]);
                }

                // Read and disable system time counter
                if (core_fds.system_time_fds[i] != -1)
                {
                    struct read_format rf;
                    if (read(core_fds.system_time_fds[i], &rf, sizeof(rf)) == sizeof(rf))
                    {
                        ioctl(core_fds.system_time_fds[i], PERF_EVENT_IOC_DISABLE, 0);
                        // Convert from ns to sec
                        core_system_time = rf.value / 1000000000.0;

                        // Update our stats with the more accurate perf event data
                        if (i < stats.core_system_times.size())
                        {
                            stats.core_system_times[i] = core_system_time;
                        }
                    }
                    close(core_fds.system_time_fds[i]);
                }

                // Update total CPU time for this core
                if (i < stats.core_cpu_times.size())
                {
                    stats.core_cpu_times[i] = stats.core_user_times[i] + stats.core_system_times[i];
                }

                // Read and disable I/O wait counter
                if (core_fds.io_wait_fds[i] != -1)
                {
                    uint64_t count;
                    if (read(core_fds.io_wait_fds[i], &count, sizeof(count)) == sizeof(count))
                    {
                        ioctl(core_fds.io_wait_fds[i], PERF_EVENT_IOC_DISABLE, 0);
                        // Each CPU migration contributes to I/O wait time estimate
                        if (count > 0)
                        {
                            stats.io_wait_time_ms += count * 10.0; // Estimate 10ms per migration
                        }
                    }
                    close(core_fds.io_wait_fds[i]);
                }

                // Read CPU cycles counter
                if (core_fds.cpu_cycles_fds[i] != -1)
                {
                    struct read_format rf;
                    if (read(core_fds.cpu_cycles_fds[i], &rf, sizeof(rf)) == sizeof(rf))
                    {
                        ioctl(core_fds.cpu_cycles_fds[i], PERF_EVENT_IOC_DISABLE, 0);

                        // Scaling factor calculation
                        if (rf.time_enabled > 0)
                        {
                            double scaling_factor = (double)rf.time_running / rf.time_enabled;

                            if (scaling_factor < 1.0 && i < stats.core_cpu_times.size())
                            {
                                stats.core_cpu_times[i] = stats.core_cpu_times[i] * scaling_factor;
                            }
                        }
                    }
                    close(core_fds.cpu_cycles_fds[i]);
                }

                // Read CPU instructions counter
                if (core_fds.cpu_instructions_fds[i] != -1)
                {
                    struct read_format rf;
                    if (read(core_fds.cpu_instructions_fds[i], &rf, sizeof(rf)) == sizeof(rf))
                    {
                        ioctl(core_fds.cpu_instructions_fds[i], PERF_EVENT_IOC_DISABLE, 0);
                    }
                    close(core_fds.cpu_instructions_fds[i]);
                }

                // Read CPU reference cycles counter
                if (core_fds.cpu_ref_cycles_fds[i] != -1)
                {
                    struct read_format rf;
                    if (read(core_fds.cpu_ref_cycles_fds[i], &rf, sizeof(rf)) == sizeof(rf))
                    {
                        ioctl(core_fds.cpu_ref_cycles_fds[i], PERF_EVENT_IOC_DISABLE, 0);
                    }
                    close(core_fds.cpu_ref_cycles_fds[i]);
                }
            }

            // Clean up
            phase_core_fds.erase(core_fds_it);
        }

        if (core_timespec_it != phase_core_timespec.end())
        {
            // Clean up
            phase_core_timespec.erase(core_timespec_it);
        }
    }

    // private
    // Helper for perf_event_open syscall
    long PerformanceMonitor::perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                                             int cpu, int group_fd, unsigned long flags)
    {
        return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
    }

    // Setup user time counter for a specific core
    int PerformanceMonitor::setup_user_time_counter(int core_id)
    {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = PERF_TYPE_SOFTWARE;
        pe.size = sizeof(struct perf_event_attr);
        pe.config = PERF_COUNT_SW_TASK_CLOCK;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;
        pe.disabled = 1;
        pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

        // Use getpid() to track only our process on the specified core
        int fd = perf_event_open(&pe, getpid(), core_id, -1, 0);
        if (fd == -1)
        {
            std::cerr << "Warning: Failed to open user time perf event for core "
                      << core_id << ": " << strerror(errno) << std::endl;
        }
        return fd;
    }

    // Setup system time counter for a specific core
    int PerformanceMonitor::setup_system_time_counter(int core_id)
    {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = PERF_TYPE_SOFTWARE;
        pe.size = sizeof(struct perf_event_attr);
        pe.config = PERF_COUNT_SW_TASK_CLOCK;
        pe.exclude_user = 1;
        pe.exclude_hv = 1;
        pe.disabled = 1;
        pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

        // Use getpid() to track only our process on the specified core
        int fd = perf_event_open(&pe, getpid(), core_id, -1, 0);
        if (fd == -1)
        {
            std::cerr << "Warning: Failed to open system time perf event for core "
                      << core_id << ": " << strerror(errno) << std::endl;
        }
        return fd;
    }

    // Setup I/O wait counter
    int PerformanceMonitor::setup_io_wait_counter(int core_id)
    {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = PERF_TYPE_SOFTWARE;
        pe.size = sizeof(struct perf_event_attr);

        // Block I/O delay tracking
        pe.config = PERF_COUNT_SW_CPU_MIGRATIONS; // As a proxy for I/O waits
        pe.disabled = 1;
        pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

        int fd = perf_event_open(&pe, getpid(), core_id, -1, 0);
        if (fd == -1)
        {
            std::cerr << "Warning: Failed to open I/O wait perf event for core "
                      << core_id << ": " << strerror(errno) << std::endl;
        }
        return fd;
    }

    // Setup CPU cycles counter for a specific core
    int PerformanceMonitor::setup_cpu_cycles_counter(int core_id)
    {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = PERF_TYPE_HARDWARE;
        pe.size = sizeof(struct perf_event_attr);
        pe.config = PERF_COUNT_HW_CPU_CYCLES;
        pe.disabled = 1;
        pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

        int fd = perf_event_open(&pe, getpid(), core_id, -1, 0);
        if (fd == -1)
        {
            std::cerr << "Warning: Failed to open CPU cycles perf event for core "
                      << core_id << ": " << strerror(errno) << std::endl;
        }
        return fd;
    }

    // Setup CPU instructions counter for a specific core
    int PerformanceMonitor::setup_cpu_instructions_counter(int core_id)
    {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = PERF_TYPE_HARDWARE;
        pe.size = sizeof(struct perf_event_attr);
        pe.config = PERF_COUNT_HW_INSTRUCTIONS;
        pe.disabled = 1;
        pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

        int fd = perf_event_open(&pe, getpid(), core_id, -1, 0);
        if (fd == -1)
        {
            std::cerr << "Warning: Failed to open CPU instructions perf event for core "
                      << core_id << ": " << strerror(errno) << std::endl;
        }
        return fd;
    }

    // Setup CPU reference cycles counter for a specific core
    int PerformanceMonitor::setup_cpu_ref_cycles_counter(int core_id)
    {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = PERF_TYPE_HARDWARE;
        pe.size = sizeof(struct perf_event_attr);
        pe.config = PERF_COUNT_HW_REF_CPU_CYCLES;
        pe.disabled = 1;
        pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

        int fd = perf_event_open(&pe, getpid(), core_id, -1, 0);
        if (fd == -1)
        {
            std::cerr << "Warning: Failed to open CPU reference cycles perf event for core "
                      << core_id << ": " << strerror(errno) << std::endl;
        }
        return fd;
    }

    // Get system I/O wait percentage
    double PerformanceMonitor::get_system_io_wait()
    {
        std::ifstream stat_file("/proc/stat");
        if (!stat_file.is_open())
        {
            return 0.0;
        }

        std::string line;
        std::getline(stat_file, line);

        std::istringstream iss(line);
        std::string cpu_label;
        unsigned long user, nice, system, idle, iowait, irq, softirq, steal;

        iss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

        unsigned long total_time = user + nice + system + idle + iowait + irq + softirq + steal;

        return (total_time > 0) ? (iowait * 100.0) / total_time : 0.0;
    }

    //////////////////////////////////////////////////////////////////////////
    /* class PerformanceMetrics: Class to collect and report performance metrics */
    // public:
    void PerformanceMetrics::RecordStats(const std::string &phase, const PerfStats &stats)
    {
        phase_stats_.emplace_back(phase, stats);
    }

    void PerformanceMetrics::PrintStats() const
    {
        std::unordered_map<std::string, std::vector<PerfStats>> grouped_stats;
        std::unordered_set<std::string> printed;

        // Group same phase names
        for (const auto &[phase, stat] : phase_stats_)
        {
            grouped_stats[phase].push_back(stat);
        }

        // Preserve insertion order
        std::cout << "\n"
                  << COLOR_GREEN << "Perf Report " << COLOR_RESET << "\n";
        for (const auto &[phase, _] : phase_stats_)
        {
            if (printed.count(phase))
                continue;
            printed.insert(phase);

            const auto &stats_vec = grouped_stats.at(phase);
            if (stats_vec.empty())
                continue;

            std::cout << "\n"
                      << COLOR_YELLOW << "=== Performance Statistics for Phase: " << phase << " ===" << COLOR_RESET << "\n";

            if (stats_vec.size() == 1)
            {
                PrintSinglePhaseStat(stats_vec[0]);
            }
            else
            {
                PrintAverageStats(stats_vec);

                if (stats_vec.size() <= 10)
                {
                    std::cout << "\nPer-step details:\n";
                    for (size_t i = 0; i < stats_vec.size(); ++i)
                    {
                        std::cout << "Step " << i << ":\n";
                        PrintSinglePhaseStat(stats_vec[i], "  ");
                    }
                }
            }
        }
    }

    void PerformanceMetrics::PrintAverageStats(const std::vector<PerfStats> &stats_vec) const
    {
        double avg_wall_time = 0, avg_user_time = 0, avg_system_time = 0;
        double avg_cpu_time = 0, avg_io_wait_time = 0;
        double avg_io_bytes_read = 0, avg_io_bytes_written = 0;

        for (const auto &stats : stats_vec)
        {
            avg_wall_time += stats.wall_time_ms;
            avg_user_time += stats.user_time_sec;
            avg_system_time += stats.system_time_sec;
            avg_cpu_time += stats.cpu_time_sec;
            avg_io_wait_time += stats.io_wait_time_ms;
            avg_io_bytes_read += stats.io_bytes_read;
            avg_io_bytes_written += stats.io_bytes_written;
        }

        size_t count = stats_vec.size();
        avg_wall_time /= count;
        avg_user_time /= count;
        avg_system_time /= count;
        avg_cpu_time /= count;
        avg_io_wait_time /= count;
        avg_io_bytes_read /= count;
        avg_io_bytes_written /= count;

        std::cout << "Number of measurements: " << count << "\n"
                  << "Average wall clock time: " << avg_wall_time << " ms\n"
                  << "Average user time: " << avg_user_time << " sec\n"
                  << "Average system time: " << avg_system_time << " sec\n"
                  << "Average CPU time (user+system): " << avg_cpu_time << " sec\n"
                  << "Average I/O wait time: " << avg_io_wait_time << " ms\n"
                  << "Average I/O bytes read: " << avg_io_bytes_read / (1024.0 * 1024.0) << " MB\n"
                  << "Average I/O bytes written: " << avg_io_bytes_written / (1024.0 * 1024.0) << " MB\n"
                  << "CPU utilization: " << (avg_cpu_time * 1000 * 100) / avg_wall_time << "%\n";
    }

    void PerformanceMetrics::PrintSinglePhaseStat(const PerfStats &stats, const std::string &prefix) const
    {
        std::cout << prefix << "Wall clock time: " << stats.wall_time_ms << " ms\n"
                  << prefix << "User time: " << stats.user_time_sec << " sec\n"
                  << prefix << "System time: " << stats.system_time_sec << " sec\n"
                  << prefix << "Total CPU time (user+system): " << stats.cpu_time_sec << " sec\n"
                  << prefix << "Process CPU time (timespec): " << stats.process_cpu_time_sec << " sec\n"
                  << prefix << "I/O wait time: " << stats.io_wait_time_ms << " ms\n"
                  << prefix << "I/O bytes read: " << stats.io_bytes_read / (1024.0 * 1024.0) << " MB\n"
                  << prefix << "I/O bytes written: " << stats.io_bytes_written / (1024.0 * 1024.0) << " MB\n"
                  << prefix << "CPU utilization: " << (stats.cpu_time_sec * 1000 * 100) / stats.wall_time_ms << "%\n";

        if (!stats.core_user_times.empty())
        {
            std::cout << prefix << "Per-core statistics:\n";
            for (size_t i = 0; i < stats.core_user_times.size(); ++i)
            {
                std::cout << prefix << "  Core " << i << ": "
                          << "User=" << stats.core_user_times[i] << "s, "
                          << "System=" << stats.core_system_times[i] << "s, "
                          << "Total=" << (stats.core_user_times[i] + stats.core_system_times[i]) << "s\n";
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    /* class DecodingMetrics: Class for measuring decoding metrics (time to first token, average times, etc.)*/
    // public:
    // Called before decoding loop starts

    void DecodingMetrics::RecordTimes(double inference_time_ms, double sampling_time_ms)
    {
        double decoding_time_ms = inference_time_ms + sampling_time_ms;

        // If this is the first token, record time to first token
        if (!first_token_recorded_)
        {
            first_token_recorded_ = true;
            first_decoding_time_ms_ = decoding_time_ms;
            return;
        }

        // Track inference time
        total_inference_time_ms_ += inference_time_ms;
        // Track sampling time
        total_sampling_time_ms_ += sampling_time_ms;
        // Track total decoding time
        total_decoding_time_ms_ += decoding_time_ms;

        // Track total tokens
        ++token_count_excluding_first_;
    }

    // Print out final decoding metrics
    void DecodingMetrics::PrintMetrics(int prefill_time_ms)
    {
        double avg_inference_time_ms = 0.0;
        double avg_sampling_time_ms = 0.0;
        double avg_decoding_time_ms = 0.0;
        double avg_inference_speed = 0.0;
        double avg_sampling_speed = 0.0;
        double avg_decoding_speed = 0.0;

        if (token_count_excluding_first_ > 0)
        {
            avg_inference_time_ms = total_inference_time_ms_ / token_count_excluding_first_;
            avg_sampling_time_ms = total_sampling_time_ms_ / token_count_excluding_first_;
            avg_decoding_time_ms = (total_sampling_time_ms_ + total_inference_time_ms_) / token_count_excluding_first_;

            avg_inference_speed = token_count_excluding_first_ / (total_inference_time_ms_ / 1000);
            avg_sampling_speed = token_count_excluding_first_ / (total_sampling_time_ms_ / 1000);
            avg_decoding_speed = token_count_excluding_first_ / (total_decoding_time_ms_ / 1000);
        }

        std::cout << "\n"
                  << COLOR_GREEN << "Decoding Metrics" << COLOR_RESET << "\n\n";

        std::cout << "[METRICS] Total Number of Generated Tokens : " << token_count_excluding_first_ + 1 << " tokens\n\n";

        std::cout << "[METRICS] Prefill Time                     : " << prefill_time_ms << " ms\n";
        std::cout << "[METRICS] First Decoding Time              : " << first_decoding_time_ms_ << " ms\n";
        std::cout << "[METRICS] Time to First Token              : " << prefill_time_ms + first_decoding_time_ms_ << " ms\n\n";
        std::cout << "[METRICS] [NOTE] First Decoding Time is excluded from Total Decoding Time \n\n";

        std::cout << "[METRICS] Total Inference Time             : " << total_inference_time_ms_ << " ms\n";
        std::cout << "[METRICS] Total Sampling Time              : " << total_sampling_time_ms_ << " ms\n";
        std::cout << "[METRICS] Total Decoding Time              : " << total_decoding_time_ms_ << " ms\n\n";

        std::cout << "[METRICS] Average Inference Time per Token : " << avg_inference_time_ms << " ms"
                  << " (" << avg_inference_speed << " tokens/s)\n";
        std::cout << "[METRICS] Average Sampling Time per Token  : " << avg_sampling_time_ms << " ms"
                  << " (" << avg_sampling_speed << " tokens/s)\n";
        std::cout << "[METRICS] Average Decoding Time per Token  : " << avg_decoding_time_ms << " ms"
                  << " (" << avg_decoding_speed << " tokens/s)\n";
    }
    // private:

    std::mutex monitor_signal_mutex;
    std::condition_variable monitor_signal_cv;
    std::atomic<bool> monitor_log_requested{false};
    std::atomic<bool> monitor_generation_done{false};
    std::string current_phase_name;
    int phase_status; // 0 for start, 1 for end
}; // ai_edge_torch::custom::profiler
