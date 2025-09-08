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
}

namespace custom::profiler
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

    void print_rusage(rusage usage_start, rusage usage_end, double wall_time_ms, const std::string phase_name)
    {
        double user_time_start = __timeval_to_ms(usage_start.ru_utime);
        double user_time_end = __timeval_to_ms(usage_end.ru_utime);
        double sys_time_start = __timeval_to_ms(usage_start.ru_stime);
        double sys_time_end = __timeval_to_ms(usage_end.ru_stime);
        double cpu_time_ms = (user_time_end - user_time_start) + (sys_time_end - sys_time_start);
        double user_time_ms = (user_time_end - user_time_start);
        double sys_time_ms = (sys_time_end - sys_time_start);

        std::cout << "[INFO] " << phase_name << " -------------------------------------\n"
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
        std::vector<RUsageRecord> filtered;
        for (const auto &rec : records)
        {
            if (phase_name_prefix.empty() ||
                rec.phase_name_.find(phase_name_prefix) == 0)
            {
                filtered.push_back(rec);
            }
        }

        if (filtered.empty())
        {
            std::cout << "[RUsage] No records to print for prefix: " << phase_name_prefix << "\n";
            return;
        }

        std::cout << "\n---------------------------------------------------\n";
        std::cout << COLOR_GREEN << "RUsage Records Report (" << phase_name_prefix << ")" << COLOR_RESET << "\n";
        std::cout << "\n";

        for (const auto &rec : filtered)
        {
            // Print each record
            print_rusage(rec.start_, rec.end_, rec.wall_time_ms_, rec.phase_name_);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    /* class DecodingMetrics: Class for measuring decoding metrics (time to first token, average times, etc.)*/
    // public:

    void GenAIMetrics::RecordPrefillTime(double prefill_time_ms)
    {
        prefill_time_ms_ = prefill_time_ms;
    }

    void GenAIMetrics::RecordDecodingTime(double inference_time_ms,
                                               double sampling_time_ms,
                                               double detok_time_ms)
    {
        double decoding_time_ms = inference_time_ms + sampling_time_ms + detok_time_ms;

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
        // Track detokenization time
        total_detokenization_time_ms_ += detok_time_ms;
        // Track total decoding time
        total_decoding_time_ms_ += decoding_time_ms;

        // Track total tokens
        ++token_count_excluding_first_;
    }

    // Print out final decoding metrics
    void GenAIMetrics::Print()
    {
        double avg_inference_time_ms = 0.0;
        double avg_sampling_time_ms = 0.0;
        double avg_detokenization_time_ms = 0.0;
        double avg_decoding_time_ms = 0.0;

        double avg_inference_speed = 0.0;
        double avg_sampling_speed = 0.0;
        double avg_detokenization_speed = 0.0;
        double avg_decoding_speed = 0.0;

        if (token_count_excluding_first_ > 0)
        {
            avg_inference_time_ms = total_inference_time_ms_ / token_count_excluding_first_;
            avg_sampling_time_ms = total_sampling_time_ms_ / token_count_excluding_first_;
            avg_detokenization_time_ms = total_detokenization_time_ms_ / token_count_excluding_first_;
            avg_decoding_time_ms = total_decoding_time_ms_ / token_count_excluding_first_;

            avg_inference_speed = token_count_excluding_first_ / (total_inference_time_ms_ / 1000);
            avg_sampling_speed = token_count_excluding_first_ / (total_sampling_time_ms_ / 1000);
            avg_detokenization_speed = token_count_excluding_first_ / (total_detokenization_time_ms_ / 1000);
            avg_decoding_speed = token_count_excluding_first_ / (total_decoding_time_ms_ / 1000);
        }
        std::cout << "---------------------------------------------------\n";
        std::cout << COLOR_GREEN << "Generation Metrics" << COLOR_RESET << "\n\n";

        std::cout << "[METRICS] Total Number of Generated Tokens : " << token_count_excluding_first_ + 1 << " tokens\n\n";

        std::cout << "[METRICS] Prefill Time                     : " << prefill_time_ms_ << " ms\n";
        std::cout << "[METRICS] First Decoding Time              : " << first_decoding_time_ms_ << " ms\n";
        std::cout << "[METRICS] Time to First Token              : " << prefill_time_ms_ + first_decoding_time_ms_ << " ms\n\n";
        std::cout << "[METRICS] [NOTE] First Decoding Time is excluded from Total Decoding Time \n\n";

        std::cout << "[METRICS] Total Inference Time             : " << total_inference_time_ms_ << " ms\n";
        std::cout << "[METRICS] Total Sampling Time              : " << total_sampling_time_ms_ << " ms\n";
        std::cout << "[METRICS] Total Detokenization Time        : " << total_detokenization_time_ms_ << " ms\n";
        std::cout << "[METRICS] Total Decoding Time              : " << total_decoding_time_ms_ << " ms\n\n";

        std::cout << "[METRICS] Average Inference Time per Token        : " << avg_inference_time_ms << " ms"
                  << " (" << avg_inference_speed << " tokens/s)\n";
        std::cout << "[METRICS] Average Sampling Time per Token         : " << avg_sampling_time_ms << " ms"
                  << " (" << avg_sampling_speed << " tokens/s)\n";
        std::cout << "[METRICS] Average Detokenization Time per Token   : " << avg_detokenization_time_ms << " ms"
                  << " (" << avg_detokenization_speed << " tokens/s)\n";
        std::cout << "[METRICS] Average Decoding Time per Token         : " << avg_decoding_time_ms << " ms"
                  << " (" << avg_decoding_speed << " tokens/s)\n";
    }



    /* Timer Utility */
    TimerUtility::TimerUtility(const std::string &name)
            : name_(name) {}
            
    void TimerUtility::Start()
    {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double TimerUtility::Stop() const
    {
        auto end_ = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - start_).count();
        return static_cast<double>(duration_us) / 1000000.0;
    }

    /* Scope Timer */
    ScopeTimer::ScopeTimer(double &out_ref, const std::string &name)
            : out_ref_(out_ref), timer_(name)
    {
        timer_.Start();
    }

    ScopeTimer::~ScopeTimer()
    {
        out_ref_ = static_cast<double>(timer_.Stop());
    }

    
    ScopeEventHandler::ScopeEventHandler(const std::string &name){
        current_phase_name_= name;
        TRACE_LOGIC_START(current_phase_name_.c_str());
    }

    ScopeEventHandler::~ScopeEventHandler()
    {
        TRACE_LOGIC_END(current_phase_name_.c_str());

    }


    /* ScopeEventPrefetcher */
    // Constructor implementation
    ScopeEventPrefetcher::ScopeEventPrefetcher(PhaseContext &ctx, const std::string &name)
        : ctx_(ctx), lock_(ctx_.mutex)
        {
            ctx_.current_phase_name = name;
            ctx_.log_requested.store(true);
            ctx_.phase_status = 0; // Start phase
            ctx_.signal_cv.notify_all();
            ctx_.signal_cv.wait(lock_, [&]() { return !ctx_.log_requested.load(); });

            TRACE_LOGIC_START(ctx_.current_phase_name.c_str());
        }

    // Destructor implementation
    ScopeEventPrefetcher::~ScopeEventPrefetcher()
        {
            TRACE_LOGIC_END(ctx_.current_phase_name.c_str());

            ctx_.log_requested.store(true);
            ctx_.phase_status = 1; // End phase
            ctx_.signal_cv.notify_all();
            ctx_.signal_cv.wait(lock_, [&]() { return !ctx_.log_requested.load(); });
        }


    /* ScopeEventListener */
    ScopeEventListener::ScopeEventListener(PhaseContext &ctx, bool log_stdout, std::vector<RUsageRecord> *usage_records)
            : ctx_(ctx),
              log_stdout_(log_stdout),
              usage_records_(usage_records),
              timer_("ScopeEventListener") {}
        
        void ScopeEventListener::Run()
        {
            std::unique_lock<std::mutex> lock(ctx_.mutex);

            while (!ctx_.generation_done.load())
            {
                ctx_.signal_cv.wait(lock, [&]()
                                    { return ctx_.log_requested.load() || ctx_.generation_done.load(); });

                if (ctx_.generation_done.load())
                {
                    std::cout << "[INFO] Monitoring thread exiting...\n";
                    break;
                }

                if (ctx_.phase_status == 1)
                { // End phase
                    getrusage(RUSAGE_SELF, &usage_end_);
                    double duration_ms = timer_.Stop();

                    if (log_stdout_)
                    {
                        custom::profiler::print_rusage(
                            usage_start_, usage_end_, duration_ms, ctx_.current_phase_name);
                    }
                    if (usage_records_)
                    {
                        usage_records_->emplace_back(usage_start_, usage_end_, duration_ms, ctx_.current_phase_name);
                    }
                }

                ctx_.log_requested.store(false);
                ctx_.signal_cv.notify_all();

                if (ctx_.phase_status == 0)
                { // Start phase
                    timer_.Start();
                    getrusage(RUSAGE_SELF, &usage_start_);
                }
            }
            std::cout << "[INFO] Monitoring finished\n";
        }

}; // custom::profiler
