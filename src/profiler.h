#ifndef AI_EDGE_TORCH_GENERATIVE_PROFILER_H_
#define AI_EDGE_TORCH_GENERATIVE_PROFILER_H_

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <istream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

// included by nclee
#include <iostream>
#include <sys/mman.h>
#include <linux/perf_event.h> // For perf_event_attr and PERF_* constants
#include <sys/syscall.h>      // For syscall and __NR_perf_event_open
#include <unistd.h>           // For syscall wrapper and pid_t
#include <sys/ioctl.h>
#include <unordered_map>
#include <stdexcept>
#include <sys/time.h>
#include <sys/resource.h>
#ifndef __NR_perf_event_open
#define __NR_perf_event_open 241 // Syscall number for aarch64
#endif

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/c/common.h"

// 예: 프로젝트 내에 정의된 매크로라 가정
#ifndef MINIMAL_CHECK
#define MINIMAL_CHECK(x)                       \
    if (!(x))                                  \
    {                                          \
        std::cerr << "Check failed: " #x "\n"; \
        std::abort();                          \
    }
#endif

// ANSI color codes
#define COLOR_GREEN  "\033[1;32m"
#define COLOR_YELLOW "\033[1;33m"
#define COLOR_RESET  "\033[0m"
#define COLOR_RED    "\033[1;31m"
#define COLOR_BLUE   "\033[1;34m"
#define COLOR_CYAN   "\033[1;36m"

// /proc/stat에서 특정 코어의 user/system jiffies 읽기
std::pair<double, double> get_core_cpu_time(int core_id);

namespace ai_edge_torch::custom::profiler
{
    //////////////////////////////////////////////////////////////
    /* Struct, Types */
    //////////////////////////////////////////////////////////////

    // Performance metrics structure to store all relevant timing data
    struct PerfStats
    {
        // Wall clock time
        double wall_time_ms;

        // CPU time (from rusage)
        double user_time_sec;
        double system_time_sec;
        double cpu_time_sec; // user + system

        // I/O time (from multiple sources)
        double io_wait_time_ms;
        double io_bytes_read;
        double io_bytes_written;

        // Per-core metrics (if available)
        std::vector<double> core_user_times;
        std::vector<double> core_system_times;
        std::vector<double> core_cpu_times; // Add this missing field

        // New fields for timespec-based CPU time verification
        double process_cpu_time_sec; // CPU time using clock_gettime(CLOCK_PROCESS_CPUTIME_ID)

        PerfStats() : wall_time_ms(0), user_time_sec(0), system_time_sec(0),
                      cpu_time_sec(0), io_wait_time_ms(0), io_bytes_read(0), io_bytes_written(0),
                      process_cpu_time_sec(0) {}
    };

    // Function to measure I/O statistics from /proc filesystem
    struct IOStats
    {
        uint64_t bytes_read;
        uint64_t bytes_written;
        uint64_t read_ops;
        uint64_t write_ops;
    };

    // RUsage record structure
    struct RUsageRecord
    {
        rusage start_;
        rusage end_;
        double wall_time_ms;

        RUsageRecord() = default;

        RUsageRecord(const struct rusage& start, const struct rusage& end, const double wall_time_ms)
            : start_(start), end_(end), wall_time_ms(wall_time_ms) {}
    };

    //////////////////////////////////////////////////////////////
    /* Functions */
    //////////////////////////////////////////////////////////////

    // 현재 프로세스가 실행 중인 CPU 코어 목록 반환
    void detect_active_cores(std::vector<int> &cores);

    // /proc/self/io에서 I/O 통계 읽기
    IOStats get_io_stats();

    // RUsage 출력 헬퍼
    void print_rusage(rusage usage_start, rusage usage_end, const std::string phase_name);
    void print_rusage_records(const std::vector<RUsageRecord> &records, const std::string &phase_name_prefix);

    // Tensors 강제 페이지 폴트 (upload) 해주는 함수
    void upload_tensors_for_all_subgraphs(tflite::Interpreter *interpreter);

    //////////////////////////////////////////////////////////////
    /* Classes */
    //////////////////////////////////////////////////////////////


    // --------------------------------------------------------------------------
    // PerformanceMonitor
    // --------------------------------------------------------------------------
    class PerformanceMonitor
    {
    public:
        //  Constructor that takes a list of core IDs to monitor
        explicit PerformanceMonitor(const std::vector<int> &cores = {})
            : monitored_cores(cores)
        {
            // If no cores specified, detect active cores
            if (monitored_cores.empty())
            {
                detect_active_cores(monitored_cores);

                // If still empty, default to core 0
                if (monitored_cores.empty())
                {
                    monitored_cores.push_back(0);
                }
            }

            std::cout << "Performance monitor tracking cores: ";
            for (int core : monitored_cores)
            {
                std::cout << core << " ";
            }
            std::cout << std::endl;
        }
        ~PerformanceMonitor() = default;

        // 어떤 "phase"가 시작될 때 기록
        void start_phase(const std::string &phase_name);

        // 어떤 "phase"가 끝났을 때, PerfStats를 계산해 반환
        void end_phase(const std::string &phase_name, PerfStats &stats);

    private:
        // perf_event_open용
        long perf_event_open(struct perf_event_attr *hw_event,
                             pid_t pid, int cpu, int group_fd,
                             unsigned long flags);

        // 세부 설정 함수
        int setup_user_time_counter(int core_id);
        int setup_system_time_counter(int core_id);
        int setup_io_wait_counter(int core_id);
        int setup_cpu_cycles_counter(int core_id);
        int setup_cpu_instructions_counter(int core_id);
        int setup_cpu_ref_cycles_counter(int core_id);

        double get_system_io_wait();

        // --- Phase별 시간 측정 ---
        std::unordered_map<std::string, struct timespec> phase_start_process_time;
        std::unordered_map<std::string, std::chrono::steady_clock::time_point> phase_start_times;
        std::unordered_map<std::string, rusage> phase_start_rusage;
        std::unordered_map<std::string, IOStats> phase_start_io;
        std::unordered_map<std::string, std::vector<std::pair<double, double>>> phase_start_core_times;

        // Per-core timespec
        struct CoreTimespec
        {
            std::vector<struct timespec> start_times;
        };
        std::unordered_map<std::string, CoreTimespec> phase_core_timespec;

        // perf_event fd들 저장
        struct CoreEventFds
        {
            std::vector<int> user_time_fds;
            std::vector<int> system_time_fds;
            std::vector<int> io_wait_fds;
            std::vector<int> cpu_cycles_fds;
            std::vector<int> cpu_instructions_fds;
            std::vector<int> cpu_ref_cycles_fds;
        };
        std::unordered_map<std::string, CoreEventFds> phase_core_fds;

        // 모니터링할 코어 목록
        std::vector<int> monitored_cores;
    };

    // --------------------------------------------------------------------------
    // PerformanceMetrics
    // --------------------------------------------------------------------------
    class PerformanceMetrics
    {
    public:
        PerformanceMetrics() = default;
        ~PerformanceMetrics() = default;

        void RecordStats(const std::string &phase, const PerfStats &stats);
        void PrintStats() const;

    private:
        std::vector<std::pair<std::string, PerfStats>> phase_stats_;

        void PrintAverageStats(const std::vector<PerfStats>& stats_vec) const;
        void PrintSinglePhaseStat(const PerfStats &stats, const std::string &prefix = "") const;


    };

    // --------------------------------------------------------------------------
    // DecodingMetrics
    // --------------------------------------------------------------------------
    class DecodingMetrics
    {
    public:
        DecodingMetrics() = default;
        ~DecodingMetrics() = default;

        void RecordTimes(double inference_time_ms, double sampling_time_ms);
        // Print out final decoding metrics
        void PrintMetrics(int prefill_time);

    private:
        // Time to first token
        double first_decoding_time_ms_ = 0.0;
        bool first_token_recorded_ = false;

        // Accumulators
        double total_inference_time_ms_ = 0.0;
        double total_sampling_time_ms_ = 0.0;
        double total_decoding_time_ms_ = 0.0;
        int token_count_ = 0;
    };

 

    // --------------------------------------------------------------------------
    // TimerUtility: basic timing utility (renamed from ScopeTimer)
    // --------------------------------------------------------------------------
    class TimerUtility {
    protected:
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_;

    public:
        explicit TimerUtility(const std::string &name)
            : name_(name) {}

        void Start() {
            start_ = std::chrono::high_resolution_clock::now();
        }

        double Stop() const {
            auto end_ = std::chrono::high_resolution_clock::now();
            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count();
            return static_cast<double>(duration_us) / 1000.0;  
        }
    };

    // --------------------------------------------------------------------------
    // A scoped timer class that log the elapsed time when going out of scope
    // --------------------------------------------------------------------------
    class ScopeTimer {
    public:
        explicit ScopeTimer(double& out_ref, const std::string &name="")
            : out_ref_(out_ref), timer_(name) 
        {
            timer_.Start();
        }

        ~ScopeTimer() {
            out_ref_ = static_cast<double>(timer_.Stop());
        }

    private:
        double&      out_ref_;
        TimerUtility timer_;
    };


    // --------------------------------------------------------------------------
    // ScopeLogger: composed version with TimerUtility and profiling
    // --------------------------------------------------------------------------
    class ScopeLogger {
    public:
        ScopeLogger(const std::string &name,
                    ai_edge_torch::custom::profiler::PerformanceMonitor &monitor,
                    ai_edge_torch::custom::profiler::PerformanceMetrics &metrics,
                    struct rusage &usage_start,
                    struct rusage &usage_end,
                    bool log_stdout=true,
                    std::vector<ai_edge_torch::custom::profiler::RUsageRecord> *usage_records=nullptr,
                    double *out_duration_ms=nullptr)
            : timer_(name), name_(name), monitor_(monitor), metrics_(metrics),
            usage_start_(usage_start), usage_end_(usage_end), 
            log_stdout(log_stdout), usage_records_(usage_records), out_duration_ms_(out_duration_ms) {

            timer_.Start();
            getrusage(RUSAGE_SELF, &usage_start_);
            monitor_.start_phase(name_);
        }

        ~ScopeLogger() {
            duration_ms_ = timer_.Stop();
            getrusage(RUSAGE_SELF, &usage_end_);
            monitor_.end_phase(name_,stats_);

            if(log_stdout)
            {
                std::cout << "\n[INFO] " << name_ << " took " << duration_ms_ << " ms\n";
                ai_edge_torch::custom::profiler::print_rusage(usage_start_, usage_end_, name_);
            }

            if(usage_records_)
            {
                usage_records_->emplace_back(usage_start_, usage_end_, duration_ms_);
            }

            if(out_duration_ms_)
            {
                *out_duration_ms_ = duration_ms_;
            }

            metrics_.RecordStats(name_, stats_);
        }
        
    private:
        TimerUtility timer_;
        std::string name_;
        double duration_ms_;
        PerformanceMonitor& monitor_;
        PerformanceMetrics& metrics_;
        PerfStats stats_;
        struct rusage &usage_start_;
        struct rusage &usage_end_;
        bool log_stdout;
        std::vector<RUsageRecord>* usage_records_;
        double* out_duration_ms_;
    };

} // namespace ai_edge_torch::custom::profiler

#endif // AI_EDGE_TORCH_GENERATIVE_PROFILER_H_
