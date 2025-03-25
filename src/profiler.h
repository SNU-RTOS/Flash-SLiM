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
        rusage start;
        rusage end;
    };

    //////////////////////////////////////////////////////////////
    /* Functions */
    //////////////////////////////////////////////////////////////

    // 현재 프로세스가 실행 중인 CPU 코어 목록 반환
    std::vector<int> detect_active_cores();
    // /proc/self/io에서 I/O 통계 읽기
    IOStats get_io_stats();

    // RUsage 출력 헬퍼
    void print_rusage(rusage usage_start, rusage usage_end, const std::string phase_name);
    void print_rusage_records(const std::vector<RUsageRecord> &records);

    // Tensors 강제 페이지 폴트 (upload) 해주는 함수
    void upload_tensors_for_all_subgraphs(tflite::Interpreter *interpreter);

    //////////////////////////////////////////////////////////////
    /* Classes */
    //////////////////////////////////////////////////////////////
    // --------------------------------------------------------------------------
    // A scoped timer class that prints the elapsed time when going out of scope
    // --------------------------------------------------------------------------
    class ScopeTimer
    {
    public:
        explicit ScopeTimer(const std::string &name)
            : name_(name),
              start_(std::chrono::high_resolution_clock::now()) {}

        ~ScopeTimer()
        {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
            std::cout << "\n[INFO] " << name_ << " took " << duration_ms << " ms\n";
        }

    private:
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_;
    };

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
                monitored_cores = detect_active_cores();

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
        PerfStats end_phase(const std::string &phase_name);

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
        void PrintSinglePhaseStat(const PerfStats &stats,
                                  const std::string &prefix = "") const;

    private:
        std::unordered_map<std::string, std::vector<PerfStats>> phase_stats;
    };

    // --------------------------------------------------------------------------
    // DecodingMetrics
    // --------------------------------------------------------------------------
    class DecodingMetrics
    {
    public:
        DecodingMetrics() = default;
        ~DecodingMetrics() = default;

        void StartDecoding();
        // Record times for each token
        //   - token_start: time point before inference/sampling starts for a token
        //   - inference_time_ms: how many ms were spent in model inference
        //   - sampling_time_ms : how many ms were spent in sampling the next token
        void RecordTimes(const std::chrono::high_resolution_clock::time_point &token_start,
                         double inference_time_ms, double sampling_time_ms);
        // Print out final decoding metrics
        void PrintMetrics();

    private:
        // Decode start time
        std::chrono::high_resolution_clock::time_point decode_start_;

        // Time to first token
        double time_to_first_token_ms_ = 0.0;
        bool first_token_recorded_ = false;

        // Accumulators
        double total_inference_time_ms_ = 0.0;
        double total_sampling_time_ms_ = 0.0;
        double total_decoding_time_ms_ = 0.0;
        int token_count_ = 0;
    };

} // namespace ai_edge_torch::custom::profiler

#endif // AI_EDGE_TORCH_GENERATIVE_PROFILER_H_
