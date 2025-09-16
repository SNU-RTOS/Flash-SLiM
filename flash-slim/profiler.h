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
#include <sys/mman.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <stdexcept>
#include <errno.h>

#include <mutex>
#include <condition_variable>

#ifndef __NR_perf_event_open
#define __NR_perf_event_open 241
#endif

#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model.h"
#include "tflite/c/common.h"

#include "utils.h"

#define COLOR_GREEN "\033[1;32m"
#define COLOR_YELLOW "\033[1;33m"
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[1;31m"
#define COLOR_BLUE "\033[1;34m"
#define COLOR_CYAN "\033[1;36m"


// USDT Probes for eBPF tracing with Phase index
#ifdef EBPF_TRACE_ENABLED
#include <sys/sdt.h>

#define TRACE_LOGIC_START(phase_name) DTRACE_PROBE1(text_gen, phase_start, phase_name)
#define TRACE_LOGIC_END(phase_name)   DTRACE_PROBE1(text_gen, phase_end, phase_name)

#else
#define TRACE_LOGIC_START(phase_name)
#define TRACE_LOGIC_END(phase_name)
#endif

namespace custom::profiler
{

    // RUsage record structure
    struct RUsageRecord
    {
        rusage start_;
        rusage end_;
        double wall_time_ms_;
        std::string phase_name_;

        RUsageRecord() = default;

        RUsageRecord(const struct rusage &start, const struct rusage &end, const double wall_time_ms,
                     const std::string &phase_name = "")
            : start_(start), end_(end), wall_time_ms_(wall_time_ms), phase_name_(phase_name)
        {
        }
    };

    //////////////////////////////////////////////////////////////
    /* Functions */
    //////////////////////////////////////////////////////////////

    void detect_active_cores(std::vector<int> &cores);
    void print_rusage(rusage usage_start, rusage usage_end, double wall_time_ms, const std::string phase_name);
    void print_rusage_records(const std::vector<RUsageRecord> &records, const std::string &phase_name_prefix = "");
    void upload_tensors_for_all_subgraphs(tflite::Interpreter *interpreter);

    //////////////////////////////////////////////////////////////
    /* Classes */
    //////////////////////////////////////////////////////////////

    // --------------------------------------------------------------------------
    // GenAIMetrics
    // --------------------------------------------------------------------------
    class GenAIMetrics
    {
    public:
        GenAIMetrics() = default;
        ~GenAIMetrics() = default;

        void RecordPrefillTime(double prefill_time_ms);
        void RecordDecodingTime(
            double inference_time_ms,
            double sampling_time_ms,
            double detok_time_ms);

        void Print();

    private:
        // Time to first token
        double prefill_time_ms_ = 0.0;
        double first_decoding_time_ms_ = 0.0;
        bool first_token_recorded_ = false;

        // Accumulators
        double total_inference_time_ms_ = 0.0;
        double total_sampling_time_ms_ = 0.0;
        double total_decoding_time_ms_ = 0.0;
        double total_detokenization_time_ms_ = 0.0;
        int token_count_excluding_first_ = 0;
    };

    // --------------------------------------------------------------------------
    // TimerUtility: basic timing utility (renamed from ScopeTimer)
    // --------------------------------------------------------------------------
    class TimerUtility
    {
    protected:
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_;

    public:
        explicit TimerUtility(const std::string &name);
        void Start();
        double Stop() const;
    };

    // --------------------------------------------------------------------------
    // A scoped timer class that log the elapsed time when going out of scope
    // --------------------------------------------------------------------------
    class ScopeTimer
    {
    public:
        explicit ScopeTimer(double &out_ref, const std::string &name = "");
        ~ScopeTimer();

    private:
        double &out_ref_;
        TimerUtility timer_;
    };

    class ScopeEventHandler
    {
    public:
        explicit ScopeEventHandler(const std::string &name);
        ~ScopeEventHandler();
        std::string current_phase_name_="Idle";
    };

    // --------------------------------------------------------------------------
    // ScopeEventPrefetcher: signals phase start/end events but doesn't log time
    // --------------------------------------------------------------------------
    struct PhaseContext
    {
        std::mutex mutex;
        std::condition_variable signal_cv;
        std::atomic<bool> log_requested{false};
        std::atomic<bool> generation_done{false};
        std::string current_phase_name = "Idle";
        int phase_status = 0; // 0=start, 1=end
    };

    class ScopeEventPrefetcher
    {
    public:
        explicit ScopeEventPrefetcher(PhaseContext &ctx, const std::string &name);
        ~ScopeEventPrefetcher();

    private:
        PhaseContext &ctx_;
        std::unique_lock<std::mutex> lock_;
    };

    
    class ScopeEventListener
    {
    public:
        ScopeEventListener(PhaseContext &ctx,
                           bool log_stdout = true,
                           std::vector<RUsageRecord> *usage_records = nullptr);

        void Run();
        
    private:
        PhaseContext &ctx_;
        TimerUtility timer_;
        bool log_stdout_;
        std::vector<RUsageRecord> *usage_records_;
        struct rusage usage_start_;
        struct rusage usage_end_;
    };
} // namespace custom::profiler

#endif // AI_EDGE_TORCH_GENERATIVE_PROFILER_H_
