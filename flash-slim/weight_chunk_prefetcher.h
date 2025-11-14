// Copyright 2025 Flash-SLiM Research Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef FLASH_SLIM_WEIGHT_CHUNK_PREFETCHER_H_
#define FLASH_SLIM_WEIGHT_CHUNK_PREFETCHER_H_

#include <atomic>
#include <array>
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <optional>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <limits>

#include "tflite/delegates/xnnpack/streaming_weight_cache.h"
#include "weight_chunk_io_engine.h"

namespace flash_slim {
namespace streaming {

using ProviderMode = tflite::xnnpack::StreamingWeightCacheProvider::ProviderMode;

struct WeightChunkInfo {
    size_t chunk_index;
    size_t aligned_offset;
    size_t offset_adjust; //abs_offset(mmap_buffer_base_offset_ + offset) - aligned_offset
    size_t aligned_size;
    size_t origin_offset;
    size_t origin_size;
    size_t weights_id;
};

struct WeightChunkGroupInfo {
  size_t group_index = std::numeric_limits<size_t>::max();
  size_t start_origin_offset = 0;
  size_t start_aligned_offset = 0;
  size_t total_aligned_size = 0;
  std::vector<size_t> chunk_indices;
  std::vector<size_t> chunk_relative_offsets;
};

class WeightChunkPrefetcher {

 public:
  enum class PrefetchMode {
    PREFILL,
    DECODE,
    UNINITIALIZED,
  };

  enum class IOEngineType {
    IO_URING,
    PARALLEL_PREAD,
    UNINITIALIZED,
  };

  struct PrefetchPlan {
    std::unordered_map<size_t, size_t> offset_to_index;  // origin_offset -> index
    std::vector<WeightChunkInfo> chunks;                 // index -> chunk metadata
  std::vector<WeightChunkGroupInfo> chunk_groups;        // group_index-sorted chunk groups
  };

  struct PrefetchRequest {
    const WeightChunkGroupInfo* chunk_group = nullptr;
    void* buffer_base = nullptr;
    int direct_io_fd = -1;
    int buffer_index = -1;
    PrefetchMode mode = PrefetchMode::UNINITIALIZED;
  };

  struct ChunkIOState {
    std::mutex mutex;
    std::condition_variable cv;
    bool in_flight = false;
    bool ready = false;
    bool success = false;
  };

  struct ChunkGroupIOState {
    std::mutex mutex;
    bool in_flight = false;
  };

  WeightChunkPrefetcher() = default;
  ~WeightChunkPrefetcher();
  
  // Configure IO Engine type and parameters
  void ConfigureIOEngine(
    const std::string& io_engine_mode_str,
    unsigned iouring_ring_depth, size_t iouring_subread_bytes,
    size_t pread_min_block_size, size_t pread_max_threads);

  IOEngineType GetIOEngineType() const { return io_engine_type_; }

  void ConfigureIOBuffers(void* buffer0, size_t size0, void* buffer1, size_t size1);

  void SetPrefetchPlan(PrefetchMode mode,
                       std::unordered_map<size_t, size_t>&& offset_to_index,
                       std::vector<WeightChunkInfo>&& chunks,
                       std::vector<WeightChunkGroupInfo>&& chunk_groups);

  bool HasPrefetchPlan(PrefetchMode mode) const;

  const PrefetchPlan* GetPrefetchPlan(PrefetchMode mode) const;

  void UpdatePrefetcherMode(PrefetchMode mode) { prefetch_mode_ = mode; }

  static constexpr const char* PrefetchModeName(PrefetchMode mode) {
    switch (mode) {
      case PrefetchMode::PREFILL:
        return "PREFILL";
      case PrefetchMode::DECODE:
        return "DECODE";
      case PrefetchMode::UNINITIALIZED:
        return "UNINITIALIZED";
      default:
        return "UNKNOWN";
    }
  }

  inline int PrefetchModeToIndex(PrefetchMode mode) const {
    switch (mode) {
      case PrefetchMode::PREFILL:
        return 0;
      case PrefetchMode::DECODE:
        return 1;
      default:
        return -1;
    }
  }

  inline std::optional<PrefetchMode> IndexToPrefetchMode(int index) const {
    switch (index) {
      case 0:
        return PrefetchMode::PREFILL;
      case 1:
        return PrefetchMode::DECODE;
      default:
        return std::nullopt;
    }
  }

  inline std::string IndexToPrefetchModeString(int index) const {
    const auto mode = IndexToPrefetchMode(index);
    if (!mode.has_value()) {
      return "UNKNOWN";
    }
    return std::string(PrefetchModeName(*mode));
  }

  PrefetchMode GetPrefetchMode() const { return prefetch_mode_; }
  int GetPrefetchModeIndex() const { return PrefetchModeToIndex(prefetch_mode_);}
  std::string GetPrefetchModeString() const;

  bool Submit(const PrefetchRequest& request);
  bool WaitReady(const WeightChunkInfo* chunk_info);
  void StartWorker();
  void StopWorker();

  // Configure the worker thread's CPU affinity. Passing an empty `cores`
  // vector clears the affinity preference. The thread will adopt the new
  // affinity upon the next `StartWorker()` call (or immediately if running).
  void SetWorkerThreadAffinity(const std::vector<int>& cores);
  
  void BuildIndexToChunksFromPlans();

  const std::vector<WeightChunkInfo>& GetIndexToChunks() const { return index_to_chunks_; }

  const WeightChunkInfo* LookupChunkInfo(PrefetchMode mode, size_t offset) const;

  const WeightChunkInfo* GetChunkInfoByIndex(size_t chunk_index) const;
  const WeightChunkGroupInfo* GetChunkGroupByChunkIndex(PrefetchMode mode, size_t chunk_index) const;
  const WeightChunkGroupInfo* GetNextChunkGroup(PrefetchMode mode, size_t current_io_order,
                                               PrefetchMode* next_mode) const;
  
 private:
  static constexpr int kPrefetchPlanCount = 2;
  
  // IO Engine configuration
  IOEngineType io_engine_type_ = IOEngineType::UNINITIALIZED;
  
  // io_uring specific parameters
  unsigned iouring_ring_depth_ = 64;
  size_t iouring_subread_bytes_ = 512 * 1024;
  
  // parallel_pread specific parameters
  size_t pread_min_block_size_ = 512 * 1024;
  size_t pread_max_threads_ = 4;

  using PrefetchJob = PrefetchRequest;

  void ApplyWorkerAffinity();
  void ResetRuntimeState();
  void MarkJobCompleted(const PrefetchJob& job, bool success);
  
  // io_uring specific methods
  void StartIoUringWorker();
  void RunIoUringSubmissionLoop();
  void RunIoUringCompletionLoop();
  
  // parallel_pread specific methods
  void StartParallelPreadWorker();
  void RunParallelPreadWorkerLoop();
  bool ExecuteParallelPread(int fd, void* buffer, size_t size, off_t offset);
  std::shared_ptr<ChunkIOState> GetChunkIOState(size_t chunk_index);
  std::shared_ptr<ChunkGroupIOState> GetChunkGroupIOState(PrefetchMode mode, size_t group_index);
  void ResetChunkStates();

  PrefetchMode prefetch_mode_ = PrefetchMode::UNINITIALIZED;

  std::array<PrefetchPlan, kPrefetchPlanCount> prefetch_plans_{};  // [PREFILL=0, DECODE=1]
  std::array<bool, kPrefetchPlanCount> has_plan_{{false, false}};
  
  std::vector<WeightChunkInfo> index_to_chunks_;
  std::unordered_map<size_t, std::shared_ptr<ChunkIOState>> index_to_chunk_io_states_;
  std::array<std::unordered_map<size_t, std::shared_ptr<ChunkGroupIOState>>, kPrefetchPlanCount> index_to_group_io_states_;
  
  // io_uring specific members
  std::unique_ptr<WeightChunkIOEngine> io_engine_;
  std::array<void*, 2> io_registered_buffers_{nullptr, nullptr};
  std::array<size_t, 2> io_registered_buffer_sizes_{0, 0};
  bool io_buffers_configured_ = false;
  
  // Worker state
  std::deque<PrefetchJob> io_job_queue_;
  std::thread io_submission_thread_;
  std::thread io_completion_thread_;  // Only used by io_uring
  std::vector<int> io_worker_cores_;
  
  std::atomic<bool> io_worker_running_{false};
  std::atomic<bool> io_worker_stop_requested_{false};
  
  // Simplified mutex design: 3 mutexes only
  mutable std::mutex state_mutex_;        // Protects all shared state (chunks, groups, config, affinity)
  std::mutex queue_mutex_;                // Protects job queue only
  std::mutex engine_mutex_;               // Protects io_engine_ operations (io_uring only)
  
  std::condition_variable io_submission_cv_;   // Submission loop 전용
  std::condition_variable io_completion_cv_;   // Completion loop 전용 (io_uring only)
  
};

using PrefetchMode = WeightChunkPrefetcher::PrefetchMode;

}  // namespace streaming
}  // namespace flash_slim

#endif  // FLASH_SLIM_WEIGHT_CHUNK_PREFETCHER_H_
