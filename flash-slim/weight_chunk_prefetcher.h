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

struct PrefetchChunkRange {
  size_t range_index = std::numeric_limits<size_t>::max();
  size_t io_order = 0;
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

  static constexpr int kPrefetchPlanCount = 2;

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

  struct PrefetchPlan {
    std::unordered_map<size_t, size_t> offset_to_index;  // origin_offset -> index
    std::vector<WeightChunkInfo> chunks;                 // index -> chunk metadata
    std::vector<PrefetchChunkRange> chunk_ranges;        // io_order-sorted ranges
  };

  struct PrefetchRequest {
    const PrefetchChunkRange* chunk_range = nullptr;
    void* buffer_base = nullptr;
    int direct_io_fd = -1;
    int buffer_index = -1;
  };

  WeightChunkPrefetcher() = default;
  ~WeightChunkPrefetcher();
  
  void ConfigureIOBuffers(void* buffer0, size_t size0, void* buffer1, size_t size1);

  void SetPrefetchPlan(PrefetchMode mode,
                       std::unordered_map<size_t, size_t>&& offset_to_index,
                       std::vector<WeightChunkInfo>&& chunks,
                       std::vector<PrefetchChunkRange>&& chunk_ranges);

  bool HasPrefetchPlan(PrefetchMode mode) const;

  const PrefetchPlan* GetPrefetchPlan(PrefetchMode mode) const;

  void UpdatePrefetcherMode(PrefetchMode mode) { prefetch_mode_ = mode; }

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

  const PrefetchChunkRange* GetChunkRangeByChunkIndex(PrefetchMode mode, size_t chunk_index) const;
  const PrefetchChunkRange* GetChunkRangeByIoOrder(PrefetchMode mode, size_t io_order) const;
  const PrefetchChunkRange* GetNextChunkRange(PrefetchMode mode, size_t current_io_order) const;
  
 private:
  struct PrefetchJob {
    const PrefetchChunkRange* chunk_range = nullptr;
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

  struct ChunkRangeIOState {
    std::mutex mutex;
    bool in_flight = false;
  };

  void WorkerLoop();
  void ApplyWorkerAffinity();
  void ResetRuntimeState();
  void MarkJobCompleted(const PrefetchJob& job, bool success);
  std::shared_ptr<ChunkIOState> GetChunkIOState(size_t chunk_index);
  std::shared_ptr<ChunkRangeIOState> GetChunkRangeState(PrefetchMode mode, size_t range_index);
  void RunAsyncWorkerLoop();
  void RunSyncWorkerLoop();

  PrefetchMode prefetch_mode_ = PrefetchMode::UNINITIALIZED;

  std::array<PrefetchPlan, kPrefetchPlanCount> prefetch_plans_{};  // [PREFILL=0, DECODE=1]
  std::array<bool, kPrefetchPlanCount> has_plan_{{false, false}};
  std::array<std::unordered_map<size_t, std::shared_ptr<ChunkRangeIOState>>, kPrefetchPlanCount> range_states_;

  std::vector<WeightChunkInfo> index_to_chunks_;
  std::unordered_map<size_t, std::shared_ptr<ChunkIOState>> index_to_chunk_states_;
  std::mutex range_state_mutex_;
  
  std::unique_ptr<WeightChunkIOEngine> io_engine_;

  std::array<void*, 2> registered_buffers_{nullptr, nullptr};
  std::array<size_t, 2> registered_buffer_sizes_{0, 0};
  bool buffers_configured_ = false;
  std::mutex io_config_mutex_;
  
  
  std::mutex chunk_state_mutex_;
  std::deque<PrefetchJob> io_job_queue_;
  std::thread io_worker_thread_;
  std::mutex io_worker_mutex_;
  std::atomic<bool> io_worker_running_{false};
  std::atomic<bool> io_worker_stop_requested_{false};
  std::condition_variable io_worker_cv_;
  std::vector<int> io_worker_cores_;
};

using PrefetchMode = WeightChunkPrefetcher::PrefetchMode;

}  // namespace streaming
}  // namespace flash_slim

#endif  // FLASH_SLIM_WEIGHT_CHUNK_PREFETCHER_H_
