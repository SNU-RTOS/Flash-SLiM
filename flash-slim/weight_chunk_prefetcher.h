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

#include "tflite/delegates/xnnpack/streaming_weight_cache.h"

namespace flash_slim {
namespace streaming {

using weight_chunk_info_t = tflite::xnnpack::StreamingWeightCacheProvider::weight_chunk_info_t;
using ProviderMode = tflite::xnnpack::StreamingWeightCacheProvider::ProviderMode;

class WeightChunkPrefetcher {

 public:
  struct PrefetchPlan {
    std::unordered_map<size_t, size_t> offset_to_index;  // origin_offset -> index
    std::vector<weight_chunk_info_t> chunks;             // index -> chunk metadata
  };

  enum class PrefetchMode {
    PREFILL,
    DECODE,
    UNINITIALIZED,
  };

  struct PrefetchRequest {
    const weight_chunk_info_t* chunk_info = nullptr;
    void* buffer_base = nullptr;
    int direct_io_fd = -1;
  };

  WeightChunkPrefetcher() = default;
  ~WeightChunkPrefetcher();
  

  void SetPrefetchPlan(PrefetchMode mode,
                       std::unordered_map<size_t, size_t>&& offset_to_index,
                       std::vector<weight_chunk_info_t>&& chunks);

  bool HasPrefetchPlan(PrefetchMode mode) const;

  const PrefetchPlan* GetPrefetchPlan(PrefetchMode mode) const;

  void BuildIndexToChunksFromPlans();

  const std::vector<weight_chunk_info_t>& GetIndexToChunks() const { return index_to_chunks_; }

  void UpdatePrefetcherMode(PrefetchMode mode) { prefetch_mode_ = mode; }

  static inline int PrefetchModeToIndex(PrefetchMode mode) {
        switch (mode) {
        case PrefetchMode::PREFILL:
        return 0;
        case PrefetchMode::DECODE:
        return 1;
        default:
        return -1;
    }
  }

  PrefetchMode GetPrefetcherMode() const { return prefetch_mode_; }
  
  std::string GetPrefetcherModeString() const;

  void StartWorker();
  void StopWorker();

  // Assign a target CPU core for the worker thread. Passing a negative value
  // clears the affinity.
  void SetWorkerThreadCpu(int core_id);

  const weight_chunk_info_t* LookupChunkInfo(PrefetchMode mode, size_t offset) const;

  bool Submit(const PrefetchRequest& request);
  bool WaitReady(const weight_chunk_info_t* chunk_info);
  bool WaitReady(PrefetchMode mode, size_t offset);

  std::optional<size_t> GetNextChunkIndex(PrefetchMode mode, size_t current_chunk_index) const;
  const weight_chunk_info_t* GetChunkInfoByIndex(size_t chunk_index) const;

 private:
  struct PrefetchJob {
    const weight_chunk_info_t* chunk_info = nullptr;
    void* buffer_base = nullptr;
    int direct_io_fd = -1;
  };

  struct ChunkIOState {
    std::mutex mutex;
    std::condition_variable cv;
    bool in_flight = false;
    bool ready = false;
    bool success = false;
  };

  void WorkerLoop();
  void ApplyWorkerAffinity();
  void ResetRuntimeState();
  void MarkJobCompleted(const PrefetchJob& job, bool success);
  std::shared_ptr<ChunkIOState> GetChunkIOState(size_t chunk_index);

  PrefetchMode prefetch_mode_ = PrefetchMode::UNINITIALIZED;

  std::array<PrefetchPlan, 2> prefetch_plans_{};  // [PREFILL=0, DECODE=1]
  std::array<bool, 2> has_plan_{{false, false}};

  std::vector<weight_chunk_info_t> index_to_chunks_;
  std::unordered_map<size_t, std::shared_ptr<ChunkIOState>> index_to_chunk_states_;
  
  
  std::mutex chunk_state_mutex_;
  std::deque<PrefetchJob> io_job_queue_;
  std::thread io_worker_thread_;
  std::mutex io_worker_mutex_;
  std::atomic<bool> io_worker_running_{false};
  std::atomic<bool> io_worker_stop_requested_{false};
  std::condition_variable io_worker_cv_;
  std::optional<int> io_worker_core_id_;
};

using PrefetchMode = WeightChunkPrefetcher::PrefetchMode;

}  // namespace streaming
}  // namespace flash_slim

#endif  // FLASH_SLIM_WEIGHT_CHUNK_PREFETCHER_H_


