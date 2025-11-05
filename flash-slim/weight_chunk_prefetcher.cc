// Copyright 2025 Flash-SLiM Research Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <iostream>
#include <thread>
#include <limits>
#include <optional>
#include <utility>
#include <unistd.h>
#include <vector>

#include <pthread.h>

#include "tflite/minimal_logging.h"
#include "weight_chunk_prefetcher.h"

namespace flash_slim {
namespace streaming {

static constexpr uint32_t kPlanIndexShift = 30;
static constexpr size_t kPlanIndexMask = (static_cast<size_t>(1) << kPlanIndexShift) - static_cast<size_t>(1);

static inline size_t EncodeRangeId(int plan_index, size_t range_index) {
  return (static_cast<size_t>(plan_index) << kPlanIndexShift) | range_index;
}

static inline int DecodePlanIndex(size_t encoded) {
  return static_cast<int>(encoded >> kPlanIndexShift);
}

static inline size_t DecodeRangeIndex(size_t encoded) {
  return encoded & kPlanIndexMask;
}

/** =============== WeightChunkPrefetcher Class =============== */
WeightChunkPrefetcher::~WeightChunkPrefetcher() { StopWorker(); }

std::string WeightChunkPrefetcher::GetPrefetchModeString() const {
  return std::string(PrefetchModeName(prefetch_mode_));
}


void WeightChunkPrefetcher::ConfigureIOEngine(unsigned ring_depth, size_t subread_bytes,
                                               size_t min_block_size, size_t max_threads) {
  ring_depth_ = ring_depth;
  subread_bytes_ = subread_bytes;
  min_block_size_ = min_block_size;
  max_threads_ = max_threads;
}

void WeightChunkPrefetcher::ConfigureIOBuffers(void* buffer0, size_t size0, void* buffer1,
                                               size_t size1) {
  std::lock_guard<std::mutex> lock(io_config_mutex_);
  io_registered_buffers_[0] = buffer0;
  io_registered_buffer_sizes_[0] = size0;
  io_registered_buffers_[1] = buffer1;
  io_registered_buffer_sizes_[1] = size1;
  io_buffers_configured_ = buffer0 != nullptr && buffer1 != nullptr && size0 > 0 && size1 > 0;
}

void WeightChunkPrefetcher::SetWorkerThreadAffinity(const std::vector<int>& cores) {
    std::lock_guard<std::mutex> lock(io_worker_mutex_);
    io_worker_cores_ = cores;
}

void WeightChunkPrefetcher::ApplyWorkerAffinity() {
  std::lock_guard<std::mutex> lock(io_worker_mutex_);
  if (io_worker_cores_.empty()) {
    return;
  }

  cpu_set_t set;
  CPU_ZERO(&set);
  for (int core : io_worker_cores_) {
    if (core >= 0) {
      CPU_SET(static_cast<unsigned long>(core), &set);
    }
  }
  
  if (io_submission_thread_.joinable()) {
    const int rc1 = pthread_setaffinity_np(io_submission_thread_.native_handle(), sizeof(set), &set);
    if (rc1 != 0) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                      "WeightChunkPrefetcher: failed to set submission thread affinity (errno=%d)", rc1);
    }
  }
  
  if (io_completion_thread_.joinable()) {
    const int rc2 = pthread_setaffinity_np(io_completion_thread_.native_handle(), sizeof(set), &set);
    if (rc2 != 0) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                      "WeightChunkPrefetcher: failed to set completion thread affinity (errno=%d)", rc2);
    }
  }
}

void WeightChunkPrefetcher::StartWorker() {
  bool expected = false;
  if (!io_worker_running_.compare_exchange_strong(expected, true)) {
    return;  // already running
  }

  io_worker_stop_requested_.store(false, std::memory_order_relaxed);

  void* buffer0 = nullptr;
  void* buffer1 = nullptr;
  size_t size0 = 0;
  size_t size1 = 0;
  bool configured = false;
  {
    std::lock_guard<std::mutex> lock(io_config_mutex_);
    buffer0 = io_registered_buffers_[0];
    buffer1 = io_registered_buffers_[1];
    size0 = io_registered_buffer_sizes_[0];
    size1 = io_registered_buffer_sizes_[1];
    configured = io_buffers_configured_;
  }

  if (!configured) {
    io_worker_running_.store(false, std::memory_order_relaxed);
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                    "WeightChunkPrefetcher: IO buffers not configured; worker start deferred");
    return;
  }

  if (!io_engine_) {
    io_engine_ = std::make_unique<WeightChunkIOEngine>();
  }
  if (io_engine_ && !io_engine_->IsReady()) {
    printf("[INFO] Initializing io_uring engine for WeightChunkPrefetcher...\n");
    if (!io_engine_->Initialize(ring_depth_, 
                                subread_bytes_, 
                                buffer0, size0, 
                                buffer1, size1)) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                      "WeightChunkPrefetcher: failed to initialize io_uring engine; using sync IO");
    }
  }

  if (io_engine_ && io_engine_->IsReady()) {
    std::cout << "[INFO] WeightChunkPrefetcher: starting io_uring submission and completion threads" << std::endl;
    io_submission_thread_ = std::thread([this]() {
      RunIoUringSubmissionLoop();
    });
    io_completion_thread_ = std::thread([this]() {
      RunIoUringCompletionLoop();
    });
  } else {
    std::cout << "[INFO] WeightChunkPrefetcher: starting parallel pread worker thread" << std::endl;
    io_submission_thread_ = std::thread([this]() {
      RunParallelPreadWorkerLoop();
    });
  }

  ApplyWorkerAffinity();
}

void WeightChunkPrefetcher::StopWorker() {
  if (!io_worker_running_.exchange(false)) {
    return;  // not running
  }

  {
    std::lock_guard<std::mutex> lock(io_worker_mutex_);
    io_worker_stop_requested_.store(true, std::memory_order_relaxed);
    io_job_queue_.clear();
  }
  io_worker_cv_.notify_all();

  if (io_submission_thread_.joinable()) {
    io_submission_thread_.join();
  }
  if (io_completion_thread_.joinable()) {
    io_completion_thread_.join();
  }

  io_worker_stop_requested_.store(false, std::memory_order_relaxed);

  if (io_engine_) {
    io_engine_->Shutdown();
  }

  ResetChunkStates();
}

bool WeightChunkPrefetcher::Submit(const PrefetchRequest& request) {
  const WeightChunkGroupInfo* group = request.chunk_group;
  if (!request.buffer_base || request.direct_io_fd < 0 || request.buffer_index < 0 || !group ||
      group->chunk_indices.empty()) {
    return false;
  }

  PrefetchMode mode = request.mode;
  if (mode == PrefetchMode::UNINITIALIZED) {
    mode = prefetch_mode_;
  }
  const int plan_idx = PrefetchModeToIndex(mode);
  if (plan_idx < 0 || !has_plan_[plan_idx]) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(io_worker_mutex_);
    auto group_state = GetChunkGroupIOState(mode, group->group_index);
    {
      std::lock_guard<std::mutex> group_lock(group_state->mutex);
      if (group_state->in_flight) {
        return true;
      }
      group_state->in_flight = true;
    }

    std::vector<std::shared_ptr<ChunkIOState>> chunk_states;
    chunk_states.reserve(group->chunk_indices.size());
    for (const size_t chunk_index : group->chunk_indices) {
      chunk_states.push_back(GetChunkIOState(chunk_index));
    }

    bool already_inflight = false;
    for (auto& state : chunk_states) {
      std::lock_guard<std::mutex> state_lock(state->mutex);
      if (state->in_flight) {
        already_inflight = true;
        break;
      }
    }

    if (already_inflight) {
      std::lock_guard<std::mutex> group_lock(group_state->mutex);
      group_state->in_flight = false;
      return true;
    }

    for (auto& state : chunk_states) {
      std::lock_guard<std::mutex> state_lock(state->mutex);
      state->in_flight = true;
      state->ready = false;
      state->success = false;
    }

    {
      PrefetchJob job;
      job.chunk_group = group;
      job.buffer_base = request.buffer_base;
      job.direct_io_fd = request.direct_io_fd;
      job.buffer_index = request.buffer_index;
      job.mode = mode;
      io_job_queue_.push_back(std::move(job));
    }
  }

  io_worker_cv_.notify_one();
  return true;
}

bool WeightChunkPrefetcher::WaitReady(const WeightChunkInfo* chunk_info) {
 
  if (!chunk_info) {
    return false;
  }

  auto state = GetChunkIOState(chunk_info->chunk_index);
  std::unique_lock<std::mutex> lock(state->mutex);
  state->cv.wait(lock, [&]() { return state->ready || !state->in_flight; });
  const bool success = state->success;
  state->ready = false;
  state->in_flight = false;
  return success;
}

void WeightChunkPrefetcher::RunIoUringSubmissionLoop() {
  while (true) {
    PrefetchJob job;
    bool stop_requested = false;
    
    {
      std::unique_lock<std::mutex> lock(io_worker_mutex_);
      io_worker_cv_.wait(lock, [this]() {
        return io_worker_stop_requested_.load(std::memory_order_relaxed) || !io_job_queue_.empty();
      });

      stop_requested = io_worker_stop_requested_.load(std::memory_order_relaxed);
      
      if (stop_requested && io_job_queue_.empty()) {
        break;
      }
      
      if (!io_job_queue_.empty()) {
        job = std::move(io_job_queue_.front());
        io_job_queue_.pop_front();
      } else {
        continue;
      }
    }

    const WeightChunkGroupInfo* group = job.chunk_group;
    if (!group || job.buffer_index < 0) {
      MarkJobCompleted(job, false);
      continue;
    }

    const int plan_index = PrefetchModeToIndex(job.mode);
    if (plan_index < 0) {
      MarkJobCompleted(job, false);
      continue;
    }

    WeightChunkIOEngine::IORequest request;
    request.range_index = EncodeRangeId(plan_index, group->group_index);
    request.aligned_offset = group->start_aligned_offset;
    request.aligned_size = group->total_aligned_size;
    request.buffer_base = job.buffer_base;
    request.direct_io_fd = job.direct_io_fd;
    request.buffer_index = job.buffer_index;

    const bool submitted = io_engine_->Submit(request);
    if (!submitted) {
      std::lock_guard<std::mutex> requeue_lock(io_worker_mutex_);
      io_job_queue_.push_front(std::move(job));
      io_worker_cv_.notify_one();
    }
  }
}

void WeightChunkPrefetcher::RunIoUringCompletionLoop() {
  while (true) {
    std::vector<WeightChunkIOEngine::Completion> completions;
    
    bool stop_requested = false;
    bool has_pending = false;
    bool queue_empty = false;
    
    {
      std::lock_guard<std::mutex> lock(io_worker_mutex_);
      stop_requested = io_worker_stop_requested_.load(std::memory_order_relaxed);
      has_pending = io_engine_ && io_engine_->HasPending();
      queue_empty = io_job_queue_.empty();
    }

    const bool should_wait = !stop_requested || has_pending;
    io_engine_->DrainCompletions(&completions, should_wait);

    for (const auto& completion : completions) {
      const int plan_index = DecodePlanIndex(completion.range_index);
      if (plan_index < 0 || plan_index >= 2 || !has_plan_[plan_index]) {
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                        "WeightChunkPrefetcher: invalid completion plan index=%d", plan_index);
        continue;
      }
      
      const size_t range_index = DecodeRangeIndex(completion.range_index);
      const auto& plan = prefetch_plans_[plan_index];
      if (range_index >= plan.chunk_groups.size()) {
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                        "WeightChunkPrefetcher: missing chunk group metadata for group_index=%zu",
                        range_index);
        continue;
      }
      
      const auto mode = IndexToPrefetchMode(plan_index);
      if (!mode.has_value()) {
        continue;
      }
      
      PrefetchJob job_snapshot;
      job_snapshot.chunk_group = &plan.chunk_groups[range_index];
      job_snapshot.mode = *mode;
      MarkJobCompleted(job_snapshot, completion.success);
    }

    {
      std::lock_guard<std::mutex> lock(io_worker_mutex_);
      stop_requested = io_worker_stop_requested_.load(std::memory_order_relaxed);
      has_pending = io_engine_ && io_engine_->HasPending();
      queue_empty = io_job_queue_.empty();
    }

    if (stop_requested && queue_empty && !has_pending) {
      break;
    }
  }
}

void WeightChunkPrefetcher::RunParallelPreadWorkerLoop() {
  while (true) {
    PrefetchJob job;
    {
      std::unique_lock<std::mutex> lock(io_worker_mutex_);
      io_worker_cv_.wait(lock, [this]() {
        return io_worker_stop_requested_.load(std::memory_order_relaxed) || !io_job_queue_.empty();
      });

      if (io_worker_stop_requested_.load(std::memory_order_relaxed) && io_job_queue_.empty()) {
        break;
      }

      job = std::move(io_job_queue_.front());
      io_job_queue_.pop_front();
    }

    const WeightChunkGroupInfo* group = job.chunk_group;
    const bool success = group &&
                         ExecuteParallelPread(job.direct_io_fd, job.buffer_base, group->total_aligned_size,
                                   static_cast<off_t>(group->start_aligned_offset));

    MarkJobCompleted(job, success);
  }
}

std::shared_ptr<WeightChunkPrefetcher::ChunkIOState> WeightChunkPrefetcher::GetChunkIOState(size_t chunk_index) {
  std::lock_guard<std::mutex> lock(chunk_io_state_mutex_);
  auto it = index_to_chunk_io_states_.find(chunk_index);
  if (it == index_to_chunk_io_states_.end() || !(it->second)) {
    auto state = std::make_shared<ChunkIOState>();
    it = index_to_chunk_io_states_.emplace(chunk_index, std::move(state)).first;
  }
  return it->second;
}

std::shared_ptr<WeightChunkPrefetcher::ChunkGroupIOState> WeightChunkPrefetcher::GetChunkGroupIOState(
    PrefetchMode mode, size_t group_index) {
  std::lock_guard<std::mutex> lock(group_io_state_mutex_);
  const int plan_idx = PrefetchModeToIndex(mode);
  if (plan_idx < 0 || plan_idx >= static_cast<int>(index_to_group_io_states_.size())) {
    static auto dummy = std::make_shared<ChunkGroupIOState>();
    return dummy;
  }
  auto& map = index_to_group_io_states_[plan_idx];
  auto it = map.find(group_index);
  if (it == map.end() || !(it->second)) {
    auto state = std::make_shared<ChunkGroupIOState>();
    it = map.emplace(group_index, std::move(state)).first;
  }
  return it->second;
}

void WeightChunkPrefetcher::ResetChunkStates() {
  std::lock_guard<std::mutex> states_lock(chunk_io_state_mutex_);
  for (auto& kv : index_to_chunk_io_states_) {
    const auto& state = kv.second;
    if (!state) {
      continue;
    }
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->in_flight = false;
      state->ready = false;
      state->success = false;
    }
    state->cv.notify_all();
  }
}

void WeightChunkPrefetcher::ResetRuntimeState() {
  {
    std::lock_guard<std::mutex> lock(io_worker_mutex_);
    io_job_queue_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(group_io_state_mutex_);
    for (auto& states : index_to_group_io_states_) {
      states.clear();
    }
  }
  ResetChunkStates();
}

void WeightChunkPrefetcher::MarkJobCompleted(const PrefetchJob& job, bool success) {
  const WeightChunkGroupInfo* group = job.chunk_group;
  if (!group) {
    return;
  }

//   std::cout << "[WeightChunkPrefetcher] MarkJobCompleted: io_order=" << group->io_order
//             << " success=" << success << std::endl;

  for (const size_t chunk_index : group->chunk_indices) {
    auto state = GetChunkIOState(chunk_index);
    if (!state) {
      continue;
    }
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->success = success;
      state->ready = true;
      state->in_flight = false;
    }
    state->cv.notify_all();
  }

  auto group_state = GetChunkGroupIOState(job.mode, group->group_index);
  if (group_state) {
    std::lock_guard<std::mutex> group_lock(group_state->mutex);
    group_state->in_flight = false;
  }

  if (!success && !group->chunk_indices.empty()) {
    const auto* info = GetChunkInfoByIndex(group->chunk_indices.front());
    const size_t origin_offset = info ? info->origin_offset : 0;
    const size_t chunk_index = info ? info->chunk_index : group->chunk_indices.front();
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
          "WeightChunkPrefetcher: prefetch failed (group_index=%zu, chunk_index=%zu, "
          "origin_offset=%zu)",
          group->group_index, chunk_index, origin_offset);
  }
}

void WeightChunkPrefetcher::SetPrefetchPlan(
  PrefetchMode mode, std::unordered_map<size_t, size_t>&& offset_to_index,
  std::vector<WeightChunkInfo>&& chunks,
  std::vector<WeightChunkGroupInfo>&& chunk_groups) {

  const int idx = PrefetchModeToIndex(mode);

  if (idx < 0) {
    return;
  }

  auto& plan = prefetch_plans_[idx];
  plan.offset_to_index = std::move(offset_to_index);
  plan.chunks = std::move(chunks);
  plan.chunk_groups = std::move(chunk_groups);

  std::sort(plan.chunk_groups.begin(), plan.chunk_groups.end(),
            [](const WeightChunkGroupInfo& lhs, const WeightChunkGroupInfo& rhs) {
              return lhs.group_index < rhs.group_index;
            });

  for (size_t group_index = 0; group_index < plan.chunk_groups.size(); ++group_index) {
    auto& group = plan.chunk_groups[group_index];
    if (group_index > kPlanIndexMask) {
      TFLITE_LOG_PROD(
          tflite::TFLITE_LOG_WARNING,
          "WeightChunkPrefetcher: group index %zu exceeds encoding capacity (%zu). Results may be "
          "undefined.",
          group_index, kPlanIndexMask);
    }
    if (group.group_index != group_index) {
      TFLITE_LOG_PROD(
          tflite::TFLITE_LOG_WARNING,
          "WeightChunkPrefetcher: adjusting group_index from %zu to %zu to maintain contiguity.",
          group.group_index, group_index);
      group.group_index = group_index;
    }
  }

  has_plan_[idx] = true;
  index_to_group_io_states_[idx].clear();
  ResetRuntimeState();
}

bool WeightChunkPrefetcher::HasPrefetchPlan(PrefetchMode mode) const {
  const int idx = PrefetchModeToIndex(mode);
  return idx >= 0 && has_plan_[idx];
}

const WeightChunkPrefetcher::PrefetchPlan* WeightChunkPrefetcher::GetPrefetchPlan( PrefetchMode mode) const {
  const int idx = PrefetchModeToIndex(mode);
  if (idx < 0 || !has_plan_[idx]) {
    return nullptr;
  }
  return &prefetch_plans_[idx];
}

void WeightChunkPrefetcher::BuildIndexToChunksFromPlans() {
  size_t max_index = 0;
  bool seen_any = false;
  for (int i = 0; i < 2; ++i) {
    if (!has_plan_[i]) {
      continue;
    }
    for (const auto& chunk : prefetch_plans_[i].chunks) {
      if (!seen_any || chunk.chunk_index > max_index) {
        max_index = chunk.chunk_index;
      }
      seen_any = true;
    }
  }

  if (!seen_any) {
    index_to_chunks_.clear();
    return;
  }

  WeightChunkInfo tmp_chunk;
  tmp_chunk.chunk_index = SIZE_MAX;
  tmp_chunk.origin_offset = 0;
  tmp_chunk.origin_size = 0;
  tmp_chunk.aligned_offset = 0;
  tmp_chunk.aligned_size = 0;
  tmp_chunk.offset_adjust = 0;
  tmp_chunk.weights_id = 0;

  index_to_chunks_.assign(max_index + 1, tmp_chunk);

  for (int i = 0; i < 2; ++i) {
    if (!has_plan_[i]) {
      continue;
    }
    for (const auto& chunk : prefetch_plans_[i].chunks) {
      const size_t idx = chunk.chunk_index;
      auto& destination = index_to_chunks_[idx];
      if (destination.chunk_index == SIZE_MAX) {
        destination = chunk;
      } else if (destination.origin_offset != chunk.origin_offset) {
        TFLITE_LOG_PROD(
            tflite::TFLITE_LOG_WARNING,
            "WeightChunkPrefetcher::BuildIndexToChunksFromPlans: conflict for "
            "chunk_index=%zu: existing origin_offset=%zu, new origin_offset=%zu."
            " Keeping existing.",
            idx, destination.origin_offset, chunk.origin_offset);
      }
    }
  }
}

const WeightChunkInfo* WeightChunkPrefetcher::LookupChunkInfo(PrefetchMode mode,
                                                                 size_t offset) const {
  const int idx = PrefetchModeToIndex(mode);
  if (idx < 0 || !has_plan_[idx]) {
    return nullptr;
  }

  const auto& plan = prefetch_plans_[idx];
  const auto it = plan.offset_to_index.find(offset);
  if (it == plan.offset_to_index.end()) {
    return nullptr;
  }

  const size_t index = it->second;
  if (index >= index_to_chunks_.size()) {
    return nullptr;
  }

  const auto& candidate = index_to_chunks_[index];
  if (candidate.chunk_index == std::numeric_limits<size_t>::max()) {
    return nullptr;
  }
  return &candidate;
}

const WeightChunkInfo* WeightChunkPrefetcher::GetChunkInfoByIndex(size_t chunk_index) const {
  if (chunk_index >= index_to_chunks_.size()) {
    return nullptr;
  }

  const auto& info = index_to_chunks_[chunk_index];
  if (info.chunk_index == std::numeric_limits<size_t>::max()) {
    return nullptr;
  }

  return &info;
}

const WeightChunkGroupInfo* WeightChunkPrefetcher::GetChunkGroupByChunkIndex(
    PrefetchMode mode, size_t chunk_index) const {
  const int plan_idx = PrefetchModeToIndex(mode);
  if (plan_idx < 0 || !has_plan_[plan_idx]) {
    return nullptr;
  }
  const auto& plan = prefetch_plans_[plan_idx];
  for (size_t i = 0; i < plan.chunk_groups.size(); ++i) {
    const auto& group = plan.chunk_groups[i];
    if (std::find(group.chunk_indices.begin(), group.chunk_indices.end(), chunk_index) !=
        group.chunk_indices.end()) {
      return &plan.chunk_groups[i];
    }
  }

  TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                  "WeightChunkPrefetcher: chunk_index=%zu not found in group list for mode %d",
                  chunk_index, plan_idx);
  return nullptr;
}


const WeightChunkGroupInfo* WeightChunkPrefetcher::GetNextChunkGroup(
    PrefetchMode mode, size_t current_group_index, PrefetchMode* next_mode) const {

  const int plan_idx = PrefetchModeToIndex(mode);
  if (plan_idx < 0 || !has_plan_[plan_idx]) {
    return nullptr;
  }

  const auto& plan = prefetch_plans_[plan_idx];
  if (plan.chunk_groups.empty()) {
    return nullptr;
  }

  PrefetchMode resolved_mode = mode;
  const size_t group_count = plan.chunk_groups.size();
  const size_t next_index = current_group_index + 1;

  if (mode == PrefetchMode::PREFILL && next_index >= group_count) {
    const int decode_idx = PrefetchModeToIndex(PrefetchMode::DECODE);
    if (decode_idx >= 0 && has_plan_[decode_idx]) {
      const auto& decode_plan = prefetch_plans_[decode_idx];
      if (!decode_plan.chunk_groups.empty()) {
        resolved_mode = PrefetchMode::DECODE;
        if (next_mode) {
          *next_mode = resolved_mode;
        }
        return &decode_plan.chunk_groups.front();
      }
    }
  }

  const size_t target_index = next_index % group_count;
  if (target_index < plan.chunk_groups.size() &&
      plan.chunk_groups[target_index].group_index == target_index) {
    if (next_mode) {
      *next_mode = resolved_mode;
    }
    return &plan.chunk_groups[target_index];
  }

  for (const auto& group : plan.chunk_groups) {
    if (group.group_index == target_index) {
      if (next_mode) {
        *next_mode = resolved_mode;
      }
      return &group;
    }
  }

  if (next_mode) {
    *next_mode = resolved_mode;
  }
  return &plan.chunk_groups.front();
}


bool WeightChunkPrefetcher::ExecuteParallelPread(int fd, void* buffer, size_t size, off_t offset) {
  if (fd < 0 || buffer == nullptr || size == 0) {
    return false;
  }

  if (size < min_block_size_ * 2) {
    const ssize_t bytes_read = pread(fd, buffer, size, offset);
    return bytes_read > 0 && static_cast<size_t>(bytes_read) == size;
  }

  const size_t num_threads = std::min(max_threads_, size / min_block_size_);
  const size_t block_size = size / num_threads;
  const size_t remainder = size % num_threads;

  std::vector<std::future<bool>> futures;
  futures.reserve(num_threads);

  for (size_t i = 0; i < num_threads; ++i) {
    const size_t current_block_size = block_size + (i < remainder ? 1 : 0);
    const size_t block_offset = i * block_size + std::min(i, remainder);

    uint8_t* block_buffer = static_cast<uint8_t*>(buffer) + block_offset;
    const off_t block_file_offset = offset + block_offset;

    futures.emplace_back(std::async(std::launch::async, [=]() -> bool {
      const ssize_t bytes_read = pread(fd, block_buffer, current_block_size, block_file_offset);
      return bytes_read > 0 && static_cast<size_t>(bytes_read) == current_block_size;
    }));
  }

  bool all_success = true;
  for (auto& future : futures) {
    if (!future.get()) {
      all_success = false;
    }
  }

  return all_success;
}


}  // namespace streaming
}  // namespace flash_slim
