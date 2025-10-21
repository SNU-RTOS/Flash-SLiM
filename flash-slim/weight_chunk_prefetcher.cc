
#include <algorithm>
#include <cstdint>
#include <future>
#include <limits>
#include <utility>
#include <unistd.h>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#endif

#include "tflite/minimal_logging.h"
#include "weight_chunk_prefetcher.h"

namespace flash_slim {
namespace streaming {

namespace {

bool ExecuteIO(int fd, void* buffer, size_t size, off_t offset) {
  constexpr size_t kMinBlockSize = 512 * 1024;
  constexpr size_t kMaxThreads = 4;

  if (fd < 0 || buffer == nullptr || size == 0) {
    return false;
  }

  if (size < kMinBlockSize * 2) {
    const ssize_t bytes_read = pread(fd, buffer, size, offset);
    return bytes_read > 0 && static_cast<size_t>(bytes_read) == size;
  }

  const size_t num_threads = std::min(kMaxThreads, size / kMinBlockSize);
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

}  // namespace

WeightChunkPrefetcher::~WeightChunkPrefetcher() { StopWorker(); }

void WeightChunkPrefetcher::SetPrefetchPlan(
    PrefetchMode mode, std::unordered_map<size_t, size_t>&& offset_to_index,
    std::vector<weight_chunk_info_t>&& chunks) {
  const int idx = PrefetchModeToIndex(mode);
  if (idx < 0) {
    return;
  }
  prefetch_plans_[idx].offset_to_index = std::move(offset_to_index);
  prefetch_plans_[idx].chunks = std::move(chunks);
  has_plan_[idx] = true;
  ResetRuntimeState();
}

bool WeightChunkPrefetcher::HasPrefetchPlan(PrefetchMode mode) const {
  const int idx = PrefetchModeToIndex(mode);
  return idx >= 0 && has_plan_[idx];
}

const WeightChunkPrefetcher::PrefetchPlan* WeightChunkPrefetcher::GetPrefetchPlan(
    PrefetchMode mode) const {
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

  weight_chunk_info_t tmp_chunk;
  tmp_chunk.chunk_index = SIZE_MAX;
  tmp_chunk.aligned_offset = 0;
  tmp_chunk.offset_adjust = 0;
  tmp_chunk.aligned_size = 0;
  tmp_chunk.origin_offset = 0;
  tmp_chunk.origin_size = 0;
  tmp_chunk.managed_buffer_index = -1;
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

std::string WeightChunkPrefetcher::GetPrefetcherModeString() const {
  switch (prefetch_mode_) {
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

void WeightChunkPrefetcher::StartWorker() {
  bool expected = false;
  if (!io_worker_running_.compare_exchange_strong(expected, true)) {
    return;  // already running
  }

  io_worker_stop_requested_.store(false, std::memory_order_relaxed);

  io_worker_thread_ = std::thread([this]() {
    WorkerLoop();
  });

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

  if (io_worker_thread_.joinable()) {
    io_worker_thread_.join();
  }

  io_worker_stop_requested_.store(false, std::memory_order_relaxed);

  {
    std::lock_guard<std::mutex> states_lock(chunk_state_mutex_);
    for (auto& kv : index_to_chunk_states_) {
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
}

const weight_chunk_info_t* WeightChunkPrefetcher::LookupChunkInfo(PrefetchMode mode,
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

bool WeightChunkPrefetcher::Submit(const PrefetchRequest& request) {
  if (!request.buffer_base || request.direct_io_fd < 0) {
    return false;
  }

  PrefetchJob job;
  if (!ResolvePrefetchJob(request, &job)) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(io_worker_mutex_);
    auto state = GetChunkIOState(job.chunk_info.chunk_index);
    {
      std::lock_guard<std::mutex> state_lock(state->mutex);
      if (state->in_flight) {
        // Already scheduled; do not enqueue a duplicate.
        return true;
      }
      state->in_flight = true;
      state->ready = false;
      state->success = false;
    }
    io_job_queue_.push_back(std::move(job));
  }

  io_worker_cv_.notify_one();
  return true;
}

bool WeightChunkPrefetcher::WaitReady(PrefetchMode mode, size_t offset) {
  const weight_chunk_info_t* info = LookupChunkInfo(mode, offset);
  if (!info) {
    return false;
  }

  auto state = GetChunkIOState(info->chunk_index);
  std::unique_lock<std::mutex> lock(state->mutex);
  state->cv.wait(lock, [&]() { return state->ready || !state->in_flight; });
  const bool success = state->success;
  state->ready = false;
  state->in_flight = false;
  return success;
}

void WeightChunkPrefetcher::WorkerLoop() {
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

    const bool success = ExecuteIO(job.direct_io_fd, job.buffer_base,
                                   job.chunk_info.aligned_size, job.chunk_info.aligned_offset);
    MarkJobCompleted(job, success);
  }
}

void WeightChunkPrefetcher::ApplyWorkerAffinity() {
#if defined(__linux__)
  if (!io_worker_core_id_.has_value() || !io_worker_thread_.joinable()) {
    return;
  }

  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(static_cast<unsigned long>(*io_worker_core_id_), &set);
  const int rc = pthread_setaffinity_np(io_worker_thread_.native_handle(), sizeof(set), &set);
  if (rc != 0) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                    "WeightChunkPrefetcher: failed to set worker affinity (core=%d, errno=%d)",
                    *io_worker_core_id_, rc);
  }
#else
  (void)worker_core_id_;
#endif
}

std::shared_ptr<WeightChunkPrefetcher::ChunkIOState> WeightChunkPrefetcher::GetChunkIOState(
    size_t chunk_index) {
  std::lock_guard<std::mutex> lock(chunk_state_mutex_);
  auto it = index_to_chunk_states_.find(chunk_index);
  if (it == index_to_chunk_states_.end() || !(it->second)) {
    auto state = std::make_shared<ChunkIOState>();
    it = index_to_chunk_states_.emplace(chunk_index, std::move(state)).first;
  }
  return it->second;
}

void WeightChunkPrefetcher::ResetRuntimeState() {
  {
    std::lock_guard<std::mutex> lock(io_worker_mutex_);
    io_job_queue_.clear();
  }
  std::lock_guard<std::mutex> states_lock(chunk_state_mutex_);
  for (auto& kv : index_to_chunk_states_) {
    const auto& state = kv.second;
    if (!state) {
      continue;
    }
    {
      std::lock_guard<std::mutex> state_lock(state->mutex);
      state->in_flight = false;
      state->ready = false;
      state->success = false;
    }
    state->cv.notify_all();
  }
}

bool WeightChunkPrefetcher::ResolvePrefetchJob(const PrefetchRequest& request, PrefetchJob* job) {
  if (!job) {
    return false;
  }
  const weight_chunk_info_t* info = LookupChunkInfo(request.mode, request.offset);
  if (!info) {
    return false;
  }

  job->mode = request.mode;
  job->offset = request.offset;
  job->chunk_info = *info;
  job->buffer_base = request.buffer_base;
  job->direct_io_fd = request.direct_io_fd;
  return true;
}

void WeightChunkPrefetcher::MarkJobCompleted(const PrefetchJob& job, bool success) {
  auto state = GetChunkIOState(job.chunk_info.chunk_index);
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->success = success;
    state->ready = true;
    state->in_flight = false;
  }
  state->cv.notify_all();

  if (!success) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                    "WeightChunkPrefetcher: prefetch failed (offset=%zu, chunk_index=%zu)",
                    job.offset, job.chunk_info.chunk_index);
  }
}

}  // namespace streaming
}  // namespace flash_slim
