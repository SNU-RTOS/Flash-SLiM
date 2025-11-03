
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

#if defined(__linux__)
#include <pthread.h>
#endif

#include "tflite/minimal_logging.h"
#include "weight_chunk_prefetcher.h"

namespace flash_slim {
namespace streaming {

namespace {

constexpr unsigned kDefaultRingDepth = 64;
constexpr size_t kDefaultSubreadBytes = 512 * 1024;
constexpr uint32_t kPlanIndexShift = 30;
constexpr size_t kPlanIndexMask = (static_cast<size_t>(1) << kPlanIndexShift) - static_cast<size_t>(1);

inline size_t EncodeRangeId(int plan_index, size_t range_index) {
  return (static_cast<size_t>(plan_index) << kPlanIndexShift) | range_index;
}

inline int DecodePlanIndex(size_t encoded) {
  return static_cast<int>(encoded >> kPlanIndexShift);
}

inline size_t DecodeRangeIndex(size_t encoded) {
  return encoded & kPlanIndexMask;
}

// constexpr unsigned kDefaultRingDepth = 128;
// constexpr size_t kDefaultSubreadBytes = 4 * 1024 * 1024;

// constexpr unsigned kDefaultRingDepth = 128;
// constexpr size_t kDefaultSubreadBytes = 2 * 1024 * 1024;


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

void WeightChunkPrefetcher::ConfigureIOBuffers(void* buffer0, size_t size0, void* buffer1,
                                               size_t size1) {
  std::lock_guard<std::mutex> lock(io_config_mutex_);
  registered_buffers_[0] = buffer0;
  registered_buffer_sizes_[0] = size0;
  registered_buffers_[1] = buffer1;
  registered_buffer_sizes_[1] = size1;
  buffers_configured_ = buffer0 != nullptr && buffer1 != nullptr && size0 > 0 && size1 > 0;
}

void WeightChunkPrefetcher::SetWorkerThreadAffinity(const std::vector<int>& cores) {
  {
    std::lock_guard<std::mutex> lock(io_worker_mutex_);
    io_worker_cores_ = cores;
  }
}

void WeightChunkPrefetcher::SetPrefetchPlan(
    PrefetchMode mode, std::unordered_map<size_t, size_t>&& offset_to_index,
    std::vector<WeightChunkInfo>&& chunks,
    std::vector<PrefetchChunkRange>&& chunk_ranges) {

  const int idx = PrefetchModeToIndex(mode);

  if (idx < 0) {
    return;
  }

  auto& plan = prefetch_plans_[idx];
  plan.offset_to_index = std::move(offset_to_index);
  plan.chunks = std::move(chunks);
  plan.chunk_ranges = std::move(chunk_ranges);

  std::sort(plan.chunk_ranges.begin(), plan.chunk_ranges.end(),
            [](const PrefetchChunkRange& lhs, const PrefetchChunkRange& rhs) {
              return lhs.io_order < rhs.io_order;
            });

  for (size_t range_index = 0; range_index < plan.chunk_ranges.size(); ++range_index) {
    auto& range = plan.chunk_ranges[range_index];
    if (range_index > kPlanIndexMask) {
      TFLITE_LOG_PROD(
          tflite::TFLITE_LOG_WARNING,
          "WeightChunkPrefetcher: range index %zu exceeds encoding capacity (%zu). Results may be "
          "undefined.",
          range_index, kPlanIndexMask);
    }
    if (range.io_order != range_index) {
      TFLITE_LOG_PROD(
          tflite::TFLITE_LOG_WARNING,
          "WeightChunkPrefetcher: adjusting io_order from %zu to %zu to maintain contiguity.",
          range.io_order, range_index);
      range.io_order = range_index;
    }
    range.range_index = range_index;
  }

  has_plan_[idx] = true;
  range_states_[idx].clear();
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

std::string WeightChunkPrefetcher::GetPrefetchModeString() const {
  return std::string(PrefetchModeName(prefetch_mode_));
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

const PrefetchChunkRange* WeightChunkPrefetcher::GetChunkRangeByChunkIndex(
    PrefetchMode mode, size_t chunk_index) const {
  const int plan_idx = PrefetchModeToIndex(mode);
  if (plan_idx < 0 || !has_plan_[plan_idx]) {
    return nullptr;
  }
  const auto& plan = prefetch_plans_[plan_idx];
  for (size_t i = 0; i < plan.chunk_ranges.size(); ++i) {
    const auto& range = plan.chunk_ranges[i];
    if (std::find(range.chunk_indices.begin(), range.chunk_indices.end(), chunk_index) !=
        range.chunk_indices.end()) {
      return &plan.chunk_ranges[i];
    }
  }

  TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                  "WeightChunkPrefetcher: chunk_index=%zu not found in range list for mode %d",
                  chunk_index, plan_idx);
  return nullptr;
}

const PrefetchChunkRange* WeightChunkPrefetcher::GetChunkRangeByIoOrder(
    PrefetchMode mode, size_t io_order) const {
  const int plan_idx = PrefetchModeToIndex(mode);
  if (plan_idx < 0 || !has_plan_[plan_idx]) {
    return nullptr;
  }
  const auto& plan = prefetch_plans_[plan_idx];
  if (io_order < plan.chunk_ranges.size() && plan.chunk_ranges[io_order].io_order == io_order) {
    return &plan.chunk_ranges[io_order];
  }
  for (size_t i = 0; i < plan.chunk_ranges.size(); ++i) {
    if (plan.chunk_ranges[i].io_order == io_order) {
      return &plan.chunk_ranges[i];
    }
  }
  return nullptr;
}

const PrefetchChunkRange* WeightChunkPrefetcher::GetNextChunkRange(
    PrefetchMode mode, size_t current_io_order) const {
  const int plan_idx = PrefetchModeToIndex(mode);
  if (plan_idx < 0 || !has_plan_[plan_idx]) {
    return nullptr;
  }

  const auto& plan = prefetch_plans_[plan_idx];
  if (plan.chunk_ranges.empty()) {
    return nullptr;
  }
  const size_t target_io_order = (current_io_order + 1) % plan.chunk_ranges.size();
  if (target_io_order < plan.chunk_ranges.size() &&
      plan.chunk_ranges[target_io_order].io_order == target_io_order) {
    return &plan.chunk_ranges[target_io_order];
  }
  for (size_t i = 0; i < plan.chunk_ranges.size(); ++i) {
    if (plan.chunk_ranges[i].io_order == target_io_order) {
      return &plan.chunk_ranges[i];
    }
  }
  return &plan.chunk_ranges.front();
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
    buffer0 = registered_buffers_[0];
    buffer1 = registered_buffers_[1];
    size0 = registered_buffer_sizes_[0];
    size1 = registered_buffer_sizes_[1];
    configured = buffers_configured_;
  }

  if (!configured) {
    io_worker_running_.store(false, std::memory_order_relaxed);
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                    "WeightChunkPrefetcher: IO buffers not configured; worker start deferred");
    return;
  }

#if defined(__linux__)
  if (!io_engine_) {
    io_engine_ = std::make_unique<WeightChunkIOEngine>();
  }
  if (io_engine_ && !io_engine_->IsReady()) {
    printf("[INFO] Initializing io_uring engine for WeightChunkPrefetcher...\n");
    if (!io_engine_->Initialize(kDefaultRingDepth, kDefaultSubreadBytes, buffer0, size0, buffer1,
                                size1)) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                      "WeightChunkPrefetcher: failed to initialize io_uring engine; using sync IO");
    }
  }
#endif

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

#if defined(__linux__)
  if (io_engine_) {
    io_engine_->Shutdown();
  }
#endif

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

bool WeightChunkPrefetcher::Submit(const PrefetchRequest& request) {
  const PrefetchChunkRange* range = request.chunk_range;
  if (!request.buffer_base || request.direct_io_fd < 0 || request.buffer_index < 0 || !range ||
      range->chunk_indices.empty()) {
    return false;
  }

  const PrefetchMode mode = prefetch_mode_;
  const int plan_idx = PrefetchModeToIndex(mode);
  if (plan_idx < 0 || !has_plan_[plan_idx]) {
    return false;
  }

//   std::cout << "[WeightChunkPrefetcher] Submit: mode=" << plan_idx
//             << " io_order=" << range->io_order
//             << " buffer_index=" << request.buffer_index
//             << " size=" << range->total_aligned_size << std::endl;

  {
    std::lock_guard<std::mutex> lock(io_worker_mutex_);
    auto range_state = GetChunkRangeState(mode, range->range_index);
    {
      std::lock_guard<std::mutex> range_lock(range_state->mutex);
      if (range_state->in_flight) {
        return true;
      }
      range_state->in_flight = true;
    }

    std::vector<std::shared_ptr<ChunkIOState>> chunk_states;
    chunk_states.reserve(range->chunk_indices.size());
    for (const size_t chunk_index : range->chunk_indices) {
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
      std::lock_guard<std::mutex> range_lock(range_state->mutex);
      range_state->in_flight = false;
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
      job.chunk_range = range;
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
//   std::cout << "[WeightChunkPrefetcher] WaitReady: chunk_index=" << chunk_info->chunk_index
//             << " waiting for ready" << std::endl;
  state->cv.wait(lock, [&]() { 
;
    return state->ready || !state->in_flight; });
  const bool success = state->success;
  state->ready = false;
  state->in_flight = false;
//   std::cout << "[WeightChunkPrefetcher] WaitReady: chunk_index=" << chunk_info->chunk_index
//             << " completed, success=" << success << std::endl;
  return success;
}

void WeightChunkPrefetcher::WorkerLoop() {
// #if defined(__linux__)
//   if (io_engine_ && io_engine_->IsReady()) {
//     std::cout << "[INFO] WeightChunkPrefetcher: using io_uring IO worker loop" << std::endl;
//     RunAsyncWorkerLoop();
//     return;
//   }
// #endif
std::cout << "[INFO] WeightChunkPrefetcher: using parallel pread IO worker loop" << std::endl;
  RunSyncWorkerLoop();
}

void WeightChunkPrefetcher::RunAsyncWorkerLoop() {
  while (true) {
    std::vector<PrefetchJob> jobs_to_submit;
    bool stop_requested = false;
    bool had_pending_before_wait = false;

    {
      std::unique_lock<std::mutex> lock(io_worker_mutex_);
      io_worker_cv_.wait(lock, [this]() {
        if (io_worker_stop_requested_.load(std::memory_order_relaxed) || !io_job_queue_.empty()) {
          return true;
        }
        return io_engine_ && io_engine_->HasPending();
      });

      stop_requested = io_worker_stop_requested_.load(std::memory_order_relaxed);
      had_pending_before_wait = io_engine_ && io_engine_->HasPending();

      while (!io_job_queue_.empty()) {
        jobs_to_submit.push_back(std::move(io_job_queue_.front()));
        io_job_queue_.pop_front();
      }
    }

    // if (!jobs_to_submit.empty()) {
    //   printf("WeightChunkPrefetcher: submitting %zu IO jobs\n", jobs_to_submit.size());
    // }

    bool needs_blocking_drain = had_pending_before_wait;
    for (auto& job : jobs_to_submit) {
      const PrefetchChunkRange* range = job.chunk_range;
      if (!range) {
        MarkJobCompleted(job, false);
        continue;
      }

      if (job.buffer_index < 0) {
        MarkJobCompleted(job, false);
        continue;
      }

      const int plan_index = PrefetchModeToIndex(job.mode);
      if (plan_index < 0) {
        MarkJobCompleted(job, false);
        continue;
      }

      WeightChunkIOEngine::IORequest request;
      request.range_index = EncodeRangeId(plan_index, range->range_index);
      request.aligned_offset = range->start_aligned_offset;
      request.aligned_size = range->total_aligned_size;
      request.buffer_base = job.buffer_base;
      request.direct_io_fd = job.direct_io_fd;
      request.buffer_index = job.buffer_index;

      const bool submitted = io_engine_->Submit(request);
      if (!submitted) {
        {
          std::lock_guard<std::mutex> requeue_lock(io_worker_mutex_);
          io_job_queue_.push_front(std::move(job));
        }
        needs_blocking_drain = true;
        io_worker_cv_.notify_one();
        continue;
      }
    //   std::cout << "[WeightChunkPrefetcher] Worker: submitted range io_order="
    //             << range->io_order << " aligned_offset=" << request.aligned_offset
    //             << " size=" << request.aligned_size << std::endl;
      needs_blocking_drain = true;
    }

    std::vector<WeightChunkIOEngine::Completion> completions;
    bool queue_empty_after_submit = false;
    {
      std::lock_guard<std::mutex> lock(io_worker_mutex_);
      queue_empty_after_submit = io_job_queue_.empty();
    }
    const bool wait_for_events = stop_requested || needs_blocking_drain ||
                                 (queue_empty_after_submit && io_engine_ && io_engine_->HasPending());
    io_engine_->DrainCompletions(&completions, wait_for_events);

    for (const auto& completion : completions) {
      PrefetchJob job_snapshot;
      const int plan_index = DecodePlanIndex(completion.range_index);
      if (plan_index < 0 || plan_index >= 2 || !has_plan_[plan_index]) {
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                        "WeightChunkPrefetcher: invalid completion plan index=%d", plan_index);
        continue;
      }
      const size_t range_index = DecodeRangeIndex(completion.range_index);
      const auto& plan = prefetch_plans_[plan_index];
      if (range_index >= plan.chunk_ranges.size()) {
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                        "WeightChunkPrefetcher: missing chunk range metadata for range_index=%zu",
                        range_index);
        continue;
      }
      const auto mode = IndexToPrefetchMode(plan_index);
      if (!mode.has_value()) {
        continue;
      }
      job_snapshot.chunk_range = &plan.chunk_ranges[range_index];
      job_snapshot.mode = *mode;
      MarkJobCompleted(job_snapshot, completion.success);
    }

    // if (!completions.empty()) {
    //   printf("WeightChunkPrefetcher: completed %zu IO jobs\n", completions.size());
    // }

    bool queue_empty = false;
    {
      std::lock_guard<std::mutex> lock(io_worker_mutex_);
      queue_empty = io_job_queue_.empty();
    }

    if (stop_requested && queue_empty && !(io_engine_ && io_engine_->HasPending())) {
      break;
    }
  }
}

void WeightChunkPrefetcher::RunSyncWorkerLoop() {
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

    const PrefetchChunkRange* range = job.chunk_range;
    const bool success = range &&
                         ExecuteIO(job.direct_io_fd, job.buffer_base, range->total_aligned_size,
                                   static_cast<off_t>(range->start_aligned_offset));
    // std::cout << "[WeightChunkPrefetcher] SyncWorker: completed range io_order="
    //           << (range ? range->io_order : static_cast<size_t>(-1))
    //           << " success=" << success << std::endl;
    MarkJobCompleted(job, success);
  }
}

void WeightChunkPrefetcher::ApplyWorkerAffinity() {
#if defined(__linux__)
  std::lock_guard<std::mutex> lock(io_worker_mutex_);
  if (io_worker_cores_.empty() || !io_worker_thread_.joinable()) {
    return;
  }

  cpu_set_t set;
  CPU_ZERO(&set);
  for (int core : io_worker_cores_) {
    if (core >= 0) {
      CPU_SET(static_cast<unsigned long>(core), &set);
    }
  }
  const int rc = pthread_setaffinity_np(io_worker_thread_.native_handle(), sizeof(set), &set);
  if (rc != 0) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                    "WeightChunkPrefetcher: failed to set worker affinity (errno=%d)", rc);
  }
#else
  (void)io_worker_cores_;
#endif
}

std::shared_ptr<WeightChunkPrefetcher::ChunkIOState> WeightChunkPrefetcher::GetChunkIOState(size_t chunk_index) {
  std::lock_guard<std::mutex> lock(chunk_state_mutex_);
  auto it = index_to_chunk_states_.find(chunk_index);
  if (it == index_to_chunk_states_.end() || !(it->second)) {
    auto state = std::make_shared<ChunkIOState>();
    it = index_to_chunk_states_.emplace(chunk_index, std::move(state)).first;
  }
  return it->second;
}

std::shared_ptr<WeightChunkPrefetcher::ChunkRangeIOState> WeightChunkPrefetcher::GetChunkRangeState(
    PrefetchMode mode, size_t range_index) {
  std::lock_guard<std::mutex> lock(range_state_mutex_);
  const int plan_idx = PrefetchModeToIndex(mode);
  if (plan_idx < 0 || plan_idx >= static_cast<int>(range_states_.size())) {
    static auto dummy = std::make_shared<ChunkRangeIOState>();
    return dummy;
  }
  auto& map = range_states_[plan_idx];
  auto it = map.find(range_index);
  if (it == map.end() || !(it->second)) {
    auto state = std::make_shared<ChunkRangeIOState>();
    it = map.emplace(range_index, std::move(state)).first;
  }
  return it->second;
}

void WeightChunkPrefetcher::ResetRuntimeState() {
  {
    std::lock_guard<std::mutex> lock(io_worker_mutex_);
    io_job_queue_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(range_state_mutex_);
    for (auto& states : range_states_) {
      states.clear();
    }
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

void WeightChunkPrefetcher::MarkJobCompleted(const PrefetchJob& job, bool success) {
  const PrefetchChunkRange* range = job.chunk_range;
  if (!range) {
    return;
  }

//   std::cout << "[WeightChunkPrefetcher] MarkJobCompleted: io_order=" << range->io_order
//             << " success=" << success << std::endl;

  for (const size_t chunk_index : range->chunk_indices) {
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

  auto range_state = GetChunkRangeState(job.mode, range->range_index);
  if (range_state) {
    std::lock_guard<std::mutex> range_lock(range_state->mutex);
    range_state->in_flight = false;
  }

  if (!success && !range->chunk_indices.empty()) {
    const auto* info = GetChunkInfoByIndex(range->chunk_indices.front());
    const size_t origin_offset = info ? info->origin_offset : 0;
    const size_t chunk_index = info ? info->chunk_index : range->chunk_indices.front();
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                    "WeightChunkPrefetcher: prefetch failed (range_io_order=%zu, chunk_index=%zu, "
                    "origin_offset=%zu)",
                    range->io_order, chunk_index, origin_offset);
  }
}

}  // namespace streaming
}  // namespace flash_slim
