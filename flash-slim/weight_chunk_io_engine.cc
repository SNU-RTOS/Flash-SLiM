#include "weight_chunk_io_engine.h"

#include <algorithm>
#include <limits>

#if defined(__linux__)
#include <cerrno>
#endif

#include "tflite/minimal_logging.h"

namespace flash_slim {
namespace streaming {

WeightChunkIOEngine::WeightChunkIOEngine() = default;

WeightChunkIOEngine::~WeightChunkIOEngine() { Shutdown(); }

bool WeightChunkIOEngine::Initialize(unsigned ring_depth, size_t subread_bytes) {
  ring_depth_ = ring_depth == 0 ? 16 : ring_depth;
  subread_bytes_ = subread_bytes == 0 ? 1024 * 1024 : subread_bytes;

#if defined(__linux__)
  if (ready_) {
    return true;
  }

  if (io_uring_queue_init(ring_depth_, &ring_, 0) != 0) {
    ready_ = false;
    return false;
  }

  ready_ = true;
  return true;
#else
  ready_ = false;
  return false;
#endif
}

void WeightChunkIOEngine::Shutdown() {
#if defined(__linux__)
  if (ready_) {
    CollectCompletions(false);
    io_uring_queue_exit(&ring_);
  }
#endif
  ready_ = false;
  inflight_.clear();
  pending_completions_.clear();
}

bool WeightChunkIOEngine::IsReady() const { return ready_; }

bool WeightChunkIOEngine::HasPending() const {
  return ready_ && !inflight_.empty();
}

bool WeightChunkIOEngine::Submit(const IORequest& request) {
#if defined(__linux__)
  if (!ready_) {
    return false;
  }

  if (!request.chunk_info || request.direct_io_fd < 0 || !request.buffer_base) {
    return false;
  }

  if (request.chunk_info->chunk_index > std::numeric_limits<uint32_t>::max()) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                    "WeightChunkIOEngine: chunk_index=%zu exceeds 32-bit limit; using fallback",
                    request.chunk_info->chunk_index);
    return false;
  }

  if (request.chunk_info->aligned_size == 0) {
    Completion completion;
    completion.chunk_index = request.chunk_info->chunk_index;
    completion.success = true;
    completion.bytes_transferred = 0;
    completion.epoch = 0;
    pending_completions_.push_back(completion);
    return true;
  }

  if (!SubmitInternal(request)) {
    return false;
  }

  CollectCompletions(false);
  return true;
#else
  (void)request;
  return false;
#endif
}

void WeightChunkIOEngine::DrainCompletions(std::vector<Completion>* completions,
                                           bool wait_for_events) {
  if (!completions) {
    return;
  }

#if defined(__linux__)
  if (ready_) {
    if (wait_for_events && pending_completions_.empty() && !inflight_.empty()) {
      CollectCompletions(true);
    }
    CollectCompletions(false);
  }
#else
  (void)wait_for_events;
#endif

  while (!pending_completions_.empty()) {
    completions->push_back(pending_completions_.front());
    pending_completions_.pop_front();
  }
}

void WeightChunkIOEngine::Reset() {
#if defined(__linux__)
  if (ready_) {
    CollectCompletions(false);
  }
#endif
  inflight_.clear();
  pending_completions_.clear();
}

#if defined(__linux__)

uint64_t WeightChunkIOEngine::PackTag(uint32_t chunk_index, uint16_t epoch,
                                      uint16_t sub_index) {
  return (static_cast<uint64_t>(chunk_index) << 32) |
         (static_cast<uint64_t>(epoch) << 16) |
         static_cast<uint64_t>(sub_index);
}

void WeightChunkIOEngine::UnpackTag(uint64_t tag, uint32_t* chunk_index, uint16_t* epoch,
                                    uint16_t* sub_index) {
  if (chunk_index) {
    *chunk_index = static_cast<uint32_t>(tag >> 32);
  }
  if (epoch) {
    *epoch = static_cast<uint16_t>((tag >> 16) & 0xFFFFu);
  }
  if (sub_index) {
    *sub_index = static_cast<uint16_t>(tag & 0xFFFFu);
  }
}

bool WeightChunkIOEngine::SubmitInternal(const IORequest& request) {
  const auto* chunk_info = request.chunk_info;
  const size_t chunk_index = chunk_info->chunk_index;

  auto& state = inflight_[chunk_index];
  state.epoch = static_cast<uint32_t>((state.epoch + 1) & 0xFFFFu);
  if ((state.epoch & 0xFFFFu) == 0) {
    state.epoch = 1;
  }
  state.pending = 0;
  state.expected_bytes = chunk_info->aligned_size;
  state.completed_bytes = 0;
  state.error = false;

  uint8_t* buffer = static_cast<uint8_t*>(request.buffer_base);
  off_t offset = static_cast<off_t>(chunk_info->aligned_offset);
  size_t remaining = chunk_info->aligned_size;
  uint16_t sub_index = 0;

  while (remaining > 0) {
    const size_t len = std::min(subread_bytes_, remaining);
    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);

    if (!sqe) {
      const int submit_rc = io_uring_submit(&ring_);
      if (submit_rc < 0) {
        state.error = true;
        break;
      }
      CollectCompletions(false);
      sqe = io_uring_get_sqe(&ring_);
      if (!sqe) {
        CollectCompletions(true);
        sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {
          state.error = true;
          break;
        }
      }
    }

    io_uring_prep_read(sqe, request.direct_io_fd, buffer, len, offset);
    sqe->user_data = PackTag(static_cast<uint32_t>(chunk_index),
                             static_cast<uint16_t>(state.epoch & 0xFFFFu),
                             sub_index++);

    buffer += len;
    offset += static_cast<off_t>(len);
    remaining -= len;
    state.pending++;
  }

  if (state.pending == 0 || state.error) {
    inflight_.erase(chunk_index);
    return false;
  }

  if (io_uring_submit(&ring_) < 0) {
    inflight_.erase(chunk_index);
    return false;
  }

  return true;
}

void WeightChunkIOEngine::CollectCompletions(bool wait_for_one) {
  while (true) {
    struct io_uring_cqe* cqe = nullptr;
    int rc = 0;
    if (wait_for_one) {
      rc = io_uring_wait_cqe(&ring_, &cqe);
      wait_for_one = false;
    } else {
      rc = io_uring_peek_cqe(&ring_, &cqe);
    }

    if (rc == -EAGAIN || cqe == nullptr) {
      break;
    }

    ProcessCqe(cqe);
    io_uring_cqe_seen(&ring_, cqe);
  }
}

void WeightChunkIOEngine::ProcessCqe(struct io_uring_cqe* cqe) {
  uint32_t chunk_index = 0;
  uint16_t epoch = 0;
  uint16_t sub_index = 0;
  UnpackTag(cqe->user_data, &chunk_index, &epoch, &sub_index);

  auto it = inflight_.find(chunk_index);
  if (it == inflight_.end()) {
    return;
  }

  auto& state = it->second;
  if (static_cast<uint16_t>(state.epoch & 0xFFFFu) != epoch) {
    return;
  }

  if (cqe->res < 0) {
    state.error = true;
  } else {
    state.completed_bytes += static_cast<size_t>(cqe->res);
  }

  if (state.pending > 0) {
    state.pending--;
  }

  if (state.pending == 0) {
    Completion completion;
    completion.chunk_index = static_cast<size_t>(chunk_index);
    completion.epoch = state.epoch;
    completion.bytes_transferred = state.completed_bytes;
    completion.success = !state.error && state.completed_bytes == state.expected_bytes;
    pending_completions_.push_back(completion);
    inflight_.erase(it);
  }
}

#endif  // defined(__linux__)

}  // namespace streaming
}  // namespace flash_slim
