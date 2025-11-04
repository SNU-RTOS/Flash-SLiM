#include "weight_chunk_io_engine.h"

#include <algorithm>
#include <limits>

#include <cerrno>

#include "tflite/minimal_logging.h"

namespace flash_slim {
namespace streaming {

WeightChunkIOEngine::WeightChunkIOEngine() = default;

WeightChunkIOEngine::~WeightChunkIOEngine() { Shutdown(); }

bool WeightChunkIOEngine::Initialize(unsigned ring_depth, size_t subread_bytes, void* buffer_0,
                                     size_t size_0, void* buffer_1, size_t size_1) {
  ring_depth_ = ring_depth == 0 ? 16 : ring_depth;
  subread_bytes_ = subread_bytes == 0 ? 1024 * 1024 : subread_bytes;

  if (ready_) {
    Shutdown();
  }

  buffers_registered_ = false;
  registered_buffers_[0] = {};
  registered_buffers_[1] = {};

  if (!buffer_0 || !buffer_1 || size_0 == 0 || size_1 == 0) {
    ready_ = false;
    return false;
  }

  if (io_uring_queue_init(ring_depth_, &ring_, 0) != 0) {
    ready_ = false;
    return false;
  }

  struct iovec iovs[2];
  iovs[0].iov_base = buffer_0;
  iovs[0].iov_len = size_0;
  iovs[1].iov_base = buffer_1;
  iovs[1].iov_len = size_1;

  if (io_uring_register_buffers(&ring_, iovs, 2) != 0) {
    io_uring_queue_exit(&ring_);
    ready_ = false;
    return false;
  }

  registered_buffers_[0].base = buffer_0;
  registered_buffers_[0].size = size_0;
  registered_buffers_[1].base = buffer_1;
  registered_buffers_[1].size = size_1;
  buffers_registered_ = true;

  ready_ = true;
  return true;

}

void WeightChunkIOEngine::Shutdown() {
  if (ready_) {
    CollectCompletions(false);
    if (buffers_registered_) {
      io_uring_unregister_buffers(&ring_);
      buffers_registered_ = false;
      registered_buffers_[0] = {};
      registered_buffers_[1] = {};
    }
    io_uring_queue_exit(&ring_);
  }
  ready_ = false;
  inflight_.clear();
  pending_completions_.clear();
}

bool WeightChunkIOEngine::IsReady() const { return ready_; }

bool WeightChunkIOEngine::HasPending() const {
  return ready_ && !inflight_.empty();
}

bool WeightChunkIOEngine::Submit(const IORequest& request) {
  if (!ready_) {
    return false;
  }

  if (request.direct_io_fd < 0 || !request.buffer_base) {
    return false;
  }

  if (!buffers_registered_ || request.buffer_index < 0 ||
      request.buffer_index >= static_cast<int>(registered_buffers_.size())) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                    "WeightChunkIOEngine: invalid buffer index %d", request.buffer_index);
    return false;
  }

  if (request.range_index > std::numeric_limits<uint32_t>::max()) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                    "WeightChunkIOEngine: range_index=%zu exceeds 32-bit limit; using fallback",
                    request.range_index);
    return false;
  }

  if (request.aligned_size == 0) {
    Completion completion;
    completion.range_index = request.range_index;
    completion.success = true;
    completion.bytes_transferred = 0;
    completion.epoch = 0;
    pending_completions_.push_back(completion);
    return true;
  }

  if (!ValidateRequestBuffer(request)) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                    "WeightChunkIOEngine: request buffer out of range for index %d",
                    request.buffer_index);
    return false;
  }

  if (!SubmitInternal(request)) {
    return false;
  }

  CollectCompletions(false);
  return true;

}

void WeightChunkIOEngine::DrainCompletions(std::vector<Completion>* completions,
                                           bool wait_for_events) {
  if (!completions) {
    return;
  }

  if (ready_) {
    if (wait_for_events && pending_completions_.empty() && !inflight_.empty()) {
      CollectCompletions(true);
    }
    CollectCompletions(false);
  }

  while (!pending_completions_.empty()) {
    completions->push_back(pending_completions_.front());
    pending_completions_.pop_front();
  }
}

void WeightChunkIOEngine::Reset() {
  if (ready_) {
    CollectCompletions(false);
  }

}


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
  const size_t range_index = request.range_index;

  auto& state = inflight_[range_index];
  state.epoch = static_cast<uint32_t>((state.epoch + 1) & 0xFFFFu);
  if ((state.epoch & 0xFFFFu) == 0) {
    state.epoch = 1;
  }
  state.pending = 0;
  state.expected_bytes = request.aligned_size;
  state.completed_bytes = 0;
  state.error = false;

  uint8_t* buffer = static_cast<uint8_t*>(request.buffer_base);
  off_t offset = static_cast<off_t>(request.aligned_offset);
  size_t remaining = request.aligned_size;
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
    io_uring_prep_read_fixed(sqe, request.direct_io_fd, buffer, len, offset,
                             static_cast<unsigned>(request.buffer_index));
    sqe->user_data = PackTag(static_cast<uint32_t>(range_index),
                             static_cast<uint16_t>(state.epoch & 0xFFFFu),
                             sub_index++);

    buffer += len;
    offset += static_cast<off_t>(len);
    remaining -= len;
    state.pending++;
  }

  if (state.pending == 0 || state.error) {
    inflight_.erase(range_index);
    return false;
  }

  if (io_uring_submit(&ring_) < 0) {
    inflight_.erase(range_index);
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
    completion.range_index = static_cast<size_t>(chunk_index);
    completion.epoch = state.epoch;
    completion.bytes_transferred = state.completed_bytes;
    completion.success = !state.error && state.completed_bytes == state.expected_bytes;
    pending_completions_.push_back(completion);
    inflight_.erase(it);
  }
}

bool WeightChunkIOEngine::ValidateRequestBuffer(const IORequest& request) const {
  const size_t index = static_cast<size_t>(request.buffer_index);
  if (index >= registered_buffers_.size()) {
    return false;
  }

  const auto& registered = registered_buffers_[index];
  if (!registered.base || registered.size == 0) {
    return false;
  }

  const uint8_t* base = static_cast<const uint8_t*>(registered.base);
  const uint8_t* ptr = static_cast<const uint8_t*>(request.buffer_base);
  const uint8_t* end = base + registered.size;

  if (ptr < base || ptr >= end) {
    return false;
  }

  const uint8_t* ptr_end = ptr + request.aligned_size;
  if (ptr_end > end) {
    return false;
  }

  return true;
}


}  // namespace streaming
}  // namespace flash_slim
