// Copyright 2025 Flash-SLiM Research Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef FLASH_SLIM_WEIGHT_CHUNK_IO_ENGINE_H_
#define FLASH_SLIM_WEIGHT_CHUNK_IO_ENGINE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <liburing.h>
#endif

#include "tflite/delegates/xnnpack/streaming_weight_cache.h"

namespace flash_slim {
namespace streaming {

class WeightChunkIOEngine {
 public:
  struct IORequest {
    size_t range_index = 0;
    size_t aligned_offset = 0;
    size_t aligned_size = 0;
    void* buffer_base = nullptr;
    int direct_io_fd = -1;
    int buffer_index = -1;
  };

  struct Completion {
    size_t range_index = 0;
    bool success = false;
    size_t bytes_transferred = 0;
    uint32_t epoch = 0;
  };

  WeightChunkIOEngine();
  ~WeightChunkIOEngine();

  bool Initialize(unsigned ring_depth, size_t subread_bytes, void* buffer_0, size_t size_0,
                  void* buffer_1, size_t size_1);
  void Shutdown();

  bool IsReady() const;
  bool HasPending() const;

  bool Submit(const IORequest& request);
  void DrainCompletions(std::vector<Completion>* completions, bool wait_for_events);
  void Reset();

 private:
  struct InflightState {
    uint32_t epoch = 0;
    uint32_t pending = 0;
    size_t expected_bytes = 0;
    size_t completed_bytes = 0;
    bool error = false;
  };

#if defined(__linux__)
  static uint64_t PackTag(uint32_t chunk_index, uint16_t epoch, uint16_t sub_index);
  static void UnpackTag(uint64_t tag, uint32_t* chunk_index, uint16_t* epoch, uint16_t* sub_index);

  bool SubmitInternal(const IORequest& request);
  void CollectCompletions(bool wait_for_one);
  void ProcessCqe(struct io_uring_cqe* cqe);
#endif

  struct RegisteredBuffer {
    void* base = nullptr;
    size_t size = 0;
  };

  bool ValidateRequestBuffer(const IORequest& request) const;

  bool ready_ = false;
  unsigned ring_depth_ = 32;
  size_t subread_bytes_ = 256 * 1024;

#if defined(__linux__)
  struct io_uring ring_ {};
#endif

  std::array<RegisteredBuffer, 2> registered_buffers_{};
  bool buffers_registered_ = false;
  std::unordered_map<size_t, InflightState> inflight_;
  std::deque<Completion> pending_completions_;
};

}  // namespace streaming
}  // namespace flash_slim

#endif  // FLASH_SLIM_WEIGHT_CHUNK_IO_ENGINE_H_
