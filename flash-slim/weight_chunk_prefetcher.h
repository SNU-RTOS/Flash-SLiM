// Copyright 2025 Flash-SLiM Research Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef FLASH_SLIM_WEIGHT_CHUNK_PREFETCHER_H_
#define FLASH_SLIM_WEIGHT_CHUNK_PREFETCHER_H_

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "tflite/delegates/xnnpack/streaming_weight_cache.h"

namespace flash_slim {
namespace streaming {

class WeightChunkPrefetcher {
 public:
  using PrefetchMode = tflite::xnnpack::WeightChunkPrefetchMode;
  using ChunkInfo = tflite::xnnpack::StreamingWeightCacheProvider::weight_chunk_info_t;

  struct PrefetchPlan {
    std::unordered_map<size_t, size_t> offset_to_index;  // origin_offset -> index
    std::vector<ChunkInfo> chunks;                      // index -> chunk metadata
  };

  WeightChunkPrefetcher() = default;
  ~WeightChunkPrefetcher() = default;

  static int PrefetchModeToIndex(PrefetchMode mode) {
    return tflite::xnnpack::WeightChunkPrefetchModeToIndex(mode);
  }

  void SetPrefetchPlan(PrefetchMode mode,
                       std::unordered_map<size_t, size_t>&& offset_to_index,
                       std::vector<ChunkInfo>&& chunks);

  bool HasPlan(PrefetchMode mode) const;

  const PrefetchPlan* GetPlan(PrefetchMode mode) const;

  void BuildIndexToChunksFromPlans();

  const std::vector<ChunkInfo>& GetIndexToChunks() const { return index_to_chunks_; }

  void UpdatePrefetcherMode(PrefetchMode mode) { prefetch_mode_ = mode; }
  PrefetchMode GetPrefetcherMode() const { return prefetch_mode_; }
  std::string GetPrefetcherModeString() const;

 private:
  PrefetchMode prefetch_mode_ = PrefetchMode::UNINITIALIZED;
  std::array<PrefetchPlan, 2> prefetch_plans_{};  // [PREFILL=0, DECODE=1]
  std::array<bool, 2> has_plan_{{false, false}};
  std::vector<ChunkInfo> index_to_chunks_;
};

}  // namespace streaming
}  // namespace flash_slim

#endif  // FLASH_SLIM_WEIGHT_CHUNK_PREFETCHER_H_


