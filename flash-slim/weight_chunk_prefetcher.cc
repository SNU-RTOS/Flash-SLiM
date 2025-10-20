#include "flash-slim/weight_chunk_prefetcher.h"

#include <limits>
#include <utility>

#include "tflite/minimal_logging.h"

namespace flash_slim {
namespace streaming {

void WeightChunkPrefetcher::SetPrefetchPlan(
    PrefetchMode mode, std::unordered_map<size_t, size_t>&& offset_to_index,
    std::vector<ChunkInfo>&& chunks) {
  const int idx = PrefetchModeToIndex(mode);
  if (idx < 0) {
    return;
  }
  prefetch_plans_[idx].offset_to_index = std::move(offset_to_index);
  prefetch_plans_[idx].chunks = std::move(chunks);
  has_plan_[idx] = true;
}

bool WeightChunkPrefetcher::HasPlan(PrefetchMode mode) const {
  const int idx = PrefetchModeToIndex(mode);
  return idx >= 0 && has_plan_[idx];
}

const WeightChunkPrefetcher::PrefetchPlan* WeightChunkPrefetcher::GetPlan(
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

  ChunkInfo sentinel;
  sentinel.chunk_index = SIZE_MAX;
  sentinel.aligned_offset = 0;
  sentinel.offset_adjust = 0;
  sentinel.aligned_size = 0;
  sentinel.origin_offset = 0;
  sentinel.origin_size = 0;
  sentinel.managed_buffer_index = -1;
  sentinel.weights_id = 0;

  index_to_chunks_.assign(max_index + 1, sentinel);

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

}  // namespace streaming
}  // namespace flash_slim
