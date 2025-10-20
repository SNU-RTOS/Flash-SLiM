#ifndef FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_H_
#define FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "flash-slim/cmt_generator_util.h"
#include "flash-slim/weight_chunk_prefetcher.h"

#include "tflite/delegates/xnnpack/streaming_weight_cache.h"

namespace flash_slim {
namespace streaming {

class WeightChunkController : public tflite::xnnpack::WeightChunkControllerInterface {
 public:
  explicit WeightChunkController(tflite::xnnpack::StreamingWeightCacheProvider* provider);
  ~WeightChunkController();

  bool LoadPrefetchPlan(const std::string& path);
  void AttachPrefetcher(std::shared_ptr<flash_slim::streaming::WeightChunkPrefetcher> prefetcher);
  void AttachMetadataWriter(tflite::xnnpack::WeightChunkMetaDataWriter* writer);
  void SetMode(tflite::xnnpack::StreamingWeightCacheProvider::ProviderMode mode);
  void UpdatePrefetcherMode( WeightChunkPrefetcher::PrefetchMode mode);
  void EnsureWeightChunkBuffer(size_t size);

  void PreInvoke(size_t offset) override;
  void PostInvoke(size_t offset) override;

  void SwitchActiveBuffer();
  void ResetBuffers();

  void DumpStatus() const;

  void TraceWeightsAddr(void* addr, size_t offset) override;
  void RecordChunkAccess(size_t offset) override;

  size_t weight_chunk_buffer_requirement() const { return weight_chunk_buffer_requirement_; }

 private:
  using ProviderMode = tflite::xnnpack::StreamingWeightCacheProvider::ProviderMode;
  using PrefetchMode = flash_slim::streaming::WeightChunkPrefetcher::PrefetchMode;
  using ChunkInfo = tflite::xnnpack::StreamingWeightCacheProvider::weight_chunk_info_t;

  void ReleaseWeightChunkBuffers();
  bool HandleRuntimePreInvoke(size_t offset);
  void HandlePreRuntimePreInvoke(size_t offset);
  const ChunkInfo* ResolveChunkInfo(size_t offset) const;
  void UpdateWeightsPointer(size_t offset, const ChunkInfo& info);
  bool LoadChunkData(const ChunkInfo& info);

  tflite::xnnpack::StreamingWeightCacheProvider* provider_ = nullptr;
  std::shared_ptr<flash_slim::streaming::WeightChunkPrefetcher> prefetcher_;
  tflite::xnnpack::WeightChunkMetaDataWriter* writer_ = nullptr;
  ProviderMode provider_mode_ = ProviderMode::RUNTIME;
  PrefetchMode prefetch_mode_ = PrefetchMode::UNINITIALIZED;
  size_t weight_chunk_buffer_requirement_ = 0;
  size_t weight_chunk_buffer_capacity_ = 0;
  std::array<void*, 2> weight_chunk_buffers_{nullptr, nullptr};
  int active_weight_chunk_buffer_index_ = 0;
  size_t next_chunk_index_ = 0;
  std::unordered_map<size_t, ChunkInfo> chunk_info_cache_;
  std::unordered_map<size_t, std::array<void*, 2>> offset_to_weights_ptr_;
};

}  // namespace streaming
}  // namespace flash_slim

#endif  // FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_H_
