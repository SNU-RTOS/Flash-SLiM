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

#include "tflite/delegates/xnnpack/streaming_weight_cache.h"

#include "weight_chunk_prefetcher.h"
#include "weight_chunk_controller_utils.h"

namespace flash_slim {
namespace streaming {

class WeightChunkController : public tflite::xnnpack::WeightChunkControllerInterface {

 public:
  explicit WeightChunkController(tflite::xnnpack::StreamingWeightCacheProvider* provider);
  ~WeightChunkController();

  void AttachPrefetcher(std::unique_ptr<WeightChunkPrefetcher> prefetcher);
  void AttachMetadataWriter(WeightChunkMetaDataWriter* writer);
  void UpdateProviderMode(ProviderMode mode);
  void UpdatePrefetcherMode(PrefetchMode mode);

  bool LoadPrefetchPlan(const std::string& path);
  
  void AllocWeightChunkBuffer(size_t size);
  void ReleaseWeightChunkBuffer();
  void SwitchActiveBufferIndex();
  void UpdateActiveBufferIndex(int index);
  void* GetActiveWeightChunkBuffer() const override;
  void* GetWeightChunkBuffer(int index) const override;
  
  void DumpStatus() const;

  void PreInvoke(size_t offset) override;
  void PostInvoke(size_t offset) override;
  void TraceWeightsAddr(void* addr, size_t offset) override;
  
  void RecordChunkAccess(size_t offset) override;

 private:
  
  void HandlePreRuntimePreInvoke(size_t offset);
  bool HandleRuntimePreInvoke(size_t offset);
  const weight_chunk_info_t* ResolveChunkInfo(size_t offset) const;
  void UpdateWeightsPointer(size_t offset, const weight_chunk_info_t& info);
  bool LoadChunkData(const weight_chunk_info_t& info);

  tflite::xnnpack::StreamingWeightCacheProvider* provider_ = nullptr;
  std::unique_ptr<WeightChunkPrefetcher> prefetcher_ = nullptr;
  WeightChunkMetaDataWriter* writer_ = nullptr;
  ProviderMode provider_mode_ = ProviderMode::RUNTIME;
  PrefetchMode prefetch_mode_ = PrefetchMode::UNINITIALIZED;
  size_t weight_chunk_buffer_requirement_ = 0;
  size_t weight_chunk_buffer_capacity_ = 0;
  std::array<void*, 2> weight_chunk_buffers_{nullptr, nullptr};
  int active_weight_chunk_buffer_index_ = 0;
  size_t next_chunk_index_ = 0;
  std::unordered_map<size_t, weight_chunk_info_t> chunk_info_cache_;
  std::unordered_map<size_t, std::array<void*, 2>> offset_to_weights_ptr_;
};

}  // namespace streaming
}  // namespace flash_slim

#endif  // FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_H_
