#ifndef FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_H_
#define FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
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
  void* GetWeightChunkBufferAddr(int index) const override;
  
  void PreInvokeImpl(size_t offset) override;
  void PostInvokeImpl(size_t offset) override;
  void TraceWeightsAddrImpl(void* addr, size_t offset) override;
  
  void RecordChunkAccess(size_t offset) override;

 private:
  using PreInvokeHandler = bool (WeightChunkController::*)(size_t);
  
  void UpdatePreinvokeHandler(ProviderMode mode);
  bool HandlePreRuntimePreInvoke(size_t offset);
  bool HandleRuntimePreInvoke(size_t offset);
  bool HandleDefaultPreInvoke(size_t offset);

  void ResetPrefetchState();
  bool EnsureChunkReady(const weight_chunk_info_t* info, int buffer_index, int fd);
  bool ScheduleNextChunk(const weight_chunk_info_t* current_info, int fd);
  int GetInactiveBufferIndex() const { return 1 - active_weight_chunk_buffer_index_; }
  
  void UpdateWeightsPointer(size_t offset, const weight_chunk_info_t& info);
  
  tflite::xnnpack::StreamingWeightCacheProvider* provider_ = nullptr;
  std::unique_ptr<WeightChunkPrefetcher> prefetcher_ = nullptr;
  WeightChunkMetaDataWriter* writer_ = nullptr;
  ProviderMode provider_mode_ = ProviderMode::RUNTIME;
  PrefetchMode prefetch_mode_ = PrefetchMode::UNINITIALIZED;
  PreInvokeHandler preinvoke_handler_ = &WeightChunkController::HandleDefaultPreInvoke;
  size_t weight_chunk_buffer_requirement_ = 0;
  size_t weight_chunk_buffer_capacity_ = 0;
  std::array<void*, 2> weight_chunk_buffers_{nullptr, nullptr};
  int active_weight_chunk_buffer_index_ = 0;
  size_t next_chunk_index_ = 0;
  std::unordered_map<size_t, weight_chunk_info_t> offset_to_chunk_info;
  std::unordered_map<size_t, std::array<void*, 2>> offset_to_weights_ptr_;
  std::array<const weight_chunk_info_t*, 2> buffer_chunks_{nullptr, nullptr};
  const weight_chunk_info_t* current_chunk_info_ = nullptr;
  const weight_chunk_info_t* next_chunk_info_ = nullptr;
  int next_chunk_buffer_index_ = -1;
  std::optional<size_t> next_chunk_expected_offset_;
};

}  // namespace streaming
}  // namespace flash_slim

#endif  // FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_H_
