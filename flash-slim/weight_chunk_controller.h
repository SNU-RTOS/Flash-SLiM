#ifndef FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_H_
#define FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <limits>
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
  void AttachPrefetcher(std::unique_ptr<WeightChunkPrefetcher> prefetcher, const std::vector<int>& io_cores);
  void AttachMetadataWriter(WeightChunkMetaDataWriter* writer);
  
  void UpdateProviderMode(ProviderMode mode);
  void UpdatePrefetcherMode(PrefetchMode mode);

  bool LoadPrefetchPlan(const std::string& path);
  
  void AllocWeightChunkBuffer(size_t size);
  void ReleaseWeightChunkBuffer();
  void SwitchActiveBufferIndex();
  void UpdateActiveBufferIndex(int index);

  bool ResetBPFProbe();
  
  void* GetActiveWeightChunkBuffer() const override;
  void* GetWeightChunkBufferAddr(int index) const override;
  
  void PreInvokeImpl(size_t offset) override;
  void PostInvokeImpl(size_t offset) override;
  void TraceWeightsAddrImpl(void* addr, size_t offset) override;
  int FetchArgIntImpl() override;
  
  void RecordChunkAccess(size_t offset) override;


 private:
  using PreInvokeHandler = bool (WeightChunkController::*)(size_t);
  using PostInvokeHandler = bool (WeightChunkController::*)();
  
  void UpdatePreInvokeHandler(ProviderMode mode);
  bool HandlePreRunWarmUpPreInvoke(size_t offset);
  bool HandlePreRunProfilePreInvoke(size_t offset);
  bool HandleRuntimePreInvoke(size_t offset);
  bool HandleDefaultPreInvoke(size_t offset);

  void UpdatePostInvokeHandler(ProviderMode mode);
  bool HandlePreRunWarmUpPostInvoke();
  bool HandlePreRunProfilePostInvoke();
  bool HandleRuntimePostInvoke();
  bool HandleDefaultPostInvoke();

  // Helper to emit BPF probe for chunk completion
  void EmitBPFProbe(size_t offset);
  
//   bool EnsureChunkReady(const weight_chunk_info_t* info, int buffer_index, int fd);
  bool ScheduleNextRange(const PrefetchChunkRange* current_range, int fd);
  int GetInactiveBufferIndex() const { return 1 - active_weight_chunk_buffer_index_; }
  size_t ComputeInactiveSlotOffset(size_t next_aligned_size) const;
  void ResetBufferSlots();
  
  void UpdateWeightsPointer(size_t offset, const weight_chunk_info_t& info,
                            const PrefetchChunkRange& range);
  size_t FindChunkRelativeOffset(const PrefetchChunkRange& range, size_t chunk_index) const;
  
  tflite::xnnpack::StreamingWeightCacheProvider* provider_ = nullptr;
  std::unique_ptr<WeightChunkPrefetcher> prefetcher_ = nullptr;
  WeightChunkMetaDataWriter* writer_ = nullptr;
  PreInvokeHandler preinvoke_handler_ = &WeightChunkController::HandleDefaultPreInvoke;
  PostInvokeHandler postinvoke_handler_ = &WeightChunkController::HandleDefaultPostInvoke;

  size_t chunk_index_ = 0;
  size_t bpf_probe_prev_offset_ = 0;
  size_t weight_chunk_buffer_requirement_ = 0;
  size_t weight_chunk_buffer_capacity_ = 0;
  bool bpf_probe_first_call_ = true;
  int active_weight_chunk_buffer_index_ = 0;
  void* weight_chunk_buffer_base_ = nullptr;
  std::array<bool, 2> first_prefetch_per_mode_{{true, true}};
  std::array<void*, 2> weight_chunk_buffers_{nullptr, nullptr};
  std::array<size_t, 2> buffer_offsets_{0, 0};
  std::array<size_t, 2> buffer_sizes_{0, 0};
  std::array<size_t, 2> buffer_range_ids_{{std::numeric_limits<size_t>::max(),
                                           std::numeric_limits<size_t>::max()}};
  std::unordered_map<size_t, weight_chunk_info_t> offset_to_chunk_info_;
  std::unordered_map<size_t, std::array<void*, 2>> offset_to_weights_ptr_;
};

}  // namespace streaming
}  // namespace flash_slim

#endif  // FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_H_
