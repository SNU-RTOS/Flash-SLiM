#ifndef FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_H_
#define FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
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
  void AttachPrefetcher(std::unique_ptr<WeightChunkPrefetcher> prefetcher, const std::vector<int>& io_cores);
  void AttachMetadataWriter(WeightChunkMetaDataWriter* writer);
  
  void UpdateProviderMode(ProviderMode mode);
  void UpdatePrefetcherMode(PrefetchMode mode);

  bool LoadPrefetchPlan(const std::string& path);
  
  void AllocWeightChunkBuffer(size_t size);
  void ReleaseWeightChunkBuffer();
  void SwitchActiveBufferIndex();
  void UpdateActiveBufferIndex(int index);
  int GetActiveBufferIndex() const { return active_weight_chunk_buffer_index_; }
  int GetInactiveBufferIndex() const { return 1 - active_weight_chunk_buffer_index_; }

  bool ResetBPFProbe();

  void RecordChunkAccess(size_t offset);

  inline void* GetActiveWeightChunkBuffer() const override;
  inline void* GetWeightChunkBufferAddr(int index) const override;
  inline void* OffsetToAddrImpl(size_t offset) override;
  inline void PreInvokeImpl(size_t offset) override;
  inline void PostInvokeImpl(size_t offset) override;
  inline void TraceWeightsAddrImpl(void* addr, size_t offset) override;
  inline int FetchArgIntImpl() override;
  
  
 private:
  using PreInvokeHandler = bool (WeightChunkController::*)(size_t);
  using PostInvokeHandler = bool (WeightChunkController::*)();

  static constexpr const char* kPrefillMode = "PREFILL";
  static constexpr const char* kDecodeMode = "DECODE";

  struct BufferSlot {
    void* base = nullptr;
    size_t offset = 0;
    size_t size = 0;
    size_t io_order = std::numeric_limits<size_t>::max();
  };

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
  
  bool ScheduleNextRange(const PrefetchChunkRange* current_range, int fd);
  size_t ComputeInactiveSlotOffset(size_t next_aligned_size) const;
  void ResetBufferSlots();
  void UpdateWeightsPointer(size_t offset, const WeightChunkInfo& info,
                            const PrefetchChunkRange& range);
  size_t FindChunkRelativeOffset(const PrefetchChunkRange& range, size_t chunk_index) const;
  
  static inline size_t AlignTo(size_t value, size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
 }

  tflite::xnnpack::StreamingWeightCacheProvider* provider_ = nullptr;
  std::unique_ptr<WeightChunkPrefetcher> prefetcher_ = nullptr;
  WeightChunkMetaDataWriter* writer_ = nullptr;

  PreInvokeHandler preinvoke_handler_ = &WeightChunkController::HandleDefaultPreInvoke;
  PostInvokeHandler postinvoke_handler_ = &WeightChunkController::HandleDefaultPostInvoke;

  size_t record_chunk_index_ = 0;
  size_t bpf_probe_prev_offset_ = 0;

  size_t weight_chunk_buffer_requirement_ = 0;
  size_t weight_chunk_buffer_capacity_ = 0;

  bool bpf_probe_first_call_ = true;
  int active_weight_chunk_buffer_index_ = 0;
  void* weight_chunk_buffer_base_ = nullptr;

  std::array<bool, 2> first_prefetch_per_mode_{{true, true}};
  std::array<BufferSlot, 2> buffer_slots_{};

  std::unordered_map<size_t, WeightChunkInfo> offset_to_chunk_info_;
  std::unordered_map<size_t, std::array<void*, 2>> offset_to_weights_ptr_;
};

}  // namespace streaming
}  // namespace flash_slim

#endif  // FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_H_
