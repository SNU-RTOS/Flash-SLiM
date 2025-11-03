#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include <sys/sdt.h>

#include "weight_chunk_controller.h"

namespace flash_slim {
namespace streaming {

WeightChunkController::WeightChunkController(
    tflite::xnnpack::StreamingWeightCacheProvider* provider)
    : provider_(provider) {
    if (provider_) {
        provider_->SetController(this);
        auto provider_mode = provider_->GetProviderMode();
        UpdatePreInvokeHandler(provider_mode);
        UpdatePostInvokeHandler(provider_mode);
    } else {
        std::cerr << "[WeightChunkController] Provider is nullptr\n";
    }
}

WeightChunkController::~WeightChunkController() {
  if (prefetcher_) {
    prefetcher_->StopWorker();
  }
  ReleaseWeightChunkBuffer();
}

void WeightChunkController::AttachPrefetcher(
    std::unique_ptr<WeightChunkPrefetcher> prefetcher,
    const std::vector<int>& io_cores) {
  if (prefetcher) {
    prefetcher->SetWorkerThreadAffinity(io_cores);
  }
  return AttachPrefetcher(std::move(prefetcher));
}

void WeightChunkController::AttachPrefetcher(
    std::unique_ptr<WeightChunkPrefetcher> prefetcher) {
  if (prefetcher_) {
    prefetcher_->StopWorker();
  }
  prefetcher_ = std::move(prefetcher);
  if (prefetcher_) {
    if (buffer_slots_[0].base && buffer_slots_[1].base &&
        weight_chunk_buffer_capacity_ > 0) {
      prefetcher_->ConfigureIOBuffers(buffer_slots_[0].base, weight_chunk_buffer_capacity_,
                                      buffer_slots_[1].base, weight_chunk_buffer_capacity_);
      prefetcher_->StartWorker();
    }
  }
}

void WeightChunkController::AttachMetadataWriter(
    WeightChunkMetaDataWriter* writer) {
  writer_ = writer;
}

void WeightChunkController::UpdateProviderMode(ProviderMode mode) {
  if (provider_) {
    provider_->UpdateProviderMode(mode);
    UpdatePreInvokeHandler(mode);
    UpdatePostInvokeHandler(mode);
  } else {
    std::cerr << "[WeightChunkController] Provider is nullptr\n";
  }
}

void WeightChunkController::UpdatePrefetcherMode(PrefetchMode mode) {
  if (prefetcher_) {
    prefetcher_->UpdatePrefetcherMode(mode);
  } else {
    std::cerr << "[WeightChunkController] Prefetcher is nullptr\n";
  }
}

bool WeightChunkController::LoadPrefetchPlan(const std::string& path) {
  if (!prefetcher_) {
    std::cerr << "[WeightChunkController] Prefetcher not attached\n";
    return false;
  }

  JsonPrefetchPlanLoader loader;
  if (!loader.LoadFromFile(path)) {
    return false;
  }

  const uint64_t plan_buffer_span = loader.weight_chunk_buffer_size();
  weight_chunk_buffer_requirement_ = plan_buffer_span;
  if (weight_chunk_buffer_requirement_ == 0) {
    weight_chunk_buffer_requirement_ = loader.max_aligned_size();
  }

  std::cout << "[INFO] Allocating weight cache buffer of size "
            << weight_chunk_buffer_requirement_ << " bytes.\n";
  std::cout << "[INFO] weight_chunk_buffer_size (max consecutive pair sum): "
            << (plan_buffer_span ? plan_buffer_span : weight_chunk_buffer_requirement_) << " bytes.\n";
  std::cout << "[INFO] Loaded prefetch plan for model: " << loader.model()
            << ", planner version: " << loader.version() << "\n";

  if (provider_) {
    AllocWeightChunkBuffer(weight_chunk_buffer_requirement_);
  }

  loader.PrintMetadata(std::cout);

  auto prefill_plan = loader.BuildModeChunkPlan(kPrefillMode);
  auto decode_plan = loader.BuildModeChunkPlan(kDecodeMode);

  std::cout << "[INFO] PREFILL: offset_to_index=" << prefill_plan.offset_to_index.size()
            << ", chunks=" << prefill_plan.chunks.size()
            << ", groups=" << prefill_plan.io_order_groups.size() << std::endl;
  std::cout << "[INFO] DECODE : offset_to_index=" << decode_plan.offset_to_index.size()
            << ", chunks=" << decode_plan.chunks.size()
            << ", groups=" << decode_plan.io_order_groups.size() << std::endl;

  prefetcher_->SetPrefetchPlan(WeightChunkPrefetcher::PrefetchMode::PREFILL,
                               std::move(prefill_plan.offset_to_index),
                               std::move(prefill_plan.chunks),
                               std::move(prefill_plan.io_order_groups));

  prefetcher_->SetPrefetchPlan(WeightChunkPrefetcher::PrefetchMode::DECODE,
                               std::move(decode_plan.offset_to_index),
                               std::move(decode_plan.chunks),
                               std::move(decode_plan.io_order_groups));

  prefetcher_->BuildIndexToChunksFromPlans();

  offset_to_weights_ptr_.clear();
  offset_to_chunk_info_.clear();
  record_chunk_index_ = 0;
  initial_prefetch_pending_ = true;

  return true;
}

void WeightChunkController::AllocWeightChunkBuffer(size_t size) {
  if (!provider_ || size == 0) {
    return;
  }

  const size_t alignment = provider_->GetDirectIOBufferSectorSize();
  const size_t aligned_size = AlignTo(size, alignment);

  if (weight_chunk_buffer_capacity_ >= aligned_size &&
      weight_chunk_buffer_base_) {
    weight_chunk_buffer_requirement_ = size;
    ResetBufferSlots();
    UpdateActiveBufferIndex(0);
    buffer_slots_[0].base = weight_chunk_buffer_base_;
    buffer_slots_[1].base = weight_chunk_buffer_base_;
    if (prefetcher_) {
      prefetcher_->ConfigureIOBuffers(weight_chunk_buffer_base_, weight_chunk_buffer_capacity_,
                                      weight_chunk_buffer_base_, weight_chunk_buffer_capacity_);
      prefetcher_->StartWorker();
    }
    return;
  }

  if (prefetcher_) {
    prefetcher_->StopWorker();
  }
  ReleaseWeightChunkBuffer();

  void* buffer = nullptr;
  if (posix_memalign(&buffer, alignment, aligned_size) != 0) {
    std::perror("WeightChunkController::AllocWeightChunkBuffer posix_memalign");
    weight_chunk_buffer_requirement_ = 0;
    weight_chunk_buffer_capacity_ = 0;
    return;
  }
  std::memset(buffer, 0, aligned_size);
  weight_chunk_buffer_base_ = buffer;
  buffer_slots_[0].base = buffer;
  buffer_slots_[1].base = buffer;

  weight_chunk_buffer_requirement_ = size;
  weight_chunk_buffer_capacity_ = aligned_size;
  ResetBufferSlots();
  UpdateActiveBufferIndex(0);
  if (prefetcher_) {
    prefetcher_->ConfigureIOBuffers(weight_chunk_buffer_base_, weight_chunk_buffer_capacity_,
                                    weight_chunk_buffer_base_, weight_chunk_buffer_capacity_);
    prefetcher_->StartWorker();
  }
}

void WeightChunkController::ReleaseWeightChunkBuffer() {
  if (weight_chunk_buffer_base_) {
    std::free(weight_chunk_buffer_base_);
    weight_chunk_buffer_base_ = nullptr;
  }
  for (auto& slot : buffer_slots_) {
    slot.base = nullptr;
  }
  ResetBufferSlots();
  if (prefetcher_) {
    prefetcher_->ConfigureIOBuffers(nullptr, 0, nullptr, 0);
  }
  weight_chunk_buffer_capacity_ = 0;
  weight_chunk_buffer_requirement_ = 0;
  active_weight_chunk_buffer_index_ = 0;
}


void WeightChunkController::SwitchActiveBufferIndex() {
  active_weight_chunk_buffer_index_ = 1 - active_weight_chunk_buffer_index_;
}

void WeightChunkController::UpdateActiveBufferIndex(int index) {
  active_weight_chunk_buffer_index_ = index;
}

void* WeightChunkController::GetWeightChunkBufferAddr(int index) const {
  if (index < 0 || index >= static_cast<int>(buffer_slots_.size())) {
    return nullptr;
  }
  if (!weight_chunk_buffer_base_) {
    return nullptr;
  }
  const WeightChunkBufferSlot& slot = buffer_slots_.at(index);
  if (!slot.base) {
    return nullptr;
  }
  if (weight_chunk_buffer_capacity_ > 0 &&
      slot.offset >= weight_chunk_buffer_capacity_) {
    return nullptr;
  }
  auto* byte_base = static_cast<uint8_t*>(slot.base);
  return byte_base + slot.offset;
}

inline void* WeightChunkController::GetActiveWeightChunkBuffer() const {
  return GetWeightChunkBufferAddr(active_weight_chunk_buffer_index_);
}

void WeightChunkController::RecordChunkAccess(size_t offset) {
  if (!provider_) {
    return;
  }

  if (offset_to_chunk_info_.find(offset) != offset_to_chunk_info_.end()) {
    return;
  }

  const int weights_id = provider_->GetWeightsId(offset);
  const size_t buffer_size = provider_->GetBufferSize(offset);
  const size_t sector_size = provider_->GetDirectIOBufferSectorSize();
  const size_t abs_offset = provider_->GetMMapBaseOffset() + offset;

  const size_t aligned_offset = (abs_offset / sector_size) * sector_size;
  const size_t offset_adjust = abs_offset - aligned_offset;
  const size_t aligned_size = AlignTo(buffer_size + offset_adjust, sector_size);

  WeightChunkInfo info;
  info.chunk_index = record_chunk_index_;
  info.aligned_offset = aligned_offset;
  info.offset_adjust = offset_adjust;
  info.aligned_size = aligned_size;
  info.origin_offset = offset;
  info.origin_size = buffer_size;
  info.weights_id = weights_id >= 0 ? static_cast<size_t>(weights_id) : 0;

  offset_to_chunk_info_.insert({info.origin_offset, info});

//   printf("[WeightChunkController] Recorded chunk_index=%zu for offset=%zu, aligned_offset=%zu \
//     aligned_size=%zu origin_offset=%zu origin_size=%zu weights_id=%zu\n",
//          info.chunk_index, offset, info.aligned_offset, info.aligned_size,
//          info.origin_offset, info.origin_size, info.weights_id);
  record_chunk_index_ = record_chunk_index_ + 1;
}

void WeightChunkController::UpdatePreInvokeHandler(ProviderMode mode) {
  switch (mode) {
    case ProviderMode::PRE_RUN_WARMUP:
      preinvoke_handler_ = &WeightChunkController::HandlePreRunWarmUpPreInvoke;
      break;
    case ProviderMode::PRE_RUN_PROFILE:
      preinvoke_handler_ = &WeightChunkController::HandlePreRunProfilePreInvoke;
      break;
    case ProviderMode::RUNTIME:
      preinvoke_handler_ = &WeightChunkController::HandleRuntimePreInvoke;
      break;
    default:
      preinvoke_handler_ = &WeightChunkController::HandleDefaultPreInvoke;
      break;
  }
}

void WeightChunkController::UpdatePostInvokeHandler(ProviderMode mode) {
  switch (mode) {
    case ProviderMode::PRE_RUN_WARMUP:
      postinvoke_handler_ = &WeightChunkController::HandlePreRunWarmUpPostInvoke;
      break;
    case ProviderMode::PRE_RUN_PROFILE:
      postinvoke_handler_ = &WeightChunkController::HandlePreRunProfilePostInvoke;
      break;
    case ProviderMode::RUNTIME:
      postinvoke_handler_ = &WeightChunkController::HandleRuntimePostInvoke;
      break;
    default:
      postinvoke_handler_ = &WeightChunkController::HandleDefaultPostInvoke;
      break;
  }
}

bool WeightChunkController::ScheduleNextGroup(const WeightChunkGroupInfo* current_group, int fd) {

  if (!prefetcher_ || !current_group ) {
    std::cerr << "[WeightChunkController] Invalid state for scheduling next chunk group\n";
    return false;
  }

  // 1. Get next chunk group info
  PrefetchMode requested_mode = prefetcher_->GetPrefetchMode();
  PrefetchMode next_mode = requested_mode;
  const WeightChunkGroupInfo* next_group = prefetcher_->GetNextChunkGroup(requested_mode, current_group->io_order, &next_mode);
  
  if (!next_group || next_group == current_group) {
    // already loaded
    return true;
  }

  // check weight chunk buffer capacity
  if (next_group->total_aligned_size > weight_chunk_buffer_capacity_) {
    std::cerr << "[WeightChunkController] Next group (io_order=" << next_group->io_order
              << ") does not fit into the configured buffer span. aligned_size="
              << next_group->total_aligned_size
              << ", span=" << weight_chunk_buffer_capacity_ << "\n";
    return false;
  }

  const size_t target_slot_offset = ComputeInactiveSlotOffset(next_group->total_aligned_size);
  if (target_slot_offset == std::numeric_limits<size_t>::max()) {
    std::cerr << "[WeightChunkController] Next group (io_order=" << next_group->io_order
              << ") does not fit into the configured buffer span.\n";
    return false;
  }

  const int buffer_index = GetInactiveBufferIndex();

  WeightChunkBufferSlot& slot = buffer_slots_[buffer_index];
  slot.offset = target_slot_offset;
  slot.size = next_group->total_aligned_size;
  
  void* buffer_base = GetWeightChunkBufferAddr(buffer_index);
  if (!buffer_base) {
    std::cerr << "[WeightChunkController] Invalid inactive buffer index " << buffer_index
              << "\n";
    slot.size = 0;
    slot.offset = 0;
    slot.io_order = std::numeric_limits<size_t>::max();
    return false;
  }

  WeightChunkPrefetcher::PrefetchRequest request;
  request.chunk_group = next_group;
  request.buffer_base = buffer_base;
  request.direct_io_fd = fd;
  request.buffer_index = buffer_index;
  request.mode = next_mode;

  if (!prefetcher_->Submit(request)) {
    std::cerr << "[WeightChunkController] Next chunk group submit failed for io_order="
        << next_group->io_order << "\n";
    slot.size = 0;
    slot.offset = 0;
    slot.io_order = std::numeric_limits<size_t>::max();
    return false;
  }

  slot.io_order = next_group->io_order;

  return true;
}

size_t WeightChunkController::ComputeInactiveSlotOffset(size_t next_aligned_size) const {
  if (weight_chunk_buffer_capacity_ == 0 ||
      next_aligned_size > weight_chunk_buffer_capacity_) {
    return std::numeric_limits<size_t>::max();
  }

  const int active_index = active_weight_chunk_buffer_index_;
  const WeightChunkBufferSlot& active_slot = buffer_slots_.at(active_index);
  const size_t active_offset = active_slot.offset;
  const size_t active_size = active_slot.size;

  if (active_size == 0) {
    // First chunk or state reset; place at the beginning.
    return 0;
  }

  if (active_offset == 0) {
    if (active_size + next_aligned_size > weight_chunk_buffer_capacity_) {
      return std::numeric_limits<size_t>::max();
    }
    return weight_chunk_buffer_capacity_ - next_aligned_size;
  }

  if (active_size + next_aligned_size > weight_chunk_buffer_capacity_) {
    return std::numeric_limits<size_t>::max();
  }

  return 0;
}

void WeightChunkController::ResetBufferSlots() {
  for (auto& slot : buffer_slots_) {
    slot.offset = 0;
    slot.size = 0;
    slot.io_order = std::numeric_limits<size_t>::max();
  }
}

void WeightChunkController::UpdateWeightsPointer(size_t offset, const WeightChunkInfo& info,
                                                 const WeightChunkGroupInfo& group) {
  const int mode_idx = prefetcher_->GetPrefetchModeIndex();
  if (mode_idx < 0) {
    return;
  }

  auto entry = offset_to_weights_ptr_.find(offset);
  if (entry == offset_to_weights_ptr_.end()) {
    printf("[WeightChunkController] Warning: no weights pointer entry for offset=%zu\n", offset);
    return;
  }

  void* target_slot = entry->second[mode_idx];
  if (!target_slot) {
    return;
  }

  void* buffer_base = GetActiveWeightChunkBuffer();
  if (!buffer_base) {
    return;
  }

  const size_t relative_offset = FindChunkRelativeOffset(group, info.chunk_index);

  auto** slot = reinterpret_cast<uint8_t**>(target_slot);
  *slot = static_cast<uint8_t*>(buffer_base) + relative_offset + info.offset_adjust;
}

size_t WeightChunkController::FindChunkRelativeOffset(const WeightChunkGroupInfo& group,
                                                      size_t chunk_index) const {
  for (size_t i = 0; i < group.chunk_indices.size(); ++i) {
    if (group.chunk_indices[i] == chunk_index) {
      if (i < group.chunk_relative_offsets.size()) {
        return group.chunk_relative_offsets[i];
      }
      break;
    }
  }
  std::cerr << "[WeightChunkController] Warning: relative offset not found for chunk_index="
            << chunk_index << " in group io_order=" << group.io_order << std::endl;
  return 0;
}

bool WeightChunkController::HandlePreRunWarmUpPreInvoke(size_t offset) {
  if (!writer_) {
    std::cerr << "[WeightChunkController] writer_ is null\n";
    return false;
  }

  auto it = offset_to_chunk_info_.find(offset);
  if (it == offset_to_chunk_info_.end()) {
    RecordChunkAccess(offset);
    it = offset_to_chunk_info_.find(offset);
    if (it == offset_to_chunk_info_.end()) {
        std::cerr << "[WeightChunkController] Failed to record chunk info for offset "
                  << offset << "\n";
      return false;
    }
  }

  writer_->WriteChunkInfo(it->second, prefetcher_->GetPrefetchMode());
  return true;
}

void WeightChunkController::EmitBPFProbe(size_t offset) {
  
  auto it = offset_to_chunk_info_.find(offset);
  if (it == offset_to_chunk_info_.end()) {
    std::cerr << "[WeightChunkController] BPF probe: chunk not found for offset=" 
              << offset << "\n";
    return;
  }
  
  const WeightChunkInfo& chunk_info = it->second;
  const std::string mode_string = prefetcher_->GetPrefetchModeString();
  const char* prefetch_mode_str = mode_string.c_str();

  DTRACE_PROBE3(text_gen, ops_check, 
    static_cast<uint64_t>(chunk_info.chunk_index),
                static_cast<uint64_t>(offset), 
                const_cast<char*>(prefetch_mode_str));
}

bool WeightChunkController::ResetBPFProbe(){
 EmitBPFProbe(bpf_probe_prev_offset_);

 bpf_probe_prev_offset_ = 0;
 bpf_probe_first_call_ = true;

 return true;
}

bool WeightChunkController::HandlePreRunProfilePreInvoke(size_t offset) {

  if (bpf_probe_first_call_) {
    DTRACE_PROBE(text_gen, ops_start);
    bpf_probe_first_call_ = false;
  }
  else{
    EmitBPFProbe(bpf_probe_prev_offset_);
    DTRACE_PROBE(text_gen, ops_start);
  }
  
  bpf_probe_prev_offset_ = offset;
  return true;
}

bool WeightChunkController::HandleRuntimePreInvoke(size_t offset) {

  // 1. Validate prerequisites
  if (!prefetcher_ || !provider_ ) {
    return false;
  }

  // 2. Get current chunk info and group
  const PrefetchMode prefetch_mode = prefetcher_->GetPrefetchMode();
  if (prefetch_mode == PrefetchMode::UNINITIALIZED) {
    return false;
  }

  const WeightChunkInfo* current_chunk_info = prefetcher_->LookupChunkInfo(prefetch_mode, offset);
  if (!current_chunk_info) {
    return false;
  }

  const WeightChunkGroupInfo* current_group = prefetcher_->GetChunkGroupByChunkIndex(prefetch_mode, current_chunk_info->chunk_index);
  if (!current_group) {
    std::cerr << "[WeightChunkController] Failed to resolve chunk group for chunk_index="
              << current_chunk_info->chunk_index << "\n";
    return false;
  }


  // 3. Ensure active buffer slot matches current group
  // if not, try to swap to the other buffer slot
  if (buffer_slots_[active_weight_chunk_buffer_index_].io_order != current_group->io_order) {
    SwitchActiveBufferIndex();
  }



  // 4. Get direct IO file descriptor
  const int fd = provider_->GetDirectIOFileDescriptor();
  if (fd < 0) {
    return false;
  }

  // 5. Submit prefetch request if first time 
  if (initial_prefetch_pending_) {  // Synchronously submit prefetch request only for the very first call
    WeightChunkBufferSlot& active_slot = buffer_slots_[active_weight_chunk_buffer_index_];
    active_slot.io_order = current_group->io_order;
    active_slot.size = current_group->total_aligned_size;
    active_slot.offset = 0;

    void* buffer_base = GetActiveWeightChunkBuffer();

    if (!buffer_base) {
      std::cerr << "[WeightChunkController] Invalid buffer index "
                << active_weight_chunk_buffer_index_ << "\n";
      return false;
    }

    WeightChunkPrefetcher::PrefetchRequest request;
    request.chunk_group = current_group;
    request.buffer_base = buffer_base;
    request.direct_io_fd = fd;
    request.buffer_index = active_weight_chunk_buffer_index_;
    request.mode = prefetch_mode;

    if (!prefetcher_->Submit(request)) {
      std::cerr << "[WeightChunkController] Submit failed for group io_order="
                << current_group->io_order << "\n";
      return false;
    }

    initial_prefetch_pending_ = false;
  } 

  // WaitReady returns immediately if already ready
  if (!prefetcher_->WaitReady(current_chunk_info)) {
    std::cerr << "[WeightChunkController] WaitReady failed for chunk_index="
            << current_chunk_info->chunk_index << "\n";
    return false;
  }

  // 6. Update state and pointer
  UpdateWeightsPointer(offset, *current_chunk_info, *current_group);

  // Check if this is the first chunk in the group
  const bool is_first_chunk_in_group = !current_group->chunk_indices.empty() &&
      current_group->chunk_indices.front() == current_chunk_info->chunk_index;

  // 7. Schedule next chunk to inactive buffer asynchronously
  if (is_first_chunk_in_group && !ScheduleNextGroup(current_group, fd)) {
      std::cerr << "[WeightChunkController] ScheduleNextGroup failed after chunk_index="
                << current_chunk_info->chunk_index << "\n";
      return false;
  }

  return true;
}

bool WeightChunkController::HandleDefaultPreInvoke(size_t /*offset*/) {
  return true;
    
}

bool WeightChunkController::HandlePreRunWarmUpPostInvoke() {
    return true;
}

bool WeightChunkController::HandlePreRunProfilePostInvoke() {
    return true;
}

bool WeightChunkController::HandleRuntimePostInvoke() {
    return true;
}

bool WeightChunkController::HandleDefaultPostInvoke() {
    return true;
}

//* ================ Hook ================
inline void* WeightChunkController::OffsetToAddrImpl(size_t offset) {
    void* addr = nullptr;
    switch (provider_->GetProviderMode()) {
        case ProviderMode::RUNTIME:  // weight streaming path
            addr = GetActiveWeightChunkBuffer();  // return weight chunk buffer
            break;
        case ProviderMode::PRE_RUN_WARMUP:  // pre-runtime (warmup) path
            RecordChunkAccess(offset);
            addr = provider_->GetMmappedAddr(offset);  // return mmaped address
            break;
        case ProviderMode::PRE_RUN_PROFILE:  // pre-runtime (profile) path
            addr = GetActiveWeightChunkBuffer();  // return weight chunk buffer
            break;
        case ProviderMode::DEBUG_MMAP:  // general path(with mmap)
            addr = provider_->GetMmappedAddr(offset);
            break;
        default:
            addr = nullptr;
            break;
    }
    return addr;
}

inline void WeightChunkController::PreInvokeImpl(size_t offset) {
  (void)(this->*preinvoke_handler_)(offset);
}

inline void WeightChunkController::PostInvokeImpl(size_t /*offset*/) {
    (void)(this->*postinvoke_handler_)();
}

void WeightChunkController::TraceWeightsAddrImpl(void* addr, size_t offset) {
  if (!addr) {
    std::cerr << "[WeightChunkController] addr is nullptr\n";
    return;
  }

  const int mode_idx = prefetcher_->GetPrefetchModeIndex();
  if (mode_idx < 0) {
    return;
  }

  auto& entry = offset_to_weights_ptr_[offset];
  entry[mode_idx] = addr;
}

inline int WeightChunkController::FetchArgIntImpl() {
    const int mode_idx = prefetcher_->GetPrefetchModeIndex();
    if (mode_idx < 0) {
        return -1;
    }
    return mode_idx;
}


}  // namespace streaming
}  // namespace flash_slim
