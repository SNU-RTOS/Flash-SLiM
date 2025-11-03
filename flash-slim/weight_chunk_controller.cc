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

namespace {

constexpr const char* kPrefillMode = "PREFILL";
constexpr const char* kDecodeMode = "DECODE";

static inline size_t AlignTo(size_t value, size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const size_t remainder = value % alignment;
  return remainder == 0 ? value : value + (alignment - remainder);
}

}  // namespace

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
    if (weight_chunk_buffers_[0] && weight_chunk_buffers_[1] &&
        weight_chunk_buffer_capacity_ > 0) {
      prefetcher_->ConfigureIOBuffers(weight_chunk_buffers_[0], weight_chunk_buffer_capacity_,
                                      weight_chunk_buffers_[1], weight_chunk_buffer_capacity_);
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
    const int mode_index = prefetcher_->PrefetchModeToIndex(mode);
    if (mode_index >= 0 &&
        mode_index < static_cast<int>(first_prefetch_per_mode_.size())) {
      first_prefetch_per_mode_[mode_index] = true;
    }
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

  // weight_chunk_buffer_requirement_ used to rely on max_aligned_size, which forced the classic
  // ping-pong allocation of two equally sized buffers. We now size the single span by the maximum
  // sum of two consecutive chunks instead, so the old double-buffer allocation is kept here as a
  // comment for reference only.
  // weight_chunk_buffer_requirement_ = loader.max_aligned_size();

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
            << ", ranges=" << prefill_plan.io_order_ranges.size() << std::endl;
  std::cout << "[INFO] DECODE : offset_to_index=" << decode_plan.offset_to_index.size()
            << ", chunks=" << decode_plan.chunks.size()
            << ", ranges=" << decode_plan.io_order_ranges.size() << std::endl;

  prefetcher_->SetPrefetchPlan(WeightChunkPrefetcher::PrefetchMode::PREFILL,
                               std::move(prefill_plan.offset_to_index),
                               std::move(prefill_plan.chunks),
                               std::move(prefill_plan.io_order_ranges),
                               std::move(prefill_plan.chunk_index_to_range),
                               std::move(prefill_plan.io_order_to_range_index));

  prefetcher_->SetPrefetchPlan(WeightChunkPrefetcher::PrefetchMode::DECODE,
                               std::move(decode_plan.offset_to_index),
                               std::move(decode_plan.chunks),
                               std::move(decode_plan.io_order_ranges),
                               std::move(decode_plan.chunk_index_to_range),
                               std::move(decode_plan.io_order_to_range_index));

  prefetcher_->BuildIndexToChunksFromPlans();

  offset_to_weights_ptr_.clear();
  offset_to_chunk_info_.clear();
  chunk_index_ = 0;
  first_prefetch_per_mode_.fill(true);

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
    weight_chunk_buffers_[0] = weight_chunk_buffer_base_;
    weight_chunk_buffers_[1] = weight_chunk_buffer_base_;
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
  weight_chunk_buffers_[0] = buffer;
  weight_chunk_buffers_[1] = buffer;

  // Legacy double-buffer allocation (based on max_aligned_size ping-pong) is intentionally kept
  // here as a comment to capture the previous design. Under the new sliding window scheme we reuse
  // a single contiguous span and only move logical offsets for the two slots.
  /*
  bool allocation_failed = false;
  for (int i = 0; i < 2; ++i) {
    void* legacy_buffer = nullptr;
    if (posix_memalign(&legacy_buffer, alignment, aligned_size) != 0) {
      std::perror("WeightChunkController::AllocWeightChunkBuffer posix_memalign");
      allocation_failed = true;
      break;
    }
    std::memset(legacy_buffer, 0, aligned_size);
    weight_chunk_buffers_[i] = legacy_buffer;
  }
  if (allocation_failed) {
    ReleaseWeightChunkBuffer();
    weight_chunk_buffer_requirement_ = 0;
    weight_chunk_buffer_capacity_ = 0;
    return;
  }
  */

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
  weight_chunk_buffers_.fill(nullptr);
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
  if (index < 0 || index >= static_cast<int>(weight_chunk_buffers_.size())) {
    return nullptr;
  }
  if (!weight_chunk_buffer_base_) {
    return nullptr;
  }
  void* base = weight_chunk_buffers_.at(index);
  if (!base) {
    return nullptr;
  }
  const size_t offset = buffer_offsets_.at(index);
  if (weight_chunk_buffer_capacity_ > 0 &&
      offset >= weight_chunk_buffer_capacity_) {
    return nullptr;
  }
  auto* byte_base = static_cast<uint8_t*>(base);
  return byte_base + offset;
}

void* WeightChunkController::GetActiveWeightChunkBuffer() const {
  return GetWeightChunkBufferAddr(active_weight_chunk_buffer_index_);
}

void WeightChunkController::RecordChunkAccess(size_t offset) {
  if (!provider_) {
    return;
  }

  if (offset_to_chunk_info_.find(offset) != offset_to_chunk_info_.end()) {
    return;
  }

  const size_t buffer_size = provider_->GetBufferSize(offset);
  const int weights_id = provider_->GetWeightsId(offset);
  const size_t sector_size = provider_->GetDirectIOBufferSectorSize();
  const size_t abs_offset = provider_->GetMMapBaseOffset() + offset;

  const size_t aligned_offset = (abs_offset / sector_size) * sector_size;
  const size_t offset_adjust = abs_offset - aligned_offset;
  const size_t aligned_size = AlignTo(buffer_size + offset_adjust, sector_size);


  weight_chunk_info_t info;
  info.chunk_index = chunk_index_;
  info.aligned_offset = aligned_offset;
  info.offset_adjust = offset_adjust;
  info.aligned_size = aligned_size;
  info.origin_offset = offset;
  info.origin_size = buffer_size;
  info.weights_id = weights_id >= 0 ? static_cast<size_t>(weights_id) : 0;


  offset_to_chunk_info_.insert({info.origin_offset, info});
//   offset_to_chunk_info_.emplace(offset, info);

//   printf("[WeightChunkController] Recorded chunk_index=%zu for offset=%zu, aligned_offset=%zu \
//     aligned_size=%zu origin_offset=%zu origin_size=%zu weights_id=%zu\n",
//          info.chunk_index, offset, info.aligned_offset, info.aligned_size,
//          info.origin_offset, info.origin_size, info.weights_id);
  chunk_index_ = chunk_index_ + 1;
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

bool WeightChunkController::ScheduleNextRange(const PrefetchChunkRange* current_range, int fd) {
  if (!prefetcher_ || !current_range ||
      prefetcher_->GetPrefetchMode() == PrefetchMode::UNINITIALIZED) {
    std::cerr << "[WeightChunkController] Invalid state for scheduling next chunk range\n";
    return false;
  }

  const PrefetchChunkRange* next_range =
      prefetcher_->GetNextChunkRange(prefetcher_->GetPrefetchMode(), current_range->io_order);
  if (!next_range || next_range == current_range) {
    return true;
  }

//   std::cout << "[WeightChunkController] ScheduleNextRange: current_io_order="
//             << current_range->io_order << " -> next_io_order=" << next_range->io_order
//             << " (size=" << next_range->total_aligned_size << ")" << std::endl;

  if (next_range->total_aligned_size > weight_chunk_buffer_capacity_) {
    std::cerr << "[WeightChunkController] Next range (io_order=" << next_range->io_order
              << ") does not fit into the configured buffer span. aligned_size="
              << next_range->total_aligned_size
              << ", span=" << weight_chunk_buffer_capacity_ << "\n";
    return false;
  }

  const size_t target_offset = ComputeInactiveSlotOffset(next_range->total_aligned_size);
  if (target_offset == std::numeric_limits<size_t>::max()) {
    std::cerr << "[WeightChunkController] Next range (io_order=" << next_range->io_order
              << ") does not fit into the configured buffer span.\n";
    return false;
  }

  const int buffer_index = GetInactiveBufferIndex();
  if (buffer_range_ids_[buffer_index] == next_range->io_order) {
    // std::cout << "[WeightChunkController] ScheduleNextRange: io_order="
    //           << next_range->io_order << " already resident in buffer_index="
    //           << buffer_index << std::endl;
    return true;
  }
  buffer_offsets_[buffer_index] = target_offset;
  buffer_sizes_[buffer_index] = next_range->total_aligned_size;
  void* buffer_base = GetWeightChunkBufferAddr(buffer_index);
  if (!buffer_base) {
    std::cerr << "[WeightChunkController] Invalid inactive buffer index " << buffer_index
              << "\n";
    buffer_sizes_[buffer_index] = 0;
    buffer_offsets_[buffer_index] = 0;
    return false;
  }

  WeightChunkPrefetcher::PrefetchRequest request;
  request.chunk_range = next_range;
  request.buffer_base = buffer_base;
  request.direct_io_fd = fd;
  request.buffer_index = buffer_index;

  if (!prefetcher_->Submit(request)) {
    std::cerr << "[WeightChunkController] Next chunk range submit failed for io_order="
              << next_range->io_order << "\n";
    buffer_sizes_[buffer_index] = 0;
    buffer_offsets_[buffer_index] = 0;
    return false;
  }

//   std::cout << "[WeightChunkController] ScheduleNextRange: submitted io_order="
//             << next_range->io_order << " to buffer_index=" << buffer_index
//             << " offset=" << target_offset << std::endl;

  buffer_range_ids_[buffer_index] = next_range->io_order;

  return true;
}

size_t WeightChunkController::ComputeInactiveSlotOffset(size_t next_aligned_size) const {
  if (weight_chunk_buffer_capacity_ == 0 ||
      next_aligned_size > weight_chunk_buffer_capacity_) {
    return std::numeric_limits<size_t>::max();
  }

  const int active_index = active_weight_chunk_buffer_index_;
  const size_t active_offset = buffer_offsets_.at(active_index);
  const size_t active_size = buffer_sizes_.at(active_index);

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
  buffer_offsets_.fill(0);
  buffer_sizes_.fill(0);
  buffer_range_ids_.fill(std::numeric_limits<size_t>::max());
}

void WeightChunkController::UpdateWeightsPointer(size_t offset, const weight_chunk_info_t& info,
                                                 const PrefetchChunkRange& range) {
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

  const size_t relative_offset = FindChunkRelativeOffset(range, info.chunk_index);

  auto** slot = reinterpret_cast<uint8_t**>(target_slot);
  *slot = static_cast<uint8_t*>(buffer_base) + relative_offset + info.offset_adjust;
}

size_t WeightChunkController::FindChunkRelativeOffset(const PrefetchChunkRange& range,
                                                      size_t chunk_index) const {
  for (size_t i = 0; i < range.chunk_indices.size(); ++i) {
    if (range.chunk_indices[i] == chunk_index) {
      if (i < range.chunk_relative_offsets.size()) {
        return range.chunk_relative_offsets[i];
      }
      break;
    }
  }
  std::cerr << "[WeightChunkController] Warning: relative offset not found for chunk_index="
            << chunk_index << " in range io_order=" << range.io_order << std::endl;
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
  
  const weight_chunk_info_t& chunk_info = it->second;
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

  const PrefetchMode prefetch_mode = prefetcher_->GetPrefetchMode();
  if (prefetch_mode == PrefetchMode::UNINITIALIZED) {
    return false;
  }

//   std::cout << "[WeightChunkController] RuntimePreInvoke: offset=" << offset << " Try to LookupChunkInfo"
            // << std::endl;

  const weight_chunk_info_t* current_chunk_info = prefetcher_->LookupChunkInfo(prefetch_mode, offset);
  if (!current_chunk_info) {
    return false;
  }

//   std::cout << "[WeightChunkController] RuntimePreInvoke: Requesting ChunkRange with chunk_index="
//             << current_chunk_info->chunk_index << " for offset=" << offset << std::endl;

  const PrefetchChunkRange* current_range =
      prefetcher_->GetChunkRangeByChunkIndex(prefetch_mode, current_chunk_info->chunk_index);
  if (!current_range) {
    std::cerr << "[WeightChunkController] Failed to resolve chunk range for chunk_index="
              << current_chunk_info->chunk_index << "\n";
    return false;
  }

  const bool is_first_chunk_in_range = !current_range->chunk_indices.empty() &&
      current_range->chunk_indices.front() == current_chunk_info->chunk_index;

  if (current_range->total_aligned_size == 0 ||
      current_range->total_aligned_size > weight_chunk_buffer_capacity_) {
    std::cerr << "[WeightChunkController] Chunk range size=" << current_range->total_aligned_size
              << " exceeds allocated buffer span=" << weight_chunk_buffer_capacity_ << "\n";
    return false;
  }

  int active_index = active_weight_chunk_buffer_index_;
  if (buffer_range_ids_[active_index] != current_range->io_order) {
    const int other_index = GetInactiveBufferIndex();
    if (buffer_range_ids_[other_index] == current_range->io_order) {
      UpdateActiveBufferIndex(other_index);
      active_index = other_index;
    }
  }
  buffer_range_ids_[active_index] = current_range->io_order;

//   std::cout << "[WeightChunkController] RuntimePreInvoke: Ensuring chunk_index="
//             << current_chunk_info->chunk_index << " for offset=" << offset << "\n";

  const int fd = provider_->GetDirectIOFileDescriptor();
  if (fd < 0) {
    return false;
  }

  // 2. Ensure chunk is ready (handles cache hit via ChunkIOState automatically)
  const int mode_index = prefetcher_->GetPrefetchModeIndex();
  if (mode_index < 0 ||
      mode_index >= static_cast<int>(first_prefetch_per_mode_.size())) {
    return false;
  }

  bool& first_prefetch = first_prefetch_per_mode_[mode_index];
  if (first_prefetch) {  // Synchronously submit prefetch request and wait only for the first call
    buffer_offsets_[active_index] = 0;
    buffer_sizes_[active_index] = current_range->total_aligned_size;
    void* buffer_base = GetWeightChunkBufferAddr(active_weight_chunk_buffer_index_);
    if (!buffer_base) {
      std::cerr << "[WeightChunkController] Invalid buffer index "
                << active_weight_chunk_buffer_index_ << "\n";
      return false;
    }
    WeightChunkPrefetcher::PrefetchRequest request;
    request.chunk_range = current_range;
    request.buffer_base = buffer_base;
    request.direct_io_fd = fd;
    request.buffer_index = active_weight_chunk_buffer_index_;
    if (!prefetcher_->Submit(request)) {
      std::cerr << "[WeightChunkController] Submit failed for range io_order="
                << current_range->io_order << "\n";
      return false;
    }
    first_prefetch = false;
  } else {
    buffer_sizes_[active_index] = current_range->total_aligned_size;
  }
    
  // WaitReady returns immediately if already ready
//   printf("WeightChunkController: Waiting for chunk_index=%zu\n", current_chunk_info->chunk_index);
//   std::cout << "[WeightChunkController] RuntimePreInvoke: Waiting for chunk_index="
//             << current_chunk_info->chunk_index << std::endl;
  if (!prefetcher_->WaitReady(current_chunk_info)) {
    std::cerr << "[WeightChunkController] WaitReady failed for chunk_index="
            << current_chunk_info->chunk_index << "\n";
    return false;
  }
//   std::cout << "[WeightChunkController] RuntimePreInvoke: chunk_index="
//             << current_chunk_info->chunk_index << " ready" << std::endl;

  // 3. Update state and pointer
  UpdateWeightsPointer(offset, *current_chunk_info, *current_range);
//   std::cout << "[WeightChunkController] RuntimePreInvoke: Updated pointer for chunk_index="
//             << current_chunk_info->chunk_index << std::endl;
  
  // 4. Schedule next chunk to inactive buffer asynchronously
  if (is_first_chunk_in_range) {
    if (!ScheduleNextRange(current_range, fd)) {
      std::cerr << "[WeightChunkController] ScheduleNextRange failed after chunk_index="
                << current_chunk_info->chunk_index << "\n";
      return false;
    }
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
void WeightChunkController::PreInvokeImpl(size_t offset) {
  (void)(this->*preinvoke_handler_)(offset);
}

void WeightChunkController::PostInvokeImpl(size_t /*offset*/) {
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

int WeightChunkController::FetchArgIntImpl() {
    const int mode_idx = prefetcher_->GetPrefetchModeIndex();
    if (mode_idx < 0) {
        return -1;
    }
    return mode_idx;
}


}  // namespace streaming
}  // namespace flash_slim
