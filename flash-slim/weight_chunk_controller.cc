#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "weight_chunk_controller.h"

namespace {

constexpr const char* kPrefillMode = "PREFILL";
constexpr const char* kDecodeMode = "DECODE";

inline size_t AlignTo(size_t value, size_t alignment) {
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
    }
  UpdatePreinvokeHandler(provider_mode_);
}

WeightChunkController::~WeightChunkController() {
  if (prefetcher_) {
    prefetcher_->StopWorker();
  }
  ReleaseWeightChunkBuffer();
}


void WeightChunkController::AttachPrefetcher(
    std::unique_ptr<WeightChunkPrefetcher> prefetcher) {
  if (prefetcher_) {
    prefetcher_->StopWorker();
  }
  prefetcher_ = std::move(prefetcher);
  if (prefetcher_) {
    prefetcher_->StartWorker();
    if (prefetch_mode_ != PrefetchMode::UNINITIALIZED) {
      prefetcher_->UpdatePrefetcherMode(prefetch_mode_);
    }
  }
}

void WeightChunkController::AttachMetadataWriter(
    WeightChunkMetaDataWriter* writer) {
  writer_ = writer;
}

void WeightChunkController::UpdateProviderMode(
    ProviderMode mode) {
  provider_mode_ = mode;
  UpdatePreinvokeHandler(mode);
  if (provider_) {
    provider_->UpdateProviderMode(mode);
  }
}

void WeightChunkController::UpdatePrefetcherMode(PrefetchMode mode) {
  prefetch_mode_ = mode;
  if (prefetcher_) {
    prefetcher_->UpdatePrefetcherMode(mode);
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

  weight_chunk_buffer_requirement_ = loader.max_aligned_size();

  std::cout << "[INFO] Allocating weight cache buffer of size "
            << weight_chunk_buffer_requirement_ << " bytes.\n";
  std::cout << "[INFO] Loaded prefetch plan for model: " << loader.model()
            << ", planner version: " << loader.version() << "\n";

  if (provider_) {
    AllocWeightChunkBuffer(weight_chunk_buffer_requirement_);
  }

  loader.PrintMetadata(std::cout);

  auto offset_to_index_prefill = loader.BuildOffsetToIndexForMode(kPrefillMode);
  auto offset_to_index_decode = loader.BuildOffsetToIndexForMode(kDecodeMode);
  auto chunks_prefill = loader.BuildIndexToChunkVectorForMode(kPrefillMode);
  auto chunks_decode = loader.BuildIndexToChunkVectorForMode(kDecodeMode);

  std::cout << "[INFO] PREFILL: offset_to_index=" << offset_to_index_prefill.size()
            << ", index_to_chunk=" << chunks_prefill.size() << std::endl;
  std::cout << "[INFO] DECODE : offset_to_index=" << offset_to_index_decode.size()
            << ", index_to_chunk=" << chunks_decode.size() << std::endl;

  prefetcher_->SetPrefetchPlan(WeightChunkPrefetcher::PrefetchMode::PREFILL,
                               std::move(offset_to_index_prefill),
                               std::move(chunks_prefill));
  prefetcher_->SetPrefetchPlan(WeightChunkPrefetcher::PrefetchMode::DECODE,
                               std::move(offset_to_index_decode),
                               std::move(chunks_decode));
  prefetcher_->BuildIndexToChunksFromPlans();

  offset_to_weights_ptr_.clear();
  offset_to_chunk_info.clear();
  next_chunk_index_ = 0;

  return true;
}

void WeightChunkController::AllocWeightChunkBuffer(size_t size) {
  if (!provider_ || size == 0) {
    return;
  }

  const size_t alignment = provider_->GetDirectIOBufferSectorSize();
  const size_t aligned_size = AlignTo(size, alignment);

  if (weight_chunk_buffer_capacity_ >= aligned_size &&
      weight_chunk_buffers_[0] && weight_chunk_buffers_[1]) {
    weight_chunk_buffer_requirement_ = size;
    UpdateActiveBufferIndex(0);
    return;
  }

  bool allocation_failed = false;
  for (int i = 0; i < 2; ++i) {
    void* buffer = nullptr;
    if (posix_memalign(&buffer, alignment, aligned_size) != 0) {
      std::perror("WeightChunkController::AllocWeightChunkBuffer posix_memalign");
      allocation_failed = true;
      break;
    }
    std::memset(buffer, 0, aligned_size);
    weight_chunk_buffers_[i] = buffer;
  }

  if (allocation_failed) {
    ReleaseWeightChunkBuffer();
    weight_chunk_buffer_requirement_ = 0;
    weight_chunk_buffer_capacity_ = 0;
    return;
  }

  weight_chunk_buffer_requirement_ = size;
  weight_chunk_buffer_capacity_ = aligned_size;
  UpdateActiveBufferIndex(0);
}

void WeightChunkController::ReleaseWeightChunkBuffer() {
  for (auto& buffer : weight_chunk_buffers_) {
    if (buffer) {
      std::free(buffer);
      buffer = nullptr;
    }
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
  return weight_chunk_buffers_.at(index);
}

void* WeightChunkController::GetActiveWeightChunkBuffer() const {
  return GetWeightChunkBufferAddr(active_weight_chunk_buffer_index_);
}

void WeightChunkController::RecordChunkAccess(size_t offset) {
  if (!provider_) {
    return;
  }

  if (offset_to_chunk_info.find(offset) != offset_to_chunk_info.end()) {
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
  info.chunk_index = next_chunk_index_++;
  info.aligned_offset = aligned_offset;
  info.offset_adjust = offset_adjust;
  info.aligned_size = aligned_size;
  info.origin_offset = offset;
  info.origin_size = buffer_size;
  info.managed_buffer_index = active_weight_chunk_buffer_index_;
  info.weights_id = weights_id >= 0 ? static_cast<size_t>(weights_id) : 0;

  offset_to_chunk_info.emplace(offset, info);
}

void WeightChunkController::UpdatePreinvokeHandler(ProviderMode mode) {
  switch (mode) {
    case ProviderMode::RUNTIME:
      preinvoke_handler_ = &WeightChunkController::HandleRuntimePreInvoke;
      break;
    case ProviderMode::PRE_RUNTIME:
      preinvoke_handler_ = &WeightChunkController::HandlePreRuntimePreInvoke;
      break;
    default:
      preinvoke_handler_ = &WeightChunkController::HandleDefaultPreInvoke;
      break;
  }
}



bool WeightChunkController::ScheduleNextChunk(const weight_chunk_info_t* current_info, int fd) {
  if (!prefetcher_ || !current_info || prefetch_mode_ == PrefetchMode::UNINITIALIZED) {
    return false;
  }

  const auto next_index =
      prefetcher_->GetNextChunkIndex(prefetch_mode_, current_info->chunk_index);
  if (!next_index.has_value()) {
    return true;  // No next chunk in plan
  }

  const weight_chunk_info_t* next_info = prefetcher_->GetChunkInfoByIndex(*next_index);
  if (!next_info) {
    std::cerr << "[WeightChunkController] Failed to resolve next chunk for index "
              << *next_index << "\n";
    return false;
  }

  const int buffer_index = GetInactiveBufferIndex();
  void* buffer_base = GetWeightChunkBufferAddr(buffer_index);
  if (!buffer_base) {
    std::cerr << "[WeightChunkController] Invalid inactive buffer index " << buffer_index
              << "\n";
    return false;
  }

  WeightChunkPrefetcher::PrefetchRequest request;
  request.chunk_info = next_info;
  request.buffer_base = buffer_base;
  request.direct_io_fd = fd;

  // Submit asynchronously; Prefetcher tracks state via ChunkIOState
  if (!prefetcher_->Submit(request)) {
    std::cerr << "[WeightChunkController] Submit failed for next chunk_index="
              << next_info->chunk_index << "\n";
    return false;
  }

  return true;
}

void WeightChunkController::UpdateWeightsPointer(size_t offset, const weight_chunk_info_t& info) {
  const int mode_idx = WeightChunkPrefetcher::PrefetchModeToIndex(prefetch_mode_);
  if (mode_idx < 0) {
    return;
  }

  auto entry = offset_to_weights_ptr_.find(offset);
  if (entry == offset_to_weights_ptr_.end()) {
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

  auto** slot = reinterpret_cast<uint8_t**>(target_slot);
  *slot = static_cast<uint8_t*>(buffer_base) + info.offset_adjust;
}

bool WeightChunkController::HandlePreRuntimePreInvoke(size_t offset) {
  if (!writer_) {
    std::cerr << "[WeightChunkController] writer_ is null\n";
    return false;
  }
  auto it = offset_to_chunk_info.find(offset);
  if (it == offset_to_chunk_info.end()) {
    RecordChunkAccess(offset);
    it = offset_to_chunk_info.find(offset);
    if (it == offset_to_chunk_info.end()) {
        std::cerr << "[WeightChunkController] Failed to record chunk info for offset "
                  << offset << "\n";
      return false;
    }
  }

  writer_->WriteChunkInfo(it->second, prefetch_mode_);
  return true;
}

bool WeightChunkController::HandleRuntimePreInvoke(size_t offset) {
  // 1. Validate prerequisites
  if (!prefetcher_ || !provider_ || prefetch_mode_ == PrefetchMode::UNINITIALIZED) {
    return false;
  }

  const weight_chunk_info_t* current_chunk_info = prefetcher_->LookupChunkInfo(prefetch_mode_, offset);
  if (!current_chunk_info) {
    return false;
  }

  const int fd = provider_->GetDirectIOFileDescriptor();
  if (fd < 0) {
    return false;
  }

  // 2. Ensure chunk is ready (handles cache hit via ChunkIOState automatically)
  static bool first_call = true;
  //   auto start = std::chrono::steady_clock::now();
  if (first_call) { // Synchronously submit prefetch request and wait only for the first call
    void* buffer_base = GetWeightChunkBufferAddr(active_weight_chunk_buffer_index_);
    if (!buffer_base) {
        std::cerr << "[WeightChunkController] Invalid buffer index " << active_weight_chunk_buffer_index_ << "\n";
        return false;
    }
    WeightChunkPrefetcher::PrefetchRequest request;
    request.chunk_info = current_chunk_info;
    request.buffer_base = buffer_base;
    request.direct_io_fd = fd;
    // Submit handles duplicate detection via ChunkIOState
    if (!prefetcher_->Submit(request)) {
        std::cerr << "[WeightChunkController] Submit failed for chunk_index="
                << current_chunk_info->chunk_index << "\n";
        return false;
    }
  }
  first_call = false;
    
  // WaitReady returns immediately if already ready
  if (!prefetcher_->WaitReady(current_chunk_info)) {
    std::cerr << "[WeightChunkController] WaitReady failed for chunk_index="
            << current_chunk_info->chunk_index << "\n";
    return false;
  }

//   auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
//   printf("[WeightChunkController] Used prefetched chunk_index=%zu for offset=%zu, chunk_size=%zu wait time=%ld us\n",
//            current_chunk_info->chunk_index, offset, current_chunk_info->aligned_size, elapsed.count());

  // 3. Update state and pointer
  UpdateWeightsPointer(offset, *current_chunk_info);
  
//   start = std::chrono::steady_clock::now();
  // 4. Schedule next chunk to inactive buffer asynchronously
  (void)ScheduleNextChunk(current_chunk_info, fd);
//   elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
//    printf("[WeightChunkController] Scheduled next chunk_index asynchronously, time=%ld us\n\n",
//            elapsed.count());

  return true;
}

bool WeightChunkController::HandleDefaultPreInvoke(size_t /*offset*/) {
  return false;
}


//* ================ Hook ================
void WeightChunkController::PreInvokeImpl(size_t offset) {
  (void)(this->*preinvoke_handler_)(offset);
}

void WeightChunkController::PostInvokeImpl(size_t /*offset*/) {
    SwitchActiveBufferIndex();
    // printf("[WeightChunkController] PostInvokeImpl called\n");
}

void WeightChunkController::TraceWeightsAddrImpl(void* addr, size_t offset) {
  if (!addr) {
    std::cerr << "[WeightChunkController] addr is nullptr\n";
    return;
  }
  const int mode_idx = WeightChunkPrefetcher::PrefetchModeToIndex(prefetch_mode_);
  if (mode_idx < 0) {
    return;
  }

  auto& entry = offset_to_weights_ptr_[offset];
  entry[mode_idx] = addr;
}


}  // namespace streaming
}  // namespace flash_slim
