#include "flash-slim/weight_chunk_controller.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <vector>
#include <unistd.h>

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

bool ExecuteIO(int fd, void* buffer, size_t size, off_t offset);

bool ExecuteIO(int fd, void* buffer, size_t size, off_t offset) {
  constexpr size_t kMinBlockSize = 512 * 1024;
  constexpr size_t kMaxThreads = 4;

  if (fd < 0 || buffer == nullptr || size == 0) {
    return false;
  }

  if (size < kMinBlockSize * 2) {
    const ssize_t bytes_read = pread(fd, buffer, size, offset);
    return bytes_read > 0 && static_cast<size_t>(bytes_read) == size;
  }

  const size_t num_threads = std::min(kMaxThreads, size / kMinBlockSize);
  const size_t block_size = size / num_threads;
  const size_t remainder = size % num_threads;

  std::vector<std::future<bool>> futures;
  futures.reserve(num_threads);

  for (size_t i = 0; i < num_threads; ++i) {
    const size_t current_block_size = block_size + (i < remainder ? 1 : 0);
    const size_t block_offset = i * block_size + std::min(i, remainder);

    uint8_t* block_buffer = static_cast<uint8_t*>(buffer) + block_offset;
    const off_t block_file_offset = offset + block_offset;

    futures.emplace_back(std::async(std::launch::async, [=]() -> bool {
      const ssize_t bytes_read = pread(fd, block_buffer, current_block_size, block_file_offset);
      return bytes_read > 0 && static_cast<size_t>(bytes_read) == current_block_size;
    }));
  }

  bool all_success = true;
  for (auto& future : futures) {
    if (!future.get()) {
      all_success = false;
    }
  }

  return all_success;
}

}  // namespace

namespace flash_slim {
namespace streaming {

WeightChunkController::WeightChunkController(
    tflite::xnnpack::StreamingWeightCacheProvider* provider)
    : provider_(provider) {}

WeightChunkController::~WeightChunkController() {
  if (provider_) {
    provider_->ClearWeightChunkBuffers();
  }
  ReleaseWeightChunkBuffers();
}

bool WeightChunkController::LoadPrefetchPlan(const std::string& path) {
  if (!prefetcher_) {
    std::cerr << "[WeightChunkController] Prefetcher not attached\n";
    return false;
  }

  flash_slim::JsonPrefetchPlanLoader loader;
  if (!loader.LoadFromFile(path)) {
    return false;
  }

  weight_chunk_buffer_requirement_ = loader.max_aligned_size();

  std::cout << "[INFO] Allocating weight cache buffer of size "
            << weight_chunk_buffer_requirement_ << " bytes.\n";
  std::cout << "[INFO] Loaded prefetch plan for model: " << loader.model()
            << ", planner version: " << loader.version() << "\n";

  if (provider_) {
    EnsureWeightChunkBuffer(weight_chunk_buffer_requirement_);
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
  chunk_info_cache_.clear();
  next_chunk_index_ = 0;

  return true;
}

void WeightChunkController::AttachPrefetcher(
    std::shared_ptr<flash_slim::streaming::WeightChunkPrefetcher> prefetcher) {
  prefetcher_ = std::move(prefetcher);
}

void WeightChunkController::AttachMetadataWriter(
    tflite::xnnpack::WeightChunkMetaDataWriter* writer) {
  writer_ = writer;
}

void WeightChunkController::SetMode(
    tflite::xnnpack::StreamingWeightCacheProvider::ProviderMode mode) {
  provider_mode_ = mode;
  if (provider_) {
    provider_->SetProviderMode(mode);
  }
}

void WeightChunkController::UpdatePrefetcherMode(PrefetchMode mode) {
  prefetch_mode_ = mode;
  if (prefetcher_) {
    prefetcher_->UpdatePrefetcherMode(mode);
  }
}

void WeightChunkController::EnsureWeightChunkBuffer(size_t size) {
  if (!provider_ || size == 0) {
    return;
  }

  const size_t alignment = provider_->GetWeightChunkBufferSectorSize();
  const size_t aligned_size = AlignTo(size, alignment);

  if (weight_chunk_buffer_capacity_ >= aligned_size &&
      weight_chunk_buffers_[0] && weight_chunk_buffers_[1]) {
    weight_chunk_buffer_requirement_ = size;
    provider_->SetWeightChunkBuffer(0, weight_chunk_buffers_[0], aligned_size);
    provider_->SetWeightChunkBuffer(1, weight_chunk_buffers_[1], aligned_size);
    ResetBuffers();
    return;
  }

  provider_->ClearWeightChunkBuffers();
  ReleaseWeightChunkBuffers();

  bool allocation_failed = false;
  for (int i = 0; i < 2; ++i) {
    void* buffer = nullptr;
    if (posix_memalign(&buffer, alignment, aligned_size) != 0) {
      std::perror("WeightChunkController::EnsureWeightChunkBuffer posix_memalign");
      allocation_failed = true;
      break;
    }
    std::memset(buffer, 0, aligned_size);
    weight_chunk_buffers_[i] = buffer;
    provider_->SetWeightChunkBuffer(i, weight_chunk_buffers_[i], aligned_size);
  }

  if (allocation_failed) {
    ReleaseWeightChunkBuffers();
    provider_->ClearWeightChunkBuffers();
    weight_chunk_buffer_requirement_ = 0;
    weight_chunk_buffer_capacity_ = 0;
    return;
  }

  weight_chunk_buffer_requirement_ = size;
  weight_chunk_buffer_capacity_ = aligned_size;
  ResetBuffers();
}

void WeightChunkController::ReleaseWeightChunkBuffers() {
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

void WeightChunkController::PreInvoke(size_t offset) {
  if (!provider_) {
    return;
  }

  if (provider_mode_ == ProviderMode::RUNTIME) {
    HandleRuntimePreInvoke(offset);
  } else if (provider_mode_ == ProviderMode::PRE_RUNTIME) {
    HandlePreRuntimePreInvoke(offset);
  }
}

void WeightChunkController::PostInvoke(size_t /*offset*/) { SwitchActiveBuffer(); }

void WeightChunkController::SwitchActiveBuffer() {
  active_weight_chunk_buffer_index_ = 1 - active_weight_chunk_buffer_index_;
  if (provider_) {
    provider_->SwitchActiveWeightChunkBuffer();
  }
}

void WeightChunkController::ResetBuffers() {
  active_weight_chunk_buffer_index_ = 0;
  if (provider_) {
    provider_->ResetActiveWeightChunkBuffer();
  }
}

void WeightChunkController::DumpStatus() const {
  std::cout << "[WeightChunkController] buffer_index=" << active_weight_chunk_buffer_index_
            << ", cached_chunks=" << chunk_info_cache_.size() << std::endl;
}

void WeightChunkController::TraceWeightsAddr(void* addr, size_t offset) {
  if (!addr) {
    std::printf("TraceWeightsAddr: nullptr\n");
    return;
  }
  const int mode_idx = WeightChunkPrefetcher::PrefetchModeToIndex(prefetch_mode_);
  if (mode_idx < 0) {
    return;
  }

  auto& entry = offset_to_weights_ptr_[offset];
  entry[mode_idx] = addr;
}

void WeightChunkController::RecordChunkAccess(size_t offset) {
  if (!provider_) {
    return;
  }

  if (chunk_info_cache_.find(offset) != chunk_info_cache_.end()) {
    return;
  }

  const size_t buffer_size = provider_->GetBufferSize(offset);
  const int weights_id = provider_->GetWeightsId(offset);
  const size_t sector_size = provider_->GetWeightChunkBufferSectorSize();
  const size_t abs_offset = provider_->GetMMapBaseOffset() + offset;

  const size_t aligned_offset = (abs_offset / sector_size) * sector_size;
  const size_t offset_adjust = abs_offset - aligned_offset;
  const size_t aligned_size = AlignTo(buffer_size + offset_adjust, sector_size);

  ChunkInfo info;
  info.chunk_index = next_chunk_index_++;
  info.aligned_offset = aligned_offset;
  info.offset_adjust = offset_adjust;
  info.aligned_size = aligned_size;
  info.origin_offset = offset;
  info.origin_size = buffer_size;
  info.managed_buffer_index = active_weight_chunk_buffer_index_;
  info.weights_id = weights_id >= 0 ? static_cast<size_t>(weights_id) : 0;

  chunk_info_cache_.emplace(offset, info);
}

bool WeightChunkController::HandleRuntimePreInvoke(size_t offset) {
  if (!prefetcher_ || prefetch_mode_ == PrefetchMode::UNINITIALIZED) {
    return false;
  }

  const ChunkInfo* info = ResolveChunkInfo(offset);
  if (!info) {
    return false;
  }

  UpdateWeightsPointer(offset, *info);
  return LoadChunkData(*info);
}

void WeightChunkController::HandlePreRuntimePreInvoke(size_t offset) {
  if (!writer_) {
    return;
  }

  auto it = chunk_info_cache_.find(offset);
  if (it == chunk_info_cache_.end()) {
    RecordChunkAccess(offset);
    it = chunk_info_cache_.find(offset);
    if (it == chunk_info_cache_.end()) {
      return;
    }
  }

  writer_->WriteChunkInfo(it->second, prefetch_mode_);
}

const WeightChunkController::ChunkInfo* WeightChunkController::ResolveChunkInfo(size_t offset) const {
  if (!prefetcher_) {
    return nullptr;
  }

  const int mode_idx = WeightChunkPrefetcher::PrefetchModeToIndex(prefetch_mode_);
  if (mode_idx < 0) {
    return nullptr;
  }

  const auto* plan = prefetcher_ ? prefetcher_->GetPlan(prefetch_mode_) : nullptr;
  if (!plan) {
    return nullptr;
  }

  const auto it = plan->offset_to_index.find(offset);
  if (it == plan->offset_to_index.end()) {
    return nullptr;
  }

  const auto& index_to_chunks = prefetcher_->GetIndexToChunks();
  const size_t index = it->second;
  if (index >= index_to_chunks.size()) {
    return nullptr;
  }

  const auto& candidate = index_to_chunks[index];
  if (candidate.chunk_index == std::numeric_limits<size_t>::max()) {
    return nullptr;
  }
  return &candidate;
}

void WeightChunkController::UpdateWeightsPointer(size_t offset, const ChunkInfo& info) {
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

  void* buffer_base = provider_->GetWeightChunkBuffer(active_weight_chunk_buffer_index_);
  if (!buffer_base) {
    return;
  }

  auto** slot = reinterpret_cast<uint8_t**>(target_slot);
  *slot = static_cast<uint8_t*>(buffer_base) + info.offset_adjust;
}

bool WeightChunkController::LoadChunkData(const ChunkInfo& info) {
  void* buffer_base = provider_->GetWeightChunkBuffer(active_weight_chunk_buffer_index_);
  if (!buffer_base) {
    return false;
  }

  const int fd = provider_->GetDirectIOFileDescriptor();
  if (fd < 0) {
    return false;
  }

  return ExecuteIO(fd, buffer_base, info.aligned_size, info.aligned_offset);
}

}  // namespace streaming
}  // namespace flash_slim
