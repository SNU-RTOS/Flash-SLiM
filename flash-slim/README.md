# flash-slim/ Source Code

This directory contains the core C++ source code for Flash-SLiM's LLM inference engine and weight streaming infrastructure. The codebase is organized into **Pre-runtime** tools and **Runtime** components.

## Architecture Overview

Flash-SLiM implements a **Storage-Aware Weight Streaming Pipeline** with two distinct stages:

- **Pre-runtime**: Offline tools for profiling and generating weight chunk metadata
- **Runtime**: Online execution with asynchronous weight prefetching and I/O–Compute overlap

## Source Files

### Pre-runtime Binaries

#### cmt_generator.cc (39KB)

**Purpose**: Generate Chunk Metadata Table (CMT) for weight streaming

**Key Components**:

- Loads TFLite model and builds interpreter
- Sets up `StreamingWeightCacheProvider` in PRE_RUNTIME mode
- Initializes `JsonWeightChunkMetaDataWriter` for metadata recording
- Applies XNNPACK delegate with weight caching
- Executes prefill and decode phases to record chunk access patterns
- Outputs `weight_chunks_metadata_table.json`

**Main Function**: `main()` (lines 632-901)

**Output Format**:

```json
{
  "metadata": {
    "version": "1.0",
    "model": "path/to/model.tflite",
    "chunk_count_by_mode": {"PREFILL": n, "DECODE": m}
  },
  "weight_chunks": {
    "PREFILL": [
      {
        "chunk_index": 0,
        "aligned_offset": 1024,
        "aligned_size": 65536,
        "weights_id": 123,
        "managed_buffer_index": 0
      }
    ],
    "DECODE": [...]
  }
}
```

**Build**: `./build_cmt_generator.sh`

**Run**: `./run_cmt_generator.sh`

#### cmt_generator_util.{h,cc} (14KB)

**Purpose**: CMT serialization and loading utilities

**Classes**:

1. **JsonWeightChunkMetaDataWriter**
   - Implements `WeightChunkMetaDataWriter` interface
   - Writes weight chunk info to JSON with per-mode organization (PREFILL/DECODE)
   - Tracks: chunk_index, aligned_offset, aligned_size, origin_offset, weights_id, etc.
   - Methods:
     - `WriteMetadata()` - Write final JSON to file
     - `OnChunkAccess()` - Record chunk access event

2. **JsonPrefetchPlanLoader**
   - Loads CMT JSON files for runtime use
   - Provides mode-based chunk access (prefill_chunks, decode_chunks)
   - Methods:
     - `LoadFromFile()` - Parse JSON and populate internal structures
     - `BuildOffsetToIndexForMode()` - Create offset→index mapping
     - `PrintMetadata()` - Diagnostic output

### Runtime Binaries

#### text_generator_main.cc (20KB)

**Purpose**: Primary LLM inference application with multi-level profiling

**Features**:

- Complete inference pipeline (prefill + decode)
- Multi-level profiling:
  - TFLite operator-level profiling
  - eBPF system-level tracing (when built with `--config=ebpf`)
  - Custom scope profiler for phase timing
- Token sampling strategies: greedy, top-k, top-p, temperature
- LoRA fine-tuning support
- Optional repetition penalty
- Weight streaming runtime (Phase 3, in progress)

**Command-line Flags**:

```bash
--tflite_model              # Path to TFLite model
--sentencepiece_model       # Path to SentencePiece tokenizer
--prompt                    # Input prompt text
--max_decode_steps          # Number of tokens to generate
--num_threads               # Thread pool size (default: 4)
--weight_cache_path         # XNNPACK weight cache directory
--temperature               # Sampling temperature (default: 0.8)
--top_k                     # Top-k sampling (default: 40)
--top_p                     # Top-p nucleus sampling (default: 0.9)
--csv_profile_output_path   # CSV profiling output file
```

**Entry Point**: `main()` (lines 33-100+)

- Parses flags using `absl::FLAGS`
- Initializes `GenAIMetrics` profiler
- Calls `__run_main()` for actual inference
- Outputs profiling results in log and CSV formats

**Build**: `./build.sh`

**Run**: `./run.sh`

### Runtime Components (Phase 3)

#### weight_chunk_controller.{h,cc} (1KB)

**Purpose**: Orchestrate weight chunk lifecycle and buffer management

**Key Responsibilities**:

- Manage weight chunk lifecycle
- Interface with `StreamingWeightCacheProvider`
- Implement buffer switching logic (active_buffer_index toggles between 0 and 1)
- Coordinate with prefetcher for async I/O

**Methods**:

- `LoadPrefetchPlan()` - Load prefetch scheduling plan from JSON
- `AttachPrefetcher()` - Attach weight chunk prefetcher
- `AttachMetadataWriter()` - Attach metadata writer (pre-runtime mode)
- `PreInvoke()` / `PostInvoke()` - Hook points for operator execution
- `SwitchActiveBuffer()` / `ResetBuffers()` - Buffer management
- `DumpStatus()` - Status logging

**Buffer States**: READY, FILLING, IN_USE

#### weight_chunk_prefetcher.{h,cc} (stub)

**Purpose**: Asynchronous weight chunk loading with double-buffering

**Planned Features**:

- Background thread managing double-buffer weight loading
- Async I/O operations (DirectIO/io_uring)
- Prefetch scheduling based on Prefetch Plan
- Coordination with WeightChunkController

**Status**: ⏳ Implementation in progress

### Shared Runtime Components

#### common.{h,cc} (11KB)

**Purpose**: Shared runtime helpers and initialization functions

**Key Functions**:

- `AllocateKVCache()` - KV cache buffer allocation
- `GetPrefillRunner()` - Prefill phase runner setup
- `GetDecodeRunner()` - Decode phase runner setup
- Unified namespace across text generation and prefetch modules

**Replaces**: `kv_cache_manager.{h,cc}` (removed)

#### profiler.{h,cc} (14KB)

**Purpose**: Custom scope profiler for stage-level timing

**Classes**:

- `GenAIMetrics` - Main profiling metrics collector
- `TimerUtility` - High-precision timer
- `ScopeTimer` - RAII-based scope timing
- `ScopeEventPrefetcher` - Event-based prefetch profiling

**Integration**:

- Works alongside TFLite profiler and eBPF profiler
- Outputs JSON-based logs for unified timeline analysis

#### sampler.{h,cc} (9KB)

**Purpose**: Token sampling strategies

**Functions**:

- `GreedySampling()` - Argmax over logits
- `TopKSampling()` - Top-k sampling
- `TopPSampling()` - Nucleus sampling
- `TemperatureSampling()` - Temperature-based sampling

#### utils.{h,cc} (7KB)

**Purpose**: Utility functions and helpers

- LiteRT model loading
- Tensor manipulation
- String utilities
- File I/O helpers

#### lora_adapter.{h,cc} (3KB)

**Purpose**: LoRA fine-tuning support

- Applies LoRA adapters to model weights
- Supports low-rank adaptation for efficient fine-tuning

#### aligned_allocator.h

**Purpose**: Memory-aligned allocator for cache optimization

- Ensures proper memory alignment for SIMD operations
- Optimizes buffer performance for weight streaming

### Third-party Dependencies

#### nlohmann/

**Purpose**: JSON serialization library

- Used for CMT and Prefetch Plan file I/O
- Header-only library embedded in source tree

## Build System

### BUILD File

Defines Bazel build targets:

**Libraries**:

- `utils` - LiteRT utilities, Abseil containers
- `sampler` - Token sampling, LiteRT framework
- `profiler` - Custom profiling, XNNPACK delegate
- `aligned_allocator` - Memory alignment
- `lora_adapter` - LoRA support, GenAI ops
- `cmt_generator_util` - JSON handling, nlohmann/json
- `weight_chunk_controller` - Streaming weight cache provider
- `common` - Comprehensive dependency aggregator

**Binaries**:

- `text_generator_main` - Primary inference application
- `cmt_generator` - Chunk metadata generation tool

### Build Configuration (.bazelrc)

```bash
# Architecture-specific
--config=avx_linux          # x86_64 with AVX instructions
--config=linux_arm64        # ARM64 optimizations
--config=avx2_linux         # AVX2 for modern CPUs

# Feature toggles
--config=ebpf               # eBPF tracing support (-DEBPF_TRACE_ENABLED)
--config=weight_streaming   # Weight streaming (-DUSE_WEIGHT_STREAMING)

# Language/compiler settings
--cxxopt=-std=c++17         # C++17 standard
--copt=-O3                  # Optimization level 3
```

## Development Workflow

### Pre-runtime Development

1. **CMT Generation**:
   ```bash
   ./build_cmt_generator.sh
   ./run_cmt_generator.sh
   # Output: weight_chunks_metadata_table.json
   ```

2. **Prefetch Planning** (planned):
   ```bash
   python tools/model_prefetch_planner/prefetch_planner.py \
     --cmt=weight_chunks_metadata_table.json \
     --output=prefetch_plan.json
   ```

### Runtime Development

1. **Build with weight streaming**:
   ```bash
   ./build.sh --config=weight_streaming
   ```

2. **Run inference**:
   ```bash
   ./run.sh --log
   ```

3. **Profiling**:
   ```bash
   # With eBPF profiling
   ./build.sh --config=ebpf
   ./run.sh --log
   ```

## Key Design Patterns

### Event Hook Integration

The weight streaming system integrates with LiteRT via event hooks:

- `PreInvoke()` - Called before operator execution
  - Check if next chunk needs loading
  - Trigger prefetcher if necessary

- `PostInvoke()` - Called after operator execution
  - Record chunk access metadata
  - Update buffer states

### Double-Buffering

Weight chunks use a double-buffer pattern:

- **Buffer A/B**: Two weight chunk buffers
- **active_buffer_index**: Toggles between 0 and 1
- **States**: READY (available), FILLING (loading), IN_USE (active)

**Flow**:

```text
1. Use Buffer A (active_buffer_index=0)
2. Prefetch into Buffer B (inactive)
3. Switch to Buffer B (active_buffer_index=1)
4. Prefetch into Buffer A (now inactive)
5. Repeat
```

### Modular Design

- **Separation of Concerns**: Pre-runtime (CMT generation) vs Runtime (execution)
- **Interface-Based**: `WeightChunkMetaDataWriter`, `StreamingWeightCacheProvider`
- **Dependency Injection**: Controller/Prefetcher attachment via methods

## Testing and Validation

### Unit Tests

Located in test files (to be added):

- `cmt_generator_test.cc`
- `weight_chunk_controller_test.cc`
- `profiler_test.cc`

### Integration Tests

Validation logs generated during execution:

- `weight_cache_structure.log` - Weight cache structure dump
- `weight_cache_tensor_id_map.log` - Tensor ID mapping
- `weight_cache_validation.log` - Validation results

## Related Documentation

- Main README: [../README.md](../README.md)
- Tools directory: [../tools/README.md](../tools/README.md)
- Development log: [../docs/dev_log.md](../docs/dev_log.md)
- Claude Code guide: [../docs/CLAUDE.md](../docs/CLAUDE.md)

## Contributing

When adding new files or modifying existing ones:

1. Follow C++17 standards
2. Use consistent naming conventions (snake_case for files/functions)
3. Add appropriate comments and documentation
4. Update BUILD file with new targets
5. Ensure compatibility with both x86_64 and ARM64
6. Test with multiple build configurations (ebpf, weight_streaming)

## Recent Changes

- **Namespace Unification**: Extracted shared helpers into common.{h,cc}
- **CMT Generator**: Fully implemented with JSON-based metadata output
- **Weight Streaming Infrastructure**: Added controller and prefetcher stubs
- **Removed KV Cache Manager**: Migrated to common utilities
- **Build System**: Added weight_chunk_controller library target
