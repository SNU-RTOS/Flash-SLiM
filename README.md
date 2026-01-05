# Flash-SLiM: Streaming LLM inference on Memory-constrained devices using Flash storage

**flash-slim** is a research framework for on-device Large Language Model (LLM) inference. Built with LiteRT (TensorFlow Lite) and Bazel, it is engineered to explore and evaluate advanced OS-aware optimization techniques, specifically targeting memory-constrained environments.

The core research goal of this project is to implement and analyze a **Storage-Aware Weight Streaming Pipeline** with Compute–I/O Overlap, designed to run large decoder-only LLMs on devices with limited DRAM. This framework provides the essential tools to profile, analyze, and optimize the complex interplay between the inference engine and the underlying operating system.

## Architecture Overview

Flash-SLiM employs a **two-stage pipeline** separating preparation and execution:

### Pre-runtime Stage (Offline Preparation)

- **Profiling**: Collects operator-level execution metrics using multi-level profilers
- **CMT Generation**: Creates Weight Chunk Metadata Table mapping operators to weight chunks
- **Prefetch Planning**: Generates optimal chunk loading timeline for I/O–Compute overlap

### Runtime Stage (Online Execution)

- **Weight Streaming**: Asynchronously loads weight chunks from storage using double-buffering
- **Controller Orchestration**: Manages chunk lifecycle and buffer states (READY/FILLING/IN_USE)
- **LiteRT Integration**: Executes LLM inference while coordinating with prefetcher via event hooks

**Key Design Principle**: Pre-runtime generates plans (CMT, Prefetch Plan) → Runtime executes with I/O–Compute overlap

## Core Features

- **OS-Aware Benchmarking**: Deep integration with `cgroups` (v1/v2) and `taskset` for precise control over memory and CPU.
- **Multi-Level Profiling Infrastructure**: A unique, three-tiered profiling system to capture a holistic view of performance:
  1. **TFLite Profiler**: Operator-level wall-clock time.
  2. **eBPF Profiler**: System-level metrics like I/O delay and page faults.
  3. **Custom Scope Profiler**: Stage-level timings for the inference pipeline (e.g., prefill, decode).
- **Standardized Scripting & Logging**: A robust suite of scripts powered by a centralized utility module (`utils.sh`), featuring color-coded console output and clean, color-free log files for easy parsing and analysis.
- **Cross-Platform Compatibility**: Successfully tested on x86_64 and aarch64 architectures (Ubuntu, Debian).

## Current Status & Accomplishments

- **✅ Integrated Profiling System**: The three core profiling tools are fully integrated.
- **✅ Standardized Log Format**: All log outputs, including the multi-level profilers, are standardized to a JSON-based format to facilitate unified post-processing.
- **✅ Robust Scripting Infrastructure**: The project's scripts have been fully refactored for consistency, modularity, and ease of use.
- **✅ Foundational Inference Pipeline**: A stable baseline inference pipeline is in place, serving as the foundation for future optimization work.
- **✅ CMT Generator (Phase 2)**: Chunk Metadata Table generator is fully implemented, replacing the previous prefetch planner.
- **🚧 Weight Streaming Runtime (Phase 3)**: Weight chunk controller and prefetcher infrastructure are under active development.

## Future Development Plan (Roadmap)

The ultimate goal is to build a fully functional **Storage-Aware Weight Streaming Pipeline** that implements I/O–Compute overlap. The development is organized into two distinct stages: **Pre-runtime** (preparation) and **Runtime** (execution).

---

## 🔧 Pre-runtime Stage: Plan Generation

> **Purpose**: Generate **Weight Chunk Metadata Table (CMT)** and **Prefetch Plan** for runtime execution

### Phase 1: Profiling and Analysis (In Progress)

- **Objective**: Collect operator-level execution metrics and analyze performance bottlenecks.
- **Components**:
  - **LiteRT Interpreter**: Executes the LLM and emits `op_start`/`op_end` event hooks
  - **Profiler (eBPF-based)**: Captures operator execution information, latency, and system metrics
  - **Multi-level Profiling**: TFLite ops, eBPF system metrics, custom scope timings
- **Tasks**:
  - Implement a post-processing script (`merge_logs.py`) to sort and merge the JSON log files
  - Define KPIs: Compute–I/O overlap ratio, I/O wait time, page fault rates
  - Develop visualization tools to identify performance bottlenecks

### Phase 2: CMT Generation and Prefetch Planning 🚧 In Progress

- **Objective**: Partition model weights into chunks and generate prefetch scheduling plans.
- **Components**:
  1. **Weight Chunk Metadata Writer**
     - Collects operator-level information from event hooks
     - Generates **CMT (Weight Chunk Metadata Table)** with chunk boundaries and metadata

  2. **Weight Chunk Partitioner**
     - Partitions weights into logical chunks based on CMT
     - Algorithm: Greedy chunking to ensure `T_IO(c_i+1) < T_compute(c_i)` while maintaining `M_peak < M_budget`

  3. **Prefetch Plan Generator**
     - Reads partition results and generates **Prefetch Plan** timeline
     - Optimizes for I/O–Compute overlap based on operator latency predictions

- **Status**:
  - ✅ **CMT Generator** (`cmt_generator` binary): Implemented and produces `weight_chunks_metadata_table.json`
  - ⏳ **Prefetch Plan Generator** (`prefetch_planner` binary): Planned to generate prefetch scheduling timeline

- **Output Files**:
  - `CMT` → Model memory access pattern information
  - `Prefetch Plan` → Timeline plan for runtime prefetcher reference

---

## ⚡ Runtime Stage: Weight Streaming Execution

> **Purpose**: Achieve **I/O–Compute Overlap** during LLM inference using double-buffered weight streaming

### Phase 3: Weight Streaming Runtime 🚧 In Progress

- **Objective**: Execute LLM inference with asynchronous weight prefetching guided by Prefetch Plan.
- **Architectural Components**:

  1. **Buffer on DRAM**
     - **KV Cache**: Fixed pinned buffer
     - **Weight Chunk Buffer A/B**: Double-buffer structure for switching

  2. **LiteRT Interpreter (with XNNPACK Delegate)**
     - Executes actual LLM inference
     - Interacts with Controller/Prefetcher via event hooks at each operation

  3. **Weight Chunk Controller**
     - Orchestrates entire execution flow:
       - Requests next chunk based on Prefetch Plan
       - Manages buffer states (READY/FILLING/IN_USE)
       - Coordinates Prefetcher calls and LiteRT hook synchronization

  4. **Weight Chunk Prefetcher**
     - Loads weight chunks from storage into buffers (handles I/O execution)
     - Pre-loads chunks according to Prefetch Plan
     - Manages asynchronous I/O operations (DirectIO/io_uring)

- **Data Flow**:

  ```text
  Storage → (Prefetcher) → Buffer A/B → (Controller) → LiteRT → XNNPACK Execution
  ```

- **Status**:
  - ✅ **Prefetch Plan Loader**: Implemented in `cmt_generator_util.{h,cc}` to load plan files at runtime
  - ✅ **Weight Chunk Controller**: Framework in place (`weight_chunk_controller.{h,cc}`) for orchestration
  - ⏳ **Weight Chunk Prefetcher**: Stub created (`weight_chunk_prefetcher.{h,cc}`), async I/O implementation in progress

---

### Phase 4: Performance Validation

- **Objective**: Quantitatively evaluate the performance of the new weight streaming runtime against baseline approaches.
- **Tasks**:
  - Benchmark various models (e.g., Gemma-3B, Llama-3.2 3B) under different memory constraints.
  - Measure key metrics: Time to First Token (TTFT), throughput (tokens/s), and peak DRAM usage.
  - Compare results against a baseline implementation that relies on standard OS page fault mechanisms for memory management.

## Project Structure

```text
Flash-SLiM/
├── flash-slim/                    # Core C++ source code for the inference engine
│   ├── text_generator_main.cc     # Main application entry point with multi-level profiling
│   ├── cmt_generator.cc           # Chunk Metadata Table generator (Phase 2)
│   ├── cmt_generator_util.{h,cc}  # JSON-based CMT writer and loader
│   ├── common.{h,cc}              # Shared runtime helpers and initialization functions
│   ├── profiler.{h,cc}            # Custom scope profiler implementation
│   ├── weight_chunk_controller.{h,cc}  # Weight chunk lifecycle management (Phase 3)
│   ├── weight_chunk_prefetcher.{h,cc}  # Async weight prefetcher (Phase 3, in progress)
│   ├── sampler.{h,cc}             # Token sampling strategies (greedy, top-k, top-p)
│   ├── lora_adapter.{h,cc}        # LoRA fine-tuning support
│   ├── utils.{h,cc}               # Utility functions and helpers
│   ├── aligned_allocator.h        # Memory-aligned allocator for cache optimization
│   ├── nlohmann/                  # JSON library for CMT serialization
│   └── BUILD                      # Bazel build definitions
│
├── tools/                         # Analysis and profiling tools
│   ├── model_dump/                # Model structure analysis tools (5.3MB binary)
│   │   ├── dump_llm_nodes.cc      # Detailed model inspection (35KB)
│   │   ├── tensor_visualization.py # Tensor analysis visualization
│   │   └── log/                   # Model analysis output logs
│   ├── ebpf/                      # System-level profiling using eBPF (68KB)
│   │   ├── profile_phase.py       # eBPF-based profiling orchestrator
│   │   └── bpfc_profile_phase.c   # eBPF bytecode for tracing
│   ├── model_prefetch_planner/    # Prefetch planning tools
│   ├── cache/                     # Cache clearing utilities (x86/ARM)
│   ├── benchmark/                 # Performance analysis tools
│   ├── plot/                      # Data visualization scripts
│   ├── power_analysis/            # Power consumption analysis
│   ├── flamegraph/                # Flamegraph generation for performance
│   └── prompt/                    # Prompt processing utilities
│
├── test/                          # PoC and experimental code
│   ├── conceptual_streaming_prototype/  # Weight streaming demo
│   │   ├── partitioner.py         # Greedy chunking algorithm PoC
│   │   ├── cmt.json               # Sample CMT output
│   │   └── chunk_streaming_litert.cpp  # Simulated prefetch runtime
│   ├── direct_io/                 # Direct I/O experiments
│   └── io_uring/                  # Async I/O experiments
│
├── scripts/                       # Build and utility scripts
│   ├── utils.sh                   # Central shell utilities with logging
│   ├── install_prerequisites.sh   # Dependency installation
│   └── build-benchmark_util.sh    # Non-LLM model benchmark build
│
├── build.sh                       # Main build orchestrator (Bazel wrapper)
├── build_cmt_generator.sh         # CMT generator build script
├── run.sh                         # Primary execution script with benchmarking
├── run_cmt_generator.sh           # CMT generator runner
├── models/                        # Model storage (see models/README.md)
├── prompt/                        # JSON prompt files with hyperparameters
├── benchmark/                     # Default output directory for results
│   ├── llm_infer_results/         # LLM inference benchmark results
│   ├── model_analysis_results/    # Model analysis outputs
│   ├── ebpf/                      # eBPF profiling results
│   └── power/                     # Power measurement logs
├── bin/                           # Compiled binaries output directory
├── docs/                          # Project documentation
│   ├── CLAUDE.md                  # Claude Code guidance documentation
│   ├── dev_log.md                 # Development log
│   └── *.md                       # Additional technical documentation
├── external/                      # External dependencies (LiteRT, XNNPACK)
├── bazel/                         # Patches and BUILD files for dependencies
├── .bazelrc                       # Bazel configuration (architecture, features)
├── MODULE.bazel                   # Bazel module dependencies
└── WORKSPACE                      # Bazel workspace definition
```

## Getting Started

### 1. Prerequisites

First, clone the repository and run the script to install all necessary build dependencies. This will install Bazelisk (a Bazel version manager), build tools, and libraries required for the project.

```sh
git clone <repository_url>
cd OS-Support-for-On-Device-LLM
./scripts/install_prerequisites.sh
```

### 2. Environment Setup

Copy the sample environment file. This file defines root paths for the project, models, and prompts. The default values are generally sufficient for most setups.

```sh
cp .env.sample .env
```

### 3. Build the Project

Use the unified build script, which acts as a wrapper around Bazel. Compiled binaries will be placed in the `bin/` directory.

```sh
# Build the main inference binary (text_generator_main)
./build.sh

# Build the CMT generator binary
./build_cmt_generator.sh

# Build all targets
./build.sh all

# Build in debug mode with debugging symbols
./build.sh debug

# Clean build artifacts
./build.sh clean

# Run tests
./build.sh test
```

**Available Build Configurations (via .bazelrc)**:

- `--config=avx_linux` - x86_64 with AVX instructions
- `--config=avx2_linux` - x86_64 with AVX2 instructions
- `--config=linux_arm64` - ARM64 optimizations
- `--config=ebpf` - Enable eBPF tracing support
- `--config=weight_streaming` - Enable weight streaming features (Phase 3)

## Usage

### Running Inference (text_generator_main)

For LLM text generation with the main inference engine:

```sh
# Quick test with default settings
./run.sh

# Run with logging enabled
./run.sh --log
```

### Generating Chunk Metadata Table (cmt_generator)

To generate the CMT for weight streaming (Phase 2):

```sh
# Generate CMT with default settings
./run_cmt_generator.sh

# Output: weight_chunks_metadata_table.json
```

The CMT contains metadata for both PREFILL and DECODE phases, including chunk indices, offsets, sizes, and buffer indices needed for the weight streaming runtime.

### Advanced Benchmarking (`run.sh`)

The `run.sh` script is the primary tool for comprehensive benchmarking. It offers extensive command-line options to control the execution environment.

**Log Path Format**:
When logging is enabled (`--log`), results are saved to a structured directory inside `benchmark/llm_infer_results/`. The path is dynamically generated based on the run configuration:
`[model_name]_[target]_[mem_limit]/`

**Common Commands**:

```sh
# See all available options and their descriptions
./run.sh --help

# Example: Run on GPU, log output, and bind to CPU cores 0-3
./run.sh --target gpu --log --core 0-3

# Example: Run a multi-stage benchmark with different memory limits.
# This will create two separate log directories:
# .../llama_q8_ekv1024_cpu_512M/
# .../llama_q8_ekv1024_cpu_1G/
./run.sh --log --memory 512M --memory 1G
```

## Key Binaries and Tools

### Pre-runtime Tools

1. **cmt_generator** (`bin/cmt_generator`)
   - **Purpose**: Generate Weight Chunk Metadata Table (CMT)
   - **Process**:
     - Loads LiteRT model and executes prefill/decode phases
     - Records weight chunk access patterns via event hooks
     - Produces `weight_chunks_metadata_table.json` with PREFILL/DECODE metadata
   - **Output**: CMT with chunk boundaries, offsets, sizes, and buffer indices

2. **prefetch_planner.py** (`tools/model_prefetch_planner/prefetch_planner.py`) ⏳ Planned
   - **Purpose**: Generate Prefetch Plan timeline
   - **Process**:
     - Reads CMT and operator profiling data
     - Applies greedy partitioning algorithm
     - Optimizes for I/O–Compute overlap scheduling
   - **Output**: Prefetch Plan JSON with chunk loading timeline

### Runtime Binaries

1. **text_generator_main** (`bin/text_generator_main`)
   - **Purpose**: Execute LLM inference with weight streaming
   - **Features**:
     - Multi-level profiling (TFLite ops, eBPF, custom scopes)
     - Token sampling strategies (greedy, top-k, top-p, temperature)
     - LoRA fine-tuning support
     - Weight streaming runtime (Phase 3, in progress)
   - **Output**: Generated text, CSV/JSON profiling data

### Analysis Tools

- **dump_llm_nodes** (`tools/model_dump/dump_llm_nodes`)
  - Model structure inspection and analysis
  - XNNPACK operator counting within delegate nodes
  - Tensor allocation tracking (Mmap, Arena)
  - Execution plan visualization

- **eBPF Profilers** (`tools/ebpf/`)
  - System-level profiling with page fault and I/O tracing
  - Phase-level (prefill/decode) and operator-level tracing
  - Integration with LiteRT event hooks

## Conceptual Prototypes

The `test/conceptual_streaming_prototype/` directory contains a set of files that serve as a Proof-of-Concept for the weight streaming pipeline. These are not integrated into the main application but demonstrate the core logic:

- `merged_profile.json`: A dummy data file simulating the output of the unified logging phase.
- `partitioner.py`: A Python script that implements the greedy chunking algorithm, taking `merged_profile.json` as input and producing `cmt.json`.
- `cmt.json`: A sample Chunk Metadata Table generated by the partitioner.
- `chunk_streaming_litert.cpp`: A C++ application that simulates the prefetch-aware runtime. It loads the `cmt.json` and uses a dummy prefetch thread to log chunk loading and execution events, demonstrating the core double-buffering and swapping logic.

## Recent Changes

- **Namespace Unification**: Extracted shared runtime helpers into [common.h](flash-slim/common.h) and [common.cc](flash-slim/common.cc), unifying namespaces across text generation and prefetch modules.

- **CMT Generator (Pre-runtime)**: Implemented `cmt_generator` binary with:
  - `JsonWeightChunkMetaDataWriter` for CMT generation
  - Event hook integration with LiteRT interpreter
  - Per-mode (PREFILL/DECODE) metadata organization

- **Weight Streaming Runtime Components (Phase 3)**:
  - Added [weight_chunk_controller.{h,cc}](flash-slim/weight_chunk_controller.h) for orchestrating chunk lifecycle and buffer state management
  - Created [weight_chunk_prefetcher.{h,cc}](flash-slim/weight_chunk_prefetcher.h) stub for async I/O implementation
  - Implemented `JsonPrefetchPlanLoader` in [cmt_generator_util.h](flash-slim/cmt_generator_util.h) for loading prefetch plans

- **Removed KV Cache Manager**: Migrated functionality from `kv_cache_manager.{h,cc}` to common utilities:
  - `AllocateKVCache()` - KV cache buffer allocation
  - `GetPrefillRunner()` - Prefill phase runner setup
  - `GetDecodeRunner()` - Decode phase runner setup

- **Build System Updates**:
  - Added `weight_chunk_controller` library to BUILD file
  - Added `build_cmt_generator.sh` and `run_cmt_generator.sh` scripts
