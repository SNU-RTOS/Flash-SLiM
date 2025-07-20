# flash-slim: A Research Framework for On-Device LLM Inference with OS-Aware Optimization

**flash-slim** is a high-performance research framework for on-device Large Language Model (LLM) inference. Built with LiteRT (TensorFlow Lite) and Bazel, it is engineered to explore and evaluate advanced OS-aware optimization techniques, specifically targeting memory-constrained environments.

The core research goal of this project is to implement and analyze a **Weight Streaming with Compute–I/O Overlap** pipeline, designed to run large decoder-only LLMs on devices with limited DRAM. This framework provides the essential tools to profile, analyze, and optimize the complex interplay between the inference engine and the underlying operating system.

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

## Future Development Plan (Roadmap)

The ultimate goal is to build a fully functional, prefetch-aware runtime that implements weight streaming. The development is planned in the following phases:

### Phase 1: Log Unification and Analysis (In Progress)

- **Objective**: Merge the disparate logs from the three profilers into a single, unified timeline based on high-precision timestamps.
- **Tasks**:
  - Implement a post-processing script (`merge_logs.py`) to sort and merge the JSON log files.
  - Define and quantify key performance indicators (KPIs) such as Compute–I/O overlap ratio, I/O wait time, and page fault rates per inference stage.
  - Develop visualization tools (e.g., using `matplotlib` or `plotly`) to chart the unified timeline and clearly identify performance bottlenecks.

### Phase 2: Weight Chunk Partitioner (Pre-runtime)

- **Objective**: Develop an offline tool that partitions a static LLM into dynamically loadable weight chunks based on profiling data from Phase 1.
- **Algorithm**: A greedy algorithm will group operators into chunks. The partitioning heuristic aims to ensure that the I/O time for the next chunk is hidden by the compute time of the current chunk (`T_IO(c_i+1) < T_compute(c_i)`), without exceeding the device's specified memory budget (`M_peak < M_budget`).
- **Output**: The tool will generate a **Chunk Metadata Table (CMT)** in JSON format. This table will contain the operator range, size, and estimated latencies for each chunk, serving as a blueprint for the runtime.

### Phase 3: Prefetch-Aware Runtime

- **Objective**: Modify the LiteRT interpreter to support asynchronous weight prefetching and compute-I/O overlap, guided by the CMT.
- **Architectural Components**:
  - **CMT Parser**: A C++ module to load the CMT and create an in-memory map from operator indices to chunk IDs.
  - **Chunk Execution Module**: A component integrated into the LiteRT operator execution loop. It will track the current operator index, detect chunk boundaries, and trigger prefetch events for subsequent chunks.
  - **Memory Prefetch Module**: A background thread managing a double-buffer system. It will be responsible for asynchronously loading the next chunk's weights from flash storage into the inactive buffer.

### Phase 4: Performance Validation

- **Objective**: Quantitatively evaluate the performance of the new weight streaming runtime against baseline approaches.
- **Tasks**:
  - Benchmark various models (e.g., Gemma-3B, Llama-3.2 3B) under different memory constraints.
  - Measure key metrics: Time to First Token (TTFT), throughput (tokens/s), and peak DRAM usage.
  - Compare results against a baseline implementation that relies on standard OS page fault mechanisms for memory management.

## Project Structure

```text
├── flash-slim/             # Core C++ source code for the inference engine.
│   ├── text_generator_main.cc # Main application logic with multi-level profiling.
│   ├── profiler.{h,cc}        # Custom scope profiler implementation.
│   └── ...
├── scripts/                  # All shell scripts for building, running, and analysis.
│   ├── build.sh              # Unified build script for all C++ targets.
│   ├── run.sh                # Comprehensive script for multi-run benchmarks.
│   ├── run_once.sh           # A simple script for a single, quick test run.
│   ├── run_benchmark_util.sh # Script for benchmarking non-LLM TFLite models.
│   ├── install_prerequisites.sh # Installs all necessary build dependencies.
│   └── utils.sh                # Centralized shell utility and helper functions.
├── models/                   # Model storage. See models/README.md for the required structure.
├── benchmark/                # Default output directory for benchmark logs and results.
├── util/                     # Source code for host-side C++ utilities (e.g., cache clearing tool).
├── test/                     # Test files, PoC scripts, and non-LLM models.
│   └── conceptual_streaming_prototype/ # Contains PoC code for the roadmap.
├── bazel/                    # Patches and BUILD files for third-party dependencies.
├── .bazelrc                  # Bazel run commands configuration.
├── MODULE.bazel              # Bazel module file defining project dependencies.
└── WORKSPACE                 # Bazel workspace definition.
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

Use the unified build script, which acts as a wrapper around Bazel. The main binary, `text_generator_main`, will be placed in the `output/` directory (which is git-ignored).

```sh
# Build the main binary in release mode (optimized)
./build.sh

# Build in debug mode with debugging symbols
./build.sh debug
```

## Usage

### Quick Test Run (`run_once.sh`)

For a simple, default test run to verify the build and basic functionality. This script is not configurable via CLI arguments and always logs its output to `result_run_once/output.log`.

```sh
./run_once.sh
```

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

## Conceptual Prototypes

The `test/conceptual_streaming_prototype/` directory contains a set of files that serve as a Proof-of-Concept for the future development plan. These are not integrated into the main application but demonstrate the core logic:

- `merged_profile.json`: A dummy data file simulating the output of the unified logging phase.
- `partitioner.py`: A Python script that implements the greedy chunking algorithm, taking `merged_profile.json` as input and producing `cmt.json`.
- `cmt.json`: A sample Chunk Metadata Table generated by the partitioner.
- `chunk_streaming_litert.cpp`: A C++ application that simulates the prefetch-aware runtime. It loads the `cmt.json` and uses a dummy prefetch thread to log chunk loading and execution events, demonstrating the core double-buffering and swapping logic.
