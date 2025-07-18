# flash-slim: On-Device LLM Inference with OS Support

A high-performance, on-device Large Language Model (LLM) inference system built with LiteRT (TensorFlow Lite) and Bazel. This project, now named **flash-slim**, provides a complete framework for running, profiling, and evaluating LLMs on resource-constrained devices. It is designed to bridge the gap between high-level model research and low-level, OS-aware performance tuning.

## Overview

This project provides a complete framework for running LLMs on-device with:

- **Modern Build System**: Powered by Bazel for fast, reproducible, and hermetic builds.
- **OS-Aware Performance Tuning**: Deep integration with OS features like cgroups and core pinning for realistic benchmarking.
- **Cross-platform compatibility**: Tested on x86_64 and aarch64 architectures.
- **Advanced Benchmarking**: Comprehensive utilities for evaluating performance under various constraints.
- **Memory Optimization**: XNNPACK weight caching and fine-grained memory management.
- **Flexible Deployment**: Support for various LLM architectures (Gemma, Llama, etc.).

## Design Philosophy

The core goal of **flash-slim** is to provide a stable, high-performance environment for on-device LLM research with a strong emphasis on OS-level resource management.

- **OS-Aware Performance Tuning**: We believe that predictable performance on-device requires deep integration with the operating system. The project provides built-in tools for memory and CPU management (`cgroups`, `taskset`) to accurately simulate and benchmark performance under realistic constraints.
- **Reproducibility and Reliability**: The entire project is built with Bazel to ensure that builds are hermetic and reproducible. This is critical for academic research and production deployments where consistency is key.
- **Modularity and Focus**: By encapsulating the core logic within the `flash-slim` module, we aim to provide a clean, focused, and extensible codebase for LLM inference, separate from the complexities of dependency management and build orchestration.

## Platform Support

Successfully tested on the following platforms:

- **x86_64**: Ubuntu 22.04, Ubuntu 24.04
- **aarch64**: Ubuntu 20.04, Ubuntu 22.04, Ubuntu 24.04, Debian 13

**Note**: GPU acceleration is available but may have model compatibility limitations.

## Project Structure

```text
├── flash-slim/             # Core source code for the inference engine
│   ├── text_generator_main.cc # Main application logic
│   ├── profiler.{h,cc}        # Performance profiling utilities
│   ├── sampler.{h,cc}         # Text generation sampling strategies
│   └── utils.{h,cc}           # Common utilities
├── scripts/                  # Build, run, and utility scripts
│   ├── build.sh              # Main build script
│   ├── run.sh                # Comprehensive run and benchmark script
│   ├── run_once.sh           # Quick test run script
│   ├── common.sh             # Shared build configurations
│   ├── build-benchmark_util.sh # Builds the TFLite benchmark utility
│   ├── build-deps.sh           # Installs build dependencies
│   ├── build-litert.sh         # Builds the main LiteRT binary
│   ├── build-litert_gpu_delegate.sh # Builds LiteRT with GPU delegate support
│   ├── parse_json_prompt.py    # Parses JSON prompt files for the main run script
│   ├── utils.sh                # Shared utility and build helper functions
│   └── ...
├── test/                     # Test files and scripts
│   ├── README.md
├── benchmark/                # Benchmarking results and analysis tools
├── models/                   # Model storage directory
├── output/                   # Build output directory (binaries)
├── bazel/                    # Patches and BUILD files for third-party deps
├── .bazelrc                  # Bazel run commands configuration
├── .bazelversion             # Specifies the official Bazel version for the project
├── MODULE.bazel              # Bazel module file
└── WORKSPACE                 # Bazel workspace definition
```

## Getting Started

### 1. Prerequisites

#### Install Bazelisk

This project uses Bazel for builds. We recommend installing Bazelisk, which automatically manages Bazel versions.

**Ubuntu/Linux:**

```sh
curl -L https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -o /usr/local/bin/bazelisk
chmod +x /usr/local/bin/bazelisk
# Create a symlink to use 'bazel' command
sudo ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel
```

#### Configure Environment

Copy the sample environment file and customize it if necessary.

```sh
cp .env.sample .env
# Edit .env to set project paths (defaults are usually fine)
```

### 2. Build

The project is built using a simple shell script that wraps Bazel commands.

```sh
# Build the main binary in release mode (default)
./build.sh

# Build in debug mode
./build.sh debug

# Build all targets, including tests
./build.sh all

# Clean all build artifacts
./build.sh clean
```

The main binary `text_generator_main` will be placed in the `output/` directory.

### 3. Run Inference

#### Quick Test

For a simple test run with default settings:

```sh
./run_once.sh
```

#### Advanced Benchmarking

The `run.sh` script provides extensive options for benchmarking, including core pinning, thread management, and memory constraints.

```sh
# See all available options
./run.sh --help

# Example: Run on GPU, log output, bind to cores 0-3, use 4 threads
./run.sh --target gpu --log --core 0-3 --threads 4

# Example: Run a benchmark with specific memory limits (512MB and 1GB)
./run.sh --memory 512M --memory 1G
```

## Features

### Performance Profiling & Benchmarking

- **cgroup Integration**: Run benchmarks under specific memory constraints (cgroup v1 and v2 supported) to simulate real-world device limitations.
- **Core Pinning**: Isolate and assign inference workloads to specific CPU cores using `taskset` for stable and predictable performance measurement.
- **Detailed Logging**: Capture comprehensive performance metrics, model outputs, and system information for in-depth analysis.

### Optimization Features

- **XNNPACK Integration**: Leverage hardware-specific optimizations for high-performance CPU inference via the XNNPACK delegate.
- **Weight Caching**: Reduce memory footprint and accelerate model loading times by caching model weights.
- **Multi-threading**: Utilize a configurable thread pool to maximize performance on multi-core processors.

### Supported Models

- Gemma (2B, 7B variants)
- Llama 3.2 (3B variants)
- Custom models converted with ai-edge-torch, or from Hugging Face: [https://huggingface.co/litert-community/](https://huggingface.co/litert-community/)

## Build Options

The build system supports multiple configurations via `scripts/utils.sh`:

- **Release (default)**: Optimized for performance (`-c opt`).
- **Debug**: Includes debugging symbols (`-c dbg`).

## Environment Variables

Key environment variables (configure in `.env`):

- `ROOT_PATH`: Project root directory.
- `MODEL_PATH`: Directory for storing model files.
- `PROMPT_PATH`: Directory for prompt JSON files.
