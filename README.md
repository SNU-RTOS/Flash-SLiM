# OS-Support-for-On-Device-LLM

A on-device Large Language Model (LLM) inference system with comprehensive profiling and evaluation capabilities, built with LiteRT (TensorFlow Lite).

## Overview

This project provides a complete framework for running LLMs on-device with:

- **Cross-platform compatibility**: Tested on x86_64 and aarch64 architectures
- **Performance profiling**: Built-in profiling tools
- **Benchmark utilities**: Comprehensive evaluation and performance analysis
- **Memory optimization**: XNNPACK weight caching and external KV cache management
- **Flexible deployment**: Support for various LLM architectures (Gemma, Llama, etc.)

## Platform Support

Successfully tested on the following platforms:

- **x86_64**: Ubuntu 22.04, Ubuntu 24.04
- **aarch64**: Ubuntu 20.04, Ubuntu 22.04, Ubuntu 24.04, Debian 13

**Note**: GPU acceleration is currently not supported due to model compatibility limitations.

## Project Structure

```text
├── src/                    # Core source code
│   ├── text_generator_main_cpu.cc   # CPU inference implementation
│   ├── text_generator_main_gpu.cc   # GPU inference (currently not supported)
│   ├── profiler.{h,cc}              # Performance profiling utilities
│   ├── sampler.{h,cc}               # Text generation sampling strategies
│   └── utils.{h,cc}                 # Common utilities
├── scripts/                # Build and utility scripts
│   ├── build-*.sh          # Dependency and component build scripts
│   ├── convert-*.sh         # Model conversion utilities
│   └── utils.sh             # Common shell utilities
├── evaluation/             # Performance evaluation tools
│   ├── plot.sh              # Plotting utilities
│   └── show-plot.py         # Visualization scripts
├── profiling/              # Profiling results and tools
│   ├── flame_graph_profiling.sh     # Flame graph generation
│   └── results/             # Profiling output directory
├── models/                 # Model storage directory
├── output/                 # Build output directory
└── external/               # External dependencies (auto-downloaded)
```

## Quick Start

### 1. Prerequisites

#### Download Bazelisk

This project requires Bazelisk, a version control tool for Bazel. Bazelisk automatically downloads and runs the appropriate Bazel version based on the `.bazelversion` file in your project.

**Ubuntu/Linux:**

```sh
curl -L https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -o /usr/local/bin/bazelisk
chmod +x /usr/local/bin/bazelisk
ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel  # Set up to use 'bazel' command
```

**macOS (Homebrew):**

```sh
brew install bazelisk
```

**Windows (Scoop):**

```sh
scoop install bazelisk
```

#### Environment Configuration

Copy the sample environment file and configure it for your system:

```sh
cp .env.sample .env
# Edit .env to match your system paths
```

### 2. Build Setup

#### Initial Setup (First Time)

Download external dependencies and build all components:

```sh
./setup.sh          # Release build (default)
./setup.sh debug    # Debug build
```

This script will:

- Download and build ai-edge-torch and TensorFlow dependencies
- Build LiteRT runtime
- Compile the LLM inference application (CPU version)

#### Subsequent Builds

After initial setup, use the faster build script:

```sh
./build.sh          # Build CPU version (default)
./build.sh cpu      # Build CPU version explicitly
```

**Note**: GPU build options (`./build.sh gpu` or `./build.sh all`) are available in the script but currently not functional due to model compatibility issues.

### 3. Running Inference

#### Quick Test Run

```sh
./run_once.sh       # Run CPU version (default and only supported)
./run_once.sh cpu   # Run CPU version explicitly
```

**Note**: GPU options (`./run_once.sh gpu`) exist in the script but are not currently functional.

#### Advanced Usage

```sh
# Run with custom parameters
./output/text_generator_main_cpu \
  --tflite_model=models/your_model.tflite \
  --prompt="Your input prompt here" \
  --num_threads=8

# Run with profiling
./profiling/flame_graph_profiling.sh
```

## Features

### Performance Profiling

- **Flame Graph Generation**: Visualize performance bottlenecks
- **Memory Usage Tracking**: Monitor memory consumption patterns
- **Benchmark Tools**: Comprehensive performance evaluation

### Optimization Features

- **XNNPACK Integration**: Hardware-optimized inference
- **Weight Caching**: Reduce memory usage and startup time
- **External KV Cache**: Efficient memory management for transformer models
- **Multi-threading**: Configurable thread pool for optimal performance

### Supported Models

- Gemma (2B, 7B variants)
- Llama 3.2 (3B variants)
- Custom models, converted with ai-edge-torch, or huggingface [https://huggingface.co/litert-community/]

## Build Options

The project supports the following build configurations:

- **Release Mode** (default): Optimized for performance
- **Debug Mode**: With debugging symbols and verbose logging
- **CPU-only**: Currently the only functional build option

**Future Work**: GPU acceleration support is planned but currently unavailable due to model compatibility limitations.

## Environment Variables

Key environment variables (configure in `.env`):

- `ROOT_PATH`: Project root directory
- `EXTERNAL_PATH`: External dependencies directory
- `AI_EDGE_TORCH_PATH`: AI Edge Torch installation path
- `LITERT_PATH`: LiteRT installation path
- `MODEL_PATH`: Model files directory
- `PROMPT_PATH`: Prompt files directory
