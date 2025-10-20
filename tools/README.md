# tools/ Directory Guide

This directory contains auxiliary tools and scripts for development, profiling, analysis, and prefetch planning in the Flash-SLiM project. Tools are organized by their role in the **Pre-runtime** and **Runtime** stages.

## Directory Overview

### Pre-runtime Tools (Plan Generation)

#### model_prefetch_planner/

- **Purpose**: Generate Prefetch Plan for weight streaming runtime
- **prefetch_planner.py** (⏳ Planned): Python script to create prefetch scheduling timeline
  - Reads CMT (Chunk Metadata Table) and operator profiling data
  - Applies greedy partitioning algorithm
  - Optimizes chunk loading timeline for I/O–Compute overlap
  - Output: Prefetch Plan JSON

#### model_dump/

- **Purpose**: Model structure analysis and inspection (Pre-runtime analysis)
- **dump_llm_nodes** (5.3MB binary): Compiled model inspection tool
- **dump_llm_nodes.cc** (35KB): Source code for detailed model analysis
  - Parses TFLite model structure and execution plans
  - Counts XNNPACK operators within delegate nodes
  - Tracks tensor allocation types (Mmap, Arena)
  - Generates execution plan visualization
- **tensor_visualization.py** (23KB): Python visualization tool for tensor analysis
- **parse_model.sh**: Build script for model parser
- **log/**: Output logs from model analysis runs

### Runtime Tools (Profiling & Analysis)

#### ebpf/

- **Purpose**: System-level profiling during inference execution
- **profile_phase.py** (29KB): eBPF-based profiling orchestrator
  - Captures page faults, I/O events, system calls
  - Phase-level (prefill/decode) tracing
  - Integrates with LiteRT event hooks via USDT probes
- **bpfc_profile_phase.c** (15KB): eBPF bytecode for profiling
- **operator-level/**: Operator-level eBPF tracing scripts
- **phase-level/**: Phase-level (prefill/decode) tracing scripts
- **Usage**: `sudo python tools/ebpf/profile_phase.py --pid <PID>`

### Analysis & Visualization Tools

#### benchmark/

- **Purpose**: Post-processing and analysis of profiling results
- **fix_profile_report.py**: Post-processes LiteRT benchmark CSV output
  - Cleans and normalizes profiling data
  - Generates unified timeline for analysis
- **Usage**: `python tools/benchmark/fix_profile_report.py <input.csv> <output.csv>`

#### plot/

- **Purpose**: Data visualization and plotting
- **plot.sh**: Shell script for automated plotting
- **show_plot.py**: Python script for visualizing benchmark results
  - Generates charts for KPIs (latency, throughput, memory usage)
  - Supports timeline visualization for compute-I/O overlap analysis
- **Usage**: `python tools/plot/show_plot.py <data.json>`

#### flamegraph/

- **Purpose**: Flamegraph generation for performance visualization
- **flame_graph_profiling.sh**: Automated flamegraph generation
- **flamegraph.svg**: Sample flamegraph output
- **Usage**: `bash tools/flamegraph/flame_graph_profiling.sh`

#### power_analysis/

- **Purpose**: Power consumption analysis during inference
- Scripts for measuring and analyzing power usage on target devices

### System Utilities

#### cache/

- **Purpose**: System cache clearing for consistent benchmarking
- **clear_cache_x86.cc**: Source code for x86_64 cache clearing
- **clear_cache_arm.cc**: Source code for ARM64 cache clearing
- **Compilation**: `g++ -o clear_cache tools/cache/clear_cache_x86.cc`
- **Usage**: `sudo ./clear_cache`

#### prompt/

- **Purpose**: Prompt processing and management
- **parse_json_prompt.py**: Parses prompt JSON files
- **update_json_prompt_hyperparams.py**: Updates hyperparameters in prompt files
- **Usage**: `python tools/prompt/parse_json_prompt.py <prompt.json>`

## Tool Workflow

### Pre-runtime Workflow

```text
1. Model Analysis (model_dump/)
   ↓
2. Profiling Execution (Run inference with profilers)
   ↓
3. Prefetch Plan Generation (model_prefetch_planner/)
   ↓
Output: CMT + Prefetch Plan
```

### Runtime Workflow

```text
1. Execute Inference (text_generator_main)
   ↓
2. Real-time Profiling (ebpf/)
   ↓
3. Post-processing (benchmark/, plot/)
   ↓
Output: Performance metrics, visualizations
```

## Common Usage Examples

### Pre-runtime Analysis

```bash
# Analyze model structure
./tools/model_dump/dump_llm_nodes --model=/path/to/model.tflite

# Generate prefetch plan (planned)
python tools/model_prefetch_planner/prefetch_planner.py \
  --cmt=weight_chunks_metadata_table.json \
  --profile=merged_profile.json \
  --output=prefetch_plan.json
```

### Runtime Profiling

```bash
# eBPF profiling during inference
sudo python tools/ebpf/profile_phase.py --pid $(pgrep text_generator)

# Post-process profiling results
python tools/benchmark/fix_profile_report.py profile.csv profile_clean.csv

# Generate visualization
python tools/plot/show_plot.py profile_clean.csv
```

### System Preparation

```bash
# Clear system cache before benchmark
g++ -o clear_cache tools/cache/clear_cache_x86.cc
sudo ./clear_cache

# Parse and update prompt
python tools/prompt/parse_json_prompt.py prompt/test_prompt.json
```

## Tool Dependencies

### Python Tools

- **numpy**, **matplotlib**, **plotly**: Visualization tools
- **json**: CMT and Prefetch Plan parsing
- **bcc** (BPF Compiler Collection): eBPF profiling

### C++ Tools

- **Bazel**: Build system for model_dump
- **TFLite/LiteRT**: Model parsing and analysis
- Standard C++ compiler for cache utilities

## Related Documentation

- Main README: [../README.md](../README.md)
- Flash-slim source: [../flash-slim/README.md](../flash-slim/README.md)
- Model dump details: [model_dump/README.md](model_dump/README.md)

For detailed usage of each tool, refer to the comments or README within the respective file/directory.
