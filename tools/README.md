
# tools/ Directory Guide

This directory contains various auxiliary tools and scripts for development, benchmarking, and analysis in the flash-slim project. Each subfolder and main file serves a specific purpose as described below.

## Directory and Key File Overview

- **benchmark/**
  - `fix_profile_report.py`: Post-processes LiteRT benchmark CSV output
- **bin/**
  - Binaries and utilities for benchmarking/system operations
- **cache/**
  - `clear_cache_arm.cc`, `clear_cache_x86.cc`: Source code to clear system cache on ARM and x86 environments
- **epbf/**
  - `tflite_gen_profile.bt`, `tflite_gen_profile_stages.bt`: eBPF-based profiling scripts
- **flamegraph/**
  - `flame_graph_profiling.sh`, `flamegraph.svg`: Flamegraph generation and visualization tools
- **model_analyzer.py**
  - Python script for model analysis
- **plot/**
  - `plot.sh`, `show_plot.py`: Scripts for visualizing and plotting benchmark results
- **prompt/**
  - `parse_json_prompt.py`, `update_json_prompt_hyperparams.py`: Scripts for parsing prompt JSON and updating hyperparameters

## Usage Examples

- Benchmark result post-processing: `python tools/benchmark/fix_profile_report.py <input.csv> <output.csv>`
- System cache clearing: Compile and run `tools/cache/clear_cache_x86.cc` or `tools/cache/clear_cache_arm.cc`
- eBPF profiling: `sudo bpftrace tools/epbf/tflite_gen_profile.bt`
- Flamegraph generation: `bash tools/flamegraph/flame_graph_profiling.sh`
- Prompt parsing: `python tools/prompt/parse_json_prompt.py <prompt.json>`

For detailed usage of each tool, refer to the comments or README within the respective file.
