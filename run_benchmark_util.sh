#!/usr/bin/env bash
# run_benchmark_util.sh
#
# TensorFlow Lite model benchmarking utility using benchmark_model tool
#
# Usage:
#   ./scripts/run_benchmark_util.sh
#
# Prerequisites:
#   * .env file with MODEL_PATH defined
#   * benchmark_model binary in util/ directory
#   * TensorFlow Lite model files (*.tflite) in model directory

set -euo pipefail

# --------------------------------------------------------------------------- #
# 0. Load environment and helpers                                            #
# --------------------------------------------------------------------------- #
if [[ ! -f .env ]]; then
    echo "[ERROR] .env file not found. Please run from project root directory." >&2
    exit 1
fi

source .env                              # Load environment variables
source ./scripts/utils.sh               # Load utility functions

banner "Build TensorFlow Lite Model Benchmarking Utility"
# --------------------------------------------------------------------------- #
# 1. Configuration                                                           #
# --------------------------------------------------------------------------- #
# Hardcoded model path for simplicity
MODEL_PATH="models/mobileone_s0.tflite"
OUTPUT_DIR="./benchmark/benchmark_model_results/tmp"
BENCHMARK_BIN="./bin/benchmark_model"

# Benchmark settings (hardcoded for simplicity)
NUM_THREADS=4
USE_XNNPACK=true
USE_GPU=false

# --------------------------------------------------------------------------- #
# 2. Validation                                                              #
# --------------------------------------------------------------------------- #
log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2; }

# Validate benchmark binary
if [[ ! -x "$BENCHMARK_BIN" ]]; then
    error "Benchmark binary not found or not executable: $BENCHMARK_BIN"
    echo "Please ensure the benchmark_model binary is available." >&2
    exit 1
fi

# Validate model file
if [[ ! -f "$MODEL_PATH" ]]; then
    error "Model file not found: $MODEL_PATH"
    exit 1
fi

# Create output directory
if [[ ! -d "$OUTPUT_DIR" ]]; then
    log "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# --------------------------------------------------------------------------- #
# 3. Benchmark execution                                                      #
# --------------------------------------------------------------------------- #
run_benchmark() {
    local model_path="$1"
    local model_name
    model_name=$(basename "$model_path" .tflite)
    
    # Generate filename suffix based on configuration
    local suffix=""
    
    # Add thread count
    if [[ "$NUM_THREADS" -eq 1 ]]; then
        suffix="${suffix}_single_thread"
    else
        suffix="${suffix}_${NUM_THREADS}threads"
    fi
    
    # Add acceleration type
    if [[ "$USE_GPU" == "true" ]]; then
        suffix="${suffix}_gpu"
    elif [[ "$USE_XNNPACK" == "true" ]]; then
        suffix="${suffix}_xnnpack"
    else
        suffix="${suffix}_cpu"
    fi
    
    local csv_file="${OUTPUT_DIR}/${model_name}${suffix}.csv"
    local log_file="${OUTPUT_DIR}/${model_name}${suffix}.log"
    
    log "Benchmarking: $model_name"
    log "Configuration: ${NUM_THREADS} threads, GPU=${USE_GPU}, XNNPACK=${USE_XNNPACK}"
    
    # Clear caches before benchmarking
    clear_caches
    
    # Execute benchmark (original hardcoded settings)
    if "$BENCHMARK_BIN" \
        --graph="$model_path" \
        --num_threads="$NUM_THREADS" \
        --enable_op_profiling=true \
        --use_xnnpack="$USE_XNNPACK" \
        --use_gpu="$USE_GPU" \
        --report_peak_memory_footprint=true \
        --op_profiling_output_mode=csv \
        --op_profiling_output_file="$csv_file" \
        > "$log_file" 2>&1; then
        
        log "✓ Completed: $model_name"
        log "  Results: $csv_file"
        log "  Log: $log_file"
    else
        error "✗ Failed: $model_name"
        error "  Check log: $log_file"
        return 1
    fi
}

# --------------------------------------------------------------------------- #
# 4. Main execution                                                          #
# --------------------------------------------------------------------------- #
main() {
    log "=== TensorFlow Lite Model Benchmarking ==="
    log "Model file: $MODEL_PATH"
    log "Output directory: $OUTPUT_DIR"
    log "Benchmark binary: $BENCHMARK_BIN"
    log "Threads: $NUM_THREADS"
    log "Use GPU: $USE_GPU"
    log "Use XNNPACK: $USE_XNNPACK"
    log "============================================="
    
    # Run benchmark for the single model file
    if run_benchmark "$MODEL_PATH"; then
        log "Benchmark completed successfully!"
    else
        error "Benchmark failed!"
        exit 1
    fi
}

# Run main function
main "$@"
