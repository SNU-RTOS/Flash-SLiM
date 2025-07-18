#!/usr/bin/env bash
# run_benchmark_util.sh
#
# TensorFlow Lite model benchmarking utility using benchmark_model tool
#
# Usage:
#   ./scripts/run_benchmark_util.sh
#
# Prerequisites:
#   * .env file with ROOT_PATH defined
#   * benchmark_model binary in util/ directory
#   * TensorFlow Lite model files (*.tflite) in model directory

set -euo pipefail

# =========================================================================== #
# 0. Load Environment and Helpers                                             #
# =========================================================================== #
if [[ ! -f .env ]]; then
    echo "[ERROR] .env file not found. Please run from project root directory." >&2
    exit 1
fi

source .env                              # Load environment variables
source ./scripts/utils.sh               # Load utility functions

# =========================================================================== #
# 1. Configuration                                                            #
# =========================================================================== #
banner "Benchmark Utility Configuration"

# --- Model and Binary Settings ---
MODEL_PATH="./models/test/mobileone_s0.tflite"
OUTPUT_DIR="./benchmark/benchmark_model_results/tmp"
BENCHMARK_BIN="./util/bin/benchmark_model"

# --- Benchmark Settings ---
NUM_THREADS=4
USE_XNNPACK=true
USE_GPU=false

# =========================================================================== #
# 2. Validation                                                               #
# =========================================================================== #
# Validate benchmark binary
if [[ ! -x "$BENCHMARK_BIN" ]]; then
    log "Benchmark binary not found: $BENCHMARK_BIN"
    banner "Building benchmark_model binary..."
    (
        cd scripts || exit 1
        ./build-benchmark_util.sh
    )
    [[ ! -x "$BENCHMARK_BIN" ]] && error "Failed to build benchmark_model binary."
    log "Successfully built benchmark_model binary: $BENCHMARK_BIN"
else
    log "Found benchmark_model binary: $BENCHMARK_BIN"
fi

# Validate model file
[[ -f "$MODEL_PATH" ]] || error "Model file not found: $MODEL_PATH"

# Create output directory
ensure_dir "$OUTPUT_DIR"

# =========================================================================== #
# 3. Core Functions                                                           #
# =========================================================================== #
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
    
    clear_caches
    
    # Execute benchmark
    local CMD=(
        "$BENCHMARK_BIN"
        --graph="$model_path"
        --num_threads="$NUM_THREADS"
        --enable_op_profiling=true
        --use_xnnpack="$USE_XNNPACK"
        --use_gpu="$USE_GPU"
        --report_peak_memory_footprint=true
        --op_profiling_output_mode=csv
        --op_profiling_output_file="$csv_file"
    )

    if execute_with_log "$log_file" "${CMD[@]}"; then
        log "Completed: $model_name"
        log "  Results: $csv_file"
        log "  Log: $log_file"
    else
        error "Failed: $model_name. Check log: $log_file"
        return 1
    fi
}

# =========================================================================== #
# 4. Main Execution                                                           #
# =========================================================================== #
main() {
    banner "TensorFlow Lite Model Benchmarking"
    log "Model file: $MODEL_PATH"
    log "Output directory: $OUTPUT_DIR"
    log "Benchmark binary: $BENCHMARK_BIN"
    log "Threads: $NUM_THREADS"
    log "Use GPU: $USE_GPU"
    log "Use XNNPACK: $USE_XNNPACK"
    
    # Enable logging for the benchmark run
    LOG_ENABLED=true

    if run_benchmark "$MODEL_PATH"; then
        banner "Benchmark completed successfully!"
    else
        error "Benchmark failed!"
    fi
}

# Run main function
main "$@"
