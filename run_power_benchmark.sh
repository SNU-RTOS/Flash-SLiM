#!/bin/bash

# Power measurement benchmark script
# Runs LLM inference with different memory constraints and measures power consumption
# Usage: ./run_power_benchmark.sh

set -e  # Exit on any error

# Configuration
MEMORY_LIMITS=("1G" "3G")
REPEAT_COUNT=10
LOG_DIR="./log"
BASE_DATE="0831"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

echo "===== Power Measurement Benchmark ====="
echo "Memory limits: ${MEMORY_LIMITS[@]}"
echo "Repeat count: ${REPEAT_COUNT}"
echo "Log directory: ${LOG_DIR}"
echo "========================================"

# Function to run single benchmark
run_benchmark() {
    local memory_limit=$1
    local run_number=$2
    local log_file="${LOG_DIR}/${BASE_DATE}_power_128_4threads_32_memory_${memory_limit}_${run_number}.log"
    
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting run ${run_number}/5 with memory limit ${memory_limit}"
    echo "Log file: ${log_file}"
    
    # Power measurement command with taskset for CPU affinity
    taskset -c 3 sudo python3 tools/power_analysis/power_util.py --mode logger \
        --csv "${log_file}" \
        --realtime --rt-priority 99 \
        --exec "./run.sh --tl 32 --target cpu --core 2,4-7 --threads 4 --log --memory ${memory_limit} --file prompt/sample_prompt_128_1.json"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed run ${run_number}/5 with memory limit ${memory_limit}"
    
    # Brief pause between runs to let system settle
    sleep 2
}

# Main benchmark loop
total_runs=$((${#MEMORY_LIMITS[@]} * ${REPEAT_COUNT}))
current_run=0

for memory_limit in "${MEMORY_LIMITS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting benchmarks with memory limit: ${memory_limit}"
    echo "=========================================="
    
    for run_num in $(seq 1 ${REPEAT_COUNT}); do
        current_run=$((current_run + 1))
        echo ""
        echo "Progress: ${current_run}/${total_runs} (Memory: ${memory_limit}, Run: ${run_num}/${REPEAT_COUNT})"
        
        run_benchmark "${memory_limit}" "${run_num}"
    done
    
    echo ""
    echo "Completed all runs for memory limit: ${memory_limit}"
done

echo ""
echo "===== Benchmark Completed ====="
echo "Total runs completed: ${total_runs}"
echo "Log files created in: ${LOG_DIR}"

# List generated log files
echo ""
echo "Generated log files:"
ls -la "${LOG_DIR}/${BASE_DATE}_power_128_4threads_32_memory_"*.log 2>/dev/null || echo "No log files found"

echo ""
echo "===== Starting Power Analysis ====="

# Define input pattern and output directory
INPUT_PATTERN="${LOG_DIR}/${BASE_DATE}_power_128_4threads_32_*.log"
echo "Input pattern: ${INPUT_PATTERN}"
OUTPUT_DIR="${LOG_DIR}"

# Check if input files exist
if ! ls ${INPUT_PATTERN} 1> /dev/null 2>&1; then
    error "No files found matching pattern: ${INPUT_PATTERN}"
    exit 1
fi

log "Found files matching pattern:"
ls ${INPUT_PATTERN}

# Process each log file
for log_file in ${INPUT_PATTERN}; do
    if [[ -f "${log_file}" ]]; then
        # Generate output filename (.log -> .txt)
        output_file="${log_file%.log}.txt"
        
        log "Processing: ${log_file} -> ${output_file}"
        
        # Run power analysis and save to txt file
        python3 tools/power_analysis/power_util.py \
            --mode parser \
            --csv "${log_file}" \
            > "${output_file}"
        
        if [[ $? -eq 0 ]]; then
            log "✓ Successfully processed: $(basename ${log_file})"
        else
            error "✗ Failed to process: $(basename ${log_file})"
        fi
    fi
done

log "Power analysis completed. Results saved as .txt files in ${OUTPUT_DIR}"

echo ""
echo "===== All Processing Completed ====="
echo "Raw data: ${LOG_DIR}/*.log"
echo "Analysis results: ${LOG_DIR}/*.txt"
