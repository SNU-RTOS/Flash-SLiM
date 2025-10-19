#!/usr/bin/env bash
# run.sh - Refactored version with better readability
#
# The JSON file should contain hyperparameters for each prompt:
# {
#   "tokens": 512,
#   "prompt": "Your prompt text here",
#   "temperature": 0.8,
#   "top_k": 50,
#   "top_p": 0.95,
#   "repetition_penalty": 1.1,
#   "enable_repetition_penalty": true
# }

set -euo pipefail

# =========================================================================== #
# 0. Load Environment and Helpers                                             #
# =========================================================================== #
if [[ ! -f .env ]]; then
    echo "[ERROR] .env file not found. Please run from project root directory." >&2
    exit 1
fi

source .env
source ./scripts/utils.sh

# =========================================================================== #
# 1. Configuration                                                            #
# =========================================================================== #
banner "Script Configuration"

# --- Model and Binary Settings ---
# You can switch models by uncommenting the desired lines.
# MODEL_DIR="${MODEL_PATH}/Llama3.2-1B"
# MODEL_NAME="llama3.2_q8_ekv1280"

MODEL_DIR="${MODEL_PATH}/Llama3.2-3B"
MODEL_NAME="llama3.2_q8_ekv1024"

# MODEL_DIR="${MODEL_PATH}/Gemma3-1B"
# MODEL_NAME="gemma3_q4_ekv2048"
# MODEL_NAME="gemma3_q8_ekv2048"

# MODEL_DIR="${MODEL_PATH}/Gemma2-2B"
# MODEL_NAME="gemma2_q8_ekv1024"

# MODEL_DIR="${MODEL_PATH}/Qwen2.5-1.5B"
# MODEL_NAME="qwen2.5-1.5b_q8_ekv1280"
# MODEL_NAME="qwen2.5-1.5b_q8_ekv4096"

# MODEL_DIR="${MODEL_PATH}/Qwen2.5-3B"
# MODEL_NAME="qwen2.5-3b_q8_ekv1280"

# MODEL_DIR="${MODEL_PATH}/SmolLM-135M"
# MODEL_NAME="smollm_q8_ekv1280"

BIN="bin/prefetch_planner"

# --- Execution Settings ---
TARGET="cpu"           # Default target: cpu | gpu
LOG_ENABLED=false      # Default logging: false
CORE_LIST="all"          # Default core list for taskset
NUM_THREADS=1          # Default number of threads
PROMPT_FILE="${PROMPT_PATH}/sample_prompt_8_1.json" # Default prompt file
MAX_TOK_LEN=16         # Default max tokens to generate
NUM_REPEATS=1          # Default number of iterations
MEMORY_LIMITS=()       # Array of memory limits for cgroup testing
ENABLE_CGROUP=false    # Default cgroup enable state

# --- Logging Settings ---
# Base directory for logs. The final path will be e.g. <LOG_DIR_BASE>/<model_target_mem>
LOG_DIR_BASE="benchmark/llm_infer_results"

log "--- Using the following configuration ---"
log "Model            : ${MODEL_DIR}/${MODEL_NAME}.tflite"
log "Binary           : ${BIN}"
log "Default prompt   : ${PROMPT_FILE}"
log "Log base dir     : ${LOG_DIR_BASE}"
log "-------------------------------------------"

# =========================================================================== #
# 2. Validate Prerequisites                                                   #
# =========================================================================== #
[[ -x "$BIN" ]] || error "Binary not found: $BIN"
[[ -f "$PROMPT_FILE" ]] || error "Prompt file not found: $PROMPT_FILE"
[[ "$PROMPT_FILE" =~ \.(json)$ ]] || error "Prompt file must be .json format"

# =========================================================================== #
# 3. Usage and CLI Parsing                                                    #
# =========================================================================== #
usage() {
    banner "Usage Information"
    cat <<'EOF'
    Usage: run.sh [OPTIONS]

    This script runs LLM inference using either a GPU or CPU binary,
    based on the specified model and prompt input from JSON files.
    Supports memory-constrained benchmarking with cgroups.

    Options:
    -g, --target <gpu|cpu>   Select binary to run (default: cpu)
    -l, --log                Enable logging to ./result_* directory (default: off)
    -c, --core <CORES>       Specify CPU core(s) to bind the process using 'taskset'
                             For example: "0-3" or "0,2". Default is to use all available cores.
    -t, --threads <N>        Set the number of threads to use for inference (default: 1)
    -f, --file <PATH>        Path to the input prompt file (JSON format)
                             Default: ./${PROMPT_PATH}/sample_prompt_8_1.json
    --tl, --max_tokens <N>   Set the maximum number of tokens to generate (default: 16)
    -r, --repeat <N>         Repeat all prompts N times (default: 1)
                                 Supports suffixes: K, M, G (e.g., 512M, 1G, 2G)
    -h, --help               Show this help message and exit

    JSON Format:
        Single prompt: {
          "tokens": 512,
          "prompt": "Hello, how are you?",
          "temperature": 0.8,
          "top_k": 50,
          "top_p": 0.95,
          "repetition_penalty": 1.1,
          "enable_repetition_penalty": true
        }
        
        Multiple prompts: [
          {"tokens": 512, "prompt": "...", "temperature": 0.8, ...},
          {"tokens": 256, "prompt": "...", "temperature": 0.7, ...}
        ]

    Examples:
        ./run.sh --target gpu --log --core 0-3 --threads 4
        ./run.sh -g cpu -f prompts.json --tl 32
        ./run.sh --target cpu --file my_prompts.json
EOF
    exit 0
}

parse_command_line(){
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    -g | --target)
        TARGET="$2"
        shift 2
        ;;
    -l | --log)
        LOG_ENABLED=true
        shift
        ;;
    -c | --core)
        CORE_LIST="$2"
        shift 2
        ;;
    -t | --threads)
        NUM_THREADS="$2"
        shift 2
        ;;
    -f | --file)
        PROMPT_FILE="$2"
        shift 2
        ;;
    --tl | --max_tokens)
        MAX_TOK_LEN="$2"
        shift 2
        ;;
    -r | --repeat)
        NUM_REPEATS="$2"
        shift 2
        ;;
    -h | --help) 
        usage 
        ;;
    *)
        error "Unknown option: $1"
        usage
        ;;
    esac
done

# Validate target
[[ "${TARGET}" =~ ^(gpu|cpu)$ ]] || error "Invalid target argument: ${TARGET}"

# Validate memory limits if specified
if [[ ${#MEMORY_LIMITS[@]} -gt 0 ]]; then
    log "Memory-constrained benchmarking enabled with limits: ${MEMORY_LIMITS[*]}"
fi

# Set default memory limits if cgroup is enabled but no limits specified
if [[ "$ENABLE_CGROUP" == true && ${#MEMORY_LIMITS[@]} -eq 0 ]]; then
    MEMORY_LIMITS=("2G")
    log "No memory limits specified, using default: ${MEMORY_LIMITS[*]}"
fi

}



# =========================================================================== #
# 5. Core Functions                                                           #
# =========================================================================== #


# Function to parse JSON and extract prompt data
parse_json_file() {
    local json_file="$1"
    
    log "Detected JSON format. Processing..." >&2
    
    # Check if Python3 is available
    if ! command -v python3 >/dev/null 2>&1; then
        error "Python3 is required for JSON parsing but not installed. Please install Python3: sudo apt-get install python3"
    fi
    
    # Check if parser script exists
    local parser_script="./tools/prompt/parse_json_prompt.py"
    if [[ ! -f "$parser_script" ]]; then
        error "JSON parser script not found: $parser_script. Please ensure the script exists in the scripts directory."
    fi
    
    # Run the Python parser
    local parse_result
    parse_result=$(python3 "$parser_script" "$json_file")
    local parse_status=$?
    
    if [[ $parse_status -ne 0 ]]; then
        error "Failed to parse JSON file:\n$parse_result"
    fi
    
    echo "$parse_result"
}

# Function to run a single prompt
run_single_prompt() {
    local TOKENS="$1"
    local PROMPT="$2"  
    local LOG_FILE="$3"
    local TEMPERATURE="$4"
    local TOP_K="$5"
    local TOP_P="$6"
    local REPETITION_PENALTY="$7"
    local ENABLE_REPETITION_PENALTY="$8"
    local MEMORY_LIMIT="${9:-}"
    local CSV_FILE="${LOG_FILE%.log}.csv"

    banner "LLM inference start (${TARGET^^})"
    log "Model                           : ${MODEL_NAME}"
    log "Tokens requested                : ${TOKENS}"
    # log "Prompt           : ${PROMPT}"
    log "Cores                           : ${CORE_LIST}"
    log "Threads                         : ${NUM_THREADS}"
    log "Max tokens                      : ${MAX_TOK_LEN}"
    log "Temperature                     : ${TEMPERATURE}"
    log "Top-k                           : ${TOP_K}"
    log "Top-p                           : ${TOP_P}"
    log "Repetition penalty              : ${REPETITION_PENALTY}"
    log "Enable rep. penalty             : ${ENABLE_REPETITION_PENALTY}"
    [[ -n "$MEMORY_LIMIT" ]] && \
    log "Memory limit                    : ${MEMORY_LIMIT}"
    log "Target Processor                : ${TARGET^^}"
    if [[ "$LOG_ENABLED" == "true" ]]; then
        log "Log file                        : ${LOG_FILE}"
        log "Op-level profiling csv results  : ${CSV_FILE}"
    fi

    clear_caches

    # Build command as array (safe quoting)

    if [[ "$LOG_ENABLED" == "true" ]]; then
        local CMD=(
            "${BIN}"
            --tflite_model "${MODEL_DIR}/${MODEL_NAME}.tflite"
            --sentencepiece_model "${MODEL_DIR}/tokenizer.model"
            --max_decode_steps "${MAX_TOK_LEN}"
            --start_token "<bos>"
            --stop_token "<end_of_turn>"
            --num_threads "${NUM_THREADS}"
            --prompt "${PROMPT}"
            --weight_cache_path "${MODEL_DIR}/${MODEL_NAME}.xnnpack_cache"
            --temperature "${TEMPERATURE}"
            --top_k "${TOP_K}"
            --top_p "${TOP_P}"
            --repetition_penalty "${REPETITION_PENALTY}"
            --csv_profile_output_path "$CSV_FILE"
            --model_dump_file_path "${MODEL_DIR}/${MODEL_NAME}_dump.log"
            --op_tensor_byte_stats
            --dump_tensor_details
        )
    else
        local CMD=(
            "${BIN}"
            --tflite_model "${MODEL_DIR}/${MODEL_NAME}.tflite"
            --sentencepiece_model "${MODEL_DIR}/tokenizer.model"
            --max_decode_steps "${MAX_TOK_LEN}"
            --start_token "<bos>"
            --stop_token "<end_of_turn>"
            --num_threads "${NUM_THREADS}"
            --prompt "${PROMPT}"
            --weight_cache_path "${MODEL_DIR}/${MODEL_NAME}.xnnpack_cache"
            --temperature "${TEMPERATURE}"
            --top_k "${TOP_K}"
            --top_p "${TOP_P}"
            --repetition_penalty "${REPETITION_PENALTY}"
            --model_dump_file_path "${MODEL_DIR}/${MODEL_NAME}_dump.log"
            --op_tensor_byte_stats
            --dump_tensor_details
        )
    fi
    


    # Add repetition penalty flag if enabled
    if [[ "${ENABLE_REPETITION_PENALTY}" == "true" ]]; then
        CMD+=(--enable_repetition_penalty)
    fi
    
    # Add taskset if specified
    [[ "${CORE_LIST}" != "all" ]] && CMD=(taskset -c "${CORE_LIST}" "${CMD[@]}")

    banner "--- C++ Binary Execution START ---"
    banner "Command: ${CMD[*]}"
    # Execute command with or without memory limit
    if [[ -n "$MEMORY_LIMIT" ]]; then
        # Memory-constrained execution with cgroup
        run_with_memlimit "$MEMORY_LIMIT" "${CMD[@]}"
    else
        # Normal execution
        sudo "${CMD[@]}"
    fi


    banner "--- C++ Binary Execution END ---"

    if [[ "$LOG_ENABLED" == "true" ]]; then
        log "Log saved to ${LOG_FILE}"
    fi

    # python3 tools/model_dump/tensor_visualization.py \
    #         "${MODEL_DIR}/${MODEL_NAME}_dump.log" \
    #          "${MODEL_DIR}/${MODEL_NAME}_analysis_report.txt" \
    #          "${MODEL_DIR}/${MODEL_NAME}_analysis_data.json"

}


# Function to process multiple prompts from array
process_multiple_prompts() {
    local parse_result="$1"
    local memory_limit="${2:-}"
    local iteration="${3:-1}"
    local log_dir="$4"

    local prompt_count
    prompt_count=$(echo "$parse_result" | head -n1 | awk '{print $2}')
    
    log "Found $prompt_count prompts in JSON file. Processing all..."
    
    local prompt_index=0
    local current_tokens current_temperature current_top_k current_top_p
    local current_repetition_penalty current_enable_repetition_penalty
    local current_prompt="" in_prompt=false
    
    while IFS= read -r line; do
        if [[ "$line" =~ ^ITEM ]]; then
            # Parse: ITEM index tokens temperature top_k top_p repetition_penalty enable_repetition_penalty
            prompt_index=$(echo "$line" | awk '{print $2}')
            current_tokens=$(echo "$line" | awk '{print $3}')
            current_temperature=$(echo "$line" | awk '{print $4}')
            current_top_k=$(echo "$line" | awk '{print $5}')
            current_top_p=$(echo "$line" | awk '{print $6}')
            current_repetition_penalty=$(echo "$line" | awk '{print $7}')
            current_enable_repetition_penalty=$(echo "$line" | awk '{print $8}')
            current_prompt=""
            in_prompt=false
            
        elif [[ "$line" == "PROMPT_START" ]]; then
            in_prompt=true
            current_prompt=""
            
        elif [[ "$line" == "PROMPT_END" ]]; then
            in_prompt=false
            
            # Process this prompt
            local prompt_index_display=$((prompt_index + 1))
            log "Processing prompt $prompt_index_display/$prompt_count (${current_tokens} tokens)..." >&2
            
            local timestamp log_file
            timestamp=$(date +'%y%m%d_%H%M%S')
            if [[ $NUM_REPEATS -gt 1 ]]; then
                log_file="${log_dir}/run_${current_tokens}_${prompt_index_display}_${iteration}_${timestamp}.log"
            else
                log_file="${log_dir}/run_${current_tokens}_${prompt_index_display}_${timestamp}.log"
            fi

            log "Executing LLM Inference"
            execute_with_log "$log_file" run_single_prompt "$current_tokens" "$current_prompt" "$log_file" \
                "$current_temperature" "$current_top_k" "$current_top_p" \
                "$current_repetition_penalty" "$current_enable_repetition_penalty" "$memory_limit"
                
        elif [[ "$in_prompt" == "true" ]]; then
            if [[ -z "$current_prompt" ]]; then
                current_prompt="$line"
            else
                current_prompt="$current_prompt"$'\n'"$line"
            fi
        fi
    done <<< "$parse_result"
}

# Function to process single prompt
process_single_prompt() {
    local parse_result="$1"
    local memory_limit="${2:-}"
    local iteration="${3:-}"
    local log_dir="$4"
    
    # Parse: SINGLE tokens temperature top_k top_p repetition_penalty enable_repetition_penalty
    local first_line tokens temperature top_k top_p repetition_penalty enable_repetition_penalty
    first_line=$(echo "$parse_result" | head -n1)
    tokens=$(echo "$first_line" | awk '{print $2}')
    temperature=$(echo "$first_line" | awk '{print $3}')
    top_k=$(echo "$first_line" | awk '{print $4}')
    top_p=$(echo "$first_line" | awk '{print $5}')
    repetition_penalty=$(echo "$first_line" | awk '{print $6}')
    enable_repetition_penalty=$(echo "$first_line" | awk '{print $7}')
    
    local prompt timestamp log_file
    prompt=$(echo "$parse_result" | sed -n '/PROMPT_START/,/PROMPT_END/p' | sed '1d;$d')
    timestamp=$(date +'%y%m%d_%H%M%S')
    if [[ $NUM_REPEATS -gt 1 ]]; then
        log_file="${log_dir}/run_${tokens}_${iteration}_${timestamp}.log"
    else
        log_file="${log_dir}/run_${tokens}_${timestamp}.log"
    fi
    
    log "Executing LLM Inference"
    execute_with_log "$log_file" run_single_prompt "$tokens" "$prompt" "$log_file" \
        "$temperature" "$top_k" "$top_p" "$repetition_penalty" "$enable_repetition_penalty" "$memory_limit"

}


# Function to execute benchmarks with or without memory constraints
execute_benchmarks() {
    local memory_limit="${1:-}"
    local current_log_dir
    
    # Setup log directory
    local model_id="${MODEL_NAME}"
    local target_id="${TARGET}"
    local mem_id
    
    if [[ -n "$memory_limit" ]]; then
        mem_id="_${memory_limit}"
        log "Memory Limit: $memory_limit"
    else
        mem_id=""
        log "Normal run (no memory limit)"
    fi
    current_log_dir="${LOG_DIR_BASE}/${model_id}_${target_id}${mem_id}"
    
    if [[ "$LOG_ENABLED" == "true" ]]; then
        if [[ -d "${current_log_dir}" ]]; then
            log "Log directory exists: ${current_log_dir} (appending)"
        else
            ensure_dir "${current_log_dir}"
            log "Created log directory: ${current_log_dir}"
        fi
    fi
    
    # Execute for each repeat
    for ((iter = 1; iter <= NUM_REPEATS; iter++)); do
        if [[ $NUM_REPEATS -gt 1 ]]; then
            banner "Iteration $iter / $NUM_REPEATS"
        fi
        
        # Parse JSON file
        local parse_result
        parse_result=$(parse_json_file "${PROMPT_FILE}")
        
        # Process based on JSON structure
        if [[ "$parse_result" =~ ^ARRAY ]]; then
            # Multiple prompts
            process_multiple_prompts "$parse_result" "$memory_limit" "$iter" "$current_log_dir"
        elif [[ "$parse_result" =~ ^SINGLE ]]; then
            # Single prompt
            process_single_prompt "$parse_result" "$memory_limit" "$iter" "$current_log_dir"
        else
            error "Invalid JSON parse result format"
        fi
    done
}

# =========================================================================== #
# 6. Main Execution                                                           #
# =========================================================================== #


# --- Main Execution Logic ---
main() {
    parse_command_line "$@"

    # Main execution logic
    if [[ "$ENABLE_CGROUP" == "true" ]]; then
        # Memory-constrained benchmarking
        for memory_limit in "${MEMORY_LIMITS[@]}"; do
            execute_benchmarks "$memory_limit"
        done
    else
        # Normal benchmarking
        execute_benchmarks
    fi

    log "All benchmarks completed successfully!"
}

# Run main function
main "$@"
