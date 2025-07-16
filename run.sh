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

# --------------------------------------------------------------------------- #
# 0. Load environment and helpers                                            #
# --------------------------------------------------------------------------- #
source .env                              # contains PROMPT_PATH, MODEL_PATH, …
source ./scripts/utils.sh               # provides clear_caches()

# ------------------------------------------------------------------------------
# 1. Defaults & CLI parsing
# ------------------------------------------------------------------------------
TARGET="cpu"           # cpu | gpu
LOG_ENABLED=false      # log to file
CORE_LIST=all          # taskset core list
NUM_THREADS=1
PROMPT_FILE="./${PROMPT_PATH}/sample_prompt_8_1.json"
MAX_TOK_LEN=16
NUM_REPEATS=1          # number of iterations
MEMORY_LIMITS=()       # array of memory limits for cgroup testing
ENABLE_CGROUP=false    # enable memory-constrained benchmarking

usage() {
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
    -m, --memory <LIMIT>     Enable memory-constrained benchmarking with specified limit
                             Can be used multiple times: -m 512M -m 1G -m 2G
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
        ./run.sh --memory 512M --memory 1G --repeat 3 --log
EOF
    exit 0
}

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
    -m | --memory)
        MEMORY_LIMITS+=("$2")
        ENABLE_CGROUP=true
        shift 2
        ;;
    -h | --help) 
        usage 
        ;;
    *)
        echo "Unknown option: $1" >&2
        usage
        ;;
    esac
done

# Validate target
[[ "${TARGET}" =~ ^(gpu|cpu)$ ]] || {
    echo "Invalid --target ${TARGET}" >&2
    exit 1
}

# Validate memory limits if specified
if [[ ${#MEMORY_LIMITS[@]} -gt 0 ]]; then
    echo "Memory-constrained benchmarking enabled with limits: ${MEMORY_LIMITS[*]}"
fi

# Set default memory limits if cgroup is enabled but no limits specified
if [[ "$ENABLE_CGROUP" == true && ${#MEMORY_LIMITS[@]} -eq 0 ]]; then
    MEMORY_LIMITS=("2G")
    echo "No memory limits specified, using default: ${MEMORY_LIMITS[*]}"
fi

# ------------------------------------------------------------------------------
# 2. Logging helper
# ------------------------------------------------------------------------------
log() { echo "$@"; }
if $LOG_ENABLED; then
    log() { echo "$@" | tee -a "$OUTPUT_FILE"; }
fi

# ------------------------------------------------------------------------------
# 3. Memory-constrained execution helper (cgroup)
# ------------------------------------------------------------------------------
run_with_memlimit() {
    local mmax="$1"
    shift
    local cmd=("$@")

    if mount | grep -q "cgroup2"; then
        # Use systemd-run for cgroup v2
        systemd-run --quiet --scope \
            -p MemoryMax="$mmax" -p MemoryHigh="$mmax" -p MemorySwapMax=0 \
            -- "${cmd[@]}"
    else
        # Use cgexec for cgroup v1
        local cg="/sys/fs/cgroup/memory/llmbench"
        
        # Create cgroup if it doesn't exist
        if [[ ! -d "$cg" ]]; then
            echo "Creating cgroup: $cg" >&2
            sudo mkdir -p "$cg"
        fi
        
        # Set memory limit (always update to current limit)
        echo 0 | sudo tee "$cg/memory.force_empty" >/dev/null || true
        echo "$(($(numfmt --from=iec "$mmax")))" | sudo tee "$cg/memory.limit_in_bytes" >/dev/null
        
        # Execute with cgroup
        sudo cgexec -g memory:llmbench -- "${cmd[@]}"
    fi
}

# --------------------------------------------------------------------------- #
# 3. Model, Binary and Prompt settings                                       #
# --------------------------------------------------------------------------- #
MODEL_DIR="${MODEL_PATH}/llama-3.2-3b-it-q8"
MODEL_NAME="llama_q8_ekv1024"

# MODEL_DIR="${MODEL_PATH}/Gemma3-1B"
# MODEL_NAME="model.q8"

# MODEL_DIR="${MODEL_PATH}/SmolLM"
# MODEL_NAME="model.q8"

BIN="output/text_generator_main_${TARGET}"

# Validate prerequisites
[[ -x "$BIN" ]] || { 
    echo "[ERROR] Binary not found: $BIN" >&2
    exit 1 
}

[[ -f "$PROMPT_FILE" ]] || { 
    echo "[ERROR] Prompt file not found: $PROMPT_FILE" >&2
    exit 1 
}

# Validate file format
[[ "$PROMPT_FILE" =~ \.(json)$ ]] || {
    echo "[ERROR] Prompt file must be .json format" >&2
    exit 1
}

# --------------------------------------------------------------------------- #
# 4. Core Functions                                                          #
# --------------------------------------------------------------------------- #

# Function to run a single prompt
run_single_prompt() {
    local TOKENS="$1"
    local PROMPT="$2"  
    local OUTPUT_FILE="$3"
    local TEMPERATURE="$4"
    local TOP_K="$5"
    local TOP_P="$6"
    local REPETITION_PENALTY="$7"
    local ENABLE_REPETITION_PENALTY="$8"
    local MEMORY_LIMIT="${9:-}"  # Optional memory limit
    
    log "------ LLM inference start (${TARGET^^}) ------"
    log "Model            : ${MODEL_NAME}"
    log "Tokens requested : ${TOKENS}"
    # log "Prompt           : ${PROMPT}"
    log "Cores            : ${CORE_LIST}"
    log "Threads          : ${NUM_THREADS}"
    log "Max tokens       : ${MAX_TOK_LEN}"
    log "Temperature      : ${TEMPERATURE}"
    log "Top-k            : ${TOP_K}"
    log "Top-p            : ${TOP_P}"
    log "Repetition penalty: ${REPETITION_PENALTY}"
    log "Enable rep. penalty: ${ENABLE_REPETITION_PENALTY}"
    [[ -n "$MEMORY_LIMIT" ]] && log "Memory limit     : ${MEMORY_LIMIT}"
    log "Target Processor : ${TARGET^^}"
    log "Log file         : ${OUTPUT_FILE}"
    log "-----------------------------------------------"
    log "Cache clear ..."
    clear_caches

    # Build command as array (safe quoting)
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
    )

    # Add repetition penalty flag if enabled
    if [[ "${ENABLE_REPETITION_PENALTY}" == "true" ]]; then
        CMD+=(--enable_repetition_penalty)
    fi
    
    # Add taskset if specified
    [[ "${CORE_LIST}" != "all" ]] && CMD=(taskset -c "${CORE_LIST}" "${CMD[@]}")

    # Execute command with or without memory limit
    if [[ -n "$MEMORY_LIMIT" ]]; then
        # Memory-constrained execution with cgroup
        if $LOG_ENABLED; then
            run_with_memlimit "$MEMORY_LIMIT" "${CMD[@]}" 2>&1 | tee -a "${OUTPUT_FILE}"
            log "Results saved to ${OUTPUT_FILE}"
        else
            run_with_memlimit "$MEMORY_LIMIT" "${CMD[@]}"
        fi
    else
        # Normal execution
        if $LOG_ENABLED; then
            sudo "${CMD[@]}" 2>&1 | tee -a "${OUTPUT_FILE}"
            log "Results saved to ${OUTPUT_FILE}"
        else
            sudo "${CMD[@]}"
        fi
    fi
    log "-----------------------------------------------"
}

# Function to parse JSON and extract prompt data
parse_json_file() {
    local json_file="$1"
    
    echo "Detected JSON format. Processing..." >&2
    
    # Check if Python3 is available
    if ! command -v python3 >/dev/null 2>&1; then
        echo "[ERROR] Python3 is required for JSON parsing but not installed." >&2
        echo "Please install Python3: sudo apt-get install python3" >&2
        exit 1
    fi
    
    # Check if parser script exists
    local parser_script="./scripts/parse_json_prompt.py"
    if [[ ! -f "$parser_script" ]]; then
        echo "[ERROR] JSON parser script not found: $parser_script" >&2
        echo "Please ensure the script exists in the scripts directory." >&2
        exit 1
    fi
    
    # Run the Python parser
    local parse_result
    parse_result=$(python3 "$parser_script" "$json_file")
    local parse_status=$?
    
    if [[ $parse_status -ne 0 ]]; then
        echo "[ERROR] Failed to parse JSON file:" >&2
        echo "$parse_result" >&2
        exit 1
    fi
    
    echo "$parse_result"
}

# Function to process multiple prompts from array
process_multiple_prompts() {
    local parse_result="$1"
    local prompt_count
    prompt_count=$(echo "$parse_result" | head -n1 | awk '{print $2}')
    
    echo "Found $prompt_count prompts in JSON file. Processing all..."
    
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
            echo "Processing prompt $prompt_index_display/$prompt_count (${current_tokens} tokens)..."
            
            local timestamp output_file
            timestamp=$(date +'%y%m%d_%H%M%S')
            output_file="${RESULTS_DIR}/output_${current_tokens}_${prompt_index_display}_${timestamp}.log"
            
            run_single_prompt "$current_tokens" "$current_prompt" "$output_file" \
                "$current_temperature" "$current_top_k" "$current_top_p" \
                "$current_repetition_penalty" "$current_enable_repetition_penalty"
                
        elif [[ "$in_prompt" == true ]]; then
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
    
    # Parse: SINGLE tokens temperature top_k top_p repetition_penalty enable_repetition_penalty
    local first_line tokens temperature top_k top_p repetition_penalty enable_repetition_penalty
    first_line=$(echo "$parse_result" | head -n1)
    tokens=$(echo "$first_line" | awk '{print $2}')
    temperature=$(echo "$first_line" | awk '{print $3}')
    top_k=$(echo "$first_line" | awk '{print $4}')
    top_p=$(echo "$first_line" | awk '{print $5}')
    repetition_penalty=$(echo "$first_line" | awk '{print $6}')
    enable_repetition_penalty=$(echo "$first_line" | awk '{print $7}')
    
    local prompt timestamp output_file
    prompt=$(echo "$parse_result" | sed -n '/PROMPT_START/,/PROMPT_END/p' | sed '1d;$d')
    timestamp=$(date +'%y%m%d_%H%M%S')
    output_file="${RESULTS_DIR}/output_${tokens}_${timestamp}.log"
    
    run_single_prompt "$tokens" "$prompt" "$output_file" \
        "$temperature" "$top_k" "$top_p" "$repetition_penalty" "$enable_repetition_penalty"
}

# --------------------------------------------------------------------------- #
# 5. Main execution                                                          #
# --------------------------------------------------------------------------- #

# Function to execute benchmarks with or without memory constraints
execute_benchmarks() {
    local memory_limit="${1:-}"
    
    # Setup results directory
    if [[ -n "$memory_limit" ]]; then
        RESULTS_DIR="result_${memory_limit}"
        echo "[INFO] === Memory Limit: $memory_limit ==="
    else
        RESULTS_DIR="result_run_once"
    fi
    
    if [[ -d "${RESULTS_DIR}" ]]; then
        echo "Warning: ${RESULTS_DIR} already exists. Results will be appended."
    else
        echo "Creating results directory: ${RESULTS_DIR}"
        mkdir -p "${RESULTS_DIR}"
    fi
    
    # Execute for each repeat
    for ((iter = 1; iter <= NUM_REPEATS; iter++)); do
        [[ $NUM_REPEATS -gt 1 ]] && echo "[INFO] --- Iteration $iter / $NUM_REPEATS ---"
        
        # Parse JSON file
        local parse_result
        parse_result=$(parse_json_file "${PROMPT_FILE}")
        
        # Process based on JSON structure
        if [[ "$parse_result" =~ ^ARRAY ]]; then
            # Multiple prompts
            process_multiple_prompts "$parse_result" "$memory_limit" "$iter"
        elif [[ "$parse_result" =~ ^SINGLE ]]; then
            # Single prompt
            process_single_prompt "$parse_result" "$memory_limit" "$iter"
        else
            echo "[ERROR] Invalid JSON parse result format" >&2
            exit 1
        fi
    done
}

# Update process functions to accept memory limit and iteration
process_multiple_prompts() {
    local parse_result="$1"
    local memory_limit="${2:-}"
    local iteration="${3:-1}"
    
    local prompt_count
    prompt_count=$(echo "$parse_result" | head -n1 | awk '{print $2}')
    
    echo "Found $prompt_count prompts in JSON file. Processing all..."
    
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
            echo "Processing prompt $prompt_index_display/$prompt_count (${current_tokens} tokens)..."
            
            local timestamp output_file
            timestamp=$(date +'%y%m%d_%H%M%S')
            if [[ $NUM_REPEATS -gt 1 ]]; then
                output_file="${RESULTS_DIR}/output_${current_tokens}_${prompt_index_display}_${iteration}_${timestamp}.log"
            else
                output_file="${RESULTS_DIR}/output_${current_tokens}_${prompt_index_display}_${timestamp}.log"
            fi
            
            run_single_prompt "$current_tokens" "$current_prompt" "$output_file" \
                "$current_temperature" "$current_top_k" "$current_top_p" \
                "$current_repetition_penalty" "$current_enable_repetition_penalty" "$memory_limit"
                
        elif [[ "$in_prompt" == true ]]; then
            if [[ -z "$current_prompt" ]]; then
                current_prompt="$line"
            else
                current_prompt="$current_prompt"$'\n'"$line"
            fi
        fi
    done <<< "$parse_result"
}

process_single_prompt() {
    local parse_result="$1"
    local memory_limit="${2:-}"
    local iteration="${3:-1}"
    
    # Parse: SINGLE tokens temperature top_k top_p repetition_penalty enable_repetition_penalty
    local first_line tokens temperature top_k top_p repetition_penalty enable_repetition_penalty
    first_line=$(echo "$parse_result" | head -n1)
    tokens=$(echo "$first_line" | awk '{print $2}')
    temperature=$(echo "$first_line" | awk '{print $3}')
    top_k=$(echo "$first_line" | awk '{print $4}')
    top_p=$(echo "$first_line" | awk '{print $5}')
    repetition_penalty=$(echo "$first_line" | awk '{print $6}')
    enable_repetition_penalty=$(echo "$first_line" | awk '{print $7}')
    
    local prompt timestamp output_file
    prompt=$(echo "$parse_result" | sed -n '/PROMPT_START/,/PROMPT_END/p' | sed '1d;$d')
    timestamp=$(date +'%y%m%d_%H%M%S')
    if [[ $NUM_REPEATS -gt 1 ]]; then
        output_file="${RESULTS_DIR}/output_${tokens}_${iteration}_${timestamp}.log"
    else
        output_file="${RESULTS_DIR}/output_${tokens}_${timestamp}.log"
    fi
    
    run_single_prompt "$tokens" "$prompt" "$output_file" \
        "$temperature" "$top_k" "$top_p" "$repetition_penalty" "$enable_repetition_penalty" "$memory_limit"
}

# Main execution logic
if [[ "$ENABLE_CGROUP" == true ]]; then
    # Memory-constrained benchmarking
    for memory_limit in "${MEMORY_LIMITS[@]}"; do
        execute_benchmarks "$memory_limit"
    done
else
    # Normal benchmarking
    execute_benchmarks
fi

echo "All benchmarks completed successfully!"
exit 0
