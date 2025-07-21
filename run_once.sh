#!/usr/bin/env bash
# run_once.sh
#
#   ./run_once.sh            # run once (default: cpu)
#   ./run_once.sh gpu        # run GPU binary once
#   ./run_once.sh cpu        # run CPU binary once
#
# Prerequisites:
#   * build.sh must have produced output/text_generator_main
#   * .env defines ROOT_PATH, MODEL_PATH, PROMPT_PATH, etc.
#   * scripts/utils.sh provides clear_caches(), banner(), ensure_dir()

set -euo pipefail

# =========================================================================== #
# 0. Load Environment and Helpers                                             #
# =========================================================================== #
if [[ ! -f .env ]]; then
    error ".env file not found. Please run from project root directory."
fi

source .env
source ./scripts/utils.sh

# =========================================================================== #
# 1. Configuration                                                            #
# =========================================================================== #
banner "Single Prompt Run Configuration"

# --- Execution Settings ---
TARGET="${1:-cpu}"
LOG_ENABLED=true # Always log for single runs
LOG_DIR="result_run_once"
LOG_FILE="${LOG_DIR}/output.log"

# --- Model and Binary Settings ---
MODEL_DIR="${MODEL_PATH}/Llama3.2-1B"
MODEL_NAME="llama_q8_ekv1024"
BIN="output/text_generator_main"
PROMPT_FILE="./${PROMPT_PATH}/sample_prompt_8_1.json"

# =========================================================================== #
# 2. Validation                                                               #
# =========================================================================== #
case "${TARGET}" in
  cpu|gpu) ;;
  *)
    error "Invalid target argument. Usage: $0 [cpu|gpu]"
    ;;
esac

[[ -x "${BIN}" ]] || error "${BIN} not found or not executable. Build it first!"
[[ -f "${PROMPT_FILE}" ]] || error "Prompt file '${PROMPT_FILE}' does not exist."

ensure_dir "${LOG_DIR}"

log "Selected binary: ${BIN}"
log "Target accelerator: ${TARGET^^}"

# =========================================================================== #
# 3. Core Functions                                                           #
# =========================================================================== #
parse_prompt_file() {
    local file_content
    file_content=$(cat "${PROMPT_FILE}")

    if [[ "$file_content" =~ ^\s*\[ || "$file_content" =~ ^\s*\{ ]]; then
        banner "Parsing JSON prompt file"
        if ! command -v python3 >/dev/null 2>&1; then
            error "Python3 is required for JSON parsing but not installed."
        fi
        local parser_script="./scripts/parse_json_prompt.py"
        [[ -f "$parser_script" ]] || error "JSON parser script not found: $parser_script"

        local parse_result
        parse_result=$(python3 "$parser_script" "$PROMPT_FILE")
        [[ $? -eq 0 ]] || error "Failed to parse JSON file."

        if [[ "$parse_result" =~ ^ARRAY ]]; then
            local first_line
            first_line=$(echo "$parse_result" | grep "^ITEM 0" | head -n1)
            TOKEN_COUNT=$(echo "$first_line" | awk '{print $3}')
            TEMPERATURE=$(echo "$first_line" | awk '{print $4}')
            TOP_K=$(echo "$first_line" | awk '{print $5}')
            TOP_P=$(echo "$first_line" | awk '{print $6}')
            REPETITION_PENALTY=$(echo "$first_line" | awk '{print $7}')
            ENABLE_REPETITION_PENALTY=$(echo "$first_line" | awk '{print $8}')
            PROMPT_TEXT=$(echo "$parse_result" | sed -n '/PROMPT_START/,/PROMPT_END/p' | sed '1d;$d' | head -n1)
        elif [[ "$parse_result" =~ ^SINGLE ]]; then
            local first_line
            first_line=$(echo "$parse_result" | head -n1)
            TOKEN_COUNT=$(echo "$first_line" | awk '{print $2}')
            TEMPERATURE=$(echo "$first_line" | awk '{print $3}')
            TOP_K=$(echo "$first_line" | awk '{print $4}')
            TOP_P=$(echo "$first_line" | awk '{print $5}')
            REPETITION_PENALTY=$(echo "$first_line" | awk '{print $6}')
            ENABLE_REPETITION_PENALTY=$(echo "$first_line" | awk '{print $7}')
            PROMPT_TEXT=$(echo "$parse_result" | sed -n '/PROMPT_START/,/PROMPT_END/p' | sed '1d;$d')
        else
            error "Invalid JSON parse result format."
        fi
    else
        banner "Parsing legacy text prompt file"
        local first_line
        IFS= read -r first_line < "${PROMPT_FILE}"
        if [[ ! "${first_line}" =~ ^([0-9]+),"(.*)"$ ]]; then
            error "Legacy prompt-file line format invalid."
        fi
        TOKEN_COUNT="${BASH_REMATCH[1]}"
        PROMPT_TEXT="${BASH_REMATCH[2]}"
        TEMPERATURE=0.8
        TOP_K=40
        TOP_P=0.9
        REPETITION_PENALTY=1.0
        ENABLE_REPETITION_PENALTY=false
    fi
}

# =========================================================================== #
# 4. Main Execution                                                           #
# =========================================================================== #
main() {
    parse_prompt_file

    banner "Prompt & Hyperparameters"
    log "Using prompt (${TOKEN_COUNT} tokens): ${PROMPT_TEXT}"
    log "Hyperparameters: temp=${TEMPERATURE}, top_k=${TOP_K}, top_p=${TOP_P}, rep_penalty=${REPETITION_PENALTY}, enable_rep=${ENABLE_REPETITION_PENALTY}"

    clear_caches

    local CMD=(
      sudo "${BIN}"
      --tflite_model="${MODEL_DIR}/${MODEL_NAME}.tflite"
      --sentencepiece_model="${MODEL_DIR}/tokenizer.model"
      --max_decode_steps="${TOKEN_COUNT}"
      --start_token="<bos>"
      --stop_token="<eos>"
      --num_threads=1
      --prompt="${PROMPT_TEXT}"
      --weight_cache_path="${MODEL_DIR}/${MODEL_NAME}.xnnpack_cache"
      --temperature="${TEMPERATURE}"
      --top_k="${TOP_K}"
      --top_p="${TOP_P}"
      --repetition_penalty="${REPETITION_PENALTY}"
    )
    if [[ "${ENABLE_REPETITION_PENALTY}" == "true" ]]; then
      CMD+=(--enable_repetition_penalty)
    fi

    banner "Running Inference"
    execute_with_log "${CMD[@]}"
    banner "Run Complete"
}

main "$@"
