#!/usr/bin/env bash
# run_once.sh
#
#   ./run_once.sh            # run GPU binary (default) once
#   ./run_once.sh cpu        # run CPU binary once
#   ./run_once.sh gpu        # explicit GPU run
#
# Prerequisites:
#   * build.sh must have produced  output/text_generator_main_gpu  and/or
#     output/text_generator_main_cpu.
#   * .env defines ROOT_PATH, MODEL_PATH, PROMPT_PATH, etc.
#   * scripts/utils.sh provides clear_caches()

set -euo pipefail

################ Load utils ################
source ./scripts/utils.sh

################ Setup environment ################
source .env

# ---------------------------------------------------------------------------
# 1. Choose target (gpu|cpu|qnn)
# ---------------------------------------------------------------------------
TARGET="${1:-gpu}"          # default = gpu

case "${TARGET}" in
  gpu|cpu|qnn) ;;
  *)
    echo "Usage: $0 [gpu|cpu|qnn]" >&2
    exit 1
    ;;
esac

BIN="output/text_generator_main_${TARGET}"
if [[ ! -x "${BIN}" ]]; then
  echo "Error: ${BIN} not found or not executable. Build it first!" >&2
  exit 2
fi

echo "[INFO] Selected binary  : ${BIN}"
echo "[INFO] Target accelerator: ${TARGET^^}"

# ---------------------------------------------------------------------------
# 2. Model & prompt settings
# ---------------------------------------------------------------------------
# MODEL_DIR="${MODEL_PATH}/llama-3.2-3b-it-q8"
# MODEL_NAME="llama_q8_ekv1024"

MODEL_DIR="${MODEL_PATH}/Gemma3-1B"
MODEL_NAME="model.q8"

# MODEL_DIR="${MODEL_PATH}/SmolLM"
# MODEL_NAME="model.q8"

PROMPT_FILE="./${PROMPT_PATH}/sample_prompt_8_1.json"
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: prompt file '${PROMPT_FILE}' does not exist." >&2
  exit 3
fi

# Check if it's JSON format and parse accordingly
FILE_CONTENT=$(cat "${PROMPT_FILE}")
if [[ "$FILE_CONTENT" =~ ^\s*\[ || "$FILE_CONTENT" =~ ^\s*\{ ]]; then
  echo "[INFO] Detected JSON format, parsing..."
  
  # Use Python script for JSON parsing
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: Python3 is required for JSON parsing but not installed." >&2
    exit 4
  fi
  
  PARSER_SCRIPT="./scripts/parse_json_prompt.py"
  if [[ ! -f "$PARSER_SCRIPT" ]]; then
    echo "Error: JSON parser script not found: $PARSER_SCRIPT" >&2
    exit 4
  fi
  
  # Parse JSON and get first prompt
  PARSE_RESULT=$(python3 "$PARSER_SCRIPT" "$PROMPT_FILE")
  if [[ $? -ne 0 ]]; then
    echo "Error: Failed to parse JSON file" >&2
    exit 4
  fi
  
  # Extract first prompt data
  if [[ "$PARSE_RESULT" =~ ^ARRAY ]]; then
    # Get first item from array
    FIRST_LINE=$(echo "$PARSE_RESULT" | grep "^ITEM 0" | head -n1)
    TOKEN_COUNT=$(echo "$FIRST_LINE" | awk '{print $3}')
    TEMPERATURE=$(echo "$FIRST_LINE" | awk '{print $4}')
    TOP_K=$(echo "$FIRST_LINE" | awk '{print $5}')
    TOP_P=$(echo "$FIRST_LINE" | awk '{print $6}')
    REPETITION_PENALTY=$(echo "$FIRST_LINE" | awk '{print $7}')
    ENABLE_REPETITION_PENALTY=$(echo "$FIRST_LINE" | awk '{print $8}')
    
    # Extract prompt text
    PROMPT_TEXT=$(echo "$PARSE_RESULT" | sed -n '/PROMPT_START/,/PROMPT_END/p' | sed '1d;$d' | head -n1)
  elif [[ "$PARSE_RESULT" =~ ^SINGLE ]]; then
    # Single prompt
    FIRST_LINE=$(echo "$PARSE_RESULT" | head -n1)
    TOKEN_COUNT=$(echo "$FIRST_LINE" | awk '{print $2}')
    TEMPERATURE=$(echo "$FIRST_LINE" | awk '{print $3}')
    TOP_K=$(echo "$FIRST_LINE" | awk '{print $4}')
    TOP_P=$(echo "$FIRST_LINE" | awk '{print $5}')
    REPETITION_PENALTY=$(echo "$FIRST_LINE" | awk '{print $6}')
    ENABLE_REPETITION_PENALTY=$(echo "$FIRST_LINE" | awk '{print $7}')
    
    # Extract prompt text
    PROMPT_TEXT=$(echo "$PARSE_RESULT" | sed -n '/PROMPT_START/,/PROMPT_END/p' | sed '1d;$d')
  else
    echo "Error: Invalid JSON parse result" >&2
    exit 4
  fi
  
else
  # Legacy text format: <token_count>,"<prompt text>"
  echo "[INFO] Detected legacy text format, parsing..."
  IFS= read -r FIRST_LINE < "${PROMPT_FILE}"
  if [[ ! "${FIRST_LINE}" =~ ^([0-9]+),\"(.*)\"$ ]]; then
    echo "Error: prompt-file line format invalid." >&2
    exit 4
  fi
  
  TOKEN_COUNT="${BASH_REMATCH[1]}"
  PROMPT_TEXT="${BASH_REMATCH[2]}"
  # Default hyperparameters for legacy format
  TEMPERATURE=0.8
  TOP_K=40
  TOP_P=0.9
  REPETITION_PENALTY=1.0
  ENABLE_REPETITION_PENALTY=false
fi

echo "[INFO] Using prompt (${TOKEN_COUNT} tokens): ${PROMPT_TEXT}"
echo "[INFO] Hyperparameters: temp=${TEMPERATURE}, top_k=${TOP_K}, top_p=${TOP_P}, rep_penalty=${REPETITION_PENALTY}, enable_rep=${ENABLE_REPETITION_PENALTY}"

# ---------------------------------------------------------------------------
# 3. Clear caches
# ---------------------------------------------------------------------------
echo "[INFO] Dropping OS page cache..."
clear_caches
echo "[INFO] CPU caches cleared."

# ---------------------------------------------------------------------------
# 4. Run once
# ---------------------------------------------------------------------------
# Build command with hyperparameters
CMD=(
  sudo "./${BIN}"
  --tflite_model="${MODEL_DIR}/${MODEL_NAME}.tflite"
  --sentencepiece_model="${MODEL_DIR}/tokenizer.model"
  --max_decode_steps=32
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

# Add repetition penalty flag if enabled
if [[ "${ENABLE_REPETITION_PENALTY}" == "true" ]]; then
  CMD+=(--enable_repetition_penalty)
fi

# Execute the command
"${CMD[@]}"
