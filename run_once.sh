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

# --------------------------------------------------------------------------- #
# 0. Load environment and helpers                                            #
# --------------------------------------------------------------------------- #
if [[ ! -f .env ]]; then
    echo "[ERROR] .env file not found. Please run from project root directory." >&2
    exit 1
fi

source .env
source ./scripts/utils.sh

banner "Single prompt run"

# ---------------------------------------------------------------------------
# 1. Choose target (cpu|gpu|qnn)
# ---------------------------------------------------------------------------
TARGET="${1:-cpu}"
case "${TARGET}" in
  cpu|gpu) ;;
  *)
    banner "Invalid target argument"
    echo "Usage: $0 [cpu|gpu]" >&2
    exit 1
    ;;
esac

BIN="output/text_generator_main"
if [[ ! -x "${BIN}" ]]; then
  banner "Binary missing"
  echo "[ERROR] ${BIN} not found or not executable. Build it first!" >&2
  exit 2
fi

banner "Selected binary: ${BIN}"
echo "[INFO] Target accelerator: ${TARGET^^}"

# ---------------------------------------------------------------------------
# 2. Model & prompt settings (unified Bazel logic)
# ---------------------------------------------------------------------------
# Default: llama-3.2-3b-it-q8
MODEL_DIR="${MODEL_PATH}/llama-3.2-3b-it-q8"
MODEL_NAME="llama_q8_ekv1024"
# If you want to use Gemma3-1B or SmolLM, uncomment below
# MODEL_DIR="${MODEL_PATH}/Gemma3-1B"
# MODEL_NAME="model.q8"
# MODEL_DIR="${MODEL_PATH}/SmolLM"
# MODEL_NAME="model.q8"

PROMPT_FILE="./${PROMPT_PATH}/sample_prompt_8_1.json"
RESULTS_DIR="result_run_once"
ensure_dir "${RESULTS_DIR}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  banner "Prompt file missing"
  echo "[ERROR] Prompt file '${PROMPT_FILE}' does not exist." >&2
  exit 3
fi

# Check if it's JSON format and parse accordingly
FILE_CONTENT=$(cat "${PROMPT_FILE}")
if [[ "$FILE_CONTENT" =~ ^\s*\[ || "$FILE_CONTENT" =~ ^\s*\{ ]]; then
  banner "Parsing JSON prompt file"
  if ! command -v python3 >/dev/null 2>&1; then
    banner "Python3 missing"
    echo "[ERROR] Python3 is required for JSON parsing but not installed." >&2
    exit 4
  fi
  PARSER_SCRIPT="./scripts/parse_json_prompt.py"
  if [[ ! -f "$PARSER_SCRIPT" ]]; then
    banner "JSON parser missing"
    echo "[ERROR] JSON parser script not found: $PARSER_SCRIPT" >&2
    exit 4
  fi
  PARSE_RESULT=$(python3 "$PARSER_SCRIPT" "$PROMPT_FILE")
  if [[ $? -ne 0 ]]; then
    banner "JSON parse failed"
    echo "[ERROR] Failed to parse JSON file" >&2
    exit 4
  fi
  if [[ "$PARSE_RESULT" =~ ^ARRAY ]]; then
    FIRST_LINE=$(echo "$PARSE_RESULT" | grep "^ITEM 0" | head -n1)
    TOKEN_COUNT=$(echo "$FIRST_LINE" | awk '{print $3}')
    TEMPERATURE=$(echo "$FIRST_LINE" | awk '{print $4}')
    TOP_K=$(echo "$FIRST_LINE" | awk '{print $5}')
    TOP_P=$(echo "$FIRST_LINE" | awk '{print $6}')
    REPETITION_PENALTY=$(echo "$FIRST_LINE" | awk '{print $7}')
    ENABLE_REPETITION_PENALTY=$(echo "$FIRST_LINE" | awk '{print $8}')
    PROMPT_TEXT=$(echo "$PARSE_RESULT" | sed -n '/PROMPT_START/,/PROMPT_END/p' | sed '1d;$d' | head -n1)
  elif [[ "$PARSE_RESULT" =~ ^SINGLE ]]; then
    FIRST_LINE=$(echo "$PARSE_RESULT" | head -n1)
    TOKEN_COUNT=$(echo "$FIRST_LINE" | awk '{print $2}')
    TEMPERATURE=$(echo "$FIRST_LINE" | awk '{print $3}')
    TOP_K=$(echo "$FIRST_LINE" | awk '{print $4}')
    TOP_P=$(echo "$FIRST_LINE" | awk '{print $5}')
    REPETITION_PENALTY=$(echo "$FIRST_LINE" | awk '{print $6}')
    ENABLE_REPETITION_PENALTY=$(echo "$FIRST_LINE" | awk '{print $7}')
    PROMPT_TEXT=$(echo "$PARSE_RESULT" | sed -n '/PROMPT_START/,/PROMPT_END/p' | sed '1d;$d')
  else
    banner "JSON parse result invalid"
    echo "[ERROR] Invalid JSON parse result" >&2
    exit 4
  fi
else
  # Legacy text format: <token_count>,"<prompt text>"
  banner "Parsing legacy text prompt file"
  IFS= read -r FIRST_LINE < "${PROMPT_FILE}"
  if [[ ! "${FIRST_LINE}" =~ ^([0-9]+),"(.*)"$ ]]; then
    banner "Legacy prompt format invalid"
    echo "[ERROR] prompt-file line format invalid." >&2
    exit 4
  fi
  TOKEN_COUNT="${BASH_REMATCH[1]}"
  PROMPT_TEXT="${BASH_REMATCH[2]}"
  TEMPERATURE=0.8
  TOP_K=40
  TOP_P=0.9
  REPETITION_PENALTY=1.0
  ENABLE_REPETITION_PENALTY=false
fi

banner "Prompt & Hyperparameters"
echo "[INFO] Using prompt (${TOKEN_COUNT} tokens): ${PROMPT_TEXT}"
echo "[INFO] Hyperparameters: temp=${TEMPERATURE}, top_k=${TOP_K}, top_p=${TOP_P}, rep_penalty=${REPETITION_PENALTY}, enable_rep=${ENABLE_REPETITION_PENALTY}"

# ---------------------------------------------------------------------------
# 3. Clear caches
# ---------------------------------------------------------------------------
banner "Dropping OS page cache"
clear_caches
banner "CPU caches cleared"

# ---------------------------------------------------------------------------
# 4. Run once
# ---------------------------------------------------------------------------
CMD=(
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

banner "Running inference"
"${CMD[@]}" | tee "${RESULTS_DIR}/output.log"
banner "Run complete"
