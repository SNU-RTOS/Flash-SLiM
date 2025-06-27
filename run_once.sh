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
# MODEL_DIR="${MODEL_PATH}/Gemma2-2B-it"
# MODEL_NAME="Gemma2-2B-IT_multi-prefill-seq_q8_ekv1280"

MODEL_DIR="${MODEL_PATH}/gemma3-1b-it-int4"
MODEL_NAME="gemma3"

PROMPT_FILE="./${PROMPT_PATH}/sample_prompt_8_1.txt"
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: prompt file '${PROMPT_FILE}' does not exist." >&2
  exit 3
fi

# Grab the first valid prompt line:  <token_count>,"<prompt text>"
IFS= read -r FIRST_LINE < "${PROMPT_FILE}"
if [[ ! "${FIRST_LINE}" =~ ^([0-9]+),\"(.*)\"$ ]]; then
  echo "Error: prompt-file line format invalid." >&2
  exit 4
fi

TOKEN_COUNT="${BASH_REMATCH[1]}"
PROMPT_TEXT="${BASH_REMATCH[2]}"

echo "[INFO] Using prompt (${TOKEN_COUNT} tokens): ${PROMPT_TEXT}"

# ---------------------------------------------------------------------------
# 3. Clear caches
# ---------------------------------------------------------------------------
echo "[INFO] Dropping OS page cache..."
clear_caches
echo "[INFO] CPU caches cleared."

# ---------------------------------------------------------------------------
# 4. Run once
# ---------------------------------------------------------------------------
sudo "./${BIN}" \
  --tflite_model="${MODEL_DIR}/${MODEL_NAME}.tflite" \
  --sentencepiece_model="${MODEL_DIR}/tokenizer.model" \
  --max_decode_steps=32 \
  --start_token="<bos>" \
  --stop_token="<eos>" \
  --num_threads=1 \
  --prompt="${PROMPT_TEXT}" \
  --weight_cache_path="${MODEL_DIR}/${MODEL_NAME}.xnnpack_cache"
