#!/usr/bin/env bash
# run_once.sh
#
# Options
#   -g, --target  gpu|cpu   Which binary to run           (default: gpu)
#   -l, --log              Enable log-to-file            (default: off)
#   -c, --core   LIST       taskset core list (e.g. 0,2) (default: all)
#   -t, --threads N         #threads for the binary      (default: 1)
#       --file    PATH      Prompt CSV file               (default: $PROMPT_PATH/sample_prompt_8_1.txt)
#       --tl      N         Max tokens to generate        (default: 10)
#   -h, --help              Show this help
#
# Result logs go to result_run_once/output_<tokens>_<timestamp>.log

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
LOG_ENABLED=true
CORE_LIST=all        # taskset core list
NUM_THREADS=1
PROMPT_FILE="./${PROMPT_PATH}/sample_prompt_8_1.txt"
MAX_TOK_LEN=16



usage() {
    cat <<'EOF'
    Usage: run_once.sh [OPTIONS]

    This script runs a single LLM inference pass using either a GPU or CPU binary,
    based on the specified model and prompt input.

    Options:
    -g, --target <gpu|cpu>   Select binary to run (default: cpu)

    -l, --log Enable logging to ./result_run_once directory. (default: on)

    -c, --core <CORES>
        Specify CPU core(s) to bind the process using 'taskset'.
        For example: "0-3" or "0,2". Default is to use all available cores.

    -t, --threads <N>
        Set the number of threads to use for inference.
        Default: 1

    -f, --file <PATH>
        Path to the input prompt file (CSV format: token_count,"prompt").
        Default: ./${PROMPT_PATH}/sample_prompt_8_1.txt

        Each line should look like:
            8,"Hello, how are you?"
        The token_count field is optional unless used for naming logs.
        ⚠️ Make sure the file exists before running.

    --tl, --max_tokens <N> Set the maximum number of tokens to generate. (default: 16)

    -h, --help Show this help message and exit.

    Examples:
        ./run_once.sh --target gpu --log --core 0-3 --threads 4
        ./run_once.sh -g cpu -f prompts.txt --tl 32
EOF
    exit 0
}

# --- parse args -------------------------------------------------------------

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
    -tl | --max_tokens)
        MAX_TOK_LEN="$2"
        shift 2
        ;;
    -h | --help) usage ;;
    *)
        echo "Unknown option: $1" >&2
        usage
        ;;
    esac
done

[[ "${TARGET}" =~ ^(gpu|cpu)$ ]] || {
    echo "Invalid --target ${TARGET}"
    exit 1
}


# ------------------------------------------------------------------------------
# 2. Logging helper (overridden if --log)
# ------------------------------------------------------------------------------
log() { echo "$@"; }
if $LOG_ENABLED; then
  log() { echo "$@" | tee -a "$OUTPUT_FILE"; }
fi

# --------------------------------------------------------------------------- #
# 3. Model, Binary and Prompt settings                                              #
# --------------------------------------------------------------------------- #
MODEL_DIR="${MODEL_PATH}/llama-3.2-3b-it-q8"
MODEL_NAME="llama_q8_ekv1024"
BIN="output/text_generator_main_${TARGET}"

[[ -x "$BIN" ]]       || { echo "[ERROR] Binary not found: $BIN"; exit 1; }
[[ -f "$PROMPT_FILE" ]]|| { echo "[ERROR] Prompt file not found: $PROMPT_FILE"; exit 1; }

if [[ "$PROMPT_FILE" =~ ^.*_.*_([0-9]+)\.txt$ ]]; then
  PROMPT_ITEM_SIZE="${BASH_REMATCH[1]}"
else
  echo "[ERROR] Prompt filename must match *_*_<N>.txt"; exit 1
fi

# --------------------------------------------------------------------------- #
# 5. Main loop 
# --------------------------------------------------------------------------- #
RESULTS_DIR="result_run_once"
if [[ -d "${RESULTS_DIR}" ]]; then
    echo "Warning: ${RESULTS_DIR} already exists. Results will be appended."
else
    echo "Creating results directory: ${RESULTS_DIR}"
fi


while IFS= read -r line; do
    [[ "$line" =~ ^([0-9]+),\"(.*)\"$ ]] || continue
    TOKENS="${BASH_REMATCH[1]}"
    PROMPT="${BASH_REMATCH[2]}"

    TIMESTAMP=$(date +'%y%m%d_%H%M%S')
    OUTPUT_FILE="${RESULTS_DIR}/output_${TOKENS}_${TIMESTAMP}.log"

    log "------ LLM inference start (${TARGET^^}) ------"
    log "Model            : ${MODEL_NAME}"
    log "Tokens requested : ${TOKENS}"
    log "Prompt           : ${PROMPT}"
    log "Cores            : ${CORE_LIST}"
    log "Threads          : ${NUM_THREADS}"
    log "Max tokens       : ${MAX_TOK_LEN}"
    log "Target Processor : ${TARGET^^}"
    log "Log file         : ${OUTPUT_FILE}"
    log "-----------------------------------------------"
    log "Cache clear ..."
    clear_caches

    # Build command as array (safe quoting)
    CMD=(
        sudo "${BIN}"
        --tflite_model "${MODEL_DIR}/${MODEL_NAME}.tflite"
        --sentencepiece_model "${MODEL_DIR}/tokenizer.model"
        --max_decode_steps "${MAX_TOK_LEN}"
        --start_token "<bos>"
        --stop_token "<eos>"
        --num_threads "${NUM_THREADS}"
        --prompt "${PROMPT}"
        --weight_cache_path "${MODEL_DIR}/${MODEL_NAME}.xnnpack_cache"
    )
    [[ "${CORE_LIST}" != "all" ]] && CMD=(taskset -c "${CORE_LIST}" "${CMD[@]}")

    if $LOG_ENABLED; then
        "${CMD[@]}" 2>&1 | tee -a "${OUTPUT_FILE}"
        log "Results saved to ${OUTPUT_FILE}"
    else
        "${CMD[@]}"
    fi
    log "-----------------------------------------------"
done <"${PROMPT_FILE}"

exit 0
