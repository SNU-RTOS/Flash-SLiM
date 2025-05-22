#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# run_cgroup.sh – Memory‑constrained LLM inference benchmark (cgroup v2 / v1)
# ------------------------------------------------------------------------------
# • Runs each prompt under a given MemoryMax (CGROUP_MMAX array)
# • Supports GPU / CPU binaries (name pattern: output/text_generator_main_<target>)
# • Auto‑detects cgroup v2 (systemd‑run) or falls back to cgroup v1 (cgexec)
# ------------------------------------------------------------------------------

set -euo pipefail

# ------------------------------------------------------------------------------
# 0. Load environment and helpers
# ------------------------------------------------------------------------------
source .env               # contains PROMPT_PATH, MODEL_PATH, …
source ./scripts/utils.sh # provides clear_caches()

# ------------------------------------------------------------------------------
# 1. Defaults & CLI parsing
# ------------------------------------------------------------------------------
TARGET="cpu" # cpu | gpu
LOG_ENABLED=false
CORE_LIST="all" # taskset core list
NUM_THREADS=1
PROMPT_FILE="./${PROMPT_PATH}/sample_prompt_8_1.txt"
MAX_TOK_LEN=16
NUM_REPEATS=1

usage() {
    cat <<EOF
Usage: run_cgroup.sh [OPTIONS]

Run LLM inference benchmarks with per‑process memory limits.

Options:
  -g, --target <gpu|cpu>   Select binary to run (default: cpu)
  -l, --log                Enable logging to ./result_<mem>/output_*.log
  -c, --core <CORES>       Bind to CPU core list (e.g. "0-3" or "1,3")
  -t, --threads <N>        Threads per run (default: 1)
  -f, --file <PATH>        Prompt CSV file (default: \$PROMPT_FILE)
  --tl, --max_tokens <N>   Max tokens to generate (default: 16)
  -r, --repeat <N>         Repeat all prompts N times (default: 1)
  -h, --help               Show this help and exit

Examples:
  ./run_cgroup.sh --target gpu --log --core 0-3 --threads 4 --repeat 3
  ./run_cgroup.sh -f prompts/sample_prompt_8_10.txt --tl 32
EOF
    exit 0
}

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
    -r | --repeat)
        NUM_REPEATS="$2"
        shift 2
        ;;
    -h | --help) usage ;;
    *)
        echo "[ERROR] Unknown option: $1" >&2
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
if [ "$LOG_ENABLED" = true ]; then
    log() { echo "$@" | tee -a "$OUTPUT_FILE"; }
fi

# ------------------------------------------------------------------------------
# 3. Model configuration
# ------------------------------------------------------------------------------
MODEL_DIR="${MODEL_PATH}/llama-3.2-3b-it-q8"
MODEL_NAME="llama_q8_ekv1024"
BIN="output/text_generator_main_${TARGET}"

[[ -x "$BIN" ]] || {
    echo "[ERROR] Binary not found: $BIN"
    exit 1
}
[[ -f "$PROMPT_FILE" ]] || {
    echo "[ERROR] Prompt file not found: $PROMPT_FILE"
    exit 1
}

if [[ "$PROMPT_FILE" =~ ^.*_.*_([0-9]+)\.txt$ ]]; then
    PROMPT_ITEM_SIZE="${BASH_REMATCH[1]}"
else
    echo "[ERROR] Prompt filename must match *_*_<N>.txt"
    exit 1
fi

# ------------------------------------------------------------------------------
# 4. Memory scenarios to test (edit as needed)
# ------------------------------------------------------------------------------
CGROUP_MMAX=(
    # "512M"
    # "1G"
    "2G"
    )

# ------------------------------------------------------------------------------
# 5. cgroup execution helper
# ------------------------------------------------------------------------------
run_with_memlimit() {
    local mmax="$1"
    shift
    local cmd=("$@")

    if mount | grep -q "cgroup2"; then
        systemd-run --quiet --scope \
            -p MemoryMax=$mmax -p MemoryHigh=$mmax -p MemorySwapMax=0 \
            -- "${cmd[@]}"
    else
        local cg="/sys/fs/cgroup/memory/llmbench"
        mkdir -p "$cg"
        echo 0 >"$cg/memory.force_empty" || true
        echo "$(($(numfmt --from=iec $mmax)))" >"$cg/memory.limit_in_bytes"
        cgexec -g memory:llmbench -- "${cmd[@]}"
    fi
}

# ------------------------------------------------------------------------------
# 6. Main loop
# ------------------------------------------------------------------------------
for MMAX in "${CGROUP_MMAX[@]}"; do
    RESULTS_DIR="result_${MMAX}"
    mkdir -p "$RESULTS_DIR"
    prompt_id=1
    echo "[INFO] === Memory Limit: $MMAX ==="

    for ((iter = 1; iter <= NUM_REPEATS; iter++)); do
        echo "[INFO] --- Iteration $iter / $NUM_REPEATS ---"

        while IFS= read -r line; do
            [[ "$line" =~ ^([0-9]+),\"(.*)\"$ ]] || continue
            token_count="${BASH_REMATCH[1]}"
            prompt="${BASH_REMATCH[2]}"

            OUTPUT_FILE="${RESULTS_DIR}/output_${token_count}_$((iter * PROMPT_ITEM_SIZE + prompt_id)).log"

            log "[INFO] Prompt #${prompt_id} | Tokens: $token_count"
            log "[INFO] CPU cores: ${CORE_LIST} | Threads: ${NUM_THREADS} | Target: ${TARGET}"
            clear_caches

            cmd=("$BIN"
                --tflite_model "${MODEL_DIR}/${MODEL_NAME}.tflite"
                --sentencepiece_model "${MODEL_DIR}/tokenizer.model"
                --max_decode_steps "${MAX_TOK_LEN}"
                --start_token "<bos>"
                --stop_token "<eos>"
                --num_threads "${NUM_THREADS}"
                --prompt "${prompt}"
                --weight_cache_path "${MODEL_DIR}/${MODEL_NAME}.xnnpack_cache")

            [[ "$CORE_LIST" != "all" ]] && cmd=(taskset -c "$CORE_LIST" "${cmd[@]}")

            if [ "$LOG_ENABLED" = true ]; then
                run_with_memlimit "$MMAX" "${cmd[@]}" 2>&1 | tee -a "$OUTPUT_FILE"
            else
                run_with_memlimit "$MMAX" "${cmd[@]}"
            fi

            ((prompt_id++))
            ((prompt_id > PROMPT_ITEM_SIZE)) && prompt_id=1
        done <"$PROMPT_FILE"
    done

done
