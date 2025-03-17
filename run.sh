#!/bin/bash

################ Load utils ################
source ./util/utils.sh

################ Setup environment ################
source .env

##################################
# Default values
LOG_ENABLED=false
CORE_LIST="all"  # 기본값: 모든 코어 사용 (taskset 사용 안 함)
NUM_THREADS=1    # 기본값: 1개 스레드 사용
FILE="./${PROMPT_PATH}/sample_prompt_8_1.txt"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--log) LOG_ENABLED=true ;;
        -c|--core) CORE_LIST="$2"; shift ;;
        -t|--threads) NUM_THREADS="$2"; shift ;;
        -file|--file) FILE="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [-l|--log] [-c|--core CORES] [-t|--threads NUM_THREADS]"
            echo "  -l, --log        Enable logging (default: disabled)"
            echo "  -c, --core CORES Set CPU cores for execution (default: all cores)"
            echo "  -t, --threads NUM_THREADS Set number of threads (default: 1)"
            echo "  -file, --file FILE Set the input file (default: sample_prompt_8_1.txt)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

log() {
    echo "$@" | tee -a "$OUTPUT_FILE"
}

##################################

# Llama 3.2 3B INT8 quantized
MODEL_PATH="${MODEL_PATH}/llama-3.2-3b-it-q8"
MODEL_NAME="llama_q8_ekv1024"



if [ ! -f "$FILE" ]; then
    echo "Error: 파일 '$FILE'이 존재하지 않습니다."
    exit 1
fi

RESULTS_DIR="result_run_once"
if [ ! -d "$RESULTS_DIR" ]; then
    mkdir "$RESULTS_DIR"
fi



################ Main Scripts ################
while read -r line; do
    if [[ "$line" =~ ^([0-9]+),\"(.*)\"$ ]]; then
        token_count="${BASH_REMATCH[1]}"
        prompt="${BASH_REMATCH[2]}"
        
        # token_count 값을 반영하여 로그 파일 동적으로 설정
        TIMESTAMP=$(date +"%y%m%d_%H%M%S")  # 타임스탬프 생성
        OUTPUT_FILE="${RESULTS_DIR}/output_${token_count}_${TIMESTAMP}.log"

        log "[INFO] Start LLM inference"
        log "[INFO] Model: ${MODEL_NAME}"
        log "[INFO] Using CPU Cores: ${CORE_LIST}"
        log "[INFO] Number of Threads: ${NUM_THREADS}"
        log "[INFO] Logging Enabled: ${LOG_ENABLED}"
        log "[INFO] Output File: ${OUTPUT_FILE}"

        log "[INFO] Dropping OS Page Caches.."
        clear_caches
        log "[INFO] Clearing CPU Caches"

        log "[INFO] Running inference for token count: $token_count"

        # taskset 적용 여부 결정
        CMD="./text_generator_main \
            --tflite_model='${MODEL_PATH}/${MODEL_NAME}.tflite' \
            --sentencepiece_model='${MODEL_PATH}/tokenizer.model' \
            --max_decode_steps=128 \
            --start_token='<bos>' \
            --stop_token='<eos>' \
            --num_threads='${NUM_THREADS}' \
            --prompt='${prompt}' \
            --weight_cache_path='${MODEL_PATH}/${MODEL_NAME}.xnnpack_cache'"

        if [ "$CORE_LIST" != "all" ]; then
            CMD="taskset -c ${CORE_LIST} ${CMD}"
        fi

        # Run the model and log everything
        if [ "$LOG_ENABLED" = true ]; then
            eval "sudo $CMD" 2>&1 | tee -a "$OUTPUT_FILE"
        else
            eval "sudo $CMD" 
        fi

        log "[INFO] Results saved in: $OUTPUT_FILE"
        log "==================================="
    fi
done <"$FILE"