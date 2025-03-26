#!/bin/bash

################ Load utils ################
source ./util/utils.sh

log() {
    echo "$@" | tee -a "$OUTPUT_FILE"
}

################ Setup Environment ################
source .env

################ CLI Parsing ################
# Default values
LOG_ENABLED=false
CORE_LIST="all" # 기본값: 모든 코어 사용 (taskset 사용 안 함)
NUM_THREADS=1   # 기본값: 1개 스레드 사용
FILE="./${PROMPT_PATH}/sample_prompt_8_1.txt"
MAX_TOKEN_LENGHT_TO_GENERATE=16
NUM_REPEATS=1

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
    -l | --log) LOG_ENABLED=true ;;
    -c | --core)
        CORE_LIST="$2"
        shift
        ;;
    -t | --threads)
        NUM_THREADS="$2"
        shift
        ;;
    -i | --input_prompt_path)
        FILE="$2"
        shift
        ;;
    -mtl | --max_token_length)
        MAX_TOKEN_LENGHT_TO_GENERATE="$2"
        shift
        ;;
    -r | --repeat)
        NUM_REPEATS="$2"
        shift
        ;;
    -h | --help)
        echo "Usage: $0 [-l|--log] [-c|--core CORES] [-t|--threads NUM_THREADS] [-i|--input_prompt_path PATH] [-mtl|--max_token_length NUM] [-r|--repeat NUM]"
        echo "  -l, --log                   Enable logging (default: disabled)"
        echo "  -c, --core                  CORES CPU cores for execution (default: all cores)"
        echo "  -t, --threads               NUM_THREADS number of threads (default: 1)"
        echo "  -i, --input_prompt_path     PATH Path of input prompt file (default: sample_prompt_8_1.txt)"
        echo "  -mtl, --max_token_length    NUM Max_token_length to generate (default: 16)"
        echo "  -r, --repeat                NUM Number of repeat (default: 1)"
        exit 0
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
done

################ Model Info ################
# Llama 3.2 3B INT8 quantized
MODEL_PATH="${MODEL_PATH}/llama-3.2-3b-it-q8"
MODEL_NAME="llama_q8_ekv1024"

if [ ! -f "$FILE" ]; then
    echo "'$FILE' no exists"
    exit 1
fi

################ Check number of prompts in given .txt file ################
if [[ "$FILE" =~ ^.*_.*_([0-9]+)\.txt$ ]]; then
    PROMPT_ITEM_SIZE="${BASH_REMATCH[1]}"
    # echo "Number of prompts: $PROMPT_ITEM_SIZE in $FILE"
else
    echo "Please check the name of prompt file. '[ANY NAME]_[ANY NAME]_[number of prompts].txt' "
    exit 1
fi

################ Set CGROUP ################
CGROUP_MMAX=(
    "8G"
    # "4G"
    # "2G"
    # "1G"
    # "512M"
    #"256M"
)
# Cleanup existing cgroup if it exists
if [ -d "/sys/fs/cgroup/memory/mygroup" ]; then
    # Kill any processes still in the cgroup
    kill -9 $(cat /sys/fs/cgroup/memory/mygroup/tasks 2>/dev/null) 2>/dev/null
    # Remove the cgroup
    rmdir /sys/fs/cgroup/memory/mygroup 2>/dev/null
fi

mkdir -p /sys/fs/cgroup/memory/mygroup

################ Main Scripts ################
for MMAX in "${CGROUP_MMAX[@]}"; do
    RESULTS_DIR="result_${MMAX}"

    if [ ! -d "$RESULTS_DIR" ]; then
        mkdir "$RESULTS_DIR"
    fi

    # Set memory limit of the cgroup
    echo $MMAX >/sys/fs/cgroup/memory/mygroup/memory.limit_in_bytes
    echo 0 >/sys/fs/cgroup/memory/mygroup/memory.force_empty

    prompt_id=1
    echo "======== Current Allocatable Memory Size: $MMAX ======="

    for ((i = 1; i <= NUM_REPEATS; i++)); do
        echo "========== Iteration $i/$NUM_REPEATS =========="

        while read -r line; do
            if [[ "$line" =~ ^([0-9]+),\"(.*)\"$ ]]; then
                token_count="${BASH_REMATCH[1]}"
                prompt="${BASH_REMATCH[2]}"

                # token_count 값을 반영하여 로그 파일 동적으로 설정
                # TIMESTAMP=$(date +"%y%m%d_%H%M%S")  # 타임스탬프 생성
                OUTPUT_FILE="${RESULTS_DIR}/output_${token_count}_$((${i} * ${PROMPT_ITEM_SIZE} + ${prompt_id})).log"

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
                    --max_decode_steps=${MAX_TOKEN_LENGHT_TO_GENERATE} \
                    --start_token='<bos>' \
                    --stop_token='<eos>' \
                    --num_threads='${NUM_THREADS}' \
                    --prompt='${prompt}' \
                    --weight_cache_path='${MODEL_PATH}/${MODEL_NAME}.xnnpack_cache'"

                if [ "$CORE_LIST" != "all" ]; then
                    CMD="cgexec -g memory:mygroup taskset -c ${CORE_LIST} ${CMD}"
                fi

                # Run the model and log everything
                if [ "$LOG_ENABLED" = true ]; then
                    eval "sudo $CMD" 2>&1 | tee -a "$OUTPUT_FILE"
                    log "[INFO] Results saved in: $OUTPUT_FILE"
                else
                    eval "sudo $CMD"
                fi
                log "==================================="

            fi

            prompt_id=$((prompt_id + 1))
            if ((prompt_id > ${PROMPT_ITEM_SIZE})); then
                prompt_id=1
            fi

        done <"$FILE"
    done
done
