#!/bin/bash

# 벤치마크 바이너리 경로 (수정 가능)
OPS_BENCHMARK_BIN="./benchmark_model_aarch64"

# 결과 저장 디렉토리
OPS_RESULT_DIR="./benchmark_ops_results"
if [ ! -d ${OPS_RESULT_DIR} ]; then
    mkdir ${OPS_RESULT_DIR}
fi

# 모델 디렉토리
MODEL_ROOT_PATH="/root/models/split"
if [ ! -d ${MODEL_ROOT_PATH} ]; then
    echo "[INFO] INVAILD MODEL_ROOT_PATH"
    exit
fi

for model_path in "$MODEL_ROOT_PATH"/*.tflite; do
    model_name=$(basename "$model_path" .tflite)

    echo "Benchmarking model: $model_name"

    # 단일 쓰레드 실행, XNNPACK 사용
    taskset -c 0 "$OPS_BENCHMARK_BIN" \
        --graph="$model_path" \
        --num_threads=1 \
        --enable_op_profiling=true \
        --use_xnnpack=true \
        --report_peak_memory_footprint=true \
        --op_profiling_output_mode=csv \
        --op_profiling_output_file="${OPS_RESULT_DIR}/${model_name}_single_thread_xnn.csv" \
        >${OPS_RESULT_DIR}/${model_name}_single_thread_xnn.log

    echo "Saved multi thread result: ${OPS_RESULT_DIR}/${model_name}_multi_thread.csv"
done

echo "Benchmarking completed. Results are saved in $OPS_RESULT_DIR."
