#!/bin/bash

# ──────────────────────────────────────────────────────────────────────────────
source common.sh
cd ..
source .env

# ── Build Configuration ───────────────────────────────────────────────────────
BUILD_MODE=${1:-release}
setup_build_config "$BUILD_MODE"

# ── Paths ─────────────────────────────────────────────────────────────────────
BENCHMARK_TOOL_PATH=${LITERT_PATH}/bazel-bin/tflite/tools/benchmark/benchmark_model
BENCHMARK_DEST_PATH=${ROOT_PATH}/benchmark_model

# ── Clean existing binary ─────────────────────────────────────────────────────
if [ -f "${BENCHMARK_DEST_PATH}" ]; then
    rm "${BENCHMARK_DEST_PATH}"
fi

# ── Build LiteRT benchmark tool ───────────────────────────────────────────────
echo "[INFO] Build LiteRT benchmark tool ($BUILD_MODE mode) with GPU delegate support…"
echo "[INFO] Path: ${BENCHMARK_TOOL_PATH}"
echo "[INFO] GPU Flags: ${GPU_FLAGS}"

cd "${LITERT_PATH}" || exit 1
pwd

# GPU delegate 의존성 문제를 해결하기 위한 추가 플래그
bazel build ${BAZEL_CONF} \
    //tflite/tools/benchmark:benchmark_model \
    ${GPU_FLAGS} \
    ${COPT_FLAGS} \
    ${GPU_COPT_FLAGS} \
    ${LINKOPTS} \
    --verbose_failures \
    --define=TFLITE_SUPPORTS_GPU_DELEGATE=1

# ── Copy binary ───────────────────────────────────────────────────────────────
echo "[INFO] Copy benchmark tool to project root…"
cp "${BENCHMARK_TOOL_PATH}" "${BENCHMARK_DEST_PATH}"

if [ -f "${BENCHMARK_DEST_PATH}" ]; then
    echo "✅ Successfully built benchmark tool with GPU delegate support: ${BENCHMARK_DEST_PATH}"
    echo "[INFO] GPU delegate and invoke loop support enabled"
    echo "[INFO] Available flags: --use_gpu, --gpu_invoke_loop_times, --gpu_backend"
else
    echo "❌ Failed to build benchmark tool"
    exit 1
fi

cd "${ROOT_PATH}/scripts"
pwd