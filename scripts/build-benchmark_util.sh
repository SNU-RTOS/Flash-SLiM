#!/bin/bash

# ──────────────────────────────────────────────────────────────────────────────
source utils.sh
# ──────────────────────────────────────────────────────────────────────────────
cd ..
source .env

# ── Build Configuration ───────────────────────────────────────────────────────
BUILD_MODE=${1:-release}
setup_build_config "$BUILD_MODE"

# ── Paths ─────────────────────────────────────────────────────────────────────
BENCHMARK_TOOL_BUILD_DIR=${ROOT_PATH}/bazel-bin/external/litert/tflite/tools/benchmark
BENCHMARK_TOOL_BUILD_BIN=${BENCHMARK_TOOL_BUILD_DIR}/benchmark_model
BENCHMARK_DIR=${ROOT_PATH}/util/bin
BENCHMARK_BIN=${BENCHMARK_DIR}/benchmark_model

# ── Clean existing binary ─────────────────────────────────────────────────────
if [ -f "${BENCHMARK_BIN}" ]; then
    rm "${BENCHMARK_BIN}"
fi

# ── Build LiteRT benchmark tool ───────────────────────────────────────────────
log "Build LiteRT benchmark tool ($BUILD_MODE mode) with GPU delegate support…"
log "Path: ${BENCHMARK_TOOL_BUILD_BIN}"
log "GPU Flags: ${GPU_FLAGS}"

cd "${ROOT_PATH}" || exit 1
pwd

bazel build ${BAZEL_CONF} \
    @litert//tflite/tools/benchmark:benchmark_model \
    ${GPU_FLAGS} \
    ${COPT_FLAGS} \
    ${GPU_COPT_FLAGS} \
    ${LINKOPTS} \
    --verbose_failures 

# ── Copy binary ───────────────────────────────────────────────────────────────
log "Copy benchmark tool to project root…"
ensure_dir "${BENCHMARK_DIR}"
cp "${BENCHMARK_TOOL_BUILD_BIN}" "${BENCHMARK_BIN}"

if [ -f "${BENCHMARK_BIN}" ]; then
    log "Successfully built benchmark tool with GPU delegate support: ${BENCHMARK_BIN}"
    log "GPU delegate and invoke loop support enabled"
    log "Available flags: --use_gpu, --gpu_invoke_loop_times, --gpu_backend"
else
    error "❌ Failed to build benchmark tool"
    exit 1
fi

cd "${ROOT_PATH}/scripts"
pwd