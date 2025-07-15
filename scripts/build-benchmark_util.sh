#!/bin/bash

# ──────────────────────────────────────────────────────────────────────────────
source common.sh
cd ..
source .env

# ── Build Configuration ───────────────────────────────────────────────────────
BUILD_MODE=${1:-release}
if [ "$BUILD_MODE" = "debug" ]; then
  BAZEL_CONF="-c dbg"
  COPT_FLAGS="--copt=-Og"
  LINKOPTS=""
else
  BAZEL_CONF="-c opt"
  COPT_FLAGS="--copt=-Os --copt=-fPIC --copt=-Wno-incompatible-pointer-types"
  LINKOPTS="--linkopt=-s"
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
BENCHMARK_TOOL_PATH=${LITERT_PATH}/bazel-bin/tflite/tools/benchmark/benchmark_model
BENCHMARK_DEST_PATH=${ROOT_PATH}/benchmark_model

# ── Clean existing binary ─────────────────────────────────────────────────────
if [ -f "${BENCHMARK_DEST_PATH}" ]; then
    rm "${BENCHMARK_DEST_PATH}"
fi

# ── Build LiteRT benchmark tool ───────────────────────────────────────────────
echo "[INFO] Build LiteRT benchmark tool ($BUILD_MODE mode)…"
echo "[INFO] Path: ${BENCHMARK_TOOL_PATH}"

cd "${LITERT_PATH}" || exit 1
pwd

bazel build ${BAZEL_CONF} \
    //tflite/tools/benchmark:benchmark_model \
    ${COPT_FLAGS} \
    ${LINKOPTS}

# ── Copy binary ───────────────────────────────────────────────────────────────
echo "[INFO] Copy benchmark tool to project root…"
cp "${BENCHMARK_TOOL_PATH}" "${BENCHMARK_DEST_PATH}"

if [ -f "${BENCHMARK_DEST_PATH}" ]; then
    echo "✅ Successfully built benchmark tool: ${BENCHMARK_DEST_PATH}"
else
    echo "❌ Failed to build benchmark tool"
    exit 1
fi

cd "${ROOT_PATH}/scripts"
pwd
