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
GPU_DELEGATE_PATH=${LITERT_PATH}/bazel-bin/tflite/delegates/gpu/libtensorflowlite_gpu_delegate.so

# ── Build GPU Delegate ────────────────────────────────────────────────────────
echo "[INFO] Build GPU Delegate ($BUILD_MODE mode) .."
echo "[INFO] Path: ${GPU_DELEGATE_PATH}"

cd "${LITERT_PATH}" || exit 1
pwd

bazel build ${BAZEL_CONF} \
  //tflite/delegates/gpu:libtensorflowlite_gpu_delegate.so \
  ${COPT_FLAGS} ${LINKOPTS}

# ── Symlinks ──────────────────────────────────────────────────────────────────
echo "[INFO] Symlink LiteRT GPU Delegate.."

create_symlink_or_fail "${GPU_DELEGATE_PATH}" \
                       "${ROOT_PATH}/lib/libtensorflowlite_gpu_delegate.so" \
                       "libtensorflowlite_gpu_delegate.so"

cd "${ROOT_PATH}/scripts"
pwd
