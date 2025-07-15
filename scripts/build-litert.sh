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
  COPT_FLAGS="--copt=-Os --copt=-fPIC --copt=-Wno-incompatible-pointer-types "
  LINKOPTS="--linkopt=-s"
fi

# ── paths ─────────────────────────────────────────────────────────────────────
LITERT_LIB_PATH=${LITERT_PATH}/bazel-bin/tflite/libtensorflowlite.so
GENAI_LIB_PATH=${LITERT_PATH}/bazel-bin/tflite/experimental/genai
RESOURCE_LIB_PATH=${LITERT_PATH}/bazel-bin/tflite/experimental/resource
CACHE_LIB_PATH=${LITERT_PATH}/bazel-bin/tflite/experimental/resource
FLATBUFFER_LIB_PATH=${LITERT_PATH}/bazel-bin/external/flatbuffers/src


TENSORFLOW_INC_PATH=${LITERT_PATH}/bazel-litert/external/org_tensorflow/tensorflow
FLATBUFFER_INC_PATH=${LITERT_PATH}/bazel-litert/external/flatbuffers/include/flatbuffers
LITERT_INC_PATH=${LITERT_PATH}/tflite

echo "[INFO] Build LiteRT ($BUILD_MODE mode)…"
echo "[INFO] Core:      ${LITERT_LIB_PATH}"
echo "[INFO] GenAI:     ${GENAI_LIB_PATH}"
echo "[INFO] Resource:  ${RESOURCE_LIB_PATH}"
echo "[INFO] CacheBuf:  ${CACHE_LIB_PATH}"

cd "${LITERT_PATH}" || exit 1
pwd

# 1) Build core + GenAI + Resource + CacheBuffer
bazel build ${BAZEL_CONF} \
    //tflite:tf_lite_runtime \
    //tflite/experimental/genai:genai_ops \
    //tflite/experimental/resource:resource \
    //tflite/experimental/resource:cache_buffer \
    @flatbuffers//src:flatbuffers \
    ${COPT_FLAGS} \
    ${LINKOPTS}

bazel shutdown

# ── symlinks ──────────────────────────────────────────────────────────────────
echo "[INFO] Symlink LiteRT and auxiliary libs…"
mkdir -p "${ROOT_PATH}/lib" "${ROOT_PATH}/inc"


## Selecet the correct library names based on the build mode
if [ "$BUILD_MODE" = "debug" ]; then
  GENAI_A="libgenai_ops.a"
  RESOURCE_A="libresource.a"
  CACHEBUF_A="libcache_buffer.a"
  FLATBUF_A="libflatbuffers.a"
else
  GENAI_A="libgenai_ops.pic.a"
  RESOURCE_A="libresource.pic.a"
  CACHEBUF_A="libcache_buffer.pic.a"
  FLATBUF_A="libflatbuffers.pic.a"
fi

## ──────────── Libs ──────────────────────────────────────────────
create_symlink_or_fail "${LITERT_LIB_PATH}" \
                       "${ROOT_PATH}/lib/libtensorflowlite.so" \
                       "libtensorflowlite.so"

create_symlink_or_fail "${GENAI_LIB_PATH}/${GENAI_A}" \
                       "${ROOT_PATH}/lib/libgenai_ops.a" \
                       "libgenai_ops.a"

create_symlink_or_fail "${RESOURCE_LIB_PATH}/${RESOURCE_A}" \
                       "${ROOT_PATH}/lib/libresource.a" \
                       "libresource.a"

create_symlink_or_fail "${CACHE_LIB_PATH}/${CACHEBUF_A}" \
                       "${ROOT_PATH}/lib/libcache_buffer.a" \
                       "libcache_buffer.a"

create_symlink_or_fail "${FLATBUFFER_LIB_PATH}/${FLATBUF_A}" \
                       "${ROOT_PATH}/lib/libflatbuffers.a" \
                       "libflatbuffers.a"

## ──────────── Headers ──────────────────────────────────────────────
create_symlink_or_fail "${FLATBUFFER_INC_PATH}" \
                       "${ROOT_PATH}/inc/" \
                       "Flatbuffers header files"

create_symlink_or_fail "${LITERT_INC_PATH}" \
                       "${ROOT_PATH}/inc/" \
                       "LiteRT header files"

create_symlink_or_fail "${TENSORFLOW_INC_PATH}" \
                        "${ROOT_PATH}/inc/" \
                        "Tensorflow header files"


cd "${ROOT_PATH}/scripts" || exit 1
pwd