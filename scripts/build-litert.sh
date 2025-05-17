#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
cd ..
source .env

BUILD_MODE=${1:-release}
if [ "$BUILD_MODE" = "debug" ]; then
  BAZEL_CONF="-c dbg"
  COPT_FLAGS="--copt=-Og --copt=-fPIC"
  LINKOPTS=""
else
  BAZEL_CONF="-c opt"
  COPT_FLAGS="--copt=-Os --copt=-fPIC --copt=-Wno-incompatible-pointer-types --copt='-DXNN_LOG_LEVEL=1' "
  LINKOPTS="--linkopt=-s"
fi

# ── paths ─────────────────────────────────────────────────────────────────────
LITERT_LIB_PATH=${TENSORFLOW_PATH}/bazel-bin/tensorflow/lite/libtensorflowlite.so
GENAI_LIB_PATH=${TENSORFLOW_PATH}/bazel-bin/tensorflow/lite/experimental/genai/libgenai_ops.a
RESOURCE_LIB_PATH=${TENSORFLOW_PATH}/bazel-bin/tensorflow/lite/experimental/resource/libresource.a
CACHE_LIB_PATH=${TENSORFLOW_PATH}/bazel-bin/tensorflow/lite/experimental/resource/libcache_buffer.a
FLATBUFFER_LIB_PATH=${TENSORFLOW_PATH}/bazel-bin/external/flatbuffers/src/libflatbuffers.a

FLATBUFFER_INC_PATH=${TENSORFLOW_PATH}/bazel-tensorflow/external/flatbuffers/include/flatbuffers
LITERT_INC_PATH=${TENSORFLOW_PATH}

echo "[INFO] Build LiteRT ($BUILD_MODE mode)…"
echo "[INFO] Core:      ${LITERT_LIB_PATH}"
echo "[INFO] GenAI:     ${GENAI_LIB_PATH}"
echo "[INFO] Resource:  ${RESOURCE_LIB_PATH}"
echo "[INFO] CacheBuf:  ${CACHE_LIB_PATH}"

cd "${TENSORFLOW_PATH}" || exit 1
pwd

# 1) Build core + GenAI + Resource + CacheBuffer
bazel build ${BAZEL_CONF} \
    //tensorflow/lite:tensorflowlite \
    //tensorflow/lite/experimental/genai:genai_ops \
    //tensorflow/lite/experimental/resource:resource \
    //tensorflow/lite/experimental/resource:cache_buffer \
    ${COPT_FLAGS} \
    ${LINKOPTS}

bazel shutdown

# ── symlinks ──────────────────────────────────────────────────────────────────
echo "[INFO] Symlink LiteRT and auxiliary libs…"
mkdir -p "${ROOT_PATH}/lib" "${ROOT_PATH}/inc"

ln -sf "${LITERT_LIB_PATH}"    "${ROOT_PATH}/lib/libtensorflowlite.so"
ln -sf "${GENAI_LIB_PATH}"     "${ROOT_PATH}/lib/libgenai_ops.a"
ln -sf "${RESOURCE_LIB_PATH}"  "${ROOT_PATH}/lib/libresource.a"
ln -sf "${CACHE_LIB_PATH}"     "${ROOT_PATH}/lib/libcache_buffer.a"
ln -sf "${FLATBUFFER_LIB_PATH}" "${ROOT_PATH}/lib/libflatbuffers.a"

ln -sf "${FLATBUFFER_INC_PATH}" "${ROOT_PATH}/inc/"
ln -sf "${LITERT_INC_PATH}/tensorflow" "${ROOT_PATH}/inc/"

cd "${ROOT_PATH}/scripts" || exit 1
pwd
