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
  USE_PIC_SUFFIX="a"
else
  BAZEL_CONF="-c opt"
  COPT_FLAGS="--copt=-Os --copt=-fPIC "
  LINKOPTS="--linkopt=-s"
  USE_PIC_SUFFIX="pic.a"
fi

# ── paths ─────────────────────────────────────────────────────────────────────
ABSEIL_LIB_PATH=${AI_EDGE_TORCH_PATH}/bazel-bin/external/com_google_absl/absl
ABSEIL_INC_PATH=${AI_EDGE_TORCH_PATH}/bazel-ai-edge-torch/external/com_google_absl/absl
SENTENCEPIECE_LIB_PATH=${AI_EDGE_TORCH_PATH}/bazel-bin/external/com_google_sentencepiece
SENTENCEPIECE_INC_PATH=${AI_EDGE_TORCH_PATH}/bazel-ai-edge-torch/external/com_google_sentencepiece
PROTOBUF_LIB_PATH=${AI_EDGE_TORCH_PATH}/bazel-bin/external/com_google_protobuf
PROTOBUF_INC_PATH=${AI_EDGE_TORCH_PATH}/bazel-ai-edge-torch/external/com_google_protobuf

cd "${AI_EDGE_TORCH_PATH}"
pwd

# ── build abseil and sentencepiece ──────────────────────────────────────────────
echo "[INFO] Build abseil ($BUILD_MODE mode) .."
bazel build ${BAZEL_CONF} \
  $(bazel query 'kind("cc_library", @com_google_absl//absl/...:*) except attr("name", ".*benchmark.*", @com_google_absl//absl/...)') \
  ${COPT_FLAGS} ${LINKOPTS}
echo "[INFO] Build sentencepiece ($BUILD_MODE mode) .."
bazel build ${BAZEL_CONF} \
  @com_google_sentencepiece//:sentencepiece_cc_proto \
  @com_google_sentencepiece//:sentencepiece_model_cc_proto \
  @com_google_sentencepiece//:sentencepiece_processor \
  @com_google_protobuf//:protobuf_lite \
  ${COPT_FLAGS} ${LINKOPTS}
# ── symlinks ──────────────────────────────────────────────────────────────────
echo "[INFO] Symlink abseil and sentencepiece .."



## ──────────── abseil ──────────────────────────────────────────────
if [ ! -d "${ROOT_PATH}/lib/absl" ]; then
  mkdir -p "${ROOT_PATH}/lib/absl"
fi

find -L "${ABSEIL_LIB_PATH}" -type f -name "lib*.${USE_PIC_SUFFIX}" | while read -r libfile; do
  base="$(basename "$libfile")"
  link_name="${base%.$USE_PIC_SUFFIX}.a"
  target_link="${ROOT_PATH}/lib/absl/${link_name}"

  create_symlink_or_fail "$libfile" "$target_link" "$link_name"
done

create_symlink_or_fail "${ABSEIL_INC_PATH}" \
                       "${ROOT_PATH}/inc" \
                       "abseil header files"

## ──────────── Sentencepiece ───────────────────────────────────────
create_symlink_or_fail "${SENTENCEPIECE_LIB_PATH}/libsentencepiece_processor.pic.a" \
                         "${ROOT_PATH}/lib/libsentencepiece_processor.a" \
                         "libsentencepiece_processor.a"

for name in sentencepiece_proto sentencepiece_model_proto; do
  libfile="${SENTENCEPIECE_LIB_PATH}/lib${name}.a"
  create_symlink_or_fail "$libfile" "${ROOT_PATH}/lib/lib${name}.a" "lib${name}.a"
done

create_symlink_or_fail "${SENTENCEPIECE_INC_PATH}" \
                       "${ROOT_PATH}/inc/sentencepiece" \
                       "sentencepiece header files"

## ──────────── Protobuf ────────────────────────────────────────────
create_symlink_or_fail "${PROTOBUF_LIB_PATH}/libprotobuf_lite.so" \
                       "${ROOT_PATH}/lib/libprotobuf_lite.so" \
                       "libprotobuf_lite.so"

cd "${ROOT_PATH}/scripts"
pwd
