#!/bin/bash
cd ..
source .env

BUILD_MODE=${1:-release}
if [ "$BUILD_MODE" = "debug" ]; then
  BAZEL_CONF="-c dbg"
  COPT_FLAGS="--copt=-Og --copt=-fPIC"
  LINKOPTS=""
else
  BAZEL_CONF="-c opt"
  COPT_FLAGS="--copt=-Os --copt=-fPIC"
  LINKOPTS="--linkopt=-s"
fi

ABSEIL_PATH=${AI_EDGE_TORCH_PATH}/bazel-bin/external/com_google_absl/absl
ABSEIL_HEADER_PATH=${AI_EDGE_TORCH_PATH}/bazel-ai-edge-torch/external/com_google_absl/absl
SENTENCEPIECE_PATH=${AI_EDGE_TORCH_PATH}/bazel-bin/external/com_google_sentencepiece
SENTENCEPIECE_HEADER_PATH=${AI_EDGE_TORCH_PATH}/bazel-ai-edge-torch/external/com_google_sentencepiece
PROTOBUF_PATH=${AI_EDGE_TORCH_PATH}/bazel-bin/external/com_google_protobuf
PROTOBUF_HEADER_PATH=${AI_EDGE_TORCH_PATH}/bazel-ai-edge-torch/external/com_google_protobuf


cd ${AI_EDGE_TORCH_PATH}
pwd

########## Build ##########

# echo "[INFO] Build abseil ($BUILD_MODE mode) .."
# bazel build ${BAZEL_CONF} \
#   $(bazel query 'kind("cc_library", @com_google_absl//absl/...:*) except attr("name", ".*benchmark.*", @com_google_absl//absl/...)') \
#                           @com_google_sentencepiece//:sentencepiece_processor \
#     @com_google_sentencepiece//:sentencepiece_proto \
#     @com_google_sentencepiece//:sentencepiece_model_proto \
#     @com_google_protobuf//:protobuf_lite \
#     ${COPT_FLAGS} ${LINKOPTS}

########## Symlink ##########
echo "[INFO] Symlink abseil and sentencepiece .."

find -L "${ABSEIL_PATH}" -type f -name "lib*.a" | while read -r libfile; do
  target_link="${ROOT_PATH}/lib/absl/$(basename "$libfile")"
  echo "→ Making symlink: $(basename "$libfile")"
  ln -sf "$libfile" "$target_link"
done

echo "→ Making symlink: abseil header files"
ln -sf ${ABSEIL_HEADER_PATH} ${ROOT_PATH}/inc/


# SentencePiece
echo "→ Making symlink: libsentencepiece_processor.a"
ln -sf ${SENTENCEPIECE_PATH}/libsentencepiece_processor.a ${ROOT_PATH}/lib/
echo "→ Making symlink: libsentencepiece_proto.a"
ln -sf ${SENTENCEPIECE_PATH}/libsentencepiece_proto.a  ${ROOT_PATH}/lib/
echo "→ Making symlink: libsentencepiece_model_proto.a"
ln -sf ${SENTENCEPIECE_PATH}/libsentencepiece_model_proto.a ${ROOT_PATH}/lib/
echo "→ Making symlink: sentencepiece header files"
ln -sf ${SENTENCEPIECE_HEADER_PATH} ${ROOT_PATH}/inc/sentencepiece

# ProtoBuf
echo "→ Making symlink: libprotobuf_lite.so"
ln -sf ${PROTOBUF_PATH}/libprotobuf_lite.so ${ROOT_PATH}/lib/


cd ${ROOT_PATH}/scripts
pwd