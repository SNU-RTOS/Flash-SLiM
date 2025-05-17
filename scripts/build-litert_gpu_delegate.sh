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

GPU_DELEGATE_PATH=${TENSORFLOW_PATH}/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so
########## Build ##########

echo "[INFO] Build GPU Delegate ($BUILD_MODE mode) .."
echo "[INFO] Path: ${GPU_DELEGATE_PATH}"

cd ${TENSORFLOW_PATH}
pwd

bazel build ${BAZEL_CONF} //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so \
    ${COPT_FLAGS} ${LINKOPTS}
bazel shutdown

########## Symlink ##########
echo "[INFO] Symlink LiteRT GPU Delegate.."
ln -sf ${GPU_DELEGATE_PATH} ${ROOT_PATH}/lib/libtensorflowlite_gpu_delegate.so

cd ${ROOT_PATH}/scripts
pwd
