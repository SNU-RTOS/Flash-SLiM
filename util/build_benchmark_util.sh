cd ..
source .env

cd ${TENSORFLOW_PATH}
pwd
bazel build -c opt --define=CL_DELEGATE_NO_GL=1 //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so
# bazel build -c opt --define=CL_DELEGATE_NO_GL=1 //tensorflow/lite/tools/benchmark:benchmark_model
# bazel shutdown

cd ${ROOT_PATH}
if [ "benchmark_model" ]; then
    rm benchmark_model
fi
cp ${TENSORFLOW_PATH}/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model benchmark_model
