source .env
cd ${TENSORFLOW_PATH}
pwd
bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model
bazel shutdown

cd ${ROOT_PATH}
if [ "benchmark_model" ]; then
    rm benchmark_model
fi
cp ${ROOT_PATH}/external/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model benchmark_model
