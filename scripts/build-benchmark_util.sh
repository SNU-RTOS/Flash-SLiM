cd ..
source .env

cd ${TENSORFLOW_PATH}
pwd


cd ${ROOT_PATH}
if [ "benchmark_model" ]; then
    rm benchmark_model
fi
cp ${TENSORFLOW_PATH}/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model benchmark_model
