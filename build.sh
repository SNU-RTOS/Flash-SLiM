#!/bin/bash
source .env

make -f Makefile_main_gpu_metric 
# cd ${AI_EDGE_TORCH_PATH}
# bazel build -c opt  //ai_edge_torch/generative/examples/cpp:text_generator_main 
# # bazel build -c opt --copt=-DTFLITE_MMAP_DISABLED //ai_edge_torch/generative/examples/cpp:text_generator_main 
# bazel shutdown
# cd ${ROOT_PATH}
# # rm text_geneartor_main_gpu
# # ln -s ${ROOT_PATH}/external/ai_edge_torch/bazel-bin/generative/examples/text_generator_main text_generator_main



# cd ${TENSORFLOW_PATH}
# pwd
# bazel build -c opt //tensorflow/lite/examples/cpp:text_generator_main
# bazel shutdown
# cd ${ROOT_PATH}
# rm text_geneartor_main_gpu
# ln -s ${ROOT_PATH}/external/tensorflow/bazel-bin/tensorflow/lite/examples/cpp/text_generator_main text_generator_main_gpu