#!/bin/bash
source .env
cd ${AI_EDGE_TORCH_PATH}
bazel build -c opt //ai_edge_torch/generative/examples/rt-llm-infer:text_generator_main 

# bazel build -c opt --copt=-DTFLITE_MMAP_DISABLED //ai_edge_torch/generative/examples/cpp:text_generator_main 
