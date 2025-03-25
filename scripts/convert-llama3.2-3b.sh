#!/bin/bash

# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
# SCRIPT_PATH="/home/rtos/workspace/ghpark/ai-edge-torch/ai_edge_torch/generative/examples/deepseek"

# MODEL_NAME=gemma-2-2b-it
# SCRIPT_PATH="/home/rtos/workspace/ghpark/ai-edge-torch/ai_edge_torch/generative/examples/gemma"

# MODEL_NAME="phi-3.5-mini-it"
# SCRIPT_PATH="/home/rtos/workspace/ghpark/ai-edge-torch/ai_edge_torch/generative/examples/phi"

MODEL_NAME=llama-3.2-3b-it
SCRIPT_PATH="/home/rtos/workspace/ghpark/ai-edge-torch/ai_edge_torch/generative/examples/llama"

MODEL_EXPORT_PATH="/home/rtos/workspace/models/export"
MODEL_ORIGIN_CHECKPOINT_PATH="/home/rtos/workspace/models/llm/${MODEL_NAME}"

exec > >(tee "log-${MODEL_NAME}-fp32.log") 2>&1  # stdout + stderr을 로그 파일에 저장 & 실시간 출력

# echo "========================================"
# EXPORT_PATH="${MODEL_EXPORT_PATH}/${MODEL_NAME}-fp32"
# if [ -d "${EXPORT_PATH}" ]; then
#   echo "Directory '${EXPORT_PATH}' already exists. Skipping creation."
# else
#   echo "Directory '${EXPORT_PATH}' does not exist. Creating now."
#   mkdir -p "${EXPORT_PATH}"
#   echo "Directory '${EXPORT_PATH}' created successfully."
# fi

# python3 ${SCRIPT_PATH}/convert_to_tflite.py \
#      --checkpoint_path  /home/rtos/workspace/ghpark/models/${MODEL_NAME} \
#      --output_path ${EXPORT_PATH} \
#      --saved_model_path ${EXPORT_PATH}

# echo "[INFO] ${EXPORT_PATH} exported"


exec > >(tee "log-${MODEL_NAME}-q8.log") 2>&1  # stdout + stderr을 로그 파일에 저장 & 실시간 출력

echo "========================================"
EXPORT_PATH="${MODEL_EXPORT_PATH}/${MODEL_NAME}-q8"
if [ -d "${EXPORT_PATH}" ]; then
  echo "Directory '${EXPORT_PATH}' already exists. Skipping creation."
else
  echo "Directory '${EXPORT_PATH}' does not exist. Creating now."
  mkdir -p "${EXPORT_PATH}"
  echo "Directory '${EXPORT_PATH}' created successfully."
fi

python3 ${SCRIPT_PATH}/convert_to_tflite.py \
     --checkpoint_path  ${MODEL_ORIGIN_CHECKPOINT_PATH} \
     --output_path ${EXPORT_PATH} \
     --saved_model_path ${EXPORT_PATH} \
     --quantize True

echo "[INFO] ${EXPORT_PATH} exported"

exit