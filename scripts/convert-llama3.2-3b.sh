#!/bin/bash

MODEL_NAME=llama-3.2-3b-it
EXPORT_PATH="/home/rtos/workspace/ghpark/export"
SCRIPT_PATH="/home/rtos/workspace/ghpark/ai-edge-torch/ai_edge_torch/generative/examples/llama"

exec > >(tee "log-${MODEL_NAME}-fp32.log") 2>&1  # stdout + stderr을 로그 파일에 저장 & 실시간 출력

# echo "========================================"
# OUTPUT_PATH="${EXPORT_PATH}/${MODEL_NAME}-fp32"
# if [ -d "${OUTPUT_PATH}" ]; then
#   echo "Directory '${OUTPUT_PATH}' already exists. Skipping creation."
# else
#   echo "Directory '${OUTPUT_PATH}' does not exist. Creating now."
#   mkdir -p "${OUTPUT_PATH}"
#   echo "Directory '${OUTPUT_PATH}' created successfully."
# fi

# python3 ${SCRIPT_PATH}/convert_to_tflite.py \
#      --checkpoint_path  /home/rtos/workspace/ghpark/models/${MODEL_NAME} \
#      --output_path ${OUTPUT_PATH} \
#      --saved_model_path ${OUTPUT_PATH}

# echo "[INFO] ${OUTPUT_PATH} exported"


exec > >(tee "log-${MODEL_NAME}-q8.log") 2>&1  # stdout + stderr을 로그 파일에 저장 & 실시간 출력

echo "========================================"
OUTPUT_PATH="${EXPORT_PATH}/${MODEL_NAME}-q8"
if [ -d "${OUTPUT_PATH}" ]; then
  echo "Directory '${OUTPUT_PATH}' already exists. Skipping creation."
else
  echo "Directory '${OUTPUT_PATH}' does not exist. Creating now."
  mkdir -p "${OUTPUT_PATH}"
  echo "Directory '${OUTPUT_PATH}' created successfully."
fi

python3 ${SCRIPT_PATH}/convert_to_tflite.py \
     --checkpoint_path  /home/rtos/workspace/ghpark/models/${MODEL_NAME} \
     --output_path ${OUTPUT_PATH} \
     --saved_model_path ${OUTPUT_PATH} \
     --quantize True

echo "[INFO] ${OUTPUT_PATH} exported"

exit