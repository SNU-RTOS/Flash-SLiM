#!/bin/bash
source ./common_env_loader.sh

# Define the list of subfolders to include
INCLUDE_FOLDERS=(
    # "DeepSeek-R1-Distill-Qwen-1.5B-q8"
    "llama-3.2-3b-it-q8"
    "gemma-2-2b-it-q8" 
    "phi-3.5-mini-it-q8"
    )  

# Upload each specified subfolder
for folder in "${INCLUDE_FOLDERS[@]}"; do
    LOCAL_PATH="${EXPORTED_MODEL_DIR}/${folder}"
    REMOTE_PATH="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_MODEL_DIR}/${folder}"

    if [ -d "${LOCAL_PATH}" ]; then
        echo "Uploading ${LOCAL_PATH} to ${REMOTE_PATH}"
        sshpass -p "${REMOTE_PASSWORD}" rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" "${LOCAL_PATH}/" "${REMOTE_PATH}/"
    else
        echo "Skipping ${LOCAL_PATH}: Directory does not exist."
    fi
done

echo "Selected folders uploaded successfully!"


