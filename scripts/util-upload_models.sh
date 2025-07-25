#!/bin/bash

if ! command -v sshpass &> /dev/null; then
    echo "sshpass could not be found. Please install it to use this script."
    exit 1
fi

if ! command -v rsync &> /dev/null; then
    echo "rsync could not be found. Please install it to use this script."
    exit 1
fi


# Define the remote server details
REMOTE_USER="root"
REMOTE_IP="192.xxx.x.x"
REMOTE_PORT="22"
REMOTE_PW="xxxx"
REMOTE_PATH="/ENTER/PATH/TO/MODELS/DIR/IN/REMOTE/MACHINE"

# Define the local path to the exported models directory
LOCAL_PATH="/ENTER/PATH/TO/MODELS/DIR/IN/LOCAL/MACHINE"

sshpass -p "${REMOTE_PW}" rsync -avz --progress -e \
    "ssh -p ${REMOTE_PORT}" \
    "${LOCAL_PATH}" \
    "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PATH}"

echo "Selected folders uploaded successfully!"



# # Define the list of subfolders to include
# INCLUDE_FOLDERS=(
#     # "DeepSeek-R1-Distill-Qwen-1.5B-q8"
#     "llama-3.2-3b-it-q8"
#     "gemma-2-2b-it-q8" 
#     "phi-3.5-mini-it-q8"
#     )  

# Upload each specified subfolder
# for folder in "${INCLUDE_FOLDERS[@]}"; do
#     LOCAL_PATH="${EXPORTED_MODEL_DIR}/${folder}"
#     REMOTE_PATH="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_MODEL_DIR}/${folder}"

#     if [ -d "${LOCAL_PATH}" ]; then
#         echo "Uploading ${LOCAL_PATH} to ${REMOTE_PATH}"
#         sshpass -p "${REMOTE_PASSWORD}" rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" "${LOCAL_PATH}/" "${REMOTE_PATH}/"
#     else
#         echo "Skipping ${LOCAL_PATH}: Directory does not exist."
#     fi
# done