#!/bin/bash

source .env

########## Setup env ##########

TENSORFLOW_COMMIT_HASH=117a62ac439ed87eb26f67208be60e01c21960de


LLM_APP_SRC=${ROOT_PATH}/src
LLM_APP_BINARY_NAME=text_generator_main
# LLM_APP_BINARY_PATH=${AI_EDGE_TORCH_PATH}/bazel-bin/ai_edge_torch/generative/examples/cpp/${LLM_APP_BINARY_NAME}
LLM_APP_BINARY_PATH=./output/${LLM_APP_BINARY_NAME}

echo "[INFO] ROOT_PATH: ${ROOT_PATH}"
echo "[INFO] EXTERNAL PATH: ${EXTERNAL_PATH}"
echo "[INFO] AI_EDGE_TORCH_PATH: ${AI_EDGE_TORCH_PATH}"
echo "[INFO] TENSORFLOW_PATH: ${TENSORFLOW_PATH}"

if [ ! -d ${EXTERNAL_PATH} ]; then
    mkdir -p ${EXTERNAL_PATH}
fi

########## Setup external sources ##########
cd ${EXTERNAL_PATH}
pwd
echo "[INFO] Installing ai-edge-torch"
# Install ai-edge-torch
if [ ! -d ${AI_EDGE_TORCH_PATH} ]; then
    # Clone customized ai-edge-torch 
    git clone https://github.com/SNU-RTOS/ai-edge-torch.git

    if [ -d "${AI_EDGE_TORCH_PATH}/ai_edge_torch/generative/examples/cpp" ]; then
        rm -r ${AI_EDGE_TORCH_PATH}/ai_edge_torch/generative/examples/cpp
        echo "[INFO] delete exisiting ${AI_EDGE_TORCH_PATH}/ai_edge_torch/generative/examples/cpp"
    fi

    ln -s ${LLM_APP_SRC} ${AI_EDGE_TORCH_PATH}/ai_edge_torch/generative/examples/cpp

    # Update Tensorflow PATH in ai-edge-torch/WORKSPACE for build
    WORKSPACE_FILE="${AI_EDGE_TORCH_PATH}/WORKSPACE"
    sed -i "s|path = \".*\"|path = \"${TENSORFLOW_PATH}\"|" "$WORKSPACE_FILE"
    echo "[INFO] Updated tensorflow local_repository path in ${TENSORFLOW_PATH}/WORKSPACE to: ${TENSORFLOW_PATH}"
else
    echo "[INFO] ai-edge-torch is already installed, skipping ..."
fi

## Install tensorflow
echo "[INFO] Installing tensorflow"
if [ ! -d ${TENSORFLOW_PATH} ]; then
    mkdir -p ${TENSORFLOW_PATH}
    cd ${TENSORFLOW_PATH}
    pwd
    git init .
    git remote add origin https://github.com/tensorflow/tensorflow.git
    git fetch --depth 1 origin  ${TENSORFLOW_COMMIT_HASH}
    git checkout FETCH_HEAD
    echo "[INFO] Patching tensorflow source to build with ai-edge-torch"
    patch -p1 <../ai-edge-torch/bazel/org_tensorflow_system_python.diff
else
    echo "[INFO] tensorflow is already installed, skipping ..."
fi

cd ${EXTERNAL_PATH}
pwd

########## Make folders ##########
cd ${ROOT_PATH}
pwd

if [ ! -d "inc" ]; then
    mkdir inc    
fi

if [ ! -d "lib" ]; then
    mkdir lib   
fi

if [ ! -d "obj" ]; then
    mkdir obj  
fi

if [ ! -d "output" ]; then
    mkdir output  
fi

########## Build LiteRT ##########
cd ${ROOT_PATH}/scripts
pwd

echo "[INFO] Build LiteRT"
./build-litert.sh  debug
./build-litert_gpu_delegate.sh  debug
./build-deps.sh  debug
echo "========================"

# ########## Build LiteRT_LLM_Inference_app ##########
echo "[INFO] Build ${LLM_APP_BINARY_NAME}"
echo "========================"
cd ${ROOT_PATH}
pwd
${ROOT_PATH}/build.sh
echo "========================"

cp ${LLM_APP_BINARY_PATH} ${LLM_APP_BINARY_NAME} 
# ########## Make soft symlink ##########
# cd ${ROOT_PATH}
# pwd
# echo "[INFO] Succefully built ${LLM_APP_BINARY_NAME}"
# echo "[INFO] Making soft symbolic link ${LLM_APP_BINARY_NAME} from ${LLM_APP_BINARY_PATH} to ${ROOT_PATH}"
# if [ ${LLM_APP_BINARY_NAME} ]; then
#     rm ${LLM_APP_BINARY_NAME}
#     echo "Deleted: ${LLM_APP_BINARY_NAME}"
# fi
# ln -s ${LLM_APP_BINARY_PATH} 

# echo "[INFO] Setup finished."
