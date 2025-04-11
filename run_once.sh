#!/bin/bash

################ Load utils ################
source ./util/utils.sh

################ Setup environment ################
source .env

##################################
# MODEL_PATH="/home/rtos/workspace/ghpark/export/gemma-2-2b-it-q8"
# MODEL_NAME="gemma2_q8_ekv1024"

# MODEL_PATH="/home/rtos/workspace/ghpark/export/gemma-2-2b-it-fp32"
# MODEL_NAME="gemma2_f32_ekv1024"

MODEL_PATH="${MODEL_PATH}/llama-3.2-3b-it-q8"
MODEL_NAME="llama_q8_ekv1024"


FILE="./${PROMPT_PATH}/sample_prompt_8_1.txt"

if [ ! -f "$FILE" ]; then
    echo "Error: 파일 '$FILE'이 존재하지 않습니다."
    exit 1
fi

echo "[INFO] Dropping OS Page Caches.."
clear_caches
echo "[INFO] Clearing CPU Caches"

################ scripts ################
while read -r line; do
    if [[ "$line" =~ ^([0-9]+),\"(.*)\"$ ]]; then
            token_count="${BASH_REMATCH[1]}"
            prompt="${BASH_REMATCH[2]}"

        sudo ./text_generator_main \
            --tflite_model="${MODEL_PATH}/${MODEL_NAME}.tflite" \
            --sentencepiece_model="${MODEL_PATH}/tokenizer.model" \
            --max_decode_steps=32 \
            --start_token="<bos>" \
            --stop_token="<eos>" \
            --num_threads=1 \
            --prompt="$prompt" \
            --weight_cache_path="${MODEL_PATH}/${MODEL_NAME}.xnnpack_cache"

    fi
done <"$FILE"

######
# text_generator_main: Warning: SetProgramUsageMessage() never called
#   Flags from ai_edge_torch/generative/examples/cpp/text_generator_main.cc:
#     --lora_path (Optional path to LoRA artifact.); default: "";
#     --max_decode_steps (The number of tokens to generate. Defaults to the KV
#       cache size defined during conversion.); default: -1;
#     --num_threads (Number of threads to use. Defaults to 4.); default: 4;
#     --prompt (Input prompt to the model.); default: "Write an email:";
#     --sentencepiece_model (Path to sentencepiece model.); default: "";
#     --start_token (Start token is appended to the beginning of input prompt to
#       signify start of sentence.); default: "";
#     --stop_token (Stop token used to deterine end of decoding loop. If not
#       provided will decode until max_kv_cache_size or max_decode_steps.);
#       default: "";
#     --tflite_model (Two-signature tflite model prepared for text generation
#       using ODML tools.); default: "";
#     --weight_cache_path (XNNPACK weight caching path, e.g.
#       /tmp/model.xnnpack_cache.); default: "";

# Try --helpfull to get a list of all flags or --help=substring shows help for
# flags which include specified substring in either in the name, or description or
# path.
