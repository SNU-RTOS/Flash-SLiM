#!/bin/bash

if [ ! -d tflite ]; then
    echo "tflite directory does not exist."
    echo "Generating tflite schema..."
    if arch=$(uname -m); then
        if [ "$arch" = "x86_64" ]; then
            flatc="./flatc_x64"
        fi
        if [ "$arch" = "aarch64" ]; then
            flatc="./flatc_arm64"
        fi
    fi
    $flatc --python schema.fbs
fi

python3 parser.py -m ../../models/SmolLM-135M/smollm_q8_ekv1280.tflite
# python3 parser.py -m ../../models/Gemma3-1B/gemma3_q8_ekv2048.tflite
# python3 parser.py -m ../../models/Llama3.2-3B/llama3.2_q8_ekv1024.tflite

