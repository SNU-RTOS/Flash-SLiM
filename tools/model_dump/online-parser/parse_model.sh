#!/bin/bash

# Online Parser : Can parse XNNPACK Delegate
## Define the source and destination directories
destination_dir="../../../models/Llama3.2-3B"
bin=./dump_llm



# Run parser.py for each .tflite file in the destination directory
for tflite_file in "$destination_dir"/*.tflite; do
    echo "Processing $tflite_file with parser"
    model_name=$(echo "$tflite_file" | grep -oP '[^/]+(?=\.tflite)')
    log="${destination_dir}/${model_name}_dump.log"
    weight_cache="${destination_dir}/${model_name}.xnnpack_cache"

    echo "Running dump_model_cpu_x64 for $tflite_file at $log"
    echo "weight_cache: $weight_cache"
    echo "log: $log"
    $bin \
        --dump_file_path "$log" \
        --tflite_model "$tflite_file" \
        --weight_cache_path "$weight_cache"
done

