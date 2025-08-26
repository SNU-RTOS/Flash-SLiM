#!/bin/bash

# Online Parser : Can parse XNNPACK Delegate
## Define the source and destination directories
destination_dir="../../../models/SmolLM-135M/"



# # Run parser.py for each .tflite file in the destination directory
for tflite_file in "$destination_dir"/*.tflite; do
    echo "Processing $tflite_file with parser"
    model_name=$(echo "$tflite_file" | grep -oP '[^/]+(?=\.tflite)')
    log="${destination_dir}/${model_name}_dump.log"
    echo "Running dump_model_cpu_x64 for $tflite_file at $log"
    # echo 1 | python3 parser.py -m "$tflite_file"
    ./dump_model_xnnpack "$tflite_file" "$log"
done

