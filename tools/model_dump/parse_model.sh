#!/bin/bash

# Online Parser : Can parse XNNPACK Delegate
## Define the source and destination directories
destination_dir="../../models/Llama3.2-3B"
# destination_dir="../../models/Llama3.2-1B"
# destination_dir="../../models/Qwen2.5-3B"
# destination_dir="../../models/Qwen2.5-1.5B"
# destination_dir="../../models/SmolLM-135M" # NEED TO PATCH -> OUTPUT FORMAT IS NOT SUPPORTED
# destination_dir="../../models/Gemma3-1B" # NEED TO PATCH -> OUTPUT FORMAT IS NOT SUPPORTED
bin=dump_llm_nodes


./build_model_parser.sh $bin

# Run parser.py for each .tflite file in the destination directory
for tflite_file in "$destination_dir"/*.tflite; do
    echo "Processing $tflite_file with parser"
    model_name=$(echo "$tflite_file" | grep -oP '[^/]+(?=\.tflite)')
    log="${destination_dir}/${model_name}_dump.log"
    weight_cache="${destination_dir}/${model_name}.xnnpack_cache"

    echo "Running dump_model_cpu_x64 for $tflite_file at $log"
    echo "weight_cache: $weight_cache"
    echo "log: $log"

    sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

    # start bpftrace
    # echo "Starting bpftrace..."
    # sudo setsid bpftrace weight_cache_check.bt > weight_cache_bpf.log 2>&1 < /dev/null &
    # bpftrace_pid=$!
    # sleep 5 # Give bpftrace some time to start
    echo -e "Running $bin...\n"
    # --------------------------------------
    ./$bin \
        --dump_file_path "$log" \
        --tflite_model "$tflite_file" \
        --weight_cache_path "$weight_cache" \
        --dump_tensor_details \
        --num_threads 1 \
        --op_tensor_byte_stats 
    # --------------------------------------
    # stop bpftrace
    # echo "Stopping bpftrace..."
    # sudo kill -INT $bpftrace_pid
    # wait $bpftrace_pid 2>/dev/null
        
    python3 tensor_visualization.py "$log" "${destination_dir}/${model_name}_analysis_report.txt" "${destination_dir}/${model_name}_analysis_data.json"

    echo "Finished processing $tflite_file"
    echo "----------------------------------------"
    echo ""
    echo "Raw log saved to $log"
    echo -e "Report saved to ${destination_dir}/${model_name}_analysis_report.txt \nand saved to ${destination_dir}/${model_name}_analysis_data.json"
done

