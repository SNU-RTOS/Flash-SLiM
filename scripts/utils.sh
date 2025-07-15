#!/bin/bash

TRACEFS="/sys/kernel/debug/tracing"
TRACE_SAVE_INTERVAL=5  # Save trace every 5 seconds

setup_ftrace() {
    # Clear existing trace
    echo > $TRACEFS/trace

    # Disable tracing temporarily
    echo 0 > $TRACEFS/tracing_on

    # Clear existing events
    echo > $TRACEFS/set_event

    # Enable memory-related events
    echo 1 > $TRACEFS/events/kmem/mm_page_alloc/enable
    echo 1 > $TRACEFS/events/kmem/mm_page_free/enable
    echo 1 > $TRACEFS/events/kmem/rss_stat/enable
    
    # Memory reclaim events
    echo 1 > $TRACEFS/events/vmscan/mm_vmscan_direct_reclaim_begin/enable
    echo 1 > $TRACEFS/events/vmscan/mm_vmscan_direct_reclaim_end/enable
    
    # Set the trace buffer size (in KB)
    echo 8192 > $TRACEFS/buffer_size_kb

    # Use function tracer
    echo function > $TRACEFS/current_tracer
    
    # Clear the trace buffer
    echo > $TRACEFS/trace
}

setup_pid_filter() {
    local pid=$1
    
    echo "common_pid==$pid" > $TRACEFS/events/kmem/mm_page_alloc/filter
    echo "common_pid==$pid" > $TRACEFS/events/kmem/mm_page_free/filter
    echo "common_pid==$pid" > $TRACEFS/events/kmem/rss_stat/filter
    echo "common_pid==$pid" > $TRACEFS/events/vmscan/mm_vmscan_direct_reclaim_begin/filter
    echo "common_pid==$pid" > $TRACEFS/events/vmscan/mm_vmscan_direct_reclaim_end/filter
}

save_trace_buffer() {
    local output_file=$1
    local elapsed_time=$2
    
    # Save current trace buffer by appending to the output file
    echo -e "\n# Elapsed time: $elapsed_time seconds" >> "$output_file"
    cat $TRACEFS/trace >> "$output_file"
    
    # Clear the buffer for next collection
    echo > $TRACEFS/trace
    
    echo "[INFO] Appended trace data at $elapsed_time seconds"
}

get_pagefault_stats() {
    local pid=$1
    local stat_line=$(cat /proc/$pid/stat 2>/dev/null)
    if [ $? -eq 0 ]; then
        local stats=($stat_line)
        echo "${stats[9]},${stats[11]}"
    else
        echo "0,0"
    fi
}


clear_caches() {
    echo "[INFO] Start: Dropping OS Page Caches.."
    sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    echo "[INFO] Waiting for caches to clear..."
    # sleep for a brief moment to ensure caches are cleared
    sleep 1
    echo "[INFO] DONE: Dropping OS Page Caches"

    echo "Dropping swapped memory..."
    # sudo swapoff -a
    # sudo swapon -a
    sleep 1
    echo "[INFO] DONE: Dropping Swapped Memory"

    echo "[INFO] Clearing CPU Caches"
    ARCH=$(uname -m)

    case "$ARCH" in
        "x86_64")
            CACHE_SCRIPT="./util/clear_cache_x86"
            ;;
        "aarch64")
            CACHE_SCRIPT="./util/clear_cache_arm"
            ;;
        *)
            echo "[ERROR] Unsupported architecture: $ARCH"
            return 1
            ;;
    esac
    
    if [[ -f "$CACHE_SCRIPT" ]]; then
        "$CACHE_SCRIPT"
    else
        echo "[WARNING] CPU cache clearing script not found: $CACHE_SCRIPT"
    fi

    echo "[INFO] DONE: Clearing Caches"
    echo ""
}

