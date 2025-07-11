#!/bin/bash
set -e

# ==============================================================================
# run_test: 지정된 모드로 I/O 테스트를 실행합니다.
# - 페이지 캐시를 비웁니다.
# - taskset을 사용하여 지정된 CPU 코어에서 테스트를 실행합니다.
#
# @param $1: 테스트 모드 (e.g., mmap, read, io_uring)
# ==============================================================================
run_test() {
    local mode=$1
    if [ -z "$mode" ]; then
        echo "[ERROR] Test mode is not provided."
        exit 1
    fi

    echo "================================================="
    echo "[INFO] Preparing test for mode: $mode"
    echo "-------------------------------------------------"

    echo "[INFO] Dropping page cache..."
    sync
    # 'here string' (<<<)을 사용하여 파이프를 대체하고, sudo 권한 실패 시 에러를 출력합니다.
    if ! echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null; then
        echo "[ERROR] Failed to drop page caches. Make sure you have sudo privileges."
        exit 1
    fi
    sleep 3

    echo "[INFO] Running io_test with mode '$mode' on CPU cores 8-15 "
    taskset -c 8-15 ./io_test "$mode"
    echo "[INFO] Test for mode '$mode' finished."
    echo "================================================="
    echo
}

# 1) 테스트 환경 준비: 기존 복사본 삭제
if [ -f dummy4g_copy.bin ]; then
    echo "[INFO] Removing existing dummy4g_copy.bin..."
    rm dummy4g_copy.bin
    echo
fi

# 2) 모든 모드에 대해 테스트 실행
run_test "mmap"
run_test "read"
run_test "io_uring"
run_test "direct_io"

echo "[INFO] All tests completed."
