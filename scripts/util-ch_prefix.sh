#!/bin/bash

# 변경할 패턴 설정
OLD_PATTERN="mem_usage_*"
NEW_PREFIX="memusage_"

# 대상 디렉토리 (현재 디렉토리 기준)
TARGET_DIR="./"

for file in "${TARGET_DIR}"$OLD_PATTERN; do
    if [[ -f "$file" ]]; then
        base_name=$(basename "$file")
        # 파일 이름이 "mem_usage_"로 시작하면 그 이후 문자열을 추출
        if [[ "$base_name" == mem_usage_* ]]; then
            suffix="${base_name#mem_usage_}"
            echo $suffix
            new_name="${NEW_PREFIX}${suffix}"
            echo "Renamed: $file -> ${TARGET_DIR}${new_name}"
            mv "$file" "${TARGET_DIR}${new_name}"
        fi
    fi
done
echo "파일 이름 변경 완료!"
