#!/bin/bash

# 1. 기존 swap 해제 및 삭제
echo "🔹 현재 swap 해제 중..."
sudo swapoff -a

echo "🔹 기존 swap 파일 삭제..."
sudo rm -f /swap.img

# 2. 새로운 swap 파일 생성 (예: 4GB)
SWAP_SIZE_GB=16
SWAP_SIZE=$((SWAP_SIZE_GB * 1024 * 1024))  # MB 단위 변환

echo "🔹 새로운 ${SWAP_SIZE_GB}GB swap 파일 생성..."
sudo fallocate -l ${SWAP_SIZE_GB}G /swap.img

# 3. swap 파일 권한 설정
echo "🔹 swap 파일 권한 설정..."
sudo chmod 600 /swap.img

# 4. swap 설정
echo "🔹 swap 영역 생성..."
sudo mkswap /swap.img

# 5. swap 활성화
echo "🔹 swap 활성화..."
sudo swapon /swap.img

exit
