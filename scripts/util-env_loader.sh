# sshpass 설치 여부 확인
if ! command -v sshpass &> /dev/null; then
    echo "sshpass가 설치되어 있지 않습니다. 설치 합니다."
    sudo apt install sshpass -y
fi

# rsync 설치 여부 확인
if ! command -v rsync &> /dev/null; then
    echo "rsync가 설치되어 있지 않습니다. 설치 설치합니다"
    sudo apt install rsync -y
fi

# .env 파일 읽기
if [ -f .env ]; then
  source .env
else
  echo ".env 파일이 존재하지 않습니다. 스크립트를 종료합니다."
  exit 1
fi
