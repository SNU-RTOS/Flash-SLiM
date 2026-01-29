#!/usr/bin/env bash
set -euo pipefail

# run_fio_sweep.sh
#
# Runs fio read sweep over SIZE with configurable ioengine/bs/iodepth/direct.
#
# Usage:
#   ./run_fio_sweep.sh /path/to/target [out_dir] [trials] [io_engine] [block_size] [io_depth] [direct]
#
# target:
#   - block device (e.g., /dev/nvme0n1p1) OR
#   - directory (will create a 1GiB test file in it) OR
#   - existing file path
#
# out_dir (optional): default ./fio_logs_{io_engine}_bs{block_size}_qd{io_depth}_direct{direct}_YYYYmmdd_HHMMSS
# trials (optional): default 100
# io_engine (optional): default io_uring
# block_size (optional): default 512k (fio accepts 4k, 64k, 512k, 1m, etc.)
# io_depth (optional): default 16
# direct (optional): default 1 (0 or 1)

TARGET="${1:-}"
OUT_DIR="${2:-}"
TRIALS="${3:-100}"
IOENGINE="${4:-io_uring}"
BLOCK_SIZE="${5:-512k}"
IODEPTH="${6:-16}"
DIRECT="${7:-1}"

if [[ -z "${TARGET}" ]]; then
  echo "ERROR: missing TARGET"
  echo "Usage: $0 /path/to/target [out_dir] [trials] [io_engine] [block_size] [io_depth] [direct]"
  exit 1
fi

command -v fio >/dev/null 2>&1 || { echo "ERROR: fio not found in PATH"; exit 1; }

# If OUT_DIR not set, build one from config
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="./fio_logs_${IOENGINE}_bs${BLOCK_SIZE}_qd${IODEPTH}_direct${DIRECT}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUT_DIR}"

# Resolve filename for fio:
FIO_FILENAME="${TARGET}"
TESTFILE_CREATED="0"

if [[ -d "${TARGET}" ]]; then
  FIO_FILENAME="${TARGET%/}/fio_testfile_1g.bin"
  echo "TARGET is a directory; creating test file: ${FIO_FILENAME}"
  rm -f "${FIO_FILENAME}"
  fallocate -l 1G "${FIO_FILENAME}"
  sync
  TESTFILE_CREATED="1"
else
  if [[ ! -e "${TARGET}" ]]; then
    echo "ERROR: TARGET does not exist: ${TARGET}"
    exit 1
  fi
fi

# Validate DIRECT
if [[ "${DIRECT}" != "0" && "${DIRECT}" != "1" ]]; then
  echo "ERROR: direct must be 0 or 1, got: ${DIRECT}"
  exit 1
fi

# Validate iodepth is positive integer
if ! [[ "${IODEPTH}" =~ ^[0-9]+$ ]] || [[ "${IODEPTH}" -lt 1 ]]; then
  echo "ERROR: io_depth must be a positive integer, got: ${IODEPTH}"
  exit 1
fi

# IMPORTANT:
# - sync/psync engines do not use iodepth>1 meaningfully; keep but warn.
# - mmap engine is buffered (page cache) and direct=1 is not meaningful; warn.
case "${IOENGINE}" in
  sync|psync)
    if [[ "${IODEPTH}" -ne 1 ]]; then
      echo "WARN: ioengine=${IOENGINE} ignores iodepth>1 in practice; consider io_depth=1."
    fi
    ;;
  mmap)
    if [[ "${DIRECT}" -eq 1 ]]; then
      echo "WARN: ioengine=mmap is page-cache backed; direct=1 is not applicable. Setting direct=0 for this run."
      DIRECT="0"
    fi
    ;;
  libaio|io_uring)
    ;;
  *)
    echo "WARN: ioengine=${IOENGINE} not in {io_uring,libaio,sync,psync,mmap}. fio may still accept it."
    ;;
esac

# Sweep sizes: mixed exponential + linear (your current policy)
MIN_SIZE=$((1 * 1024 * 1024))
MIN_LINEAR=$((16 * 1024 * 1024))
MAX_LINEAR=$((512 * 1024 * 1024))
MAX_SIZE=$((1024 * 1024 * 1024))
STEP_LINEAR=$((16 * 1024 * 1024))

echo "Writing logs to: ${OUT_DIR}"
echo "fio filename: ${FIO_FILENAME}"
echo "trials per size: ${TRIALS}"
echo "ioengine: ${IOENGINE}"
echo "block_size: ${BLOCK_SIZE}"
echo "iodepth: ${IODEPTH}"
echo "direct: ${DIRECT}"
echo "size sweep: ${MIN_SIZE} .. ${MAX_SIZE} (mixed linear/exponential)"

SIZE="${MIN_SIZE}"
while [[ "${SIZE}" -le "${MAX_SIZE}" ]]; do
  SIZE_DIR="${OUT_DIR}/size_${SIZE}"
  mkdir -p "${SIZE_DIR}"

  echo "== SIZE ${SIZE} bytes =="

  for ((i=1; i<=TRIALS; i++)); do
    OUT_JSON="${SIZE_DIR}/run_$(printf "%03d" "${i}").json"

    fio \
      --name="read_size${SIZE}_iter${i}" \
      --filename="${FIO_FILENAME}" \
      --rw=read \
      --direct="${DIRECT}" \
      --ioengine="${IOENGINE}" \
      --iodepth="${IODEPTH}" \
      --numjobs=1 \
      --bs="${BLOCK_SIZE}" \
      --size="${SIZE}" \
      --offset=0 \
      --readonly \
      --group_reporting=1 \
      --output-format=json \
      --output="${OUT_JSON}" >/dev/null
  done

  # mixed step: linear in [16MiB, 512MiB), otherwise exponential
  # if (( SIZE >= MIN_LINEAR && SIZE < MAX_LINEAR )); then
  #   SIZE=$(( SIZE + STEP_LINEAR ))
  # else
  #   SIZE=$(( SIZE * 2 ))
  # fi
  SIZE=$(( SIZE * 2 ))
done

echo "Done."

if [[ "${TESTFILE_CREATED}" == "1" ]]; then
  echo "Note: test file remains at: ${FIO_FILENAME}"
fi
