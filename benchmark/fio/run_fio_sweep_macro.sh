#!/usr/bin/env bash
set -euo pipefail

# run_fio_sweep_macro.sh
#
# Runs run_fio_sweep.sh over multiple configurations.
#
# Usage:
#   ./run_fio_sweep_macro.sh /path/to/target [out_root] [trials]
#
# Defaults:
#   out_root: ./fio_macro_YYYYmmdd_HHMMSS
#   trials:   100
#
# Config grid (edit below as needed):
#   io_engine:  {io_uring, libaio, sync, psync, mmap}
#   block_size: {4k, 64k, 128k, 512k, 1m}
#   io_depth:   {1, 16, 32, 64}
#   direct:     {1}   (macro keeps 1; mmap will auto-force 0)

TARGET="${1:-}"
OUT_ROOT="${2:-}"
TRIALS="${3:-10}"

if [[ -z "${TARGET}" ]]; then
  echo "ERROR: missing TARGET"
  echo "Usage: $0 /path/to/target [out_root] [trials]"
  exit 1
fi

if [[ -z "${OUT_ROOT}" ]]; then
  OUT_ROOT="./fio_macro_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUT_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_SCRIPT="${SCRIPT_DIR}/run_fio_sweep.sh"

if [[ ! -x "${SWEEP_SCRIPT}" ]]; then
  echo "ERROR: cannot execute ${SWEEP_SCRIPT}"
  echo "Make sure run_fio_sweep.sh is in the same directory and chmod +x."
  exit 1
fi

IO_ENGINES=(libaio sync psync mmap) # io_uring
BLOCK_SIZES=(32k 64k 128k 512k 1m) # 2m 4m 8m)
IO_DEPTHS=(1 2 4 8 16 32 64)
DIRECTS=(1)

echo "Macro output root: ${OUT_ROOT}"
echo "Trials per size: ${TRIALS}"
echo "Target: ${TARGET}"

# Record config for reproducibility
CONFIG_TXT="${OUT_ROOT}/macro_config.txt"
{
  echo "date: $(date -Is)"
  echo "target: ${TARGET}"
  echo "trials: ${TRIALS}"
  echo "io_engines: ${IO_ENGINES[*]}"
  echo "block_sizes: ${BLOCK_SIZES[*]}"
  echo "io_depths: ${IO_DEPTHS[*]}"
  echo "directs: ${DIRECTS[*]}"
} > "${CONFIG_TXT}"

for eng in "${IO_ENGINES[@]}"; do
  for bs in "${BLOCK_SIZES[@]}"; do
    for qd in "${IO_DEPTHS[@]}"; do
      for direct in "${DIRECTS[@]}"; do
        OUT_DIR="${OUT_ROOT}/${eng}/bs_${bs}/qd_${qd}/direct_${direct}"
        mkdir -p "${OUT_DIR}"
        echo "== RUN: ioengine=${eng}, bs=${bs}, iodepth=${qd}, direct=${direct} =="

        # Call sweep script; pass OUT_DIR explicitly so it doesn't add its own timestamp.
        "${SWEEP_SCRIPT}" "${TARGET}" "${OUT_DIR}" "${TRIALS}" "${eng}" "${bs}" "${qd}" "${direct}"
      done
    done
  done
done

echo "All macro runs complete."
echo "Config saved to: ${CONFIG_TXT}"
