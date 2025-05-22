#!/usr/bin/env bash
# build.sh
#
# Build the LLM sample in CPU and/or GPU flavours.
#
# Usage:
#   ./build.sh            # defaults to "gpu"
#   ./build.sh cpu        # CPU-only build
#   ./build.sh gpu        # GPU-only build
#   ./build.sh all        # build both variants
#
# Resulting binaries:
#   output/text_generator_main_gpu
#   output/text_generator_main_cpu
#
# A convenience symlink ${ROOT_PATH}/text_generator_main will point to
# the most recently built variant.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Environment
# ---------------------------------------------------------------------------
source .env
: "${ROOT_PATH:?ROOT_PATH must be set in .env}"

APP_BASE=text_generator_main         # basename without suffix
OUT_DIR=output

CPU_BIN="${OUT_DIR}/${APP_BASE}_cpu"
GPU_BIN="${OUT_DIR}/${APP_BASE}_gpu"

# ---------------------------------------------------------------------------
# 1. Parse argument
# ---------------------------------------------------------------------------
BUILD_TARGET="${1:-gpu}"    # default = gpu

case "${BUILD_TARGET}" in
  cpu|gpu|all) ;;
  *)
    echo "Invalid target '${BUILD_TARGET}' (expected cpu|gpu|all)" >&2
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# 2. Build helpers
# ---------------------------------------------------------------------------
run() { echo "+ $*"; "$@"; }

do_gpu_build() {
  echo "[INFO] Building GPU variant..."
  run make -f Makefile-gpu -j"$(nproc)"
  # Rename / move artefact
  [[ -f "${OUT_DIR}/${APP_BASE}" ]] \
      && mv -f "${OUT_DIR}/${APP_BASE}" "${GPU_BIN}"
  echo "[INFO] GPU binary -> ${GPU_BIN}"
}

do_cpu_build() {
  echo "[INFO] Building CPU variant..."
  run make -f Makefile-cpu -j"$(nproc)"
  [[ -f "${OUT_DIR}/${APP_BASE}" ]] \
      && mv -f "${OUT_DIR}/${APP_BASE}" "${CPU_BIN}"
  echo "[INFO] CPU binary -> ${CPU_BIN}"
}

# ---------------------------------------------------------------------------
# 3. Invoke builds
# ---------------------------------------------------------------------------
[[ "${BUILD_TARGET}" == gpu || "${BUILD_TARGET}" == all ]] && do_gpu_build
[[ "${BUILD_TARGET}" == cpu || "${BUILD_TARGET}" == all ]] && do_cpu_build

echo "[SUCCESS] Finished build target='${BUILD_TARGET}'."

# ---------------------------------------------------------------------------
# 4. Update convenience symlink
# ---------------------------------------------------------------------------
cd "${ROOT_PATH}"

case "${BUILD_TARGET}" in
  gpu) TARGET_BIN="${GPU_BIN}" ;;
  cpu) TARGET_BIN="${CPU_BIN}" ;;
  all) TARGET_BIN="${GPU_BIN}" ;;   # default symlink to GPU when building both
esac

if [[ -L "${APP_BASE}" || -e "${APP_BASE}" ]]; then
  rm -f "${APP_BASE}"
fi

ln -s "${TARGET_BIN}" "${APP_BASE}"
echo "[INFO] Symlink ${APP_BASE} → ${TARGET_BIN}"
