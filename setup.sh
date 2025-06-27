#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# setup.sh
#
# 1) Prepare third-party sources (ai-edge-torch, TensorFlow at a fixed commit)
# 2) Build LiteRT and the GPU delegate
# 3) Build the LLM inference application
#
# Required environment variables (.env):
#   ROOT_PATH           – project root directory
#   EXTERNAL_PATH       – directory for external sources (e.g. ${ROOT_PATH}/external)
#   AI_EDGE_TORCH_PATH  – "${EXTERNAL_PATH}/ai-edge-torch"
#   TENSORFLOW_PATH     – "${EXTERNAL_PATH}/tensorflow"
#   LLM_APP_SRC         – "${ROOT_PATH}/src/llm_app"
#
# Usage:
#   ./setup.sh          # build in “release” mode (default)
#   ./setup.sh debug    # build in “debug”  mode
# ------------------------------------------------------------------------------

set -euo pipefail

source .env

# --- Configuration ------------------------------------------------------------
# TF_COMMIT=117a62ac439ed87eb26f67208be60e01c21960de
BUILD_TYPE="${1:-release}"   # debug | release

# --- Helpers ------------------------------------------------------------------
run()         { echo "+ $*"; "$@"; }
ensure_dir()  { [[ -d $1 ]] || run mkdir -p "$1"; }
banner()      { printf "\n\033[1;34m========== %s ==========\033[0m\n" "$*"; }

# --- Show environment ---------------------------------------------------------
cat <<EOF
[ENV]
  ROOT_PATH          = ${ROOT_PATH}
  EXTERNAL_PATH      = ${EXTERNAL_PATH}
  AI_EDGE_TORCH_PATH = ${AI_EDGE_TORCH_PATH}
  LITERT_PATH        = ${LITERT_PATH}
  BUILD_TYPE         = ${BUILD_TYPE}
EOF
# TENSORFLOW_PATH    = ${TENSORFLOW_PATH}
# ------------------------------------------------------------------------------
# 1. Ensure external directory exists
# ------------------------------------------------------------------------------
ensure_dir "${EXTERNAL_PATH}"

# ------------------------------------------------------------------------------
# 2. Clone ai-edge-torch and create sample-app symlink
# ------------------------------------------------------------------------------
banner "Installing ai-edge-torch"
if [[ ! -d "${AI_EDGE_TORCH_PATH}" ]]; then
#   run git clone https://github.com/SNU-RTOS/ai-edge-torch.git "${AI_EDGE_TORCH_PATH}"

#   # Link our app sources into examples/cpp
#   EX_CPP="${AI_EDGE_TORCH_PATH}/ai_edge_torch/generative/examples/cpp"
#   [[ -e "${EX_CPP}" ]] && run rm -rf "${EX_CPP}"
#   run ln -s "${LLM_APP_SRC}" "${EX_CPP}"

#   # Update TensorFlow path in WORKSPACE
#   WORKSPACE="${AI_EDGE_TORCH_PATH}/WORKSPACE"
#   run sed -i "s|path = \".*\"|path = \"${TENSORFLOW_PATH}\"|" "${WORKSPACE}"
else
  echo "[SKIP] ai-edge-torch already present."
fi

# ------------------------------------------------------------------------------
# 3. Clone LiteRT at the specified commit
# ------------------------------------------------------------------------------
# banner "Installing TensorFlow (${TF_COMMIT})"
# if [[ ! -d "${TENSORFLOW_PATH}" ]]; then
#   ensure_dir "${TENSORFLOW_PATH}"
#   pushd "${TENSORFLOW_PATH}" >/dev/null
#   run git init .
#   run git remote add origin https://github.com/tensorflow/tensorflow.git
#   run git fetch --depth 1 origin "${TF_COMMIT}"
#   run git checkout FETCH_HEAD
#   echo "[INFO] Applying LiteRT patch ..."
#   run patch -p1 <"${AI_EDGE_TORCH_PATH}/bazel/org_tensorflow_system_python.diff"
#   popd >/dev/null
# else
#   echo "[SKIP] TensorFlow already present."
# fi

banner "Installing LiteRT"
if [[ ! -d "${LITERT_PATH}" ]]; then
  ensure_dir "${LITERT_PATH}"
  pushd "${LITERT_PATH}" >/dev/null
  pwd
  run git clone https://github.com/google-ai-edge/litert.git
  cd LiteRT
  pwd
  ./configure
  popd >/dev/null
else
  echo "[SKIP] LiteRT already present."
fi
# ------------------------------------------------------------------------------
# 4. Create local build directories (inc/lib/obj/output)
# ------------------------------------------------------------------------------
for d in inc lib obj output; do ensure_dir "${ROOT_PATH}/${d}"; done

# ------------------------------------------------------------------------------
# 5. Build LiteRT and its dependencies
# ------------------------------------------------------------------------------
banner "Building LiteRT and delegates (${BUILD_TYPE})"
pushd "${ROOT_PATH}/scripts" >/dev/null
run ./build-litert.sh              "${BUILD_TYPE}"
run ./build-litert_gpu_delegate.sh "${BUILD_TYPE}"
run ./build-deps.sh                "${BUILD_TYPE}"
popd >/dev/null

# ------------------------------------------------------------------------------
# 6. Build the LLM inference application
# ------------------------------------------------------------------------------
# banner "Building LLM inference app (${BUILD_TYPE})"
# run "${ROOT_PATH}/build.sh" "${BUILD_TYPE}"

banner "Setup complete"
