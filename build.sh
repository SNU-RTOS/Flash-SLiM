#!/usr/bin/env bash
# build.sh
#
# Build the LLM sample using Bazel build system.
#
# Usage:
#   ./build.sh            # defaults to "main"
#   ./build.sh main       # build main text generator
#   ./build.sh clean      # clean build artifacts
#   ./build.sh test       # run tests
#   ./build.sh all        # build all targets
#
# Resulting binaries:
#   bazel-bin/flash-slim/text_generator_main
#
# The binary will be copied to output/ directory for convenience.

set -euo pipefail


# ---------------------------------------------------------------------------
# 0. Environment & Build Config
# ---------------------------------------------------------------------------
source .env
source ./scripts/utils.sh
source ./scripts/common.sh
: "${ROOT_PATH:?ROOT_PATH must be set in .env}"

# Apply build configuration (release by default)
setup_build_config

# ---------------------------------------------------------------------------
# Print build configuration
# ---------------------------------------------------------------------------
banner "Bazel Build Configuration"
log "  BAZEL_CONF: $BAZEL_CONF"
log "  COPT_FLAGS: $COPT_FLAGS"
log "  LINKOPTS: $LINKOPTS"
log "  GPU_FLAGS: $GPU_FLAGS"
log "  GPU_COPT_FLAGS: $GPU_COPT_FLAGS"

APP_BASE=text_generator_main
OUT_DIR=output
BAZEL_BIN="bazel-bin/flash-slim/${APP_BASE}"
OUTPUT_BIN="${OUT_DIR}/${APP_BASE}"

# ---------------------------------------------------------------------------
# 1. Parse argument
# ---------------------------------------------------------------------------
BUILD_TARGET="${1:-main}"    # default = main

case "${BUILD_TARGET}" in
    main|clean|test|all) ;;
    *)
        echo "Invalid target '${BUILD_TARGET}' (expected main|clean|test|all)" >&2
        echo "Usage: $0 [main|clean|test|all]" >&2
        exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# 2. Build helpers
# ---------------------------------------------------------------------------
run() { echo "+ $*"; "$@"; }

do_main_build() {
    banner "Building main text generator with Bazel"
    
    bazel build $BAZEL_CONF \
        //flash-slim:text_generator_main \
        $COPT_FLAGS \
        $LINKOPTS \
        $GPU_FLAGS \
        $GPU_COPT_FLAGS \
        --config=linux \
        --config=ebpf

    ensure_dir "${OUT_DIR}"
    banner "Copying binary to output directory"
    if [[ -f "${BAZEL_BIN}" ]]; then
        [[ -f "${OUTPUT_BIN}" ]] && rm -f "${OUTPUT_BIN}"
        cp "${BAZEL_BIN}" "${OUTPUT_BIN}"
        log "Binary copied to ${OUTPUT_BIN}"
    else
        banner "Binary not found"
        log "[ERROR] Binary not found at ${BAZEL_BIN}"
        exit 1
    fi
}

do_clean() {
    banner "Cleaning build artifacts"
    run bazel clean
    rm -rf "${OUT_DIR}"
    log "Clean complete"
}

do_test() {
    banner "Running tests"
    run bazel test //flash-slim:all
    log "Tests complete"
}

do_all_build() {
    banner "Building all targets"
    run bazel build $BAZEL_CONF //flash-slim:all $COPT_FLAGS $LINKOPTS $GPU_FLAGS $GPU_COPT_FLAGS

    ensure_dir "${OUT_DIR}"
    banner "Copying main binary to output directory"
    if [[ -f "${BAZEL_BIN}" ]]; then
        [[ -f "${OUTPUT_BIN}" ]] && rm -f "${OUTPUT_BIN}"
        cp "${BAZEL_BIN}" "${OUTPUT_BIN}"
        log "Main binary copied to ${OUTPUT_BIN}"
    fi

    banner "Copying additional binaries"
    for binary in bazel-bin/flash-slim/*; do
        if [[ -f "$binary" && -x "$binary" && "$binary" != "${BAZEL_BIN}" ]]; then
            binary_name=$(basename "$binary")
            cp "$binary" "${OUT_DIR}/${binary_name}"
            log "Additional binary copied to ${OUT_DIR}/${binary_name}"
        fi
    done
}

# ---------------------------------------------------------------------------
# 3. Invoke builds
# ---------------------------------------------------------------------------
case "${BUILD_TARGET}" in
    main) do_main_build ;;
    clean) do_clean ;;
    test) do_test ;;
    all) do_all_build ;;
esac

echo "[SUCCESS] Finished build target='${BUILD_TARGET}'."

# ---------------------------------------------------------------------------
# 4. Display build info
# ---------------------------------------------------------------------------
if [[ "${BUILD_TARGET}" != "clean" ]]; then
    echo ""
    echo "Build Information:"
    echo "  Target: ${BUILD_TARGET}"
    echo "  Bazel workspace: $(pwd)"
    echo "  Main binary: ${OUTPUT_BIN}"
    
    if [[ -f "${OUTPUT_BIN}" ]]; then
        echo "  Binary size: $(ls -lh "${OUTPUT_BIN}" | awk '{print $5}')"
        echo "  Binary permissions: $(ls -l "${OUTPUT_BIN}" | awk '{print $1}')"
    fi
fi

echo "[INFO] BUILD COMPLETE"
