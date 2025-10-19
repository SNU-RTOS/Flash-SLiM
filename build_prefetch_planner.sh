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
if [[ ! -f .env ]]; then
    echo "[ERROR] .env file not found. Please run from project root directory." >&2
    exit 1
fi

source .env
source ./scripts/utils.sh

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

APP_NAME="prefetch_planner"
OUT_DIR=bin
BAZEL_BIN="bazel-bin/flash-slim/${APP_NAME}"
OUTPUT_BIN="${OUT_DIR}/${APP_NAME}"

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

setup_dependencies() {
    banner "Setting up external dependencies"
    
    # Use variables from .env, with fallbacks to default values
    local external_dir="${EXTERNAL_DIR:-external}"
    local litert_repo="${LITERT_REPO_URL:-https://github.com/Seunmul/LiteRT.git}"
    local litert_dir="$external_dir/litert" # Corrected to match WORKSPACE
    local XNNPACK_repo="${XNNPACK_REPO_URL:-https://github.com/Seunmul/XNNPACK.git}"
    local XNNPACK_dir="$external_dir/XNNPACK"

    ensure_dir "$external_dir"

    if [ ! -d "$litert_dir" ]; then
        log "Cloning LiteRT repository into $litert_dir..."
        git clone "$litert_repo" "$litert_dir" --depth 1
    else
        log "LiteRT repository already exists. Skipping clone."
    fi

    if [ ! -d "$XNNPACK_dir" ]; then
        log "Cloning XNNPACK repository into $XNNPACK_dir..."
        git clone "$XNNPACK_repo" "$XNNPACK_dir" --depth 1
    else
        log "XNNPACK repository already exists. Skipping clone."
    fi

}

do_main_build() {
    banner "Building main text generator with Bazel"
    
    local arch_optimize_config=""
    # Determine architecture
    arch=$(uname -m)

    # Set config based on architecture
    case "${arch}" in
        x86_64)
            arch_optimize_config="--config=avx_linux"
            ;;
        aarch64*)
            arch_optimize_config="--config=linux_arm64"
            ;;
        *)
            echo "Unsupported architecture: ${arch}. Using default config."
            arch_optimize_config=""
            ;;
    esac

    bazel $BAZEL_LAUNCH_CONF \
        build $BAZEL_CONF \
        //flash-slim:$APP_NAME \
        $COPT_FLAGS \
        $LINKOPTS \
        $GPU_FLAGS \
        $GPU_COPT_FLAGS \
        $arch_optimize_config \
        --config=ebpf \
        --config=weight_streaming

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
# Setup dependencies before building
# ---------------------------------------------------------------------------
if [[ "${BUILD_TARGET}" != "clean" ]]; then
    setup_dependencies
fi

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
