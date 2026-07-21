#!/usr/bin/env bash
# build.sh — Configure and compile hpcsched.
# Usage:
#   bash build.sh                   # debug build (default)
#   bash build.sh --release         # release build
#   bash build.sh --asan            # debug + AddressSanitizer
#   bash build.sh --tsan            # debug + ThreadSanitizer
#   bash build.sh --clean           # wipe build dir first, then debug build

set -euo pipefail

# ==============================================================================
# Defaults
# ==============================================================================
BUILD_TYPE="Debug"
USE_ASAN="OFF"
USE_TSAN="OFF"
CLEAN=false

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
BINARY_NAME="cqsimcpp"

# ==============================================================================
# Argument parsing
# ==============================================================================
for arg in "$@"; do
    case "$arg" in
        --release) BUILD_TYPE="Release" ;;
        --asan)    USE_ASAN="ON" ;;
        --tsan)    USE_TSAN="ON" ;;
        --clean)   CLEAN=true ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash build.sh [--release] [--asan] [--tsan] [--clean]"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Environment
# ==============================================================================
module load cmake 2>/dev/null || true

if ! command -v cmake &>/dev/null; then
    echo "ERROR: cmake not found. Load the module or install cmake."
    exit 1
fi

# ==============================================================================
# Build
# ==============================================================================
if $CLEAN; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
fi

if [[ -f "${PROJECT_DIR}/${BINARY_NAME}" ]]; then
    echo "Removing existing ${BINARY_NAME} at repo root..."
    rm -f "${PROJECT_DIR}/${BINARY_NAME}"
fi

mkdir -p "${BUILD_DIR}"

echo "Configuring (type=${BUILD_TYPE}, ASAN=${USE_ASAN}, TSAN=${USE_TSAN})..."
cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"      \
    -DUSE_ASAN="${USE_ASAN}"                \
    -DUSE_TSAN="${USE_TSAN}"

echo "Compiling..."
cmake --build "${BUILD_DIR}" -- -j"$(nproc)"

echo "Copying ${BINARY_NAME} to repo root..."
cp "${BUILD_DIR}/${BINARY_NAME}" "${PROJECT_DIR}/${BINARY_NAME}"

echo "Build complete: ${BUILD_DIR}"
echo "Binary available at: ${PROJECT_DIR}/${BINARY_NAME}"
