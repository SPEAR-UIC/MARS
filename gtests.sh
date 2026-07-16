#!/usr/bin/env bash
# test.sh — Build (if needed) and run the gtest suite.
# Usage:
#   bash test.sh                    # run all tests
#   bash test.sh --filter FooTest   # run tests matching a pattern
#   bash test.sh --verbose          # show per-test output
#   bash test.sh --rebuild          # force a fresh build before testing

set -euo pipefail

# ==============================================================================
# Defaults
# ==============================================================================
FILTER=""
VERBOSE=false
REBUILD=false

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"

# ==============================================================================
# Argument parsing
# ==============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter)
            shift
            FILTER="${1:-}"
            ;;
        --verbose) VERBOSE=true ;;
        --rebuild) REBUILD=true ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash test.sh [--filter PATTERN] [--verbose] [--rebuild]"
            exit 1
            ;;
    esac
    shift
done

# ==============================================================================
# Build
# ==============================================================================
if $REBUILD || [[ ! -d "${BUILD_DIR}" ]]; then
    echo "Running build first..."
    bash "${PROJECT_DIR}/build.sh"
fi

# ==============================================================================
# Run tests via ctest
# ==============================================================================
CTEST_ARGS=("--test-dir" "${BUILD_DIR}" "--output-on-failure")

if [[ -n "${FILTER}" ]]; then
    CTEST_ARGS+=("--tests-regex" "${FILTER}")
fi

if $VERBOSE; then
    CTEST_ARGS+=("-V")
fi

echo "Running tests..."
ctest "${CTEST_ARGS[@]}"
