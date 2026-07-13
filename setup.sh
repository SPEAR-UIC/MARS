#!/usr/bin/env bash
# setup.sh — CQSimPrivate project setup
#
# Installs CMake (optional), builds the C++ simulator, and sets up the Python environment.
#
# Usage:
#   ./setup.sh                    # full setup: C++ build + Python venv
#   ./setup.sh --cpp              # C++ build only
#   ./setup.sh --python           # Python venv only
#   ./setup.sh --install-cmake    # install a recent CMake from upstream, then build
#   ./setup.sh --clean            # wipe build/ and .venv/ then redo full setup
#   ./setup.sh --help

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$REPO_DIR/build"
VENV_DIR="$REPO_DIR/.venv"
PYTHON="${PYTHON:-python3}"
CMAKE_VERSION="${CMAKE_VERSION:-4.2.3}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[setup]${NC} $*"; }
success() { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[setup]${NC} $*"; }
error()   { echo -e "${RED}[setup]${NC} $*" >&2; exit 1; }

# ── Parse flags ──────────────────────────────────────────────────────────────
DO_CPP=true
DO_PYTHON=true
INSTALL_CMAKE=false
CLEAN=false

for arg in "$@"; do
    case "$arg" in
        --cpp)           DO_PYTHON=false ;;
        --python)        DO_CPP=false ;;
        --install-cmake) INSTALL_CMAKE=true ;;
        --clean)         CLEAN=true ;;
        --help|-h)
            echo "Usage: $0 [--cpp] [--python] [--install-cmake] [--clean]"
            echo ""
            echo "  --cpp            Build the C++ simulator only (skip Python)"
            echo "  --python         Set up Python venv only (skip C++ build)"
            echo "  --install-cmake  Download and install CMake ${CMAKE_VERSION} from upstream"
            echo "                   (use when system cmake is too old)"
            echo "  --clean          Remove build/ and .venv/ before setup"
            echo ""
            echo "Environment variables:"
            echo "  CMAKE_VERSION    CMake version to install (default: ${CMAKE_VERSION})"
            echo "  PYTHON           Python executable to use  (default: python3)"
            exit 0 ;;
        *) error "Unknown argument: $arg (run $0 --help)" ;;
    esac
done

# ── Clean ────────────────────────────────────────────────────────────────────
if $CLEAN; then
    info "Cleaning build/ and .venv/ ..."
    rm -rf "$BUILD_DIR" "$VENV_DIR"
fi

# ── CMake installation (optional) ────────────────────────────────────────────
install_cmake_upstream() {
    local ver="$1"
    local arch
    arch="$(uname -m)"
    local tarball="cmake-${ver}-linux-${arch}.tar.gz"
    local url="https://github.com/Kitware/CMake/releases/download/v${ver}/${tarball}"

    info "Installing CMake ${ver} for ${arch} from upstream..."

    # Remove system cmake if present
    if dpkg -l cmake &>/dev/null 2>&1; then
        info "  Removing system cmake package..."
        sudo apt-get remove --purge cmake -y
    fi

    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT

    info "  Downloading ${url} ..."
    wget -q --show-progress -O "${tmp}/${tarball}" "$url"

    info "  Installing to /usr/local ..."
    sudo tar -xzf "${tmp}/${tarball}" -C /usr/local --strip-components=1

    success "  CMake installed: $(cmake --version | head -1)"
}

if $INSTALL_CMAKE; then
    install_cmake_upstream "$CMAKE_VERSION"
fi

# ── C++ build ────────────────────────────────────────────────────────────────
if $DO_CPP; then
    info "=== C++ build ==="

    # Check cmake
    if ! command -v cmake &>/dev/null; then
        warn "cmake not found. Run:  $0 --install-cmake"
        error "cmake required for C++ build."
    fi
    CMAKE_VER_INSTALLED="$(cmake --version | head -1 | awk '{print $3}')"
    info "  Using CMake ${CMAKE_VER_INSTALLED}"

    # Check C++ compiler
    if ! command -v g++ &>/dev/null && ! command -v clang++ &>/dev/null; then
        error "No C++ compiler found. Install: sudo apt-get install build-essential"
    fi

    # Check OpenMP (required by parallel MCTS)
    if ! g++ -fopenmp -x c++ - -o /dev/null <<< '#include<omp.h>
int main(){return 0;}' 2>/dev/null; then
        info "  OpenMP not found; installing libomp-dev..."
        sudo apt-get install -y libomp-dev
    fi
    info "  OpenMP OK"

    mkdir -p "$BUILD_DIR"

    info "Configuring CMake (Release, -O3 -march=native)..."
    cmake -S "$REPO_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-O3 -march=native" \
        2>&1 | grep -v "^--" || true

    NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    info "Building cqsimcpp (using $NPROC cores)..."
    cmake --build "$BUILD_DIR" --target cqsimcpp -- -j"$NPROC"

    # Copy binary to repo root for convenient use
    cp "$BUILD_DIR/cqsimcpp" "$REPO_DIR/cqsimcpp"
    success "C++ build complete → $REPO_DIR/cqsimcpp"
fi

# ── Python venv ───────────────────────────────────────────────────────────────
if $DO_PYTHON; then
    info "=== Python setup ==="

    # Verify Python
    if ! command -v "$PYTHON" &>/dev/null; then
        error "$PYTHON not found. Install Python 3.8+ first."
    fi
    PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    info "  Using $PYTHON (version $PY_VERSION)"
    "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)" \
        || error "Python 3.8+ required (found $PY_VERSION)"

    # Create or reuse venv
    if [ ! -d "$VENV_DIR" ]; then
        info "Creating virtual environment at .venv/ ..."
        "$PYTHON" -m venv "$VENV_DIR"
    else
        info "Reusing existing virtual environment at .venv/ ..."
    fi

    VENV_PY="$VENV_DIR/bin/python"
    VENV_PIP="$VENV_DIR/bin/pip"

    info "Upgrading pip..."
    "$VENV_PIP" install --upgrade pip --quiet

    info "Installing Python dependencies..."
    "$VENV_PIP" install \
        "numpy>=1.21" \
        "scipy>=1.7" \
        "matplotlib>=3.5" \
        --quiet

    # Sanity check
    INSTALLED=$("$VENV_PY" -c "
import numpy as np, scipy, matplotlib as mpl
print(f'numpy {np.__version__}, scipy {scipy.__version__}, matplotlib {mpl.__version__}')
")
    success "Python venv ready — $INSTALLED"
    info "  Activate with:  source .venv/bin/activate"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
success "=== Setup complete ==="
if $DO_CPP;    then echo "  Simulator:  $REPO_DIR/cqsimcpp"; fi
if $DO_PYTHON; then echo "  Python env: $VENV_DIR"; fi
echo ""
echo "Quick start:"
if $DO_PYTHON; then echo "  source .venv/bin/activate"; fi
echo "  ./cqsimcpp run config experiments/exp4a.json"
echo "  python3 scripts/analyze_hot_windows.py"
echo "  python3 scripts/plot_window_sensitivity.py experiments/exp4a.json"
