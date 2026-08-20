#!/usr/bin/env bash
# Configure and build. Defaults to a fat binary spanning Turing to Hopper;
# pass an architecture to cut compile time when targeting one machine.
#
#   scripts/build.sh              # everything, for rental GPUs of any vintage
#   scripts/build.sh 87           # AGX Orin only - much faster to compile
#   scripts/build.sh 86           # A10 / A100-adjacent / RTX 30xx
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
ARCH="${1:-}"

CMAKE_ARGS=(-S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release)
if [[ -n "$ARCH" ]]; then
  CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=$ARCH")
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

echo
echo "built into $BUILD_DIR"
ls -1 "$BUILD_DIR"/src/hsi_detect "$BUILD_DIR"/bench/bench_sam 2>/dev/null || true
