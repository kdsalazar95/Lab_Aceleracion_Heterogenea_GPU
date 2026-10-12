#!/usr/bin/env bash
# Configura (si hace falta) y compila la version CPU y todas las versiones GPU.
# Uso: scripts/build_all.sh            # RTX 4060 Ti (sm_89)
#      GPU_ARCH=53 scripts/build_all.sh  # Jetson Nano
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_ARCH="${GPU_ARCH:-89}"

# nvcc no acepta los LDFLAGS que conda define por defecto.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LDFLAGS="-L$CONDA_PREFIX/lib"
fi

if [[ ! -f build/build.ninja ]]; then
    meson setup build -Dgpu_arch="$GPU_ARCH"
fi
meson compile -C build
