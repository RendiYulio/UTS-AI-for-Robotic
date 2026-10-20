#!/usr/bin/env bash
# setenv.sh - menyiapkan lingkungan Jetson Nano untuk GPU (CUDA + TensorRT).
#
# Dua hal yang dilakukan:
#  1. Memperbaiki bug OpenBLAS pada kernel tegra 4.9 (instruksi `mrs midr_el1`
#     memicu SIGILL). Solusinya memaksa tipe core OpenBLAS.
#  2. Menambahkan library CUDA 10.2 + TensorRT 8.2 + cuDNN + cuBLAS yang ada di
#     rootfs SD card ke LD_LIBRARY_PATH, serta modul python `tensorrt`.
#
# Pakai:  source ./setenv.sh

# 1) Fix SIGILL pada numpy/OpenBLAS
export OPENBLAS_CORETYPE=CORTEXA57

# 2) Cari rootfs SD card yang berisi TensorRT
SD_ROOT=""
for base in /media/*/* /mnt/* /media/*; do
    if [ -e "$base/usr/lib/aarch64-linux-gnu/libnvinfer.so.8" ]; then
        SD_ROOT="$base"
        break
    fi
done

if [ -z "$SD_ROOT" ]; then
    echo "[setenv] ERROR: rootfs SD dengan TensorRT tidak ditemukan." >&2
    echo "[setenv]        Pastikan SD card ter-mount (cek: ls /media/*/)." >&2
    return 1 2>/dev/null || exit 1
fi

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export SD_ROOT
export APP_DIR

# 3) Library runtime dari SD card
export LD_LIBRARY_PATH="$SD_ROOT/usr/local/cuda-10.2/targets/aarch64-linux/lib:$SD_ROOT/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"

# 4) Modul python tensorrt (symlink khusus agar numpy sistem tidak tertimpa)
mkdir -p "$APP_DIR/pylib"
if [ ! -e "$APP_DIR/pylib/tensorrt" ]; then
    ln -s "$SD_ROOT/usr/lib/python3.6/dist-packages/tensorrt" "$APP_DIR/pylib/tensorrt"
fi
export PYTHONPATH="$APP_DIR/pylib:$PYTHONPATH"

echo "[setenv] SD_ROOT = $SD_ROOT"
echo "[setenv] CUDA/TensorRT siap (GPU Jetson Nano)."
