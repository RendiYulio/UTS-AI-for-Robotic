#!/usr/bin/env bash
# run.sh - jalankan sistem pengenalan rambu + kendali LED.
# Menyiapkan environment (setenv.sh) lalu menjalankan aplikasi.
#
# Contoh:
#   ./run.sh                 # kamera + GPIO + tampilan
#   ./run.sh --no-gpio       # uji tanpa LED
#   ./run.sh --no-display    # headless/SSH
#   ./run.sh --camera 1 --conf 0.5

DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$DIR/setenv.sh" || exit 1
exec python3 -u "$DIR/sign_led.py" "$@"
