"""
test_led.py - Uji wiring 3 LED (merah=pin11, kuning=pin13, hijau=pin18).

Menjalankan beberapa fase bergantian agar mudah memastikan:
  Fase 1 : semua MATI          -> semua LED harus padam
  Fase 2 : hanya MERAH nyala
  Fase 3 : hanya KUNING nyala
  Fase 4 : hanya HIJAU nyala
  Fase 5 : semua NYALA
  Fase 6 : semua MATI

Jalankan:
    python3 test_led.py
    python3 test_led.py --active-low    # bila wiring common-anode (GPIO sink)
"""

import argparse
import time

import Jetson.GPIO as GPIO

PINS = {"merah": 11, "kuning": 13, "hijau": 18}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--active-low", action="store_true",
                    help="wiring common-anode: LED nyala saat pin LOW")
    args = ap.parse_args()

    ON = GPIO.LOW if args.active_low else GPIO.HIGH
    OFF = GPIO.HIGH if args.active_low else GPIO.LOW

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    for p in PINS.values():
        GPIO.setup(p, GPIO.OUT, initial=OFF)

    def all_set(state):
        for p in PINS.values():
            GPIO.output(p, state)

    try:
        print("Fase 1: SEMUA MATI (3 dtk)  -> semua LED harus PADAM")
        all_set(OFF); time.sleep(3)
        for name, p in PINS.items():
            print("Fase  : hanya %s (2 dtk) -> cek LED %s" % (name.upper(), name))
            all_set(OFF); GPIO.output(p, ON); time.sleep(2)
        print("Fase  : SEMUA NYALA (2 dtk)")
        all_set(ON); time.sleep(2)
        all_set(OFF)
        print("Selesai. (active_low=%s)" % args.active_low)
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
