"""
sign_led.py - Sistem pengenalan rambu (sign) + kendali LED via GPIO Jetson Nano.

Alur:  Camera -> Sign Recognition (YOLOv8-seg + TensorRT) -> Decision Logic
       -> GPIO -> LED

Jalankan:
    source setenv.sh
    python3 sign_led.py                 # kamera + GPIO + tampilan (auto)
    python3 sign_led.py --no-gpio       # uji tanpa LED
    python3 sign_led.py --no-display    # uji headless / SSH

Rambu yang dikenali : GO -> hijau, SLOW -> kuning, STOP -> merah.
Kondisi aman (tidak ada rambu / confidence < threshold): lihat SAFE_ACTION.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time

import numpy as np
import cv2

from yolo_trt import YoloTRT, CLASS_NAMES

# ---------------------------------------------------------------------------
# KONFIGURASI (silakan sesuaikan dengan wiring kelompok)
# ---------------------------------------------------------------------------
CONFIG = {
    "engine": "best_fp16.engine",
    "conf_thres": 0.40,          # ambang confidence; di bawah ini -> kondisi aman
    "iou_thres": 0.45,
    "hold_seconds": 1.0,         # tahan keputusan terakhir saat frame kosong
    "camera": {
        "index": 0,
        "width": 1280,
        "height": 720,
        "fps": 30,
        # Kamera CSI Jetson (nvarguscamerasrc). Set None untuk kamera USB (V4L2).
        # Catatan: sensor IMX219 tidak punya mode 1280x720@30; pakai 1080p@30
        # lalu di-scale nvvidconv ke 1280x720.
        "gst": ("nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! "
                "nvvidconv ! video/x-raw,format=BGRx,width=1280,height=720 ! "
                "videoconvert ! video/x-raw,format=BGR ! fdsink fd=1 sync=false"),
    },
    # Pemetaan nomor pin fisik header Jetson Nano (mode BOARD)
    "led_pins": {
        "red": 11,      # STOP
        "yellow": 13,   # SLOW
        "green": 18,    # GO
    },
    "sign_to_led": {"STOP": "red", "SLOW": "yellow", "GO": "green"},
    # Polaritas wiring LED:
    #   False = active-high: GPIO(pin) -> R -> LED(+) -> LED(-) -> GND
    #   True  = active-low : 3.3V -> R -> LED(+) -> LED(-) -> GPIO(pin) (GPIO sink)
    "active_low": False,
    # Kondisi aman saat rambu tidak dikenali / confidence rendah:
    #   "STOP" -> LED merah menyala (paling konservatif untuk robot)
    #   "NONE" -> semua LED mati
    #   "BLINK"-> LED merah berkedip (tanda hazard)
    "safe_action": "NONE",
}

LED_COLORS = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 200, 0)}


# ---------------------------------------------------------------------------
# GPIO
# ---------------------------------------------------------------------------
class LedController(object):
    def __init__(self, pins, enabled, active_low=False):
        self.enabled = enabled
        self.pins = pins
        self.active_low = active_low
        self.GPIO = None
        self._on = None
        self._off = None
        if enabled:
            import Jetson.GPIO as GPIO
            self.GPIO = GPIO
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BOARD)
            self._on = GPIO.LOW if active_low else GPIO.HIGH
            self._off = GPIO.HIGH if active_low else GPIO.LOW
            for pin in pins.values():
                GPIO.setup(pin, GPIO.OUT, initial=self._off)

    def output(self, active):
        """active: nama warna yang menyala, atau None untuk semua mati."""
        if not self.enabled:
            return
        for name, pin in self.pins.items():
            self.GPIO.output(pin, self._on if name == active else self._off)

    def cleanup(self):
        if self.enabled and self.GPIO is not None:
            self.GPIO.cleanup()


# ---------------------------------------------------------------------------
# Logika keputusan
# ---------------------------------------------------------------------------
class DecisionEngine(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_label = None
        self.last_conf = 0.0
        self.last_time = 0.0

    def decide(self, detections, now):
        """Kembalikan (sign_label, led_color, is_safe, confidence)."""
        best = detections[0] if detections else None
        if best is not None and best[4] >= self.cfg["conf_thres"]:
            label = CLASS_NAMES[best[5]]
            led = self.cfg["sign_to_led"].get(label)
            self.last_label, self.last_conf, self.last_time = label, best[4], now
            return label, led, False, best[4]

        # tidak ada deteksi / confidence rendah
        if (self.last_label is not None and
                (now - self.last_time) < self.cfg["hold_seconds"]):
            label = self.last_label
            led = self.cfg["sign_to_led"].get(label)
            return label, led, False, self.last_conf

        return self._safe_state()

    def _safe_state(self):
        action = self.cfg["safe_action"]
        if action == "NONE":
            return "SAFE", None, True, 0.0
        if action == "BLINK":
            return "SAFE", "blink", True, 0.0
        # default: perlakukan sebagai STOP
        return "SAFE", self.cfg["sign_to_led"].get(action, "red"), True, 0.0


# ---------------------------------------------------------------------------
# Kamera
# ---------------------------------------------------------------------------
class GstCamera(object):
    """Tangkap frame BGR dari pipeline GStreamer via subprocess gst-launch-1.0.

    Dipakai karena:
      - OpenCV wheel ini tidak dibangun dengan GStreamer (kamera CSI tidak bisa
        dibaca lewat cv2.VideoCapture V4L2 -> hasilnya frame hijau polos), dan
      - binding gi/Gst di GStreamer 1.14 tidak mengekspos appsink.pull_sample.

    Pipeline harus diakhiri `fdsink fd=1`; frame BGR mentah dibaca dari stdout.
    """

    def __init__(self, pipeline_str, width, height):
        self.width = int(width)
        self.height = int(height)
        self._frame_bytes = self.width * self.height * 3
        cmd = ["gst-launch-1.0", "-q"] + shlex.split(pipeline_str)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=0)
        self._opened = self.proc.poll() is None

    def isOpened(self):
        return self._opened and self.proc.poll() is None

    def read(self):
        if not self.isOpened():
            return False, None
        n = self._frame_bytes
        chunks = []
        got = 0
        while got < n:
            chunk = self.proc.stdout.read(n - got)
            if not chunk:
                return False, None
            chunks.append(chunk)
            got += len(chunk)
        frame = np.frombuffer(b"".join(chunks), dtype=np.uint8)
        frame = frame.reshape(self.height, self.width, 3)   # BGR
        return True, frame.copy()

    def release(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
            finally:
                if self.proc.stdout is not None:
                    self.proc.stdout.close()
                self.proc = None
        self._opened = False


class NullCamera(object):
    def isOpened(self):
        return False

    def read(self):
        return False, None

    def release(self):
        pass


def open_camera(ccfg):
    if ccfg.get("gst"):
        for attempt in range(1, 4):
            try:
                cam = GstCamera(ccfg["gst"], ccfg["width"], ccfg["height"])
                for _ in range(60):                 # warmup sampai frame valid
                    ok, _ = cam.read()
                    if ok:
                        return cam
                cam.release()
                print("[kamera] percobaan %d: pipeline tidak menghasilkan frame." % attempt)
            except Exception as e:
                print("[kamera] percobaan %d gagal: %s" % (attempt, e))
            time.sleep(1.0)
        print("[kamera] ERROR: kamera CSI (nvarguscamerasrc) gagal dibuka.")
        print("        Penyebab umum: daemon Argus macet. Jalankan:")
        print("            sudo systemctl restart nvargus-daemon")
        print("        lalu ulangi. (Atau pakai kamera USB.)")
        return NullCamera()

    cap = cv2.VideoCapture(ccfg["index"], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ccfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ccfg["height"])
    cap.set(cv2.CAP_PROP_FPS, ccfg["fps"])
    return cap


# ---------------------------------------------------------------------------
# Tampilan
# ---------------------------------------------------------------------------
def display_available():
    """Cek apakah OpenCV punya dukungan GUI (imshow) dan DISPLAY tersedia."""
    if not os.environ.get("DISPLAY"):
        return False, "DISPLAY tidak di-set"
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__probe__")
        return True, ""
    except cv2.error:
        return False, "OpenCV tanpa dukungan GUI (build headless)"


def draw(frame, detections, label, led, is_safe, conf):
    for (x1, y1, x2, y2, c, cid) in detections:
        color = LED_COLORS.get(CONFIG["sign_to_led"].get(CLASS_NAMES[cid]), (255, 255, 255))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, "%s %.2f" % (CLASS_NAMES[cid], c),
                    (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    status = "SAFE" if is_safe else label
    line1 = "Sign : %s" % status
    line2 = "LED  : %s" % (led if led else "OFF")
    line3 = "Conf : %.2f" % conf
    cv2.rectangle(frame, (0, 0), (260, 92), (0, 0, 0), -1)
    cv2.putText(frame, line1, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, line2, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, line3, (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpio", action="store_true", help="nonaktifkan GPIO (mode uji)")
    ap.add_argument("--no-display", action="store_true", help="tanpa jendela OpenCV (headless)")
    ap.add_argument("--camera", type=int, default=None, help="indeks kamera V4L2")
    ap.add_argument("--conf", type=float, default=None, help="ambang confidence")
    args = ap.parse_args()

    cfg = CONFIG
    if args.camera is not None:
        cfg["camera"]["index"] = args.camera
    if args.conf is not None:
        cfg["conf_thres"] = args.conf

    display = False
    if not args.no_display:
        display, reason = display_available()
        if not display:
            print("[info] mode headless (%s) -> status lewat terminal." % reason)

    print("[info] memuat engine TensorRT...")
    model = YoloTRT(cfg["engine"], conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])

    led = LedController(cfg["led_pins"], enabled=not args.no_gpio,
                        active_low=cfg.get("active_low", False))
    decision = DecisionEngine(cfg)

    print("[info] membuka kamera...")
    cap = open_camera(cfg["camera"])
    if not cap.isOpened():
        print("[error] kamera tidak bisa dibuka.", file=sys.stderr)
        led.cleanup()
        return 1

    print("[info] sistem berjalan. tekan 'q' untuk keluar.\n")
    blink_on = False
    last_blink = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[warn] gagal baca frame.")
                continue

            t0 = time.time()
            dets, _, _ = model.infer(frame)
            label, led_color, is_safe, conf = decision.decide(dets, time.time())
            fps = 1.0 / max(1e-6, time.time() - t0)

            # kendalikan LED
            if led_color == "blink":
                if time.time() - last_blink > 0.25:
                    blink_on = not blink_on
                    last_blink = time.time()
                led.output("red" if blink_on else None)
            else:
                led.output(led_color)

            print("Sign=%-5s LED=%-6s Conf=%.2f FPS=%.1f" %
                  (label, led_color or "OFF", conf, fps))

            if display:
                try:
                    cv2.imshow("Sign Recognition + LED",
                               draw(frame, dets, label, led_color, is_safe, conf))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as e:
                    print("[warn] tampilan gagal (%s) -> lanjut headless." % e)
                    display = False
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        led.cleanup()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("\n[info] selesai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
