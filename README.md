# AIROBOT — Sistem Pengenalan Rambu + Kendali LED (Jetson Nano) Kelompok 3

Rendi Yulio Pramudita - 2023071047
Rizky Kamil Ajie - 2023071051
Ferdinan Indra Fata - 2023071059
Fadil Muhamad Prasetya - 2023071068
Raihan Ilham Ramadhan - 2023071075

Sistem yang membaca gambar rambu dari kamera, mengenali jenis rambu
(**GO / SLOW / STOP**) memakai **YOLOv8-seg + TensorRT** di GPU Jetson Nano,
lalu menyalakan **LED** yang sesuai melalui **GPIO**. Bila tidak ada rambu,
semua LED mati.

---

## 1. Ringkasan

| Aspek | Keterangan |
|-------|------------|
| Tujuan | Mengenali rambu (GO/SLOW/STOP) dari kamera dan menyalakan LED indikator |
| Input | Frame kamera CSI (IMX219) 1280×720 |
| Model | YOLOv8n-seg (custom, 3 kelas), input `1×3×640×640`, engine FP16 |
| Kelas | `0 = GO`, `1 = SLOW`, `2 = STOP` |
| Output | Bounding box + confidence → keputusan LED |
| LED | STOP→Merah, SLOW→Kuning, GO→Hijau; tanpa rambu → semua mati |
| Perangkat | NVIDIA Jetson Nano Developer Kit (4 GB) |
| Perkiraan kecepatan | ~11–12 FPS (inferensi GPU + kamera) |

Alur singkat:

```
Kamera CSI → Preprocess (letterbox) → YOLOv8-seg (TensorRT, GPU)
          → NMS + postprocess → Logika keputusan → GPIO → LED
```

---

## 2. Spesifikasi Hardware

- **Board**: NVIDIA Jetson Nano Developer Kit (4 GB, Tegra X1, Cortex-A57)
- **Storage sistem**: eMMC (`/dev/mmcblk0p1`) — JetPack 4.6
- **SD card**: berisi CUDA/TensorRT (lihat bagian Software)
- **Kamera**: CSI **IMX219** (8MP); mode dipakai 1080p@30 → di-scale ke 720p
- **Indikator**: 3 × LED (Merah, Kuning, Hijau)
- **Resistor**: 3 × 330 Ω (satu per LED, seri)
- **Lain-lain**: breadboard, kabel jumper secukupnya

> Untuk kamera USB (V4L2), set `"gst": None` pada konfigurasi `camera` di `sign_led.py`.

---

## 3. Spesifikasi Software

- **OS/Platform**: JetPack 4.6 (L4T R32.6.1)
- **Bahasa**: Python 3.6.9
- **CUDA**: 10.2 (runtime library dari SD card)
- **TensorRT**: 8.2.1 (`libnvinfer.so.8`)
- **OpenCV**: `opencv-python==4.5.5.64` (build QT5, mendukung `imshow`)
- **GStreamer**: 1.14.5 (`nvarguscamerasrc`, `nvvidconv`)
- **GPIO**: `Jetson.GPIO` 2.0.17 (mode `BOARD`)
- **NumPy**: 1.x

### Catatan penting lingkungan

1. **OpenBLAS/SIGILL** — kernel tegra 4.9 memicu `Illegal instruction` pada
   numpy/OpenBLAS (instruksi `mrs midr_el1`). Diperbaiki dengan menetapkan
   `OPENBLAS_CORETYPE=CORTEXA57` (sudah otomatis lewat `setenv.sh`).
2. **Stack CUDA/TensorRT dari SD card** — rootfs sistem (eMMC) tidak punya
   CUDA runtime/TensorRT lengkap. `setenv.sh` otomatis mencari SD card yang
   berisi `libnvinfer.so.8` lalu menambahkan `LD_LIBRARY_PATH` dan modul
   Python `tensorrt`. **Sistem bergantung pada SD card harus ter-mount.**

---

## 4. Struktur File

```
AIROBOT/
├── setenv.sh          # Menyiapkan environment (OpenBLAS + CUDA/TensorRT dari SD)
├── run.sh             # Launcher: setenv + jalankan sign_led.py
├── sign_led.py        # Aplikasi utama (kamera, GPIO, logika keputusan, tampilan)
├── yolo_trt.py        # Inferensi YOLOv8-seg + TensorRT + NMS
├── test_led.py        # Uji wiring LED (tanpa kamera/model)
├── best.onnx          # Model ONNX hasil training
├── best.pt            # Checkpoint YOLOv8-seg
├── best_fp16.engine   # Engine TensorRT FP16 (hasil build dari best.onnx)
├── pylib/             # Symlink modul `tensorrt` (dibuat otomatis oleh setenv.sh)
└── README.md
```

---

## 5. Cara Kerja Sistem

1. **Akuisisi gambar** — Kamera CSI dibaca via GStreamer
   (`nvarguscamerasrc → nvvidconv → videoconvert → fdsink`). Frame BGR
   1280×720 dibaca dari stdout subprocess.
   *(OpenCV wheel ini tidak punya dukungan GStreamer, dan V4L2 pada CSI
   menghasilkan frame hijau polos, sehingga dipakai jalur subprocess ini.)*
2. **Pra-proses** — Frame di-*letterbox* ke 640×640, dikonversi BGR→RGB,
   dinormalisasi (`/255`).
3. **Inferensi GPU** — Tensor dimasukkan ke engine TensorRT (`execute_async_v2`).
   Keluaran `output0` `(1, 39, 8400)` berisi 4 box + 3 kelas + 32 koefisien mask.
4. **Post-process** — Ambang confidence, konversi box ke koordinat asli,
   lalu **NMS per kelas** (numpy).
5. **Logika keputusan** (`DecisionEngine`):
   - Ada deteksi (`conf ≥ conf_thres`) → pilih rambu dengan confidence tertinggi.
   - Tidak ada / conf rendah → **tahan** keputusan terakhir selama
     `hold_seconds` (default 1 s) agar LED tidak berkedip.
   - Setelah itu → **kondisi aman** sesuai `safe_action`
     (default `"NONE"` = semua LED mati).
6. **Kendali LED** — `LedController` menyalakan satu LED sesuai
   `sign_to_led` (`STOP→red`, `SLOW→yellow`, `GO→green`).
7. **Tampilan** — Jendela OpenCV menampilkan bounding box, label, status LED,
   dan confidence. Tekan `q` untuk keluar.

### Konfigurasi utama (`sign_led.py` → `CONFIG`)

| Kunci | Default | Fungsi |
|-------|---------|--------|
| `conf_thres` | `0.40` | Ambang confidence minimum |
| `iou_thres` | `0.45` | Ambang NMS |
| `hold_seconds` | `1.0` | Tahan keputusan terakhir saat frame kosong |
| `led_pins` | `red:11, yellow:13, green:18` | Nomor pin fisik (BOARD) |
| `sign_to_led` | `STOP→red, SLOW→yellow, GO→green` | Pemetaan rambu→LED |
| `active_low` | `False` | `False`=active-high, `True`=active-low |
| `safe_action` | `"NONE"` | Aksi saat tanpa rambu: `NONE` (mati), `STOP` (merah), `BLINK` (merah kedip) |

---

## 6. Wiring

Wiring **active-high**: pin GPIO → resistor → kaki **panjang** LED (+),
kaki **pendek** LED (−) → GND.

```
Jetson pin (11 / 13 / 18) ──> [330 Ω] ──> kaki PANJANG LED (+)
                                            kaki PENDEK LED (−) ──> rel GND breadboard
rel GND breadboard ──> GND Jetson (mis. pin 14)
```

Pemetaan pin (mode `BOARD`, nomor **fisik** header):

| LED | Rambu | Pin fisik (BOARD) | Label casing (Raspberry-style) |
|-----|-------|-------------------|--------------------------------|
| Merah | STOP | **11** | D17 |
| Kuning | SLOW | **13** | D27 |
| Hijau | GO | **18** | D24 |

> **Penting**: kode memakai **nomor pin fisik** (BOARD), bukan label `Dxx` di
> casing. Pin 17 = 3.3 V dan pin 27 tidak bisa dipakai sebagai GPIO.

Setiap LED memerlukan resistor 330 Ω sendiri, dan **tidak boleh** tersambung
langsung ke rail 3.3 V/5 V (harus seri dengan pin GPIO).

### Uji wiring (tanpa kamera/model)

```bash
source setenv.sh
python3 test_led.py                 # wiring active-high
python3 test_led.py --active-low    # bila wiring active-low
```

Fase pertama harus membuat **semua LED padam**, lalu bergantian
merah → kuning → hijau.

---

## 7. Instalasi

1. **Pastikan JetPack 4.6 (L4T 32.6.x)** sudah terpasang, dan **SD card** yang
   berisi CUDA 10.2 + TensorRT 8.2 sudah ter-*mount* (biasanya di
   `/media/<user>/<uuid>/`). Cek:
   ```bash
   ls /media/*/*/usr/lib/aarch64-linux-gnu/libnvinfer.so.8
   ```

2. **Dependensi Python** (sekali saja):
   ```bash
   pip3 install --user "numpy<2"
   pip3 install --user opencv-python==4.5.5.64   # build GUI (QT5)
   pip3 install --user Jetson.GPIO
   ```

3. **(Opsional) Bangun ulang engine TensorRT** dari `best.onnx`:
   ```bash
   source setenv.sh
   "$SD_ROOT/usr/src/tensorrt/bin/trtexec" \
       --onnx=best.onnx --saveEngine=best_fp16.engine --fp16
   ```

4. **Siapkan environment** (dilakukan otomatis oleh `run.sh`, atau manual):
   ```bash
   cd AIROBOT
   source setenv.sh
   ```

> Bila kamera CSI error `Failed to create CaptureSession`, restart daemon Argus:
> ```bash
> sudo systemctl restart nvargus-daemon
> ```

---

## 8. Menjalankan Program

```bash
cd AIROBOT
./run.sh
```

Opsi yang tersedia:

```bash
./run.sh --conf 0.5        # ubah ambang confidence
./run.sh --no-gpio         # uji tanpa menggerakkan LED
./run.sh --no-display      # mode headless (SSH / tanpa layar)
./run.sh --camera 1        # pilih indeks kamera V4L2 (kamera USB)
```

Ekivalen secara manual:

```bash
source setenv.sh
python3 -u sign_led.py
```

Status akan tercetak di terminal, contoh:

```
Sign=SAFE  LED=OFF    Conf=0.00 FPS=11.6   # tidak ada rambu -> LED mati
Sign=STOP  LED=red    Conf=0.87 FPS=11.2   # rambu STOP     -> LED merah
```

Tekan **`q`** pada jendela tampilan untuk keluar.

---

## 9. Pemecahan Masalah (Troubleshooting)

| Gejala | Penyebab / Solusi |
|--------|-------------------|
| `Illegal instruction` saat impor numpy | `source setenv.sh` (menyetel `OPENBLAS_CORETYPE=CORTEXA57`) |
| `TensorRT` tidak ditemukan | SD card belum ter-mount; pasang lalu `source setenv.sh` |
| Kamera hitam/hijau / `CaptureSession` gagal | `sudo systemctl restart nvargus-daemon`, lalu ulangi |
| LED tidak menyala / menyala terbalik | Cek wiring & polaritas; ubah `active_low` di `CONFIG` |
| `Channel XX is invalid` | Pin bukan GPIO (mis. pin 17 = 3.3 V, pin 27 = I2C) |
| Jendela tidak muncul | Layar/DISPLAY tidak tersedia → jalankan `--no-display` |
