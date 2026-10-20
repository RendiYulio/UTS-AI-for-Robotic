"""
yolo_trt.py - Inferensi YOLOv8-seg dengan TensorRT di Jetson Nano.

Hanya keluaran DETEKSI (output0) yang dipakai; keluaran mask (output1) diabaikan
karena sistem ini cukup mengenali jenis sign, bukan membuat segmentasi piksel.

Butuh library CUDA/TensorRT dari SD card -> jalankan lewat `source setenv.sh`.
"""

import ctypes
import numpy as np
import cv2
import tensorrt as trt

CLASS_NAMES = ["GO", "SLOW", "STOP"]


# ---------------------------------------------------------------------------
# Binding CUDA runtime via ctypes (tanpa pycuda)
# ---------------------------------------------------------------------------
class CudaRuntime(object):
    H2D = 1   # cudaMemcpyHostToDevice
    D2H = 2   # cudaMemcpyDeviceToHost

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so")
        L = self.lib
        L.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        L.cudaMalloc.restype = ctypes.c_int
        L.cudaFree.argtypes = [ctypes.c_void_p]
        L.cudaFree.restype = ctypes.c_int
        L.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_size_t, ctypes.c_int]
        L.cudaMemcpy.restype = ctypes.c_int
        L.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        L.cudaStreamCreate.restype = ctypes.c_int
        L.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        L.cudaStreamDestroy.restype = ctypes.c_int
        L.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        L.cudaStreamSynchronize.restype = ctypes.c_int

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        rc = self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        if rc != 0:
            raise RuntimeError("cudaMalloc gagal rc=%d (butuh %d byte)" % (rc, nbytes))
        return ptr

    def free(self, ptr):
        if ptr:
            self.lib.cudaFree(ptr)

    def memcpy_h2d(self, dptr, host_arr):
        rc = self.lib.cudaMemcpy(dptr,
                                 host_arr.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_size_t(host_arr.nbytes), self.H2D)
        if rc != 0:
            raise RuntimeError("cudaMemcpy H2D gagal rc=%d" % rc)

    def memcpy_d2h(self, host_arr, dptr):
        rc = self.lib.cudaMemcpy(host_arr.ctypes.data_as(ctypes.c_void_p),
                                 dptr,
                                 ctypes.c_size_t(host_arr.nbytes), self.D2H)
        if rc != 0:
            raise RuntimeError("cudaMemcpy D2H gagal rc=%d" % rc)

    def stream(self):
        s = ctypes.c_void_p()
        self.lib.cudaStreamCreate(ctypes.byref(s))
        return s


# ---------------------------------------------------------------------------
# Utilitas citra & NMS
# ---------------------------------------------------------------------------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize + pad citra, kembalikan citra, rasio, dan offset (kiri, atas)."""
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw = (new_shape[1] - new_w) / 2.0
    dh = (new_shape[0] - new_h) / 2.0
    if (w, h) != (new_w, new_h):
        im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def nms(boxes, scores, iou_thres):
    """NMS sederhana (numpy). boxes: (N,4) xyxy."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[np.where(iou <= iou_thres)[0] + 1]
    return keep


# ---------------------------------------------------------------------------
# Inferensi
# ---------------------------------------------------------------------------
class YoloTRT(object):
    def __init__(self, engine_path, class_names=None, imgsz=(640, 640),
                 conf_thres=0.35, iou_thres=0.45):
        self.class_names = class_names or CLASS_NAMES
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self._cuda = CudaRuntime()

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Gagal deserialize engine: %s" % engine_path)

        self.context = self.engine.create_execution_context()
        self.stream = self._cuda.stream()

        self.inputs = []    # (name, host_arr, device_ptr, nbytes)
        self.outputs = []   # (name, host_arr, device_ptr, nbytes, shape)
        self.bindings = [None] * self.engine.num_bindings

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = np.float32 if self.engine.get_binding_dtype(i) == trt.float32 \
                else np.float16
            host = np.empty(shape, dtype=dtype)
            dev = self._cuda.malloc(host.nbytes)
            self.bindings[i] = int(dev.value)
            if self.engine.binding_is_input(i):
                self.inputs.append((name, host, dev, host.nbytes))
                self.input_shape = shape
            else:
                self.outputs.append((name, host, dev, host.nbytes, shape))

        self.output0 = next(o for o in self.outputs if o[0] == "output0")

    # -- pra-proses: BGR frame -> tensor NCHW RGB ternormalisasi --------------
    def preprocess(self, frame_bgr):
        img, ratio, pad = letterbox(frame_bgr, self.imgsz)
        img = img[:, :, ::-1]                      # BGR -> RGB
        img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
        img /= 255.0
        return np.ascontiguousarray(img[None]), ratio, pad

    def infer(self, frame_bgr):
        """Kembalikan (detections, ratio, pad).
        detections: list of (x1, y1, x2, y2, conf, class_id) pada koordinat asli.
        """
        tensor, ratio, pad = self.preprocess(frame_bgr)
        self._cuda.memcpy_h2d(self.inputs[0][2], tensor)

        ok = self.context.execute_async_v2(bindings=self.bindings,
                                           stream_handle=self.stream.value)
        self._cuda.lib.cudaStreamSynchronize(self.stream)
        if not ok:
            raise RuntimeError("execute_async_v2 gagal")

        for (_, host, dev, _, _) in self.outputs:
            self._cuda.memcpy_d2h(host, dev)

        preds = self.output0[1]                    # (1, 39, 8400)
        dets = self._postprocess(preds, ratio, pad, frame_bgr.shape)
        return dets, ratio, pad

    def _postprocess(self, preds, ratio, pad, orig_shape):
        p = preds[0].transpose(1, 0)               # (8400, 39)
        nc = len(self.class_names)
        boxes_xywh = p[:, 0:4]
        scores = p[:, 4:4 + nc]
        cls_ids = scores.argmax(axis=1)
        confs = scores[np.arange(len(scores)), cls_ids]
        mask = confs >= self.conf_thres
        boxes_xywh, cls_ids, confs = boxes_xywh[mask], cls_ids[mask], confs[mask]
        if len(boxes_xywh) == 0:
            return []

        # cx,cy,w,h (ruang 640) -> xyxy, lalu balikkan letterbox
        cx, cy, w, h = (boxes_xywh[:, 0], boxes_xywh[:, 1],
                        boxes_xywh[:, 2], boxes_xywh[:, 3])
        x1 = (cx - w / 2 - pad[0]) / ratio
        y1 = (cy - h / 2 - pad[1]) / ratio
        x2 = (cx + w / 2 - pad[0]) / ratio
        y2 = (cy + h / 2 - pad[1]) / ratio
        oh, ow = orig_shape[:2]
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)

        # NMS per kelas
        results = []
        for c in np.unique(cls_ids):
            idx = np.where(cls_ids == c)[0]
            keep = nms(boxes[idx], confs[idx], self.iou_thres)
            for k in keep:
                j = idx[k]
                results.append((float(boxes[j, 0]), float(boxes[j, 1]),
                                float(boxes[j, 2]), float(boxes[j, 3]),
                                float(confs[j]), int(cls_ids[j])))
        results.sort(key=lambda d: d[4], reverse=True)
        return results
