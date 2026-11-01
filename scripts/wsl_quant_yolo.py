"""R5 · WSL 侧 FP32→INT8 量化（onnxruntime QDQ 静态量化，供昇腾 ATC 编译）。

步骤：
 1) ultralytics 用 opset=13 重导 yolov8n.onnx（QDQ per-channel 需 opset>=13）；
 2) 用若干真实图片做标定，quantize_static 输出 QDQ INT8 ONNX；
 3) onnxruntime 各跑一张，打印 FP32 vs INT8 输出的数值差异（量化自检）。
产物：/root/quant/yolov8n_fp32.onnx, /root/quant/yolov8n_int8.onnx
"""
import glob
import os
import numpy as np
import cv2
from pathlib import Path

WORK = Path("/root/quant")
WORK.mkdir(parents=True, exist_ok=True)
PT = "/mnt/g/Git_repo/yolov8n.pt"

# ---------- 1) 重导 opset13 ONNX ----------
from ultralytics import YOLO
print("[1] export onnx opset=13 ...")
m = YOLO(PT)
p = m.export(format="onnx", opset=13, imgsz=640, simplify=False, half=False,
             dynamic=False, batch=1)
fp32 = WORK / "yolov8n_fp32.onnx"
fp32.write_bytes(Path(p).read_bytes())
print(f"    fp32 onnx -> {fp32}  ({fp32.stat().st_size/1024/1024:.1f} MB)")

# ---------- 2) 标定 + 静态量化 ----------
IMG_GLOBS = [
    "/mnt/g/Git_repo/Image_deal_class/*.jpg",
    "/mnt/g/Git_repo/edge-perception-bench/results/image/*.jpg",
    "/mnt/g/Git_repo/Image_deal_class/实验*/**/*.jpg",
    "/mnt/g/Git_repo/edge-perception-bench/results/doc/*.jpg",
]
imgs = []
for g in IMG_GLOBS:
    imgs += glob.glob(g, recursive=True)
imgs = [i for i in imgs if os.path.getsize(i) > 3000][:40]
print(f"[2] calibration images: {len(imgs)}")


def letterbox(img, sz=640):
    h0, w0 = img.shape[:2]
    r = min(sz / h0, sz / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    rs = cv2.resize(img, (nw, nh))
    c = np.full((sz, sz, 3), 114, np.uint8)
    l, t = (sz - nw) // 2, (sz - nh) // 2
    c[t:t + nh, l:l + nw] = rs
    x = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))[None]


feeds = [letterbox(cv2.imread(i)) for i in imgs if cv2.imread(i) is not None]
print(f"    valid feeds: {len(feeds)}")

import onnx
from onnxruntime.quantization import QuantType, QuantFormat, CalibrationMethod
from onnxruntime.quantization import quantize_static

input_name = onnx.load(str(fp32)).graph.input[0].name
print("    input node:", input_name)


class Reader:
    def __init__(self):
        self.i = 0

    def get_next(self):
        if self.i >= len(feeds):
            return None
        f = feeds[self.i]
        self.i += 1
        return {input_name: f}

    def rewind(self):
        self.i = 0


int8 = WORK / "yolov8n_int8.onnx"
quantize_static(
    str(fp32), str(int8), Reader(),
    quant_format=QuantFormat.QDQ,
    per_channel=True,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.MinMax,
    op_types_to_quantize=["Conv", "MatMul"],
)
print(f"    int8 onnx -> {int8}  ({int8.stat().st_size/1024/1024:.1f} MB)")

# ---------- 3) 量化自检：FP32 vs INT8 输出差异 ----------
import onnxruntime as ort
x = feeds[0]
so_f = ort.SessionOptions(); so_f.log_verbosity_level = 0
sf = ort.InferenceSession(str(fp32), so_f, providers=["CPUExecutionProvider"])
si = ort.InferenceSession(str(int8), so_f, providers=["CPUExecutionProvider"])
of = sf.run(None, {input_name: x})[0]
oi = si.run(None, {input_name: x})[0]
diff = np.abs(of - oi)
cos = float(np.dot(of.ravel(), oi.ravel()) /
            (np.linalg.norm(of.ravel()) * np.linalg.norm(oi.ravel()) + 1e-12))
print("[3] raw-output check (frame0):")
print(f"    shape={of.shape}  fp32[min,max]=({of.min():.2f},{of.max():.2f})")
print(f"    mean_abs_diff={diff.mean():.4f}  max_abs_diff={diff.max():.4f}  cosine={cos:.5f}")
Path("/root/quant/quant_selfcheck.json").write_text(
    '{"mean_abs_diff":%.6f,"max_abs_diff":%.6f,"cosine":%.6f,"out_shape":list(%s)}'
    % (diff.mean(), diff.max(), cos, of.shape))
print("DONE")
