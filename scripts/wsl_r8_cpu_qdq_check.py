#!/usr/bin/env python3
"""R8 诊断：onnxruntime CPU 跑 modelslim 中间 QDQ 模型，判断框退化发生在
量化阶段（quantize_static）还是 ATC/NPU 阶段。"""
import sys
sys.path.insert(0, "/mnt/g/Git_repo/edge-perception-bench/scripts")
import glob
import cv2
import numpy as np
import onnxruntime as ort
from infer_dms_pipeline import LetterboxPreprocess

FP32 = "/root/quant/yolov8n_fp32.onnx"
QDQ = "/root/quant/r8/yolov8n_s3_quant_v2_cpu_quant.onnx"

imgs = [i for i in glob.glob("/mnt/g/Git_repo/Image_deal_class/**/*.jpg", recursive=True)
        if __import__("os").path.getsize(i) > 3000][:1]
img = cv2.imread(imgs[0])
print("img:", imgs[0], img.shape)

pre = LetterboxPreprocess(imgsz=640)
t, _, _ = pre(img)
t = t.astype(np.float32)

opt = ort.SessionOptions()
opt.log_verbosity_level = 0
s_fp = ort.InferenceSession(FP32, opt, providers=["CPUExecutionProvider"])
s_q = ort.InferenceSession(QDQ, opt, providers=["CPUExecutionProvider"])
iname = s_fp.get_inputs()[0].name
o_fp = s_fp.run(None, {iname: t})[0].squeeze().transpose()
o_q = s_q.run(None, {iname: t})[0].squeeze().transpose()

cos = float(np.dot(o_fp.ravel(), o_q.ravel()) /
            (np.linalg.norm(o_fp.ravel()) * np.linalg.norm(o_q.ravel()) + 1e-12))
print(f"cosine(fp32, qdq) = {cos:.5f}")

print("--- FP32 top8 ---")
for idx in (-o_fp[:, 4:].max(1)).argsort()[:8]:
    cls = int(o_fp[idx, 4:].argmax())
    print(f"  idx={idx:5d} cls={cls} conf={o_fp[idx, 4 + cls]:.4f} xywh={[round(v, 2) for v in o_fp[idx, :4]]}")
print("--- QDQ(CPU) top8 ---")
for idx in (-o_q[:, 4:].max(1)).argsort()[:8]:
    cls = int(o_q[idx, 4:].argmax())
    print(f"  idx={idx:5d} cls={cls} conf={o_q[idx, 4 + cls]:.4f} xywh={[round(v, 2) for v in o_q[idx, :4]]}")
