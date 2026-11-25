#!/usr/bin/env python3
"""R8 中间特征对照·板端：FP32 mid vs INT8 v2 mid 同帧逐中间点 cosine/误差统计。

定位 INT8 精度崩坏的第一层（找到 cos 显著下降的 stage）。
"""
import glob
import sys

import cv2
import numpy as np

from infer_dms_pipeline import AclContext, AclExecutor, LetterboxPreprocess

INT8_OM = sys.argv[1] if len(sys.argv) > 1 else "/root/epb/models/yolov8n_s3_int8_v2_mid.om"

STAGE_NAMES = [
    "s0_conv", "s2_cv2", "s4_cv2", "s6_cv2",
    "s7_conv", "s8_cv1", "s8_m0_cv1", "s8_m0_cv2", "s8_cv2",
    "s9_cv2", "s12_cv2", "s15_cv2", "s18_cv2", "s21_cv2", "final",
]

clips = sorted(glob.glob("/mnt/e61/workspace/dmd/gA/*/*/*_mosaic.avi"))
clip = clips[0]
cap = cv2.VideoCapture(clip)
ok, frame = cap.read()
cap.release()
assert ok, f"cannot read {clip}"
print(f"clip: {clip}")

pre = LetterboxPreprocess(imgsz=640)
tensor, r, pad = pre(frame)

ctx = AclContext()
m_fp = AclExecutor(ctx, "/root/epb/models/yolov8n_fp32_mid.om")
m_q = AclExecutor(ctx, INT8_OM)
print(f"fp32 outputs={m_fp.n_out}  int8 outputs={m_q.n_out}  ({INT8_OM})")

of, _ = m_fp.execute([tensor])
oq, _ = m_q.execute([tensor])

print(f"{'stage':<10} {'shape':<22} {'cosine':<10} {'max_abs':<12} {'mean_abs':<12} {'rel_err'}")
for i, name in enumerate(STAGE_NAMES):
    a = of[i].ravel().astype(np.float64)
    b = oq[i].ravel().astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    d = np.abs(a - b)
    rel = float(np.mean(d) / (np.mean(np.abs(a)) + 1e-12))
    print(f"{name:<10} {str(tuple(of[i].shape)):<22} {cos:<10.6f} {d.max():<12.4g} {d.mean():<12.4g} {rel:.4f}")
