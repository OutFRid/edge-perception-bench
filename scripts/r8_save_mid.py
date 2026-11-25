#!/usr/bin/env python3
"""R8 保存真实中间特征：跑 fp32 mid OM，保存 s6_cv2 输出与预处理输入，
供「s7 单层复现实验」使用（输入真实、CPU 精确手算对照板端）。
"""
import glob

import cv2
import numpy as np

from infer_dms_pipeline import AclContext, AclExecutor, LetterboxPreprocess

clips = sorted(glob.glob("/mnt/e61/workspace/dmd/gA/*/*/*_mosaic.avi"))
cap = cv2.VideoCapture(clips[0])
ok, frame = cap.read()
cap.release()
assert ok

pre = LetterboxPreprocess(imgsz=640)
tensor, r, pad = pre(frame)

ctx = AclContext()
m = AclExecutor(ctx, "/root/epb/models/yolov8n_fp32_mid.om")
of, _ = m.execute([tensor])

np.savez("/root/epb/results/r8_mid_feats.npz",
         input=tensor.astype(np.float32),
         s6=of[3].astype(np.float32))
print("saved input", tensor.shape, "s6", of[3].shape)
