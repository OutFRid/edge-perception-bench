#!/usr/bin/env python3
"""R8 下钻：单帧打印 FP32 / INT8 OM 的全部检测框，排查 IoU=0 异常。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
from infer_dms_pipeline import AclContext, AclExecutor, LetterboxPreprocess, YoloV8Postprocess  # noqa

CLIP = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/e61/workspace/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_rgb_mosaic.avi"
FP32 = "/mnt/e61/workspace/dmd/model/yolov8n.om"
INT8 = "/root/epb/models/yolov8n_s3_int8.om"

cap = cv2.VideoCapture(CLIP)
ok, frame = cap.read()
cap.release()
h0, w0 = frame.shape[:2]
pre = LetterboxPreprocess(imgsz=640)
post = YoloV8Postprocess(conf=0.25, iou=0.45)
tensor, r, pad = pre(frame)

ctx = AclContext()
m_fp = AclExecutor(ctx, FP32)
m_qt = AclExecutor(ctx, INT8)
of, _ = m_fp.execute([tensor])
oq, _ = m_qt.execute([tensor])
m_fp.destroy(); m_qt.destroy(); ctx.destroy()

df = post(of[0], r, pad, (w0, h0))
dq = post(oq[0], r, pad, (w0, h0))

print(f"frame {w0}x{h0}  ratio={r:.4f} pad={pad}")
print("--- FP32 detections ---")
for d in df:
    print(f"  {d['label']:<14} conf={d['confidence']:.3f} box={d['bbox']}")
print("--- INT8 detections ---")
for d in dq:
    print(f"  {d['label']:<14} conf={d['confidence']:.3f} box={d['bbox']}")

# 高置信 anchor 差异（NMS 前 top-k 候选）也能看出漂移来源
o = of[0].squeeze().transpose()
q = oq[0].squeeze().transpose()
print("--- FP32 top10 anchors (conf desc) ---")
for idx in (-o[:, 4:].max(1)).argsort()[:10]:
    cls = int(o[idx, 4:].argmax())
    print(f"  idx={idx:5d} cls={cls} conf={o[idx, 4 + cls]:.4f} xywh={[round(v,1) for v in o[idx,:4]]}")
print("--- INT8 top10 anchors (conf desc) ---")
for idx in (-q[:, 4:].max(1)).argsort()[:10]:
    cls = int(q[idx, 4:].argmax())
    print(f"  idx={idx:5d} cls={cls} conf={q[idx, 4 + cls]:.4f} xywh={[round(v,1) for v in q[idx,:4]]}")
