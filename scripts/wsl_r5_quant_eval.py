"""R5 · INT8 量化评估（WSL 侧，onnxruntime CPU 口径）。

产出两组数据：
  A. 精度损失代理：同一批真实图片，FP32 vs INT8 —— 原始输出 cosine / 检出数差 /
     person 平均置信度差 / 匹配框 IoU。
  B. 延时收益：单线程 onnxruntime CPU 纯推理（session.run）FP32 vs INT8。

背景：本项目 R2/R3 已实测「Host CPU 侧 pre+post 才是 E2E 延时大头」，
故 CPU 口径的量化延时同样有参考价值；NPU 侧 INT8 受限于工具链（见周志）。
"""
import glob
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, "/mnt/g/Git_repo/edge-perception-bench/scripts")
from infer_dms_pipeline import LetterboxPreprocess, YoloV8Postprocess  # noqa

WORK = Path("/root/quant")
FP32 = str(WORK / "yolov8n_fp32.onnx")
INT8 = str(WORK / "yolov8n_int8.onnx")

# 收集评测图片
imgs = []
for g in ["/mnt/g/Git_repo/Image_deal_class/**/*.jpg",
          "/mnt/g/Git_repo/Image_deal_class/**/*.png",
          "/mnt/g/Git_repo/edge-perception-bench/results/image/*.jpg",
          "/mnt/g/Git_repo/edge-perception-bench/results/doc/*.jpg"]:
    imgs += glob.glob(g, recursive=True)
imgs = [i for i in imgs if os.path.isfile(i) and os.path.getsize(i) > 3000]
imgs = list(dict.fromkeys(imgs))
print(f"eval images: {len(imgs)}")

import onnxruntime as so  # noqa
opt = so.SessionOptions()
opt.log_verbosity_level = 0
opt.intra_op_num_threads = 1        # 单线程，公平且可复现
s_fp = so.InferenceSession(FP32, opt, providers=["CPUExecutionProvider"])
s_qt = so.InferenceSession(INT8, opt, providers=["CPUExecutionProvider"])
input_name = s_fp.get_inputs()[0].name
pre = LetterboxPreprocess()
post = YoloV8Postprocess(conf=0.25)


def run(sess, t):
    return sess.run(None, {input_name: t.astype(np.float32)})[0]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xx1, yy1 = max(ax1, bx1), max(ay1, by1)
    xx2, yy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


cosines, det_fp, det_qt, conf_fp, conf_qt, match_iou = [], [], [], [], [], []
for p in imgs:
    im = cv2.imread(p)
    if im is None:
        continue
    h0, w0 = im.shape[:2]
    t, r, pad = pre(im)
    of, oq = run(s_fp, t), run(s_qt, t)
    cosines.append(float(np.dot(of.ravel(), oq.ravel()) /
                   (np.linalg.norm(of.ravel()) * np.linalg.norm(oq.ravel()) + 1e-12)))
    df = post(of, r, pad, (w0, h0))
    dq = post(oq, r, pad, (w0, h0))
    det_fp.append(len(df)); det_qt.append(len(dq))
    pf = [d["confidence"] for d in df if d["label"] == "person"]
    pq = [d["confidence"] for d in dq if d["label"] == "person"]
    conf_fp += pf; conf_qt += pq
    # 同类别贪心匹配 IoU
    for a in df:
        cands = [b for b in dq if b["label"] == a["label"]]
        if cands:
            best = max(cands, key=lambda b: iou(a["bbox"], b["bbox"]))
            match_iou.append(iou(a["bbox"], best["bbox"]))

print(f"[A] output cosine   mean={statistics.mean(cosines):.5f} min={min(cosines):.5f}")
print(f"[A] det count       FP32={statistics.mean(det_fp):.2f} INT8={statistics.mean(det_qt):.2f} per img")
print(f"[A] person conf     FP32={statistics.mean(conf_fp):.4f} INT8={statistics.mean(conf_qt):.4f}")
print(f"[A] matched box IoU mean={statistics.mean(match_iou):.4f} min={min(match_iou):.3f}")

# B · 延时
def bench(sess, t, iters=30, warmup=5):
    for _ in range(warmup):
        sess.run(None, {input_name: t})
    xs = []
    for _ in range(iters):
        t0 = time.perf_counter(); sess.run(None, {input_name: t}); t1 = time.perf_counter()
        xs.append((t1 - t0) * 1000)
    return {"mean": round(statistics.mean(xs), 3), "median": round(statistics.median(xs), 3),
            "p95": round(sorted(xs)[int(len(xs) * 0.95) - 1], 3)}


tq, _, _ = pre(cv2.imread(imgs[0]))
tq = tq.astype(np.float32)
lat_fp = bench(s_fp, tq)
lat_qt = bench(s_qt, tq)
print(f"[B] CPU infer FP32 {lat_fp}  INT8 {lat_qt}")

import json
out = {
    "eval_images": len(cosines),
    "model_size_mb": {"fp32": round(os.path.getsize(FP32) / 1048576, 2),
                      "int8": round(os.path.getsize(INT8) / 1048576, 2)},
    "accuracy": {
        "output_cosine_mean": round(statistics.mean(cosines), 5),
        "output_cosine_min": round(min(cosines), 5),
        "det_count_mean": {"fp32": round(statistics.mean(det_fp), 2),
                           "int8": round(statistics.mean(det_qt), 2)},
        "person_conf_mean": {"fp32": round(statistics.mean(conf_fp), 4),
                             "int8": round(statistics.mean(conf_qt), 4)},
        "matched_box_iou_mean": round(statistics.mean(match_iou), 4),
        "matched_box_iou_min": round(min(match_iou), 3),
    },
    "cpu_infer_ms_1thread": {"fp32": lat_fp, "int8": lat_qt,
                             "speedup": round(lat_fp["mean"] / lat_qt["mean"], 2)},
}
Path("/root/quant/r5_int8_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("WROTE /root/quant/r5_int8_eval.json")
