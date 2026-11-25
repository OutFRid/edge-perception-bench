#!/usr/bin/env python3
"""R6 · PTQ 量化策略组合对比（WSL onnxruntime CPU 口径）。

背景（承接 R5）：
  R5 已产出单一策略 INT8 模型（MinMax / per-channel / QDQ / Conv+MatMul）：
  精度近无损但 CPU 延时反慢 1.3×；真·NPU INT8 OM 受 AMCT 缺失阻塞。
  R6 任务：横向对比多组 PTQ 策略组合，沉淀「精度-延时」权衡方法，
  为后续 NPU 上板（R6 顺延/R8 缓冲）选定策略。

策略矩阵（7 组，S1 复用 R5 产物保证可比）：
  S1  MinMax   / per-channel / Conv+MatMul      （R5 基线，复用已有模型）
  S2  Entropy  / per-channel / Conv+MatMul
  S3  Percentile(99.99) / per-channel / Conv+MatMul
  S4  MinMax   / per-tensor  / Conv+MatMul
  S5  MinMax   / per-channel / 仅 Conv
  S6  MinMax   / per-channel / 全网络（ORT 默认 op 列表）
  S7  动态量化（仅权重，激活保持 FP32，ORT 默认 MatMul/Gemm）

统一口径：同一批标定集（与 R5 同源）、同一批 16 张评估集、
  单线程 CPU、30 iters 延时。评估维度：体积 / 输出 cosine / 检出数 /
  person 置信度 / 匹配框 IoU / CPU 推理延时。

产物：/root/quant/r6/*.onnx + /root/quant/r6_ptq_strategies.json
      （脚本末自动拷贝 JSON 到仓库 results/doc/）
"""
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                      QuantType, quantize_dynamic,
                                      quantize_static)

sys.path.insert(0, "/mnt/g/Git_repo/edge-perception-bench/scripts")
from infer_dms_pipeline import LetterboxPreprocess, YoloV8Postprocess  # noqa

WORK = Path("/root/quant/r6")
WORK.mkdir(parents=True, exist_ok=True)
FP32 = "/root/quant/yolov8n_fp32.onnx"
R5_INT8 = "/root/quant/yolov8n_int8.onnx"   # S1 复用
OUT_JSON = Path("/root/quant/r6_ptq_strategies.json")
REPO_JSON = Path("/mnt/g/Git_repo/edge-perception-bench/results/doc/r6_ptq_strategies.json")

# ---------- 图片收集（与 R5 同源同过滤，保证可比） ----------
IMGS = []
for g in ["/mnt/g/Git_repo/Image_deal_class/**/*.jpg",
          "/mnt/g/Git_repo/Image_deal_class/**/*.png",
          "/mnt/g/Git_repo/edge-perception-bench/results/image/*.jpg",
          "/mnt/g/Git_repo/edge-perception-bench/results/doc/*.jpg"]:
    IMGS += glob.glob(g, recursive=True)
IMGS = [i for i in IMGS if os.path.isfile(i) and os.path.getsize(i) > 3000]
IMGS = list(dict.fromkeys(sorted(IMGS)))
print(f"image pool: {len(IMGS)}")


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


CAL_N = 12
cal_imgs = [cv2.imread(p) for p in IMGS[:CAL_N]]
cal_feeds = [letterbox(im) for im in cal_imgs if im is not None]
print(f"calibration feeds: {len(cal_feeds)}")

# 评估集：图片池全部有效图（当前池 16 张，与 R5 评估口径一致；与标定集同池，见 JSON cal_eval_overlap）
EVAL_N = 16
eval_paths = [p for p in IMGS[:EVAL_N] if cv2.imread(p) is not None]
print(f"eval images: {len(eval_paths)}")

input_name = onnx.load(FP32).graph.input[0].name


class CalReader:
    def __init__(self, feeds):
        self.feeds, self.i = feeds, 0

    def get_next(self):
        if self.i >= len(self.feeds):
            return None
        f = self.feeds[self.i]
        self.i += 1
        return {input_name: f}

    def rewind(self):
        self.i = 0


# ---------- 策略定义 ----------
STRATEGIES = [
    {
        "id": "S1_minmax_pc_convmatmul",
        "desc": "MinMax / per-channel / Conv+MatMul (R5 基线)",
        "reuse": R5_INT8,
    },
    {
        "id": "S2_entropy_pc_convmatmul",
        "desc": "Entropy / per-channel / Conv+MatMul",
        "kwargs": dict(quant_format=QuantFormat.QDQ, per_channel=True,
                       calibrate_method=CalibrationMethod.Entropy,
                       op_types_to_quantize=["Conv", "MatMul"]),
    },
    {
        "id": "S3_pct9999_pc_convmatmul",
        "desc": "Percentile(99.99) / per-channel / Conv+MatMul",
        "kwargs": dict(quant_format=QuantFormat.QDQ, per_channel=True,
                       calibrate_method=CalibrationMethod.Percentile,
                       op_types_to_quantize=["Conv", "MatMul"]),
    },
    {
        "id": "S4_minmax_pt_convmatmul",
        "desc": "MinMax / per-tensor / Conv+MatMul",
        "kwargs": dict(quant_format=QuantFormat.QDQ, per_channel=False,
                       calibrate_method=CalibrationMethod.MinMax,
                       op_types_to_quantize=["Conv", "MatMul"]),
    },
    {
        "id": "S5_minmax_pc_convonly",
        "desc": "MinMax / per-channel / 仅 Conv",
        "kwargs": dict(quant_format=QuantFormat.QDQ, per_channel=True,
                       calibrate_method=CalibrationMethod.MinMax,
                       op_types_to_quantize=["Conv"]),
    },
    {
        "id": "S6_minmax_pc_allops",
        "desc": "MinMax / per-channel / 全网络（ORT 默认 op 列表）",
        "kwargs": dict(quant_format=QuantFormat.QDQ, per_channel=True,
                       calibrate_method=CalibrationMethod.MinMax),
    },
    {
        "id": "S7_dynamic_weightonly",
        "desc": "动态量化（仅权重 int8，激活保持 FP32，限 MatMul）",
        "dynamic": True,
    },
]


def make_strategy(st):
    out = str(WORK / (st["id"] + ".onnx"))
    if "reuse" in st:
        print(f"[{st['id']}] reuse {st['reuse']}")
        return st["reuse"]
    if st.get("dynamic"):
        # 限定 MatMul：动态量化默认会生成 ConvInteger，ort CPU 无对应 kernel（NotImplemented）
        quantize_dynamic(FP32, out, weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul"])
        return out
    common = dict(weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
    quantize_static(FP32, out, CalReader(cal_feeds), **common, **st["kwargs"])
    return out


def count_qdq(path):
    m = onnx.load(path)
    q = sum(1 for n in m.graph.node if n.op_type == "QuantizeLinear")
    return q, len(m.graph.node)


# ---------- 评估（同 R5 口径） ----------
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xx1, yy1 = max(ax1, bx1), max(ay1, by1)
    xx2, yy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def build_session(path):
    opt = ort.SessionOptions()
    opt.log_verbosity_level = 0
    opt.intra_op_num_threads = 1
    return ort.InferenceSession(path, opt, providers=["CPUExecutionProvider"])


def bench(sess, iname, t, iters=30, warmup=5):
    for _ in range(warmup):
        sess.run(None, {iname: t})
    xs = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {iname: t})
        xs.append((time.perf_counter() - t0) * 1000)
    return {"mean": round(statistics.mean(xs), 3),
            "median": round(statistics.median(xs), 3),
            "p95": round(sorted(xs)[int(len(xs) * 0.95) - 1], 3)}


def _flush(result):
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    REPO_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    pre = LetterboxPreprocess()
    post = YoloV8Postprocess(conf=0.25)
    result = {"generated": time.strftime("%Y-%m-%d %H:%M"),
              "cal_images": len(cal_feeds), "eval_images": EVAL_N,
              "cal_eval_overlap": bool(set(IMGS[:CAL_N]) & set(eval_paths)),
              "strategies": {}}

    # 参考模型 FP32：cosine 基准
    fp_sess = build_session(FP32)
    fp_iname = fp_sess.get_inputs()[0].name

    # 先跑各策略：量化（或复用）+ 评估
    dets_by_strategy = {}
    for st in STRATEGIES:
        sid = st["id"]
        print(f"\n===== {sid}: {st['desc']} =====")
        try:
            path = make_strategy(st)
            qdq_n, node_n = count_qdq(path)
            print(f"    onnx={os.path.getsize(path)/1048576:.2f}MB qdq_nodes={qdq_n}/{node_n}")
        except Exception as e:  # 该策略失败不阻塞整体
            result["strategies"][sid] = {"desc": st["desc"], "error": str(e)}
            print(f"    FAILED: {e}")
            continue

        sess = build_session(path)
        iname = sess.get_inputs()[0].name
        cosines, det, conf, boxes = [], [], [], []
        try:
            for p in eval_paths:
                im = cv2.imread(p)
                if im is None:
                    continue
                h0, w0 = im.shape[:2]
                t, r, pad = pre(im)
                t = t.astype(np.float32)
                o = sess.run(None, {iname: t})[0]
                ofp = fp_sess.run(None, {fp_iname: t})[0]
                cosines.append(float(np.dot(o.ravel(), ofp.ravel()) /
                                     (np.linalg.norm(o.ravel()) *
                                      np.linalg.norm(ofp.ravel()) + 1e-12)))
                d = post(o, r, pad, (w0, h0))
                det.append(len(d))
                conf += [x["confidence"] for x in d if x["label"] == "person"]
                boxes.append(d)
        except Exception as e:  # 单策略评估失败不丢其他策略结果
            result["strategies"][sid] = {
                "desc": st["desc"],
                "model_mb": round(os.path.getsize(path) / 1048576, 2),
                "qdq_nodes": qdq_n, "error": str(e)}
            _flush(result)
            print(f"    EVAL_FAILED: {e}")
            continue
        dets_by_strategy[sid] = boxes

        # 延时（固定输入，单线程）
        tq = pre(cv2.imread(eval_paths[0]))[0].astype(np.float32)
        lat = bench(sess, iname, tq)
        result["strategies"][sid] = {
            "desc": st["desc"],
            "model_mb": round(os.path.getsize(path) / 1048576, 2),
            "qdq_nodes": qdq_n,
            "accuracy": {
                "output_cosine_mean": round(statistics.mean(cosines), 5),
                "output_cosine_min": round(min(cosines), 5),
                "det_count_mean": round(statistics.mean(det), 2),
                "person_conf_mean": round(statistics.mean(conf), 4)
                if conf else None,
            },
            "cpu_ms_1thread": lat,
        }
        _flush(result)
        print(f"    cosine={result['strategies'][sid]['accuracy']['output_cosine_mean']}"
              f"  det={result['strategies'][sid]['accuracy']['det_count_mean']}"
              f"  lat={lat['mean']}ms")

    # 匹配框 IoU：各策略 vs S1 基线（R5 口径：FP32 基线）
    s1_key = STRATEGIES[0]["id"]
    for sid, boxes in dets_by_strategy.items():
        if sid == s1_key:
            continue
        mious = []
        for df, dq in zip(dets_by_strategy[s1_key], boxes):
            for a in df:
                cands = [b for b in dq if b["label"] == a["label"]]
                if cands:
                    best = max(cands, key=lambda b: iou(a["bbox"], b["bbox"]))
                    mious.append(iou(a["bbox"], best["bbox"]))
        if mious:
            result["strategies"][sid]["accuracy"]["vs_S1_box_iou_mean"] = \
                round(statistics.mean(mious), 4)
            result["strategies"][sid]["accuracy"]["vs_S1_box_iou_min"] = \
                round(min(mious), 3)

    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    REPO_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nWROTE {OUT_JSON} -> {REPO_JSON}")


if __name__ == "__main__":
    main()