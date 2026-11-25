#!/usr/bin/env python3
"""R8 · NPU INT8 上板精度对照（FP32 OM vs INT8 OM，同帧同口径）。

口径对齐 R5（wsl_r5_quant_eval.py，CPU 侧）：
  - 原始输出张量 cosine（84x8400 全张量保真度）
  - 检出框数 / 帧（conf=0.25，NMS 0.45）
  - person 平均置信度
  - 匹配框 IoU（同类别贪心匹配，均值/最小）

帧源：DMD 真实数据（/mnt/e61/workspace/dmd/<group>/<subject>/<session>/*_mosaic.avi），
跨 group 抽 16 个 clip 各取首帧，覆盖多主体多场景。

用法（板端）:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 r8_board_eval.py --fp32 <om> --int8 <om> [--n-frames 16] [--json out.json]
"""
import argparse
import glob
import json
import os
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_dms_pipeline import AclContext, AclExecutor, LetterboxPreprocess, YoloV8Postprocess  # noqa

DMD_ROOT = "/mnt/e61/workspace/dmd"


def pick_clips(n):
    """跨 group/subject 均匀抽 clip（每 clip 用首帧）。"""
    clips = sorted(glob.glob(f"{DMD_ROOT}/*/*/*/*_mosaic.avi"))
    by_group = {}
    for c in clips:
        g = Path(c).parts[-4]          # .../dmd/<group>/<subject>/<session>/xx.avi
        by_group.setdefault(g, []).append(c)
    picked = []
    groups = sorted(by_group)
    gi = 0
    while len(picked) < n and groups:
        g = groups[gi % len(groups)]
        lst = by_group[g]
        if lst:
            picked.append(lst.pop(0))
        gi += 1
    return picked[:n]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xx1, yy1 = max(ax1, bx1), max(ay1, by1)
    xx2, yy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--json", default="/root/epb/results/r8_board_eval.json")
    args = ap.parse_args()

    clips = pick_clips(args.n_frames)
    print(f"clips picked: {len(clips)}")
    for c in clips:
        print(f"  {c}")

    pre = LetterboxPreprocess(imgsz=640)
    post = YoloV8Postprocess(conf=0.25, iou=0.45)

    ctx = AclContext()
    print("ACL inited")
    m_fp = AclExecutor(ctx, args.fp32)
    print(f"FP32 OM: inputs={m_fp.n_in} outputs={m_fp.n_out} out_dims={m_fp.out_dims} "
          f"out_sizes={m_fp.out_sizes}")
    m_qt = AclExecutor(ctx, args.int8)
    print(f"INT8 OM: inputs={m_qt.n_in} outputs={m_qt.n_out} out_dims={m_qt.out_dims} "
          f"out_sizes={m_qt.out_sizes}")

    cosines, det_fp, det_qt, conf_fp, conf_qt, match_iou = [], [], [], [], [], []
    t_exec_fp, t_exec_qt = [], []
    per_frame = []

    for ci, clip in enumerate(clips):
        cap = cv2.VideoCapture(clip)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"  [skip] cannot read {clip}")
            continue
        h0, w0 = frame.shape[:2]
        tensor, r, pad = pre(frame)

        of, tf = m_fp.execute([tensor])
        oq, tq = m_qt.execute([tensor])
        of, oq = of[0], oq[0]

        cos = float(np.dot(of.ravel(), oq.ravel()) /
                    (np.linalg.norm(of.ravel()) * np.linalg.norm(oq.ravel()) + 1e-12))
        cosines.append(cos)

        df = post(of, r, pad, (w0, h0))
        dq = post(oq, r, pad, (w0, h0))
        det_fp.append(len(df)); det_qt.append(len(dq))
        pf = [d["confidence"] for d in df if d["label"] == "person"]
        pq = [d["confidence"] for d in dq if d["label"] == "person"]
        conf_fp += pf; conf_qt += pq
        for a in df:
            cands = [b for b in dq if b["label"] == a["label"]]
            if cands:
                best = max(cands, key=lambda b: iou(a["bbox"], b["bbox"]))
                match_iou.append(iou(a["bbox"], best["bbox"]))

        t_exec_fp.append(tf["execute"] + tf["sync"])
        t_exec_qt.append(tq["execute"] + tq["sync"])
        per_frame.append({
            "clip": clip, "cosine": round(cos, 5),
            "det_fp32": len(df), "det_int8": len(dq),
            "exec_fp32_ms": round(tf["execute"] + tf["sync"], 3),
            "exec_int8_ms": round(tq["execute"] + tq["sync"], 3),
        })
        print(f"[{ci + 1}/{len(clips)}] cos={cos:.5f} det FP32={len(df)} INT8={len(dq)} "
              f"exec FP32={tf['execute'] + tf['sync']:.2f}ms INT8={tq['execute'] + tq['sync']:.2f}ms")

    m_fp.destroy(); m_qt.destroy(); ctx.destroy()

    out = {
        "eval_frames": len(cosines),
        "models": {"fp32": args.fp32, "int8": args.int8},
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
        "board_exec_ms": {
            "fp32": {"mean": round(statistics.mean(t_exec_fp), 3),
                     "median": round(statistics.median(t_exec_fp), 3)},
            "int8": {"mean": round(statistics.mean(t_exec_qt), 3),
                     "median": round(statistics.median(t_exec_qt), 3)},
        },
        "per_frame": per_frame,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"accuracy": out["accuracy"], "board_exec_ms": out["board_exec_ms"]},
                     indent=2, ensure_ascii=False))
    print(f"WROTE {args.json}")


if __name__ == "__main__":
    main()
