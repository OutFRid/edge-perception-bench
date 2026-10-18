"""DMS 推理流水线 · 底层 ACL 裸调（仿 MindSDK YOLOv7Detection 的 Pipeline 骨架）。

设计对应 `docs/一帧图的旅程-ACL推理数据流.md` 的 6 阶段数据流：

    Decode -> Preprocess(letterbox) -> H2D -> Execute(OM) -> Sync -> D2H -> Postprocess -> Render

与 `infer_yolo_acl.py`（单帧、把 H2D+execute+D2H 混在一个 latency 里）不同，本脚本：
  1. 以「流水线阶段」组织代码（Preprocessor / AclExecutor / Postprocessor / Renderer），
     对齐参考骨架 MxBase 的 ImageProcessor / Model / PostProcess 分层；
  2. 逐帧流式处理整段视频（DMS 连续帧场景）；
  3. 对每个阶段**单独计时**，直接产出 §9 待补的分阶段耗时表，为 R4 profiling 铺路。

注意：当前加载的是 COCO yolov8n 检测模型，DMS 语义上应换人脸/驾驶状态模型（R3 计划），
本流水线骨架换模型时只需替换 OM 路径与 postprocess 的类别/解码，阶段结构不变。

用法（板端）:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 infer_dms_pipeline.py \
        --om /mnt/e61/workspace/dmd/model/yolov8n.om \
        --video /mnt/e61/workspace/dmd/gA/1/s1/xxx_mosaic.avi \
        --out  /tmp/dms_out.mp4 --max-frames 120
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ACL_SUCCESS = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _check(ret, msg=""):
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"ACL error {ret}: {msg}")


# ============================================================
# Stage 0: ACL 运行时（context + 模型生命周期）
# ============================================================
class AclContext:
    """acl.init -> set_device -> create_context（对应参考骨架 MxInit）。"""

    def __init__(self, device=0):
        import acl
        self.acl = acl
        _check(acl.init(""), "acl.init")
        _check(acl.rt.set_device(device), "set_device")
        self.context, ret = acl.rt.create_context(device)
        _check(ret, "create_context")

    def destroy(self):
        self.acl.rt.destroy_context(self.context)
        self.acl.rt.reset_device(0)
        self.acl.finalize()


class AclExecutor:
    """加载 OM 并执行推理；把 H2D / execute / sync / D2H 拆开分别计时。"""

    def __init__(self, ctx: AclContext, om_path: str):
        self.ctx = ctx
        self.acl = ctx.acl
        self.model_id, ret = self.acl.mdl.load_from_file(om_path)
        _check(ret, "load_from_file")
        self.desc = self.acl.mdl.create_desc()
        _check(self.acl.mdl.get_desc(self.desc, self.model_id), "get_desc")
        self.n_in = self.acl.mdl.get_num_inputs(self.desc)
        self.n_out = self.acl.mdl.get_num_outputs(self.desc)
        self.in_sizes = [self.acl.mdl.get_input_size_by_index(self.desc, i) for i in range(self.n_in)]
        self.out_sizes = [self.acl.mdl.get_output_size_by_index(self.desc, i) for i in range(self.n_out)]
        self.out_dims = [self._dims(self.acl.mdl.get_output_dims, i) for i in range(self.n_out)]

    def _dims(self, fn, i):
        info, ret = fn(self.desc, i)
        _check(ret, "get_output_dims")
        d = info.get("dims") or info.get("dim")
        if isinstance(d, list) and d and isinstance(d[0], dict):
            return [x["value"] for x in d]
        return d if isinstance(d, list) else []

    def execute(self, inputs):
        """返回 (outputs, timing_dict)。timing 单位 ms。"""
        acl = self.acl
        t_h2d = t_exec = t_sync = t_d2h = 0.0
        in_dev, out_dev, in_bufs, out_bufs = [], [], [], []
        in_ds = out_ds = None
        try:
            # --- H2D ---
            t0 = time.perf_counter()
            for i, arr in enumerate(inputs):
                arr = np.ascontiguousarray(arr.astype(np.float32))
                if arr.nbytes != self.in_sizes[i]:
                    raise ValueError(f"input[{i}] size {arr.nbytes} != {self.in_sizes[i]}")
                p, ret = acl.rt.malloc(arr.nbytes, 0)
                _check(ret, "malloc(in)")
                in_dev.append(p)
                _check(acl.rt.memcpy(p, arr.nbytes, arr.ctypes.data, arr.nbytes,
                                     ACL_MEMCPY_HOST_TO_DEVICE), "memcpy H2D")
                b = acl.create_data_buffer(p, arr.nbytes)
                in_bufs.append(b)
            for i, size in enumerate(self.out_sizes):
                p, ret = acl.rt.malloc(size, 0)
                _check(ret, "malloc(out)")
                out_dev.append(p)
                out_bufs.append(acl.create_data_buffer(p, size))
            t_h2d = (time.perf_counter() - t0) * 1000

            # --- datasets ---
            in_ds = acl.mdl.create_dataset()
            for b in in_bufs:
                _check(acl.mdl.add_dataset_buffer(in_ds, b)[1], "add in")
            out_ds = acl.mdl.create_dataset()
            for b in out_bufs:
                _check(acl.mdl.add_dataset_buffer(out_ds, b)[1], "add out")

            # --- execute（异步入队）---
            t0 = time.perf_counter()
            _check(acl.mdl.execute(self.model_id, in_ds, out_ds), "execute")
            t_exec = (time.perf_counter() - t0) * 1000

            # --- sync（真正的栅栏）---
            t0 = time.perf_counter()
            _check(acl.rt.synchronize_stream(0), "sync")
            t_sync = (time.perf_counter() - t0) * 1000

            # --- D2H ---
            t0 = time.perf_counter()
            outputs = []
            for i in range(self.n_out):
                size = self.out_sizes[i]
                host = np.empty(size // 4, dtype=np.float32)
                _check(acl.rt.memcpy(host.ctypes.data, size, out_dev[i], size,
                                     ACL_MEMCPY_DEVICE_TO_HOST), "memcpy D2H")
                shape = self.out_dims[i] if self.out_dims[i] else [size // 4]
                outputs.append(host.reshape(shape))
            t_d2h = (time.perf_counter() - t0) * 1000

            return outputs, {"h2d": t_h2d, "execute": t_exec, "sync": t_sync, "d2h": t_d2h}
        finally:
            for b in in_bufs + out_bufs:
                try:
                    acl.destroy_data_buffer(b)
                except Exception:
                    pass
            for ds in (in_ds, out_ds):
                if ds is not None:
                    try:
                        acl.mdl.destroy_dataset(ds)
                    except Exception:
                        pass
            for p in in_dev + out_dev:
                try:
                    acl.rt.free(p)
                except Exception:
                    pass

    def destroy(self):
        self.acl.mdl.unload(self.model_id)
        self.acl.mdl.destroy_desc(self.desc)


# ============================================================
# Stage 1: Preprocess（letterbox，Host 侧）
# ============================================================
class LetterboxPreprocess:
    def __init__(self, imgsz=640, color=(114, 114, 114)):
        self.imgsz, self.color = imgsz, color

    def __call__(self, frame):
        h0, w0 = frame.shape[:2]
        r = min(self.imgsz / h0, self.imgsz / w0)
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), self.color, dtype=np.uint8)
        left, top = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        canvas[top:top + nh, left:left + nw] = resized
        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, 0), r, (left, top)


# ============================================================
# Stage 2: Postprocess（YOLOv8 解码 + NMS + 坐标逆映射，Host 侧）
# ============================================================
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class YoloV8Postprocess:
    def __init__(self, conf=0.25, iou=0.45):
        self.conf, self.iou = conf, iou

    def __call__(self, output, ratio, pad, orig_wh):
        o = np.squeeze(output)
        o = np.transpose(o)
        boxes = o[:, :4]
        scores = o[:, 4:]
        cls = np.argmax(scores, 1)
        conf = np.max(scores, 1)
        m = conf > self.conf
        boxes, cls, conf = boxes[m], cls[m], conf[m]
        if len(boxes) == 0:
            return []
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        nms_in = [[float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])]
                  for i in range(len(x1))]
        idx = cv2.dnn.NMSBoxes(nms_in, conf.tolist(), self.conf, self.iou)
        pl, pt = pad
        ow, oh = orig_wh
        dets = []
        for i in (idx.flatten() if len(idx) else []):
            i = int(i)
            bx1 = min(max((x1[i] - pl) / ratio, 0), ow)
            by1 = min(max((y1[i] - pt) / ratio, 0), oh)
            bx2 = min(max((x2[i] - pl) / ratio, 0), ow)
            by2 = min(max((y2[i] - pt) / ratio, 0), oh)
            dets.append({"class": int(cls[i]), "label": COCO_CLASSES[int(cls[i])],
                         "confidence": round(float(conf[i]), 4),
                         "bbox": [round(float(bx1), 1), round(float(by1), 1),
                                  round(float(bx2), 1), round(float(by2), 1)]})
        return dets


# ============================================================
# 流水线编排 + 分段计时
# ============================================================
STAGES = ["decode", "preprocess", "h2d", "execute", "sync", "d2h", "postprocess", "render"]


def run_pipeline(om_path, video_path, out_path=None, max_frames=0, conf=0.25, sample_jpeg=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video : {Path(video_path).name}  {w}x{h}  {total} frames  {fps:.1f}fps")

    ok, first = cap.read()
    if not ok:
        sys.exit("cannot read first frame")
    cap.release()

    ctx = AclContext()
    executor = AclExecutor(ctx, om_path)
    pre = LetterboxPreprocess()
    post = YoloV8Postprocess(conf=conf)

    # 用首帧打印一次输入/输出形状，验证数据流
    tensor, ratio, pad = pre(first)
    print(f"Input tensor: {tensor.shape} {tensor.dtype}  ratio={ratio:.4f} pad={pad}")

    outs, t = executor.execute([tensor])
    dets = post(outs[0], ratio, pad, (w, h))
    print(f"Output tensor: {outs[0].shape}  detections(first frame): {len(dets)}")
    for d in dets:
        print(f"  [{d['label']:<12}] conf={d['confidence']:.3f} box={d['bbox']}")

    # ---- 逐帧流式跑全片，累计各阶段耗时 ----
    cap = cv2.VideoCapture(str(video_path))
    acc = {s: 0.0 for s in STAGES}
    n = 0
    writer = None
    person_frames = 0
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            break
        acc["decode"] += (t1 - t0) * 1000

        tensor, ratio, pad = pre(frame)
        acc["preprocess"] += (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()

        outs, te = executor.execute([tensor])
        acc["h2d"] += te["h2d"]
        acc["execute"] += te["execute"]
        acc["sync"] += te["sync"]
        acc["d2h"] += te["d2h"]

        dets = post(outs[0], ratio, pad, (w, h))
        acc["postprocess"] += (time.perf_counter() - t2) * 1000

        if any(d["label"] == "person" for d in dets):
            person_frames += 1

        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{d['label']} {d['confidence']:.2f}",
                        (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 保存第一张含 person 检出的标注单帧，供周志/对比使用
        if sample_jpeg and person_frames == 1:
            cv2.imwrite(str(sample_jpeg), frame)

        if out_path:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            t3 = time.perf_counter()
            writer.write(frame)
            acc["render"] += (time.perf_counter() - t3) * 1000

        n += 1
        if max_frames and n >= max_frames:
            break
    cap.release()
    if writer:
        writer.release()

    executor.destroy()
    ctx.destroy()

    print(f"\n{'='*58}")
    print(f"Processed {n} frames   person 检出帧 {person_frames} ({person_frames / max(n,1):.0%})")
    if out_path:
        print(f"Annotated video -> {out_path}")
    print("\n分段耗时均值 (ms/frame)  [§9 分阶段耗时表 · 直接可填]")
    e2e = sum(acc[s] for s in STAGES)
    for s in STAGES:
        mean = acc[s] / max(n, 1)
        pct = acc[s] / e2e * 100 if e2e else 0
        print(f"  {s:<12} {mean:7.2f} ms   {pct:5.1f}%")
    print(f"  {'-'*40}")
    print(f"  {'E2E 合计':<10} {e2e / max(n,1):7.2f} ms")
    print(f"  {'吞吐':<12} {n / (e2e / 1000) if e2e else 0:.2f} fps")
    print(f"{'='*58}")

    return {"frames": n, "person_frames": person_frames,
            "stage_ms_mean": {s: round(acc[s] / max(n, 1), 2) for s in STAGES},
            "e2e_ms_mean": round(e2e / max(n, 1), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--om", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", help="annotated output .mp4 path")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--json", help="dump stats json")
    ap.add_argument("--sample-jpeg", help="save first annotated frame with a person detection")
    args = ap.parse_args()
    if not Path(args.om).exists():
        sys.exit(f"OM not found: {args.om}")
    if not Path(args.video).exists():
        sys.exit(f"video not found: {args.video}")
    stats = run_pipeline(args.om, args.video, args.out, args.max_frames, args.conf, args.sample_jpeg)
    if args.json:
        Path(args.json).write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        print(f"stats -> {args.json}")


if __name__ == "__main__":
    main()
