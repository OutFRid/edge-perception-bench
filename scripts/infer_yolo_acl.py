"""YOLOv8-nano ACL inference on Ascend — first-frame DMD.

Usage:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 infer_yolo_acl.py --om /path/to/yolov8n.om --video /path/to/video.avi
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# ACL Python API wrapper
# ============================================================
ACL_SUCCESS = 0
ACL_DEVICE = 0  # device 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_FLOAT = 0


def _check(ret, msg=""):
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"ACL error {ret}: {msg}")


class AclContext:
    """ACL init → device set → context create."""

    def __init__(self):
        import acl
        self.acl = acl

        ret = acl.init("")
        _check(ret, "acl.init")

        _check(acl.rt.set_device(ACL_DEVICE), "acl.rt.set_device")

        self.context, ret = acl.rt.create_context(ACL_DEVICE)
        _check(ret, "acl.rt.create_context")

    def destroy(self):
        self.acl.rt.destroy_context(self.context)
        self.acl.rt.reset_device(ACL_DEVICE)
        self.acl.finalize()


class AclModel:
    """OM model loader + executor."""

    def __init__(self, acl_ctx, om_path: str):
        self.ctx = acl_ctx
        self.acl = acl_ctx.acl

        self.model_id, ret = self.acl.mdl.load_from_file(om_path)
        _check(ret, "acl.mdl.load_from_file")

        self.desc = self.acl.mdl.create_desc()
        _check(self.acl.mdl.get_desc(self.desc, self.model_id), "acl.mdl.get_desc")

        self.num_inputs = self.acl.mdl.get_num_inputs(self.desc)
        self.num_outputs = self.acl.mdl.get_num_outputs(self.desc)

        self.input_dims = self._get_dims(self.num_inputs, self.acl.mdl.get_input_dims)
        self.input_sizes = self._get_sizes(self.num_inputs, self.acl.mdl.get_input_size_by_index)
        self.output_dims = self._get_dims(self.num_outputs, self.acl.mdl.get_output_dims)
        self.output_sizes = self._get_sizes(self.num_outputs, self.acl.mdl.get_output_size_by_index)

        print(f"  Inputs  [{self.num_inputs}]: dims={self.input_dims}  sizes={self.input_sizes}")
        print(f"  Outputs [{self.num_outputs}]: dims={self.output_dims}  sizes={self.output_sizes}")

    def _get_dims(self, count, fn):
        result = []
        for i in range(count):
            info, ret = fn(self.desc, i)
            _check(ret, "get_dims")
            dims_info = info.get("dims") or info.get("dim")
            if isinstance(dims_info, list) and len(dims_info) > 0:
                if isinstance(dims_info[0], dict):
                    result.append([d["value"] for d in dims_info])
                else:
                    result.append(dims_info)
            else:
                result.append([])
        return result

    def _get_sizes(self, count, fn):
        sizes = []
        for i in range(count):
            sizes.append(fn(self.desc, i))
        return sizes

    def run(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        acl = self.acl
        in_dev_ptrs = []
        out_dev_ptrs = []
        in_bufs = []
        out_bufs = []

        try:
            # --- input: alloc device mem, copy, wrap into data_buffer ---
            for i, arr in enumerate(inputs):
                arr = np.ascontiguousarray(arr.astype(np.float32))
                size = arr.nbytes
                if size != self.input_sizes[i]:
                    raise ValueError(f"Input[{i}] size mismatch: {arr.nbytes} vs {self.input_sizes[i]}")

                dev_ptr, ret = acl.rt.malloc(size, 0)
                _check(ret, f"rt.malloc(in,{i})")
                in_dev_ptrs.append(dev_ptr)

                ret = acl.rt.memcpy(dev_ptr, size, arr.ctypes.data, size, ACL_MEMCPY_HOST_TO_DEVICE)
                _check(ret, f"memcpy H2D({i})")

                buf = acl.create_data_buffer(dev_ptr, size)
                in_bufs.append(buf)

            # --- output: alloc device mem, wrap into data_buffer ---
            for i, size in enumerate(self.output_sizes):
                dev_ptr, ret = acl.rt.malloc(size, 0)
                _check(ret, f"rt.malloc(out,{i})")
                out_dev_ptrs.append(dev_ptr)

                buf = acl.create_data_buffer(dev_ptr, size)
                out_bufs.append(buf)

            # --- datasets ---
            in_ds = acl.mdl.create_dataset()
            for i, buf in enumerate(in_bufs):
                _, ret = acl.mdl.add_dataset_buffer(in_ds, buf)
                _check(ret, f"add_dataset_buffer(in,{i})")

            out_ds = acl.mdl.create_dataset()
            for i, buf in enumerate(out_bufs):
                _, ret = acl.mdl.add_dataset_buffer(out_ds, buf)
                _check(ret, f"add_dataset_buffer(out,{i})")

            # --- execute ---
            _check(acl.mdl.execute(self.model_id, in_ds, out_ds), "acl.mdl.execute")
            _check(acl.rt.synchronize_stream(0), "synchronize_stream")

            # --- copy outputs back ---
            outputs = []
            for i in range(self.num_outputs):
                size = self.output_sizes[i]
                out_arr = np.empty(size // 4, dtype=np.float32)
                ret = acl.rt.memcpy(out_arr.ctypes.data, size, out_dev_ptrs[i], size, ACL_MEMCPY_DEVICE_TO_HOST)
                _check(ret, f"memcpy D2H({i})")
                shape = self.output_dims[i] if self.output_dims[i] else [size // 4]
                outputs.append(out_arr.reshape(shape))

            return outputs

        finally:
            for buf in in_bufs + out_bufs:
                try:
                    acl.destroy_data_buffer(buf)
                except Exception:
                    pass
            try:
                acl.mdl.destroy_dataset(in_ds)
            except Exception:
                pass
            try:
                acl.mdl.destroy_dataset(out_ds)
            except Exception:
                pass
            for ptr in in_dev_ptrs + out_dev_ptrs:
                try:
                    acl.rt.free(ptr)
                except Exception:
                    pass

    def destroy(self):
        self.acl.mdl.unload(self.model_id)
        self.acl.mdl.destroy_desc(self.desc)


# ============================================================
# YOLOv8 postprocessing
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


def preprocess(frame: np.ndarray, imgsz: int = 640) -> np.ndarray:
    img = cv2.resize(frame, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


def postprocess(output: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> list[dict]:
    output = np.squeeze(output)
    output = np.transpose(output)

    boxes_raw = output[:, :4]
    scores = output[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confs = np.max(scores, axis=1)

    mask = confs > conf
    boxes_raw, class_ids, confs = boxes_raw[mask], class_ids[mask], confs[mask]
    if len(boxes_raw) == 0:
        return []

    cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).clip(0, None)

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xyxy.tolist(),
        scores=confs.tolist(),
        score_threshold=conf,
        nms_threshold=iou,
    )

    detections = []
    indices_flat = indices.flatten() if len(indices) else []
    for i in indices_flat:
        idx = int(i)
        detections.append({
            "class": int(class_ids[idx]),
            "label": COCO_CLASSES[int(class_ids[idx])],
            "confidence": round(float(confs[idx]), 4),
            "bbox": [round(float(v), 1) for v in boxes_xyxy[idx]],
        })
    return detections


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--om", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", help="Save result image path")
    args = parser.parse_args()

    om_path = Path(args.om)
    video_path = Path(args.video)
    if not om_path.exists():
        sys.exit(f"OM not found: {om_path}")
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    # ---- 1. Read video ----
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    if not ok:
        sys.exit("Cannot read frame")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    h, w = frame.shape[:2]
    print(f"Video : {video_path.name}")
    print(f"       {w}x{h}, {total} frames, {fps:.1f} fps")

    # ---- 2. Preprocess ----
    tensor = preprocess(frame)
    print(f"Tensor: {tensor.shape}  {tensor.dtype}")

    # ---- 3. Init ACL + load model + infer ----
    print("\n--- ACL Init ---")
    ctx = AclContext()
    print("  OK")

    print("\n--- Load OM ---")
    model = AclModel(ctx, str(om_path))

    print("\n--- Infer ---")
    t0 = time.perf_counter()
    outputs = model.run([tensor])
    t1 = time.perf_counter()

    model.destroy()
    ctx.destroy()

    # ---- 4. Postprocess ----
    detections = postprocess(outputs[0])

    # ---- 5. Report ----
    latency_ms = (t1 - t0) * 1000
    print(f"\n{'='*50}")
    print(f"Latency   : {latency_ms:.1f} ms")
    print(f"Detections: {len(detections)}")
    for d in detections:
        print(f"  [{d['label']:<14}] conf={d['confidence']:.3f}  box={d['bbox']}")
    print(f"{'='*50}")

    result = {
        "video": str(video_path),
        "resolution": [w, h],
        "fps": fps,
        "model": str(om_path),
        "latency_ms": round(latency_ms, 2),
        "detections": detections,
    }
    print(f"\n{json.dumps(result, indent=2, ensure_ascii=False)}")

    if args.save:
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{d['label']} {d['confidence']:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(args.save, frame)
        print(f"\nSaved: {args.save}")


if __name__ == "__main__":
    main()
