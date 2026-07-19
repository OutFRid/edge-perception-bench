"""YOLOv8-nano ONNX Runtime Ascend — first-frame inference on DMD.

Usage:
    python3 infer_yolo_frame.py --model /mnt/e61/workspace/dmd/model/yolov8n.onnx \
                                --video /mnt/e61/workspace/dmd/gA/1/s6/*_mosaic.avi \
                                --task hands
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

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
    """Resize + normalize frame to [1, 3, H, W] float32."""
    img = cv2.resize(frame, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))       # HWC -> CHW
    return np.expand_dims(img, axis=0)        # -> NCHW


def postprocess(output: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> list[dict]:
    """Decode YOLOv8 output [1, 84, 8400] -> list of detections."""
    output = np.squeeze(output[0])            # [84, 8400]
    output = np.transpose(output)              # [8400, 84]

    boxes = output[:, :4]
    scores = output[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confs = np.max(scores, axis=1)

    mask = confs > conf
    boxes, class_ids, confs = boxes[mask], class_ids[mask], confs[mask]

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=confs.tolist(),
        score_threshold=conf,
        nms_threshold=iou,
    )

    detections = []
    for i in (indices.flatten() if len(indices) else []):
        x, y, w, h = boxes[i]
        detections.append({
            "class": int(class_ids[i]),
            "label": COCO_CLASSES[int(class_ids[i])],
            "confidence": float(confs[i]),
            "bbox": [float(x), float(y), float(w), float(h)],
        })
    return detections


def read_first_frame(video_path: Path) -> tuple[np.ndarray, float]:
    """Read the first frame of a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read first frame: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    print(f"  Video: {video_path.name}  ({int(total)} frames, {fps:.1f} fps, {frame.shape[1]}x{frame.shape[0]})")
    return frame, fps


def find_video(dmd_root: Path, group: str, session: int, sub_session: str, camera: str) -> Path:
    """Find a specific video in DMD dataset."""
    pattern = f"**/{group}/{session}/{sub_session}/*rgb_{camera}.*"
    matches = list(dmd_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {camera} video for {group}/{session}/{sub_session}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to YOLOv8 .onnx model")
    parser.add_argument("--video", help="Path to a specific video file (overrides --dmd-root)")
    parser.add_argument("--dmd-root", default="/mnt/e61/workspace/dmd", help="DMD dataset root")
    parser.add_argument("--group", default="gA", help="Group name e.g. gA")
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--sub-session", default="s6")
    parser.add_argument("--camera", default="mosaic", help="mosaic / face / body / hands")
    parser.add_argument("--provider", default="AscendExecutionProvider", help="ONNX Runtime EP")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", help="Save result image path")
    parser.add_argument("--json-only", action="store_true", help="Only print JSON, no image output")
    args = parser.parse_args()

    dmd_root = Path(args.dmd_root)

    # ---- 1. Find video ----
    if args.video:
        video_path = Path(args.video)
    else:
        video_path = find_video(dmd_root, args.group, args.session, args.sub_session, args.camera)
    print(f"Video: {video_path}")

    # ---- 2. Read first frame ----
    frame, fps = read_first_frame(video_path)
    h, w = frame.shape[:2]

    # ---- 3. Preprocess ----
    tensor = preprocess(frame)
    print(f"  Input tensor: {tensor.shape}  ({tensor.dtype})")

    # ---- 4. Load ONNX Runtime session ----
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 2  # warnings only

    providers = [args.provider, "CPUExecutionProvider"]
    print(f"\nONNX Runtime available providers: {ort.get_available_providers()}")
    print(f"Requested provider: {args.provider}")

    try:
        sess = ort.InferenceSession(args.model, sess_opts=sess_opts, providers=providers)
    except Exception as e:
        print(f"  {args.provider} failed: {e}")
        print("  Falling back to CPUExecutionProvider")
        sess = ort.InferenceSession(args.model, sess_opts=sess_opts, providers=["CPUExecutionProvider"])

    input_name = sess.get_inputs()[0].name
    print(f"  Model input:  {input_name}  {sess.get_inputs()[0].shape}")
    print(f"  Model output: {sess.get_outputs()[0].name}  {sess.get_outputs()[0].shape}")

    # ---- 5. Inference ----
    t0 = time.perf_counter()
    outputs = sess.run(None, {input_name: tensor})
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    # ---- 6. Postprocess ----
    detections = postprocess(outputs[0], conf=args.conf)
    print(f"\n--- Inference Result ---")
    print(f"  Latency:  {latency_ms:.1f} ms")
    print(f"  Detections: {len(detections)}")
    for d in detections:
        print(f"    [{d['label']}]  conf={d['confidence']:.3f}  bbox={[f'{v:.0f}' for v in d['bbox']]}")

    # ---- 7. Output JSON ----
    result = {
        "video": str(video_path),
        "resolution": [w, h],
        "fps": fps,
        "model": args.model,
        "provider": sess.get_providers()[0],
        "latency_ms": round(latency_ms, 2),
        "detections": detections,
    }
    print(f"\n{json.dumps(result, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
