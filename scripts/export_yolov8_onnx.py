"""Export YOLOv8-nano to ONNX for Ascend ONNX Runtime inference.

Usage:
    conda activate torchGPU
    python scripts/export_yolov8_onnx.py
"""

from pathlib import Path
from ultralytics import YOLO

MODEL_NAME = "yolov8n"
INPUT_SIZE = 640
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models"
OPSET = 11  # Ascend CANN best compatibility: opset 11


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_NAME}.pt ...")
    model = YOLO(f"{MODEL_NAME}.pt")

    print(f"Exporting to ONNX (opset={OPSET}, imgsz={INPUT_SIZE}) ...")
    model.export(
        format="onnx",
        opset=OPSET,
        imgsz=INPUT_SIZE,
        simplify=True,
        half=False,          # FP32 for Ascend
        dynamic=False,       # fixed shape 640x640
        batch=1,
    )

    onnx_path = OUTPUT_DIR / f"{MODEL_NAME}.onnx"
    print(f"Done → {onnx_path}  ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
