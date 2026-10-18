"""在 WSL 本地用 ultralytics 导出 YOLO 模型为 ONNX（给 ATC 编译用）。
用法: python3 wsl_export_yolo.py [model=yolov8s] [outdir=/root/onnx_out]
"""
import sys
from pathlib import Path
from ultralytics import YOLO

name = sys.argv[1] if len(sys.argv) > 1 else "yolov8s"
outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "/root/onnx_out")
outdir.mkdir(parents=True, exist_ok=True)

print(f"loading {name}.pt ...")
model = YOLO(f"{name}.pt")
print("exporting onnx ...")
p = model.export(format="onnx", opset=11, imgsz=640, simplify=False,
                 half=False, dynamic=False, batch=1)
dst = outdir / f"{name}.onnx"
src = Path(p)
if src.resolve() != dst.resolve():
    dst.write_bytes(src.read_bytes())
print(f"DONE onnx -> {dst}  ({dst.stat().st_size/1024/1024:.1f} MB)")
