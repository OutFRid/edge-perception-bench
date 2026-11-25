#!/usr/bin/env python3
"""R8 诊断：dump v2 量化 ONNX 中 AscendQuant/AscendDequant 的 scale/offset，找异常值。"""
import onnx
import sys
import numpy as np
from onnx import numpy_helper

M = sys.argv[1] if len(sys.argv) > 1 else "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
m = onnx.load(M)
g = m.graph
inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}

def attr_float(n, name):
    for a in n.attribute:
        if a.name == name:
            return a.f
    return None

scales = []
print(f"{'op':16s} {'scale':>12s} {'offset':>8s}  name")
for n in g.node:
    if n.op_type in ("AscendQuant", "AscendDequant"):
        s = attr_float(n, "scale")
        o = attr_float(n, "offset")
        # 有的版本 scale 是张量输入（initializer）
        if s is None:
            for i in n.input:
                if i in inits and inits[i].size <= 8:
                    s = float(inits[i].ravel()[0])
        scales.append(s)
        flag = ""
        if s is not None and (abs(s) > 1e3 or abs(s) < 1e-12):
            flag = "   <<<<< SUSPICIOUS"
        print(f"{n.op_type:16s} {str(s):>12s} {str(o):>8s}  {n.name}{flag}")

if scales:
    sc = np.array([s for s in scales if s is not None])
    print(f"\ntotal {len(sc)} scales: min={sc.min():.3e} max={sc.max():.3e} "
          f"mean={sc.mean():.3e} median={np.median(sc):.3e}")
