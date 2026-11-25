#!/usr/bin/env python3
"""R8 诊断⑮：dump v2 的 AscendDequant 输出 tensor 名（建立中间对照点映射）。"""
import onnx

V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
m = onnx.load(V2)

for n in m.graph.node:
    if n.op_type == "AscendDequant":
        print(f"{n.name:<50} -> {n.output[0]}")
