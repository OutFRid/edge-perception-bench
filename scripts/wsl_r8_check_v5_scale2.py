#!/usr/bin/env python3
"""R8 检查报错节点的 scale shape + 对比 v2 同层 dequant 的 scale shape"""
import onnx
from onnx import numpy_helper

# v5 报错节点
m5 = onnx.load("/root/quant/r8/yolov8n_s3_quant_v5.onnx")
inits5 = {i.name: i for i in m5.graph.initializer}
print("=== v5 requant_dequant scale shapes (model.12/15/16/18/19/21) ===")
for n in m5.graph.node:
    if n.name.endswith("_requant_dequant") and any(
            k in n.name for k in ("model.12", "model.15", "model.16", "model.18", "model.19", "model.21")):
        arr = numpy_helper.to_array(inits5[n.input[1]])
        print(f"{n.name:65s} scale={arr.shape} dtype={arr.dtype}")

# 对比 v2 原始 dequant 的 scale shape（能编译成功的对照）
m2 = onnx.load("/root/quant/r8/yolov8n_s3_quant_v2.onnx")
inits2 = {i.name: i for i in m2.graph.initializer}
print("=== v2 AscendDequant scale shapes (model.12/15/16/18/19/21) ===")
for n in m2.graph.node:
    if n.op_type == "AscendDequant" and any(
            k in n.name for k in ("model.12", "model.15", "model.16", "model.18", "model.19", "model.21")):
        arr = numpy_helper.to_array(inits2[n.input[1]])
        print(f"{n.name:65s} scale={arr.shape} dtype={arr.dtype}")
