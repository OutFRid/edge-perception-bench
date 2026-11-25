#!/usr/bin/env python3
"""R8 诊断④：dump v2 的 AscendDequant scale（Constant 节点 *_dequant_scale）值，判断方向。"""
import onnx
from onnx import numpy_helper

V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
m = onnx.load(V2)
g = m.graph

consts = {}
for n in g.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value" and a.t.raw_data:
                arr = numpy_helper.to_array(a.t)
                if arr.size == 1:
                    consts[n.output[0]] = float(arr.ravel()[0])

print("=== AscendDequant scale (first 8) ===")
cnt = 0
for n in g.node:
    if n.op_type == "AscendDequant":
        cnt += 1
        if cnt <= 8:
            print(f"  {n.name}: inputs={list(n.input)}")
            for i in n.input[1:]:
                if i in consts:
                    print(f"      {i} = {consts[i]}")
print(f"total Dequant: {cnt}")
