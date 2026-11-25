#!/usr/bin/env python3
"""R8 诊断⑦：QDQ 参考模型 dequant scale 的 dtype/dims + v2 同节点的 raw 结构。"""
import onnx
from onnx import numpy_helper

for path, tag in [("/root/quant/r8/yolov8n_s3_quant_v2_cpu_quant.onnx", "QDQ-ref"),
                  ("/root/quant/r8/yolov8n_s3_quant_v2.onnx", "v2")]:
    m = onnx.load(path)
    g = m.graph
    print(f"=== {tag} ===")
    cnt = 0
    for i in g.initializer:
        if "scale" in i.name:
            cnt += 1
            if cnt <= 3:
                print(f"  {i.name}: dtype={i.data_type} dims={list(i.dims)} raw={len(i.raw_data)}B")
    print(f"  total scale inits: {cnt}")
    # QDQ 的 DequantizeLinear scale 输入
    if tag == "QDQ-ref":
        for n in g.node:
            if n.op_type == "DequantizeLinear":
                print(f"  DQ {n.name}: scale input = {n.input[1]}")
                break
