#!/usr/bin/env python3
"""R8 诊断⑥：直接 dump v2 里 *_dequant_scale initializer 的值。"""
import onnx
from onnx import numpy_helper

V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
m = onnx.load(V2)
g = m.graph

for i in g.initializer:
    if "dequant_scale" in i.name:
        a = numpy_helper.to_array(i)
        print(f"  {i.name}: dtype={i.data_type} dims={list(i.dims)} raw={len(i.raw_data)}B val={a.ravel()[:3]}")
        break  # 只打第一个
print("--- all dequant_scale values ---")
for i in g.initializer:
    if "dequant_scale" in i.name:
        a = numpy_helper.to_array(i)
        print(f"  {i.name.split('/')[-2] if '/' in i.name else i.name}: {float(a.ravel()[0]) if a.size else None}")
