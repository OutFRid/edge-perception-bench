#!/usr/bin/env python3
"""R8 诊断⑧：解释 v2 dequant_scale initializer 的 raw 字节（uint64/float64/float32 视角）。"""
import onnx
import numpy as np

V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
m = onnx.load(V2)

for i in m.graph.initializer:
    if "dequant_scale" in i.name:
        raw = i.raw_data
        print(f"{i.name}: dtype={i.data_type} dims={list(i.dims)} raw={len(raw)}B")
        if raw:
            b = np.frombuffer(raw, dtype=np.uint8)
            print(f"  as uint64 : {b.view(np.uint64)[:4]}")
            print(f"  as float64: {b.view(np.float64)[:4]}")
            print(f"  as float32: {b.view(np.float32)[:6]}")
        break

# 参考：QDQ 的对应 y_scale 值
REF = "/root/quant/r8/yolov8n_s3_quant_v2_cpu_quant.onnx"
m2 = onnx.load(REF)
for i in m2.graph.initializer:
    if i.name == "/model.0/conv/Conv_output_0_scale":
        a = onnx.numpy_helper.to_array(i)
        print(f"QDQ ref y_scale = {a}")
        break
