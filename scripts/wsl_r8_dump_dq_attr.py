#!/usr/bin/env python3
"""R8 诊断⑨：AscendDequant 节点 attribute 全貌 + dequant_scale initializer 双视角。"""
import onnx
import numpy as np
from onnx import numpy_helper

V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
m = onnx.load(V2)
g = m.graph

seen = 0
for n in g.node:
    if n.op_type == "AscendDequant":
        if seen < 2:
            print(f"=== {n.name}")
            print(f"  inputs: {n.input}")
            for a in n.attribute:
                extra = ""
                if a.type == onnx.AttributeProto.INTS:
                    extra = f"ints={list(a.ints)[:8]}"
                elif a.type == onnx.AttributeProto.FLOATS:
                    extra = f"floats={list(a.floats)[:8]}"
                elif a.type == onnx.AttributeProto.FLOAT:
                    extra = f"f={a.f}"
                elif a.type == onnx.AttributeProto.INT:
                    extra = f"i={a.i}"
                elif a.type == onnx.AttributeProto.TENSOR:
                    extra = f"tensor={numpy_helper.to_array(a.t).ravel()[:4]}"
                print(f"  attr {a.name}: type={a.type} {extra}")
        seen += 1
print(f"total AscendDequant: {seen}")

# 其 scale initializer：uint32 双视角 + 与 QDQ ref y_scale 对照
for i in g.initializer:
    if i.name.endswith("_dequant_scale"):
        b = np.frombuffer(i.raw_data, dtype=np.uint8)
        print(f"{i.name}: dtype={i.data_type} dims={list(i.dims)} raw={len(i.raw_data)}B")
        print(f"  as u32 lo/hi pairs: {list(zip(b.view(np.uint32)[0::2], b.view(np.uint32)[1::2]))[:8]}")
        break

REF = "/root/quant/r8/yolov8n_s3_quant_v2_cpu_quant.onnx"
m2 = onnx.load(REF)
for i in m2.graph.initializer:
    if i.name == "/model.0/conv/Conv_output_0_scale":
        print(f"QDQ ref y_scale = {numpy_helper.to_array(i)}")
        break
