#!/usr/bin/env python3
"""R8 检查 v5 requant scale shape / 输入 dtype / 输出 dtype"""
import onnx
from onnx import numpy_helper
import numpy as np

m = onnx.load("/root/quant/r8/yolov8n_s3_quant_v5.onnx")
inits = {i.name: i for i in m.graph.initializer}
nodes = {n.name: n for n in m.graph.node}

shown = 0
for n in m.graph.node:
    if n.name.endswith("_requant_dequant"):
        scale_name = n.input[1]
        sc = inits.get(scale_name)
        arr = numpy_helper.to_array(sc)
        # 输入 cast 节点的输出类型未知，看 cast 的 to 属性
        cast_node = nodes.get(n.input[0].rsplit("/", 0)[0]) if False else None
        print(f"{n.name[:60]:62s} scale_shape={arr.shape} scale_dtype={arr.dtype} in={n.input[0][-30:]}")
        shown += 1
        if shown >= 5:
            break

# 检查一个 requant 的 cast 节点
for n in m.graph.node:
    if n.name.endswith("_requant_cast"):
        print("cast node:", n.name, "to=", [a.i for a in n.attribute], "input:", n.input[0][-30:])
        break
