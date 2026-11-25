#!/usr/bin/env python3
"""R8 诊断③b：直接看 QDQ 参考模型的量化结构（QuantizeLinear 消费者）。"""
import onnx
from onnx import numpy_helper

REF = "/root/quant/r8/yolov8n_s3_quant_v2_cpu_quant.onnx"
m = onnx.load(REF)
g = m.graph
inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}

consumers = {}
for n in g.node:
    for i in n.input:
        consumers.setdefault(i, []).append(n)

print("=== QuantizeLinear nodes (first 6) ===")
cnt = 0
for n in g.node:
    if n.op_type == "QuantizeLinear":
        cnt += 1
        if cnt <= 6:
            scale = inits.get(n.input[1], None)
            s = float(scale.ravel()[0]) if scale is not None else None
            cons = [(c.op_type, c.name) for c in consumers.get(n.output[0], [])]
            print(f"  QL in={list(n.input)} scale={s} -> {cons}")
print(f"total QuantizeLinear: {cnt}")
print("=== op histogram ===")
ops = {}
for n in g.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
for k in sorted(ops):
    print(f"  {k:20s} {ops[k]}")
