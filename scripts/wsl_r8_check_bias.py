#!/usr/bin/env python3
"""R8 诊断⑰：检查 v2 所有 int8 Conv 的 int32 bias 量级（溢出回绕嫌疑）。"""
import numpy as np
import onnx

V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
m = onnx.load(V2)
g = m.graph

inits = {i.name: i for i in g.initializer}
print(f"{'conv':<40} {'bias_min':<12} {'bias_max':<12} {'C_out':<8} {'w_scale_est'}")
for n in g.node:
    if n.op_type != "Conv":
        continue
    w = inits.get(n.input[1])
    if w is None or w.data_type != onnx.TensorProto.INT8:
        continue
    if len(n.input) < 3:
        print(f"{n.name:<40} NO BIAS")
        continue
    b = inits.get(n.input[2])
    if b is None:
        print(f"{n.name:<40} bias init MISSING")
        continue
    ba = np.frombuffer(b.raw_data, dtype=np.int32)
    flag = "  <-- OVERFLOW RISK" if (abs(ba.min()) > 2**30 or abs(ba.max()) > 2**30) else ""
    print(f"{n.name:<40} {ba.min():<12} {ba.max():<12} {len(ba):<8}{flag}")
