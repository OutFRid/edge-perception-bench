#!/usr/bin/env python3
"""R8 诊断③：同一 Conv 节点在「QDQ 参考模型」与「v2 AscendQuant 图」中的激活量化 scale 对比。"""
import numpy as np
import onnx
from onnx import numpy_helper

REF = "/root/quant/r8/yolov8n_s3_quant_v2_cpu_quant.onnx"
V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"


def conv_scales(model_path, qtype):
    m = onnx.load(model_path)
    g = m.graph
    table = {}
    for i in g.initializer:
        a = numpy_helper.to_array(i)
        if a.size == 1:
            table[i.name] = float(a.ravel()[0])
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value" and a.t.raw_data:
                    arr = numpy_helper.to_array(a.t)
                    if arr.size == 1:
                        table[n.output[0]] = float(arr.ravel()[0])
    # 张量 -> 消费者
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    out = {}
    if qtype == "qdq":
        for n in g.node:
            if n.op_type == "QuantizeLinear":
                scale = table.get(n.input[1], None)
                for c in consumers.get(n.output[0], []):
                    if c.op_type == "Conv":
                        out[c.name] = scale
    else:  # ascend
        for n in g.node:
            if n.op_type == "AscendQuant":
                s = None
                for a in n.attribute:
                    if a.name == "scale":
                        s = a.f
                for c in consumers.get(n.output[0], []):
                    if c.op_type == "Conv":
                        out[c.name.replace("_quant", "")] = s
    return out


ref = conv_scales(REF, "qdq")
v2 = conv_scales(V2, "ascend")

print(f"{'Conv node':58s} {'QDQ y_scale':>12s} {'AscendQuant':>12s} {'v2/ref':>10s}")
bad = 0
for name in sorted(set(ref) | set(v2)):
    r = ref.get(name)
    v = v2.get(name)
    if r is None or v is None:
        continue
    ratio = v / r if r else float("inf")
    mark = ""
    if abs(ratio - 1.0) > 0.05:
        bad += 1
        mark = "  <<< DIFF"
    print(f"{name:58s} {r:12.6f} {v:12.6f} {ratio:10.4f}{mark}")
print(f"\nmatched={len(set(ref) & set(v2))}  diff(ratio!=1)={bad}")
