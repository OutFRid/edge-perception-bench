#!/usr/bin/env python3
"""R8 诊断⑤：找 *_dequant_scale 张量的生产者及完整属性。"""
import onnx

V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
m = onnx.load(V2)
g = m.graph

prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n

seen = set()
for n in g.node:
    if n.op_type == "AscendDequant":
        t = n.input[1]
        if t in seen:
            continue
        seen.add(t)
        p = prod.get(t)
        print(f"tensor {t}")
        if p is None:
            print("   producer: <none>")
        else:
            print(f"   producer: {p.op_type} {p.name}")
            for a in p.attribute:
                print(f"      attr {a.name} type={a.type} f={a.f} i={a.i} "
                      f"floats={list(a.floats)[:4] if a.floats else None}")
