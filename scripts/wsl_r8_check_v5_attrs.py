#!/usr/bin/env python3
"""R8 检查 v5 ONNX requant 节点属性 + mid OM 时间戳"""
import onnx
import os
import time

m = onnx.load("/root/quant/r8/yolov8n_s3_quant_v5.onnx")
shown = 0
for n in m.graph.node:
    if n.name.endswith("_requant") and n.op_type == "AscendQuant":
        attrs = [(a.name, a.i if a.type == 2 else a.f) for a in n.attribute]
        print(n.name[:50], attrs)
        shown += 1
        if shown >= 3:
            break

for p in ["/root/quant/r8/yolov8n_s3_quant_v5.onnx",
          "/root/quant/r8/mid/yolov8n_s3_int8_v5_mid.om",
          "/root/quant/r8/yolov8n_s3_int8_v5.om"]:
    st = os.stat(p)
    print(f"{time.strftime('%H:%M:%S', time.localtime(st.st_mtime))}  {st.st_size:>10}  {p}")
