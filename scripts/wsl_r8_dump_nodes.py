#!/usr/bin/env python3
"""R8 诊断⑭：对照 FP32 与 v2 图的节点命名（找中间输出 tensor 名，供 atc --out_nodes 用）。"""
import onnx

FP32 = "/root/quant/yolov8n_fp32.onnx"
V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"

for path, tag in [(FP32, "FP32"), (V2, "v2")]:
    m = onnx.load(path)
    print(f"=== {tag} ({len(m.graph.node)} nodes) ===")
    for n in m.graph.node:
        if n.op_type in ("Conv", "MaxPool", "Concat", "Upsample", "Resize", "Split"):
            print(f"  {n.op_type:<8} {n.name:<42} out={n.output[0]}")
    print()
