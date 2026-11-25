#!/usr/bin/env python3
"""R8 诊断：检查 v2 量化 ONNX 的 DFL 子图结构，定位 w/h→0 退化的量化节点。"""
import onnx
import sys

M = sys.argv[1] if len(sys.argv) > 1 else "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
m = onnx.load(M)
g = m.graph

# 张量 -> 生产者
prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n

print("=== 节点: 名字含 dfl / model.22 且与量化相关 ===")
for n in g.node:
    if "dfl" in n.name.lower() or "model.22" in n.name:
        tag = ""
        if n.op_type in ("AscendQuant", "AscendDequant"):
            tag = " <<<< QUANT"
        print(f"  {n.op_type:16s} {n.name}")
        if tag:
            print(f"    in: {n.input}  out: {n.output}{tag}")

print("")
print("=== Softmax 输入链 ===")
for n in g.node:
    if n.op_type == "Softmax":
        print(f"Softmax node: {n.name}  in={list(n.input)}  out={list(n.output)}")
        src = prod.get(n.input[0])
        if src is not None:
            print(f"   producer: {src.op_type} {src.name}")
            if src.op_type in ("AscendDequant", "AscendQuant"):
                print(f"     quant in={list(src.input)} out={list(src.output)}")

print("")
print("=== Mul / Sum 是否吃 quant 输出（DFL 解码链） ===")
for n in g.node:
    if n.op_type in ("Mul", "Sum") and any("dfl" in i or "model.22" in i for i in n.input):
        srcs = [prod.get(i) for i in n.input]
        print(f"  {n.op_type:5s} {n.name}")
        for i, s in zip(n.input, srcs):
            t = f"{s.op_type} {s.name}" if s is not None else "<initializer/input>"
            print(f"     in: {i}  <- {t}")
