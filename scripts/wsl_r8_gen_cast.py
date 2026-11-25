#!/usr/bin/env python3
"""R8 修复 v4：v2 每个 AscendDequant 输出后插 Cast(FLOAT16->FLOAT)，
把 SiLU/Add/Concat/Split 等非 Conv 算子全部拉回 fp32 域，
验证「fp16 域中间特征/SiLU 实现精度」是否为精度崩坏根因。

产物: /root/quant/r8/yolov8n_s3_quant_v4.onnx
"""
import onnx
from onnx import helper, TensorProto

V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
V4 = "/root/quant/r8/yolov8n_s3_quant_v4.onnx"

m = onnx.load(V2)
g = m.graph

deq_outs = []
for n in g.node:
    if n.op_type == "AscendDequant":
        deq_outs.append(n.output[0])

# 插入 Cast 节点并重定向消费者
nodes = list(g.node)
added = 0
for n in nodes:
    changed = []
    for idx, inp in enumerate(n.input):
        if inp in deq_outs and n.op_type != "Cast":
            changed.append(idx)
    if not changed:
        continue
    for idx in changed:
        old = n.input[idx]
        new = old + "_to_f32"
        # 若该 tensor 的 Cast 尚未插入
        if not any(x.op_type == "Cast" and x.output[0] == new for x in g.node):
            cast = helper.make_node("Cast", [old], [new], name=new, to=TensorProto.FLOAT)
            g.node.append(cast)
            added += 1
        n.input[idx] = new

onnx.save(m, V4)
print(f"inserted {added} Cast nodes -> {V4}")

# 自检：所有 SiLU/Add/Concat/Split/MaxPool/Resize 的输入 dtype 上游是否 fp32
m2 = onnx.load(V4)
cast_in = {n.input[0]: n for n in m2.graph.node if n.op_type == "Cast"}
fp_ops = ("Sigmoid", "Mul", "Add", "Concat", "Split", "MaxPool", "Resize", "Conv")
n_fp_in = 0
for n in m2.graph.node:
    if n.op_type in fp_ops:
        for i in n.input:
            if i in cast_in:
                n_fp_in += 1
print(f"fp ops fed by Cast: {n_fp_in}")
