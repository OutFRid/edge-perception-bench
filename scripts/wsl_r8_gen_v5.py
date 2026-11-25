#!/usr/bin/env python3
"""R8 修复 v5：模拟 QDQ 的「输出重量化」（误差重置）——
v2 每个 AscendDequant 输出后插
  AscendQuant(1/y_scale, int8) -> Cast(int8→fp32) -> Mul(y_scale)
等价于 QDQ DequantizeLinear 语义 y_reset = round(y/y_scale)*y_scale，
阻止量化误差跨层累积。

为什么不用 Mul(1/ys)->Round->Mul(ys) 纯 fp32 链：
板端实测（mini_f5）310B 上 Mul/Round 以 fp16 执行，t=y_fp/sy 数值达
数百~数千，fp16 ulp 0.5-1 格，round 前即被污染（max_abs≈4 格）。
而 y_q 是 int8 小整数（|y_q|≤127），Cast 无损、Mul(y_q*sy) 的 fp16
ulp 仅 ~0.12 格，格子化截断依然有效。

y_scale 取自 QDQ 参考模型（QLinearConv 的 y_scale input[6]）。

产物: /root/quant/r8/yolov8n_s3_quant_v5.onnx
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

REF = "/root/quant/r8/yolov8n_s3_quant_v2_cpu_quant.onnx"
V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
V5 = "/root/quant/r8/yolov8n_s3_quant_v5.onnx"


def f32_bits(v):
    return int(np.float32(v).view(np.uint32))


# 1) QDQ 参考：QLinearConv 名 -> y_scale（标量）
m_ref = onnx.load(REF)
g_ref = m_ref.graph
ref_inits = {i.name: i for i in g_ref.initializer}
conv_y_scale = {}
for n in g_ref.node:
    if n.op_type == "QLinearConv":
        t = ref_inits.get(n.input[6])
        if t is not None:
            a = numpy_helper.to_array(t).ravel()
            conv_y_scale[n.name] = float(a[0])
print(f"QDQ y_scale count: {len(conv_y_scale)}")

# 2) v2：每个 AscendDequant 输出后插 重量化链
m = onnx.load(V2)
g = m.graph
deq_nodes = [n for n in g.node if n.op_type == "AscendDequant"]
print(f"AscendDequant count: {len(deq_nodes)}")

added = 0
for n in deq_nodes:
    conv_name = n.name[:-len("_dequant")]
    ys = conv_y_scale.get(conv_name)
    if ys is None:
        print(f"MISSING y_scale for {conv_name}")
        continue
    out = n.output[0]
    qf_out = out + "_req_f32"
    d_out = out + "_redq"
    r2_name = conv_name + "_requant_scale"

    # 先重定向消费者（此时 g.node 中尚无本层新节点，
    # 否则新插入节点的 input 会被改成自己的下游输出形成自环）
    for c in g.node:
        if c.name == n.name:
            continue
        for idx, inp in enumerate(c.input):
            if inp == out:
                c.input[idx] = d_out

    # 再插入重量化链 AscendQuant(1/ys, dst_type=1=输出fp32) -> Mul(ys)
    # dst_type=1 直接输出 fp32 格子化值，避开 int8 张量的 trans_data
    # 布局转换路径（310B 上该路径崩溃/结果错乱，mini_f4 已验证）。
    g.node.append(helper.make_node(
        "AscendQuant", [out], [qf_out], name=conv_name + "_requant", domain="",
        scale=1.0 / ys, offset=0.0, dst_type=1))
    g.initializer.append(numpy_helper.from_array(
        np.array([ys], dtype=np.float32), r2_name))
    g.node.append(helper.make_node(
        "Mul", [qf_out, r2_name], [d_out], name=conv_name + "_requant_mul"))
    added += 1

onnx.save(m, V5)
print(f"inserted {added} requant chains -> {V5}")

# 自检：节点数、无残留消费
m2 = onnx.load(V5)
q = sum(1 for n in m2.graph.node if n.op_type == "AscendQuant" and n.name.endswith("_requant"))
mul = sum(1 for n in m2.graph.node if n.op_type == "Mul" and n.name.endswith("_requant_mul"))
print(f"self-check: requant AscendQuant={q} Mul={mul} (expect {added} each)")
