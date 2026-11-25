#!/usr/bin/env python3
"""R8 诊断⑬：v2 量化图结构 sanity 检查——
每个 int8 Conv 是否都被 AscendQuant/AscendDequant 正确包裹、
deq_scale dims 是否等于 Conv 输出通道、offset/quant_bit 等属性。
"""
import numpy as np
import onnx

V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
m = onnx.load(V2)
g = m.graph

inits = {i.name: i for i in g.initializer}
producers = {}
for n in g.node:
    for o in n.output:
        producers[o] = n
consumers = {}
for n in g.node:
    for i in n.input:
        consumers.setdefault(i, []).append(n)

int8_convs = [n for n in g.node if n.op_type == "Conv"]
print(f"total Conv: {len(int8_convs)}")

bad = 0
for c in int8_convs:
    w = inits.get(c.input[1])
    wt = onnx.TensorProto.DataType.Name(w.data_type) if w is not None else "MISSING"
    # 1) 输入来自 AscendQuant
    src = producers.get(c.input[0])
    src_op = src.op_type if src is not None else "?"
    # 2) 输出被 AscendDequant 消费
    dst = consumers.get(c.output[0])
    dst_op = [d.op_type for d in (dst or [])]
    # 3) deq_scale dims vs C_out
    deq = None
    if dst:
        for d in dst:
            if d.op_type == "AscendDequant":
                deq = inits.get(d.input[1])
    dq_dims = list(deq.dims) if deq is not None else None
    w_shape = list(w.dims) if w is not None else None
    ok = (src_op == "AscendQuant" and "AscendDequant" in dst_op
          and dq_dims is not None and w_shape is not None
          and dq_dims[0] == w_shape[0])
    if not ok:
        bad += 1
        print(f"BAD {c.name}: src={src_op} dst={dst_op} w_dtype={wt} w_shape={w_shape} dq_dims={dq_dims}")

print(f"int8 Conv sanity: {len(int8_convs) - bad}/{len(int8_convs)} OK")

# offset 全 0？quant_bit 全 8？
offsets = {}
bits = {}
for n in g.node:
    if n.op_type == "AscendQuant":
        attrs = {a.name: a for a in n.attribute}
        off = attrs.get("offset")
        qb = attrs.get("quant_bit")
        offsets[off.f if off is not None else -1] = offsets.get(off.f if off is not None else -1, 0) + 1
        bits[qb.i if qb is not None else -1] = bits.get(qb.i if qb is not None else -1, 0) + 1
print(f"offset values: {offsets}")
print(f"quant_bit values: {bits}")

# 权重量级分布（确认 int8 合理范围）
wmin, wmax = [], []
for c in int8_convs:
    w = inits.get(c.input[1])
    if w is not None and w.raw_data:
        a = np.frombuffer(w.raw_data, dtype=np.int8)
        wmin.append(a.min())
        wmax.append(a.max())
print(f"weight int8 range: min={min(wmin)} max={max(wmax)}")
