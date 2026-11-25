#!/usr/bin/env python3
"""R8 修复(v2)：v2 量化 ONNX 的 *_dequant_scale initializer 被 modelslim 误写成
UINT64 类型（dtype=13，raw=128B），ATC 解析不出正确 scale。

修复：用 QDQ 参考模型中对应 DequantizeLinear 的 y_scale（float32 标量）覆盖之。
（AscendQuant 的 scale 属性 = 1/y_scale 乘数语义，板端证据确认执行正确，不动。）
"""
import numpy as np
import onnx
from onnx import numpy_helper

REF = "/root/quant/r8/yolov8n_s3_quant_v2_cpu_quant.onnx"
V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
V3 = "/root/quant/r8/yolov8n_s3_quant_v3.onnx"

# 1) QDQ 参考: QLinearConv 名 -> dequant y_scale
m_ref = onnx.load(REF)
g_ref = m_ref.graph
ref_inits = {i.name: numpy_helper.to_array(i) for i in g_ref.initializer}
producers = {}
for n in g_ref.node:
    for o in n.output:
        producers[o] = n
conv_dq_scale = {}
for n in g_ref.node:
    if n.op_type == "DequantizeLinear":
        src = producers.get(n.input[0])
        if src is not None and src.op_type == "QLinearConv":
            s = ref_inits.get(n.input[1])
            if s is not None:
                conv_dq_scale[src.name] = float(s.ravel()[0])
print(f"QDQ ref dequant scales: {len(conv_dq_scale)}")

# 2) v2: 覆盖 dtype 错误（UINT64）的 dequant_scale initializer
#  ATC 对独立 AscendDequant 的 scale 期望 per-channel 张量（dims=[C] FLOAT），
#  标量（dims=[]）会触发「无 kernel」报错——故按原 dims 广播 y_scale。
m = onnx.load(V2)
g = m.graph
init_by_name = {i.name: i for i in g.initializer}
dims_by_name = {i.name: list(i.dims) for i in g.initializer}
fixed = 0
for n in g.node:
    if n.op_type == "AscendDequant":
        conv_name = n.name.replace("_dequant", "")   # /model.X/.../Conv_quant
        s = conv_dq_scale.get(conv_name)
        if s is None:
            print(f"MISSING scale for {conv_name}")
            continue
        tname = n.input[1]
        dims = dims_by_name.get(tname, [1])
        arr = np.full(dims, s, dtype=np.float32) if len(dims) > 0 and dims[0] > 1 \
            else np.array([s], dtype=np.float32)
        t = numpy_helper.from_array(arr, tname)
        if tname in init_by_name:
            idx = [i.name for i in g.initializer].index(tname)
            g.initializer[idx].CopyFrom(t)
        else:
            g.initializer.append(t)
        fixed += 1

onnx.save(m, V3)
print(f"overwrote {fixed} dequant scale tensors -> {V3}")

# 自检：dtype 应为 FLOAT(1)，dims 与原一致（per-channel），值均为 y_scale
m3 = onnx.load(V3)
for i in m3.graph.initializer:
    if "dequant_scale" in i.name:
        a = numpy_helper.to_array(i)
        assert i.data_type == 1, f"dtype still {i.data_type}: {i.name}"
        assert 0 < a.ravel()[0] < 10, f"value weird: {i.name} {a.ravel()[0]}"
print("self-check OK: all dequant_scale are FLOAT, values in (0, 10)")
