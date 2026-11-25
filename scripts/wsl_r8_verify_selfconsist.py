#!/usr/bin/env python3
"""R8 诊断⑪：CPU 端自洽性验证——modelslim 输出（v2）的
AscendQuant scale 属性 × AscendDequant deq_scale（按 float32 位模式解读）
是否等于 QDQ 参考模型的 w_scale（weight scale）。

若等式成立：modelslim 输出格式自洽（位模式打包正确），问题在 ATC/vdeq_cast 执行端。
若不成立：modelslim 转换本身有 bug。
"""
import numpy as np
import onnx
from onnx import numpy_helper

V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
REF = "/root/quant/r8/yolov8n_s3_quant_v2_cpu_quant.onnx"

# QDQ 参考：QLinearConv 名 -> w_scale（weight 的 scale，per-channel 向量）
m_ref = onnx.load(REF)
g_ref = m_ref.graph
ref_inits = {i.name: numpy_helper.to_array(i) for i in g_ref.initializer}
qwscale = {}
for n in g_ref.node:
    if n.op_type == "QLinearConv":
        qwscale[n.name] = ref_inits.get(n.input[4])  # QLinearConv 输入[4] = w_scale
print(f"QDQ QLinearConv count: {len(qwscale)}")

# v2：Conv 名 -> (quant_scale_attr, deq_scale_float32)
m = onnx.load(V2)
g = m.graph
init_by_name = {i.name: i for i in g.initializer}
quant_scale = {}
for n in g.node:
    if n.op_type == "AscendQuant":
        for a in n.attribute:
            if a.name == "scale":
                conv_name = n.name.rsplit("_quant", 1)[0]
                quant_scale[conv_name] = a.f
deq_scale = {}
for n in g.node:
    if n.op_type == "AscendDequant":
        t = init_by_name.get(n.input[1])
        if t is not None and t.raw_data:
            # uint64 元素 = 高32位0 + 低32位 float32 位模式；取偶数索引（低32位）做位模式重解释
            u32 = np.frombuffer(t.raw_data, dtype=np.uint32)
            f32 = u32[0::2].view(np.float32)
            conv_name = n.name.rsplit("_dequant", 1)[0]
            deq_scale[conv_name] = f32

print(f"v2 quant_scale: {len(quant_scale)}  deq_scale: {len(deq_scale)}")

# 校验：x_scale = 1/quant_attr；w_scale_est = deq_scale / x_scale = deq_scale * quant_attr
# 与 QDQ w_scale 对照
print(f"{'conv':<40} {'w_scale(QDQ)[:3]':<28} {'deq*qattr[:3]':<28} {'ratio':<10}")
ok = bad = 0
for conv_name, w in sorted(qwscale.items()):
    dq = deq_scale.get(conv_name)
    qa = quant_scale.get(conv_name)
    if dq is None or qa is None or w is None:
        bad += 1
        print(f"{conv_name:<40} MISSING (dq={dq is not None}, qa={qa is not None}, w={w is not None})")
        continue
    est = dq.astype(np.float64) * qa
    w64 = w.ravel().astype(np.float64)
    if len(est) == len(w64):
        ratio = np.mean(est / np.maximum(w64, 1e-30))
    else:
        ratio = float("nan")
    match = np.allclose(est, w64, rtol=1e-2) if len(est) == len(w64) else False
    ok += match
    bad += not match
    print(f"{conv_name:<40} {np.array2string(w64[:3], precision=5):<28} {np.array2string(est[:3], precision=5):<28} {ratio:<10.4f}")

print(f"\nRESULT: match={ok} mismatch={bad}")
