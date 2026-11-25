#!/usr/bin/env python3
"""R8 诊断②(v2)：把 v2 量化 ONNX 的 AscendQuant/AscendDequant 还原为标准 QDQ，
CPU 推理对照原始 QDQ 模型——二分定位「图转换」还是「ATC 执行」的问题。
（scale 从 Constant 节点/initializer 取；QDQ 的 scale/zp 以 initializer 输入形式给）"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto

sys.path.insert(0, "/mnt/g/Git_repo/edge-perception-bench/scripts")
import glob
import cv2
import onnxruntime as ort
from infer_dms_pipeline import LetterboxPreprocess

M = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
QDQ_REF = "/root/quant/r8/yolov8n_s3_quant_v2_cpu_quant.onnx"
FP32 = "/root/quant/yolov8n_fp32.onnx"
REST = "/root/quant/r8/yolov8n_s3_quant_v2_restored.onnx"

m = onnx.load(M)
g = m.graph

# 张量名 -> 标量值（initializer + Constant 节点）
scalars = {}
for i in g.initializer:
    a = numpy_helper.to_array(i)
    if a.size == 1:
        scalars[i.name] = float(a.ravel()[0])
for n in g.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value" and a.t.raw_data:
                arr = numpy_helper.to_array(a.t)
                if arr.size == 1:
                    scalars[n.output[0]] = float(arr.ravel()[0])

def add_scalar_init(name, value, dtype):
    arr = np.array([value], dtype=dtype)
    t = numpy_helper.from_array(arr, name)
    g.initializer.append(t)
    return name

new_nodes = []
for n in g.node:
    if n.op_type == "AscendQuant":
        scale = offset = None
        for a in n.attribute:
            if a.name == "scale":
                scale = a.f
            if a.name == "offset":
                offset = a.f
        for i in n.input:
            if i in scalars:
                if scale is None:
                    scale = scalars[i]
                elif offset is None:
                    offset = scalars[i]
        sname = add_scalar_init(n.name + "_scale", scale, np.float32)
        zname = add_scalar_init(n.name + "_zp", offset or 0.0, np.int8)
        nn = helper.make_node("QuantizeLinear", [n.input[0], sname, zname], n.output,
                              name=n.name.replace("_quant_quant", "_qlinear"))
        new_nodes.append(nn)
    elif n.op_type == "AscendDequant":
        scale = offset = None
        for i in n.input[1:]:
            if i in scalars:
                if scale is None:
                    scale = scalars[i]
                elif offset is None:
                    offset = scalars[i]
        sname = add_scalar_init(n.name + "_scale", scale, np.float32)
        zname = add_scalar_init(n.name + "_zp", offset or 0.0, np.int8)
        nn = helper.make_node("DequantizeLinear", [n.input[0], sname, zname], n.output,
                              name=n.name.replace("_quant_dequant", "_dqlinear"))
        new_nodes.append(nn)
    else:
        new_nodes.append(n)

g.ClearField("node")
g.node.extend(new_nodes)
onnx.save(m, REST)
print("saved", REST)
print("sample scales:", {k: v for k, v in list(scalars.items())[:3]})

# ---------- CPU 对照 ----------
imgs = [i for i in glob.glob("/mnt/g/Git_repo/Image_deal_class/**/*.jpg", recursive=True)
        if __import__("os").path.getsize(i) > 3000][:1]
img = cv2.imread(imgs[0])
pre = LetterboxPreprocess(imgsz=640)
t, _, _ = pre(img)
t = t.astype(np.float32)

opt = ort.SessionOptions()
opt.log_verbosity_level = 0
s_fp = ort.InferenceSession(FP32, opt, providers=["CPUExecutionProvider"])
s_q = ort.InferenceSession(QDQ_REF, opt, providers=["CPUExecutionProvider"])
s_r = ort.InferenceSession(REST, opt, providers=["CPUExecutionProvider"])
iname = s_fp.get_inputs()[0].name
o_fp = s_fp.run(None, {iname: t})[0].squeeze().transpose()
o_q = s_q.run(None, {iname: t})[0].squeeze().transpose()
o_r = s_r.run(None, {iname: t})[0].squeeze().transpose()

def cos(a, b):
    return float(np.dot(a.ravel(), b.ravel()) /
                 (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12))

print(f"\ncosine(fp32, qdq_ref)      = {cos(o_fp, o_q):.5f}")
print(f"cosine(fp32, restored)     = {cos(o_fp, o_r):.5f}")
print(f"cosine(qdq_ref, restored)  = {cos(o_q, o_r):.5f}")

print("\n--- restored top8 ---")
for idx in (-o_r[:, 4:].max(1)).argsort()[:8]:
    cls = int(o_r[idx, 4:].argmax())
    print(f"  idx={idx:5d} cls={cls} conf={o_r[idx, 4 + cls]:.4f} xywh={[round(v, 2) for v in o_r[idx, :4]]}")
