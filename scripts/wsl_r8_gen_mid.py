#!/usr/bin/env python3
"""R8 诊断⑯：生成「中间输出对照版」ONNX——
FP32 与 v2 各把 10 个 stage 点 + 原最终输出 暴露为 graph output，
板端同帧跑两个 OM 逐点 cosine，定位第一个崩坏的层。

映射: FP32 Conv X 的输出 X_output_0  <->  v2 对应 AscendDequant 输出 X_quant_dequant_output
"""
import os
import sys

import onnx
from onnx import helper, TensorProto

FP32 = "/root/quant/yolov8n_fp32.onnx"
V2 = sys.argv[1] if len(sys.argv) > 1 else "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
TAG = sys.argv[2] if len(sys.argv) > 2 else "v2"
OUT_DIR = "/root/quant/r8/mid"

STAGES = [
    "/model.0/conv",
    "/model.2/cv2/conv",
    "/model.4/cv2/conv",
    "/model.6/cv2/conv",
    # s6 -> s8 加密：/model.7 downsample + /model.8 C2f 内部
    "/model.7/conv",
    "/model.8/cv1/conv",
    "/model.8/m.0/cv1/conv",
    "/model.8/m.0/cv2/conv",
    "/model.8/cv2/conv",
    "/model.9/cv2/conv",
    "/model.12/cv2/conv",
    "/model.15/cv2/conv",
    "/model.18/cv2/conv",
    "/model.21/cv2/conv",
]


def set_outputs(m, out_names):
    new_outs = [helper.make_empty_tensor_value_info(n) for n in out_names]
    m.graph.ClearField("output")
    m.graph.output.extend(new_outs)
    return m


import os
os.makedirs(OUT_DIR, exist_ok=True)

# ---- FP32 mid ----
m_fp = onnx.load(FP32)
final_fp = [o.name for o in m_fp.graph.output]
print(f"FP32 final outputs: {final_fp}")
fp_outs = [f"{s}/Conv_output_0" for s in STAGES] + final_fp
set_outputs(m_fp, fp_outs)
onnx.save(m_fp, f"{OUT_DIR}/yolov8n_fp32_mid.onnx")
print(f"fp32_mid outputs ({len(fp_outs)}): {fp_outs}")

# ---- v2 mid ----
m_v2 = onnx.load(V2)
final_v2 = [o.name for o in m_v2.graph.output]
print(f"{TAG} final outputs: {final_v2}")
if TAG == "v5":
    # v5 每层 deq 后插了重量化对，暴露重量化后的输出
    v2_outs = [f"{s}/Conv_quant_dequant_output_redq" for s in STAGES] + final_v2
else:
    v2_outs = [f"{s}/Conv_quant_dequant_output" for s in STAGES] + final_v2
set_outputs(m_v2, v2_outs)
onnx.save(m_v2, f"{OUT_DIR}/yolov8n_s3_quant_{TAG}_mid.onnx")
print(f"{TAG}_mid outputs ({len(v2_outs)}): {v2_outs}")
