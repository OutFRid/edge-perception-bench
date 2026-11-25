#!/usr/bin/env python3
"""R8 · 真·NPU INT8 链打通（CANN 7.0.1 自带 modelslim.onnx.post_training_quant）。

背景（承接 R5/R6）：
  R5/R6 结论：onnxruntime 产 QDQ ONNX 被 ATC 7.0.1 拒绝（No parser for
  QuantizeLinear）；AMCT 包需官网登录人工下载，成为 NPU INT8 上板的阻塞。
  R8 突破点：CANN 7.0.1 的 python 站点包自带 modelslim.onnx.post_training_quant
  （新版 AMCT 的 ONNX PTQ 实现），插入的是 AscendQuant/AscendDequant 自定义算子，
  而非标准 QDQ——ATC 可解析。无需再等 AMCT 包。

策略口径（复用 R6 拍板的 S3）：
  Percentile 标定 / per-channel 权重 / 仅 Conv+MatMul（modelslim label-free
  默认量化 Conv/Gemm/MatMul，与 S3 的算子集一致）；
  标定集与 R5/R6 同源同过滤（前 12 张），letterbox 640 同口径。

产物：/root/quant/r8/yolov8n_s3_quant.onnx
用法（WSL）:
    bash scripts/wsl_r8_run.sh
"""
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/mnt/g/Git_repo/edge-perception-bench/scripts")
from infer_dms_pipeline import LetterboxPreprocess  # noqa

WORK = Path("/root/quant/r8")
WORK.mkdir(parents=True, exist_ok=True)
FP32 = "/root/quant/yolov8n_fp32.onnx"
OUT_QUANT = str(WORK / "yolov8n_s3_quant_v2.onnx")

# ---------- 标定集（与 R5/R6 同源同过滤，前 12 张） ----------
IMGS = []
for g in ["/mnt/g/Git_repo/Image_deal_class/**/*.jpg",
          "/mnt/g/Git_repo/Image_deal_class/**/*.png",
          "/mnt/g/Git_repo/edge-perception-bench/results/image/*.jpg",
          "/mnt/g/Git_repo/edge-perception-bench/results/doc/*.jpg"]:
    IMGS += glob.glob(g, recursive=True)
IMGS = [i for i in IMGS if os.path.isfile(i) and os.path.getsize(i) > 3000]
IMGS = list(dict.fromkeys(sorted(IMGS)))
print(f"image pool: {len(IMGS)}")

pre = LetterboxPreprocess(imgsz=640)
CAL_N = 12
cal_feeds = []
for p in IMGS[:CAL_N]:
    im = cv2.imread(p)
    if im is None:
        continue
    tensor, _, _ = pre(im)
    cal_feeds.append(tensor.astype(np.float32))
print(f"calibration feeds: {len(cal_feeds)} (shape {cal_feeds[0].shape})")

# ---------- modelslim PTQ（S3：Percentile / per-channel / Conv+MatMul） ----------
# onnx>=1.16 已删除 onnx.mapping，而 modelslim(2022) 的 util.py 仍从中导入
# STORAGE_TENSOR_TYPE_TO_FIELD（标量 initializer 取值用）——注入等价 shim 而非降级 onnx。
import sys as _sys
import types as _types
from onnx import TensorProto as _TP


def _storage_tensor_type_to_field():
    m = {
        int(_TP.FLOAT): "float_data",
        int(_TP.UINT8): "int32_data",
        int(_TP.INT8): "int32_data",
        int(_TP.UINT16): "int32_data",
        int(_TP.INT16): "int32_data",
        int(_TP.INT32): "int32_data",
        int(_TP.INT64): "int64_data",
        int(_TP.STRING): "string_data",
        int(_TP.BOOL): "int32_data",
        int(_TP.FLOAT16): "int32_data",
        int(_TP.DOUBLE): "double_data",
        int(_TP.UINT32): "uint64_data",
        int(_TP.UINT64): "uint64_data",
    }
    for _name in ("INT4", "UINT4"):
        if hasattr(_TP, _name):
            m[int(getattr(_TP, _name))] = f"{_name.lower()}_data"
    return m


_mapping = _types.ModuleType("onnx.mapping")
_mapping.STORAGE_TENSOR_TYPE_TO_FIELD = _storage_tensor_type_to_field()
_sys.modules["onnx.mapping"] = _mapping

# ort>=1.17 已从 quantize_static 移除 optimize_model 参数，而 modelslim(2022)
# 硬编码传入——包一层 wrapper 剥掉，必须在 modelslim import 之前生效。
import onnxruntime.quantization as _ortq
_orig_quantize_static = _ortq.quantize_static


def _quantize_static_compat(model_input, model_output, calibration_data_reader, **kw):
    kw.pop("optimize_model", None)
    return _orig_quantize_static(model_input, model_output, calibration_data_reader, **kw)


_ortq.quantize_static = _quantize_static_compat

from modelslim.onnx.post_training_quant import QuantConfig, run_quantize  # noqa

# v2 修正（板端实测发现）：v1 全量量化把 DFL 回归分支的小数值吞成 0/粗网格，
# 框坐标崩溃（w,h→0）→ 检测头（/model.22/*）排除量化，保持 FP32。
import onnx as _onnx_detect
_m_fp = _onnx_detect.load(FP32)
DETECT_HEAD_NODES = [n.name for n in _m_fp.graph.node
                     if n.name.startswith("/model.22/")]
print(f"exclude detect head nodes: {len(DETECT_HEAD_NODES)}")

cfg = QuantConfig(
    quant_mode=1,            # label-free（需要标定数据）
    is_signed_quant=True,    # 激活 Int8
    is_per_channel=True,     # 权重 per-channel（S3）
    calib_data=[[f] for f in cal_feeds],
    calib_method=1,          # 1: percentile（S3）
    input_shape=[1, 3, 640, 640],
    is_dynamic_shape=False,
    exclude_nodes=DETECT_HEAD_NODES,  # 检测头不量化（DFL 回归对 int8 敏感）
    # 跳过 modelslim 内置图优化：其用 onnx 1.23.2 重建的优化模型 IR=14，
    # 超出本机 onnxruntime（max IR 11）加载能力；原始模型 IR=7 可直接量化。
    is_optimize_graph=False,
)
print("start run_quantize (modelslim onnx PTQ, S3 口径) ...")
t0 = time.time()
run_quantize(FP32, OUT_QUANT, cfg)
print(f"quantize done in {time.time() - t0:.1f}s -> {OUT_QUANT}")

q = Path(OUT_QUANT)
print(f"quant onnx size: {q.stat().st_size / 1e6:.2f} MB (fp32: {Path(FP32).stat().st_size / 1e6:.2f} MB)")

# 自检：统计量化前后算子分布
import onnx  # noqa
m_q = onnx.load(str(q))
print(f"quant model ir_version: {m_q.ir_version}, opset: {m_q.opset_import}")
ops = {}
for n in m_q.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("quant model op histogram:")
for k in sorted(ops):
    print(f"  {k:24s} {ops[k]}")
