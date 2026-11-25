#!/usr/bin/env python3
"""R8 决定性实验：两层链 CPU 精确模拟，对比 v2 语义 vs QDQ 语义的误差传播
链: x=SiLU(s6) -> s7(conv s2p1) -> SiLU -> s8_cv1(conv 1x1)
分支 A (v2 语义):   输出无重量化（误差自由传播）
分支 B (QDQ 语义):  每层输出重量化 y_q=round(y_acc*deq/sy) clip(-127,127), y=y_q*sy
参考: FP32 权重 numpy 精确 conv
结论判据: 若 B 的 s8_cv1 输出 cosine 显著高于 A -> 修复方向正确（板端实现问题）
         若 B≈A -> 诊断错误，误差另有根源
"""
import numpy as np
import onnx
from onnx import numpy_helper

V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
QDQ = "/root/quant/r8/yolov8n_s3_quant_v2_cpu_quant.onnx"
FP32 = "/root/quant/yolov8n_fp32.onnx"
NPZ = "/mnt/g/Git_repo/edge-perception-bench/results/r8_mid_feats.npz"

CONV7 = "/model.7/conv/Conv_quant"
CONV8 = "/model.8/cv1/conv/Conv_quant"


def silu(x):
    x = x.astype(np.float64)
    return x / (1.0 + np.exp(-x))


def conv_int8(xq, W, B, stride, pad):
    """xq [1,C,H,W] int32, W [O,C,k,k] int32 -> [1,O,OH,OW] int64"""
    xp = np.pad(xq[0], ((0, 0), (pad, pad), (pad, pad)))
    C, H, Wd = xp.shape
    k = W.shape[2]
    OH = (H - k) // stride + 1
    OW = (Wd - k) // stride + 1
    O = W.shape[0]
    patches = np.zeros((OH * OW, C * k * k), dtype=np.int64)
    for i in range(OH):
        for j in range(OW):
            patches[i * OW + j] = xp[:, i * stride:i * stride + k,
                                     j * stride:j * stride + k].reshape(-1)
    y = patches @ W.reshape(O, -1).T.astype(np.int64)
    y = y + B[None, :]
    return y.T.reshape(1, O, OH, OW)


def conv_fp32(x, W, B, stride, pad):
    """fp32 numpy conv (im2col)"""
    xp = np.pad(x[0], ((0, 0), (pad, pad), (pad, pad)))
    C, H, Wd = xp.shape
    k = W.shape[2]
    OH = (H - k) // stride + 1
    OW = (Wd - k) // stride + 1
    O = W.shape[0]
    patches = np.zeros((OH * OW, C * k * k), dtype=np.float64)
    for i in range(OH):
        for j in range(OW):
            patches[i * OW + j] = xp[:, i * stride:i * stride + k,
                                     j * stride:j * stride + k].reshape(-1)
    y = patches @ W.reshape(O, -1).T.astype(np.float64)
    y = y + B[None, :]
    return y.T.reshape(1, O, OH, OW)


def cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def get_params(m_path, conv, fp32=False):
    m = onnx.load(m_path)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    cn = conv if fp32 else conv  # 同名
    node = next(n for n in g.node if n.name == cn)
    W = numpy_helper.to_array(inits[node.input[1]])
    B = numpy_helper.to_array(inits[node.input[2]])
    if not fp32:
        W = W.astype(np.int32)
        B = B.astype(np.int32)
    return W, B


def get_scales(m_path, conv):
    """返回 (quant_scale, deq_scale[C], y_scale)"""
    m = onnx.load(m_path)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    quant_scale = None
    dq = None
    for n in g.node:
        if n.name == conv + "_quant":
            for a in n.attribute:
                if a.name == "scale":
                    quant_scale = a.f
        if n.name == conv + "_dequant":
            dq = n
    u32 = np.frombuffer(inits[dq.input[1]].raw_data, dtype=np.uint32)
    deq_scale = u32[0::2].view(np.float32)
    return quant_scale, deq_scale


def get_y_scale(m_path, conv):
    m = onnx.load(m_path)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    for n in g.node:
        if n.op_type == "QLinearConv" and n.name == conv:
            a = numpy_helper.to_array(inits[n.input[6]]).ravel()
            return float(a[0])
    return None


# ============ 参数提取 ============
qs7, deq7 = get_scales(V2, CONV7)
qs8, deq8 = get_scales(V2, CONV8)
sy7 = get_y_scale(QDQ, CONV7)
sy8 = get_y_scale(QDQ, CONV8)
W7, B7 = get_params(V2, CONV7)
W8, B8 = get_params(V2, CONV8)
FW7, FB7 = get_params(FP32, "/model.7/conv/Conv", fp32=True)
FW8, FB8 = get_params(FP32, "/model.8/cv1/conv/Conv", fp32=True)
print(f"qs7={qs7:.4f} qs8={qs8:.4f} sy7={sy7:.5f} sy8={sy8:.5f}")
print(f"W7{W7.shape} W8{W8.shape} FW7{FW7.shape} FW8{FW8.shape}")

d = np.load(NPZ)
s6 = d["s6"]
x = silu(s6).astype(np.float32)          # SiLU(s6_fp32)

# ============ FP32 参考 ============
y7f = conv_fp32(x, FW7, FB7, 2, 1)
a7f = silu(y7f)
y8f = conv_fp32(a7f.astype(np.float32), FW8, FB8, 1, 0)
print(f"\nFP32 ref: y7f range [{y7f.min():.2f},{y7f.max():.2f}] y8f range [{y8f.min():.2f},{y8f.max():.2f}]")

# ============ 公共前半：s7 量化+卷积 ============
z7q = np.round(x * qs7).clip(-128, 127).astype(np.int32)
y7acc = conv_int8(z7q, W7, B7, 2, 1)

# 分支 A (v2): y7 无重量化
y7a = (y7acc * deq7[None, :, None, None].astype(np.float64)).astype(np.float32)

# 分支 B (QDQ): y7 重量化
y7qb = np.round(y7acc * deq7[None, :, None, None] / sy7).clip(-127, 127)
y7b = (y7qb * sy7).astype(np.float32)

print(f"\ncos(A_y7, fp32)={cos(y7a, y7f):.6f}   cos(B_y7, fp32)={cos(y7b, y7f):.6f}")

# 分支 A 继续
a7a = silu(y7a)
z8qa = np.round(a7a * qs8).clip(-128, 127).astype(np.int32)
y8a = conv_int8(z8qa, W8, B8, 1, 0) * deq8[None, :, None, None].astype(np.float64)

# 分支 B 继续
a7b = silu(y7b)
z8qb = np.round(a7b * qs8).clip(-128, 127).astype(np.int32)
y8b = conv_int8(z8qb, W8, B8, 1, 0) * deq8[None, :, None, None].astype(np.float64)

print(f"\n===== 决定性对比（s8_cv1 输出 vs FP32）=====")
print(f"A (v2 语义, 无输出重量化): cosine={cos(y8a, y8f):.6f}")
print(f"B (QDQ 语义, 有输出重量化): cosine={cos(y8b, y8f):.6f}")
