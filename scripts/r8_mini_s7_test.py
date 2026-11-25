#!/usr/bin/env python3
"""R8 s7 单层复现·板端：mini_s7 OM 跑真实输入，三方对照——
board(板端 int8 单层) vs cpu_ref(精确手算) vs fp32_s7(FP32 OM 同层输出)。
1) board vs ref  ~1.0 -> ATC 单层正确；否则 ATC 有 bug
2) ref  vs fp32 -> int8 量化固有误差
"""
import numpy as np

from infer_dms_pipeline import AclContext, AclExecutor

ctx = AclContext()

x = np.load("/root/epb/results/mini_s7_input.npy")
ref = np.load("/root/epb/results/mini_s7_ref.npy")

m = AclExecutor(ctx, "/root/epb/models/mini_s7.om")
out, t = m.execute([x])
y = out[0].astype(np.float32)
print(f"mini_s7 exec: {t.get('execute', 0):.2f}ms")

def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def stats(a, b):
    d = np.abs(a.ravel() - b.ravel())
    return f"cos={cos(a, b):.6f} max_abs={d.max():.4f} mean_abs={d.mean():.4f}"

print(f"1) board vs cpu_ref : {stats(y, ref)}")

# FP32 OM 同层输出（s7 = 第 5 个中间点 of[4]）
d = np.load("/root/epb/results/r8_mid_feats.npz")
img = d["input"]
m_fp = AclExecutor(ctx, "/root/epb/models/yolov8n_fp32_mid.om")
of, _ = m_fp.execute([img])
fp_s7 = of[4].astype(np.float32)
print(f"2) cpu_ref vs fp32_s7 : {stats(ref, fp_s7)}")
print(f"3) board vs fp32_s7 : {stats(y, fp_s7)}")
