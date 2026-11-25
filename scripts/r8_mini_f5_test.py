#!/usr/bin/env python3
"""R8 板端：mini_f5（Mul-Round-Mul 重量化链）执行精度实测
板端 OM 输出 vs CPU 手算 round(x/0.003)*0.003
若偏差 ~1e-3 量级 -> 310B 上 fp32 逐元素算子以 fp16 执行（假设证实）
"""
import numpy as np
from infer_dms_pipeline import AclContext, AclExecutor

ctx = AclContext()
m = AclExecutor(ctx, "/root/epb/models/mini_f5.om")

rng = np.random.default_rng(42)
x = (rng.standard_normal((1, 16, 8, 8)) * 3.0).astype(np.float32)

out, t = m.execute([x])
y = out[0].astype(np.float32)

ref = np.round(x / 0.003) * 0.003

d = np.abs(y - ref)
print(f"exec: {t.get('execute', 0):.2f}ms")
print(f"x range: [{x.min():.3f}, {x.max():.3f}]")
print(f"max_abs={d.max():.6f}  mean_abs={d.mean():.8f}")
cos = np.dot(y.ravel(), ref.ravel()) / (np.linalg.norm(y) * np.linalg.norm(ref))
print(f"cos={cos:.8f}")
# fp16 特征检测：误差是否与 |x| 成正比（fp16 舍入特征）
rel = (d / (np.abs(ref) + 1e-6)).max()
print(f"max_rel={rel:.6f}")
