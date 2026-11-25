#!/usr/bin/env python3
"""R8 受控最小实验·板端：验证 ATC 对 AscendQuant scale / AscendDequant uint64 deq_scale 的解析语义。

期望:
  mini_a: quant scale=2.0, deq 位模式 1.0 -> 乘数语义 y=200 ; 除数语义 y=50
  mini_b: quant scale=1.0, deq 位模式 0.5 -> 位模式解析 y=50 ; uint64 数值解析 y~1e11(溢出)
  mini_c: quant scale=1.0, deq 位模式 1.0 -> 恒等 y=100
"""
import numpy as np
from infer_dms_pipeline import AclContext, AclExecutor

ctx = AclContext(device=0)
for name, note in [
    ("mini_a", "quant s=2: mul->200, div->50"),
    ("mini_b", "deq bits 0.5: bitpattern->50, u64num->overflow"),
    ("mini_c", "identity -> 100"),
    ("mini_d", "bias int32: 103=correct, 124=bias-in-fp-domain, 100=bias-dropped"),
]:
    m = AclExecutor(ctx, f"/root/epb/models/{name}.om")
    out, timing = m.execute([np.array([[[[100.0]]]], dtype=np.float32)])
    v = out[0].reshape(-1).astype(np.float64)
    print(f"{name}: y={v}  | {note}  | exec={timing.get('execute', 0):.3f}ms")
