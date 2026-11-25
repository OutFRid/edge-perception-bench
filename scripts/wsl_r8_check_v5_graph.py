#!/usr/bin/env python3
"""R8 检查 v5 ONNX 图完整性：graph output 引用、悬空输出、连接链"""
import onnx
from collections import Counter

M = onnx.load("/root/quant/r8/yolov8n_s3_quant_v5.onnx")
g = M.graph

print("nodes:", len(g.node))
print(Counter(n.op_type for n in g.node))
print("initializers:", len(g.initializer))

# 1) graph output 是否有生产者
prod = set()
for n in g.node:
    for o in n.output:
        prod.add(o)
print("\n--- graph outputs ---")
for o in g.output:
    ok = "PRODUCED" if o.name in prod else "ORPHAN!!"
    print(f"{ok}  {o.name}")

# 2) 各 deq 输出被谁消费
used = Counter()
for n in g.node:
    for i in n.input:
        used[i] += 1
print("\n--- AscendDequant outputs with consumers ---")
zero = []
for n in g.node:
    if n.op_type == "AscendDequant":
        o = n.output[0]
        if used[o] == 0:
            zero.append((n.name, o))
        elif used[o] > 3:
            print(f"MANY({used[o]}) {n.name} -> {o}")
print(f"unconsumed AscendDequant outputs: {len(zero)}")

# 2b) 自环检测：requant(AscendQuant) 的 input 不能是自己的下游 _redq
print("\n--- self-loop check (requant input == redq output?) ---")
loops = 0
for n in g.node:
    if n.op_type == "AscendQuant" and n.name.endswith("_requant"):
        if n.input[0].endswith("_redq"):
            loops += 1
            if loops <= 3:
                print(f"  SELF-LOOP: {n.name} input={n.input[0]}")
print(f"self-loops: {loops}")
for nm, o in zero:
    print("  ZERO:", nm, "->", o)

# 3) 最后 10 个节点
print("\n--- last 10 nodes ---")
for n in g.node[-10:]:
    print(f"{n.op_type:16s} {n.name:60s} out={n.output}")

# 4) 检查 graph output 名与最后一层 deq/redq 的关系
print("\n--- tail dequant/requant chain ---")
for n in g.node:
    if "model.22" in n.name and (n.op_type in ("AscendQuant", "AscendDequant")):
        print(f"{n.op_type:16s} {n.name:70s} in={n.input} out={n.output}")
