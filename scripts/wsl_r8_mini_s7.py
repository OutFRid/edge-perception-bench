#!/usr/bin/env python3
"""R8 诊断⑱：s7 单层复现——从 v2 提取 /model.7/conv 真实参数，
生成 mini_s7 ONNX（q→conv→deq），CPU 精确手算参考输出。

产物:
  /root/quant/r8/mini/mini_s7.onnx
  /root/quant/r8/mini/mini_s7_input.npy   (SiLU(FP32 s6) 输入)
  /root/quant/r8/mini/mini_s7_ref.npy     (CPU 精确参考)
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

V2 = "/root/quant/r8/yolov8n_s3_quant_v2.onnx"
NPZ = "/mnt/g/Git_repo/edge-perception-bench/results/r8_mid_feats.npz"
OUT = "/root/quant/r8/mini"

m = onnx.load(V2)
g = m.graph
inits = {i.name: i for i in g.initializer}

CONV = "/model.7/conv/Conv_quant"
qname = CONV + "_quant"
dqname = CONV + "_dequant"

# 1) 提取参数
quant_scale = None
for n in g.node:
    if n.name == qname:
        for a in n.attribute:
            if a.name == "scale":
                quant_scale = a.f
assert quant_scale is not None, "quant scale not found"

w = numpy_helper.to_array(inits[CONV.replace("/model.7/conv/", "") + "weight"]) if False else None
# 直接从 Conv 输入提取
conv_node = next(n for n in g.node if n.name == CONV)
W = numpy_helper.to_array(inits[conv_node.input[1]]).astype(np.int32)   # [256,128,3,3]
B = numpy_helper.to_array(inits[conv_node.input[2]]).astype(np.int32)   # [256]
dq = next(n for n in g.node if n.name == dqname)
S = inits[dq.input[1]]
u32 = np.frombuffer(S.raw_data, dtype=np.uint32)
deq_scale = u32[0::2].view(np.float32)                                  # [256]

print(f"quant_scale={quant_scale}  W{W.shape}  B{B.shape}  deq_scale{deq_scale.shape}")

# 2) 输入：SiLU(FP32 s6 输出)
d = np.load(NPZ)
s6 = d["s6"]                                  # [1,128,40,40]
x_in = s6 * (1.0 / (1.0 + np.exp(-s6)))       # SiLU
x_in = x_in.astype(np.float32)
np.save(f"{OUT}/mini_s7_input.npy", x_in)

# 3) CPU 精确参考
x_q = np.round(x_in * quant_scale).clip(-128, 127).astype(np.int32)     # [1,128,40,40]
xp = np.pad(x_q[0], ((0, 0), (1, 1), (1, 1)))                            # pad=1
H, Wd = 40, 40
oh = ow = 20
patches = np.zeros((oh * ow, 128 * 9), dtype=np.int32)
for i in range(oh):
    for j in range(ow):
        win = xp[:, i * 2:i * 2 + 3, j * 2:j * 2 + 3].reshape(-1)       # [128*9]
        patches[i * ow + j] = win
y_acc = patches.astype(np.int64) @ W.reshape(256, -1).T.astype(np.int64)  # [400,256]
y_acc = (y_acc + B[None, :]).astype(np.int32)
y_ref = (y_acc * deq_scale[None, :].astype(np.float64)).astype(np.float32)
y_ref = y_ref.T.reshape(1, 256, oh, ow)
np.save(f"{OUT}/mini_s7_ref.npy", y_ref)
print(f"y_ref {y_ref.shape} range [{y_ref.min():.3f}, {y_ref.max():.3f}]")

# 4) 生成 ONNX
nodes = []
nodes.append(helper.make_node(
    "AscendQuant", ["x"], ["xq"], name=qname, domain="",
    scale=float(quant_scale), offset=0.0, quant_bit=8))
nodes.append(helper.make_node(
    "Conv", ["xq", conv_node.input[1], conv_node.input[2]], ["yacc"], name=CONV,
    kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]))
nodes.append(helper.make_node(
    "AscendDequant", ["yacc", dq.input[1]], ["y"], name=dqname, domain=""))

graph = helper.make_graph(
    nodes, "mini_s7",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 128, 40, 40])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 256, 20, 20])],
    [inits[conv_node.input[1]], inits[conv_node.input[2]], inits[dq.input[1]]])
mm = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
mm.ir_version = 7
onnx.save(mm, f"{OUT}/mini_s7.onnx")
print("gen mini_s7.onnx")
