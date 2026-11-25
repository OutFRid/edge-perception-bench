#!/usr/bin/env python3
"""R8 诊断⑫：受控最小实验——生成 3 个单层 Quant→Conv(int8)→Dequant 模型，
板端跑后即可判定 ATC 对 AscendQuant scale 与 AscendDequant uint64 deq_scale 的解析语义。

模型结构: x(1x1x1x1 FP32) -> AscendQuant -> Conv(1x1, w=int8[1], 无bias) -> AscendDequant -> y
输入 x=100 常量语义（由外部喂入）。

变体:
  mini_a: quant scale=2.0, deq 位模式 1.0
          若 quant 乘数语义: xq=200 -> y=200 ; 若除数语义: xq=50 -> y=50
  mini_b: quant scale=1.0, deq 位模式 0.5
          若 deq 按低32位 float32 位模式: y=100*0.5=50 ; 若按 uint64 数值: y=1.06e11(溢出)
  mini_c: quant scale=1.0, deq 位模式 1.0  (恒等对照: 期望 y=100)
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

OUT = "/root/quant/r8/mini"


def f32_bits(v):
    return int(np.float32(v).view(np.uint32))


def make_model(name, quant_scale, deq_value):
    nodes = []
    inits = []

    nodes.append(helper.make_node(
        "AscendQuant", ["x"], ["xq"], name="q", domain="",
        scale=float(quant_scale), offset=0.0, quant_bit=8))
    inits.append(numpy_helper.from_array(
        np.array([[[[1]]]], dtype=np.int8), "W"))
    nodes.append(helper.make_node(
        "Conv", ["xq", "W"], ["yacc"], name="conv",
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0]))
    inits.append(numpy_helper.from_array(
        np.array([f32_bits(deq_value)], dtype=np.uint64), "S"))
    nodes.append(helper.make_node(
        "AscendDequant", ["yacc", "S"], ["y"], name="dq", domain=""))

    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 1, 1, 1])],
        inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 7
    onnx.save(m, f"{OUT}/{name}.onnx")
    print(f"gen {name}.onnx  quant_scale={quant_scale} deq_bits=0x{f32_bits(deq_value):08X}")


def make_bias_model(name):
    """带 int32 bias 的 Conv：x=100, quant scale=2 (x_scale=1/2),
    w_q=4 (等价 w=1, w_scale=1/4), bias_q=24 (等价 bias_fp=3),
    deq 位模式 = x_scale*w_scale = 0.125。
    正确链: xq=200 -> acc=200*4=800 -> +bias_q=824 -> *0.125 = 103
    bias 加 fp 域(错误): 800*0.125+24 = 124 ; bias 忽略: 100
    """
    nodes = []
    inits = []
    nodes.append(helper.make_node(
        "AscendQuant", ["x"], ["xq"], name="q", domain="",
        scale=2.0, offset=0.0, quant_bit=8))
    inits.append(numpy_helper.from_array(
        np.array([[[[4]]]], dtype=np.int8), "W"))
    inits.append(numpy_helper.from_array(
        np.array([24], dtype=np.int32), "B"))
    nodes.append(helper.make_node(
        "Conv", ["xq", "W", "B"], ["yacc"], name="conv",
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0]))
    inits.append(numpy_helper.from_array(
        np.array([f32_bits(0.125)], dtype=np.uint64), "S"))
    nodes.append(helper.make_node(
        "AscendDequant", ["yacc", "S"], ["y"], name="dq", domain=""))

    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 1, 1, 1])],
        inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 7
    onnx.save(m, f"{OUT}/{name}.onnx")
    print(f"gen {name}.onnx  (bias case: expect 103 if bias in int32 domain)")


import os
os.makedirs(OUT, exist_ok=True)
make_model("mini_a", 2.0, 1.0)
make_model("mini_b", 1.0, 0.5)
make_model("mini_c", 1.0, 1.0)
make_bias_model("mini_d")
print("done")
