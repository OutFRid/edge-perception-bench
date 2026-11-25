#!/usr/bin/env python3
"""R8 mini 实验：AscendDequant 标量路径触发条件验证
mini_f1: x(int32) + scale=[16] uint64 全同 -> AscendDequant -> fp32
mini_f2: x(int32) + scale=[16] uint64 各异 -> AscendDequant -> fp32
mini_f3: x(int32) -> Cast(fp32) -> Mul(fp32 标量) -> fp32   (方案 B)
产物: /root/quant/r8/mini/f{1,2,3}.onnx
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

OUT = "/root/quant/r8/mini"


def f32_bits(v):
    return int(np.float32(v).view(np.uint32))


import os
os.makedirs(OUT, exist_ok=True)
OPSET = [helper.make_opsetid("", 11)]


def save_model(g, name):
    onnx.save(helper.make_model(g, opset_imports=OPSET), f"{OUT}/{name}")
    print(f"{name} written")


def make_f1():
    x = helper.make_tensor_value_info("x", TensorProto.INT32, [1, 16, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16, 8, 8])
    scale = numpy_helper.from_array(
        np.full(16, f32_bits(0.003), dtype=np.uint64), "s1")
    n = helper.make_node("AscendDequant", ["x", "s1"], ["y"], name="dq", domain="")
    g = helper.make_graph([n], "mini_f1", [x], [y], [scale])
    save_model(g, "mini_f1.onnx")


def make_f2():
    x = helper.make_tensor_value_info("x", TensorProto.INT32, [1, 16, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16, 8, 8])
    vals = [f32_bits(0.003 * (1 + i * 0.01)) for i in range(16)]
    scale = numpy_helper.from_array(np.array(vals, dtype=np.uint64), "s2")
    n = helper.make_node("AscendDequant", ["x", "s2"], ["y"], name="dq", domain="")
    g = helper.make_graph([n], "mini_f2", [x], [y], [scale])
    save_model(g, "mini_f2.onnx")


def make_f3():
    x = helper.make_tensor_value_info("x", TensorProto.INT32, [1, 16, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16, 8, 8])
    s = numpy_helper.from_array(np.array([0.003], dtype=np.float32), "s3")
    n1 = helper.make_node("Cast", ["x"], ["xf"], name="cast", to=TensorProto.FLOAT)
    n2 = helper.make_node("Mul", ["xf", "s3"], ["y"], name="mul")
    g = helper.make_graph([n1, n2], "mini_f3", [x], [y], [s])
    save_model(g, "mini_f3.onnx")


def make_f4():
    """完整 requant 链：fp32 -> AscendQuant(1/ys, int8) -> Cast(int8->fp32) -> Mul(ys) -> fp32"""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16, 8, 8])
    s = numpy_helper.from_array(np.array([0.003], dtype=np.float32), "s4")
    n1 = helper.make_node("AscendQuant", ["x"], ["xq"], name="q", domain="",
                          scale=1.0 / 0.003, offset=0.0, quant_bit=8)
    n2 = helper.make_node("Cast", ["xq"], ["xqf"], name="cast", to=TensorProto.FLOAT)
    n3 = helper.make_node("Mul", ["xqf", "s4"], ["y"], name="mul")
    g = helper.make_graph([n1, n2, n3], "mini_f4", [x], [y], [s])
    save_model(g, "mini_f4.onnx")


def make_f5():
    """纯 ONNX 备选链：fp32 -> Mul(1/ys) -> Round -> Mul(ys) -> fp32"""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16, 8, 8])
    r1 = numpy_helper.from_array(np.array([1.0 / 0.003], dtype=np.float32), "r5")
    r2 = numpy_helper.from_array(np.array([0.003], dtype=np.float32), "s5")
    n1 = helper.make_node("Mul", ["x", "r5"], ["xt"], name="mul1")
    n2 = helper.make_node("Round", ["xt"], ["xtr"], name="round")
    n3 = helper.make_node("Mul", ["xtr", "s5"], ["y"], name="mul2")
    g = helper.make_graph([n1, n2, n3], "mini_f5", [x], [y], [r1, r2])
    save_model(g, "mini_f5.onnx")


make_f1()
make_f2()
make_f3()
make_f4()
make_f5()
