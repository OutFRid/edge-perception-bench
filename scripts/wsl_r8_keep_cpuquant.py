#!/usr/bin/env python3
"""R8 诊断：临时保留 modelslim 中间 QDQ 模型（注释掉删除逻辑），用于 CPU 对照。"""
import re
P = "/root/Ascend/ascend-toolkit/latest/python/site-packages/modelslim/onnx/post_training_quant/label_free/quantize_tool.py"
src = open(P).read()
pat = re.compile(
    r'[ \t]*logger\.debug\("Remove temp cpu quant model file: %s", cpu_quant_model_path\)\n'
    r'[ \t]*if os\.path\.exists\(cpu_quant_model_path\):\n'
    r'[ \t]*os\.remove\(cpu_quant_model_path\)')
new_src, n = pat.subn(
    '    logger.debug("KEEP temp cpu quant model file for R8 diag: %s", cpu_quant_model_path)',
    src)
if n == 0:
    print("PATTERN NOT FOUND, no change")
else:
    open(P, "w").write(new_src)
    print(f"patched {P} ({n} site)")
