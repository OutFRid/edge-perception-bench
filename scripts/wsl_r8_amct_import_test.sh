#!/bin/bash
# R8 · 修复 amct_onnx protobuf 兼容：用纯 Python 实现（不降级 protobuf）
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "=== import test ==="
python3 - <<'PY'
import amct_onnx
print("amct_onnx", amct_onnx.__version__)
from amct_onnx.quantize_tool import create_quant_config, quantize_model
print("quantize_tool OK: create_quant_config / quantize_model")
PY

echo "=== cli entry ==="
which amct_onnx onnx_quant 2>/dev/null || echo "(no cli wrapper)"
echo "=== custom op 包结构 ==="
find /root/amct/op -maxdepth 3 | head -30
echo "=== path.cfg / version.info ==="
cat /root/amct/extract/amct/conf/path.cfg 2>/dev/null
cat /root/amct/extract/amct/version.info 2>/dev/null
echo IMPORT_TEST_DONE
