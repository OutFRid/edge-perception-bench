#!/bin/bash
set -e
echo "=== pip install onnxruntime ==="
pip install --quiet onnxruntime 2>&1 | tail -8 || pip install onnxruntime 2>&1 | tail -8
echo "=== verify ==="
python3 - <<'PY'
import onnxruntime as o
print("ort", o.__version__)
from onnxruntime.quantization import quantize_static, quantize_dynamic
print("quant modules OK")
PY
