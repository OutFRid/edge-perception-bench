#!/bin/bash
# R6 · WSL 环境诊断：onnxruntime 量化模块 + AMCT 可得性 + 网络可达性
echo "=== whoami / py ==="
whoami
python3 --version
echo "=== ort & quant modules ==="
python3 - <<'PY'
import onnxruntime as o
print("ort", o.__version__)
from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantFormat, QuantType
print("quant modules OK")
import onnx
print("onnx", onnx.__version__)
try:
    import torch
    print("torch", torch.__version__)
except Exception as e:
    print("torch MISSING", e)
try:
    import ultralytics
    print("ultralytics", ultralytics.__version__)
except Exception as e:
    print("ultralytics MISSING", e)
PY
echo "=== amct/msct packages ==="
pip list 2>/dev/null | grep -i -E 'amct|msct' || echo "(no amct/msct pip packages)"
which amct amct_onnx onnx_quant.py amct_onnx_quant.py 2>/dev/null || echo "(no amct cli on PATH)"
echo "=== /root/quant contents ==="
ls -la /root/quant/ 2>/dev/null
echo "=== net check (gitee ascenda hub) ==="
timeout 15 curl -sI https://gitee.com 2>&1 | head -3 || echo "CURL_GITEE_FAIL"
timeout 15 curl -sI https://ascend-repo.obs.cn-east-2.myhuaweicloud.com 2>&1 | head -3 || echo "CURL_HW_FAIL"
echo DIAG_DONE
