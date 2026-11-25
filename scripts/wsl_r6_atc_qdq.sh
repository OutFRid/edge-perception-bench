#!/bin/bash
# R6 · 验证 WSL CANN 7.0.1 ATC 是否接受 onnxruntime 产出的 QDQ INT8 ONNX。
# 背景：板端 7.0.RC1 atc 拒绝 QDQ（No parser is registered for Op DequantizeLinear）。
# 本脚本在 WSL 7.0.1 上试编同一份模型，补齐「工具链版本 vs QDQ 支持」的验证矩阵。
set -e
export HOME=/root
TK=/root/Ascend/ascend-toolkit
source "$TK/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/x86_64-linux/devlib:$ASCEND_TOOLKIT_HOME/runtime/lib64/stub:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH"

MODEL=/root/quant/yolov8n_int8.onnx
OUT=/root/quant/r6/test_qdq_om
mkdir -p /root/quant/r6
echo "=== atc on QDQ INT8 ONNX (soc=Ascend310B1) ==="
atc --model="$MODEL" --framework=5 --output="$OUT" --soc_version=Ascend310B1 \
    --input_format=NCHW --input_shape="images:1,3,640,640" --log=error 2>&1 | tail -20 || true
if [ -f "$OUT.om" ]; then
  echo "=== ATC_ACCEPT_QDQ ==="; ls -lh "$OUT.om"
else
  echo "=== ATC_REJECT_QDQ ==="
fi
