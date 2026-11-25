#!/bin/bash
# R8 · WSL 端：modelslim ONNX PTQ（S3 口径）产出 QDQ ONNX → ATC 编 OM
# 用法: 在 WSL 内执行  bash scripts/wsl_r8_run.sh
set -e
export HOME=/root
TK=/root/Ascend/ascend-toolkit
source "$TK/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/x86_64-linux/devlib:$ASCEND_TOOLKIT_HOME/runtime/lib64/stub:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH"

cd /root
echo "=== 1) modelslim quant (S3) ==="
python3 /mnt/g/Git_repo/edge-perception-bench/scripts/wsl_r8_modelslim_quant.py

echo ""
echo "=== 2) ATC compile quant onnx ==="
QUANT_ONNX=/root/quant/r8/yolov8n_s3_quant_v2.onnx
atc \
  --model="$QUANT_ONNX" \
  --framework=5 \
  --output=/root/om_out/yolov8n_s3_int8_v2 \
  --soc_version=Ascend310B3 \
  --input_format=NCHW \
  --input_shape="images:1,3,640,640" \
  --log=info 2>&1 | tail -30

if [ -f /root/om_out/yolov8n_s3_int8_v2.om ]; then
  echo "=== OK ==="
  ls -lh /root/om_out/yolov8n_s3_int8_v2.om
else
  echo "=== ATC FAILED: no om produced ==="
  exit 1
fi
