#!/bin/bash
# R8 · WSL 端：编译中间对照 OM（FP32 mid / v2 mid，11 输出）
set -e
export HOME=/root
TK=/root/Ascend/ascend-toolkit
source "$TK/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/x86_64-linux/devlib:$ASCEND_TOOLKIT_HOME/runtime/lib64/stub:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH"

python3 /mnt/g/Git_repo/edge-perception-bench/scripts/wsl_r8_gen_mid.py

for t in fp32 quant_v2; do
  if [ "$t" = "fp32" ]; then
    SRC=/root/quant/r8/mid/yolov8n_fp32_mid.onnx
    DST=/root/quant/r8/mid/yolov8n_fp32_mid
  else
    SRC=/root/quant/r8/mid/yolov8n_s3_quant_v2_mid.onnx
    DST=/root/quant/r8/mid/yolov8n_s3_int8_v2_mid
  fi
  atc \
    --model="$SRC" \
    --framework=5 \
    --output="$DST" \
    --soc_version=Ascend310B3 \
    --input_format=NCHW \
    --input_shape="images:1,3,640,640" \
    --log=error 2>&1 | tail -5
  echo "--- $t done: $(ls -lh $DST.om 2>/dev/null | awk '{print $5}')"
done

cp -f /root/quant/r8/mid/*_mid.om /mnt/g/Git_repo/edge-perception-bench/models/
ls -lh /mnt/g/Git_repo/edge-perception-bench/models/*_mid.om
