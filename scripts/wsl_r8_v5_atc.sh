#!/bin/bash
# R8 · WSL 端：v5（输出重量化）生成 + ATC 编译（完整版 + mid 版）
set -e
export HOME=/root
TK=/root/Ascend/ascend-toolkit
source "$TK/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/x86_64-linux/devlib:$ASCEND_TOOLKIT_HOME/runtime/lib64/stub:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH"

python3 /mnt/g/Git_repo/edge-perception-bench/scripts/wsl_r8_gen_v5.py
python3 /mnt/g/Git_repo/edge-perception-bench/scripts/wsl_r8_gen_mid.py /root/quant/r8/yolov8n_s3_quant_v5.onnx v5

for t in full mid; do
  if [ "$t" = "full" ]; then
    SRC=/root/quant/r8/yolov8n_s3_quant_v5.onnx
    DST=/root/quant/r8/yolov8n_s3_int8_v5
  else
    SRC=/root/quant/r8/mid/yolov8n_s3_quant_v5_mid.onnx
    DST=/root/quant/r8/mid/yolov8n_s3_int8_v5_mid
  fi
  atc \
    --model="$SRC" \
    --framework=5 \
    --output="$DST" \
    --soc_version=Ascend310B3 \
    --input_format=NCHW \
    --input_shape="images:1,3,640,640" \
    --log=error 2>&1 | tail -4
  echo "--- $t done: $(ls -lh $DST.om 2>/dev/null | awk '{print $5}')"
done

cp -f /root/quant/r8/yolov8n_s3_int8_v5.om /root/quant/r8/mid/yolov8n_s3_int8_v5_mid.om /mnt/g/Git_repo/edge-perception-bench/models/
ls -lh /mnt/g/Git_repo/edge-perception-bench/models/yolov8n_s3_int8_v5*.om
