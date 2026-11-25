#!/bin/bash
# R8 · WSL 端：v4（deq 后 Cast fp32）生成 + ATC 编译（完整版 + mid 版）
set -e
export HOME=/root
TK=/root/Ascend/ascend-toolkit
source "$TK/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/x86_64-linux/devlib:$ASCEND_TOOLKIT_HOME/runtime/lib64/stub:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH"

python3 /mnt/g/Git_repo/edge-perception-bench/scripts/wsl_r8_gen_cast.py
python3 /mnt/g/Git_repo/edge-perception-bench/scripts/wsl_r8_gen_mid.py /root/quant/r8/yolov8n_s3_quant_v4.onnx v4

for t in full mid; do
  if [ "$t" = "full" ]; then
    SRC=/root/quant/r8/yolov8n_s3_quant_v4.onnx
    DST=/root/quant/r8/yolov8n_s3_int8_v4
  else
    SRC=/root/quant/r8/mid/yolov8n_s3_quant_v4_mid.onnx
    DST=/root/quant/r8/mid/yolov8n_s3_int8_v4_mid
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

cp -f /root/quant/r8/yolov8n_s3_int8_v4.om /root/quant/r8/mid/yolov8n_s3_int8_v4_mid.om /mnt/g/Git_repo/edge-perception-bench/models/
ls -lh /mnt/g/Git_repo/edge-perception-bench/models/yolov8n_s3_int8_v4*.om
