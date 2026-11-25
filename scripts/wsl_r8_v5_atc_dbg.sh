#!/bin/bash
# R8 · v5 ATC 详细日志排查
set -e
export HOME=/root
TK=/root/Ascend/ascend-toolkit
source "$TK/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/x86_64-linux/devlib:$ASCEND_TOOLKIT_HOME/runtime/lib64/stub:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH"

atc \
  --model=/root/quant/r8/yolov8n_s3_quant_v5.onnx \
  --framework=5 \
  --output=/root/quant/r8/yolov8n_s3_int8_v5dbg \
  --soc_version=Ascend310B3 \
  --input_format=NCHW \
  --input_shape="images:1,3,640,640" \
  --log=error 2>&1 | grep -v '^\[WARNING\]' | head -80
