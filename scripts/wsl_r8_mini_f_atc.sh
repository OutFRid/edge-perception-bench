#!/bin/bash
# R8 · mini_f 实验 ATC 编译对比
set -e
export HOME=/root
TK=/root/Ascend/ascend-toolkit
source "$TK/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/x86_64-linux/devlib:$ASCEND_TOOLKIT_HOME/runtime/lib64/stub:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH"

python3 /mnt/g/Git_repo/edge-perception-bench/scripts/wsl_r8_mini_f_gen.py

for i in 1 2 3 4 5; do
  echo "===== mini_f$i ====="
  if atc \
    --model=/root/quant/r8/mini/mini_f$i.onnx \
    --framework=5 \
    --output=/root/quant/r8/mini/mini_f$i \
    --soc_version=Ascend310B3 \
    --input_format=NCHW \
    --input_shape="x:1,16,8,8" \
    --log=error > /tmp/mini_f$i.log 2>&1; then
    echo "SUCCESS: $(ls -lh /root/quant/r8/mini/mini_f$i.om | awk '{print $5}')"
  else
    echo "FAILED:"
    grep -oE 'E[0-9]+: [^\\]*|Check failed[^\\]*|No supported Ops[^\\]*' /tmp/mini_f$i.log | head -3
  fi
done
