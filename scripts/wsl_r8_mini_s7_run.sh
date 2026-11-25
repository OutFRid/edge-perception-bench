#!/bin/bash
# R8 · WSL：s7 单层复现（提取参数 + CPU 参考 + ATC 编译）
set -e
export HOME=/root
TK=/root/Ascend/ascend-toolkit
source "$TK/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/x86_64-linux/devlib:$ASCEND_TOOLKIT_HOME/runtime/lib64/stub:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH"

cd /root/quant/r8/mini
python3 /mnt/g/Git_repo/edge-perception-bench/scripts/wsl_r8_mini_s7.py

atc \
  --model=mini_s7.onnx \
  --framework=5 \
  --output=mini_s7 \
  --soc_version=Ascend310B3 \
  --input_format=NCHW \
  --input_shape="x:1,128,40,40" \
  --log=error 2>&1 | tail -3

ls -lh mini_s7.om
cp -f mini_s7.om /mnt/g/Git_repo/edge-perception-bench/models/
cp -f mini_s7_input.npy mini_s7_ref.npy /mnt/g/Git_repo/edge-perception-bench/results/
ls -lh /mnt/g/Git_repo/edge-perception-bench/models/mini_s7.om /mnt/g/Git_repo/edge-perception-bench/results/mini_s7_*.npy
