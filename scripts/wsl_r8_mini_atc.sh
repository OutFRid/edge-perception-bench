#!/bin/bash
# R8 · WSL 端：受控最小实验——生成 3 个 mini ONNX 并 ATC 编译
set -e
export HOME=/root
TK=/root/Ascend/ascend-toolkit
source "$TK/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/x86_64-linux/devlib:$ASCEND_TOOLKIT_HOME/runtime/lib64/stub:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH"

echo "=== 1) gen mini onnx ==="
python3 /mnt/g/Git_repo/edge-perception-bench/scripts/wsl_r8_mini_gen.py

echo ""
echo "=== 2) ATC compile ==="
for m in mini_a mini_b mini_c mini_d; do
  atc \
    --model="/root/quant/r8/mini/$m.onnx" \
    --framework=5 \
    --output="/root/quant/r8/mini/$m" \
    --soc_version=Ascend310B3 \
    --input_format=NCHW \
    --input_shape="x:1,1,1,1" \
    --log=error 2>&1 | tail -5
  echo "--- $m: $(ls -lh /root/quant/r8/mini/$m.om 2>/dev/null | awk '{print $5}')"
done

echo ""
echo "=== 3) copy om -> /mnt/g ==="
cp -f /root/quant/r8/mini/mini_*.om /mnt/g/Git_repo/edge-perception-bench/models/
ls -lh /mnt/g/Git_repo/edge-perception-bench/models/mini_*.om
