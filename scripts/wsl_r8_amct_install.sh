#!/bin/bash
# R8 · 安装 AMCT 7.0.1：amct_onnx whl + custom op 包
set -e
cd /root/amct/extract/amct/amct_onnx

echo "=== 1) pip install amct_onnx whl ==="
pip install --quiet amct_onnx-0.11.1-py3-none-linux_x86_64.whl 2>&1 | tail -5
python3 -c "import amct_onnx as a; print('amct_onnx', a.__version__)"

echo "=== 2) custom op 包结构 ==="
mkdir -p /root/amct/op && tar -xzf amct_onnx_op.tar.gz -C /root/amct/op
find /root/amct/op -maxdepth 3 | head -30

echo "=== 3) path.cfg ==="
cat ../conf/path.cfg 2>/dev/null || true
cat ../version.info 2>/dev/null || true
echo AMCT_INSTALL_STEP_DONE
