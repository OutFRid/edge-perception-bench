#!/bin/bash
# R8 前置探测：modelslim 全部 onnx 依赖点 + FP32 模型 opset
MS=/root/Ascend/ascend-toolkit/latest/python/site-packages/modelslim
echo "=== modelslim onnx imports ==="
grep -rhn "import onnx\|from onnx" "$MS" --include='*.py' | sort -u
echo ""
echo "=== FP32 onnx opset ==="
python3 - <<'EOF'
import onnx
m = onnx.load('/root/quant/yolov8n_fp32.onnx')
print('opset_import:', m.opset_import, ' ir_version:', m.ir_version)
EOF
