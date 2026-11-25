#!/bin/bash
# R8 依赖探测：ort 版本 + modelslim label_free 调用链
echo "=== onnxruntime version ==="
python3 -c "import onnxruntime; print(onnxruntime.__version__)"
echo ""
echo "=== label_free/quantize_tool.py (full) ==="
cat /root/Ascend/ascend-toolkit/latest/python/site-packages/modelslim/onnx/post_training_quant/label_free/quantize_tool.py
echo ""
echo "=== quantize_static kwargs 构造点 ==="
grep -rn "optimize_model\|quantize_static\|quant_kwargs" /root/Ascend/ascend-toolkit/latest/python/site-packages/modelslim/onnx/post_training_quant/ --include='*.py'
