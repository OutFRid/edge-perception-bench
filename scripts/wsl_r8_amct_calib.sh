#!/bin/bash
# R8 · AMCT calibration（py3.9 环境 + 已编译 custom op）
export PATH=/root/miniconda3/envs/amct/bin:$PATH
QA=/mnt/g/Git_repo/edge-perception-bench/results/doc
CAL=/root/amct/calib_12

echo "=== 0) 验证 custom op 库加载 ==="
python -c "import amct_onnx; print('amct_onnx', amct_onnx.__version__)"
ls -lh /root/amct/calib_12/ | head -4

echo "=== 1) 清理旧产物 ==="
rm -rf "$QA/yolov8n_amct_default"
mkdir -p "$QA"

echo "=== 2) amct_onnx calibration ==="
amct_onnx calibration \
  --model /root/quant/yolov8n_fp32.onnx \
  --save_path "$QA/yolov8n_amct_default" \
  --input_shape "images:1,3,640,640" \
  --data_dir "$CAL/" \
  --data_types "float32" \
  --batch_num 12 2>&1 | tail -40
echo "CALIB_EXIT=$?"

echo "=== 3) 产物检查 ==="
ls -lhR "$QA/yolov8n_amct_default" 2>/dev/null | head -30
echo CALIB_DONE
