#!/bin/bash
# Convert YOLOv8 ONNX to OM (offline model) via ATC
# Usage: bash convert_onnx_to_om.sh [onnx_path] [soc_version]

MODEL="${1:-../models/yolov8n.onnx}"
SOC="${2:-Ascend310}"
OUTPUT="${MODEL%.onnx}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "=== ATC: ONNX → OM ==="
echo "  Model : $MODEL"
echo "  SoC   : $SOC"
echo "  Output: ${OUTPUT}.om"

atc \
    --model="$MODEL" \
    --framework=5 \
    --output="$OUTPUT" \
    --soc_version="$SOC" \
    --input_format=NCHW \
    --input_shape="images:1,3,640,640" \
    --log=info

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Done ==="
    ls -lh "${OUTPUT}.om"
else
    echo ""
    echo "=== ATC FAILED ==="
    exit 1
fi
