#!/bin/bash
# R8 · NPU INT8 上板：精度同帧对照 + 纯推理延时基准（FP32 vs INT8 OM）
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/epb
mkdir -p models results

FP32=/mnt/e61/workspace/dmd/model/yolov8n.om
INT8=/root/epb/models/yolov8n_s3_int8.om
ls -lh "$FP32" "$INT8"

echo "=== 1) 精度对照（16 帧同帧） ==="
python3 scripts/r8_board_eval.py \
  --fp32 "$FP32" \
  --int8 "$INT8" \
  --n-frames 16 \
  --json results/r8_board_eval.json

echo ""
echo "=== 2) 延时基准 FP32 ==="
python3 scripts/bench_latency_acl.py --om "$FP32" \
  --iters 200 --warmup 20 --tag yolov8n_fp32_r8 --json results/r8_latency.json

echo ""
echo "=== 3) 延时基准 INT8 ==="
python3 scripts/bench_latency_acl.py --om "$INT8" \
  --iters 200 --warmup 20 --tag yolov8n_int8_r8 --json results/r8_latency.json

echo "=== R8 board done ==="
ls -lh results/r8_board_eval.json results/r8_latency.json
