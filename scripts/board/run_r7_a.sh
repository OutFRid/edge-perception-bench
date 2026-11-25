#!/bin/bash
# R7-A 连板实跑：DVPP vs OpenCV 预处理对比（DMD gA/1/s1 真实帧）
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh

SCRIPT=/root/epb/scripts/r7_preprocess_bench.py
OUT=/root/epb/results/r7_preprocess_bench.json

CLIP=$(ls /mnt/e61/workspace/dmd/gA/1/s1/*_mosaic.avi 2>/dev/null | head -1)
echo "CLIP = $CLIP"

python3 "$SCRIPT" \
  --video "$CLIP" \
  --frames 30 --warmup 5 \
  --json "$OUT"
