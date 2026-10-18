#!/bin/bash
# R2 连板实跑：底层 ACL DMS 流水线在真实 DMD 片段上逐帧跑通 + 分段计时
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh

SCRIPT=/root/epb/scripts/infer_dms_pipeline.py
OM=/mnt/e61/workspace/dmd/model/yolov8n.om
OUTDIR=/root/epb/results
mkdir -p "$OUTDIR"

# 用 glob 解析含分号的 DMD 文件名，避免在命令行手打特殊字符
CLIP=$(ls /mnt/e61/workspace/dmd/gA/1/s1/*_mosaic.avi 2>/dev/null | head -1)
echo "CLIP = $CLIP"

python3 "$SCRIPT" \
  --om "$OM" \
  --video "$CLIP" \
  --out  "$OUTDIR/dms_gA1s1.mp4" \
  --sample-jpeg "$OUTDIR/dms_gA1s1_frame.jpg" \
  --json "$OUTDIR/dms_gA1s1_stats.json" \
  --max-frames 60
