#!/bin/bash
# R3-B 连板实跑：把 yolov8s（s 代替 n，同为 COCO 检测、输出布局一致）
# 跑通底层 ACL DMS 流水线，逐帧分段计时，与 R2 的 yolov8n 基线横向对比。
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh

SCRIPT=/root/epb/scripts/infer_dms_pipeline.py
OM=/mnt/e61/workspace/dmd/model/yolov8s.om
OUTDIR=/root/epb/results
mkdir -p "$OUTDIR"

# 用 glob 解析含分号的 DMD 文件名，避免在命令行手打特殊字符
CLIP=$(ls /mnt/e61/workspace/dmd/gA/1/s1/*_mosaic.avi 2>/dev/null | head -1)
echo "CLIP = $CLIP"
echo "OM   = $OM"

python3 "$SCRIPT" \
  --om "$OM" \
  --video "$CLIP" \
  --out  "$OUTDIR/dms_yolov8s_gA1s1.mp4" \
  --sample-jpeg "$OUTDIR/dms_yolov8s_gA1s1_frame.jpg" \
  --json "$OUTDIR/dms_yolov8s_gA1s1_stats.json" \
  --max-frames 60
