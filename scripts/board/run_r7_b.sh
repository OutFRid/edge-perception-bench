#!/bin/bash
# R7-B 连板实跑：多路并发吞吐探测（yolov8n，纯推理口径同 bench_latency_acl.py）
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh

SCRIPT=/root/epb/scripts/r7_multi_stream.py
OM=/mnt/e61/workspace/dmd/model/yolov8n.om
OUT=/root/epb/results/r7_multi_stream.json

python3 "$SCRIPT" \
  --om "$OM" \
  --streams 1 2 4 8 \
  --iters 100 --warmup 10 \
  --json "$OUT"
