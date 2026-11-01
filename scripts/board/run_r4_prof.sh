#!/bin/bash
# R4 · msprof 设备侧逐层 profiling：对 yolov8n.om 的纯推理循环采集
#   - 逐层(AI-core task)延时  - H2D/D2H 拷贝耗时  - 内存/度量
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/epb
OUT=/root/epb/prof_r4_yolov8n
rm -rf "$OUT"; mkdir -p "$OUT"

# 用已有的模型无关 bench 作为被采集应用（纯 execute + 两次 memcpy），iters 控制采样量
msprof \
  --output="$OUT" \
  --application="python3 scripts/bench_latency_acl.py --om /mnt/e61/workspace/dmd/model/yolov8n.om --iters 40 --warmup 5 --tag yolov8n_prof" \
  --ai-core=on \
  --aic-mode=task-based \
  --task-time=on \
  --runtime-api=on \
  --ascendcl=on \
  --model-execution=on

echo "=== collection done, tree ==="
find "$OUT" -maxdepth 3 -type d | sort
echo "=== SUMMARY csv list ==="
find "$OUT" -name "*.csv" | sort
