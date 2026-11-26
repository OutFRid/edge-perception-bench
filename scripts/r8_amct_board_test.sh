#!/bin/bash
# R8 · AMCT 路线 INT8 OM 板端测试（延时 + 精度，与 ModelSlim 路线同口径对比）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/epb/scripts

echo "=== 1) latency bench (yolov8n_amct_int8.om) ==="
python3 bench_latency_acl.py \
  --om /root/epb/models/yolov8n_amct_int8.om \
  --iters 200 --warmup 20 \
  --tag yolov8n_amct_int8_r8 \
  --json /root/epb/results/r8_latency.json

echo "=== 2) accuracy eval (FP32 vs AMCT-INT8, 16 frames DMD2) ==="
python3 r8_board_eval.py \
  --fp32 /mnt/e61/workspace/dmd/model/yolov8n.om \
  --int8 /root/epb/models/yolov8n_amct_int8.om \
  --n-frames 16 \
  --json /root/epb/results/r8_board_eval_amct.json

echo "=== 3) 结果摘要 ==="
python3 - <<'PYEOF'
import json
lat = json.load(open('/root/epb/results/r8_latency.json'))
for r in lat:
    print(f"  {r['tag']:24s} compute={r['compute_ms']['mean']:7.3f}ms total={r['total_infer_ms']:7.3f}ms fps={r['fps']}")
ev = json.load(open('/root/epb/results/r8_board_eval_amct.json'))
a = ev['accuracy']
print(f"  AMCT-INT8 accuracy: cosine={a['output_cosine_mean']:.5f} "
      f"det(fp32/int8)={a['det_count_mean']['fp32']}/{a['det_count_mean']['int8']} "
      f"conf={a['person_conf_mean']['int8']:.4f} IoU={a['matched_box_iou_mean']:.4f}")
PYEOF
echo BOARD_TEST_DONE
