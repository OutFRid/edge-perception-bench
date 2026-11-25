#!/bin/bash
# R8 · AMCT 7.0.1 PTQ 量化：yolov8n FP32 → 昇腾 QDQ INT8 ONNX
# 标定集与 R5/R6 同口径（12 张真实图，letterbox 640，RGB，/255，NCHW float32 raw bin）。
set -e
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HOME=/root
QA=/root/amct/quant
CAL=/root/amct/calib_12
mkdir -p "$QA" "$CAL"

echo "=== 1) 生成标定 bin（12 张，与 R5/R6 同源同预处理） ==="
python3 - <<'PY'
import glob, os, sys
import cv2, numpy as np
sys.path.insert(0, "/mnt/g/Git_repo/edge-perception-bench/scripts")
from infer_dms_pipeline import LetterboxPreprocess

imgs = []
for g in ["/mnt/g/Git_repo/Image_deal_class/**/*.jpg",
          "/mnt/g/Git_repo/Image_deal_class/**/*.png",
          "/mnt/g/Git_repo/edge-perception-bench/results/image/*.jpg",
          "/mnt/g/Git_repo/edge-perception-bench/results/doc/*.jpg"]:
    imgs += glob.glob(g, recursive=True)
imgs = [i for i in imgs if os.path.isfile(i) and os.path.getsize(i) > 3000]
imgs = list(dict.fromkeys(sorted(imgs)))[:12]
pre = LetterboxPreprocess()
out = "/root/amct/calib_12"
for idx, p in enumerate(imgs):
    t, _, _ = pre(cv2.imread(p))          # (1,3,640,640) float32, letterbox+RGB+/255
    t.astype(np.float32).tofile(f"{out}/{idx:04d}.bin")
print(f"wrote {len(imgs)} calib bins")
PY

echo "=== 2) amct_onnx calibration（默认量化配置，先跑通链路） ==="
amct_onnx calibration \
  --model /root/quant/yolov8n_fp32.onnx \
  --save_path "$QA/yolov8n_amct_default" \
  --input_shape "images:1,3,640,640" \
  --data_dir "$CAL/" \
  --data_types "float32" \
  --batch_num 12 2>&1 | grep -v -E 'device_discovery' | tail -25

echo "=== 3) 产物检查 ==="
ls -lh "$QA"/ 2>/dev/null
echo AMCT_QUANT_DONE
