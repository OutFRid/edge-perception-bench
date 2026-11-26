#!/bin/bash
# R8 · 补装 onnx + 提取 ORT 1.8.0 头文件（不用 set -e，各段独立执行）
PY=/root/miniconda3/envs/amct/bin/python
PIP=/root/miniconda3/envs/amct/bin/pip
INC=/root/amct/op/amct_onnx_op/inc

echo "=== 1) 补装 onnx 1.12.0（与 ORT 1.8 / protobuf 3.20 时代匹配） ==="
$PIP install -q onnx==1.12.0 2>&1 | grep -v "WARNING: Running pip" | tail -2
$PY -c "import onnx; print('onnx', onnx.__version__)"
$PY -c "import amct_onnx; print('amct_onnx OK', amct_onnx.__version__)" || echo "AMCT_IMPORT_FAIL"

echo "=== 2) 下载 ORT 1.8.0 sdist ==="
mkdir -p /root/amct/ort_src
cd /root/amct/ort_src
if [ ! -f onnxruntime-1.8.0.tar.gz ]; then
  SDIST_URL=$($PY - <<'PYEOF'
import json, urllib.request
d = json.load(urllib.request.urlopen('https://pypi.org/pypi/onnxruntime/1.8.0/json'))
for u in d['urls']:
    if u['packagetype'] == 'sdist':
        print(u['url']); break
PYEOF
)
  echo "sdist: $SDIST_URL"
  curl -sS -o onnxruntime-1.8.0.tar.gz "$SDIST_URL"
fi
ls -lh onnxruntime-1.8.0.tar.gz

echo "=== 3) 提取 4 个头文件 ==="
tar xzf onnxruntime-1.8.0.tar.gz --wildcards "*/include/onnxruntime/core/session/*.h" 2>/dev/null
SRC_H=$(find . -path "*/include/onnxruntime/core/session/onnxruntime_cxx_api.h" 2>/dev/null | head -1)
if [ -z "$SRC_H" ]; then
  echo "wildcards 提取失败，全量解压"
  tar xzf onnxruntime-1.8.0.tar.gz
  SRC_H=$(find . -path "*/include/onnxruntime/core/session/onnxruntime_cxx_api.h" | head -1)
fi
echo "found: $SRC_H"
SRC_DIR=$(dirname "$SRC_H")
cp "$SRC_DIR/onnxruntime_cxx_api.h" "$SRC_DIR/onnxruntime_cxx_inline.h" \
   "$SRC_DIR/onnxruntime_c_api.h" "$SRC_DIR/onnxruntime_session_options_config_keys.h" "$INC/" && echo "headers copied"
ls -lh "$INC"/*.h

echo "=== 4) setup.py build 入口 ==="
grep -n -E "class |def build|cmdclass|ext_modules|setup\(" /root/amct/op/amct_onnx_op/setup.py | head -20
echo FIX3_DONE
