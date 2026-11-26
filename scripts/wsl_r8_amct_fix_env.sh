#!/bin/bash
# R8 · 修复 numpy 兼容 + 提取 ORT 1.8.0 头文件（custom op 编译依赖，GitHub raw 不可达改走 PyPI sdist）
set -e
PY=/root/miniconda3/envs/amct/bin/python
PIP=/root/miniconda3/envs/amct/bin/pip
INC=/root/amct/op/amct_onnx_op/inc

echo "=== 1) numpy 降级（ORT 1.8.0 由 numpy 1.x 编译） ==="
$PIP install -q "numpy<2" 2>&1 | grep -v "WARNING: Running pip" | tail -3 || true
$PY -c "import numpy; print('numpy', numpy.__version__)"
$PY -c "import onnxruntime as ort; print('ort', ort.__version__, ort.get_available_providers())"
$PY -c "import amct_onnx; print('amct_onnx', amct_onnx.__version__)"

echo "=== 2) 下载 ORT 1.8.0 sdist 并提取 4 个头文件 ==="
mkdir -p /root/amct/ort_src
cd /root/amct/ort_src
if [ ! -f onnxruntime-1.8.0.tar.gz ]; then
  SDIST_URL=$($PY -c "
import json, urllib.request
d = json.load(urllib.request.urlopen('https://pypi.org/pypi/onnxruntime/1.8.0/json'))
for u in d['urls']:
    if u['packagetype'] == 'sdist':
        print(u['url']); break
")
  echo "sdist: $SDIST_URL"
  curl -sS -o onnxruntime-1.8.0.tar.gz "$SDIST_URL"
fi
ls -lh onnxruntime-1.8.0.tar.gz
tar xzf onnxruntime-1.8.0.tar.gz --wildcards "*/include/onnxruntime/core/session/onnxruntime_cxx_api.h" \
  "*/include/onnxruntime/core/session/onnxruntime_cxx_inline.h" \
  "*/include/onnxruntime/core/session/onnxruntime_c_api.h" \
  "*/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h" 2>/dev/null || \
  tar xzf onnxruntime-1.8.0.tar.gz
find . -name "onnxruntime_cxx_api.h" | head -3

echo "=== 3) 复制头文件到 custom op inc/ ==="
SRC_H=$(find . -path "*/include/onnxruntime/core/session/onnxruntime_cxx_api.h" | head -1)
SRC_DIR=$(dirname "$SRC_H")
cp "$SRC_DIR/onnxruntime_cxx_api.h" "$SRC_DIR/onnxruntime_cxx_inline.h" \
   "$SRC_DIR/onnxruntime_c_api.h" "$SRC_DIR/onnxruntime_session_options_config_keys.h" "$INC/"
ls -lh "$INC"/*.h

echo "=== 4) setup.py build 入口确认 ==="
grep -n -E "class |def build|cmdclass|ext_modules|install_requires" /root/amct/op/amct_onnx_op/setup.py | head -20
echo FIX_ENV_DONE
