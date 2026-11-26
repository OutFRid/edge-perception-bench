#!/bin/bash
# R8 · 头文件获取：jsdelivr CDN（GitHub 镜像通道）按 v1.8.0 tag 取 4 个头文件
INC=/root/amct/op/amct_onnx_op/inc
cd /root/amct/ort_src

echo "=== npm 4.1K 包内容 ==="
tar tzf ort-node.tgz 2>/dev/null | head -20

echo "=== jsdelivr 探测 ==="
curl -s --max-time 20 -o /tmp/jsd_test.h -w "HTTP %{http_code}, %{size_download} bytes\n" \
  "https://cdn.jsdelivr.net/gh/microsoft/onnxruntime@v1.8.0/include/onnxruntime/core/session/onnxruntime_cxx_api.h"
head -5 /tmp/jsd_test.h

echo "=== 下载 4 个头文件（v1.8.0） ==="
for H in onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_c_api.h onnxruntime_session_options_config_keys.h; do
  curl -s --max-time 30 -o "$INC/$H" -w "$H: HTTP %{http_code}, %{size_download}B\n" \
    "https://cdn.jsdelivr.net/gh/microsoft/onnxruntime@v1.8.0/include/onnxruntime/core/session/$H"
done
ls -lh "$INC"/onnxruntime_*.h
echo "=== 验证头文件版本标记 ==="
grep -m1 -E "ORT_API_VERSION|Copyright" "$INC/onnxruntime_c_api.h" | head -2
grep -m2 "ORT_API_VERSION" "$INC/onnxruntime_c_api.h" | head -2
echo JSD_DONE
