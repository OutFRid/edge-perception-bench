#!/bin/bash
# R8 · 下载 AMCT 7.0.1（匹配 WSL toolkit 7.0.1）并解压检查
set -e
BASE="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.1"
mkdir -p /root/amct
cd /root/amct

echo "=== HEAD probe ==="
for f in "Ascend-cann-amct_7.0.1_linux-x86_64.zip" "Ascend-cann-amct_7.0.1_linux-x86_64.tar.gz"; do
  echo "== $f"
  timeout 20 curl -sI "$BASE/$f" | head -3
done

echo "=== download zip ==="
timeout 300 curl -sS -o Ascend-cann-amct_7.0.1_linux-x86_64.zip "$BASE/Ascend-cann-amct_7.0.1_linux-x86_64.zip"
ls -lh Ascend-cann-amct_7.0.1_linux-x86_64.zip
sha256sum Ascend-cann-amct_7.0.1_linux-x86_64.zip

echo "=== unzip & list ==="
mkdir -p extract
unzip -o -q Ascend-cann-amct_7.0.1_linux-x86_64.zip -d extract
find extract -maxdepth 3 -type f | head -40
echo DOWNLOAD_DONE
