#!/bin/bash
# R8 · 解压 AMCT 7.0.1 包并查看内容结构
cd /root/amct || exit 1
python3 - <<'PY'
import zipfile
zipfile.ZipFile('Ascend-cann-amct_7.0.1_linux-x86_64.zip').extractall('extract')
print("extracted OK")
PY
echo "=== file list (depth<=4) ==="
find extract -maxdepth 4 -type f | head -50
echo "=== dir sizes ==="
du -sh extract/* 2>/dev/null
echo "=== install docs ==="
find extract -iname "*readme*" -o -iname "*.md" -o -iname "*install*" | head -20
echo UNZIP_DONE
