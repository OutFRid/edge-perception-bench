#!/bin/bash
# R8 · 网络可达性检查：custom op 头文件源 + deadsnakes PPA + PyPI ORT wheel
echo "== github raw (custom op headers) =="
timeout 15 curl -sI https://raw.githubusercontent.com/microsoft/onnxruntime/v1.6.0/include/onnxruntime/core/session/onnxruntime_cxx_api.h 2>&1 | head -3
echo "== launchpad deadsnakes =="
timeout 15 curl -sI https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu/dists/jammy/Release 2>&1 | head -3
echo "== pypi onnxruntime 1.8.0 cp39 wheel =="
timeout 20 curl -s https://pypi.org/pypi/onnxruntime/1.8.0/json > /tmp/ort180.json
python3 - <<'PY'
import json
try:
    d = json.load(open('/tmp/ort180.json'))
    files = [u['filename'] for u in d['urls'] if 'cp39' in u['filename']]
    print("cp39 wheels:", files)
except Exception as e:
    print("parse fail:", e)
PY
echo "== ubuntu archive (apt) =="
timeout 15 curl -sI http://archive.ubuntu.com/ubuntu/dists/jammy/Release 2>&1 | head -3
echo NET_DONE
