#!/bin/bash
# 在 WSL 本地用 ATC 把 ONNX 编译成 OM（不占板端内存）。
# 用法: bash wsl_atc_compile.sh <onnx_path> <output_name_no_ext> [soc_version] [input_shape]
set -e
export HOME=/root
TK=/root/Ascend/ascend-toolkit
source "$TK/set_env.sh"

# 无驱动机器上 atc.bin 需要 libascend_hal.so（toolkit 自带 stub / devlib），补进 LD 路径
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/x86_64-linux/devlib:$ASCEND_TOOLKIT_HOME/runtime/lib64/stub:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH"

ONNX="$1"
OUT="${2:-$(basename "$ONNX" .onnx)}"
SOC="${3:-Ascend310B3}"
SHAPE="${4:-images:1,3,640,640}"
OUTDIR=/root/om_out
mkdir -p "$OUTDIR"

echo "=== sanity: atc help reachable ==="
atc --help 2>&1 | head -2 || true

echo "=== ATC compile ==="
echo "  model  : $ONNX"
echo "  soc    : $SOC"
echo "  shape  : $SHAPE"
echo "  output : $OUTDIR/$OUT.om"

atc \
  --model="$ONNX" \
  --framework=5 \
  --output="$OUTDIR/$OUT" \
  --soc_version="$SOC" \
  --input_format=NCHW \
  --input_shape="$SHAPE" \
  --log=info

if [ -f "$OUTDIR/$OUT.om" ]; then
  echo "=== OK ==="; ls -lh "$OUTDIR/$OUT.om"
else
  echo "=== ATC FAILED: no om produced ==="; exit 1
fi
