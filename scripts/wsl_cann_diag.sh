#!/bin/bash
# 诊断 WSL 里 CANN toolkit 的 set_env 与 atc 可用性
export HOME=/root
TK=/root/Ascend/ascend-toolkit
echo "=== toolkit dirs ==="
ls -la "$TK"
echo "=== readlink latest ==="
readlink -f "$TK/latest"
echo "=== source set_env (as script) ==="
source "$TK/set_env.sh"
echo "ASCEND_TOOLKIT_HOME=[$ASCEND_TOOLKIT_HOME]"
echo "PATH has atc? -> $(command -v atc)"
echo "LD_LIBRARY_PATH=[$LD_LIBRARY_PATH]"
echo "=== try atc --version (with fallback LD) ==="
if [ -z "$LD_LIBRARY_PATH" ]; then
  REAL=$(readlink -f "$TK/latest")
  export ASCEND_TOOLKIT_HOME="$REAL"
  export LD_LIBRARY_PATH="$REAL/x86_64-linux/devlib:$REAL/runtime/lib64:$REAL/compiler/lib64:$REAL/tools/tile/development/standard/opp/kernel/3rdparty/hol/hol_modules:$LD_LIBRARY_PATH"
  export PATH="$REAL/bin:$REAL/compiler/bin:$PATH"
  echo "fallback set: ASCEND_TOOLKIT_HOME=$REAL"
fi
atc --version 2>&1 | head -10
echo "DIAG_DONE"
