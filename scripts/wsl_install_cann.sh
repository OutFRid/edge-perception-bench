#!/bin/bash
# 在 PC 本地 WSL(Ubuntu-22.04) 安装 CANN Toolkit（ATC 编译环境），避免占用板端内存。
# 幂等：已装则跳过。日志走 stdout，后台执行时重定向到 ~/cann_install.log
set -e
RUN=/mnt/g/Git_repo/edge-perception-bench/Ascend-cann-toolkit_7.0.1_linux-x86_64.run

if [ -x /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  echo "CANN already installed, skip"; exit 0
fi

echo "[1/4] apt deps ..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  gcc g++ make cmake python3 python3-dev python3-pip python3-setuptools \
  zlib1g-dev libgomp1 openssl libssl-dev

echo "[2/4] copy .run to native fs (fast cp) ..."
mkdir -p ~/cann
cp -f "$RUN" ~/cann/toolkit.run
chmod +x ~/cann/toolkit.run

echo "[3/4] install CANN toolkit -> /usr/local/Ascend ..."
sudo ~/cann/toolkit.run --install --install-path=/usr/local/Ascend --framework=0 2>&1 || \
sudo ~/cann/toolkit.run --install --install-path=/usr/local/Ascend 2>&1

echo "[4/4] verify atc ..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh
which atc && atc --version 2>&1 | head -3 || echo "ATC not on PATH, check install log"
echo "CANN_INSTALL_DONE"
