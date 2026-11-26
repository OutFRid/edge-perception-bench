#!/bin/bash
# R8 · 为 AMCT 建独立 Python 3.9 环境（custom op 只支持 ORT 1.5.2/1.6.0/1.8.0，均无 py3.10 wheel）
set -e
cd /root

echo "=== 1) 下载 miniconda（TUNA 镜像） ==="
if [ ! -f Miniconda3-latest-Linux-x86_64.sh ]; then
  curl -sS -o Miniconda3-latest-Linux-x86_64.sh \
    https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi
ls -lh Miniconda3-latest-Linux-x86_64.sh

echo "=== 2) 静默安装 ==="
if [ ! -d /root/miniconda3 ]; then
  bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
fi
/root/miniconda3/bin/conda --version

echo "=== 3) 配置 TUNA conda 镜像 ==="
cat > /root/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

echo "=== 4) 创建 py3.9 环境 ==="
/root/miniconda3/bin/conda create -n amct python=3.9 -y -q 2>&1 | tail -5

echo "=== 5) 查 conda-forge 是否有 onnxruntime 1.8.0 ==="
/root/miniconda3/bin/conda search -n amct -c conda-forge "onnxruntime=1.8.0" 2>&1 | tail -10 || echo "(search fail, will use pip)"

echo "=== 6) 环境内 pip 装 ORT 1.8.0 + amct_onnx whl ==="
/root/miniconda3/envs/amct/bin/pip install -q onnxruntime==1.8.0 2>&1 | tail -3
/root/miniconda3/envs/amct/bin/pip install -q /root/amct/extract/amct/amct_onnx/amct_onnx-0.11.1-py3-none-linux_x86_64.whl 2>&1 | tail -3
/root/miniconda3/envs/amct/bin/python -c "import onnxruntime; print('ort', onnxruntime.__version__)"
echo CONDA_ENV_DONE
