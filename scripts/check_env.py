import subprocess
import sys
from datetime import datetime


def run(cmd: str) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip() or result.stderr.strip()
    except Exception as e:
        return f"ERROR: {e}"


def main():
    print("=" * 50)
    print(f"Edge Perception Bench — 环境检测")
    print(f"检测时间: {datetime.now().isoformat()}")
    print("=" * 50)

    sections = [
        ("Python 版本", "python --version"),
        ("操作系统", "ver"),
        ("pip 版本", "pip --version"),
        ("CANN 版本", "cat /usr/local/Ascend/ascend-toolkit/version.cfg 2>nul || echo NOT_FOUND"),
        ("NPU 状态", "npu-smi info 2>nul || echo NOT_FOUND"),
        ("昇腾驱动", "cat /usr/local/Ascend/driver/version.info 2>nul || echo NOT_FOUND"),
        ("ONNX Runtime", "python -c \"import onnxruntime; print(onnxruntime.__version__)\""),
        ("PyTorch", "python -c \"import torch; print(torch.__version__)\""),
        ("NumPy", "python -c \"import numpy; print(numpy.__version__)\""),
        ("CUDA", "nvidia-smi 2>nul || echo NOT_FOUND"),
    ]

    for label, cmd in sections:
        result = run(cmd)
        print(f"\n[{label}]")
        print(result or "(no output)")

    print("\n" + "=" * 50)
    print("环境检测完成。将上述输出保存为 results/env_info.txt")


if __name__ == "__main__":
    main()
