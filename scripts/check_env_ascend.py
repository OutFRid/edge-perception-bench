"""Check ONNX Runtime + Ascend environment on the board."""

import subprocess
import sys


def run(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        return (r.stdout.strip() or r.stderr.strip())
    except Exception as e:
        return f"ERROR: {e}"


def main():
    print("=" * 50)
    print("Ascend ONNX Runtime — Board Env Check")
    print("=" * 50)

    checks = [
        ("Python", "python3 --version"),
        ("CANN version", "cat /usr/local/Ascend/ascend-toolkit/latest/arm64-linux/ascend_toolkit_install.info 2>/dev/null | head -3 || echo NOT_FOUND"),
        ("NPU status", "npu-smi info 2>/dev/null | grep -E 'Chip|Version|Health|AICore' || echo NOT_FOUND"),
        ("LD_LIBRARY_PATH", "echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -i ascend | head -5"),
        ("onnxruntime installed", "python3 -c \"import onnxruntime; print(onnxruntime.__version__)\""),
        ("Available EPs", "python3 -c \"import onnxruntime as ort; print(ort.get_available_providers())\""),
        ("ascend-toolkit path", "ls /usr/local/Ascend/ascend-toolkit/ 2>/dev/null || echo NOT_FOUND"),
        ("OpenCV", "python3 -c \"import cv2; print(cv2.__version__)\""),
        ("NumPy", "python3 -c \"import numpy; print(numpy.__version__)\""),
    ]

    for label, cmd in checks:
        result = run(cmd)
        print(f"\n[{label}]")
        print(result or "(no output)")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
