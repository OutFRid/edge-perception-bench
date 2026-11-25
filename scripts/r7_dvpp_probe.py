"""R7 · 板端 DVPP 能力探测：Python 绑定、VPC/JPEGD 模块、可用 API 面。

目标（对照 R7 主线「DVPP vs OpenCV 预处理」）：
  1. acldvpp Python 绑定是否存在、版本；
  2. 可用能力面（vpc resize/crop/pad、jpeg decode）与 cap 枚举；
  3. ACL Python 里 Device 内存句柄形态（DVPP 输出的数据怎么搬回 Host）。

用法（板端）:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 r7_dvpp_probe.py
"""
import importlib
import json
import sys


def probe_acldvpp():
    out = {}
    try:
        acldvpp = importlib.import_module("acldvpp")
        out["import"] = "OK"
        out["attrs"] = [a for a in dir(acldvpp) if not a.startswith("_")]
    except Exception as e:  # noqa: BLE001
        out["import"] = f"FAIL: {type(e).__name__}: {e}"
        out["attrs"] = []
    return out


def probe_caps():
    """枚举 ACL Python 绑定暴露的全部 DVPP 函数面（acl.media 模块）。"""
    out = {"acl_media_all": [], "acl_media_vpc": [], "acl_media_jpeg": [],
           "acl_media_png": [], "acl_media_other": []}
    try:
        import acl
        m = getattr(acl, "media", None)
        if m is None:
            out["acl_media_all"] = ["NO acl.media"]
            return out
        attrs = sorted(a for a in dir(m) if not a.startswith("_"))
        out["acl_media_all"] = attrs
        for a in attrs:
            u = a.upper()
            if "VPC" in u:
                out["acl_media_vpc"].append(a)
            elif "JPEG" in u:
                out["acl_media_jpeg"].append(a)
            elif "PNG" in u:
                out["acl_media_png"].append(a)
            else:
                out["acl_media_other"].append(a)
    except Exception as e:  # noqa: BLE001
        out["acl_media_all"] = [f"FAIL {e}"]
    return out


def probe_device_mem_shape():
    """ACL Python 的 Device 内存返回值形态（决定 DVPP 输出如何落 Host）。"""
    out = {}
    try:
        import acl
        acl.init("")
        acl.rt.set_device(0)
        p, ret = acl.rt.malloc(64, 0)
        out["malloc_ret"] = ret
        out["malloc_type"] = type(p).__name__
        out["malloc_value"] = str(p)[:60]
        # 看看 memcpy 的源/目的参数怎么吃这个句柄
        import numpy as np
        buf = np.zeros(64, dtype=np.uint8)
        r = acl.rt.memcpy(buf.ctypes.data, 64, p, 64, 2)  # D2H
        out["memcpy_d2h_ret"] = r
        acl.rt.free(p)
        acl.rt.reset_device(0)
        acl.finalize()
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def probe_dvpp_import_paths():
    """可能的 DVPP Python 绑定安装位置。"""
    out = {}
    for name in ("acldvpp", "acl"):
        try:
            m = importlib.import_module(name)
            out[name] = getattr(m, "__file__", "?")
        except Exception as e:  # noqa: BLE001
            out[name] = f"FAIL {e}"
    return out


def main():
    report = {
        "acldvpp": probe_acldvpp(),
        "caps": probe_caps(),
        "device_mem": probe_device_mem_shape(),
        "paths": probe_dvpp_import_paths(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        with open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/r7_dvpp_probe.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
