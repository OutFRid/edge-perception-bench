"""R7 · 探测 acl.media Python 绑定的调用签名 / 常量枚举 / 返回形态。

用三种手段确认「Python 里怎么调 DVPP」：
  1. __doc__ 里的 C 签名（help 文本）；
  2. 常量面（PIXEL_FORMAT_* / DVPP_CHNMODE_* 等，可能在 acl.media / acl.const / acl 顶层）；
  3. 实测 create/destroy 系列的返回形态（指针是 int？错误码是 tuple 还是 int？）。
"""
import json

import numpy as np

FUNCS = [
    "dvpp_create_channel_desc", "dvpp_set_channel_desc_mode", "dvpp_create_channel",
    "dvpp_create_pic_desc", "dvpp_set_pic_desc_data", "dvpp_set_pic_desc_size",
    "dvpp_set_pic_desc_format", "dvpp_set_pic_desc_width", "dvpp_set_pic_desc_height",
    "dvpp_set_pic_desc_width_stride", "dvpp_set_pic_desc_height_stride",
    "dvpp_create_resize_config", "dvpp_set_resize_config_interpolation",
    "dvpp_vpc_resize_async", "dvpp_vpc_crop_resize_make_border_async",
    "dvpp_create_border_config", "dvpp_set_border_config_left", "dvpp_set_border_config_top",
    "dvpp_set_border_config_value", "dvpp_set_border_config_border_type",
    "dvpp_malloc", "dvpp_free", "dvpp_destroy_channel", "dvpp_destroy_channel_desc",
]


def main():
    import acl
    media = acl.media

    report = {"docs": {}, "consts": {}, "runtime": {}}

    # 1) __doc__ 签名
    for name in FUNCS:
        fn = getattr(media, name, None)
        report["docs"][name] = (fn.__doc__ or "").strip().replace("\n", " ")[:220] if fn else "MISSING"

    # 2) 常量枚举：PIXEL_FORMAT / 通道模式 / 插值方式
    def scan(mod, prefix, out):
        if mod is None:
            return
        for a in dir(mod):
            if a.startswith(prefix):
                out[a] = repr(getattr(mod, a))[:80]
    consts = {}
    for modname in ("acl.media", "acl.const", "acl"):
        try:
            mod = __import__(modname, fromlist=["*"])
        except Exception:
            continue
        scan(mod, "PIXEL_FORMAT_", consts)
        scan(mod, "DVPP_CHNMODE_", consts)
        scan(mod, "VPC_PT_", consts)
        scan(mod, "RESIZE_", consts)
        scan(mod, "BORDER_", consts)
    report["consts"] = {k: consts[k] for k in sorted(consts)}

    # 3) 实测 create/set 返回形态
    rt = {}
    try:
        cd = media.dvpp_create_channel_desc()
        rt["create_channel_desc"] = repr(cd)
        r = media.dvpp_set_channel_desc_mode(cd, 1)  # 1 = VPC
        rt["set_channel_desc_mode"] = repr(r)
        chan = media.dvpp_create_channel(cd)
        rt["create_channel"] = repr(chan)

        pd = media.dvpp_create_pic_desc()
        rt["create_pic_desc"] = repr(pd)

        rc = media.dvpp_create_resize_config()
        rt["create_resize_config"] = repr(rc)
        r = media.dvpp_set_resize_config_interpolation(rc, 1)
        rt["set_resize_config_interpolation"] = repr(r)

        bc = media.dvpp_create_border_config()
        rt["create_border_config"] = repr(bc)

        # dvpp_malloc 返回形态
        m = media.dvpp_malloc(4096)
        rt["dvpp_malloc"] = repr(m)
    except Exception as e:  # noqa: BLE001
        rt["error"] = f"{type(e).__name__}: {e}"
    report["runtime"] = rt

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    # 4) 顺带验证：acl.rt.malloc 的 device 内存能否直接被 dvpp_pic_desc 吃下
    try:
        import acl as a
        a.init("")
        a.rt.set_device(0)
        ctx, _ = a.rt.create_context(0)
        p, ret = a.rt.malloc(4096, 0)
        ok = media.dvpp_set_pic_desc_data(pd, p)
        print(f"\nacl.rt.malloc ptr as pic data: malloc_ret={ret} set_pic_desc_data_ret={ok}")
        a.rt.free(p)
        a.rt.destroy_context(ctx)
        a.rt.reset_device(0)
        a.finalize()
    except Exception as e:  # noqa: BLE001
        print(f"\npic_desc cross check error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
