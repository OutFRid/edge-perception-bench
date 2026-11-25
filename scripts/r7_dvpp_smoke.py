"""R7 · DVPP VPC 端到端冒烟：BGR888 输入 → vpc_resize → 输出，验证全链路。

验证点：
  1. 正确顺序下（acl.init → set_device → create_context）DVPP 通道/描述符能否创建；
  2. dvpp_vpc_resize_async 的 Python 调用形态（channel desc / pic desc / resize config / stream）；
  3. 输出内容与 OpenCV resize 结果逐像素比对（数值应近似）；
  4. BGR888 打包格式在 310B 上的 stride 对齐要求。

用法（板端）:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 r7_dvpp_smoke.py --width 1280 --height 720 --out-size 640x360
"""
import argparse
import json
import sys
import time

import cv2
import numpy as np

# ---- C 头文件 acldvpp 枚举值（板端 acl_dvpp.h 抄录）----
PIXEL_FORMAT_BGR_888 = 13
PIXEL_FORMAT_RGB_888 = 12
DVPP_CHNMODE_VPC = 1
BORDER_CONSTANT = 0
VPC_PT_LINEAR = 1  # 双线性插值（hi_dvpp_vpc.h: VpcPointType）

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def align_up(x, a):
    return (x + a - 1) // a * a


class DvppVpc:
    """最小 VPC 封装：一个 resize 通道 + 输入/输出 pic desc 生命周期。"""

    def __init__(self, in_w, in_h, out_w, out_h, fmt_in=PIXEL_FORMAT_BGR_888,
                 fmt_out=PIXEL_FORMAT_BGR_888):
        import acl
        self.acl = acl
        self.media = acl.media
        acl.init("")
        acl.rt.set_device(0)
        self.ctx, ret = acl.rt.create_context(0)
        if ret != 0:
            raise RuntimeError(f"create_context failed: {ret}")
        self.stream, ret = acl.rt.create_stream()
        if ret != 0:
            raise RuntimeError(f"create_stream failed: {ret}")

        # ---- channel ----
        # 注意：CANN ACL API 中 channel desc 本身就是通道句柄，
        # dvpp_create_channel 只返回错误码 int（不像 hi_mpi 返回 channel id）。
        self.chan_desc = self.media.dvpp_create_channel_desc()
        assert self.chan_desc, f"create_channel_desc -> {self.chan_desc}"
        r = self.media.dvpp_set_channel_desc_mode(self.chan_desc, DVPP_CHNMODE_VPC)
        assert r == 0, f"set_channel_desc_mode -> {r}"
        r = self.media.dvpp_create_channel(self.chan_desc)
        assert r == 0, f"create_channel -> {r}"
        self.channel = self.chan_desc

        # ---- resize config ----
        self.resize_cfg = self.media.dvpp_create_resize_config()
        assert self.resize_cfg, f"create_resize_config -> {self.resize_cfg}"
        r = self.media.dvpp_set_resize_config_interpolation(self.resize_cfg, VPC_PT_LINEAR)
        assert r == 0, f"set_resize_config_interpolation -> {r}"

        # ---- 输入/输出 pic desc（data 在每次 resize 前 set）----
        self.in_pic = self.media.dvpp_create_pic_desc()
        self.out_pic = self.media.dvpp_create_pic_desc()
        assert self.in_pic and self.out_pic, "create_pic_desc NULL"

        self.in_w, self.in_h, self.out_w, self.out_h = in_w, in_h, out_w, out_h
        # BGR888 打包：width_stride 按字节 16 对齐，height_stride 2 对齐
        self.in_ws = align_up(in_w * 3, 16)
        self.in_hs = align_up(in_h, 2)
        self.out_ws = align_up(out_w * 3, 16)
        self.out_hs = align_up(out_h, 2)
        self.in_size = self.in_ws * self.in_hs
        self.out_size = self.out_ws * self.out_hs

        # ---- device 内存 ----
        self.in_dev, ret = self.media.dvpp_malloc(self.in_size)
        assert ret == 0, f"dvpp_malloc(in) -> {self.in_dev!r}"
        self.out_dev, ret = self.media.dvpp_malloc(self.out_size)
        assert ret == 0, f"dvpp_malloc(out) -> {self.out_dev!r}"

    def _set_pic(self, pic, data, size, w, h, ws, hs, fmt):
        m = self.media
        for r, name in ((m.dvpp_set_pic_desc_data(pic, data), "data"),
                        (m.dvpp_set_pic_desc_size(pic, size), "size"),
                        (m.dvpp_set_pic_desc_format(pic, fmt), "format"),
                        (m.dvpp_set_pic_desc_width(pic, w), "width"),
                        (m.dvpp_set_pic_desc_height(pic, h), "height"),
                        (m.dvpp_set_pic_desc_width_stride(pic, ws), "width_stride"),
                        (m.dvpp_set_pic_desc_height_stride(pic, hs), "height_stride")):
            assert r == 0, f"set_pic_desc_{name} -> {r}"

    def resize(self, bgr_host):
        """BGR888 Host ndarray (H,W,3) -> BGR888 Host ndarray（resize 到 out 尺寸）。"""
        acl = self.acl
        h, w = bgr_host.shape[:2]
        assert (w, h) == (self.in_w, self.in_h), f"{(w, h)} != {(self.in_w, self.in_h)}"
        src = np.ascontiguousarray(bgr_host)
        # H2D：逐行拷，行内补 stride 对齐（src 展平为 (h, w*3) 后再贴）
        in_buf = np.zeros(self.in_size, dtype=np.uint8)
        in_buf.reshape(self.in_hs, self.in_ws)[:h, : w * 3] = src.reshape(h, w * 3)
        r = acl.rt.memcpy(self.in_dev, self.in_size, in_buf.ctypes.data, self.in_size,
                          ACL_MEMCPY_HOST_TO_DEVICE)
        assert r == 0, f"memcpy H2D -> {r}"

        self._set_pic(self.in_pic, self.in_dev, self.in_size, w, h, self.in_ws, self.in_hs,
                      PIXEL_FORMAT_BGR_888)
        self._set_pic(self.out_pic, self.out_dev, self.out_size, self.out_w, self.out_h,
                      self.out_ws, self.out_hs, PIXEL_FORMAT_BGR_888)

        r = self.media.dvpp_vpc_resize_async(self.channel, self.in_pic, self.out_pic,
                                             self.resize_cfg, self.stream)
        assert r == 0, f"vpc_resize_async -> {r}"
        r = acl.rt.synchronize_stream(self.stream)
        assert r == 0, f"synchronize_stream -> {r}"

        out_buf = np.zeros(self.out_size, dtype=np.uint8)
        r = acl.rt.memcpy(out_buf.ctypes.data, self.out_size, self.out_dev, self.out_size,
                          ACL_MEMCPY_DEVICE_TO_HOST)
        assert r == 0, f"memcpy D2H -> {r}"
        return (out_buf.reshape(self.out_hs, self.out_ws)[: self.out_h, : self.out_w * 3]
                .reshape(self.out_h, self.out_w, 3).copy())

    def destroy(self):
        m = self.media
        acl = self.acl
        m.dvpp_free(self.in_dev)
        m.dvpp_free(self.out_dev)
        m.dvpp_destroy_resize_config(self.resize_cfg)
        m.dvpp_destroy_pic_desc(self.in_pic)
        m.dvpp_destroy_pic_desc(self.out_pic)
        m.dvpp_destroy_channel(self.channel)
        m.dvpp_destroy_channel_desc(self.chan_desc)
        acl.rt.destroy_stream(self.stream)
        acl.rt.destroy_context(self.ctx)
        acl.rt.reset_device(0)
        acl.finalize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out-size", default="640x360")
    args = ap.parse_args()
    ow, oh = [int(x) for x in args.out_size.split("x")]

    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)

    vpc = DvppVpc(args.width, args.height, ow, oh)
    try:
        t0 = time.perf_counter()
        dvpp_out = vpc.resize(frame)
        t_dvpp = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cv_out = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_LINEAR)
        t_cv = (time.perf_counter() - t0) * 1000

        diff = np.abs(dvpp_out.astype(np.int16) - cv_out.astype(np.int16))
        report = {
            "dvpp_ms": round(t_dvpp, 3),
            "cv_ms": round(t_cv, 3),
            "dvpp_shape": list(dvpp_out.shape),
            "cv_shape": list(cv_out.shape),
            "max_abs_diff": int(diff.max()),
            "mean_abs_diff": round(float(diff.mean()), 3),
            "pixels_mismatch_gt2": int((diff > 2).sum()),
        }
        print(json.dumps(report, indent=2))
        print(f"\nDVPP first row[:30]: {dvpp_out[0,:30,0].tolist()}")
        print(f"CV   first row[:30]: {cv_out[0,:30,0].tolist()}")
    finally:
        vpc.destroy()


if __name__ == "__main__":
    main()
