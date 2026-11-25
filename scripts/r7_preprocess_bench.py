"""R7 · 预处理延时对比：DVPP VPC vs OpenCV（letterbox 640，DMD 真实帧）。

对比口径（对齐 R2 的 LetterboxPreprocess，1280x720 -> 640x640x3 NCHW float）：
  A. OpenCV 路径（R2 现状）：
       cv2.resize -> canvas 补边 -> BGR2RGB + /255 -> transpose -> expand_dims
  B. DVPP 路径（本周新增）：
       H2D(帧) -> vpc_resize 1280x720->640x360 -> crop_and_paste 贴进预填充
       640x640 灰底 -> D2H -> host 尾部(BGR2RGB+/255+transpose)
  两条路径都逐子步骤计时，并做输出 tensor 一致性校验（max_abs_diff）。

用法（板端）:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 r7_preprocess_bench.py \
        --video "$(ls /mnt/e61/workspace/dmd/gA/1/s1/*_mosaic.avi | head -1)" \
        --frames 30 --json /root/epb/results/r7_preprocess_bench.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---- acldvpp 枚举（板端 acl_dvpp.h 抄录）----
PIXEL_FORMAT_BGR_888 = 13
DVPP_CHNMODE_VPC = 1
VPC_PT_LINEAR = 1  # 双线性插值

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

IMGSZ = 640
PAD_COLOR = 114


def align_up(x, a):
    return (x + a - 1) // a * a


def opencv_preprocess(frame):
    """与 R2 LetterboxPreprocess 完全一致，返回 (tensor, 子步骤耗时 ms dict)。"""
    t = {}
    t0 = time.perf_counter()
    h0, w0 = frame.shape[:2]
    r = min(IMGSZ / h0, IMGSZ / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    t["resize"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    canvas = np.full((IMGSZ, IMGSZ, 3), PAD_COLOR, dtype=np.uint8)
    left, top = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    t["pad"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t["norm"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    img = np.transpose(img, (2, 0, 1))
    tensor = np.expand_dims(img, 0)
    t["transpose"] = (time.perf_counter() - t0) * 1000
    return tensor, t


class DvppLetterbox:
    """DVPP 版 letterbox：VPC resize + crop_and_paste 进预填充灰底，host 只留归一化尾部。"""

    def __init__(self, in_w, in_h):
        import acl
        self.acl = acl
        self.m = acl.media
        acl.init("")
        acl.rt.set_device(0)
        self.ctx, ret = acl.rt.create_context(0)
        assert ret == 0, f"create_context {ret}"
        self.stream, ret = acl.rt.create_stream()
        assert ret == 0, f"create_stream {ret}"

        self.in_w, self.in_h = in_w, in_h
        self.r = min(IMGSZ / in_h, IMGSZ / in_w)
        self.nw, self.nh = int(round(in_w * self.r)), int(round(in_h * self.r))
        self.left = (IMGSZ - self.nw) // 2
        self.top = (IMGSZ - self.nh) // 2

        # ---- channel（VPC 模式）----
        chan_desc = self.m.dvpp_create_channel_desc()
        assert chan_desc, "create_channel_desc NULL"
        assert self.m.dvpp_set_channel_desc_mode(chan_desc, DVPP_CHNMODE_VPC) == 0
        assert self.m.dvpp_create_channel(chan_desc) == 0
        self.chan_desc = chan_desc

        # ---- resize config ----
        self.resize_cfg = self.m.dvpp_create_resize_config()
        assert self.resize_cfg, "create_resize_config NULL"
        assert self.m.dvpp_set_resize_config_interpolation(self.resize_cfg, VPC_PT_LINEAR) == 0

        # ---- roi config（参数顺序 left, right, top, bottom；right/bottom 为含端偏移需奇数）----
        self.crop_roi = self.m.dvpp_create_roi_config(0, self.nw - 1, 0, self.nh - 1)
        self.paste_roi = self.m.dvpp_create_roi_config(
            self.left, self.left + self.nw - 1, self.top, self.top + self.nh - 1)
        assert self.crop_roi and self.paste_roi, "create_roi_config NULL"

        # ---- 内存布局（BGR888 打包：width_stride 按字节 16 对齐）----
        self.in_ws, self.in_hs = align_up(in_w * 3, 16), align_up(in_h, 2)
        self.mid_ws, self.mid_hs = align_up(self.nw * 3, 16), align_up(self.nh, 2)
        self.out_ws, self.out_hs = align_up(IMGSZ * 3, 16), align_up(IMGSZ, 2)
        self.in_size = self.in_ws * self.in_hs
        self.mid_size = self.mid_ws * self.mid_hs
        self.out_size = self.out_ws * self.out_hs

        self.in_dev, ret = self.m.dvpp_malloc(self.in_size)
        assert ret == 0, f"dvpp_malloc(in) {ret}"
        self.mid_dev, ret = self.m.dvpp_malloc(self.mid_size)
        assert ret == 0, f"dvpp_malloc(mid) {ret}"
        self.out_dev, ret = self.m.dvpp_malloc(self.out_size)
        assert ret == 0, f"dvpp_malloc(out) {ret}"

        # ---- pic desc ----
        self.in_pic = self.m.dvpp_create_pic_desc()
        self.mid_pic = self.m.dvpp_create_pic_desc()
        self.out_pic = self.m.dvpp_create_pic_desc()

        # ---- 预填充输出灰底（只做一次）----
        canvas_host = np.full((IMGSZ, IMGSZ, 3), PAD_COLOR, dtype=np.uint8)
        self.out_buf_host = np.zeros(self.out_size, dtype=np.uint8)
        self.out_buf_host.reshape(self.out_hs, self.out_ws)[:IMGSZ, :IMGSZ * 3] = \
            canvas_host.reshape(IMGSZ, IMGSZ * 3)
        assert self.acl.rt.memcpy(self.out_dev, self.out_size, self.out_buf_host.ctypes.data,
                                  self.out_size, ACL_MEMCPY_HOST_TO_DEVICE) == 0

    def _set_pic(self, pic, data, size, w, h, ws, hs):
        m = self.m
        assert m.dvpp_set_pic_desc_data(pic, data) == 0
        assert m.dvpp_set_pic_desc_size(pic, size) == 0
        assert m.dvpp_set_pic_desc_format(pic, PIXEL_FORMAT_BGR_888) == 0
        assert m.dvpp_set_pic_desc_width(pic, w) == 0
        assert m.dvpp_set_pic_desc_height(pic, h) == 0
        assert m.dvpp_set_pic_desc_width_stride(pic, ws) == 0
        assert m.dvpp_set_pic_desc_height_stride(pic, hs) == 0

    def __call__(self, frame):
        """BGR frame -> (tensor, 子步骤耗时 ms dict)。"""
        t = {}
        acl, m = self.acl, self.m

        t0 = time.perf_counter()
        in_buf = np.zeros(self.in_size, dtype=np.uint8)
        in_buf.reshape(self.in_hs, self.in_ws)[: self.in_h, : self.in_w * 3] = \
            frame.reshape(self.in_h, self.in_w * 3)
        assert acl.rt.memcpy(self.in_dev, self.in_size, in_buf.ctypes.data,
                             self.in_size, ACL_MEMCPY_HOST_TO_DEVICE) == 0
        t["h2d"] = (time.perf_counter() - t0) * 1000

        self._set_pic(self.in_pic, self.in_dev, self.in_size, self.in_w, self.in_h,
                      self.in_ws, self.in_hs)
        self._set_pic(self.mid_pic, self.mid_dev, self.mid_size, self.nw, self.nh,
                      self.mid_ws, self.mid_hs)

        t0 = time.perf_counter()
        assert m.dvpp_vpc_resize_async(self.chan_desc, self.in_pic, self.mid_pic,
                                       self.resize_cfg, self.stream) == 0
        assert acl.rt.synchronize_stream(self.stream) == 0
        t["vpc_resize"] = (time.perf_counter() - t0) * 1000

        self._set_pic(self.out_pic, self.out_dev, self.out_size, IMGSZ, IMGSZ,
                      self.out_ws, self.out_hs)

        t0 = time.perf_counter()
        r = m.dvpp_vpc_crop_and_paste_async(self.chan_desc, self.mid_pic, self.out_pic,
                                            self.crop_roi, self.paste_roi, self.stream)
        if r != 0:
            raise RuntimeError(f"vpc_crop_and_paste_async ret={r}")
        r = acl.rt.synchronize_stream(self.stream)
        if r != 0:
            raise RuntimeError(f"synchronize_stream ret={r}")
        t["vpc_paste"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        assert acl.rt.memcpy(self.out_buf_host.ctypes.data, self.out_size, self.out_dev,
                             self.out_size, ACL_MEMCPY_DEVICE_TO_HOST) == 0
        canvas = (self.out_buf_host.reshape(self.out_hs, self.out_ws)[:IMGSZ, :IMGSZ * 3]
                  .reshape(IMGSZ, IMGSZ, 3))
        t["d2h"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        tensor = np.expand_dims(img, 0)
        t["host_tail"] = (time.perf_counter() - t0) * 1000
        return tensor, t

    def destroy(self):
        m, acl = self.m, self.acl
        m.dvpp_free(self.in_dev)
        m.dvpp_free(self.mid_dev)
        m.dvpp_free(self.out_dev)
        m.dvpp_destroy_pic_desc(self.in_pic)
        m.dvpp_destroy_pic_desc(self.mid_pic)
        m.dvpp_destroy_pic_desc(self.out_pic)
        m.dvpp_destroy_roi_config(self.crop_roi)
        m.dvpp_destroy_roi_config(self.paste_roi)
        m.dvpp_destroy_resize_config(self.resize_cfg)
        m.dvpp_destroy_channel(self.chan_desc)
        m.dvpp_destroy_channel_desc(self.chan_desc)
        acl.rt.destroy_stream(self.stream)
        acl.rt.destroy_context(self.ctx)
        acl.rt.reset_device(0)
        acl.finalize()


def stat(x):
    return {"mean": round(statistics.mean(x), 3), "median": round(statistics.median(x), 3),
            "p95": round(sorted(x)[int(len(x) * 0.95) - 1], 3), "min": round(min(x), 3),
            "max": round(max(x), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--json", help="dump results to json")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"cannot open video: {args.video}")
    frames = []
    for _ in range(args.frames + args.warmup):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) <= args.warmup:
        sys.exit(f"not enough frames: {len(frames)}")
    h, w = frames[0].shape[:2]
    print(f"Video: {Path(args.video).name}  {w}x{h}  frames={len(frames)}")

    dvpp = DvppLetterbox(w, h)
    try:
        acc_a = {k: [] for k in ("resize", "pad", "norm", "transpose")}
        acc_b = {k: [] for k in ("h2d", "vpc_resize", "vpc_paste", "d2h", "host_tail")}
        diffs = []

        for i, frame in enumerate(frames):
            ta, ta_sub = opencv_preprocess(frame)
            tb, tb_sub = dvpp(frame)
            for k in acc_a:
                acc_a[k].append(ta_sub[k])
            for k in acc_b:
                acc_b[k].append(tb_sub[k])
            if i >= args.warmup:  # 校验只统计稳态
                diffs.append(float(np.abs(ta - tb).max()))

        a_tot = {k: stat(v) for k, v in acc_a.items()}
        b_tot = {k: stat(v) for k, v in acc_b.items()}
        report = {
            "video": str(args.video),
            "frames": len(frames),
            "size": [w, h],
            "opencv": {**a_tot, "total_mean_ms": round(sum(v["mean"] for v in a_tot.values()), 3)},
            "dvpp": {**b_tot, "total_mean_ms": round(sum(v["mean"] for v in b_tot.values()), 3)},
            "tensor_max_abs_diff": round(max(diffs), 5) if diffs else None,
            "speedup_vs_opencv": round(
                sum(v["mean"] for v in a_tot.values()) / max(sum(v["mean"] for v in b_tot.values()), 1e-9), 3),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))

        print("\n| 阶段 | OpenCV ms | DVPP ms |")
        print("|---|---:|---:|")
        for k in ("resize", "pad", "norm", "transpose"):
            print(f"| cv.{k} | {a_tot[k]['mean']:.2f} | — |")
        for k in ("h2d", "vpc_resize", "vpc_paste", "d2h", "host_tail"):
            print(f"| dvpp.{k} | — | {b_tot[k]['mean']:.2f} |")
        print(f"| **合计** | **{report['opencv']['total_mean_ms']}** | "
              f"**{report['dvpp']['total_mean_ms']}** |")
        print(f"| 输出张量最大绝对差 | {report['tensor_max_abs_diff']} | |")

        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
            print(f"\n-> {args.json}")
    finally:
        dvpp.destroy()


if __name__ == "__main__":
    main()
