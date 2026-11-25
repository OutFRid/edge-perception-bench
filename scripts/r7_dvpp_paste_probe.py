"""R7 · crop_and_paste 最小复现探针：定位 BGR888 失败根因。

试验矩阵：
  T1  BGR888 in -> BGR888 out（paste 到 (0,140)）
  T2  BGR888 in -> BGR888 out（paste 到 (0,0)，排除偏移对齐嫌疑）
  T3  NV12  in -> NV12  out（paste 到 (0,140)，验证 YUV 是否必需）
  T4  BGR888 in -> BGR888 out 但输入直接用原始 1280x720（排除 mid 缓冲嫌疑）
"""
import time

import numpy as np

PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 1
PIXEL_FORMAT_BGR_888 = 13
DVPP_CHNMODE_VPC = 1
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def align_up(x, a):
    return (x + a - 1) // a * a


def setup():
    import acl
    acl.init("")
    acl.rt.set_device(0)
    ctx, ret = acl.rt.create_context(0)
    assert ret == 0, f"create_context {ret}"
    stream, ret = acl.rt.create_stream()
    assert ret == 0, f"create_stream {ret}"
    m = acl.media
    chan_desc = m.dvpp_create_channel_desc()
    assert chan_desc, "channel desc NULL"
    assert m.dvpp_set_channel_desc_mode(chan_desc, DVPP_CHNMODE_VPC) == 0
    assert m.dvpp_create_channel(chan_desc) == 0
    return acl, m, chan_desc, stream, ctx


def make_pic(m, data, size, w, h, ws, hs, fmt):
    pic = m.dvpp_create_pic_desc()
    m.dvpp_set_pic_desc_data(pic, data)
    m.dvpp_set_pic_desc_size(pic, size)
    m.dvpp_set_pic_desc_format(pic, fmt)
    m.dvpp_set_pic_desc_width(pic, w)
    m.dvpp_set_pic_desc_height(pic, h)
    m.dvpp_set_pic_desc_width_stride(pic, ws)
    m.dvpp_set_pic_desc_height_stride(pic, hs)
    return pic


def h2d(acl, dev, size, host_buf):
    return acl.rt.memcpy(dev, size, host_buf.ctypes.data, size, ACL_MEMCPY_HOST_TO_DEVICE)


def test(acl, m, chan, stream, fmt, in_w, in_h, out_w, out_h, paste_l, paste_t,
         label, use_mid=False):
    if fmt == PIXEL_FORMAT_BGR_888:
        in_ws, in_hs = align_up(in_w * 3, 16), align_up(in_h, 2)
        out_ws, out_hs = align_up(out_w * 3, 16), align_up(out_h, 2)
        in_size = in_ws * in_hs
        out_size = out_ws * out_hs
        rng = np.random.default_rng(3)
        src = rng.integers(0, 256, (in_h, in_w, 3), dtype=np.uint8)
        in_host = np.zeros(in_size, dtype=np.uint8)
        in_host.reshape(in_hs, in_ws)[:in_h, :in_w * 3] = src.reshape(in_h, in_w * 3)
        cw, ch = in_w, in_h
    else:  # NV12
        in_ws, in_hs = align_up(in_w, 16), align_up(in_h, 2)
        out_ws, out_hs = align_up(out_w, 16), align_up(out_h, 2)
        in_size = in_ws * in_hs * 3 // 2
        out_size = out_ws * out_hs * 3 // 2
        rng = np.random.default_rng(3)
        y = rng.integers(0, 256, (in_h, in_w), dtype=np.uint8)
        uv = rng.integers(0, 256, (in_h // 2, in_w), dtype=np.uint8)
        in_host = np.zeros(in_size, dtype=np.uint8)
        plane = in_host.reshape(in_hs * 3 // 2, in_ws)  # NV12：Y 面 + 交错 UV 面
        plane[:in_h, :in_w] = y
        plane[in_hs:in_hs + in_h // 2, :in_w] = uv
        cw, ch = in_w, in_h

    in_dev, ret = m.dvpp_malloc(in_size)
    assert ret == 0, f"dvpp_malloc(in) {ret}"
    out_dev, ret = m.dvpp_malloc(out_size)
    assert ret == 0, f"dvpp_malloc(out) {ret}"
    assert h2d(acl, in_dev, in_size, in_host) == 0

    if use_mid:
        # 先把输入 resize 到 (cw_out, ch_out) 再 crop_and_paste（复现 mid 缓冲嫌疑）
        mid_w, mid_h = 640, 360
        mid_ws, mid_hs = align_up(mid_w * 3, 16), align_up(mid_h, 2)
        mid_size = mid_ws * mid_hs
        mid_dev, ret = m.dvpp_malloc(mid_size)
        assert ret == 0
        in_pic = make_pic(m, in_dev, in_size, in_w, in_h, in_ws, in_hs, fmt)
        mid_pic = make_pic(m, mid_dev, mid_size, mid_w, mid_h, mid_ws, mid_hs, fmt)
        rcfg = m.dvpp_create_resize_config()
        m.dvpp_set_resize_config_interpolation(rcfg, 1)
        r = m.dvpp_vpc_resize_async(chan, in_pic, mid_pic, rcfg, stream)
        assert r == 0, f"resize {r}"
        assert acl.rt.synchronize_stream(stream) == 0
        src_pic, sw, sh, sws, shs = mid_pic, mid_w, mid_h, mid_ws, mid_hs
    else:
        src_pic, sw, sh, sws, shs = make_pic(m, in_dev, in_size, in_w, in_h, in_ws, in_hs, fmt), \
            in_w, in_h, in_ws, in_hs

    # 预填充输出缓冲（灰）
    out_host = np.full(out_size, 114, dtype=np.uint8)
    assert h2d(acl, out_dev, out_size, out_host) == 0
    out_pic = make_pic(m, out_dev, out_size, out_w, out_h, out_ws, out_hs, fmt)

    crop_roi = m.dvpp_create_roi_config(0, 0, sw, sh)
    paste_roi = m.dvpp_create_roi_config(paste_l, paste_t, paste_l + sw, paste_t + sh)

    r = m.dvpp_vpc_crop_and_paste_async(chan, src_pic, out_pic, crop_roi, paste_roi, stream)
    if r != 0:
        print(f"{label}: crop_and_paste enqueue ret={r}")
    else:
        r2 = acl.rt.synchronize_stream(stream)
        print(f"{label}: enqueue OK, sync ret={r2}")

    # 清理
    m.dvpp_free(in_dev)
    m.dvpp_free(out_dev)
    if use_mid:
        m.dvpp_free(mid_dev)
        m.dvpp_destroy_resize_config(rcfg)
        m.dvpp_destroy_pic_desc(in_pic)
        m.dvpp_destroy_pic_desc(mid_pic)
    else:
        m.dvpp_destroy_pic_desc(src_pic)
    m.dvpp_destroy_pic_desc(out_pic)
    m.dvpp_destroy_roi_config(crop_roi)
    m.dvpp_destroy_roi_config(paste_roi)


def main():
    acl, m, chan, stream, ctx = setup()
    try:
        test(acl, m, chan, stream, PIXEL_FORMAT_BGR_888, 640, 360, 640, 640, 0, 140, "T1 BGR->BGR (0,140)")
        test(acl, m, chan, stream, PIXEL_FORMAT_BGR_888, 640, 360, 640, 640, 0, 0, "T2 BGR->BGR (0,0)")
        test(acl, m, chan, stream, PIXEL_FORMAT_YUV_SEMIPLANAR_420, 640, 360, 640, 640, 0, 140, "T3 NV12->NV12 (0,140)")
        test(acl, m, chan, stream, PIXEL_FORMAT_BGR_888, 1280, 720, 640, 640, 0, 140, "T4 BGR 1280x720 直接 (0,140)")
        test(acl, m, chan, stream, PIXEL_FORMAT_BGR_888, 1280, 720, 640, 640, 0, 140, "T5 BGR mid-resize 复现", use_mid=True)
    finally:
        m.dvpp_destroy_channel(chan)
        m.dvpp_destroy_channel_desc(chan)
        acl.rt.destroy_stream(stream)
        acl.rt.destroy_context(ctx)
        acl.rt.reset_device(0)
        acl.finalize()


if __name__ == "__main__":
    main()
