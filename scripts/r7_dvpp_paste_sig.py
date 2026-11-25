"""R7 · crop_and_paste 调用形态深探：绑定期望参数个数 / roi 返回值类型。"""
import numpy as np

PIXEL_FORMAT_BGR_888 = 13
DVPP_CHNMODE_VPC = 1
ACL_MEMCPY_HOST_TO_DEVICE = 1


def align_up(x, a):
    return (x + a - 1) // a * a


def main():
    import acl
    acl.init("")
    acl.rt.set_device(0)
    ctx, ret = acl.rt.create_context(0)
    assert ret == 0
    stream, ret = acl.rt.create_stream()
    assert ret == 0
    m = acl.media
    chan = m.dvpp_create_channel_desc()
    assert m.dvpp_set_channel_desc_mode(chan, DVPP_CHNMODE_VPC) == 0
    assert m.dvpp_create_channel(chan) == 0

    # 1) roi config 返回值形态
    roi = m.dvpp_create_roi_config(0, 0, 640, 360)
    print(f"create_roi_config -> {roi!r} type={type(roi)}")
    roi2 = m.dvpp_create_roi_config(0, 0, 640, 360)
    r = m.dvpp_set_roi_config(roi2, 0, 140, 640, 500)
    print(f"dvpp_set_roi_config -> {r!r}")

    # 2) 绑定期望参数个数（不用 dummy 指针，只用 0 探「参数个数」报错，避免段错误）
    for label, args in (
        ("5args(无stream)", (chan, 0, 0, roi, roi)),
        ("6args(带stream)", (chan, 0, 0, roi, roi, stream)),
    ):
        try:
            ret = m.dvpp_vpc_crop_and_paste_async(*args)
            print(f"crop_and_paste {label} -> {ret!r}", flush=True)
        except TypeError as e:
            print(f"crop_and_paste {label} -> TypeError: {e}", flush=True)

    # 3) resize 绑定签名对照（已知能跑；只查参数个数，用 0 指针）
    try:
        ret = m.dvpp_vpc_resize_async(chan, 0, 0, 0, stream)
        print(f"resize 5args -> {ret!r}", flush=True)
    except TypeError as e:
        print(f"resize 5args -> TypeError: {e}", flush=True)

    # 4) 用 dummy pic desc 试 crop_and_paste，看是否参数校验（非法指针）100000 与合法指针不同码
    pic = m.dvpp_create_pic_desc()
    print(f"create_pic_desc -> {pic!r}")
    in_dev, r = m.dvpp_malloc(4096)
    print(f"dvpp_malloc -> ({in_dev!r}, {r})")
    print(f"set_pic_desc_data(pic, in_dev) -> {m.dvpp_set_pic_desc_data(pic, in_dev)!r}")
    print(f"set_pic_desc_size -> {m.dvpp_set_pic_desc_size(pic, 4096)!r}")
    print(f"set_pic_desc_format -> {m.dvpp_set_pic_desc_format(pic, PIXEL_FORMAT_BGR_888)!r}")
    print(f"set_pic_desc_width -> {m.dvpp_set_pic_desc_width(pic, 640)!r}")
    print(f"set_pic_desc_height -> {m.dvpp_set_pic_desc_height(pic, 360)!r}")
    print(f"set_pic_desc_width_stride -> {m.dvpp_set_pic_desc_width_stride(pic, 1920)!r}")
    print(f"set_pic_desc_height_stride -> {m.dvpp_set_pic_desc_height_stride(pic, 360)!r}")

    pic2 = m.dvpp_create_pic_desc()
    out_dev, r = m.dvpp_malloc(4096 * 4)
    print(f"dvpp_malloc(out) -> ({out_dev!r}, {r})")
    m.dvpp_set_pic_desc_data(pic2, out_dev)
    m.dvpp_set_pic_desc_size(pic2, 4096 * 4)
    m.dvpp_set_pic_desc_format(pic2, PIXEL_FORMAT_BGR_888)
    m.dvpp_set_pic_desc_width(pic2, 640)
    m.dvpp_set_pic_desc_height(pic2, 640)
    m.dvpp_set_pic_desc_width_stride(pic2, 1920)
    m.dvpp_set_pic_desc_height_stride(pic2, 640)

    for label, args in (
        ("合法pic/roi/stream", (chan, pic, pic2, roi, roi, stream)),
    ):
        ret = m.dvpp_vpc_crop_and_paste_async(*args)
        print(f"crop_and_paste {label} -> {ret!r}")

    m.dvpp_destroy_channel(chan)
    m.dvpp_destroy_channel_desc(chan)
    acl.rt.destroy_stream(stream)
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()


if __name__ == "__main__":
    main()
