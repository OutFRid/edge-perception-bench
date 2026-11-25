# DVPP 生命周期精读 · 310B 上的预处理加速单元

> 来源：R1 遗留前置学习③「DVPP 生命周期精读」，R7 连板实测后成文。
> 配套代码：`scripts/r7_dvpp_probe.py`（能力面探测）、`scripts/r7_dvpp_smoke.py`（VPC resize 冒烟）、
> `scripts/r7_preprocess_bench.py`（DVPP vs OpenCV 对比）、`scripts/r7_multi_stream.py`（并发探测）。
> 数据：`results/doc/r7_preprocess_bench.json`。

---

## 1. 一句话结论

DVPP 是 NPU 上的**图像/视频处理专用加速单元**（VPC 缩放裁剪拼贴、JPEGD/JPEGE 编解码、PNGD、VDEC/VENC 视频编解码）。
它按「**通道 + 描述符 + 异步任务 + 同步栅栏**」组织生命周期：一切 DVPP 操作必须先 `acl.init → set_device → create_context`，
再创建 channel desc 并 `create_channel`，图片用 pic desc 描述（含 stride 对齐），操作以 `_async` 入队、必须 `acl.rt.synchronize_stream` 后才算完成，最后按创建逆序销毁。
**在 310B 上它支持 BGR888 打包格式，且 resize 结果与 OpenCV 双线性逐像素一致（本仓库实测 max_abs_diff=0）。**

---

## 2. DVPP 是什么、能干什么

| 模块 | 能力 | 本仓库实测状态 |
|------|------|------|
| VPC | resize / crop / crop_and_paste / make_border / convert_color / pyr_down | ✅ BGR888 resize+crop_and_paste 实测跑通 |
| JPEGD/JPEGE | JPEG 解码/编码 | 绑定存在，未实测（DMD 源为 AVI） |
| PNGD | PNG 解码 | 绑定存在，未实测 |
| VDEC/VENC | H264/H265 视频编解码 | 绑定存在（vdec_* 全函数面），未实测 |

关键点：DVPP 处理的是 **YUV/RGB 整型图像**，做不了 float 归一化（/255）与 NCHW 转置——这两步永远留在 Host 或交给 AIPP/ATC 编译期算子。

---

## 3. 生命周期全景（对照 C API `acl_dvpp.h`）

```
acl.init("")  ──►  acl.rt.set_device(0)  ──►  acl.rt.create_context(0)
      │
      ├─► dvpp_create_channel_desc()
      │     └─► dvpp_set_channel_desc_mode(desc, DVPP_CHNMODE_VPC=1)   # VPC/JPEGD=2/JPEGE=4/PNGD=8
      │           └─► dvpp_create_channel(desc)   # ★ desc 本身就是通道句柄（非 hi_mpi 的 channel id）
      │
      ├─► 每帧/每图：
      │     dvpp_malloc(size)                  # device 内存（也可 acl.rt.malloc，dvpp_malloc 保证 DVPP 对齐）
      │     acl.rt.memcpy(dev, size, host, size, H2D)
      │     dvpp_create_pic_desc() + dvpp_set_pic_desc_{data,size,format,width,height,width_stride,height_stride}
      │     dvpp_create_resize_config() + dvpp_set_resize_config_interpolation(cfg, 1=双线性)
      │     dvpp_create_roi_config(left, right, top, bottom)   # ★ 参数顺序陷阱见 §4
      │     dvpp_vpc_resize_async(chan, in_pic, out_pic, resize_cfg, stream)   # 异步入队
      │     acl.rt.synchronize_stream(stream)   # ★ 栅栏：不 sync 就读输出 = 脏数据
      │     acl.rt.memcpy(host, size, dev, size, D2H)
      │
      └─► 销毁（逆序）：destroy_pic_desc → destroy_roi_config → destroy_resize_config
            → destroy_channel → destroy_channel_desc → dvpp_free → destroy_stream
            → destroy_context → reset_device → acl.finalize()
```

**两条铁律**（本周踩坑实录）：
1. **不 `acl.init` 就调 DVPP 一切返回错误**：`create_channel_desc` 返回 NULL(0)，`dvpp_malloc` 返回 `(0, 107002)`——CANN 的 107002/100000/500000 都是「运行时未就绪或参数非法」族错误码。
2. **channel desc 即通道句柄**：`dvpp_create_channel(desc)` 只返回错误码 int（不是 tuple），后续 `vpc_resize_async` 第一参直接传这个 desc。试图解包成 `(handle, ret)` 会 TypeError。

---

## 4. Python 绑定形态（本板：acl.media，无独立 acldvpp 包）

- 板端 `import acldvpp` **不存在**；全部 DVPP 函数在 `acl.media` 下（`acl.so` 内置，路径 `.../python/site-packages/acl.so`）。
- 返回形态分三类：
  - 指针型：`dvpp_create_*_desc/config` 返回 **int 指针值**；
  - 带输出参数的错误码型：`dvpp_malloc(size)` 返回 **tuple `(ptr, ret)`**；
  - 纯错误码型：`dvpp_set_*`、`dvpp_*_async`、`dvpp_create_channel` 返回 **int 错误码**。
- **参数顺序陷阱（本周最深的坑）**：`dvpp_create_roi_config(left, right, top, bottom)` 与直觉 `(left, top, right, bottom)` **不同**；且 `right/bottom` 是**含端偏移**（= left+width−1），要求 left/top 为偶数、right/bottom 为奇数（YUV420 约束，BGR888 按 16 字节对齐后同规则可用）。
  错序传参不报错、只返回 100000（ACL_ERROR_INVALID_PARAM），排查了整整一轮（格式怀疑 → 对齐怀疑 → 头文件 + 官方 acllite 样例对拍才锁定）。
- 枚举值本绑定不导出常量，需从 `aarch64-linux/include/acl/ops/acl_dvpp.h` 抄录：`PIXEL_FORMAT_BGR_888=13`、`YUV_SEMIPLANAR_420=1`、`RGB_888=12`；`DVPP_CHNMODE_VPC=1`；`BORDER_CONSTANT=0`；双线性插值 `VPC_PT_LINEAR=1`。

---

## 5. 内存与 stride 对齐（BGR888 打包格式实测）

- BGR888 输入：`width_stride = align_up(width*3, 16)`（字节对齐 16）、`height_stride = align_up(height, 2)`；buffer 大小 = 两 stride 之积。
- Host→Device 拷贝时需把 BGR ndarray 按 stride 逐行铺进对齐缓冲（`in_buf.reshape(hs, ws)[:h, :w*3] = src.reshape(h, w*3)`），否则行首错位、图像撕裂。
- 输出同理：D2H 后按 stride 切片再 reshape 回 `(h, w, 3)`。
- 官方样例（`/opt/samples-master/.../vpc_resnet50_.../acl_dvpp.py`）用 YUV420SP(NV12)：JPEG 解码输出 stride 按 128 对齐，VPC resize 输出仅要求宽高为偶数。

---

## 6. 实测结论（R7-A，DMD 1280×720 → letterbox 640×640，30 帧均值）

| 路径 | 子步骤分解 | 总延时 |
|------|------|---:|
| OpenCV（R2 现状） | resize 3.56 + pad 0.68 + norm 3.20 + transpose 0.10 | **7.55 ms** |
| DVPP 加速版 | h2d 2.15 + vpc_resize 1.24 + vpc_paste 0.73 + d2h 0.48 + host_tail 3.50 | **8.11 ms** |

- 输出张量**逐位一致**（tensor max_abs_diff = 0.0），DVPP 与 OpenCV 双线性在 BGR888 上等价。
- device 侧确实快：resize 1.24ms vs OpenCV 3.56ms（2.9×），paste 0.73ms vs pad 0.68ms 打平。
- 但**总延时 DVPP 反而慢 7%**：为了把图送上 DVPP 再取回，多付出 H2D 2.15 + D2H 0.48 = 2.63ms 往返，恰好吃掉 device 侧省下的 2.32ms。
- **结论**：Host 源（OpenCV decode 的 BGR 帧）前提下，DVPP 预处理在此板无净收益；DVPP 的经典收益场景是**数据已落在 device 内存**（相机直接上板、JPEGD 直出 device、VDEC 直出 device），可省掉整条 BGR 往返。R10 报告将按此口径写「DVPP 何时值得用」。
- 附带数据（R7 探针）：单次 VPC resize 冒烟 6.83ms vs OpenCV 10.55ms（冷缓存单发，稳态见上表）。

---

## 7. 多路并发吞吐（R7-B，yolov8n 纯推理口径）

| 并发路数 | 聚合吞吐 fps | 单路均值 ms | 单路 p95 ms |
|---:|---:|---:|---:|
| 1 | 64.0 | 15.6 | 16.0 |
| 2 | 77.6 | 25.3 | 38.2 |
| 4 | **88.1（峰值）** | 42.4 | 93.3 |
| 8 | 69.8（退化） | 102.3 | 251.4 |

- 聚合吞吐 1→4 路仅 1.38×（远非线性），4 路饱和；8 路**吞吐反而下降 21%**、p95 延时飙到 251ms——单 NPU 计算与访存带宽共享 + 8 个 Python 进程压 4 核 A55 的双重瓶颈。
- 部署建议：310B 上 yolov8n 多路部署的甜区在 **2~4 路**，超 4 路需换更小模型或做 NPU 上多实例隔离。
