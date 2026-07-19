# DMD 数据集 + YOLOv8-nano ONNX → OM → ACL 第一帧推理记录

> 日期: 2026-07-19 ~ 2026-07-20
> 硬件: Huawei Atlas 200I DK (Ascend 310B4, 3.4G DDR, 8G eMMC swap)
> 系统: CANN 7.0.RC1, Python 3.9.2
> 数据集: DMD (Driver Monitoring Dataset) — distraction + drowsiness + gaze

---

## 1. 数据集解压

36 个 `.tar.gz` 包（总计约 150GB）解压到 1TB 移动 SSD (闪迪 E61)：

```bash
mount /dev/sda1 /mnt/e61
cd /mnt/e61/workspace
for f in dmd/*.tar.gz; do tar -xzf "$f" & done; wait
```

DMD 三个子任务（distraction / drowsiness / gaze）共享同一批视频文件，解压后合并为同一个目录树。每个 session 包含：

```
dmd/gA/1/s6/
  *_rgb_mosaic.avi      # 多视角拼接视频
  *_rgb_face.mp4        # 人脸视频 ← 人脸检测/视线估计用这个
  *_rgb_body.mp4        # 身体视频
  *_rgb_hands.mp4       # 手势视频 ← 手部检测用这个
  *_ann_gaze.json       # 视线标注
  *_ann_hands.json      # 手势标注
  *_ann_distraction.json / *_ann_drowsiness.json  # 分心/疲劳标注（不是每个 session 都有）
```

解压完成后删除 `.tar.gz` 释放约 150GB 空间：

```bash
rm /mnt/e61/workspace/dmd/*.tar.gz
```

---

## 2. YOLOv8-nano → ONNX 导出

**环境**: Windows 本地 conda `torchGPU` (Python 3.12, PyTorch 2.6+cu124)

**脚本**: `edge-perception-bench/scripts/export_yolov8_onnx.py`

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="onnx", opset=11, imgsz=640, simplify=True, half=False, dynamic=False, batch=1)
```

**产物**:

| 文件 | 大小 | 说明 |
|------|------|------|
| `models/yolov8n.onnx` | 12.3 MB | IR v6, Opset 11, Input: `images [1,3,640,640]`, Output: `output0 [1,84,8400]` |

---

## 3. ONNX → OM (离线模型编译)

### 3.1 板端编译（失败）

**原因**: 板子 3.4G DDR，Ascend310B4 编译需 2-3GB 峰值内存，DDR 不足导致 swap 颠簸卡死。

```
ATC run failed
→ swap 涨到 7.4/8.0 GB  → kill 进程
→ 单线程 OMP_NUM_THREADS=1  → 仍然 swap 爆炸
```

### 3.2 WSL2 交叉编译（成功）

**环境**: Windows WSL2 Ubuntu 22.04 x86_64

下载 CANN 7.0.1 BETA1 (与板端 7.0.RC1 同大版本，兼容) → 安装到 `/root/Ascend`

**踩坑记录**:

| 错误 | 原因 | 解决 |
|------|------|------|
| `/root` permission too small | WSL2 默认 700 | `chmod 755 /root` |
| `ModuleNotFoundError: numpy` | WSL2 无 Python 依赖 | `apt install python3-pip python3-numpy` |
| `AttributeError: np.float_ removed` | NumPy 2.x 不兼容 CANN 7.0 | `pip install "numpy<2"` |
| `libascend_hal.so not found` | set_env.sh 未设 stub 路径 | 手动 export LD_LIBRARY_PATH |

**编译命令**:

```bash
source /root/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/root/Ascend/ascend-toolkit/latest/runtime/lib64/stub:\
/root/Ascend/ascend-toolkit/latest/x86_64-linux/devlib:$LD_LIBRARY_PATH

atc --model=models/yolov8n.onnx --framework=5 --output=models/yolov8n \
    --soc_version=Ascend310B4 --input_format=NCHW \
    --input_shape="images:1,3,640,640"
```

**产物**:

| 文件 | 大小 | 说明 |
|------|------|------|
| `models/yolov8n.om` | 7.0 MB | Ascend 310B4 离线模型，算子融合后压缩 43% |

---

## 4. ACL Python 推理

### 4.1 AC (Ascend Tensor Compiler) 与 ACL (Ascend Computing Language)

| | ATC | ACL |
|---|---|---|
| 角色 | 编译器 (类比 gcc) | 运行时 SDK (类比 CUDA Runtime) |
| 输入 | ONNX/TF/PyTorch | OM 离线模型 |
| 输出 | `.om` | 推理结果 |
| 类比 | `gcc main.c -o a.out` | `./a.out` |

### 4.2 CANN 7.0.RC1 Python API 坑 (重要！)

CANN 不同版本 Python ACL API 签名差异大，本次实例总结：

```python
import acl

# ✅ 正确签名
acl.init("")                                              # str
acl.rt.set_device(0)                                      # int → ret
ctx, ret = acl.rt.create_context(0)                       # int → (context, ret)
mid, ret = acl.mdl.load_from_file("model.om")            # str → (model_id, ret)
desc = acl.mdl.create_desc()                              # desc
acl.mdl.get_desc(desc, mid)                               # → ret

# ⚠️ create_data_buffer: CANN 7.0 是 (ptr, size)，不是 (size, type)!
buf = acl.create_data_buffer(dev_ptr, size)               # (int, int) → buf_id
addr = acl.get_data_buffer_addr(buf)                      # buf_id → int
buf_size = acl.get_data_buffer_size(buf)                  # buf_id → int

# ⚠️ add_dataset_buffer 返回 tuple!
_, ret = acl.mdl.add_dataset_buffer(ds, buf)              # → (buf_handle, ret)

# ⚠️ 必须先用 rt.malloc 分配设备内存
dev_ptr, ret = acl.rt.malloc(size, 0)                     # → (ptr, ret)
ret = acl.rt.memcpy(dst, dst_max, src, src_max, kind)     # 参数都是 int
acl.rt.free(dev_ptr)                                      # 释放设备内存

# 推理
ret = acl.mdl.execute(model_id, in_ds, out_ds)            # → ret
acl.rt.synchronize_stream(0)
```

### 4.3 完整推理流程

```
1. acl.init("")
2. acl.rt.set_device(0)
3. acl.rt.create_context(0)
4. acl.mdl.load_from_file(om_path)
5. acl.mdl.create_desc() + get_desc()
6. acl.rt.malloc(in_size) → dev_ptr_in         # 分配设备内存
7. acl.rt.memcpy(dev_ptr_in, ..., host_ptr, ..., H2D)   # 拷贝输入到设备
8. acl.create_data_buffer(dev_ptr_in, in_size)  # 包装设备指针
9. acl.mdl.add_dataset_buffer(in_ds, in_buf)
10. acl.rt.malloc(out_size) → dev_ptr_out
11. acl.create_data_buffer(dev_ptr_out, out_size)
12. acl.mdl.add_dataset_buffer(out_ds, out_buf)
13. acl.mdl.execute(model_id, in_ds, out_ds)     # NPU 推理
14. acl.rt.memcpy(host_ptr, ..., dev_ptr_out, ..., D2H)  # 拷贝输出回主机
15. 后处理 (YOLO decode + NMS)
16. 清理: destroy_data_buffer → destroy_dataset → rt.free → mdl.unload → destroy_context → finalize
```

### 4.4 推理脚本

**文件**: `edge-perception-bench/scripts/infer_yolo_acl.py`

```bash
python3 scripts/infer_yolo_acl.py \
    --om model/yolov8n.om \
    --video gA/1/s6/gA_1_s6_2019-03-08T09\;15\;15+01\;00_rgb_mosaic.avi
```

---

## 5. 第一帧推理结果

```
Video : gA_1_s6_2019-03-08T09;15;15+01;00_rgb_mosaic.avi
       1280x720, 5340 frames, 29.8 fps
Tensor: (1, 3, 640, 640)  float32

--- ACL Init ---
  OK

--- Load OM ---
  Inputs  [1]: dims=[[1, 3, 640, 640]]  sizes=[4915200]
  Outputs [1]: dims=[[1, 84, 8400]]  sizes=[2822400]

--- Infer ---
==================================================
Latency   : 31.3 ms
Detections: 1
  [person        ] conf=0.651  box=[95.5, 381.5, 263.2, 639.5]
==================================================
```

### 5.1 结果分析

**为什么框不准？**

1. **用了 mosaic.avi 而不是 face.mp4**: DMD 的 mosaic 是多视角拼接，人眼看到的分辨率很差，640×640 缩略图中人脸只有几十像素
2. **YOLOv8-nano COCO 预训练只有 80 个通用类别** (person, car...)，没有 face/hand 专用类
3. **0.651 的 person 检测可能误检了方向盘/座椅纹理**，不是真正的驾驶员人脸

### 5.2 正确的 DMD 推理方案

DMD 每个 session 已经提供了分离的视频流，不需要在 mosaic 上做目标检测：

| 输入视频 | 模型 | 任务 |
|----------|------|------|
| `_rgb_face.mp4` | SCRFD / RetinaFace | 人脸检测 |
| `_rgb_face.mp4` | L2CS-Net | 视线估计 (pitch, yaw) |
| `_rgb_hands.mp4` | YOLO-hand | 手部检测 |
| `_rgb_body.mp4` | YOLOv8-cls / MobileNetV3 | 分心/疲劳分类 |

---

## 6. 项目文件结构

```
edge-perception-bench/
├── models/
│   ├── yolov8n.onnx          # 12.3 MB, PyTorch → ONNX 导出
│   └── yolov8n.om            #  7.0 MB, ONNX → ATC → Ascend 310B4
├── scripts/
│   ├── export_yolov8_onnx.py # ONNX 导出脚本
│   ├── convert_onnx_to_om.sh # ATC 转换脚本
│   ├── infer_yolo_acl.py     # ACL Python 推理脚本
│   ├── check_env.py          # 环境检测
│   ├── check_env_ascend.py   # Ascend 环境检测
│   └── acl_diag.py           # ACL 诊断脚本
├── results/
│   └── doc/
│       └── 2026-07-20-dmd-yolov8n-acl-first-frame.md  # 本文
└── wheels/                   # onnxruntime ARM64 wheel 文件 (备选方案，未使用)
```

---

## 7. 后续计划

- [ ] 人脸检测: SCRFD / RetinaFace ONNX → OM
- [ ] 视线估计: L2CS-Net ONNX → OM
- [ ] 手部检测: YOLO-hand ONNX → OM
- [ ] 分心/疲劳分类: YOLOv8-cls ONNX → OM
- [ ] 多模型 pipeline 串联
- [ ] 性能分析 (latency, memory, throughput)
- [ ] 论文实验章节
