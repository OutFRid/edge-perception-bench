# edge-perception-bench

> 8TOPS 边缘设备上轻量级感知模型的性能基准测试

## 硬件环境

| 项目 | 详情 |
|------|------|
| 设备 | 华为昇腾 Atlas 200I DK（8 TOPS INT8） |
| 推理框架 | CANN / ONNX Runtime |
| 工具链 | Python 3.x, MindSpore |

## 为什么做这个

车载端侧感知（DMS、环视、舱内行为识别）受限于 5-15 TOPS 的 NPU 预算。
学术界的方法在 A100 上效果惊艳，但没人告诉你它们在 8TOPS 设备上能不能跑、延时多少、功耗多大。

这个仓库回答一件事：
**一个感知模型部署到端侧芯片上，到底花多少时间、占多少内存、精度掉多少。**

## 当前进度

- [x] 昇腾环境部署（CANN 驱动 + ONNX Runtime Ascend）
- [x] DMD 数据集加载 + YOLOv8-nano → OM → **ACL 推理第一帧**（2026-07-20，见 `results/doc/`）
- [ ] 修复第一帧人框偏移
- [ ] 换模型（MobileNetV3）基准推理 + 逐层 profiling
- [ ] INT8 量化对比
- [ ] 延时 / 功耗 / 精度 可视化

> 📅 完整周度排期与进度追踪见 **[docs/项目日历.md](docs/项目日历.md)**，每周更新一条周志（`docs/weekly/`）。
> 项目于 2026-07-21 ~ 10-03 因故停摆，2026-10-04 起重新基线推进。

## 基准模型

| 模型 | 输入尺寸 | 参数量 | 精度 (DMD) | 推理延时 | 备注 |
|------|----------|--------|-----------|----------|------|
| YOLOv8-nano | 640×640 | 3.2M | — | — ms | 目标检测基线 |
| MobileNetV3-Small | 224×224 | 2.5M | — | — ms | 分类基线 |

*表格随实验推进逐步填充。*

## 环境配置

### DK 网络设置（手动执行，重启后需重新运行）

```bash
# Windows 端：先开启网络共享
# ncpa.cpl → 右键上网网卡 → 属性 → 共享 → 勾选并选择 USB RNDIS6 适配器

# DK 端：删除旧 IP，设置固定 IP + 路由 + DNS
# 执行后 SSH 会断，用 ssh root@192.168.137.2 重连
ip addr del 192.168.0.2/24 dev usb0 2>/dev/null
ip addr add 192.168.137.2/24 dev usb0
ip route replace default via 192.168.137.1 dev usb0
echo "nameserver 8.8.8.8" > /etc/resolv.conf
echo "nameserver 114.114.114.114" >> /etc/resolv.conf
```

### CANN 环境初始化

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import acl; print('pyACL OK')"
```

## 项目结构

```
edge-perception-bench/
├── README.md
├── docs/
│   ├── 项目日历.md            # 周度排期 + 进度追踪（每周更新）
│   ├── 技术路线.md            # 方法论 + 学习资源索引
│   └── weekly/                # 每周一条周志
├── scripts/
│   ├── check_env.py           # 环境检测
│   ├── check_env_ascend.py    # 昇腾环境检测
│   ├── acl_diag.py            # ACL 诊断
│   ├── export_yolov8_onnx.py  # 模型导出 ONNX
│   ├── convert_onnx_to_om.sh  # ONNX → OM 离线编译
│   ├── infer_yolo_frame.py    # 单帧推理
│   └── infer_yolo_acl.py      # ACL 端到端推理流水线
├── results/
│   ├── doc/                   # 阶段记录文档
│   ├── image/                 # 推理结果图
│   └── charts/                # 性能图表
└── requirements.txt
```

## 关联文章

本项目的进度日志和设计思路见 [OutFRidBlog](https://github.com/你的用户名/OutFRidBlog)：
- 待发布...

## License

MIT
