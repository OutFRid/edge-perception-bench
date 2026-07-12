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

- [ ] 昇腾环境部署（CANN 驱动 + ONNX Runtime Ascend）
- [ ] DMD 数据集加载与预处理
- [ ] YOLOv8-nano 基准推理 + 逐层 profiling
- [ ] MobileNetV3 基准推理 + 逐层 profiling
- [ ] INT8 量化对比
- [ ] 延时 / 功耗 / 精度 可视化

## 基准模型

| 模型 | 输入尺寸 | 参数量 | 精度 (DMD) | 推理延时 | 备注 |
|------|----------|--------|-----------|----------|------|
| YOLOv8-nano | 640×640 | 3.2M | — | — ms | 目标检测基线 |
| MobileNetV3-Small | 224×224 | 2.5M | — | — ms | 分类基线 |

*表格随实验推进逐步填充。*

## 项目结构

```
edge-perception-bench/
├── README.md
├── scripts/
│   └── check_env.py       # 环境检测脚本
├── results/
│   └── charts/            # 性能图表
└── requirements.txt
```

## 关联文章

本项目的进度日志和设计思路见 [OutFRidBlog](https://github.com/你的用户名/OutFRidBlog)：
- 待发布...

## License

MIT
