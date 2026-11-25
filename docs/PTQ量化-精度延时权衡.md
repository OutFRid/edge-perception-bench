# PTQ 量化「精度-延时」权衡方法

> 沉淀自 R5（INT8 量化）+ R6（策略对比）实测，面向「低算力边缘 NPU 上的轻量感知模型」。
> 验证环境：yolov8n（8TOPS Ascend310B 边缘设备项目）/ onnxruntime 1.23 / CANN 7.0.x 工具链。

## 一、先问硬件：量化收益是硬件属性

动手前第一个决策点，不是「选哪个量化策略」，而是：

**目标硬件是否有 int8 MAC 单元？**

| 硬件 | 实测结论 | 依据 |
|---|---|---|
| 通用 x86 CPU（onnxruntime） | INT8 反而慢 1.2~1.35× | R5 单策略 96.4ms vs FP32 74.2ms；R6 七策略 89~100ms 无一变快 |
| 昇腾 310B NPU（待上板验证） | 预期收益=体积 + 带宽减半 | R4 已证 310B 上 yolov8n 是访存 bound，算力仅 3.5ms；int8 收益主要在带宽 |

> 推论：**在没有 int8 定点单元的平台上，PTQ 的延时收益约等于零（甚至为负），策略对比应聚焦「精度损失最小」而非「延时最低」。** 有 int8 MAC 的平台上，策略对比才同时有精度轴和延时轴。

## 二、精度要分层报（cosine 会藏住致命伤）

单看张量级 cosine 相似度会漏掉语义级崩溃。R6 实测：**全网络量化（S6）cosine=0.99823，检出数=0**——量化后的 Softmax/Reshape/Concat 破坏了检测头，但 cosine 完全看不出来。

| 层级 | 指标 | 能发现的问题 |
|---|---|---|
| 张量级 | 原始输出 cosine | 数值漂移 |
| 检测级 | 置信度均值 | 打分漂移 |
| 语义级 | 检出数 / 框位 IoU | 目标丢失、框跑偏 |

> 规则：**至少报三层；语义级指标为 0/异常时，cosine 再高也是不合格。**

## 三、策略维度与实测结论

四个可调维度（以 onnxruntime quantize_static 为例）：

1. **激活标定方法**：MinMax / Entropy / Percentile(99.99)
2. **权重量化粒度**：per-channel / per-tensor
3. **量化范围**：仅 Conv / Conv+MatMul / 全网络（默认 op 列表）
4. **量化类型**：静态 QDQ（权重+激活）/ 动态量化（仅权重）

R6 七策略实测（yolov8n，CPU 单线程，FP32 基线 74.2ms / cosine 1.0 / 检出 0.94）：

| 策略 | 检出数 | conf | IoU(vs S1) | 延时 | 结论 |
|---|---:|---:|---:|---:|---|
| S1 MinMax / pc / Conv+MatMul | 1.12 | 0.689 | — | 91.1ms | R5 基线，可用 |
| S2 Entropy / pc / Conv+MatMul | 1.0 | 0.691 | 0.926 | 94.3ms | 可用，无优势 |
| **S3 Percentile99.99 / pc / Conv+MatMul** | 1.12 | **0.716** | **0.965** | 90.1ms | **推荐：精度最稳** |
| S4 MinMax / per-tensor / Conv+MatMul | 1.06 | 0.740 | 0.893 | 89.2ms | IoU 明显低于 per-channel |
| S5 MinMax / pc / 仅 Conv | 1.0 | 0.691 | 0.926 | 89.2ms | 与 S2 等价 |
| S6 MinMax / pc / 全网络 | **0 ❌** | — | — | 100.2ms | **检测头被量化破坏，禁用** |
| S7 动态量化（仅权重，限 MatMul） | 0.94 | 0.7104 | 0.959 | 71.0ms | 卷积网络几乎无压缩收益 |

### 决策规则
1. **per-channel 优先于 per-tensor**：IoU 0.965 vs 0.893，精度代价小。
2. **标定方法三者皆可，Percentile 略优**：本任务差异在噪声范围，但 Percentile 对激活分布含离群值的网络更稳（截断 99.99% 保护 range）。
3. **禁止全网络量化**（检测/分割任务）：检测头（Softmax/DFL/Reshape/Concat）必须排除；量化范围锁定「Conv+MatMul」。
4. **卷积为主的小网络不要用动态量化**：MatMul 权重占比极小，体积 12.25MB≈FP32（12.2MB），无收益。
5. **延时轴在无 int8 MAC 平台无效**：策略间延时差异 ≤12% 且全部慢于 FP32，选策略看精度即可。

## 四、可复用决策流程

```
目标硬件有 int8 MAC？
├─ 否 → 量化只为体积；策略选精度最稳者（per-channel + 排除敏感算子）；
│        延时收益不抱期望（实测反而可能变慢）。
└─ 是 → 精度与延时双轴评估：
        1. per-channel + 静态 QDQ
        2. 标定方法：MinMax 起步，Percentile 备选（离群值敏感时）
        3. 量化范围：Conv+MatMul；检测头显式排除
        4. 精度报三层（cosine / conf / 检出数+IoU）
        5. 延时与 FP32 基线对比，收益归属带宽/算力分层归因
```

## 五、工具链边界（昇腾 310B + CANN 7.0.x）

- **ATC 不解析通用 QDQ ONNX**：onnxruntime 产出的 QuantizeLinear/DequantizeLinear 节点被 ATC 直接拒绝（7.0.RC1 板端与 7.0.1 WSL 均实测 `No parser is registered for Op`）。「任意 QDQ ONNX → ATC」假设不成立。
- **唯一官方路径**：昇腾 AMCT（`Ascend-cann-amct_7.0.x_linux-x86_64.zip`，含 `amct_onnx` + custom op 注册包）产出 QDQ ONNX → ATC。AMCT 包仅官网登录下载，需与 CANN 版本匹配。
- **排期教训**：涉及「某平台离线量化」的排期，先用「一个 QDQ 模型试编 + 查 amct CLI」验证工具链假设，再排上板验收；别把未上板的 INT8 当达标。
