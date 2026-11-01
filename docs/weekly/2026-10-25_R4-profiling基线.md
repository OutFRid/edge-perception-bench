# R4 · 主题：逐层 Profiling 基线

> 日期区间：2026-10-25 ~ 2026-10-31（**顺延执行**，连板补做于 2026-11-01）
> 状态：✅ 已完成（device 侧逐层延时、内存占用、Host↔Device 拷贝三块基线全部落地）

## 背景
- 日历 R4 技术任务：**Profiling 基线——逐层延时、内存峰值、Host↔Device 拷贝耗时**。
- 承接：R2/R3 用 `wall-clock` 分段计时把 E2E 拆成 8 段（`一帧图的旅程` §9、R3 周志 B 表），但那是 **Host 视角**；R4 用板端 `msprof` 把 NPU 侧「13ms compute」这一格进一步下钻到 **device 逐层（AI-core task）口径**——即 R3 周志「下周调整」预告的升级。

## 本周目标（≤3 个）
1. ☑ 用 `msprof` 对 `yolov8n.om` 纯推理循环采集 device 侧 profiling，产出逐层（per-op / per-AI-core-task）延时。
2. ☑ 采集 Host↔Device 拷贝耗时与 runtime API（malloc / memcpy）口径，与 R3 的 bench wall-clock 交叉验证。
3. ☑ 采集推理过程中的 NPU 显存占用（基线 vs 持续推理峰值）。

## 动手链路（可复现）
- 采集：`scripts/board/run_r4_prof.sh` → `msprof --application="python3 scripts/bench_latency_acl.py --om yolov8n.om --iters 40" --ai-core=on --aic-mode=task-based --task-time=on --runtime-api=on --ascendcl=on --model-execution=on`
- 解析：`scripts/parse_r4.py <PROF_DIR> <OUT_DIR>` → 读 `op_summary / api_statistic` CSV + 后台持续推理轮询 `npu-smi` 采样显存 → 出 `r4_profiling.{md,json}`。
- 产物目录：`results/doc/r4_prof/`（含 `op_summary` 原始 CSV、md、json）。
- 板端环境：CANN 7.0.RC1 / Ascend310B(310B4)、`msprof` 在 `ascend-toolkit/latest/bin`。

## 本周实际产出（核心数据）

### ① 单次前向：device 侧逐层延时
- 单次前向 **105 个 AI-core 任务**，device 任务总时长 **11.905 ms**。
- **交叉验证口径**：msprof device 总时长 11.9ms ≈ R3 bench `compute(execute+sync)` 13.0ms（差 ~1.1ms 为 launch/sync 调度开销）——两条独立口径对齐，device 数据可信。

**按算子类型聚合（总时长降序）：**
| OP Type | 实例数 | 总时长(ms) |
|---|---:|---:|
| Conv2D | 64 | 7.049 |
| Transpose | 2 | 1.288 |
| TransData | 9 | 1.135 |
| SoftmaxV2 | 1 | 0.817 |
| ConcatD | 10 | 0.653 |
| Cast | 2 | 0.496 |
| Add / Sigmoid / Resize / MaxPool … | 余 | <0.2 各 |

> **Conv2D 独占 59%**（7.05/11.9ms）是 NPU 侧绝对大头；但注意 **非卷积的「搬运/reshape 类」算子（Transpose+TransData+Cast+ConcatD = 3.57ms，30%）** 才是隐藏成本——它们几乎不算数，只在搬布局（NC1HWC0 转换、FP32→FP16 cast、DFL 头 reshape）。

**Top 5 耗时单算子：**
1. `/model.0` 首层 Conv（1258us）——输入 640×640 全分辨率，mte2 搬数 489us 占主导（**访存 bound**，非算力）。
2. `/model.22/dfl/Transpose`（862us）、`dfl/Softmax`（818us）——检测头 DFL 解码的 reshape/softmax，纯 Host 语义必需但在 NPU 上很贵。
3. `trans_TransData_1`（755us）、`trans_Cast_0`（368us）——图入口的数据格式/精度转换。

### ② 分管道耗时归因（算力 vs 访存）
| pipe | 含义 | 耗时(ms) |
|---|---|---:|
| mte2 | DDR→L2 输入搬数 | 2.601 |
| vec | 向量化算子 | 1.872 |
| mac | 矩阵乘(M) | 1.663 |
| mte3 | L2→输出搬数 | 1.804 |
| mte1 | L1→L2 搬数 | 0.665 |
| fixpipe | 输出写回 DDR | 0.590 |

> **关键认知**：纯算力（mac+vec=3.53ms） < 纯搬数（mte1+mte2+mte3+fixpipe=5.66ms）。yolov8n 在 310B 上是 **访存/搬运 bound，不是算力 bound**——这与 R2/R3「Host CPU 才是 E2E 大头」呼应，构成「延时三层论」：E2E 大头在 Host 前后处理；即便只看 NPU，大头也在搬数而非算。

### ③ Host↔Device 拷贝 & runtime API
- msprof runtime 口径：`DevMalloc` 4 次共 584us；`MemCopySync`（模型加载态）。
- 每帧推理拷贝（与 R3 bench wall-clock 对齐）：**H2D 4.9MB ≈ 1.52ms（~3.2GB/s）**、**D2H 2.8MB ≈ 1.60ms**——拷贝量与 I/O 字节数线性（延续 R3「memcpy≈I/O 字节数函数」结论）。

### ④ NPU 显存
- 空闲基线 used **1681 MiB**；持续单路推理峰值 **1688 MiB**（npu-smi 口径，采样 14 次）。
- **单路 yolov8n 增量仅 ~7 MiB**（≈ 输入 4.9MB + 输出 2.8MB 设备缓冲；权重随模型加载计入基线）→ 边缘多路并发（R7 方向）内存压力极小。

## 三条被数据敲实的认知
1. **延时三层论成形**：E2E(144ms) 大头在 Host pre/post(51.8ms+) → NPU(13ms) 内部大头又在搬数(5.66ms) 而非算力(3.53ms) → 优化优先级：先砍 Host 前后处理(R7/DVPP)，再降 NPU 搬运（算子融合/布局）。
2. **检测头 DFL（Transpose+Softmax≈1.7ms）是 NPU 上的"贵而不算"环节**：anchor-free 解码在硬件上 reshape 密集，是后处理可下沉/替换的候选。
3. **口径必须标注**：msprof device 11.9ms、bench compute 13ms、E2E 144ms 是三个不同口径，逐层分析只在 device 口径内自洽，跨层比较要先统一口径（延续 R3「单看 fps 会骗人」）。

## 遇到阻碍 / 遗留
- `--model-execution` 对 OM（`aclmdlExecute` 离线路径）的 GE 逐算子拆分未额外产出（AI-core task 口径已足够）；如需 timeline 可视化，可在 PC 侧 MindStudio 打开 `timeline/*.json`。
- 本次只跑 yolov8n 单模型基线；yolov8s / mobilenetv3 的逐层口径可后续按需补（脚本已模型无关，换 `--om` 即可）。

## 下周调整（衔接 R5）
- R5 进入 INT8 量化：FP32→INT8，记录精度损失 vs 延时收益。
- 注意：310B 上既然算力只占 3.5ms、搬数占 5.7ms，INT8 主要压缩 **权重体积与部分访存**，对 mac 收益未必直接转化为 E2E——R5 需带着「延时收益到底发生在哪一层」的问题去测。
