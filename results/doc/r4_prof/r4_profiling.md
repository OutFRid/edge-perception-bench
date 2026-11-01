# R4 · Profiling 基线（yolov8n · msprof device 侧）
- 单次前向算子数：**105**；device 侧 AI-core 任务总时长：**11.905 ms**

## 1. 分管道耗时归因（一次前向，ms）
| pipe | 含义 | 耗时 |
|---|---|---:|
| mac | 矩阵乘(M) | 1.663 |
| vec | 向量化算子(Vector) | 1.872 |
| mte2 | DDR→L2输入搬数 | 2.601 |
| fixpipe | 输出写回DDR | 0.590 |
| mte3 | L2→输出 | 1.804 |
| mte1 | L1→L2搬数 | 0.665 |

> 合计（含重叠）对照 device 总时长 11.905ms；mte2+fixpipe 为纯搬数（访存），mac+vec 为纯算力。

## 2. Top 15 耗时算子（逐层延时）
| op | type | 总时长(us) | aicore(us) | vec | mac | mte2 | fixpipe |
|---|---|---:|---:|---:|---:|---:|---:|
| /model.0/conv/Conv/model.0/act/Sigmoid/model.0/act/Mul | Conv2D | 1258.2 | 533.0 | 121.7 | 92.6 | 489.1 | 188.1 |
| PartitionedCall_/model.22/dfl/Transpose_1_Transpose_64 | Transpose | 861.9 | 366.2 | 97.2 | 0.0 | 57.5 | 0.0 |
| /model.22/dfl/Softmax | SoftmaxV2 | 817.5 | 347.3 | 346.3 | 0.0 | 22.3 | 0.0 |
| trans_TransData_1 | TransData | 754.9 | 320.6 | 129.5 | 0.0 | 131.0 | 0.0 |
| PartitionedCall_/model.22/dfl/Transpose_Transpose_61 | Transpose | 426.0 | 180.7 | 35.7 | 0.0 | 132.6 | 0.0 |
| /model.1/conv/Conv/model.1/act/Sigmoid/model.1/act/Mul | Conv2D | 408.0 | 173.0 | 60.9 | 46.2 | 147.6 | 27.1 |
| trans_Cast_0 | Cast | 368.0 | 155.0 | 21.1 | 0.0 | 153.0 | 0.0 |
| /model.22/cv3.0/cv3.0.1/conv/Conv/model.22/cv3.0/cv3.0.1/act/Sigmoid/model.22/cv3.0/cv3.0.1/act/Mul | Conv2D | 360.5 | 152.8 | 41.6 | 136.4 | 80.5 | 19.6 |
| /model.22/cv3.0/cv3.0.0/conv/Conv/model.22/cv3.0/cv3.0.0/act/Sigmoid/model.22/cv3.0/cv3.0.0/act/Mul | Conv2D | 298.8 | 126.6 | 41.0 | 74.9 | 79.5 | 9.5 |
| /model.2/cv1/conv/Conv/model.2/cv1/act/Sigmoid/model.2/cv1/act/Mul | Conv2D | 217.8 | 92.1 | 60.4 | 85.4 | 27.3 | 12.9 |
| /model.2/cv2/conv/Conv/model.2/cv2/act/Sigmoid/model.2/cv2/act/Mul | Conv2D | 217.4 | 91.9 | 60.4 | 69.8 | 31.1 | 12.4 |
| /model.22/cv2.0/cv2.0.0/conv/Conv/model.22/cv2.0/cv2.0.0/act/Sigmoid/model.22/cv2.0/cv2.0.0/act/Mul | Conv2D | 209.1 | 88.4 | 32.8 | 47.9 | 19.3 | 7.3 |
| /model.22/cv2.0/cv2.0.1/conv/Conv/model.22/cv2.0/cv2.0.1/act/Sigmoid/model.22/cv2.0/cv2.0.1/act/Mul | Conv2D | 180.5 | 76.2 | 32.8 | 47.9 | 7.2 | 7.3 |
| /model.15/cv1/conv/Conv/model.15/cv1/act/Sigmoid/model.15/cv1/act/Mul | Conv2D | 146.5 | 61.8 | 30.7 | 38.0 | 42.0 | 24.4 |
| /model.5/conv/Conv/model.5/act/Sigmoid/model.5/act/Mul | Conv2D | 143.3 | 60.4 | 18.2 | 48.1 | 31.0 | 8.1 |

## 3. 按算子类型聚合
| OP Type | 实例数 | 总时长(ms) |
|---|---:|---:|
| Conv2D | 64 | 7.049 |
| Transpose | 2 | 1.288 |
| TransData | 9 | 1.135 |
| SoftmaxV2 | 1 | 0.817 |
| ConcatD | 10 | 0.653 |
| Cast | 2 | 0.496 |
| Add | 7 | 0.178 |
| Sigmoid | 1 | 0.115 |
| ResizeNearestNeighborV2 | 2 | 0.109 |
| MaxPoolV3 | 3 | 0.044 |
| StridedSliceD | 2 | 0.009 |
| Mul | 1 | 0.006 |
| Sub | 1 | 0.006 |

## 4. Host↔Device 拷贝 & runtime API（msprof 采集，单位 us）
| API | level | 总时长(us) | 次数 | 均值(us) | 最大(us) |
|---|---|---:|---:|---:|---:|
| DevMalloc | runtime | 584.229 | 4 | 146.0573 | 301.459 |
| MemCopySync | runtime | 2244.437 | 1 | 2244.437 | 2244.437 |

## 5. NPU 显存
- 空闲基线 used：**1681 MiB**
- 持续推理峰值 used：**1688 MiB**（采样 14 次，npu-smi 口径）
