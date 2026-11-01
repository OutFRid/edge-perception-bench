#!/usr/bin/env python3
"""R4 · 解析 msprof 输出为逐层延时 / Host-Device 拷贝 / 内存报告。
用法（板端）: python3 parse_r4.py <PROF_DIR> <OUT_DIR>
"""
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

prof = Path(sys.argv[1])
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)

summary_dir = next(prof.glob("device_*/summary"))


def read_csv(name):
    f = next(summary_dir.glob(f"{name}_*.csv"))
    with open(f, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------- 1) 逐层延时（单次前向 op_summary）----------
ops = read_csv("op_summary")


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


rows = []
for r in ops:
    rows.append({
        "op_name": r["Op Name"], "op_type": r["OP Type"], "task_type": r["Task Type"],
        "duration_us": f(r["Task Duration(us)"]),
        "aicore_us": f(r.get("aicore_time(us)")),
        "vec_us": f(r.get("vec_exe_time(us)")), "mac_us": f(r.get("mac_exe_time(us)")),
        "mte1_us": f(r.get("mte1_exe_time(us)")), "mte2_us": f(r.get("mte2_exe_time(us)")),
        "mte3_us": f(r.get("mte3_exe_time(us)")), "fixpipe_us": f(r.get("fixpipe_exe_time(us)")),
    })
total_dur = sum(x["duration_us"] for x in rows)
by_type = defaultdict(lambda: {"count": 0, "us": 0.0})
for x in rows:
    by_type[x["op_type"]]["count"] += 1
    by_type[x["op_type"]]["us"] += x["duration_us"]
by_pipe = {k: sum(x[f"{k}_us"] for x in rows) for k in
           ("vec", "mac", "mte1", "mte2", "mte3", "fixpipe")}
top15 = sorted(rows, key=lambda x: -x["duration_us"])[:15]

# ---------- 2) Host<->Device / runtime API ----------
apis = read_csv("api_statistic")
api_rows = [r for r in apis if any(k in r["API Name"] for k in
            ("Memcpy", "MemCopy", "Malloc", "ModelLoad", "Execute", "Compute"))]

# ---------- 3) NPU 显存峰值（后台持续推理 + 轮询 npu-smi）----------
import re
_MEM_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*$")


def npu_mem_mb():
    """从 npu-smi info 主表解析设备显存 used（'1673 / 3513'）。
    表里同时有 Hugepages 'x / 15' 行，按 total 最大者取设备显存行。
    """
    try:
        txt = subprocess.check_output("npu-smi info", shell=True, text=True)
        best = None
        for line in txt.splitlines():
            if "|" not in line:
                continue
            seg = [s.strip() for s in line.split("|") if s.strip()]
            if len(seg) < 2:
                continue
            m = _MEM_RE.search(seg[-1])
            if m:
                used, total = int(m.group(1)), int(m.group(2))
                if best is None or total > best[1]:
                    best = (used, total)
        return best[0] if best else None
    except Exception:
        pass
    return None


peak = None
base = npu_mem_mb()
sys.stderr.write(f"[mem] base npu-smi used = {base} MiB; starting sustained loop...\n")
# 起一个持续推理的后台进程占住 NPU
loop = subprocess.Popen(
    "python3 /root/epb/scripts/bench_latency_acl.py "
    "--om /mnt/e61/workspace/dmd/model/yolov8n.om --iters 4000 --warmup 5 >/dev/null 2>&1",
    shell=True)
samples = []
t_end = time.time() + 12
while time.time() < t_end:
    m = npu_mem_mb()
    if m is not None:
        samples.append(m)
    time.sleep(0.4)
loop.terminate()
peak = max(samples) if samples else None

# ---------- 输出 ----------
report = {
    "model": "yolov8n.om (FP32 OM, HW:3519 / Ascend310B3)",
    "single_forward": {
        "num_ops": len(rows),
        "total_task_time_ms": round(total_dur / 1000, 3),
        "by_pipe_ms": {k: round(v / 1000, 3) for k, v in by_pipe.items()},
        "top15_ops": [{"op": x["op_name"], "type": x["op_type"],
                       "dur_us": round(x["duration_us"], 1),
                       "aicore_us": round(x["aicore_us"], 1),
                       "vec_us": round(x["vec_us"], 1), "mac_us": round(x["mac_us"], 1),
                       "mte2_us": round(x["mte2_us"], 1), "fixpipe_us": round(x["fixpipe_us"], 1)}
                      for x in top15],
        "by_op_type": {k: {"count": v["count"], "total_ms": round(v["us"] / 1000, 3)}
                       for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]["us"])},
    },
    "runtime_api": [{"api": r["API Name"], "level": r["Level"], "total_us": r["Time(us)"],
                     "count": r["Count"], "avg_us": r["Avg(us)"], "max_us": r["Max(us)"]}
                    for r in api_rows],
    "npu_memory_mib": {"base_used": base, "peak_used_samples": peak,
                       "samples": len(samples)},
}
(out / "r4_profiling.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

# markdown
md = ["# R4 · Profiling 基线（yolov8n · msprof device 侧）\n",
      f"- 单次前向算子数：**{len(rows)}**；device 侧 AI-core 任务总时长：**{total_dur/1000:.3f} ms**\n",
      "\n## 1. 分管道耗时归因（一次前向，ms）\n",
      "| pipe | 含义 | 耗时 |\n|---|---|---:|\n"]
pipe_desc = {"vec": "向量化算子(Vector)", "mac": "矩阵乘(M)", "mte1": "L1→L2搬数",
             "mte2": "DDR→L2输入搬数", "mte3": "L2→输出", "fixpipe": "输出写回DDR"}
for k in ("mac", "vec", "mte2", "fixpipe", "mte3", "mte1"):
    md.append(f"| {k} | {pipe_desc[k]} | {by_pipe[k]/1000:.3f} |\n")
md.append(f"\n> 合计（含重叠）对照 device 总时长 {total_dur/1000:.3f}ms；"
          "mte2+fixpipe 为纯搬数（访存），mac+vec 为纯算力。\n")
md.append("\n## 2. Top 15 耗时算子（逐层延时）\n")
md.append("| op | type | 总时长(us) | aicore(us) | vec | mac | mte2 | fixpipe |\n")
md.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
for x in top15:
    md.append(f"| {x['op_name']} | {x['op_type']} | {x['duration_us']:.1f} | "
              f"{x['aicore_us']:.1f} | {x['vec_us']:.1f} | {x['mac_us']:.1f} | "
              f"{x['mte2_us']:.1f} | {x['fixpipe_us']:.1f} |\n")
md.append("\n## 3. 按算子类型聚合\n")
md.append("| OP Type | 实例数 | 总时长(ms) |\n|---|---:|---:|\n")
for k, v in report["single_forward"]["by_op_type"].items():
    md.append(f"| {k} | {v['count']} | {v['total_ms']:.3f} |\n")
md.append("\n## 4. Host↔Device 拷贝 & runtime API（msprof 采集，单位 us）\n")
md.append("| API | level | 总时长(us) | 次数 | 均值(us) | 最大(us) |\n|---|---|---:|---:|---:|---:|\n")
for r in report["runtime_api"]:
    md.append(f"| {r['api']} | {r['level']} | {r['total_us']} | {r['count']} | "
              f"{r['avg_us']} | {r['max_us']} |\n")
md.append("\n## 5. NPU 显存\n")
md.append(f"- 空闲基线 used：**{base} MiB**\n- 持续推理峰值 used：**{peak} MiB**"
          f"（采样 {len(samples)} 次，npu-smi 口径）\n")
(out / "r4_profiling.md").write_text("".join(md))

sys.stderr.write(f"[done] wrote {out/'r4_profiling.json'} and {out/'r4_profiling.md'}\n")
sys.stderr.write(f"[done] mem base={base} peak={peak}\n")
