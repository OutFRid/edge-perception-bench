"""通用 ACL 推理延时基准（模型无关）。

只看 OM 的输入/输出字节数，喂随机张量，测三段延时：
  - compute = mdl.execute + synchronize_stream（NPU 真正算的时间）
  - h2d     = 输入 memcpy Host->Device
  - d2h     = 输出 memcpy Device->Host
适合横向对比不同模型（检测 / 分类）的推理延时，不依赖前后处理语义。

用法（板端，先 source set_env.sh）:
    python3 bench_latency_acl.py --om /path/x.om --iters 200 --warmup 20 --tag yolov8n
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

ACL_SUCCESS = 0
H2D, D2H = 1, 2


def _check(r, m=""):
    if r != ACL_SUCCESS:
        raise RuntimeError(f"ACL error {r}: {m}")


def bench(om_path, iters, warmup):
    import acl
    _check(acl.init(""))
    _check(acl.rt.set_device(0))
    ctx, ret = acl.rt.create_context(0)
    _check(ret, "create_context")

    mid, ret = acl.mdl.load_from_file(om_path)
    _check(ret, "load_from_file")
    desc = acl.mdl.create_desc()
    _check(acl.mdl.get_desc(desc, mid), "get_desc")
    n_in = acl.mdl.get_num_inputs(desc)
    n_out = acl.mdl.get_num_outputs(desc)
    in_sizes = [acl.mdl.get_input_size_by_index(desc, i) for i in range(n_in)]
    out_sizes = [acl.mdl.get_output_size_by_index(desc, i) for i in range(n_out)]

    # 预分配 device 输入/输出缓冲 + dataset（隔离出纯 execute/sync 与 memcpy）
    in_dev, out_dev, in_bufs, out_bufs = [], [], [], []
    for sz in in_sizes:
        p, _ = acl.rt.malloc(sz, 0)
        in_dev.append(p)
        in_bufs.append(acl.create_data_buffer(p, sz))
    for sz in out_sizes:
        p, _ = acl.rt.malloc(sz, 0)
        out_dev.append(p)
        out_bufs.append(acl.create_data_buffer(p, sz))
    in_ds = acl.mdl.create_dataset()
    for b in in_bufs:
        _check(acl.mdl.add_dataset_buffer(in_ds, b)[1])
    out_ds = acl.mdl.create_dataset()
    for b in out_bufs:
        _check(acl.mdl.add_dataset_buffer(out_ds, b)[1])

    # host 侧随机输入缓冲
    host_in = [np.frombuffer(bytes(sz), dtype=np.uint8) for sz in in_sizes]
    host_out = [np.empty(sz, dtype=np.uint8) for sz in out_sizes]

    def do_h2d():
        for i, sz in enumerate(in_sizes):
            _check(acl.rt.memcpy(in_dev[i], sz, host_in[i].ctypes.data, sz, H2D))

    def do_exec_sync():
        _check(acl.mdl.execute(mid, in_ds, out_ds))
        _check(acl.rt.synchronize_stream(0))

    def do_d2h():
        for i, sz in enumerate(out_sizes):
            _check(acl.rt.memcpy(host_out[i].ctypes.data, sz, out_dev[i], sz, D2H))

    # warmup
    for _ in range(warmup):
        do_h2d(); do_exec_sync(); do_d2h()

    t_comp, t_h2d, t_d2h = [], [], []
    for _ in range(iters):
        t0 = time.perf_counter(); do_h2d(); t1 = time.perf_counter()
        do_exec_sync(); t2 = time.perf_counter()
        do_d2h(); t3 = time.perf_counter()
        t_h2d.append((t1 - t0) * 1000)
        t_comp.append((t2 - t1) * 1000)
        t_d2h.append((t3 - t2) * 1000)

    def stat(x):
        return {"mean": round(statistics.mean(x), 3), "median": round(statistics.median(x), 3),
                "p95": round(sorted(x)[int(len(x) * 0.95) - 1], 3), "min": round(min(x), 3),
                "max": round(max(x), 3)}

    sc, sh, sd = stat(t_comp), stat(t_h2d), stat(t_d2h)
    total = sc["mean"] + sh["mean"] + sd["mean"]
    print(f"\n{'='*56}")
    print(f"Model : {Path(om_path).name}   in_bytes={in_sizes} out_bytes={out_sizes}")
    print(f"Iters : {iters} (warmup {warmup})")
    print(f"  compute(execute+sync) mean={sc['mean']:7.3f} ms  p95={sc['p95']:.3f}")
    print(f"  H2D memcpy            mean={sh['mean']:7.3f} ms  p95={sh['p95']:.3f}")
    print(f"  D2H memcpy            mean={sd['mean']:7.3f} ms  p95={sd['p95']:.3f}")
    print(f"  推理调用合计           mean={total:7.3f} ms   -> {1000/total:.1f} fps")
    print(f"{'='*56}")

    # 清理
    for ds in (in_ds, out_ds):
        acl.mdl.destroy_dataset(ds)
    for b in in_bufs + out_bufs:
        acl.destroy_data_buffer(b)
    for p in in_dev + out_dev:
        acl.rt.free(p)
    acl.mdl.unload(mid)
    acl.mdl.destroy_desc(desc)
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()

    return {"om": str(om_path), "in_bytes": in_sizes, "out_bytes": out_sizes,
            "iters": iters, "compute_ms": sc, "h2d_ms": sh, "d2h_ms": sd,
            "total_infer_ms": round(total, 3), "fps": round(1000 / total, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--om", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--tag", default="")
    ap.add_argument("--json", help="append result to json file")
    args = ap.parse_args()
    if not Path(args.om).exists():
        raise SystemExit(f"OM not found: {args.om}")
    res = bench(args.om, args.iters, args.warmup)
    res["tag"] = args.tag
    if args.json:
        p = Path(args.json)
        data = json.loads(p.read_text()) if p.exists() else []
        data.append(res)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"appended -> {p}")


if __name__ == "__main__":
    main()
