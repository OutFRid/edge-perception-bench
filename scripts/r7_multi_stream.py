"""R7 · 多路并发吞吐探测：N 个独立进程各跑一路推理，测聚合吞吐与单路退化。

背景（R7 主线 2）：真实部署是「多路摄像头各自一路流」，310B 单卡上
并发路数 N 上升时，聚合吞吐是否线性、单路延时如何退化，是容量规划关键。

方法：N 个 worker 进程（每进程独立 acl.init/context/stream/模型实例），
Event 同时起跑，各跑 iters 次纯推理（H2D+execute+sync+D2H，输入随机张量），
统计单路 mean/p95 延时与 N 路聚合吞吐。与 bench_latency_acl.py 同口径。

用法（板端）:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 r7_multi_stream.py \
        --om /mnt/e61/workspace/dmd/model/yolov8n.om \
        --streams 1 2 4 8 --iters 100 --warmup 10 \
        --json /root/epb/results/r7_multi_stream.json
"""
import argparse
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

ACL_SUCCESS = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _check(r, m=""):
    if r != ACL_SUCCESS:
        raise RuntimeError(f"ACL error {r}: {m}")


def worker(om_path, iters, warmup, wid, start_evt, q):
    """单个推理流：独立进程。把单路延时与总墙钟送回父进程。"""
    import numpy as np
    import acl

    _check(acl.init(""), "init")
    _check(acl.rt.set_device(0), "set_device")
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

    host_in = [np.frombuffer(bytes(sz), dtype=np.uint8) for sz in in_sizes]
    host_out = [np.empty(sz, dtype=np.uint8) for sz in out_sizes]

    def one_iter():
        for i, sz in enumerate(in_sizes):
            _check(acl.rt.memcpy(in_dev[i], sz, host_in[i].ctypes.data, sz,
                                 ACL_MEMCPY_HOST_TO_DEVICE))
        _check(acl.mdl.execute(mid, in_ds, out_ds))
        _check(acl.rt.synchronize_stream(0))
        for i, sz in enumerate(out_sizes):
            _check(acl.rt.memcpy(host_out[i].ctypes.data, sz, out_dev[i], sz,
                                 ACL_MEMCPY_DEVICE_TO_HOST))

    for _ in range(warmup):
        one_iter()

    start_evt.wait()  # 所有 worker 同刻起跑
    lats = []
    t0 = time.perf_counter()
    for _ in range(iters):
        t = time.perf_counter()
        one_iter()
        lats.append((time.perf_counter() - t) * 1000)
    wall = time.perf_counter() - t0

    q.put({
        "wid": wid,
        "iters": iters,
        "wall_s": round(wall, 3),
        "lat_mean_ms": round(statistics.mean(lats), 3),
        "lat_median_ms": round(statistics.median(lats), 3),
        "lat_p95_ms": round(sorted(lats)[int(len(lats) * 0.95) - 1], 3),
    })

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


def run_case(om_path, n_streams, iters, warmup):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    start_evt = ctx.Event()
    procs = [ctx.Process(target=worker, args=(om_path, iters, warmup, i, start_evt, q))
             for i in range(n_streams)]
    for p in procs:
        p.start()
    start_evt.set()
    for p in procs:
        p.join()
    stats = [q.get() for _ in range(n_streams)]
    stats.sort(key=lambda x: x["wid"])
    wall = max(s["wall_s"] for s in stats)
    agg = {
        "streams": n_streams,
        "iters_per_stream": iters,
        "aggregate_wall_s": round(wall, 3),
        "aggregate_fps": round(n_streams * iters / wall, 2),
        "aggregate_latency_ms": round(n_streams * wall * 1000 / (n_streams * iters), 3),
        "per_stream_lat_mean_ms": round(statistics.mean(s["lat_mean_ms"] for s in stats), 3),
        "per_stream_lat_p95_ms": round(statistics.mean(s["lat_p95_ms"] for s in stats), 3),
        "single_stream_fps": round(1000 / statistics.mean(s["lat_mean_ms"] for s in stats), 2),
    }
    return agg, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--om", required=True)
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--json", help="dump to json")
    args = ap.parse_args()
    if not Path(args.om).exists():
        raise SystemExit(f"OM not found: {args.om}")

    results = []
    for n in args.streams:
        print(f"\n=== 并发 {n} 路 ===", flush=True)
        agg, stats = run_case(args.om, n, args.iters, args.warmup)
        results.append({"case": agg, "workers": stats})
        print(f"  聚合吞吐 : {agg['aggregate_fps']} fps (墙钟 {agg['aggregate_wall_s']}s)")
        print(f"  单路均值 : {agg['per_stream_lat_mean_ms']} ms / p95 {agg['per_stream_lat_p95_ms']} ms")
        for s in stats:
            print(f"    w{s['wid']}: mean={s['lat_mean_ms']} p95={s['lat_p95_ms']}")

    # 汇总对比表
    print("\n| 路数 | 聚合吞吐 fps | 单路均值 ms | 单路 p95 ms | 单路等效 fps |")
    print("|---:|---:|---:|---:|---:|")
    for r in results:
        c = r["case"]
        print(f"| {c['streams']} | {c['aggregate_fps']} | {c['per_stream_lat_mean_ms']} "
              f"| {c['per_stream_lat_p95_ms']} | {c['single_stream_fps']} |")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
