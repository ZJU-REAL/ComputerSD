#!/usr/bin/env python3
"""压测 sglang generate:扫并发级别 × 统一真实3图payload,测全指标。
用法:
  python bench_io_workers.py <label> [router_url]
  label      = 这次的标签(如 io4 / io32),结果存 /tmp/bench_<label>.json
  router_url = 默认 http://sglang-router-host:4394/generate
两组跑完后:python bench_io_workers.py --compare io4 io32

每个并发级别报告 dispatch(图像处理) / prefill+decode / e2e / 客户端wall 的 中位/p90/max。
- 真实3图payload(从 trajectory.json 重建),保证可比
- 多进程并发(绕 GIL,对齐训练 64 独立 worker)
- 打 router(训练真实路径)
"""
import json, sys, time, statistics, urllib.request
from multiprocessing import Pool

PAYLOAD_FILE = "/tmp/payloads_3img.json"
CONC_LEVELS = [1, 4, 8, 16, 32, 64]
ROUTER = "http://sglang-router-host:4394/generate"

def pct(a, p):
    a = sorted(a); return a[min(len(a) - 1, int(len(a) * p))]

def _one(args):
    # body 是预先序列化好的单条请求体 bytes(只传一条，避免把整个大payload列表复制到每个worker→OOM)
    body, router = args
    req = urllib.request.Request(router, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=300)); wall = time.time() - t0
    except Exception as e:
        return {"err": str(e)[:80]}
    mi = r.get("meta_info", {})
    rcv = mi.get("request_received_ts"); disp = mi.get("api_server_dispatch_finish_ts")
    fin = mi.get("request_finished_ts")
    return {"wall": wall, "e2e": mi.get("e2e_latency", 0),
            "disp": (disp - rcv) if (disp and rcv) else None,
            "pf": (fin - disp) if (fin and disp) else None,
            "q": mi.get("queue_time", 0), "pt": mi.get("prompt_tokens")}

REPEATS = 3  # 每个并发级别重复轮数

def _stat(v):
    return {"mean": round(statistics.mean(v), 3),
            "std": round(statistics.pstdev(v), 3) if len(v) > 1 else 0.0,
            "median": round(statistics.median(v), 3),
            "p90": round(pct(v, 0.9), 3), "max": round(max(v), 3),
            "min": round(min(v), 3), "n": len(v)}

def run(label, router, repeats=REPEATS, payload_file=PAYLOAD_FILE):
    payloads = json.load(open(payload_file))
    print(f"[{label}] {len(payloads)} 个真实payload({payload_file}), router={router}, 每级重复 {repeats} 轮")
    out = {"label": label, "router": router, "n_payload": len(payloads), "repeats": repeats, "levels": {}}
    for N in CONC_LEVELS:
        agg = {k: [] for k in ["wall", "e2e", "disp", "pf", "q"]}
        pts = []; n_ok = 0; n_err = 0; round_walls = []
        for _ in range(repeats):
            T = time.time()
            # 预序列化每个并发位的请求体(轮转取 payload)，只把单条 bytes 发给 worker
            bodies = [(json.dumps(payloads[i % len(payloads)]).encode(), router) for i in range(N)]
            with Pool(N) as pool:
                res = pool.map(_one, bodies)
            round_walls.append(time.time() - T)
            for r in res:
                if "err" in r: n_err += 1; continue
                n_ok += 1
                for k in agg:
                    if r.get(k) is not None: agg[k].append(r[k])
                if r.get("pt"): pts.append(r["pt"])
            time.sleep(1)
        lvl = {"n_ok": n_ok, "n_err": n_err, "rounds": repeats,
               "round_wall_s": [round(w, 1) for w in round_walls]}
        for k, v in agg.items():
            if v: lvl[k] = _stat(v)
        if pts: lvl["prompt_tokens_median"] = int(statistics.median(pts))
        out["levels"][N] = lvl
        e = lvl.get("e2e", {}); d = lvl.get("disp", {}); pf = lvl.get("pf", {})
        print(f"  N={N:2d}  e2e={e.get('mean','?')}±{e.get('std','?')}(中位{e.get('median','?')})  "
              f"dispatch={d.get('mean','?')}±{d.get('std','?')}  "
              f"prefill+dec={pf.get('mean','?')}±{pf.get('std','?')}  "
              f"({n_ok}ok/{n_err}err,{repeats}轮)")
    f = f"/tmp/bench_{label}.json"
    json.dump(out, open(f, "w"), indent=2)
    print(f"已存 {f}")

def _g(d, N, key, stat="mean"):
    lv = d["levels"].get(str(N), d["levels"].get(N, {}))
    return lv.get(key, {}).get(stat)

def compare(la, lb):
    a = json.load(open(f"/tmp/bench_{la}.json")); b = json.load(open(f"/tmp/bench_{lb}.json"))
    print(f"\n=== 对比 {la} vs {lb} (mean±std, 每级{a.get('repeats','?')}轮) ===")
    print(f"{'并发':>4} | {'dispatch '+la:>18} {'dispatch '+lb:>18} {'变化':>7} | {'e2e '+la:>16} {'e2e '+lb:>16}")
    for N in CONC_LEVELS:
        da, sa = _g(a, N, "disp", "mean"), _g(a, N, "disp", "std")
        db, sb = _g(b, N, "disp", "mean"), _g(b, N, "disp", "std")
        ea, eas = _g(a, N, "e2e", "mean"), _g(a, N, "e2e", "std")
        eb, ebs = _g(b, N, "e2e", "mean"), _g(b, N, "e2e", "std")
        if da and db:
            chg = f"{(db-da)/da*100:+.0f}%"
            print(f"{N:>4} | {f'{da}±{sa}':>18} {f'{db}±{sb}':>18} {chg:>7} | {f'{ea}±{eas}':>16} {f'{eb}±{ebs}':>16}")

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2:
        # 用法: bench_io_workers.py <label> [router_url] [payload_file]
        run(sys.argv[1],
            sys.argv[2] if len(sys.argv) > 2 else ROUTER,
            payload_file=sys.argv[3] if len(sys.argv) > 3 else PAYLOAD_FILE)
    else:
        print(__doc__)
