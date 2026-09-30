#!/usr/bin/env python3
"""Analyze sglang prefix-cache effectiveness from a run's engine logs.

Purpose:判断 consistent_hashing 路由是否真的提升了 prefix cache 命中（同一轨迹
的连续 step 是否落到同一 engine、历史是否被复用）。

Engine 日志（``SGLangEngine pid=... Prefill batch, ... #new-token, #cached-token``）
不携带 routing-key / session_id，所以无法直接"按 routing key 分组"。改为用三个
可从日志算出的、能一锤定音的代理指标：

1. 全局 prefix-cache 命中率 = Σcached / Σ(new+cached)。consistent_hashing 生效时
   应显著高于 round-robin/失效时（基线 ~20%）。
2. 每个 engine 的命中率 + 负载（prefill 次数）。consistent_hashing 把会话钉到固定
   worker；若某些 engine 命中率高、另一些低，或负载极不均，说明钉定/容量有问题。
3. 命中率随时间演化（按 prefill 出现顺序分桶）。真正的会话亲和应让命中率随轨迹推进
   爬升（engine 累积了该轨迹历史）。一直平 = 没复用上。

也解析 RouterArgs（policy / cache_threshold / assignment_mode / max_tree_size），
这些直接决定 consistent_hashing 的实际行为。

Usage:
    python scripts/analyze_prefix_cache.py <engine_log_path_or_glob>
    python scripts/analyze_prefix_cache.py wandb/run-XXXX/files/output_*.log
"""

from __future__ import annotations

import glob
import re
import sys
from collections import defaultdict

PREFILL_RE = re.compile(
    r"SGLangEngine pid=(\d+).*?Prefill batch.*?#new-token: (\d+), #cached-token: (\d+)"
)
ROUTER_RE = re.compile(r"Launch router with args: RouterArgs\((.*)\)")
ROUTER_KEYS = ("policy", "cache_threshold", "assignment_mode", "max_tree_size", "eviction_interval_secs")


def _fmt_pct(num: float, den: float) -> str:
    return f"{100 * num / den:.1f}%" if den else "n/a"


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths: list[str] = []
    for arg in sys.argv[1:]:
        paths.extend(glob.glob(arg))
    if not paths:
        print(f"No files matched: {sys.argv[1:]}", file=sys.stderr)
        sys.exit(1)

    # (new, cached) prefill records per engine, in order of appearance.
    per_engine: dict[str, list[tuple[int, int]]] = defaultdict(list)
    ordered: list[tuple[int, int]] = []  # global order for time-evolution buckets
    router_cfg: dict[str, str] = {}

    for path in paths:
        for line in open(path, errors="ignore"):
            if not router_cfg:
                rm = ROUTER_RE.search(line)
                if rm:
                    body = rm.group(1)
                    for k in ROUTER_KEYS:
                        km = re.search(rf"{k}=([^,)]+)", body)
                        if km:
                            router_cfg[k] = km.group(1).strip().strip("'\"")
            m = PREFILL_RE.search(line)
            if m:
                pid, new, cached = m.group(1), int(m.group(2)), int(m.group(3))
                # Skip warmup/1-token prefills (no real prompt).
                if new + cached <= 1:
                    continue
                per_engine[pid].append((new, cached))
                ordered.append((new, cached))

    if not ordered:
        print("No real prefill records found (only warmup?). Check the log path / that the run did rollout.", file=sys.stderr)
        sys.exit(1)

    tot_new = sum(n for n, _ in ordered)
    tot_cached = sum(c for _, c in ordered)

    print("=== Router config ===")
    if router_cfg:
        for k in ROUTER_KEYS:
            print(f"  {k} = {router_cfg.get(k, '?')}")
    else:
        print("  (router launch line not found in these logs)")

    print("\n=== Global prefix-cache hit rate ===")
    print(f"  prefills: {len(ordered)}")
    print(f"  hit rate = {_fmt_pct(tot_cached, tot_new + tot_cached)}  (Σcached={tot_cached:,} / Σ(new+cached)={tot_new + tot_cached:,})")
    print(f"  baseline (routing off / round-robin) was ~20%. consistent_hashing should be clearly higher.")

    print("\n=== Per-engine (load balance + hit rate) ===")
    print(f"  {'engine pid':<14}{'prefills':>9}{'hit rate':>10}{'avg new-tok':>13}")
    print("  " + "-" * 44)
    for pid, recs in sorted(per_engine.items(), key=lambda kv: -len(kv[1])):
        n = sum(x for x, _ in recs)
        c = sum(y for _, y in recs)
        avg_new = n / len(recs) if recs else 0
        print(f"  {pid:<14}{len(recs):>9}{_fmt_pct(c, n + c):>10}{avg_new:>13.0f}")
    counts = [len(r) for r in per_engine.values()]
    if len(counts) > 1:
        print(f"  load spread: min {min(counts)} / max {max(counts)} prefills per engine "
              f"({'BALANCED' if max(counts) <= 2 * max(1, min(counts)) else 'SKEWED — some engines idle while others busy'})")

    print("\n=== Hit-rate evolution (global order, 5 buckets) ===")
    nb = 5
    sz = max(1, len(ordered) // nb)
    print("  (consistent_hashing 生效时应随时间爬升：engine 累积了各轨迹历史)")
    for b in range(nb):
        chunk = ordered[b * sz : (b + 1) * sz] if b < nb - 1 else ordered[b * sz :]
        if not chunk:
            continue
        cn = sum(x for x, _ in chunk)
        cc = sum(y for _, y in chunk)
        print(f"  bucket {b + 1}/{nb} ({len(chunk):4d} prefills): hit {_fmt_pct(cc, cn + cc)}")

    # Verdict
    overall = 100 * tot_cached / (tot_new + tot_cached) if (tot_new + tot_cached) else 0
    print("\n=== Verdict ===")
    if overall >= 50:
        print(f"  ✅ 命中率 {overall:.0f}% — consistent_hashing 明显起效，prefix 大量复用。")
    elif overall >= 35:
        print(f"  ⚠️  命中率 {overall:.0f}% — 比基线略好但远未到位。可能：cache_threshold/max_tree_size 限制、"
              f"engine 数 vs 并发轨迹比例太高(KV 互相驱逐)、或会话亲和未真正生效。")
    else:
        print(f"  ❌ 命中率 {overall:.0f}% — 与基线(~20%)无本质差别。consistent_hashing 很可能未真正把同轨迹钉到同 engine"
              f"（检查 header 是否送达 router / session_id 是否每步一致）。")


if __name__ == "__main__":
    main()
