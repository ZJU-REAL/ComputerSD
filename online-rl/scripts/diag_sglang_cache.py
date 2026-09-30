#!/usr/bin/env python3
"""Diagnose training-side sglang prefix-cache hit rate & concurrency.

WHY: training eval (~7.9s/sglang call) is ~4x slower than the standalone 8gpu
eval (~1.7s) despite same 8 cards / same model / same ~20% nominal cache. The
leading hypothesis is that slime's cache_aware ROUTER scatters a trajectory's
consecutive steps across different engines, destroying each engine's radix
prefix cache (every step re-prefills ~7600 tokens). The standalone eval pins
each trajectory to ONE engine via OPENAI_BASE_URLS sharding, keeping the prefix
warm.

This script reads the sglang ENGINE stdout (Ray worker logs) and measures the
real per-engine cache hit rate and #running-req. Run it ON THE TRAINING NODE
(where the sglang engines actually run), ideally while / right after a rollout.

Cross-check target (from the standalone eval log): cache hit ≈ 22.3%.
- If training per-engine hit rate is MUCH lower (≈0-5%): router scatter confirmed.
- If it's similar (~20%): the bottleneck is elsewhere (client path / batching).

Usage:
    python scripts/diag_sglang_cache.py                 # auto: /tmp/ray/session_latest/logs
    python scripts/diag_sglang_cache.py path/to/ray-logs # explicit Ray logs dir
    python scripts/diag_sglang_cache.py file1.out file2.out ...
"""

from __future__ import annotations

import glob
import os
import re
import sys


def _find_log_files(argv: list[str]) -> list[str]:
    if argv:
        files: list[str] = []
        for a in argv:
            if os.path.isdir(a):
                files += glob.glob(os.path.join(a, "worker-*.out"))
                files += glob.glob(os.path.join(a, "**", "*.out"), recursive=True)
            else:
                files.append(a)
        return sorted(set(files))
    # Auto-discover Ray worker logs.
    candidates = []
    for base in ["/tmp/ray/session_latest/logs", *glob.glob("/tmp/ray/session_*/logs")]:
        candidates += glob.glob(os.path.join(base, "worker-*.out"))
    # Keep only files that actually contain sglang batch lines.
    out = []
    for f in sorted(set(candidates)):
        try:
            with open(f, "r", errors="ignore") as fh:
                head = fh.read(200000)
            if "Prefill batch" in head or "Decode batch" in head:
                out.append(f)
        except Exception:
            pass
    return out


_PREFILL = re.compile(r"Prefill batch.*?#new-token: (\d+).*?#cached-token: (\d+)")
_RUNREQ = re.compile(r"#running-req: (\d+)")


def main() -> None:
    files = _find_log_files(sys.argv[1:])
    if not files:
        print("No sglang engine logs found.")
        print("Pass the training node's Ray log dir, e.g.:")
        print("  python scripts/diag_sglang_cache.py /tmp/ray/session_latest/logs")
        print("Each sglang engine is a Ray actor; its stdout is worker-*.out there.")
        return

    print(f"Scanning {len(files)} log file(s) with sglang batch lines:\n")
    grand_new = grand_cached = 0
    runreq_hist: dict[int, int] = {}

    for f in files:
        new_tok = cached_tok = n_prefill = 0
        with open(f, "r", errors="ignore") as fh:
            for line in fh:
                if "Prefill batch" in line:
                    m = _PREFILL.search(line)
                    if m:
                        new_tok += int(m.group(1))
                        cached_tok += int(m.group(2))
                        n_prefill += 1
                mr = _RUNREQ.search(line)
                if mr:
                    r = int(mr.group(1))
                    runreq_hist[r] = runreq_hist.get(r, 0) + 1
        if n_prefill == 0:
            continue
        denom = new_tok + cached_tok
        hit = (cached_tok / denom * 100) if denom else 0.0
        grand_new += new_tok
        grand_cached += cached_tok
        print(f"  {os.path.basename(f):<40} prefills={n_prefill:>5}  "
              f"new={new_tok:>10}  cached={cached_tok:>10}  hit={hit:5.1f}%")

    denom = grand_new + grand_cached
    overall = (grand_cached / denom * 100) if denom else 0.0
    print("\n" + "=" * 60)
    print(f"OVERALL prefix-cache hit rate = {overall:.1f}%   "
          f"(cached={grand_cached}, new={grand_new})")
    print("Standalone eval reference     = 22.3%")
    print("=" * 60)
    if overall < 10:
        print(">>> VERDICT: training cache hit is FAR below eval's 22% — "
              "router scatter is destroying prefix cache. FIX = pin each "
              "trajectory to one engine (sharded routing) or change router policy.")
    elif overall < 18:
        print(">>> VERDICT: training cache hit is somewhat below eval — "
              "router scatter is a PARTIAL contributor.")
    else:
        print(">>> VERDICT: cache hit is comparable to eval — the 4x sglang gap "
              "is NOT mainly prefix-cache; look at client path (router/httpx) or batching.")

    if runreq_hist:
        print("\n#running-req distribution (concurrency on engines):")
        for r in sorted(runreq_hist)[:20]:
            print(f"  running-req={r:>3}: {runreq_hist[r]:>6} samples")


if __name__ == "__main__":
    main()
