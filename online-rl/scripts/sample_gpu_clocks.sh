#!/usr/bin/env bash
# Read-only GPU clock/power/throttle sampler — verify the "training prefill is
# slow because GPUs throttle under load" hypothesis (cause C).
#
# WHY: training sglang prefill measures ~1585 tok/s vs eval ~2879 tok/s (1.8x).
# Config / sglang version / GPU model are all identical (verified from logs), so
# the leading remaining suspect is the GPUs running at a lower SM clock under the
# sustained 8-card prefill load of training. This cannot be seen at idle (idle
# already sits at max 1980 MHz). It must be sampled WHILE training is doing
# rollout. This script ONLY reads nvidia-smi — it injects no load and does not
# touch the live sglang engines.
#
# Usage (run on the training node, DURING a rollout/generate phase):
#   bash scripts/sample_gpu_clocks.sh            # samples 60s @ 0.5s, all GPUs
#   DURATION=120 INTERVAL=0.5 bash scripts/sample_gpu_clocks.sh
#
# Interpretation:
#   - SM clock stays ~1980 MHz, throttle=0x0  -> NOT throttling; C is ruled out,
#     the 1.8x must come from elsewhere (revisit kernel/config or measurement).
#   - SM clock drops well below 1980 with throttle bits set (sw_power_cap /
#     hw_thermal_slowdown / hw_power_brake) -> throttling confirmed; fix power
#     budget / cooling / stagger the 8 engines.
set -euo pipefail

DURATION="${DURATION:-60}"
INTERVAL="${INTERVAL:-0.5}"
OUT="${OUT:-/tmp/gpu_clocks_$(date +%H%M%S).csv}"

echo "timestamp,gpu,sm_mhz,max_mhz,power_w,power_limit_w,throttle,util,temp_c" > "$OUT"
n=$(python3 -c "print(int($DURATION/$INTERVAL))")
echo "Sampling ${DURATION}s @ ${INTERVAL}s ($n rows) -> $OUT  (read-only, no load injected)"
for _ in $(seq 1 "$n"); do
  nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.max.sm,power.draw,power.limit,clocks_throttle_reasons.active,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits >> "$OUT"
  sleep "$INTERVAL"
done

echo "=== Per-GPU summary (rows with util>0 = under load) ==="
python3 - "$OUT" <<'PY'
import sys, csv, statistics
from collections import defaultdict
rows=defaultdict(list)
with open(sys.argv[1]) as f:
    r=csv.DictReader(f)
    for x in r:
        try:
            g=int(x['gpu']); sm=float(x['sm_mhz']); util=float(x['util']); pw=float(x['power_w'])
        except: continue
        rows[g].append((sm,util,pw,x['throttle'].strip()))
print(f"{'gpu':>3} {'load_rows':>9} {'sm@load_med':>11} {'sm@load_min':>11} {'pw@load_med':>11} {'throttle@load':>22}")
for g in sorted(rows):
    load=[t for t in rows[g] if t[1]>0]
    if not load:
        print(f"{g:>3} {'(idle whole window)':>9}"); continue
    sm=[t[0] for t in load]; pw=[t[2] for t in load]
    thr=set(t[3] for t in load if t[3]!='0x0000000000000000')
    print(f"{g:>3} {len(load):>9} {statistics.median(sm):>11.0f} {min(sm):>11.0f} {statistics.median(pw):>11.0f} {(','.join(thr) or '0x0 (none)'):>22}")
print("\nmax SM = 1980 MHz. If sm@load stays ~1980 & throttle=none -> NOT throttling (C ruled out).")
PY
