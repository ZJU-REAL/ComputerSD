#!/bin/bash
# DEBUG: 汇总所有 worker 进程当前持有的 env lease 数 = 真实环境占用数。
# 数据来自 trajectory.py 的 _dump_inflight: 所有进程共享一个 inflight.json {pid:[leases]}。
# 用法: bash scripts/count_env.sh [GUI_RESULT_DIR]   (不传则用 $GUI_RESULT_DIR 或 results/最新)
DIR="${1:-${GUI_RESULT_DIR:-}}"
if [[ -z "$DIR" ]]; then
  DIR="$(ls -dt "$(dirname "$0")"/../results/*/ 2>/dev/null | head -1)"
fi
F="$DIR/_env_inflight/inflight.json"
if [[ ! -f "$F" ]]; then
  echo "no inflight file: $F"; exit 1
fi
python3 - "$F" <<'PY'
import sys, json
data = json.load(open(sys.argv[1]))
total = sum(len(v) for v in data.values())
print(f"真实占用 env = {total}  (across {len(data)} worker procs holding leases)")
for pid, leases in sorted(data.items(), key=lambda x: -len(x[1]))[:10]:
    print(f"  pid {pid}: {len(leases)}")
PY
