#!/usr/bin/env python3
"""Aggregate GUI rollout profiling output across many trajectories.

Each trajectory writes a ``timings.json`` (per-stage wall-clock totals: acquire/
reset/turn_loop/sglang_generate/env_step/gen_post/... see rollout/_timing.py) and
a ``step_times.jsonl`` (one line per policy step). This script sweeps a results
tree, sums each stage across all trajectories, and prints a table sorted by total
time with each stage's share — so you can see where wall-clock actually goes
without cat-ing files one by one.

Usage:
    python scripts/agg_timings.py [RESULTS_DIR]
    python scripts/agg_timings.py results/slime_gui_8b_..._20260611_115840
    python scripts/agg_timings.py --steps        # also break down step_times.jsonl
    python scripts/agg_timings.py --top 20 results/

RESULTS_DIR defaults to $GUI_RESULT_DIR or ./results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _fmt_secs(s: float) -> str:
    if s >= 3600:
        return f"{s/3600:.2f}h"
    if s >= 60:
        return f"{s/60:.2f}m"
    return f"{s:.2f}s"


def aggregate_timings(root: Path) -> tuple[dict[str, dict], int]:
    """Sum each stage across all timings.json under ``root``.

    Returns ({stage: {total_s, count, traj}}, n_trajectories). ``count`` is the
    total number of span occurrences (e.g. one per step for sglang_generate);
    ``traj`` is how many trajectories reported that stage.
    """
    agg: dict[str, dict] = {}
    n_traj = 0
    for path in root.rglob("timings.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # skip half-written / corrupt files
            print(f"WARN skip {path}: {e}", file=sys.stderr)
            continue
        n_traj += 1
        for stage, v in data.items():
            slot = agg.setdefault(stage, {"total_s": 0.0, "count": 0, "traj": 0})
            slot["total_s"] += float(v.get("total_s", 0.0))
            slot["count"] += int(v.get("count", 0))
            slot["traj"] += 1
    return agg, n_traj


def aggregate_steps(root: Path) -> tuple[dict[str, dict], int]:
    """Sum per-step phase fields across all step_times.jsonl under ``root``.

    Step fields: heartbeat / build_msg / sglang / parse / env_step / step_time.
    Returns ({field: {total_s, count}}, n_steps).
    """
    fields = ("heartbeat", "build_msg", "sglang", "parse", "env_step", "step_time")
    agg: dict[str, dict] = {f: {"total_s": 0.0, "count": 0} for f in fields}
    n_steps = 0
    for path in root.rglob("step_times.jsonl"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                n_steps += 1
                for f in fields:
                    if f in row:
                        agg[f]["total_s"] += float(row[f])
                        agg[f]["count"] += 1
        except Exception as e:
            print(f"WARN skip {path}: {e}", file=sys.stderr)
    return agg, n_steps


def step0_stats(root: Path) -> dict[str, Any]:
    """Stats over the FIRST step (step_idx==0) of every trajectory.

    step 0 is the cleanest cross-run comparison point: the prompt has no
    accumulated image history yet, so it isolates the base per-step cost
    (one sglang call + one env action) before context grows. Returns
    {n, step_time:{mean,median,min,max}, sglang_mean, env_step_mean}.
    """
    import statistics
    first: list[dict] = []
    for path in root.rglob("step_times.jsonl"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("step_idx") == 0:
                    first.append(row)
                    break  # only the first step of this trajectory
        except Exception as e:
            print(f"WARN skip {path}: {e}", file=sys.stderr)
    if not first:
        return {"n": 0}
    st = [d.get("step_time", 0.0) for d in first]
    return {
        "n": len(first),
        "step_time": {
            "mean": round(statistics.mean(st), 1),
            "median": round(statistics.median(st), 1),
            "min": round(min(st), 1),
            "max": round(max(st), 1),
        },
        "sglang_mean": round(statistics.mean([d.get("sglang", 0.0) for d in first]), 1),
        "env_step_mean": round(statistics.mean([d.get("env_step", 0.0) for d in first]), 1),
    }


def print_step0(root: Path) -> None:
    s = step0_stats(root)
    print("\n=== Step 0 (first step) timing — cleanest cross-run baseline ===")
    if not s.get("n"):
        print("  (no step_times.jsonl with step_idx==0)")
        return
    t = s["step_time"]
    print(f"  trajectories: {s['n']}")
    print(f"  step0 total : mean {t['mean']}s  median {t['median']}s  min {t['min']}s  max {t['max']}s")
    print(f"    sglang    : mean {s['sglang_mean']}s")
    print(f"    env_step  : mean {s['env_step_mean']}s")


def print_gen_post_breakdown(agg: dict[str, dict]) -> None:
    """Split gen_post into server-side vs network/router, and diagnose the cause.

    gen_post (client wall-clock of the HTTP POST) decomposes as:
        gen_post = network/router (gen_post - e2e_latency) + e2e_latency
        e2e_latency (server) = srv_queue (wait for GPU) + srv_dispatch
                               (tokenize+dispatch) + prefill + decode
    Requires sglang_enable_metrics=True so meta_info carries these fields; if
    gen_post is present but e2e_latency is absent/zero, metrics were off (or the
    run predates the qwen3vl_agent recording code) — say so instead of guessing.
    """
    gp = agg.get("gen_post")
    if not gp or not gp.get("count"):
        return
    print("\n=== gen_post breakdown (why each generate call costs what it does) ===")
    n = gp["count"]
    gp_avg = gp["total_s"] / n

    def avg(key: str) -> float | None:
        v = agg.get(key)
        if not v or not v.get("count"):
            return None
        return v["total_s"] / v["count"]

    e2e = avg("e2e_latency")
    if e2e is None:
        print(f"  gen_post avg {gp_avg:.3f}s/call  (n={n})")
        print("  server-side fields ABSENT — sglang_enable_metrics was off, or this")
        print("  run predates the meta_info recording in qwen3vl_agent. Restart with")
        print("  metrics on to populate e2e_latency/srv_queue_time/srv_dispatch.")
        return

    queue = avg("srv_queue_time") or 0.0
    dispatch = avg("srv_dispatch") or 0.0
    net = gp_avg - e2e               # client/network/router overhead outside engine
    compute = e2e - queue - dispatch  # prefill + decode (no separate field)
    comp_tok = avg("completion_tokens")
    prompt_tok = avg("prompt_tokens")

    def row(label: str, val: float) -> None:
        print(f"    {label:<28}{val:>8.3f}s {100.0*val/gp_avg:>6.1f}% of gen_post")

    print(f"  gen_post avg {gp_avg:.3f}s/call  (n={n})")
    row("e2e_latency (server total)", e2e)
    row("├ srv_queue (wait for GPU)", queue)
    row("├ srv_dispatch (tok+route)", dispatch)
    row("└ prefill+decode (compute)", compute)
    row("network/router (gp - e2e)", net)
    if prompt_tok is not None or comp_tok is not None:
        pt = f"{prompt_tok:.0f}" if prompt_tok is not None else "?"
        ct = f"{comp_tok:.0f}" if comp_tok is not None else "?"
        print(f"    tokens/req: prompt {pt}  completion {ct}")

    # Diagnosis: which slice dominates → which bottleneck.
    parts = {"srv_queue": queue, "prefill+decode": compute, "network/router": net}
    worst = max(parts, key=parts.get)
    print("  → dominant cost:", worst, end="  ")
    if worst == "srv_queue":
        print("→ requests waiting for the GPU (engine saturated under concurrency).")
    elif worst == "prefill+decode":
        print("→ compute itself; check prompt_tokens (multimodal prefill is heavy).")
    else:
        print("→ outside the engine: network / sgl-router dispatch / client send.")
        print("    cross-check `Failed to send typed request` count in the train log —")
        print("    router retries show up here, not in any server-side field.")


def print_table(title: str, agg: dict[str, dict], denom_key: str | None) -> None:
    """Print stages sorted by total_s with share-of-`denom_key` percentages."""
    if not agg:
        print(f"\n{title}: (no data)")
        return
    denom = agg[denom_key]["total_s"] if denom_key and denom_key in agg else sum(
        v["total_s"] for v in agg.values()
    )
    denom = denom or 1.0
    print(f"\n{title}")
    print(f"{'stage':<26}{'total':>12}{'share':>9}{'count':>9}{'avg/call':>12}")
    print("-" * 68)
    for stage, v in sorted(agg.items(), key=lambda x: -x[1]["total_s"]):
        total = v["total_s"]
        cnt = v.get("count", 0)
        avg = total / cnt if cnt else 0.0
        share = 100.0 * total / denom
        print(f"{stage:<26}{_fmt_secs(total):>12}{share:>8.1f}%{cnt:>9}{avg:>11.3f}s")


# A/B/C/D layer nesting for the tree view. Top-level roots are printed first,
# each with its children indented underneath. A child's share is relative to its
# parent; "(self)" is the parent's unaccounted-for remainder.
# B-layer children that live alongside turn_loop in timings.json (episode_run
# itself is only on sample.metadata, not on disk, so we synthesize a virtual
# "trajectory" root summing these one-shot stages + turn_loop).
_B_LAYER = ["acquire", "reset", "turn_loop", "prm_collect", "evaluate", "close"]
# E layer — server-side breakdown of one gen_post (HTTP round-trip). These are
# NOT timed spans: qwen3vl_agent._add()s them from sglang meta_info (needs
# sglang_enable_metrics=True). All seconds. Their sum is < gen_post; the leftover
# (gen_post - e2e_latency) is pure network/router/client overhead outside the
# engine, surfaced as the "(self)" line. srv_dispatch may have a smaller count
# than e2e_latency (only recorded when meta_info carries the dispatch timestamps).
_TREE = {
    "trajectory": _B_LAYER,  # virtual root: B layer
    "turn_loop": ["heartbeat", "build_policy_messages", "sglang_generate", "parse_response", "env_step"],  # C layer
    "sglang_generate": ["gen_apply_chat_template", "gen_extract_mm", "gen_post", "gen_decode"],  # D layer
    "gen_post": ["e2e_latency", "srv_queue_time", "srv_dispatch"],  # E layer (server-side, seconds)
}
# Keys whose accumulated value is a TOKEN COUNT, not seconds (see qwen3vl_agent).
# Their avg/call = avg tokens/request; never format these with _fmt_secs.
_TOKEN_KEYS = {"completion_tokens", "prompt_tokens"}


def print_tree(title: str, agg: dict[str, dict]) -> None:
    """Render the A/B/C/D hierarchy as an indented tree with parent-relative %.

    Each line: <indented stage> <total> <share-of-parent> <count> <avg/call>.
    Children listed in _TREE are nested under their parent; a parent's leftover
    time (total minus sum of listed children) prints as a ``(self)`` line.
    """
    if not agg:
        print(f"\n{title}: (no data)")
        return

    # Synthesize the virtual "trajectory" root: sum of B-layer stages present.
    agg = dict(agg)  # don't mutate caller's dict
    b_present = [s for s in _B_LAYER if s in agg]
    if b_present and "trajectory" not in agg:
        agg["trajectory"] = {
            "total_s": sum(agg[s]["total_s"] for s in b_present),
            "count": max(agg[s].get("count", 0) for s in b_present),
        }

    print(f"\n{title}")
    print(f"{'stage':<30}{'total':>11}{'/parent':>9}{'count':>8}{'avg/call':>11}")
    print("-" * 69)

    printed: set[str] = set()

    def line(stage: str, depth: int, parent_total: float | None) -> None:
        v = agg.get(stage)
        if v is None:
            return
        total = v["total_s"]
        cnt = v.get("count", 0)
        avg = total / cnt if cnt else 0.0
        share = (100.0 * total / parent_total) if parent_total else 100.0
        indent = "  " * depth + ("└ " if depth else "")
        name = f"{indent}{stage}"
        print(f"{name:<30}{_fmt_secs(total):>11}{share:>8.1f}%{cnt:>8}{avg:>10.3f}s")
        printed.add(stage)
        # recurse into children
        children = _TREE.get(stage, [])
        child_sum = 0.0
        for c in children:
            if c in agg:
                child_sum += agg[c]["total_s"]
                line(c, depth + 1, total)
        if children and total - child_sum > 0.01:
            rem = total - child_sum
            rname = "  " * (depth + 1) + "└ (self)"
            rshare = 100.0 * rem / total if total else 0.0
            print(f"{rname:<30}{_fmt_secs(rem):>11}{rshare:>8.1f}%{'':>8}{'':>10} ")

    # Anchor on the virtual trajectory root (B layer); else turn_loop.
    root = "trajectory" if "trajectory" in agg else ("turn_loop" if "turn_loop" in agg else None)
    if root:
        line(root, 0, None)
    # Print any stages not covered by the tree (e.g. A-layer external) as flat tail.
    leftover = [s for s in sorted(agg, key=lambda x: -agg[x]["total_s"]) if s not in printed]
    if leftover:
        print("  (not nested under the root above:)")
        for s in leftover:
            v = agg[s]; cnt = v.get("count", 0)
            avg = v["total_s"] / cnt if cnt else 0.0
            if s in _TOKEN_KEYS:  # value is a token count, not seconds
                print(f"  {s:<28}{v['total_s']:>10.0f}t{'':>9}{cnt:>8}{avg:>9.0f}t")
            else:
                print(f"  {s:<28}{_fmt_secs(v['total_s']):>11}{'':>9}{cnt:>8}{avg:>10.3f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", nargs="?", default=None, help="results tree to scan")
    ap.add_argument("--steps", action="store_true", help="also aggregate step_times.jsonl")
    ap.add_argument("--top", type=int, default=0, help="only print the top-N slowest stages (0=all)")
    ap.add_argument("--flat", action="store_true", help="flat sorted table instead of the A/B/C/D tree")
    args = ap.parse_args()

    root = Path(args.results_dir or os.getenv("GUI_RESULT_DIR", "./results"))
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    agg, n_traj = aggregate_timings(root)
    print(f"Scanned {root}  —  {n_traj} trajectories (timings.json)")
    if args.flat:
        flat = dict(sorted(agg.items(), key=lambda x: -x[1]["total_s"])[: args.top]) if args.top else agg
        denom = "turn_loop" if "turn_loop" in flat else None
        print_table("=== Per-stage totals (flat, across all trajectories) ===", flat, denom)
    else:
        # Default: hierarchical A/B/C/D tree with parent-relative percentages.
        print_tree("=== Per-stage hierarchy (across all trajectories) ===", agg)

    # Always print the step-0 baseline (cleanest cross-run comparison point).
    print_step0(root)

    # gen_post server-side breakdown (populated when sglang_enable_metrics=True).
    print_gen_post_breakdown(agg)

    if args.steps:
        sagg, n_steps = aggregate_steps(root)
        print(f"\nScanned {n_steps} steps (step_times.jsonl)")
        print_table("=== Per-step phase totals ===", sagg, "step_time")


if __name__ == "__main__":
    main()
