"""Summarize GPU kernel time in a vLLM torch-profiler (Chrome) trace.

Usage:
    python scripts/summarize_vllm_trace.py /tmp/vllm_prof/<trace>.json[.gz] [--top 40]

Prints total traced GPU kernel time and the top-N kernels by summed duration,
so a 1.25 s decode step can be apportioned to FP6 GEMM / quantizer / attention /
GDN / sampler / elementwise overhead at a glance.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict


def _load(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="chrome trace .json or .json.gz from torch profiler")
    ap.add_argument("--top", type=int, default=40, help="rows to print")
    args = ap.parse_args()

    data = _load(args.trace)
    events = data.get("traceEvents", data if isinstance(data, list) else [])

    kern_us: dict[str, float] = defaultdict(float)
    kern_n: dict[str, int] = defaultdict(int)
    t_min, t_max = float("inf"), float("-inf")
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = str(ev.get("cat", "")).lower()
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur = float(ev.get("dur", 0.0))
        name = ev["name"]
        kern_us[name] += dur
        kern_n[name] += 1
        ts = float(ev.get("ts", 0.0))
        t_min = min(t_min, ts)
        t_max = max(t_max, ts + dur)

    if not kern_us:
        print("no GPU kernel events found (is this a torch profiler chrome trace?)")
        sys.exit(1)

    total_us = sum(kern_us.values())
    wall_us = t_max - t_min
    print(f"traced wall span : {wall_us / 1e3:10.1f} ms")
    print(f"GPU kernel time  : {total_us / 1e3:10.1f} ms  ({100 * total_us / wall_us:.0f}% of span)")
    print()
    print(f"{'total ms':>10} {'calls':>8} {'avg us':>9}  kernel")
    print("-" * 100)
    rows = sorted(kern_us.items(), key=lambda kv: kv[1], reverse=True)
    for name, us in rows[: args.top]:
        n = kern_n[name]
        print(f"{us / 1e3:10.2f} {n:8d} {us / n:9.1f}  {name[:120]}")
    rest = sum(us for _, us in rows[args.top :])
    if rest:
        print(f"{rest / 1e3:10.2f} {'':8} {'':9}  ... remaining {len(rows) - args.top} kernels")


if __name__ == "__main__":
    main()
