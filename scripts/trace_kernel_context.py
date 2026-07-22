"""Attribute GPU kernels in a vLLM torch-profiler trace by timeline context.

For every kernel whose name contains a ``--match`` substring, report the
kernels that immediately precede and follow it on the same GPU stream. That
pinpoints which model region owns an anonymous kernel (e.g. a bf16 cutlass
GEMM or an ``at::native`` copy): its neighbors are distinctive (quantizer /
DenseGemm / flash attention / GDN gating / sampler...).

Usage:
    python scripts/trace_kernel_context.py /tmp/vllm_prof/<trace>.json.gz \
        --match cutlass_80_wmma --match direct_copy --match gemvx
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict


def _load(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _short(name: str, width: int = 72) -> str:
    return name if len(name) <= width else name[: width - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="chrome trace .json or .json.gz")
    ap.add_argument(
        "--match",
        action="append",
        required=True,
        help="substring of kernel names to attribute (repeatable)",
    )
    ap.add_argument("--top", type=int, default=12, help="context rows per match")
    args = ap.parse_args()

    data = _load(args.trace)
    events = data.get("traceEvents", data if isinstance(data, list) else [])

    # GPU kernel events grouped per stream track, in timeline order.
    streams: dict[tuple, list] = defaultdict(list)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        if str(ev.get("cat", "")).lower() not in (
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        ):
            continue
        streams[(ev.get("pid"), ev.get("tid"))].append(ev)
    for track in streams.values():
        track.sort(key=lambda e: float(e.get("ts", 0.0)))

    for needle in args.match:
        n_hits = 0
        total_us = 0.0
        prev_ctx: Counter = Counter()
        next_ctx: Counter = Counter()
        pair_ctx: Counter = Counter()
        for track in streams.values():
            for i, ev in enumerate(track):
                if needle not in ev["name"]:
                    continue
                n_hits += 1
                total_us += float(ev.get("dur", 0.0))
                prev_name = track[i - 1]["name"] if i > 0 else "<start>"
                next_name = track[i + 1]["name"] if i + 1 < len(track) else "<end>"
                prev_ctx[prev_name] += 1
                next_ctx[next_name] += 1
                pair_ctx[(prev_name, next_name)] += 1

        print("=" * 100)
        print(f"match: {needle!r}  hits: {n_hits}  total: {total_us / 1e3:.2f} ms")
        if not n_hits:
            continue
        print(f"\n  top (prev -> SELF -> next) contexts:")
        for (p, n), c in pair_ctx.most_common(args.top):
            print(f"  {c:7d}x")
            print(f"      prev: {_short(p)}")
            print(f"      next: {_short(n)}")
        print(f"\n  top predecessors:")
        for name, c in prev_ctx.most_common(6):
            print(f"  {c:7d}x  {_short(name, 88)}")
        print(f"\n  top successors:")
        for name, c in next_ctx.most_common(6):
            print(f"  {c:7d}x  {_short(name, 88)}")
        print()


if __name__ == "__main__":
    main()
