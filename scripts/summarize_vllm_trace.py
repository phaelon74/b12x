#!/usr/bin/env python3
"""Attribute a vLLM Torch profiler chrome-trace into kernel/host classes.

Used for Phase A of the FP6 performance recovery plan: split a ctx-8192
prefill and a decode-window capture into GEMM / act-quant / host GS /
allreduce / attention / other so each Phase-B lever has a measured budget.

Accepts ``*.pt.trace.json`` or ``*.pt.trace.json.gz`` (vLLM torch profiler
output). Prints a ranked table and optional JSON.

Examples
--------
::

    python scripts/summarize_vllm_trace.py /tmp/vllm_prof/fp6_prefill/*.pt.trace.json.gz
    python scripts/summarize_vllm_trace.py /tmp/vllm_prof --json-out /tmp/attr.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class Bucket:
    name: str
    patterns: tuple[re.Pattern[str], ...]
    cuda_us: float = 0.0
    cpu_us: float = 0.0
    count: int = 0
    examples: list[str] = field(default_factory=list)


# Order matters: first match wins. Keep specific kernels ahead of generic aten.
_BUCKET_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "fp6_gemm",
        (
            r"dense_gemm",
            r"sparkinfer.*fp6",
            r"fp6_dense",
            r"mxfp6",
            r"mxf8f6f4",
            r"PackedB",
            r"b_packed",
        ),
    ),
    (
        "fp6_act_quant",
        (
            r"bf16_to_fp6",
            r"SmallMQuant",
            r"quantize.*fp6",
            r"quantize_block_fp8_e4m3",
            r"fp6.*quant",
        ),
    ),
    (
        "fp8_gemm",
        (
            r"cutlass_scaled_mm",
            r"cutlass.*fp8",
            r"blockwise.*fp8",
            r"fp8.*gemm",
            r"scaled_mm",
            r"cublasLt.*fp8",
            r"nvjet",
        ),
    ),
    (
        "fp8_act_quant",
        (
            r"quantize.*fp8",
            r"per_token.*fp8",
            r"dynamic.*fp8",
            r"fp8_quant",
        ),
    ),
    (
        "host_gs_chain",
        (
            r"aten::amax",
            r"aten::_aminmax",
            r"aten::linalg_vector_norm",
            r"aten::mul(\.|_)",
            r"aten::div(\.|_)",
            r"aten::reciprocal",
            r"aten::to(\.|$)",
            r"aten::_to_copy",
            r"aten::copy_",
            r"aten::empty",
            r"aten::zeros",
            r"aten::clamp",
        ),
    ),
    (
        "allreduce",
        (
            r"nccl",
            r"all_reduce",
            r"AllReduce",
            r"vllm::all_reduce",
            r"custom_all_reduce",
        ),
    ),
    (
        "attention",
        (
            r"flash",
            r"FlashInfer",
            r"flashinfer",
            r"fmha",
            r"attention",
            r"cutlass_mha",
            r"cudnn.*attn",
            r"reshape_and_cache",
            r"paged_attention",
            r"unified_attention",
        ),
    ),
    (
        "norm_act_misc",
        (
            r"rms_norm",
            r"rmsnorm",
            r"silu",
            r"SwiGLU",
            r"rotary",
            r"rope",
        ),
    ),
    (
        "sampler_logit",
        (
            r"sample",
            r"softmax",
            r"topk",
            r"gather",
            r"lm_head",
        ),
    ),
)


def _compile_buckets() -> list[Bucket]:
    return [
        Bucket(name=name, patterns=tuple(re.compile(p, re.I) for p in pats))
        for name, pats in _BUCKET_SPECS
    ]


def _open_trace(path: pathlib.Path) -> Any:
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _iter_events(trace: Any) -> Iterable[dict]:
    if isinstance(trace, dict):
        events = trace.get("traceEvents") or trace.get("events") or []
    elif isinstance(trace, list):
        events = trace
    else:
        events = []
    for ev in events:
        if isinstance(ev, dict):
            yield ev


def _duration_us(ev: dict) -> float:
    # Chrome trace uses microseconds in `dur`. Some exporters stash ns in args.
    if "dur" in ev:
        return float(ev["dur"])
    args = ev.get("args") or {}
    for key in ("dur", "duration", "cuda_time_total", "cpu_time_total"):
        if key in args:
            val = float(args[key])
            # Heuristic: values that look like nanoseconds.
            return val / 1000.0 if val > 1e7 else val
    return 0.0


def _classify(name: str, buckets: list[Bucket]) -> Bucket:
    for bucket in buckets:
        if any(p.search(name) for p in bucket.patterns):
            return bucket
    other = buckets[-1]
    return other


def summarize_trace(path: pathlib.Path) -> dict[str, Any]:
    buckets = _compile_buckets()
    other = Bucket(name="other", patterns=())
    buckets.append(other)

    trace = _open_trace(path)
    cuda_total = 0.0
    cpu_total = 0.0
    by_name_cuda: dict[str, float] = defaultdict(float)
    by_name_count: dict[str, int] = defaultdict(int)

    for ev in _iter_events(trace):
        # X = complete events; also accept C++ CUDA kernels tagged similarly.
        if ev.get("ph") not in ("X", "x"):
            continue
        name = str(ev.get("name") or "")
        if not name:
            continue
        cat = str(ev.get("cat") or "")
        dur = _duration_us(ev)
        if dur <= 0:
            continue
        is_cuda = bool(
            re.search(r"cuda|gpu|kernel", cat, re.I)
            or re.search(r"cuda|gemm|nccl|cutlass|triton", name, re.I)
        )
        bucket = _classify(name, buckets)
        if is_cuda:
            bucket.cuda_us += dur
            cuda_total += dur
            by_name_cuda[name] += dur
            by_name_count[name] += 1
        else:
            bucket.cpu_us += dur
            cpu_total += dur
        if len(bucket.examples) < 5 and name not in bucket.examples:
            bucket.examples.append(name)
        bucket.count += 1

    top_kernels = sorted(by_name_cuda.items(), key=lambda kv: kv[1], reverse=True)[:40]
    return {
        "path": str(path),
        "cuda_total_us": cuda_total,
        "cpu_total_us": cpu_total,
        "buckets": [
            {
                "name": b.name,
                "cuda_us": b.cuda_us,
                "cuda_pct": (100.0 * b.cuda_us / cuda_total) if cuda_total else 0.0,
                "cpu_us": b.cpu_us,
                "count": b.count,
                "examples": b.examples,
            }
            for b in buckets
            if b.cuda_us > 0 or b.cpu_us > 0 or b.count > 0
        ],
        "top_cuda_kernels": [
            {"name": n, "cuda_us": us, "count": by_name_count[n]}
            for n, us in top_kernels
        ],
    }


def _find_traces(root: pathlib.Path) -> list[pathlib.Path]:
    if root.is_file():
        return [root]
    found: list[pathlib.Path] = []
    for pat in ("*.pt.trace.json.gz", "*.pt.trace.json"):
        found.extend(root.rglob(pat))
    if not found:
        # Fallback: any chrome-trace-looking gzip JSON under the tree.
        for pat in ("*.json.gz", "*.json"):
            found.extend(
                p
                for p in root.rglob(pat)
                if "trace" in p.name.lower() or "chrome" in p.name.lower()
            )
    return sorted({p.resolve() for p in found})


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"\n=== {summary['path']} ===")
    cuda_ms = summary["cuda_total_us"] / 1000.0
    cpu_ms = summary["cpu_total_us"] / 1000.0
    print(f"CUDA total: {cuda_ms:.2f} ms    CPU total: {cpu_ms:.2f} ms")
    print(
        f"{'bucket':18s} {'cuda_ms':>10s} {'cuda_%':>8s} {'cpu_ms':>10s} {'count':>8s}"
    )
    for b in sorted(summary["buckets"], key=lambda x: x["cuda_us"], reverse=True):
        print(
            f"{b['name']:18s} {b['cuda_us']/1000:10.2f} {b['cuda_pct']:7.1f}% "
            f"{b['cpu_us']/1000:10.2f} {b['count']:8d}"
        )
    print("\nTop CUDA kernels:")
    for row in summary["top_cuda_kernels"][:20]:
        print(
            f"  {row['cuda_us']/1000:10.2f} ms  x{row['count']:<5d}  {row['name'][:100]}"
        )


def _self_test() -> None:
    """Offline smoke: classify a tiny synthetic chrome trace without GPU."""
    import tempfile

    fake = {
        "traceEvents": [
            {"ph": "X", "name": "sparkinfer::fp6_dense_linear", "cat": "cpu_op", "dur": 100.0},
            {"ph": "X", "name": "dense_gemm_mxfp6", "cat": "kernel", "dur": 5000.0},
            {"ph": "X", "name": "bf16_to_fp6_tma", "cat": "kernel", "dur": 200.0},
            {"ph": "X", "name": "aten::amax", "cat": "cpu_op", "dur": 50.0},
            {"ph": "X", "name": "ncclAllReduce", "cat": "kernel", "dur": 800.0},
            {"ph": "X", "name": "flash_attn_varlen", "cat": "kernel", "dur": 1200.0},
            {"ph": "X", "name": "cutlass_scaled_mm", "cat": "kernel", "dur": 3000.0},
        ]
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "fake.pt.trace.json"
        path.write_text(json.dumps(fake), encoding="utf-8")
        summary = summarize_trace(path)
    by_name = {b["name"]: b for b in summary["buckets"]}
    assert by_name["fp6_gemm"]["cuda_us"] == 5000.0
    assert by_name["fp6_act_quant"]["cuda_us"] == 200.0
    assert by_name["allreduce"]["cuda_us"] == 800.0
    assert by_name["attention"]["cuda_us"] == 1200.0
    assert by_name["fp8_gemm"]["cuda_us"] == 3000.0
    assert by_name["host_gs_chain"]["cpu_us"] == 50.0
    print("self-test OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Trace file(s) or directories containing vLLM torch profiler output",
    )
    parser.add_argument("--json-out", default="", help="Write combined JSON summary")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run offline classification smoke test and exit",
    )
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.paths:
        parser.error("paths required unless --self-test")

    traces: list[pathlib.Path] = []
    for raw in args.paths:
        traces.extend(_find_traces(pathlib.Path(raw)))
    if not traces:
        raise SystemExit("no trace files found")

    summaries = [summarize_trace(p) for p in traces]
    for s in summaries:
        _print_summary(s)

    # Cross-trace budget table when exactly two traces (e.g. fp6 vs fp8).
    if len(summaries) == 2:
        print("\n=== Side-by-side CUDA bucket ms ===")
        names = sorted(
            {
                b["name"]
                for s in summaries
                for b in s["buckets"]
            }
        )
        a, b = summaries
        print(f"{'bucket':18s} {pathlib.Path(a['path']).name[:22]:>22s} "
              f"{pathlib.Path(b['path']).name[:22]:>22s} {'ratio_a/b':>10s}")
        a_map = {x["name"]: x["cuda_us"] for x in a["buckets"]}
        b_map = {x["name"]: x["cuda_us"] for x in b["buckets"]}
        for name in names:
            au = a_map.get(name, 0.0) / 1000.0
            bu = b_map.get(name, 0.0) / 1000.0
            ratio = f"{au / bu:.2f}x" if bu > 0 else ("inf" if au > 0 else "n/a")
            print(f"{name:18s} {au:22.2f} {bu:22.2f} {ratio:>10s}")

    if args.json_out:
        out = pathlib.Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
