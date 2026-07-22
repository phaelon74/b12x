#!/usr/bin/env python3
"""Small-batch W6A6 MoE benchmark using synthetic MX-FP6 expert weights."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_dense_gemm import capture_graph_replay, fmt_us, make_l2_flush_fn, resolve_l2_flush_bytes
from b12x.integration.tp_moe import allocate_tp_moe_workspace, b12x_moe_fp6, clear_tp_moe_caches

from tests.test_fp6_gpu import _synthetic_mxfp6_moe_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--flush-l2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    clear_tp_moe_caches()
    device = torch.device("cuda")
    torch.manual_seed(0)
    m, k, n, e, topk = args.m, args.k, args.n, args.experts, args.topk
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
    topk_ids = torch.randint(0, e, (m, topk), device=device, dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device=device), dim=-1)
    w1, w1_bs, w2, w2_bs = _synthetic_mxfp6_moe_weights(
        experts=e, k=k, n=n, device=device
    )
    a1 = torch.ones(1, device=device)
    a2 = torch.ones(1, device=device)
    w1a = torch.ones(e, device=device)
    w2a = torch.ones(e, device=device)
    workspace = allocate_tp_moe_workspace(
        x, a1, w1, a2, w2, topk_ids, quant_mode="w6a6", input_scales_static=True
    )
    out = torch.empty(m, k, device=device, dtype=torch.bfloat16)

    def launch() -> None:
        b12x_moe_fp6(
            x,
            a1,
            w1,
            w1_bs,
            w1a,
            a2,
            w2,
            w2_bs,
            w2a,
            topk_weights,
            topk_ids,
            workspace=workspace,
            output=out,
            input_scales_static=True,
            source_format="mxfp6_default",
        )

    replay = capture_graph_replay(launch)
    l2_flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
    for _ in range(args.warmup):
        if l2_flush is not None:
            l2_flush()
        replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    for i in range(args.iters):
        if l2_flush is not None:
            l2_flush()
        starts[i].record()
        replay()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    med = statistics.median(times)
    print(
        f"W6A6 MoE synthetic m={m} k={k} n={n} E={e} topk={topk}: "
        f"{fmt_us(times)}  (median {med:.3f} ms)"
    )


if __name__ == "__main__":
    main()
