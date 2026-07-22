#!/usr/bin/env python
"""Microbenchmark: B12X W6A6 FP6 fused MoE vs a BF16 grouped-MoE baseline.

Times ``b12x_moe_fp6`` across token counts that exercise both the static
(decode / small batch) and dynamic (prefill / large batch) backends, and
compares against a straightforward BF16 grouped MoE (the same gated-SiLU math in
full precision) to document the FP6-vs-BF16 latency gap.

This is a *latency* microbench (correctness is covered by tests/test_fp6_gpu.py).
Weights are random and quantized once via the offline W6A6 path.

Example:
    python scripts/bench_fp6_moe.py --experts 256 --k 2048 --n 512 --topk 8 \
        --tokens 1,8,128,512,4096
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from b12x.quantization import quantize_moe_weights_to_fp6


def _backend(num_tokens: int, num_topk: int) -> str:
    from b12x.integration.tp_moe import select_tp_moe_backend

    return select_tp_moe_backend(
        num_tokens=num_tokens, num_topk=num_topk, quant_mode="w6a6"
    )


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _bf16_grouped_moe(x, w1_bf, w2_bf, topk_ids, topk_weights, n):
    """Reference BF16 gated-SiLU MoE grouped by expert ([up; gate] row order)."""
    m, k = x.shape
    out = torch.zeros(m, k, device=x.device, dtype=torch.float32)
    xf = x.float()
    for e in range(w1_bf.shape[0]):
        sel = topk_ids == e
        if not bool(sel.any()):
            continue
        rows, cols = sel.nonzero(as_tuple=True)
        xe = xf[rows]
        h = xe @ w1_bf[e].float().T
        up, gate = h[:, :n], h[:, n:]
        inter = F.silu(gate) * up
        down = inter @ w2_bf[e].float().T
        out.index_add_(0, rows, down * topk_weights[rows, cols].float().unsqueeze(1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--k", type=int, default=2048, help="hidden size")
    p.add_argument("--n", type=int, default=512, help="intermediate size")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--tokens", default="1,8,128,512,4096")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-bf16", action="store_true", help="skip the BF16 baseline")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (the FP6 kernel runs on SM120)")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    e, k, n, topk = args.experts, args.k, args.n, args.topk

    print(f"quantizing random weights: E={e} K={k} N={n} topk={topk}")
    w1_bf = torch.randn(e, 2 * n, k, device=device, dtype=torch.bfloat16) * 0.15
    w2_bf = torch.randn(e, k, n, device=device, dtype=torch.bfloat16) * 0.15
    w = quantize_moe_weights_to_fp6(w1_bf, w2_bf, source_format="mxfp6_default")

    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace,
        b12x_moe_fp6,
        clear_tp_moe_caches,
    )

    print(f"\n{'tokens':>7} {'backend':>8} {'fp6_ms':>9} {'fp6_tok/s':>11} "
          f"{'bf16_ms':>9} {'speedup':>8}")
    print("-" * 60)
    for m in [int(t) for t in args.tokens.split(",")]:
        tk = min(topk, e)
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
        topk_ids = torch.randint(0, e, (m, tk), device=device, dtype=torch.int32)
        topk_weights = torch.softmax(
            torch.randn(m, tk, device=device), dim=-1
        ).to(torch.float32)

        clear_tp_moe_caches()
        workspace = allocate_tp_moe_workspace(
            x, w.a1_gscale, w.w1_fp6, w.a2_gscale, w.w2_fp6, topk_ids,
            quant_mode="w6a6", input_scales_static=True,
        )

        def _fp6():
            return b12x_moe_fp6(
                x, w.a1_gscale, w.w1_fp6, w.w1_blockscale, w.w1_alphas,
                w.a2_gscale, w.w2_fp6, w.w2_blockscale, w.w2_alphas,
                topk_weights, topk_ids,
                workspace=workspace, input_scales_static=True,
                source_format=w.source_format,
            )

        fp6_ms = _time_ms(_fp6, args.warmup, args.iters)
        tok_s = m / (fp6_ms * 1e-3)
        bf16_ms = float("nan")
        speedup = float("nan")
        if not args.no_bf16:
            bf16_ms = _time_ms(
                lambda: _bf16_grouped_moe(x, w1_bf, w2_bf, topk_ids, topk_weights, n),
                args.warmup, args.iters,
            )
            speedup = bf16_ms / fp6_ms
        print(f"{m:>7} {_backend(m, tk):>8} {fp6_ms:>9.3f} {tok_s:>11.0f} "
              f"{bf16_ms:>9.3f} {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
