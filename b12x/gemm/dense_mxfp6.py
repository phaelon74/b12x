"""Inline MX-FP6 warp MMA helpers for dense block-scaled GEMM."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32

from b12x.cute.fp6 import (
    mxfp6_mma_m16n8k32_f32_e2m3_e2m3,
    mxfp6_mma_m16n8k32_f32_e2m3_e3m2,
    mxfp6_mma_m16n8k32_f32_e3m2_e2m3,
    mxfp6_mma_m16n8k32_f32_e3m2_e3m2,
)


@cute.jit
def _mxfp6_scale_u32_from_sf_frag(sf_frag: cute.Tensor) -> Uint32:
    """Pack the first UE8M0 scale lane from an SF fragment into a single u32 operand."""
    return Uint32(sf_frag[0])


@cute.jit
def emit_mxfp6_dense_mma_k_block(
    accumulators: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCrSFA: cute.Tensor,
    tCrSFB: cute.Tensor,
    mt: int,
    nt: int,
    k_block_idx: int,
    a_dtype,
    b_dtype,
) -> None:
    """Emit one MX-FP6 ``m16n8k32`` block-scaled MMA for a single (M,N) output tile."""
    acc = accumulators[None, mt, nt]
    a_frag = tCrA[None, mt, k_block_idx]
    b_frag = tCrB[None, nt, k_block_idx]
    sfa_frag = tCrSFA[None, mt, k_block_idx]
    sfb_frag = tCrSFB[None, nt, k_block_idx]
    sfa = _mxfp6_scale_u32_from_sf_frag(sfa_frag)
    sfb = _mxfp6_scale_u32_from_sf_frag(sfb_frag)

    if cutlass.const_expr(
        a_dtype == cutlass.Float6E3M2FN and b_dtype == cutlass.Float6E3M2FN
    ):
        d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e3m2_e3m2(
            acc[0], acc[1], acc[2], acc[3],
            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
            b_frag[0], b_frag[1],
            sfa, sfb,
        )
    elif cutlass.const_expr(
        a_dtype == cutlass.Float6E2M3FN and b_dtype == cutlass.Float6E2M3FN
    ):
        d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e2m3_e2m3(
            acc[0], acc[1], acc[2], acc[3],
            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
            b_frag[0], b_frag[1],
            sfa, sfb,
        )
    elif cutlass.const_expr(
        a_dtype == cutlass.Float6E2M3FN and b_dtype == cutlass.Float6E3M2FN
    ):
        d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e2m3_e3m2(
            acc[0], acc[1], acc[2], acc[3],
            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
            b_frag[0], b_frag[1],
            sfa, sfb,
        )
    else:
        d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e3m2_e2m3(
            acc[0], acc[1], acc[2], acc[3],
            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
            b_frag[0], b_frag[1],
            sfa, sfb,
        )
    acc[0] = d0
    acc[1] = d1
    acc[2] = d2
    acc[3] = d3
