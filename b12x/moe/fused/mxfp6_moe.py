"""Shared MX-FP6 helpers for fused MoE static/dynamic kernels."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint8, Uint64
from cutlass.cutlass_dsl import Int32, Int64
from cutlass.cute.nvgpu.warp.mma import Field as WarpField

from b12x.cute.fp4 import get_ptr_as_int64, st_global_u64
from b12x.cute.fp6 import (
    quantize_block_fp6_e2m3,
    quantize_block_fp6_e2m3_fast,
    quantize_block_fp6_e3m2,
    quantize_block_fp6_e3m2_fast,
)
from b12x.gemm.dense_mxfp6 import emit_mxfp6_dense_mma_k_block


@cute.jit
def moe_mxfp6_quantize_input_block(
    values: cute.Tensor,
    block_max: Float32,
    gs_value: Float32,
    a_dtype,
    fast_math: bool,
) -> tuple[Uint64, Uint64, Uint64, Uint8]:
    """Quantize one 32-element MX-FP6 block for route-pack / FC2 requant."""
    if cutlass.const_expr(a_dtype == cutlass.Float6E3M2FN):
        if fast_math:
            return quantize_block_fp6_e3m2_fast(values, block_max, gs_value)
        return quantize_block_fp6_e3m2(values, block_max, gs_value)
    if fast_math:
        return quantize_block_fp6_e2m3_fast(values, block_max, gs_value)
    return quantize_block_fp6_e2m3(values, block_max, gs_value)


@cute.jit
def moe_mxfp6_store_packed_global(
    storage: cute.Tensor,
    byte_offset: Int32,
    lo: Uint64,
    mid: Uint64,
    hi: Uint64,
) -> None:
    """Store 24 packed MX-FP6 bytes (three u64 lanes) to global uint8 storage."""
    base = get_ptr_as_int64(storage, byte_offset)
    st_global_u64(base, lo)
    st_global_u64(base + Int64(8), mid)
    st_global_u64(base + Int64(16), hi)


@cute.jit
def moe_mxfp6_store_packed_smem_swizzled(
    sA_u8: cute.Tensor,
    row: Int32,
    sf_block: Int32,
    packed_cols: Int32,
    lo: Uint64,
    mid: Uint64,
    hi: Uint64,
) -> None:
    """Scatter 24 packed bytes into swizzled activation smem (matches FP4 layout)."""
    packed_base = sf_block * Int32(24)
    dst_pcol = row & Int32(63)
    xor_bits = ((dst_pcol >> Int32(1)) & Int32(0x3)) << Int32(4)
    row_high = row >> Int32(6)
    for byte_idx in cutlass.range_constexpr(24):
        src_pcol = packed_base + Int32(byte_idx)
        dst_row = ((src_pcol ^ xor_bits) << Int32(1)) + row_high
        dst_flat = dst_row * packed_cols + dst_pcol
        if byte_idx < 8:
            packed_u64 = lo
            shift = byte_idx
        elif byte_idx < 16:
            packed_u64 = mid
            shift = byte_idx - 8
        else:
            packed_u64 = hi
            shift = byte_idx - 16
        byte_val = Uint8((packed_u64 >> Uint64(shift * 8)) & Uint64(0xFF))
        sA_u8[dst_flat] = byte_val


@cute.jit
def moe_emit_fp4_mma(
    mma_atom,
    accumulators: cute.Tensor,
    tCrA: cute.Tensor,
    tCrSFA: cute.Tensor,
    tCrB: cute.Tensor,
    tCrSFB: cute.Tensor,
    mt: int,
    nt: int,
    k_block_idx: int,
) -> None:
    """One FP4 block-scaled warp MMA (SFA/SFB via mma_atom)."""
    mma_atom.set(WarpField.SFA, tCrSFA[None, mt, k_block_idx].iterator)
    mma_atom.set(WarpField.SFB, tCrSFB[None, nt, k_block_idx].iterator)
    cute.gemm(
        mma_atom,
        accumulators[None, mt, nt],
        tCrA[None, mt, k_block_idx],
        tCrB[None, nt, k_block_idx],
        accumulators[None, mt, nt],
    )


@cute.jit
def moe_emit_mma_k_block(
    mma_atom,
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
    """Dispatch one K-block MMA for MoE (inline MX-FP6 or native FP4)."""
    if cutlass.const_expr(
        a_dtype == cutlass.Float6E3M2FN or a_dtype == cutlass.Float6E2M3FN
    ):
        emit_mxfp6_dense_mma_k_block(
            accumulators,
            tCrA,
            tCrB,
            tCrSFA,
            tCrSFB,
            mt,
            nt,
            k_block_idx,
            a_dtype,
            b_dtype,
        )
    else:
        moe_emit_fp4_mma(
            mma_atom,
            accumulators,
            tCrA,
            tCrSFA,
            tCrB,
            tCrSFB,
            mt,
            nt,
            k_block_idx,
        )
