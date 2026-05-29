"""Inline MX-FP6 attention MMA helpers (mirrors the MXFP8 attention path)."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr

from b12x.cute.fp4 import (
    bfloat2_mul,
    broadcast_f32_to_bfloat2,
    frag_layout_swizzle_16b_to_8b,
    frag_layout_swizzle_16b_to_8b_trans,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x4_left_half_b16,
    ldmatrix_m8n8x4_right_half_b16,
    ldmatrix_m8n8x4_trans_left_half_b16,
    ldmatrix_m8n8x4_trans_right_half_b16,
)
from b12x.cute.fp6 import (
    cvt_bf16x2_to_e2m3x2,
    cvt_bf16x2_to_e3m2x2,
    cvt_bf16x2x2_to_e2m3x4,
    cvt_bf16x2x2_to_e3m2x4,
    mxfp6_mma_m16n8k32_f32_e2m3_e2m3,
    mxfp6_mma_m16n8k32_f32_e3m2_e3m2,
)


@cute.jit
def _permuted_offset_128b(row_idx, vec_idx, stride_128b):
    return row_idx * stride_128b + (vec_idx ^ (row_idx % 8))


@cute.jit
def _smem_addr_from_b128_offset(base_addr: Int32, offset_128b):
    return base_addr + Int32(offset_128b * 16)


@cute.jit
def _advance_offset_by_row_128b(offset_128b, step_size, row_stride_128b):
    return offset_128b + step_size * row_stride_128b


@cute.jit
def _advance_offset_by_column_128b_2(offset_128b, step_idx):
    xor_term = Int32(0x2) + (Int32(0x4) if const_expr(step_idx % 2 == 1) else Int32(0))
    extra = Int32(8) if const_expr(step_idx % 4 == 3) else Int32(0)
    return (offset_128b ^ xor_term) + extra


@cute.jit
def _pack_q_mxfp6_reg(
    s_q_bytes: cute.Tensor,
    row: Int32,
    col_pair_base: Int32,
) -> Uint32:
    """Pack four byte-container lanes from quantized Q smem (same layout as MXFP8)."""
    b0 = Uint32(s_q_bytes[row, col_pair_base + Int32(0)])
    b1 = Uint32(s_q_bytes[row, col_pair_base + Int32(1)])
    b2 = Uint32(s_q_bytes[row, col_pair_base + Int32(8)])
    b3 = Uint32(s_q_bytes[row, col_pair_base + Int32(9)])
    return b0 | (b1 << Int32(8)) | (b2 << Int32(16)) | (b3 << Int32(24))


@cute.jit
def _literal_qk_mma_into_sfrag_mxfp6_raw_paged(
    s_frag: cute.Tensor,
    q_base_addr: Int32,
    k_base_addr: Int32,
    lane,
    warp_q_idx,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_qk,
    upcast_stride_q,
    upcast_stride_k,
    kv_dtype,
):
    """Paged-attention QK MMA with MX-FP6 K (Q converted from BF16 smem)."""
    unit_scale = Uint32(0x7F7F7F7F)
    q_offset = _permuted_offset_128b(
        warp_q_idx * num_mma_q * 16 + lane % 16,
        lane // 16,
        upcast_stride_q,
    )
    k_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * 16 + 8 * (lane // 16) + lane % 8,
        (lane % 16) // 8,
        upcast_stride_k,
    )
    for mma_pair in cutlass.range_constexpr(num_mma_d_qk // 2):
        a_regs_k0 = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        q_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )

        q_offset_cur = q_offset
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                _smem_addr_from_b128_offset(q_base_addr, q_offset_cur)
            )
            a_regs_k0[mma_q, 0] = a0
            a_regs_k0[mma_q, 1] = a1
            a_regs_k0[mma_q, 2] = a2
            a_regs_k0[mma_q, 3] = a3
            q_offset_cur = _advance_offset_by_row_128b(q_offset_cur, 16, upcast_stride_q)

        mma_d0 = mma_pair * 2
        q_offset_mid = _advance_offset_by_column_128b_2(q_offset_cur, mma_d0) - Int32(
            num_mma_q * 16 * upcast_stride_q
        )
        q_offset_cur = q_offset_mid
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                _smem_addr_from_b128_offset(q_base_addr, q_offset_cur)
            )
            if cutlass.const_expr(kv_dtype == cutlass.Float6E3M2FN):
                q_regs[mma_q, 0] = cvt_bf16x2x2_to_e3m2x4(a_regs_k0[mma_q, 0], a0)
                q_regs[mma_q, 1] = cvt_bf16x2x2_to_e3m2x4(a_regs_k0[mma_q, 1], a1)
                q_regs[mma_q, 2] = cvt_bf16x2x2_to_e3m2x4(a_regs_k0[mma_q, 2], a2)
                q_regs[mma_q, 3] = cvt_bf16x2x2_to_e3m2x4(a_regs_k0[mma_q, 3], a3)
            else:
                q_regs[mma_q, 0] = cvt_bf16x2x2_to_e2m3x4(a_regs_k0[mma_q, 0], a0)
                q_regs[mma_q, 1] = cvt_bf16x2x2_to_e2m3x4(a_regs_k0[mma_q, 1], a1)
                q_regs[mma_q, 2] = cvt_bf16x2x2_to_e2m3x4(a_regs_k0[mma_q, 2], a2)
                q_regs[mma_q, 3] = cvt_bf16x2x2_to_e2m3x4(a_regs_k0[mma_q, 3], a3)
            q_offset_cur = _advance_offset_by_row_128b(q_offset_cur, 16, upcast_stride_q)
        q_offset = _advance_offset_by_column_128b_2(q_offset_cur, mma_d0 + 1) - Int32(
            num_mma_q * 16 * upcast_stride_q
        )

        k_offset_cur = k_offset
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            b0_k0, b0_k1, b1_k0, b1_k1 = ldmatrix_m8n8x4_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
            b1_k0 = frag_layout_swizzle_16b_to_8b(b1_k0)
            b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1)
            b1_k1 = frag_layout_swizzle_16b_to_8b(b1_k1)
            k_offset_cur = _advance_offset_by_row_128b(k_offset_cur, 16, upcast_stride_k)

            for mma_q in cutlass.range_constexpr(num_mma_q):
                qa0 = q_regs[mma_q, 0]
                qa1 = q_regs[mma_q, 1]
                qa2 = q_regs[mma_q, 2]
                qa3 = q_regs[mma_q, 3]
                if cutlass.const_expr(kv_dtype == cutlass.Float6E3M2FN):
                    d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e3m2_e3m2(
                        s_frag[mma_q, mma_kv, 0],
                        s_frag[mma_q, mma_kv, 1],
                        s_frag[mma_q, mma_kv, 2],
                        s_frag[mma_q, mma_kv, 3],
                        qa0,
                        qa1,
                        qa2,
                        qa3,
                        b0_k0,
                        b0_k1,
                        unit_scale,
                        unit_scale,
                    )
                    d4, d5, d6, d7 = mxfp6_mma_m16n8k32_f32_e3m2_e3m2(
                        s_frag[mma_q, mma_kv, 4],
                        s_frag[mma_q, mma_kv, 5],
                        s_frag[mma_q, mma_kv, 6],
                        s_frag[mma_q, mma_kv, 7],
                        qa0,
                        qa1,
                        qa2,
                        qa3,
                        b1_k0,
                        b1_k1,
                        unit_scale,
                        unit_scale,
                    )
                else:
                    d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e2m3_e2m3(
                        s_frag[mma_q, mma_kv, 0],
                        s_frag[mma_q, mma_kv, 1],
                        s_frag[mma_q, mma_kv, 2],
                        s_frag[mma_q, mma_kv, 3],
                        qa0,
                        qa1,
                        qa2,
                        qa3,
                        b0_k0,
                        b0_k1,
                        unit_scale,
                        unit_scale,
                    )
                    d4, d5, d6, d7 = mxfp6_mma_m16n8k32_f32_e2m3_e2m3(
                        s_frag[mma_q, mma_kv, 4],
                        s_frag[mma_q, mma_kv, 5],
                        s_frag[mma_q, mma_kv, 6],
                        s_frag[mma_q, mma_kv, 7],
                        qa0,
                        qa1,
                        qa2,
                        qa3,
                        b1_k0,
                        b1_k1,
                        unit_scale,
                        unit_scale,
                    )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7

        k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_pair) - Int32(
            num_mma_kv * 16 * upcast_stride_k
        )


@cute.jit
def _literal_pv_mma_into_ofrag_mxfp6_raw_paged(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    lane,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_vo,
    upcast_stride_v,
    v_scale,
    kv_dtype,
):
    """Paged-attention PV MMA with MX-FP6 V (P converted from BF16, V from smem)."""
    unit_scale = Uint32(0x7F7F7F7F)
    mask16 = Uint32(0xFFFF)
    shift16 = Uint32(16)
    v_scale_bf2 = broadcast_f32_to_bfloat2(v_scale)
    v_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * 16 + lane % 16,
        lane // 16,
        upcast_stride_v,
    )
    for mma_pair in cutlass.range_constexpr(num_mma_kv // 2):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        mma_kv0 = mma_pair * 2
        mma_kv1 = mma_kv0 + 1
        for mma_q in cutlass.range_constexpr(num_mma_q):
            if cutlass.const_expr(kv_dtype == cutlass.Float6E3M2FN):
                a_regs[mma_q, 0] = (
                    cvt_bf16x2_to_e3m2x2(bfloat2_mul(p_frag[mma_q, mma_kv0, 0], v_scale_bf2))
                    & mask16
                ) | (
                    (
                        cvt_bf16x2_to_e3m2x2(
                            bfloat2_mul(p_frag[mma_q, mma_kv1, 0], v_scale_bf2)
                        )
                        & mask16
                    )
                    << shift16
                )
                a_regs[mma_q, 1] = (
                    cvt_bf16x2_to_e3m2x2(bfloat2_mul(p_frag[mma_q, mma_kv0, 1], v_scale_bf2))
                    & mask16
                ) | (
                    (
                        cvt_bf16x2_to_e3m2x2(
                            bfloat2_mul(p_frag[mma_q, mma_kv1, 1], v_scale_bf2)
                        )
                        & mask16
                    )
                    << shift16
                )
                a_regs[mma_q, 2] = (
                    cvt_bf16x2_to_e3m2x2(bfloat2_mul(p_frag[mma_q, mma_kv0, 2], v_scale_bf2))
                    & mask16
                ) | (
                    (
                        cvt_bf16x2_to_e3m2x2(
                            bfloat2_mul(p_frag[mma_q, mma_kv1, 2], v_scale_bf2)
                        )
                        & mask16
                    )
                    << shift16
                )
                a_regs[mma_q, 3] = (
                    cvt_bf16x2_to_e3m2x2(bfloat2_mul(p_frag[mma_q, mma_kv0, 3], v_scale_bf2))
                    & mask16
                ) | (
                    (
                        cvt_bf16x2_to_e3m2x2(
                            bfloat2_mul(p_frag[mma_q, mma_kv1, 3], v_scale_bf2)
                        )
                        & mask16
                    )
                    << shift16
                )
            else:
                a_regs[mma_q, 0] = (
                    cvt_bf16x2_to_e2m3x2(bfloat2_mul(p_frag[mma_q, mma_kv0, 0], v_scale_bf2))
                    & mask16
                ) | (
                    (
                        cvt_bf16x2_to_e2m3x2(
                            bfloat2_mul(p_frag[mma_q, mma_kv1, 0], v_scale_bf2)
                        )
                        & mask16
                    )
                    << shift16
                )
                a_regs[mma_q, 1] = (
                    cvt_bf16x2_to_e2m3x2(bfloat2_mul(p_frag[mma_q, mma_kv0, 1], v_scale_bf2))
                    & mask16
                ) | (
                    (
                        cvt_bf16x2_to_e2m3x2(
                            bfloat2_mul(p_frag[mma_q, mma_kv1, 1], v_scale_bf2)
                        )
                        & mask16
                    )
                    << shift16
                )
                a_regs[mma_q, 2] = (
                    cvt_bf16x2_to_e2m3x2(bfloat2_mul(p_frag[mma_q, mma_kv0, 2], v_scale_bf2))
                    & mask16
                ) | (
                    (
                        cvt_bf16x2_to_e2m3x2(
                            bfloat2_mul(p_frag[mma_q, mma_kv1, 2], v_scale_bf2)
                        )
                        & mask16
                    )
                    << shift16
                )
                a_regs[mma_q, 3] = (
                    cvt_bf16x2_to_e2m3x2(bfloat2_mul(p_frag[mma_q, mma_kv0, 3], v_scale_bf2))
                    & mask16
                ) | (
                    (
                        cvt_bf16x2_to_e2m3x2(
                            bfloat2_mul(p_frag[mma_q, mma_kv1, 3], v_scale_bf2)
                        )
                        & mask16
                    )
                    << shift16
                )

        v_offset_k0 = v_offset
        v_offset_k1 = _advance_offset_by_row_128b(v_offset, 16, upcast_stride_v)
        for mma_d in cutlass.range_constexpr(num_mma_d_vo):
            if const_expr(mma_d % 2 == 0):
                b0_k0, b1_k0 = ldmatrix_m8n8x4_trans_left_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_k0)
                )
                b0_k1, b1_k1 = ldmatrix_m8n8x4_trans_left_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_k1)
                )
            else:
                b0_k0, b1_k0 = ldmatrix_m8n8x4_trans_right_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_k0)
                )
                b0_k1, b1_k1 = ldmatrix_m8n8x4_trans_right_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_k1)
                )
            b0_k0 = frag_layout_swizzle_16b_to_8b_trans(b0_k0)
            b1_k0 = frag_layout_swizzle_16b_to_8b_trans(b1_k0)
            b0_k1 = frag_layout_swizzle_16b_to_8b_trans(b0_k1)
            b1_k1 = frag_layout_swizzle_16b_to_8b_trans(b1_k1)

            for mma_q in cutlass.range_constexpr(num_mma_q):
                if cutlass.const_expr(kv_dtype == cutlass.Float6E3M2FN):
                    d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e3m2_e3m2(
                        o_frag[mma_q, mma_d, 0],
                        o_frag[mma_q, mma_d, 1],
                        o_frag[mma_q, mma_d, 2],
                        o_frag[mma_q, mma_d, 3],
                        a_regs[mma_q, 0],
                        a_regs[mma_q, 1],
                        a_regs[mma_q, 2],
                        a_regs[mma_q, 3],
                        b0_k0,
                        b0_k1,
                        unit_scale,
                        unit_scale,
                    )
                    d4, d5, d6, d7 = mxfp6_mma_m16n8k32_f32_e3m2_e3m2(
                        o_frag[mma_q, mma_d, 4],
                        o_frag[mma_q, mma_d, 5],
                        o_frag[mma_q, mma_d, 6],
                        o_frag[mma_q, mma_d, 7],
                        a_regs[mma_q, 0],
                        a_regs[mma_q, 1],
                        a_regs[mma_q, 2],
                        a_regs[mma_q, 3],
                        b1_k0,
                        b1_k1,
                        unit_scale,
                        unit_scale,
                    )
                else:
                    d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e2m3_e2m3(
                        o_frag[mma_q, mma_d, 0],
                        o_frag[mma_q, mma_d, 1],
                        o_frag[mma_q, mma_d, 2],
                        o_frag[mma_q, mma_d, 3],
                        a_regs[mma_q, 0],
                        a_regs[mma_q, 1],
                        a_regs[mma_q, 2],
                        a_regs[mma_q, 3],
                        b0_k0,
                        b0_k1,
                        unit_scale,
                        unit_scale,
                    )
                    d4, d5, d6, d7 = mxfp6_mma_m16n8k32_f32_e2m3_e2m3(
                        o_frag[mma_q, mma_d, 4],
                        o_frag[mma_q, mma_d, 5],
                        o_frag[mma_q, mma_d, 6],
                        o_frag[mma_q, mma_d, 7],
                        a_regs[mma_q, 0],
                        a_regs[mma_q, 1],
                        a_regs[mma_q, 2],
                        a_regs[mma_q, 3],
                        b1_k0,
                        b1_k1,
                        unit_scale,
                        unit_scale,
                    )
                o_frag[mma_q, mma_d, 0] = d0
                o_frag[mma_q, mma_d, 1] = d1
                o_frag[mma_q, mma_d, 2] = d2
                o_frag[mma_q, mma_d, 3] = d3
                o_frag[mma_q, mma_d, 4] = d4
                o_frag[mma_q, mma_d, 5] = d5
                o_frag[mma_q, mma_d, 6] = d6
                o_frag[mma_q, mma_d, 7] = d7
            if const_expr(mma_d % 2 == 1):
                v_offset_k0 = _advance_offset_by_column_128b_2(v_offset_k0, mma_d // 2)
                v_offset_k1 = _advance_offset_by_column_128b_2(v_offset_k1, mma_d // 2)

        v_offset = _advance_offset_by_row_128b(v_offset, 32, upcast_stride_v)


@cute.jit
def _literal_qk_mma_into_sfrag_mxfp6_raw_mla(
    s_frag: cute.Tensor,
    q_base_addr: Int32,
    k_base_addr: Int32,
    lane: Int32,
    row_base: Int32,
    num_mma_q: Int32,
    num_mma_kv: Int32,
    num_mma_d_qk: Int32,
    upcast_stride_q: Int32,
    upcast_stride_k: Int32,
    kv_dtype,
):
    """MLA sparse QK MMA with MX-FP6 K."""
    unit_scale = Uint32(0x7F7F7F7F)
    mask16 = Uint32(0xFFFF)
    shift16 = Uint32(16)
    q_offset = _permuted_offset_128b(
        lane % Int32(16),
        lane // Int32(16),
        upcast_stride_q,
    )
    k_offset = _permuted_offset_128b(
        row_base + Int32(8) * (lane // Int32(16)) + lane % Int32(8),
        (lane % Int32(16)) // Int32(8),
        upcast_stride_k,
    )
    for mma_pair in cutlass.range_constexpr(num_mma_d_qk // 2):
        a_regs_k0 = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        q_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )

        q_offset_cur = q_offset
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                _smem_addr_from_b128_offset(q_base_addr, q_offset_cur)
            )
            a_regs_k0[mma_q, 0] = a0
            a_regs_k0[mma_q, 1] = a1
            a_regs_k0[mma_q, 2] = a2
            a_regs_k0[mma_q, 3] = a3
            q_offset_cur = _advance_offset_by_row_128b(q_offset_cur, 16, upcast_stride_q)

        mma_d0 = mma_pair * 2
        q_offset_mid = _advance_offset_by_column_128b_2(q_offset_cur, mma_d0) - Int32(
            num_mma_q * 16 * upcast_stride_q
        )
        q_offset_cur = q_offset_mid
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                _smem_addr_from_b128_offset(q_base_addr, q_offset_cur)
            )
            if cutlass.const_expr(kv_dtype == cutlass.Float6E3M2FN):
                q_regs[mma_q, 0] = (
                    cvt_bf16x2_to_e3m2x2(a_regs_k0[mma_q, 0]) & mask16
                ) | ((cvt_bf16x2_to_e3m2x2(a_regs_k0[mma_q, 2]) & mask16) << shift16)
                q_regs[mma_q, 1] = (
                    cvt_bf16x2_to_e3m2x2(a_regs_k0[mma_q, 1]) & mask16
                ) | ((cvt_bf16x2_to_e3m2x2(a_regs_k0[mma_q, 3]) & mask16) << shift16)
                q_regs[mma_q, 2] = (cvt_bf16x2_to_e3m2x2(a0) & mask16) | (
                    (cvt_bf16x2_to_e3m2x2(a2) & mask16) << shift16
                )
                q_regs[mma_q, 3] = (cvt_bf16x2_to_e3m2x2(a1) & mask16) | (
                    (cvt_bf16x2_to_e3m2x2(a3) & mask16) << shift16
                )
            else:
                q_regs[mma_q, 0] = (
                    cvt_bf16x2_to_e2m3x2(a_regs_k0[mma_q, 0]) & mask16
                ) | ((cvt_bf16x2_to_e2m3x2(a_regs_k0[mma_q, 2]) & mask16) << shift16)
                q_regs[mma_q, 1] = (
                    cvt_bf16x2_to_e2m3x2(a_regs_k0[mma_q, 1]) & mask16
                ) | ((cvt_bf16x2_to_e2m3x2(a_regs_k0[mma_q, 3]) & mask16) << shift16)
                q_regs[mma_q, 2] = (cvt_bf16x2_to_e2m3x2(a0) & mask16) | (
                    (cvt_bf16x2_to_e2m3x2(a2) & mask16) << shift16
                )
                q_regs[mma_q, 3] = (cvt_bf16x2_to_e2m3x2(a1) & mask16) | (
                    (cvt_bf16x2_to_e2m3x2(a3) & mask16) << shift16
                )
            q_offset_cur = _advance_offset_by_row_128b(q_offset_cur, 16, upcast_stride_q)
        q_offset = _advance_offset_by_column_128b_2(q_offset_cur, mma_d0 + 1) - Int32(
            num_mma_q * 16 * upcast_stride_q
        )

        k_offset_cur = k_offset
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            b0_k0, b1_k0 = ldmatrix_m8n8x4_left_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            b0_k1, b1_k1 = ldmatrix_m8n8x4_right_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
            b1_k0 = frag_layout_swizzle_16b_to_8b(b1_k0)
            b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1)
            b1_k1 = frag_layout_swizzle_16b_to_8b(b1_k1)
            k_offset_cur = _advance_offset_by_row_128b(k_offset_cur, 16, upcast_stride_k)

            for mma_q in cutlass.range_constexpr(num_mma_q):
                if cutlass.const_expr(kv_dtype == cutlass.Float6E3M2FN):
                    d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e3m2_e3m2(
                        s_frag[mma_q, mma_kv, 0],
                        s_frag[mma_q, mma_kv, 1],
                        s_frag[mma_q, mma_kv, 2],
                        s_frag[mma_q, mma_kv, 3],
                        q_regs[mma_q, 0],
                        q_regs[mma_q, 1],
                        q_regs[mma_q, 2],
                        q_regs[mma_q, 3],
                        b0_k0,
                        b0_k1,
                        unit_scale,
                        unit_scale,
                    )
                    d4, d5, d6, d7 = mxfp6_mma_m16n8k32_f32_e3m2_e3m2(
                        s_frag[mma_q, mma_kv, 4],
                        s_frag[mma_q, mma_kv, 5],
                        s_frag[mma_q, mma_kv, 6],
                        s_frag[mma_q, mma_kv, 7],
                        q_regs[mma_q, 0],
                        q_regs[mma_q, 1],
                        q_regs[mma_q, 2],
                        q_regs[mma_q, 3],
                        b1_k0,
                        b1_k1,
                        unit_scale,
                        unit_scale,
                    )
                else:
                    d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e2m3_e2m3(
                        s_frag[mma_q, mma_kv, 0],
                        s_frag[mma_q, mma_kv, 1],
                        s_frag[mma_q, mma_kv, 2],
                        s_frag[mma_q, mma_kv, 3],
                        q_regs[mma_q, 0],
                        q_regs[mma_q, 1],
                        q_regs[mma_q, 2],
                        q_regs[mma_q, 3],
                        b0_k0,
                        b0_k1,
                        unit_scale,
                        unit_scale,
                    )
                    d4, d5, d6, d7 = mxfp6_mma_m16n8k32_f32_e2m3_e2m3(
                        s_frag[mma_q, mma_kv, 4],
                        s_frag[mma_q, mma_kv, 5],
                        s_frag[mma_q, mma_kv, 6],
                        s_frag[mma_q, mma_kv, 7],
                        q_regs[mma_q, 0],
                        q_regs[mma_q, 1],
                        q_regs[mma_q, 2],
                        q_regs[mma_q, 3],
                        b1_k0,
                        b1_k1,
                        unit_scale,
                        unit_scale,
                    )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7

        k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_pair) - Int32(
            num_mma_kv * 16 * upcast_stride_k
        )


@cute.jit
def _compute_mxfp6_tile_partials(
    s_q_bytes: cute.Tensor,
    s_w: cute.Tensor,
    num_heads: Int32,
    k_perm_base_addr: Int32,
    token_base: Int32,
    head_tile_base: Int32,
    lane: Int32,
    s_partial_logits: cute.Tensor,
    partial_row_base: Int32,
    head_tile_slot: Int32,
    kv_dtype,
    index_head_dim: int,
):
    """Indexer tile partials with MX-FP6 K (mirrors ``_compute_mxfp8_tile_partials``)."""
    from b12x.attention._cute import ops as attention_ops

    group_id = lane // Int32(4)
    thread_id_in_group = lane % Int32(4)
    col_pair_base = thread_id_in_group * Int32(2)
    q0_acc = Float32(0.0)
    q1_acc = Float32(0.0)
    q2_acc = Float32(0.0)
    q3_acc = Float32(0.0)
    k_offset = _permuted_offset_128b(
        token_base + Int32(8) * (lane // Int32(16)) + lane % Int32(8),
        (lane % Int32(16)) // Int32(8),
        Int32(index_head_dim // 16),
    )
    for mma_pair in cutlass.range_constexpr(index_head_dim // 32):
        pair_base = Int32(mma_pair * 32) + col_pair_base
        q0 = _pack_q_mxfp6_reg(s_q_bytes, head_tile_base + group_id, pair_base)
        q1 = _pack_q_mxfp6_reg(s_q_bytes, head_tile_base + group_id + Int32(8), pair_base)
        q2 = _pack_q_mxfp6_reg(s_q_bytes, head_tile_base + group_id, pair_base + Int32(16))
        q3 = _pack_q_mxfp6_reg(
            s_q_bytes,
            head_tile_base + group_id + Int32(8),
            pair_base + Int32(16),
        )
        b0_k0, _ = ldmatrix_m8n8x4_left_half_b16(
            _smem_addr_from_b128_offset(k_perm_base_addr, k_offset)
        )
        b0_k1, _ = ldmatrix_m8n8x4_right_half_b16(
            _smem_addr_from_b128_offset(k_perm_base_addr, k_offset)
        )
        b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
        b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1)
        k_offset_cur = _advance_offset_by_row_128b(
            k_offset,
            Int32(16),
            Int32(index_head_dim // 16),
        )
        unit_scale = Uint32(0x7F7F7F7F)
        if cutlass.const_expr(kv_dtype == cutlass.Float6E3M2FN):
            d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e3m2_e3m2(
                q0_acc,
                q1_acc,
                q2_acc,
                q3_acc,
                q0,
                q1,
                q2,
                q3,
                b0_k0,
                b0_k1,
                unit_scale,
                unit_scale,
            )
        else:
            d0, d1, d2, d3 = mxfp6_mma_m16n8k32_f32_e2m3_e2m3(
                q0_acc,
                q1_acc,
                q2_acc,
                q3_acc,
                q0,
                q1,
                q2,
                q3,
                b0_k0,
                b0_k1,
                unit_scale,
                unit_scale,
            )
        q0_acc = d0
        q1_acc = d1
        q2_acc = d2
        q3_acc = d3
        k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_pair) - Int32(
            16 * (index_head_dim // 16)
        )

    head0 = head_tile_base + group_id
    head1 = head0 + Int32(8)
    w0 = Float32(0.0)
    w1 = Float32(0.0)
    if head0 < num_heads:
        w0 = Float32(s_w[head0])
    if head1 < num_heads:
        w1 = Float32(s_w[head1])
    col0 = col_pair_base
    col1 = col_pair_base + Int32(1)
    partial0 = Float32(attention_ops.fmax(q0_acc, Float32(0.0)) * w0)
    partial0 = Float32(partial0 + attention_ops.fmax(q2_acc, Float32(0.0)) * w1)
    partial1 = Float32(attention_ops.fmax(q1_acc, Float32(0.0)) * w0)
    partial1 = Float32(partial1 + attention_ops.fmax(q3_acc, Float32(0.0)) * w1)
    s_partial_logits[partial_row_base, col0, head_tile_slot] = partial0
    s_partial_logits[partial_row_base, col1, head_tile_slot] = partial1
