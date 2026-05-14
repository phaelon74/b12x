"""FC2-only W4A16 direct-intermediate CuTe DSL kernel for SM120/SM121."""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, Uint32, dsl_user_op

from b12x.cute.fp4 import (
    cvt_e4m3_to_f32_via_f16,
    f16x2_to_f32x2,
    fp4_decode_4bytes,
    ld_global_nc_u32,
)
from b12x.cute.utils import current_cuda_stream, make_ptr
from b12x.runtime_control import raise_if_kernel_resolution_frozen


_SF_VEC_SIZE = 16
_DEFAULT_TILE_N = 128
_DEFAULT_TILE_K = 128
_DEFAULT_TILE_M = 32
_DEFAULT_MMA_WARPS = 4


@dsl_user_op
def _ld_global_nc_u8(base_ptr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u8 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _align_up(value: int, align: int) -> int:
    return ((int(value) + int(align) - 1) // int(align)) * int(align)


@dataclass(frozen=True)
class W4A16DirectFC2CompileResult:
    compiled: object
    output_tile_n: int
    intermediate_tile_k: int


class W4A16DirectFC2TokenTileMmaKernel:
    """Direct FC2 token/output-tile kernel."""

    def __init__(
        self,
        *,
        num_topk: int,
        intermediate_n: int,
        tile_n: int = _DEFAULT_TILE_N,
        tile_k: int = _DEFAULT_TILE_K,
        num_mma_warps: int = _DEFAULT_MMA_WARPS,
        unit_scale_contract: bool = True,
    ):
        if num_topk <= 0 or num_topk > 8:
            raise ValueError("num_topk must be in [1, 8]")
        if intermediate_n <= 0 or intermediate_n % 16 != 0:
            raise ValueError("intermediate_n must be positive and divisible by 16")
        if tile_n % 16 != 0 or tile_k % 16 != 0:
            raise ValueError("tile_n and tile_k must be multiples of 16")
        if num_mma_warps <= 0 or num_mma_warps > 8:
            raise ValueError("num_mma_warps must be in [1, 8]")
        self.num_topk = int(num_topk)
        self.intermediate_n = int(intermediate_n)
        self.sf_vec_size = _SF_VEC_SIZE
        self.tile_shape_mnk = (_DEFAULT_TILE_M, int(tile_n), int(tile_k))
        self.epi_tile = (_DEFAULT_TILE_M, int(tile_n))
        self.cluster_shape_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)
        self.acc_dtype = cutlass.Float32
        self.a_dtype = cutlass.BFloat16
        self.b_dtype = cutlass.BFloat16
        self.c_dtype = cutlass.BFloat16
        self.num_mma_warps = int(num_mma_warps)
        self.num_threads_per_warp = 32
        self.threads_per_cta = self.num_mma_warps * self.num_threads_per_warp
        self.buffer_align_bytes = 1024
        self.mma_register_requirement = 232
        self.unit_scale_contract = bool(unit_scale_contract)

        self.mma_inst_mnk = (16, 8, 16)
        self.tiled_mma = None
        self.cta_layout_mnk = None
        self.num_k_blocks = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None

    def _make_a_smem_layout(self, ab_stage: int):
        a_is_k_major = self.a_layout.is_k_major_a()
        a_major_mode_size = self.tile_shape_mnk[1 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                self.a_layout,
                self.a_dtype,
                a_major_mode_size,
            ),
            self.a_dtype,
        )
        return cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append((self.tile_shape_mnk[0], self.tile_shape_mnk[2]), ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

    def _make_b_smem_layout(self, ab_stage: int):
        b_smem_shape = cute.slice_(self.tile_shape_mnk, (0, None, None))
        b_is_k_major = self.b_layout.is_k_major_b()
        b_major_mode_size = self.tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                self.b_layout,
                self.b_dtype,
                b_major_mode_size,
            ),
            self.b_dtype,
        )
        return cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

    def _setup_attributes(self):
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            self.a_dtype,
            self.acc_dtype,
            self.mma_inst_mnk,
        )
        atom_layout_m = max(1, self.tile_shape_mnk[0] // self.mma_inst_mnk[0])
        atom_layout = cute.make_layout((atom_layout_m, 2, 1))
        permutation_mnk = (
            atom_layout_m * self.mma_inst_mnk[0],
            2 * self.mma_inst_mnk[1] * 2,
            self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.num_k_blocks = self.tile_shape_mnk[2] // self.mma_inst_mnk[2]
        self.a_smem_layout_staged = self._make_a_smem_layout(1)
        self.b_smem_layout_staged = self._make_b_smem_layout(1)
        self.epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            cutlass.BFloat16,
            self.c_layout,
            self.epi_tile,
            1,
        )

    @cute.jit
    def _load_u8_as_u32(self, base_addr: Int64, byte_offset: Int64) -> Uint32:
        return _ld_global_nc_u8(base_addr + byte_offset)

    @cute.jit
    def _swizzled_e4m3_offset(
        self,
        row: Int32,
        sf_block: Int32,
        sf_cols: Int32,
    ) -> Int64:
        row_rb = row >> Int32(7)
        mode_a = (row >> Int32(5)) & Int32(3)
        mode_32 = row & Int32(31)
        cb_idx = sf_block >> Int32(2)
        mode_c = sf_block & Int32(3)
        return (
            Int64(row_rb) * Int64(sf_cols * Int32(128))
            + Int64(cb_idx) * Int64(512)
            + Int64(mode_32) * Int64(16)
            + Int64(mode_a) * Int64(4)
            + Int64(mode_c)
        )

    @cute.jit
    def _stage_direct_intermediate_a_tile(
        self,
        intermediate_u32: cute.Tensor,
        sA: cute.Tensor,
        token_idx: Int32,
        route_idx: Int32,
        intermediate_tile_idx: Int32,
        down_cols: Int32,
        num_topk: Int32,
        copy_start: Int32,
        copy_stride: Int32,
    ):
        tile_m = Int32(self.tile_shape_mnk[0])
        tile_k = Int32(self.tile_shape_mnk[2])
        direct_fc2_chunks = ((down_cols // Int32(2)) + Int32(127)) // Int32(128)
        direct_route_stride = direct_fc2_chunks * Int32(128)
        direct_token_stride = direct_route_stride * num_topk

        copy_idx = copy_start
        while copy_idx < tile_m * tile_k:
            local_m = copy_idx // tile_k
            local_k = copy_idx - local_m * tile_k
            global_col = intermediate_tile_idx * tile_k + local_k
            value = cutlass.BFloat16(0.0)
            if local_m == Int32(0):
                if global_col < down_cols:
                    pair_col = global_col >> Int32(1)
                    n_blk = pair_col // Int32(128)
                    h_i = pair_col - n_blk * Int32(128)
                    packed_idx = (
                        token_idx * direct_token_stride
                        + route_idx * direct_route_stride
                        + n_blk * Int32(128)
                        + (h_i % Int32(4)) * Int32(32)
                        + (h_i // Int32(4))
                    )
                    h01 = Uint32(intermediate_u32[packed_idx])
                    v0, v1 = f16x2_to_f32x2(h01)
                    value = (
                        cutlass.BFloat16(v0)
                        if (global_col & Int32(1)) == Int32(0)
                        else cutlass.BFloat16(v1)
                    )
            sA[local_m, local_k, Int32(0)] = value
            copy_idx += copy_stride

    @cute.jit
    def _stage_down_fp4_b_tile(
        self,
        packed_w: cute.Tensor,
        sfb_ptr: cute.Pointer,
        sB: cute.Tensor,
        expert_idx: Int32,
        output_tile_idx: Int32,
        intermediate_tile_idx: Int32,
        weight_rows: Int32,
        weight_cols: Int32,
        sf_cols: Int32,
        copy_start: Int32,
        copy_stride: Int32,
    ):
        w_base = packed_w.iterator.toint()
        sf_base = sfb_ptr.toint()
        packed_cols = weight_cols // Int32(2)
        tile_n = Int32(self.tile_shape_mnk[1])
        tile_k = Int32(self.tile_shape_mnk[2])
        blocks_per_row = tile_k // Int32(self.sf_vec_size)
        total_blocks = tile_n * blocks_per_row
        copy_idx = copy_start
        while copy_idx < total_blocks:
            local_n = copy_idx // blocks_per_row
            local_sf_block = copy_idx - local_n * blocks_per_row
            local_k = local_sf_block * Int32(self.sf_vec_size)
            global_n = output_tile_idx * tile_n + local_n
            global_k = intermediate_tile_idx * tile_k + local_k

            scale = cutlass.Float32(0.0)
            q_word0 = Uint32(0)
            q_word1 = Uint32(0)
            if global_n < weight_rows:
                if global_k < weight_cols:
                    packed_offset = (
                        Int64(expert_idx) * Int64(weight_rows * packed_cols)
                        + Int64(global_n) * Int64(packed_cols)
                        + Int64(global_k // Int32(2))
                    )
                    scale_offset = (
                        Int64(expert_idx)
                        * Int64(
                            ((weight_rows + Int32(127)) // Int32(128))
                            * Int32(128)
                            * sf_cols
                        )
                        + self._swizzled_e4m3_offset(
                            global_n,
                            global_k // Int32(self.sf_vec_size),
                            sf_cols,
                        )
                    )
                    scale_byte = self._load_u8_as_u32(sf_base, scale_offset)
                    scale = cvt_e4m3_to_f32_via_f16(scale_byte)
                    q_word0 = ld_global_nc_u32(w_base + packed_offset)
                    q_word1 = ld_global_nc_u32(w_base + packed_offset + Int64(4))

            d0, d1, d2, d3 = fp4_decode_4bytes(q_word0)
            f0, f1 = f16x2_to_f32x2(d0)
            sB[local_n, local_k, Int32(0)] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(1), Int32(0)] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d1)
            sB[local_n, local_k + Int32(2), Int32(0)] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(3), Int32(0)] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d2)
            sB[local_n, local_k + Int32(4), Int32(0)] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(5), Int32(0)] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d3)
            sB[local_n, local_k + Int32(6), Int32(0)] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(7), Int32(0)] = cutlass.BFloat16(f1 * scale)

            d0, d1, d2, d3 = fp4_decode_4bytes(q_word1)
            f0, f1 = f16x2_to_f32x2(d0)
            sB[local_n, local_k + Int32(8), Int32(0)] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(9), Int32(0)] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d1)
            sB[local_n, local_k + Int32(10), Int32(0)] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(11), Int32(0)] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d2)
            sB[local_n, local_k + Int32(12), Int32(0)] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(13), Int32(0)] = cutlass.BFloat16(f1 * scale)
            f0, f1 = f16x2_to_f32x2(d3)
            sB[local_n, local_k + Int32(14), Int32(0)] = cutlass.BFloat16(f0 * scale)
            sB[local_n, local_k + Int32(15), Int32(0)] = cutlass.BFloat16(f1 * scale)
            copy_idx += copy_stride

    @cute.jit
    def __call__(
        self,
        intermediate_storage: cute.Tensor,
        b_down: cute.Tensor,
        sfb_down_ptr: cute.Pointer,
        down_alpha: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        scatter_output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.a_layout = utils.LayoutEnum.ROW_MAJOR
        self.b_layout = utils.LayoutEnum.from_tensor(b_down)
        self.c_layout = utils.LayoutEnum.ROW_MAJOR
        self._setup_attributes()

        grid = (
            (scatter_output.shape[1] + self.tile_shape_mnk[1] - 1)
            // self.tile_shape_mnk[1],
            scatter_output.shape[0],
            1,
        )
        self.kernel(
            intermediate_storage,
            b_down,
            sfb_down_ptr,
            down_alpha,
            topk_ids,
            topk_weights,
            scatter_output,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        intermediate_storage: cute.Tensor,
        b_down: cute.Tensor,
        sfb_down_ptr: cute.Pointer,
        down_alpha: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        scatter_output: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        epi_smem_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        output_tile_idx = Int32(bidx)
        token_idx = Int32(bidy)
        tid = Int32(tidx)

        cute.arch.setmaxregister_increase(self.mma_register_requirement)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(a_smem_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(epi_smem_staged)],
                self.buffer_align_bytes,
            ]
            # Keep the original shared-memory tail while validating the
            # register-accumulation epilogue. The CuTe stmatrix epilogue layout
            # can touch a slightly padded footprint for this tile shape.
            epi_scratch: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.tile_shape_mnk[1]],
                128,
            ]

        storage = smem.allocate(Storage)
        sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
        sB = storage.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        sC = storage.sC.get_tensor(epi_smem_staged.outer, swizzle=epi_smem_staged.inner)

        intermediate_u32 = cute.recast_tensor(intermediate_storage, cutlass.Uint32)

        tile_n = Int32(self.tile_shape_mnk[1])
        tile_k = Int32(self.tile_shape_mnk[2])
        out_cols = Int32(scatter_output.shape[1])
        down_rows = out_cols
        down_cols = Int32(self.intermediate_n)
        down_sf_cols = (
            ((down_cols // Int32(self.sf_vec_size)) + Int32(3))
            // Int32(4)
            * Int32(4)
        )
        num_topk = Int32(self.num_topk)
        intermediate_tile_cnt = (down_cols + tile_k - Int32(1)) // tile_k

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCsB = thr_mma.partition_B(sB)
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCsC_for_shape = thr_mma.partition_C(sC[None, None, 0])
        acc_shape = tCsC_for_shape.shape[:3]
        down_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        out_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        out_acc.fill(0.0)

        atom_ld_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
            self.a_dtype,
        )
        atom_ld_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
            self.b_dtype,
        )
        smem_copy_A = cute.make_tiled_copy_A(atom_ld_A, tiled_mma)
        smem_copy_B = cute.make_tiled_copy_B(atom_ld_B, tiled_mma)
        thr_ld_A = smem_copy_A.get_slice(tidx)
        thr_ld_B = smem_copy_B.get_slice(tidx)
        csA = thr_ld_A.partition_S(sA)
        csB = thr_ld_B.partition_S(sB)
        csA_stage = csA[None, None, None, 0]
        csB_stage = csB[None, None, None, 0]
        crA = thr_ld_A.retile(tCrA)
        crB = thr_ld_B.retile(tCrB)
        num_k_blocks = cute.size(tCrA, mode=[2])

        copy_atom_r2s = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
        )
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 2),
            cutlass.BFloat16,
        )
        tiled_copy_c_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_c_atom)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
        tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, cutlass.BFloat16)
        mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rD, mode=[1])
        mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rD, mode=[2])
        MmaMPerEpiM = self.epi_tile[0] // mma_tile_m
        MmaNPerEpiN = self.epi_tile[1] // mma_tile_n

        for route in cutlass.range_constexpr(self.num_topk):
            expert_idx = topk_ids[token_idx, Int32(route)].to(Int32)
            route_scale = topk_weights[token_idx, Int32(route)].to(cutlass.Float32)
            if cutlass.const_expr(not self.unit_scale_contract):
                route_scale = route_scale * down_alpha[expert_idx].to(cutlass.Float32)

            down_acc.fill(0.0)
            for intermediate_tile_idx in range(0, intermediate_tile_cnt, 1, unroll=4):
                self._stage_direct_intermediate_a_tile(
                    intermediate_u32,
                    sA,
                    token_idx,
                    Int32(route),
                    intermediate_tile_idx,
                    down_cols,
                    num_topk,
                    tid,
                    Int32(self.threads_per_cta),
                )
                self._stage_down_fp4_b_tile(
                    b_down,
                    sfb_down_ptr,
                    sB,
                    expert_idx,
                    output_tile_idx,
                    intermediate_tile_idx,
                    down_rows,
                    down_cols,
                    down_sf_cols,
                    tid,
                    Int32(self.threads_per_cta),
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_threads()

                cute.copy(smem_copy_A, csA_stage[None, None, 0], crA[None, None, 0])
                cute.copy(smem_copy_B, csB_stage[None, None, 0], crB[None, None, 0])
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    cute.gemm(
                        tiled_mma,
                        down_acc,
                        tCrA[None, None, k_block_idx],
                        tCrB[None, None, k_block_idx],
                        down_acc,
                    )
                    if k_next > 0:
                        cute.copy(
                            smem_copy_A,
                            csA_stage[None, None, k_next],
                            crA[None, None, k_next],
                        )
                        cute.copy(
                            smem_copy_B,
                            csB_stage[None, None, k_next],
                            crB[None, None, k_next],
                        )
                cute.arch.sync_threads()

            for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                    mma_m = mma_m_in_epi
                    mma_n = mma_n_in_epi
                    out_acc_slice = out_acc[(None, mma_m, mma_n)]
                    down_acc_slice = down_acc[(None, mma_m, mma_n)]
                    for elem_idx in cutlass.range_constexpr(cute.size(out_acc_slice)):
                        out_acc_slice[elem_idx] = (
                            out_acc_slice[elem_idx]
                            + route_scale * down_acc_slice[elem_idx]
                        )

        for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
            for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                mma_m = mma_m_in_epi
                mma_n = mma_n_in_epi
                tRS_rD_slice = tRS_rD[(None, mma_m_in_epi, mma_n_in_epi)]
                out_acc_slice = out_acc[(None, mma_m, mma_n)]
                for elem_idx in cutlass.range_constexpr(cute.size(tRS_rD_slice)):
                    tRS_rD_slice[elem_idx] = out_acc_slice[elem_idx]

        acc_vec = tRS_rD.load()
        acc_vec = acc_vec.to(cutlass.BFloat16)
        tRS_rD_out.store(acc_vec)
        cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, 0)])
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        store_idx = tid
        while store_idx < tile_n:
            global_col = output_tile_idx * tile_n + store_idx
            if global_col < out_cols:
                scatter_output[token_idx, global_col] = cutlass.BFloat16(
                    sC[Int32(0), store_idx, Int32(0)]
                )
            store_idx += Int32(self.threads_per_cta)


def compile_w4a16_fc2_direct_mma(
    *,
    m: int,
    k: int,
    n: int,
    weight_e: int,
    num_topk: int,
    topk_ids_dtype: torch.dtype,
    unit_scale_contract: bool = True,
    tile_n: int = _DEFAULT_TILE_N,
    tile_k: int = _DEFAULT_TILE_K,
) -> W4A16DirectFC2CompileResult:
    if k <= 0 or n <= 0 or m <= 0 or weight_e <= 0:
        raise ValueError("m, k, n, and weight_e must be positive")
    if n % 16 != 0:
        raise ValueError("W4A16 FC2 requires n to be divisible by 16")
    if topk_ids_dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids_dtype must be torch.int32 or torch.int64")

    weight_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    alpha_dtype = cutlass.Float32
    out_dtype = cutlass.BFloat16
    topk_ids_cutlass_dtype = cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8

    direct_route_stride_u32 = _align_up(n // 2, 128)
    intermediate_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (m * num_topk * direct_route_stride_u32 * 4,),
        assumed_align=16,
    )
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        (k, n, weight_e),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_e,),
        assumed_align=16,
    )
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        topk_ids_cutlass_dtype,
        (m, num_topk),
        stride_order=(1, 0),
        assumed_align=topk_ids_align,
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (m, num_topk),
        stride_order=(1, 0),
        assumed_align=4,
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        out_dtype,
        (m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )

    kernel = W4A16DirectFC2TokenTileMmaKernel(
        num_topk=num_topk,
        intermediate_n=n,
        tile_n=tile_n,
        tile_k=tile_k,
        unit_scale_contract=unit_scale_contract,
    )
    cache_key = (
        "w4a16_fc2_direct_mma_token_tile",
        m,
        k,
        n,
        weight_e,
        num_topk,
        topk_ids_dtype,
        unit_scale_contract,
        tile_n,
        tile_k,
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=cache_key)
    compiled = cute.compile(
        kernel,
        intermediate_storage_fake,
        b_down_fake,
        sfb_down_fake,
        down_alpha_fake,
        topk_ids_fake,
        topk_weights_fake,
        scatter_fake,
        current_cuda_stream(),
    )
    return W4A16DirectFC2CompileResult(
        compiled=compiled,
        output_tile_n=tile_n,
        intermediate_tile_k=tile_k,
    )


__all__ = [
    "W4A16DirectFC2TokenTileMmaKernel",
    "W4A16DirectFC2CompileResult",
    "compile_w4a16_fc2_direct_mma",
]
