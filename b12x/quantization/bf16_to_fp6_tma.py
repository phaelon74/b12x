"""Standalone BF16→MX-FP6 TMA quantize kernel (32-elt UE8M0 blocks, 4-in-3-byte pack)."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cuda.bindings.driver as cuda
from cutlass.cutlass_dsl import Int32, Uint8
from cutlass.cute.nvgpu import cpasync

from b12x.cute.fp4 import fabs_f32, fmax_f32
from b12x.cute.fp6 import quantize_block_fp6_e3m2_fast
from b12x.cute.utils import sm120_make_smem_layout_sfa
from b12x.moe.fused.mxfp6_moe import moe_mxfp6_store_packed_smem_swizzled
from b12x.quantization.bf16_to_fp4_tma import (
    make_ptr,
    shared_ptr_to_u32,
    st_shared_u8,
)

_NUM_STAGES = 1
_SF_VEC_SIZE = 32
_FP6_BLOCK_ELEMS = 32


class TestKernel:
    def __init__(self):
        self.tile_shape_mnk = (128, 128, 128)
        self.threads_per_cta = 160
        self.num_mma_warps = 4
        self.tma_warp = 4
        self.num_stages = _NUM_STAGES
        self.load_register_requirement = 40

    @cute.jit
    def __call__(
        self,
        bf16_input: cute.Tensor,
        global_scale: cute.Tensor,
        packed_a: cute.Tensor,
        sfa_ptr: cute.Pointer,
        mac: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        ab = cutlass.Float6E3M2FN
        bf = cutlass.BFloat16
        sf = cutlass.Float8E8M0FNU
        fp6_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.COL_MAJOR, ab, 128),
            ab,
        )
        fp6_staged = cute.tile_to_shape(fp6_atom, (128, 128, 1), order=(0, 1, 2))
        bf16_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.COL_MAJOR, bf, 128),
            bf,
        )
        bf16_staged = cute.tile_to_shape(bf16_atom, (128, 128, self.num_stages), order=(0, 1, 2))
        mma_op = cute.nvgpu.warp.MmaMXF8Op(
            cutlass.Float8E4M3FN,
            cutlass.Float32,
            sf,
        )
        perm = sm120_utils.get_permutation_mnk(self.tile_shape_mnk, _SF_VEC_SIZE, True)
        tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((4, 2, 1)), permutation_mnk=perm)
        sfa_staged = sm120_make_smem_layout_sfa(
            tiled_mma, self.tile_shape_mnk, _SF_VEC_SIZE, 1
        )
        sfa_logical_shape = (
            packed_a.shape[0],
            self.tile_shape_mnk[2],
            packed_a.shape[2],
        )
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            sfa_logical_shape, _SF_VEC_SIZE
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
        bf16_smem1 = cute.slice_(bf16_staged, (None, None, 0))
        tma_load, gInput = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            bf16_input,
            bf16_smem1,
            (128, 128),
            num_multicast=1,
        )
        fp6_smem1 = cute.slice_(fp6_staged, (None, None, 0))
        tile_mk = cute.slice_(self.tile_shape_mnk, (None, 0, None))
        tma_store_a, gOA = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), packed_a, fp6_smem1, tile_mk
        )
        sfa_smem1 = cute.slice_(sfa_staged, (None, None, 0))
        tma_store_sfa, gOSFA = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            sfa_tensor,
            sfa_smem1,
            tile_mk,
            internal_type=cutlass.Int16,
        )
        self.kernel(
            tma_load,
            gInput,
            tma_store_a,
            gOA,
            tma_store_sfa,
            gOSFA,
            global_scale,
            bf16_staged,
            fp6_staged,
            sfa_staged,
            cute.cosize(bf16_staged),
            cute.cosize(fp6_staged),
            cute.cosize(sfa_staged),
        ).launch(
            grid=(mac, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_load: cute.CopyAtom,
        mInput: cute.Tensor,
        tma_store_a: cute.CopyAtom,
        mOA: cute.Tensor,
        tma_store_sfa: cute.CopyAtom,
        mOSFA: cute.Tensor,
        global_scale: cute.Tensor,
        bf16_smem: cute.ComposedLayout,
        fp6_smem: cute.ComposedLayout,
        sfa_smem: cute.Layout,
        bf16_cs: cutlass.Constexpr,
        fp6_cs: cutlass.Constexpr,
        sfa_cs: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        warp_idx = tidx // Int32(32)

        k_tiles = Int32(mInput.shape[1]) // Int32(128)
        total_tiles = (Int32(mInput.shape[0]) // Int32(128)) * k_tiles
        packed_cols = Int32((128 * 3) // 4)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class S:
            pmem: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sBF16: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, bf16_cs], 1024]
            sFP6: cute.struct.Align[cute.struct.MemRange[cutlass.Float6E3M2FN, fp6_cs], 1024]
            sSFA: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E8M0FNU, sfa_cs], 1024]

        st = smem.allocate(S)
        sBF16 = st.sBF16.get_tensor(bf16_smem.outer, swizzle=bf16_smem.inner)
        sFP6 = st.sFP6.get_tensor(fp6_smem.outer, swizzle=fp6_smem.inner)
        sSFA = st.sSFA.get_tensor(sfa_smem)
        sA_u8 = cute.recast_tensor(sFP6[None, None, 0], cutlass.Uint8)
        sfa_base = shared_ptr_to_u32(st.sSFA.data_ptr())
        gs_value = global_scale[Int32(0)].to(cutlass.Float32)

        cta_layout = cute.make_layout(1)
        gI = cute.local_tile(mInput, (128, 128), (None, None))
        tLsI, tLgI = cpasync.tma_partition(
            tma_load,
            0,
            cta_layout,
            cute.group_modes(sBF16, 0, 2),
            cute.group_modes(gI, 0, 2),
        )
        tile_mk = cute.slice_(self.tile_shape_mnk, (None, 0, None))
        gOA = cute.local_tile(mOA, tile_mk, (None, None, None))
        bSsA, bSgA = cpasync.tma_partition(
            tma_store_a,
            0,
            cta_layout,
            cute.group_modes(sFP6, 0, 2),
            cute.group_modes(gOA, 0, 2),
        )
        gOSFA = cute.local_tile(mOSFA, tile_mk, (None, None, None))
        bSsSFA, bSgSFA = cpasync.tma_partition(
            tma_store_sfa,
            0,
            cta_layout,
            cute.group_modes(sSFA, 0, 2),
            cute.group_modes(gOSFA, 0, 2),
        )

        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mma_warps
            ),
            tx_count=128 * 128 * cutlass.BFloat16.width // 8,
            barrier_storage=st.pmem.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mma_warps * 32
            ),
        )

        if tidx == Int32(0):
            cpasync.prefetch_descriptor(tma_load)
            cpasync.prefetch_descriptor(tma_store_a)
            cpasync.prefetch_descriptor(tma_store_sfa)
        cute.arch.sync_threads()

        tile_idx = Int32(bidx)
        blocks_per_row = Int32(128 // _FP6_BLOCK_ELEMS)

        if warp_idx < Int32(self.num_mma_warps):
            cs = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )
            while tile_idx < total_tiles:
                mt = tile_idx // k_tiles
                kt = tile_idx % k_tiles
                load_pipeline.consumer_wait(cs)
                stage = cs.index

                blk = Int32(tidx)
                while blk < Int32(128 * blocks_per_row):
                    row = blk // blocks_per_row
                    sf_block = blk % blocks_per_row
                    col0 = sf_block * Int32(_FP6_BLOCK_ELEMS)
                    vals = cute.make_rmem_tensor((_FP6_BLOCK_ELEMS,), cutlass.Float32)
                    bmax = cutlass.Float32(0.0)
                    for e in cutlass.range_constexpr(_FP6_BLOCK_ELEMS):
                        v = cutlass.Float32(sBF16[row, col0 + Int32(e), stage])
                        vals[e] = v
                        bmax = fmax_f32(bmax, fabs_f32(v))
                    lo, mid, hi, sbyte = quantize_block_fp6_e3m2_fast(
                        vals, bmax, gs_value
                    )
                    moe_mxfp6_store_packed_smem_swizzled(
                        sA_u8, row, sf_block, packed_cols, lo, mid, hi
                    )
                    outer_m_idx = row % Int32(32)
                    inner_m_idx = row // Int32(32)
                    inner_k_idx = sf_block % Int32(4)
                    k_tile_idx = sf_block // Int32(4)
                    sf_raw_idx = (
                        k_tile_idx * Int32(512)
                        + outer_m_idx * Int32(16)
                        + inner_m_idx * Int32(4)
                        + inner_k_idx
                    )
                    st_shared_u8(sfa_base + sf_raw_idx, sbyte)
                    blk += Int32(self.num_mma_warps * 32)

                load_pipeline.consumer_release(cs)
                cs.advance()

                cute.arch.fence_proxy("async.shared", space="cta")
                if warp_idx == Int32(0) and (tidx & Int32(31)) == Int32(0):
                    cute.copy(
                        tma_store_a,
                        bSsA[(None, Int32(0))],
                        bSgA[(None, mt, kt, Int32(0))],
                    )
                    cute.copy(
                        tma_store_sfa,
                        bSsSFA[(None, Int32(0))],
                        bSgSFA[(None, mt, kt, Int32(0))],
                    )
                    store_pipeline.producer_commit()
                    store_pipeline.producer_acquire()

                tile_idx += Int32(gdim)

        elif warp_idx == Int32(self.tma_warp):
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            ps = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            while tile_idx < total_tiles:
                mt = tile_idx // k_tiles
                kt = tile_idx % k_tiles
                load_pipeline.producer_acquire(ps)
                cute.copy(
                    tma_load,
                    tLgI[(None, mt, kt)],
                    tLsI[(None, ps.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(ps),
                )
                load_pipeline.producer_commit(ps)
                ps.advance()
                tile_idx += Int32(gdim)
            load_pipeline.producer_tail(ps)
