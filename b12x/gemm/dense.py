# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file is ported from the CUTLASS dense block-scaled GEMM example
# and adapted for the current Blackwell GeForce target.

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import b12x.cute.sm120_compat as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.hopper_helpers as sm90_utils
import functools
import logging
import os
import time
import torch
from cutlass import Int32, Uint8, Uint32
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp.mma import Field as WarpField
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo

from b12x.cute.compiler import KernelCompileSpec, compile as b12x_compile
from b12x.cute.fp4 import (
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    ld_global_v4_u32,
    ld_shared_v4_u32,
    shared_ptr_to_u32,
    st_global_f32,
    st_shared_u8,
    st_shared_u64,
    st_shared_v4_u32,
    u32_as_f32,
    warp_reduce,
)
from b12x.cute.utils import (
    current_cuda_stream,
    cutlass_to_torch_dtype,
    get_cutlass_dtype,
    get_max_active_clusters,
    get_num_sm,
    is_mxfp6_ab_dtype,
    make_ptr,
    mxfp6_logical_k_from_packed_bytes,
    mxfp6_tile_k,
    sm120_make_smem_layout_sfa,
    sm120_make_smem_layout_sfb,
)
from b12x.gemm.dense_mxfp6 import emit_mxfp6_dense_mma_k_block
from b12x.cute.fp6 import (
    FLOAT6_E2M3_MAX,
    FLOAT6_E3M2_MAX,
    FLOAT8_E4M3_MAX,
    cvt_f32_to_e2m3x2,
    cvt_f32_to_e3m2x2,
    cvt_f32_to_e4m3x2,
    expand_mxfp6_packed_to_bytes,
    fp6_block_ue8m0_exact,
    mx_gs_numerator,
    quantize_block_fp6_e2m3_bytes,
    quantize_block_fp6_e3m2_bytes,
    quantize_block_fp8_e4m3_bytes,
    ue8m0_output_scale_exact,
)
from b12x.cute.runtime_control import raise_if_kernel_resolution_frozen
from b12x.cute.warp_mma_compat import MmaMXF8Op as _MmaMXF8Op

logger = logging.getLogger(__name__)
_B12X_TIMING = os.getenv("B12X_TIMING", "0") == "1" or os.getenv(
    "VLLM_B12X_TIMING", "0"
) == "1"
_B12X_TIMING_THRESHOLD_MS = float(
    os.getenv(
        "B12X_TIMING_THRESHOLD_MS",
        os.getenv("VLLM_B12X_TIMING_THRESHOLD_MS", "0"),
    )
)


# @dsl_user_op on PersistentTileSchedulerParams.__init__ can rename attributes
# (e.g. raster_along_m -> _raster_along_m, cluster_shape_major_fdd ->
# cluster_shape_m_fdd) but __extract_mlir_values__ (used by TVM-FFI)
# still references the original names.
_orig_extract = utils.PersistentTileSchedulerParams.__extract_mlir_values__

# Map of source-code attr name -> runtime attr name set by @dsl_user_op
_ATTR_RENAMES = {
    "raster_along_m": "_raster_along_m",
    "cluster_shape_major_fdd": "cluster_shape_m_fdd",
    "cluster_shape_minor_fdd": "cluster_shape_n_fdd",
}


def _patched_extract(self):
    for src_name, dst_name in _ATTR_RENAMES.items():
        if not hasattr(self, src_name) and hasattr(self, dst_name):
            setattr(self, src_name, getattr(self, dst_name))
    return _orig_extract(self)


utils.PersistentTileSchedulerParams.__extract_mlir_values__ = _patched_extract


@dataclass(frozen=True)
class _DenseGemmPolicy:
    single_work_tile_per_cta: bool
    direct_one_m_tile_scheduler: bool
    use_m1_non_tma: bool


def _max_active_clusters_for(
    cluster_shape_mn: Tuple[int, int],
    sm_count: int,
) -> int:
    cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
    # For the default single-cluster launch, occupancy is bounded only by
    # the SM count. Avoid the CUTLASS hardware-info probe here because it
    # can fail on some driver/runtime combinations with INVALID_HANDLE
    # while providing no additional information for cluster_size == 1.
    return (
        sm_count
        if cluster_size == 1
        else min(get_max_active_clusters(cluster_size), sm_count)
    )


def _dense_gemm_policy_for(
    *,
    m: int,
    n: int,
    l: int,
    ab_dtype: Type[cutlass.Numeric],
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    sm_count: int,
) -> _DenseGemmPolicy:
    max_active_clusters = _max_active_clusters_for(cluster_shape_mn, sm_count)
    tile_m, tile_n = mma_tiler_mn
    one_work_tile_per_cta = (
        ((m + tile_m - 1) // tile_m)
        * ((n + tile_n - 1) // tile_n)
        * l
        <= max_active_clusters
    )
    direct_one_m_tile_scheduler = (
        one_work_tile_per_cta and m < 16 and m <= tile_m and l == 1
    )
    use_m1_non_tma = ab_dtype == cutlass.Float8E4M3FN and m == 1
    return _DenseGemmPolicy(
        single_work_tile_per_cta=direct_one_m_tile_scheduler,
        direct_one_m_tile_scheduler=direct_one_m_tile_scheduler,
        use_m1_non_tma=use_m1_non_tma,
    )


# Expand-ahead for packed-B: at k_block 0 the MMA warps wait for stage s+1 and
# expand it in place, overlapping the expansion with ALL of stage s's MMA work
# instead of putting it on the critical path at the stage boundary. This needs
# >= 4 pipeline stages of producer slack (it measurably REGRESSED at 3 stages:
# the consumer stalled on the wait before its MMA work instead of after), so
# it is gated per-kernel in _setup_attributes. Env kill-switch for A/B runs:
# B12X_PACKED_B_EXPAND_AHEAD=0 (read once at import).
_PACKED_B_EXPAND_AHEAD = os.environ.get(
    "B12X_PACKED_B_EXPAND_AHEAD", "1"
).lower() not in ("0", "false")

# Phase 4.1: fuse BF16 activation quantization into the GEMM's DMA producer
# prologue, eliminating the separate quant kernel and the HBM round-trip for
# activation codes+scales. Currently m=1 only (decode hot path). The GEMM's
# producer warp does a full-row amax scan, derives gs/alpha, then quantizes
# each K-tile's 32-element blocks directly into sA/sSFA smem.
_DENSE_FUSED_QUANT = os.environ.get(
    "B12X_DENSE_FUSED_QUANT", "0"
).lower() not in ("0", "false")


@cute.jit
def _spread_fp6_group_u32(g: Uint32) -> Uint32:
    """Spread 4 packed 6-bit codes (24 bits) into 4 byte lanes (bits[5:0] each).

    Output byte ``i`` holds ``(g >> 6*i) & 0x3F`` — identical per-group math to
    :func:`b12x.cute.fp6.expand_mxfp6_packed_to_bytes`. Two-step binary spread
    (12-bit halves to 16-bit lanes, then 6-bit fields to byte lanes): 2 shifts +
    2 LOP3-fusable mask/or pairs, ~30% fewer ops than masking each field out of
    ``g`` individually.
    """
    a = (g & Uint32(0x00000FFF)) | ((g << Uint32(4)) & Uint32(0x0FFF0000))
    return (a & Uint32(0x003F003F)) | ((a << Uint32(2)) & Uint32(0x3F003F00))


@cute.jit
def _expand_packed_b_triplet(
    a: Uint32, b: Uint32, c: Uint32
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Expand 12 packed bytes (3 LE u32 words = 16 FP6 codes) to 4 output words.

    Little-endian regroup into four 24-bit groups of 4 codes, then byte-lane
    spread per group.
    """
    g0 = a & Uint32(0x00FFFFFF)
    g1 = (a >> Uint32(24)) | ((b & Uint32(0xFFFF)) << Uint32(8))
    g2 = (b >> Uint32(16)) | ((c & Uint32(0xFF)) << Uint32(16))
    g3 = c >> Uint32(8)
    return (
        _spread_fp6_group_u32(g0),
        _spread_fp6_group_u32(g1),
        _spread_fp6_group_u32(g2),
        _spread_fp6_group_u32(g3),
    )


@cute.jit
def _expand_packed_b_stage_smem(
    sb_base_addr: Int32,
    stage: Int32,
    tidx: Int32,
    tile_n: cutlass.Constexpr,
    tile_k: cutlass.Constexpr,
    num_threads: cutlass.Constexpr,
    sync_barrier,
) -> None:
    """Expand one 3:4-packed B stage IN PLACE into the byte-container sB stage.

    TMA stages the packed tile (96 B/row, plain k-major, no swizzle) into the
    BOTTOM ``tile_n * 3*tile_k/4`` bytes of the sB stage itself — no separate
    staging buffer, which buys an extra pipeline stage (3 -> 4 for the decode
    tile). The expanded output (128 B/row, swizzled) overlaps the packed input
    region, so expansion is two-phase: every thread loads ALL of its packed
    rows into registers, one MMA-group named-barrier, then writes the expanded
    swizzled rows. Each thread owns whole rows (``tile_n / num_threads`` of
    them), so within a row reads always precede writes; the barrier orders
    them across threads.

    Raw addressing is deliberate: ``cute.recast_tensor(sB, Uint8)`` STRIPS the
    smem swizzle on this build (confirmed in the fused-MoE FC2 requant path).
    The 8-bit K-major SW128 atom (Sw<3,4,3>, 8x128 = 1024 B) gives
    ``physical = flat ^ ((row & 7) << 4)`` — the XOR touches bits 4..6 only,
    so every 16-byte unit stays contiguous and the traffic is fully 128-bit
    (6 x ld.shared.v4 in, 8 x st.shared.v4 out per row). Per-group bit math
    matches :func:`b12x.cute.fp6.expand_mxfp6_packed_to_bytes` exactly, so
    downstream ldmatrix/MMA sees bytes identical to the pre-expanded path.

    Runs on the MMA warp group only; the caller must still follow with the
    MMA-group named barrier before any ldmatrix reads of this stage.
    """
    # Whole-row ownership; ceil-divide so configs with more threads than rows
    # (e.g. the 256-thread (128,128) tile) stay correct — guarded threads skip
    # the row I/O but ALL threads reach the phase barrier.
    rows_per_thread = (tile_n + num_threads - 1) // num_threads
    packed_row_bytes = tile_k * 3 // 4
    words_per_row = packed_row_bytes // 4
    sb_stage = sb_base_addr + stage * Int32(tile_n * tile_k)

    # Phase 1: read all owned packed rows into registers (static rmem layout).
    w = cute.make_rmem_tensor((rows_per_thread * words_per_row,), Uint32)
    for r_i in cutlass.range_constexpr(rows_per_thread):
        row = Int32(tidx) + Int32(r_i * num_threads)
        if row < Int32(tile_n):
            src = sb_stage + row * Int32(packed_row_bytes)
            for c in cutlass.range_constexpr(words_per_row // 4):
                w0, w1, w2, w3 = ld_shared_v4_u32(src + Int32(c * 16))
                w[r_i * words_per_row + c * 4 + 0] = w0
                w[r_i * words_per_row + c * 4 + 1] = w1
                w[r_i * words_per_row + c * 4 + 2] = w2
                w[r_i * words_per_row + c * 4 + 3] = w3

    # All packed reads must complete before any expanded write lands.
    sync_barrier.arrive_and_wait()

    # Phase 2: expand and write the swizzled byte-container rows.
    for r_i in cutlass.range_constexpr(rows_per_thread):
        row = Int32(tidx) + Int32(r_i * num_threads)
        if row < Int32(tile_n):
            sw = (row & Int32(7)) << Int32(4)
            flat = row * Int32(tile_k)
            for o in cutlass.range_constexpr(tile_k // 16):
                base = r_i * words_per_row + o * 3
                o0, o1, o2, o3 = _expand_packed_b_triplet(
                    w[base + 0], w[base + 1], w[base + 2]
                )
                st_shared_v4_u32(
                    sb_stage + ((flat + Int32(o * 16)) ^ sw), o0, o1, o2, o3
                )


class DenseGemmKernel:
    """Implements batched matrix multiplication (C = A x SFA x B x SFB) for
    Blackwell GeForce architecture using warp-level MMA.

    Key architectural differences from the tcgen05 donor path:
    - No TMEM, no tcgen05, no 2-CTA instructions, no multi-cluster
    - Warp-level MMA: MmaMXF4NVF4Op atom m16n8k64, atom_layout=(4,2,1)
    - 256 MMA threads + 32 DMA = 288 total threads
    - PipelineTmaAsync (not PipelineTmaUmma)
    - Manual atom unroll workaround for CuTe DSL compiler SF address space bug
    - Cluster shape always (1,1,1)

    Notes:
        - Supported combinations:
            * NVF4: A/B: Float4E2M1FN, SF: Float8E4M3FN, sf_vec_size: 16
            * MXF4: A/B: Float4E2M1FN, SF: Float8E8M0FNU, sf_vec_size: 32
            * MXFP8: A/B: Float8E4M3FN, SF: Float8E8M0FNU, sf_vec_size: 32
            * MX-FP6: A/B: Float6E3M2FN or Float6E2M3FN, SF: Float8E8M0FNU,
              sf_vec_size: 32 (inline ``mxf8f6f4`` MMA, m16n8k32)
        - Tile shape constraints:
            * tile_m must be divisible by 128
            * tile_n must be divisible by 128
            * tile_k must be divisible by 64 (sf_vec_size=16) or 128 (MXFP8 / MX-FP6)
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        mma_k: int = 64,
        tile_k: Optional[int] = None,
        single_work_tile_per_cta: bool = False,
        use_prefetch: bool = False,
        enable_pdl: bool = True,
        direct_one_m_tile_scheduler: bool = False,
        use_m1_non_tma_a: bool = False,
        use_m1_non_tma_c: bool = False,
        use_m1_non_tma_sfa: bool = False,
        mxfp6_fmt: Optional[str] = None,
        mxfp6_fmt_a: Optional[str] = None,
        mxfp6_fmt_b: Optional[str] = None,
        b_packed: bool = False,
    ):
        # When set, A/B operands are MX codes carried in Float8E4M3FN
        # byte-containers: the whole kernel runs the MXFP8 smem/TMA/ldmatrix
        # machinery, and only the mainloop MMA is emitted as the inline
        # ``mxf8f6f4`` instruction (cutlass has no working 6-bit smem layout).
        # ``mxfp6_fmt_a`` / ``mxfp6_fmt_b`` may differ (W6A8: e4m3 acts, e2m3
        # weights). A single ``mxfp6_fmt`` still means both operands match.
        if mxfp6_fmt_a is None and mxfp6_fmt_b is None:
            mxfp6_fmt_a = mxfp6_fmt
            mxfp6_fmt_b = mxfp6_fmt
        elif mxfp6_fmt_a is None or mxfp6_fmt_b is None:
            raise ValueError("mxfp6_fmt_a and mxfp6_fmt_b must both be set or both None")
        self.mxfp6_fmt_a = mxfp6_fmt_a
        self.mxfp6_fmt_b = mxfp6_fmt_b
        # Native packed-FP6 streaming: B arrives 3:4-packed ``(N, 3K/4, L)`` in
        # gmem, TMA stages the packed tile into a plain smem buffer, and the MMA
        # warps expand it into the swizzled byte-container sB right after
        # consumer_wait. Cuts B HBM traffic by 25% vs the byte-container layout;
        # the ldmatrix/MMA path is unchanged.
        assert not b_packed or mxfp6_fmt_b is not None, (
            "b_packed requires the MX-FP6 path (mxfp6_fmt_b set)"
        )
        # The in-kernel expansion computes swizzled sB addresses assuming the
        # 8-bit K-major SW128 atom (8x128 = 1024 B): physical =
        # flat ^ ((row & 7) << 4). That atom is selected iff the smem major
        # size is 128, i.e. tile_k == 128 (always true for MX-FP6).
        assert not b_packed or (tile_k or sf_vec_size * 8) == 128, (
            "b_packed expansion assumes tile_k == 128 (SW128 smem atom)"
        )
        self.b_packed = b_packed
        self.a_bf16_fused = _DENSE_FUSED_QUANT and use_m1_non_tma_a
        if self.a_bf16_fused:
            _fused_fmt = mxfp6_fmt_a or "e4m3"
            self._fused_gs_num = mx_gs_numerator(_fused_fmt)
            self._fused_act_fmt = _fused_fmt
            self._fused_fmt_max = {
                "e4m3": FLOAT8_E4M3_MAX,
                "e3m2": FLOAT6_E3M2_MAX,
                "e2m3": FLOAT6_E2M3_MAX,
            }[_fused_fmt]
        else:
            self._fused_gs_num = 0.0
            self._fused_act_fmt = ""
            self._fused_fmt_max = 0.0
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.mma_k = mma_k
        if tile_k is None:
            tile_k = sf_vec_size * 8
        self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], tile_k)
        self.sfa_tile_shape_mk = (max(128, mma_tiler_mn[0]), tile_k)
        self.sfa_tiles_per_block = self.sfa_tile_shape_mk[0] // mma_tiler_mn[0]
        self.sfb_tile_shape_nk = (max(128, mma_tiler_mn[1]), tile_k)
        self.sfb_tiles_per_block = self.sfb_tile_shape_nk[0] // mma_tiler_mn[1]
        self.cluster_shape_mnk = (1, 1, 1)  # Always (1,1,1) on the current target
        self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])
        self.single_work_tile_per_cta = single_work_tile_per_cta
        self.use_prefetch = use_prefetch
        self.enable_pdl = enable_pdl
        self.direct_one_m_tile_scheduler = direct_one_m_tile_scheduler
        self.use_m1_non_tma_a = use_m1_non_tma_a
        self.use_m1_non_tma_c = use_m1_non_tma_c
        self.use_m1_non_tma_sfa = use_m1_non_tma_sfa
        if mma_tiler_mn in ((16, 64), (16, 128)):
            self.atom_shape = (1, 2, 1)
        elif mma_tiler_mn in ((32, 64), (32, 128)):
            self.atom_shape = (2, 2, 1)
        else:
            self.atom_shape = (4, 2, 1)

        self.tiled_mma = None
        self.occupancy = 1
        if mma_tiler_mn in ((16, 64), (16, 128)):
            self.num_mma_warps = 2
        elif mma_tiler_mn in ((32, 64), (32, 128)):
            self.num_mma_warps = 4
        else:
            self.num_mma_warps = 8
        # NOTE: widening the small-M atom to (1,4,1) to double the packed-B
        # expansion threads was tried and aborts in MLIR tiled_copy_retile (the
        # ldmatrix retile geometry only supports atom_n=2 here). Expansion
        # parallelism is capped at 2 warps for the decode tile.
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + 1  # 1 warp for DMA
        ) * self.num_threads_per_warp

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None

        self.buffer_align_bytes = 1024

        self.mma_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
            mma_op = _MmaMXF8Op(
                self.a_dtype,
                self.acc_dtype,
                self.sf_dtype,
            )
        elif cutlass.const_expr(
            self.a_dtype == cutlass.Float6E3M2FN
            or self.a_dtype == cutlass.Float6E2M3FN
        ):
            # MX-FP6 uses inline ``mxf8f6f4`` MMA in the mainloop. Build tiled_mma
            # with the MXFP8 op so smem/SF layouts match m16n8k32 geometry.
            mma_op = _MmaMXF8Op(
                cutlass.Float8E4M3FN,
                self.acc_dtype,
                self.sf_dtype,
            )
        else:
            mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
                self.a_dtype,
                self.acc_dtype,
                self.sf_dtype,
            )
        atom_shape = self.atom_shape
        atom_layout = cute.make_layout(atom_shape)
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk,
            self.sf_vec_size,
            cutlass.const_expr(
                self.a_dtype == cutlass.Float8E4M3FN
                or self.a_dtype == cutlass.Float6E3M2FN
                or self.a_dtype == cutlass.Float6E2M3FN
            ),
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )
        # Bare atom for manual unroll workaround (avoids hasAuxTensor address space bug)
        self.mma_atom = cute.make_mma_atom(mma_op)
        # Compute atom loop bounds from tile shape and atom/layout shape
        # MMA atom: m16n8k64 for FP4, m16n8k32 for MXFP8.
        mma_m, mma_n, mma_k = 16, 8, self.mma_k
        self.num_m_tiles = self.tile_shape_mnk[0] // (mma_m * atom_shape[0])
        self.num_n_tiles = self.tile_shape_mnk[1] // (mma_n * atom_shape[1])
        self.num_k_blocks = self.tile_shape_mnk[2] // mma_k

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        # Compute the smem size of SFA/SFB
        sfa_smem_layout_per_stage = sm120_make_smem_layout_sfa(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )
        sfb_smem_layout_per_stage = sm120_make_smem_layout_sfb(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            sfa_smem_layout_per_stage,
            sfb_smem_layout_per_stage,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
            self.b_packed,
        )

        assert self.epi_stage > 0, (
            "epi_stage <= 0, not enough shared memory. This configuration will be skipped."
        )

        # Decided here because it depends on the computed stage depth; see the
        # _PACKED_B_EXPAND_AHEAD module comment for the >= 4 stage rationale.
        self.packed_expand_ahead = (
            self.b_packed and _PACKED_B_EXPAND_AHEAD and self.ab_stage >= 4
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )

        # Plain (non-swizzled) k-major staging layout for the 3:4-packed B
        # tile, ALIASED into the bottom of each sB stage: TMA writes the 96
        # packed bytes/row there (16-byte aligned box, within TMA's
        # non-swizzled constraints) and the MMA warps expand IN PLACE into the
        # full swizzled 128 B/row stage (two-phase, see
        # _expand_packed_b_stage_smem). No separate staging buffer means the
        # packed mode pays zero extra smem -> one more pipeline stage. The
        # stage stride is sB's full stage size, NOT the packed tile size.
        if self.b_packed:
            self.b_packed_smem_layout_staged = cute.make_layout(
                (
                    self.tile_shape_mnk[1],
                    self.tile_shape_mnk[2] * 3 // 4,
                    self.ab_stage,
                ),
                stride=(
                    self.tile_shape_mnk[2] * 3 // 4,
                    1,
                    self.tile_shape_mnk[1] * self.tile_shape_mnk[2],
                ),
            )
        else:
            self.b_packed_smem_layout_staged = None

    def _build_shared_storage(self):
        """Define the kernel smem struct (trace-time only).

        Single variant for both modes: in ``b_packed`` mode the packed TMA
        tile is aliased into the bottom of each sB stage (in-place expansion),
        so no extra region exists. The DSL preprocessor traces into this
        method when called from the JIT ``__call__``; keeping it a plain
        helper with a single return avoids the struct-flattening pitfalls.
        """
        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        x_bf16: cute.Tensor = None,
        w_gscale: cute.Tensor = None,
    ):
        """Execute the GEMM operation.

        Args:
            a: Input tensor A (byte-containers, or dummy when a_bf16_fused)
            b: Input tensor B
            sfa: Scale factor tensor for A (or dummy when a_bf16_fused)
            sfb: Scale factor tensor for B
            c: Output tensor C
            alpha: Alpha scaling factor tensor, shape (1,), float32.
                   In fused mode the kernel WRITES alpha here.
            max_active_clusters: Max active clusters
            stream: CUDA stream
            epilogue_op: Elementwise epilogue function
            x_bf16: BF16 activation input (fused quant mode only)
            w_gscale: Weight global scale, shape (1,), f32 (fused mode only)
        """
        # Setup static attributes
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.sf_dtype = sfa.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        self.sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa.iterator, self.sfa_layout)

        # With packed B the gmem extent is 3K/4 bytes, but scale-factor geometry
        # follows the LOGICAL K (one UE8M0 per 32 codes), so rebuild the shape.
        if cutlass.const_expr(self.b_packed):
            sfb_shape_nkl = (b.shape[0], b.shape[1] * 4 // 3, b.shape[2])
        else:
            sfb_shape_nkl = b.shape
        self.sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            sfb_shape_nkl, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb.iterator, self.sfb_layout)

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )
        if cutlass.const_expr(self.b_packed):
            # TMA loads the packed tile (tile_n, 3*tile_k/4 bytes) into the
            # plain staging layout; the swizzled sB is filled in-kernel.
            tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
                b,
                self.b_packed_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2] * 3 // 4),
                1,
            )
        else:
            tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
                b,
                self.b_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
                1,
            )
        if cutlass.const_expr(self.use_m1_non_tma_sfa):
            tma_atom_sfa = tma_atom_b
            tma_tensor_sfa = sfa_tensor
        else:
            tma_atom_sfa, tma_tensor_sfa = self._make_tma_atoms_and_tensors(
                sfa_tensor,
                self.sfa_smem_layout_staged,
                self.sfa_tile_shape_mk,
                1,
                internal_type=cutlass.Int16,
            )
        tma_atom_sfb, tma_tensor_sfb = self._make_tma_atoms_and_tensors(
            sfb_tensor,
            self.sfb_smem_layout_staged,
            self.sfb_tile_shape_nk,
            1,
            internal_type=cutlass.Int16,
        )
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
        )

        # Built in a plain (non-JIT) method: conditional @cute.struct class
        # definitions inside a JIT-traced function body have no precedent on
        # this build, while plain helper methods execute as ordinary Python.
        self.shared_storage = self._build_shared_storage()

        # Unused (never traced) when b_packed is off; pass the expanded layout
        # as a stand-in so the kernel signature stays uniform.
        if cutlass.const_expr(self.b_packed):
            b_packed_smem_layout_arg = self.b_packed_smem_layout_staged
        else:
            b_packed_smem_layout_arg = self.b_smem_layout_staged

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            sfa_tensor,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            c,
            self.tiled_mma,
            self.mma_atom,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            b_packed_smem_layout_arg,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
            epilogue_op,
            alpha,
            x_bf16,
            w_gscale,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )
        return

    def _partition_fragment_SFA(
        self,
        sfa_tensor: cute.Tensor,
        thr_mma: cute.ThrMma,
        tidx: int,
    ):
        return sm120_utils.partition_fragment_SFA(sfa_tensor, thr_mma, tidx)

    def _partition_fragment_SFB(
        self,
        sfb_tensor: cute.Tensor,
        thr_mma: cute.ThrMma,
        tidx: int,
    ):
        return sm120_utils.partition_fragment_SFB(sfb_tensor, thr_mma, tidx)

    def _thrfrg_SFA(
        self, sfa_tensor, tiled_mma: cute.TiledMma
    ):
        return sm120_utils.thrfrg_SFA(sfa_tensor, tiled_mma)

    def _thrfrg_SFB(
        self, sfb_tensor, tiled_mma: cute.TiledMma
    ):
        return sm120_utils.thrfrg_SFB(sfb_tensor, tiled_mma)

    def _get_layoutSFA_TV(self, tiled_mma: cute.TiledMma):
        return sm120_utils.get_layoutSFA_TV(tiled_mma)

    def _get_layoutSFB_TV(self, tiled_mma: cute.TiledMma):
        return sm120_utils.get_layoutSFB_TV(tiled_mma)

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        directA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        directSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        directC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        mma_atom: cute.MmaAtom,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        b_packed_smem_layout_staged,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
        directX_bf16: cute.Tensor,
        w_gscale: cute.Tensor,
    ):
        alpha_value = alpha[0].to(cutlass.Float32)

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == 0:
            if cutlass.const_expr(not self.use_m1_non_tma_a):
                cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(not self.use_m1_non_tma_sfa):
                cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            if cutlass.const_expr(not self.use_m1_non_tma_c):
                cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        # B's TMA transaction covers the packed staging tile when b_packed (the
        # expanded sB is filled by the MMA warps, not by TMA).
        if cutlass.const_expr(self.b_packed):
            b_tma_smem_layout = cute.slice_(
                b_packed_smem_layout_staged, (None, None, 0)
            )
        else:
            b_tma_smem_layout = b_smem_layout
        if cutlass.const_expr(self.use_m1_non_tma_sfa):
            tma_copy_bytes = (
                cute.size_in_bytes(self.b_dtype, b_tma_smem_layout)
                + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
            )
            if cutlass.const_expr(not self.use_m1_non_tma_a):
                tma_copy_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)
        else:
            tma_copy_bytes = (
                cute.size_in_bytes(self.a_dtype, a_smem_layout)
                + cute.size_in_bytes(self.b_dtype, b_tma_smem_layout)
                + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
                + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
            )

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipeline setup
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # Generate smem tensors
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        if cutlass.const_expr(self.b_packed):
            # Packed TMA destination ALIASED into sB's storage (bottom 96 B of
            # each row span, stage stride = full sB stage): no separate buffer,
            # expansion happens in place (see _expand_packed_b_stage_smem).
            sBPacked = cute.make_tensor(
                storage.sB.data_ptr(), b_packed_smem_layout_staged
            )
            # Raw u32 smem address for the packed->container expansion. Two
            # constraints force this exact spot: (1) cute.recast_tensor strips
            # sB's swizzle (see _expand_packed_b_stage_smem), so the expansion
            # needs raw addresses; (2) @cute.struct instances cannot be
            # flattened across DYNAMIC ifs (NVIDIA/cutlass#3268) and the if-
            # capture analysis is syntactic, so ``storage`` must never be
            # referenced inside the warp-dispatch branches - only this Int32
            # address may cross into them.
            sb_base_addr = shared_ptr_to_u32(storage.sB.data_ptr())
        if cutlass.const_expr(self.a_bf16_fused):
            sa_base_addr = shared_ptr_to_u32(storage.sA.data_ptr())
            ssfa_base_addr = shared_ptr_to_u32(storage.sSFA.data_ptr())
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        # Local_tile partition global tensors
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        if cutlass.const_expr(self.b_packed):
            # Packed gmem extent: 96 bytes per 128-wide logical K-tile, same
            # K-tile count as the A side ((3K/4)/96 == K/128).
            gB_nkl = cute.local_tile(
                mB_nkl,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2] * 3 // 4),
                (None, None, None),
            )
        else:
            gB_nkl = cute.local_tile(
                mB_nkl,
                cute.slice_(self.tile_shape_mnk, (0, None, None)),
                (None, None, None),
            )
        if cutlass.const_expr(not self.use_m1_non_tma_sfa):
            gSFA_mkl = cute.local_tile(
                mSFA_mkl,
                self.sfa_tile_shape_mk,
                (None, None, None),
            )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            self.sfb_tile_shape_nk,
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # Partition for TiledMMA
        thr_mma = tiled_mma.get_slice(tidx)

        # TMA partitions for A
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        if cutlass.const_expr(not self.use_m1_non_tma_a):
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                a_cta_crd,
                a_cta_layout,
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA_mkl, 0, 2),
            )

        # TMA partitions for B (targets the packed staging buffer when b_packed)
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        if cutlass.const_expr(self.b_packed):
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                b_cta_crd,
                b_cta_layout,
                cute.group_modes(sBPacked, 0, 2),
                cute.group_modes(gB_nkl, 0, 2),
            )
        else:
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                b_cta_crd,
                b_cta_layout,
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB_nkl, 0, 2),
            )

        # TMA partitions for SFA
        if cutlass.const_expr(not self.use_m1_non_tma_sfa):
            tAsSFA, tAgSFA = cpasync.tma_partition(
                tma_atom_sfa,
                a_cta_crd,
                a_cta_layout,
                cute.group_modes(sSFA, 0, 2),
                cute.group_modes(gSFA_mkl, 0, 2),
            )
            tAsSFA = cute.filter_zeros(tAsSFA)
            tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA partitions for SFB
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(gSFB_nkl, 0, 2),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        # Make fragments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)

        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrSFA_full = self._partition_fragment_SFA(sSFA[None, None, 0], thr_mma, tidx)
        tCrSFB_full = self._partition_fragment_SFB(sSFB[None, None, 0], thr_mma, tidx)

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # Cluster/thread sync
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Tile scheduler
        block_idx = cute.arch.block_idx()
        if cutlass.const_expr(self.direct_one_m_tile_scheduler):
            work_tile = WorkTileInfo(
                (Int32(0), Int32(block_idx[2]), Int32(0)),
                cutlass.Boolean(1),
            )
        else:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, block_idx, cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

        # Pipeline states
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        if cutlass.const_expr(self.packed_expand_ahead):
            # Second consumer-side view of the mainloop pipeline, kept exactly
            # one stage ahead of mainloop_consumer_state: it advances once at
            # each work-tile prologue and once per k_tile at k_block 0, for a
            # total of k_tile_cnt per tile — the same as the main state's
            # (k_tile_cnt - 1) in-loop advances plus 1 in the hoisted tail —
            # so the two stay phase-aligned across persistent work tiles.
            packed_b_lookahead_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )

        # MMA warp group
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # Copy atoms for SMEM->RMEM
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            atom_copy_ldmatrix_SF = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.sf_dtype,
            )
            smem_tiled_copy_SFA = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                self._get_layoutSFA_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_tiled_copy_SFB = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                self._get_layoutSFB_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            thr_copy_ldmatrix_SFA = smem_tiled_copy_SFA.get_slice(tidx)
            thr_copy_ldmatrix_SFB = smem_tiled_copy_SFB.get_slice(tidx)
            tCsSFA_copy_view_full = thr_copy_ldmatrix_SFA.partition_S(sSFA)
            tCrSFA_copy_view_full = thr_copy_ldmatrix_SFA.retile(tCrSFA_full)
            tCsSFB_copy_view_full = thr_copy_ldmatrix_SFB.partition_S(sSFB)
            tCrSFB_copy_view_full = thr_copy_ldmatrix_SFB.retile(tCrSFB_full)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                sfa_tile_offset = tile_coord_mnl[0] % self.sfa_tiles_per_block
                sfb_tile_offset = tile_coord_mnl[1] % self.sfb_tiles_per_block
                if cutlass.const_expr(self.sfa_tiles_per_block > 1):
                    sSFA_tile = cute.local_tile(
                        sSFA,
                        cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                        (sfa_tile_offset, 0, None),
                    )
                    tCsSFA_tile_copy_view = thr_copy_ldmatrix_SFA.partition_S(sSFA_tile)
                    tCrSFA_tile = self._partition_fragment_SFA(
                        sSFA_tile[None, None, 0], thr_mma, tidx
                    )
                    tCrSFA_tile_copy_view = thr_copy_ldmatrix_SFA.retile(tCrSFA_tile)
                else:
                    tCsSFA_tile_copy_view = tCsSFA_copy_view_full
                    tCrSFA_tile = tCrSFA_full
                    tCrSFA_tile_copy_view = tCrSFA_copy_view_full
                if cutlass.const_expr(self.sfb_tiles_per_block > 1):
                    sSFB_tile = cute.local_tile(
                        sSFB,
                        cute.slice_(self.tile_shape_mnk, (0, None, None)),
                        (sfb_tile_offset, 0, None),
                    )
                    tCsSFB_tile_copy_view = thr_copy_ldmatrix_SFB.partition_S(sSFB_tile)
                    tCrSFB_tile = self._partition_fragment_SFB(
                        sSFB_tile[None, None, 0], thr_mma, tidx
                    )
                    tCrSFB_tile_copy_view = thr_copy_ldmatrix_SFB.retile(tCrSFB_tile)
                else:
                    tCsSFB_tile_copy_view = tCsSFB_copy_view_full
                    tCrSFB_tile = tCrSFB_full
                    tCrSFB_tile_copy_view = tCrSFB_copy_view_full
                accumulators.fill(0.0)

                # Pipelined MAINLOOP
                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                if cutlass.const_expr(self.b_packed):
                    # Expand the TMA-staged packed B tile in place into the
                    # swizzled byte-container sB BEFORE any ldmatrix touches
                    # this stage (two-phase; internal read/write barrier).
                    _expand_packed_b_stage_smem(
                        sb_base_addr,
                        mainloop_consumer_state.index,
                        Int32(tidx),
                        self.tile_shape_mnk[1],
                        self.tile_shape_mnk[2],
                        self.num_mma_warps * self.num_threads_per_warp,
                        self.mma_sync_barrier,
                    )
                    self.mma_sync_barrier.arrive_and_wait()
                    if cutlass.const_expr(self.packed_expand_ahead):
                        # Lookahead expanded this stage too (same index as the
                        # consumer state at the prologue); move it one ahead.
                        packed_b_lookahead_state.advance()
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsSFA_p = tCsSFA_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                tCsSFB_p = tCsSFB_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

                tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_tile_copy_view)
                tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_tile_copy_view)

                cute.copy(
                    smem_tiled_copy_SFA,
                    tCsSFA_p_filtered[None, None, 0],
                    tCrSFA_copy_view_filtered[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_SFB,
                    tCsSFB_p_filtered[None, None, 0],
                    tCrSFB_copy_view_filtered[None, None, 0],
                )

                for k_tile in range(0, k_tile_cnt - 1, 1, unroll=2):
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )

                        if cutlass.const_expr(self.packed_expand_ahead):
                            if k_block_idx == 0:
                                # Expand-ahead: wait for stage s+1 (producer
                                # runs >= 2 stages ahead at this depth, so the
                                # wait is usually free) and expand it now, so
                                # the whole expansion overlaps stage s's MMA
                                # k-blocks instead of sitting at the stage
                                # boundary. The matching write->ldmatrix fence
                                # is the deferred barrier after the last
                                # k-block's MMA below.
                                lookahead_peek = (
                                    mainloop_pipeline.consumer_try_wait(
                                        packed_b_lookahead_state
                                    )
                                )
                                mainloop_pipeline.consumer_wait(
                                    packed_b_lookahead_state, lookahead_peek
                                )
                                _expand_packed_b_stage_smem(
                                    sb_base_addr,
                                    packed_b_lookahead_state.index,
                                    Int32(tidx),
                                    self.tile_shape_mnk[1],
                                    self.tile_shape_mnk[2],
                                    self.num_mma_warps
                                    * self.num_threads_per_warp,
                                    self.mma_sync_barrier,
                                )
                                packed_b_lookahead_state.advance()

                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )

                            tCsA_p = tCsA_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsB_p = tCsB_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsSFA_p = tCsSFA_tile_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsSFB_p = tCsSFB_tile_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )
                            if cutlass.const_expr(
                                self.b_packed and not self.packed_expand_ahead
                            ):
                                # Shallow-pipeline fallback (< 4 stages): the
                                # new stage just became full (packed bytes
                                # only); expand before the k_block_next=0
                                # ldmatrix of this stage later in this
                                # iteration. Reads of the PREVIOUS stage
                                # finished at the prior iteration's copies, so
                                # writing here is safe. The matching barrier
                                # sits AFTER the MMA block below: the last
                                # k-block's MMA reads only registers, so it
                                # overlaps expansion stragglers instead of
                                # waiting on the barrier first. (In expand-
                                # ahead mode this stage was already expanded
                                # at k_block 0; the consumer_wait above then
                                # returns immediately.)
                                _expand_packed_b_stage_smem(
                                    sb_base_addr,
                                    mainloop_consumer_state.index,
                                    Int32(tidx),
                                    self.tile_shape_mnk[1],
                                    self.tile_shape_mnk[2],
                                    self.num_mma_warps
                                    * self.num_threads_per_warp,
                                    self.mma_sync_barrier,
                                )

                        # Manual atom unroll: avoids hasAuxTensor address space bug
                        for _mt in range(self.num_m_tiles):
                            for _nt in range(self.num_n_tiles):
                                if cutlass.const_expr(self.mxfp6_fmt_a is not None):
                                    emit_mxfp6_dense_mma_k_block(
                                        accumulators,
                                        tCrA,
                                        tCrB,
                                        tCrSFA_tile,
                                        tCrSFB_tile,
                                        _mt,
                                        _nt,
                                        k_block_idx,
                                        self.mxfp6_fmt_a,
                                        self.mxfp6_fmt_b,
                                    )
                                else:
                                    mma_atom.set(
                                        WarpField.SFA,
                                        tCrSFA_tile[None, _mt, k_block_idx].iterator,
                                    )
                                    mma_atom.set(
                                        WarpField.SFB,
                                        tCrSFB_tile[None, _nt, k_block_idx].iterator,
                                    )
                                    cute.gemm(
                                        mma_atom,
                                        accumulators[None, _mt, _nt],
                                        tCrA[None, _mt, k_block_idx],
                                        tCrB[None, _nt, k_block_idx],
                                        accumulators[None, _mt, _nt],
                                    )
                        if cutlass.const_expr(self.b_packed):
                            # Deferred expansion barrier (see expansion call
                            # above): must precede the k_block_next=0 ldmatrix
                            # of the just-expanded next stage right below.
                            if k_block_idx == num_k_blocks - 1:
                                self.mma_sync_barrier.arrive_and_wait()
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )

                        tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                        tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                        tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_tile_copy_view)
                        tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_tile_copy_view)
                        cute.copy(
                            smem_tiled_copy_SFA,
                            tCsSFA_p_filtered[None, None, k_block_next],
                            tCrSFA_copy_view_filtered[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_SFB,
                            tCsSFB_p_filtered[None, None, k_block_next],
                            tCrSFB_copy_view_filtered[None, None, k_block_next],
                        )

                # Hoist out last k_tile
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )

                    if k_block_idx == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                        tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                        tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                        tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_tile_copy_view)
                        tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_tile_copy_view)
                        cute.copy(
                            smem_tiled_copy_SFA,
                            tCsSFA_p_filtered[None, None, k_block_next],
                            tCrSFA_copy_view_filtered[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_SFB,
                            tCsSFB_p_filtered[None, None, k_block_next],
                            tCrSFB_copy_view_filtered[None, None, k_block_next],
                        )
                    # Manual atom unroll: avoids hasAuxTensor address space bug
                    for _mt in range(self.num_m_tiles):
                        for _nt in range(self.num_n_tiles):
                            if cutlass.const_expr(self.mxfp6_fmt_a is not None):
                                emit_mxfp6_dense_mma_k_block(
                                    accumulators,
                                    tCrA,
                                    tCrB,
                                    tCrSFA_tile,
                                    tCrSFB_tile,
                                    _mt,
                                    _nt,
                                    k_block_idx,
                                    self.mxfp6_fmt_a,
                                    self.mxfp6_fmt_b,
                                )
                            else:
                                mma_atom.set(
                                    WarpField.SFA,
                                    tCrSFA_tile[None, _mt, k_block_idx].iterator,
                                )
                                mma_atom.set(
                                    WarpField.SFB,
                                    tCrSFB_tile[None, _nt, k_block_idx].iterator,
                                )
                                cute.gemm(
                                    mma_atom,
                                    accumulators[None, _mt, _nt],
                                    tCrA[None, _mt, k_block_idx],
                                    tCrB[None, _nt, k_block_idx],
                                    accumulators[None, _mt, _nt],
                                )

                # EPILOGUE
                _is_m_major = self.c_layout.is_m_major_c()
                if cutlass.const_expr(self.c_dtype.width == 16):
                    copy_atom_r2s = cute.make_copy_atom(
                        cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2), self.c_dtype,
                    )
                else:
                    copy_atom_r2s = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), self.c_dtype,
                    )

                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.c_layout.is_m_major_c(),
                        2,
                    ),
                    self.c_dtype,
                )

                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s,
                    tiled_copy_C_Atom,
                )

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)

                sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                tcgc_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)

                bSG_sD, bSG_gD = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    sepi_for_tma_partition,
                    tcgc_for_tma_partition,
                )

                epi_rest_m = bSG_gD.shape[1][0]
                epi_rest_n = bSG_gD.shape[1][1]
                epi_tile_m = self.epi_tile[0]
                epi_tile_n = self.epi_tile[1]
                mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rAcc, mode=[1])
                mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rAcc, mode=[2])
                has_multi_epi_store = cutlass.const_expr(
                    not (self.epi_stage == 1 and epi_rest_m == 1 and epi_rest_n == 1)
                )
                tma_store_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.num_mma_warps * self.num_threads_per_warp,
                )
                tma_store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.epi_stage,
                    producer_group=tma_store_producer_group,
                )

                for epi_m in cutlass.range_constexpr(epi_rest_m):
                    for epi_n in cutlass.range_constexpr(epi_rest_n):
                        MmaMPerEpiM = epi_tile_m // mma_tile_m
                        MmaNPerEpiN = epi_tile_n // mma_tile_n
                        for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                            for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                                mma_n = (epi_n * MmaNPerEpiN) + mma_n_in_epi
                                mma_m = (epi_m * MmaMPerEpiM) + mma_m_in_epi
                                tRS_rD_slice = tRS_rD[
                                    (None, mma_m_in_epi, mma_n_in_epi)
                                ]
                                tRS_rAcc_slice = tRS_rAcc[(None, mma_m, mma_n)]
                                for elem_idx in cutlass.range_constexpr(
                                    cute.size(tRS_rD_slice)
                                ):
                                    tRS_rD_slice[elem_idx] = tRS_rAcc_slice[elem_idx]

                        # Type conversion with alpha scaling
                        tRS_rD_out = cute.make_rmem_tensor(
                            tRS_rD_layout.shape, self.c_dtype
                        )
                        acc_vec = tRS_rD.load()
                        acc_vec = epilogue_op((alpha_value * acc_vec).to(self.c_dtype))
                        tRS_rD_out.store(acc_vec)

                        # Register to shared memory
                        epi_buffer = (epi_m * epi_rest_n + epi_n) % cute.size(
                            tRS_sD, mode=[3]
                        )
                        if has_multi_epi_store:
                            self.epilog_sync_barrier.arrive_and_wait()
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()

                        # Copy from shared memory to global memory
                        gmem_coord = (epi_m, epi_n)
                        if cutlass.const_expr(self.use_m1_non_tma_c):
                            for n_iter in cutlass.range_constexpr(
                                (
                                    self.epi_tile[1]
                                    + self.num_mma_warps * self.num_threads_per_warp
                                    - 1
                                )
                                // (self.num_mma_warps * self.num_threads_per_warp)
                            ):
                                n_local = Int32(tidx) + Int32(
                                    n_iter
                                    * self.num_mma_warps
                                    * self.num_threads_per_warp
                                )
                                if n_local < Int32(self.epi_tile[1]):
                                    n_coord = (
                                        tile_coord_mnl[1]
                                        * Int32(self.tile_shape_mnk[1])
                                        + Int32(epi_n * self.epi_tile[1])
                                        + n_local
                                    )
                                    if n_coord < Int32(directC_mnl.shape[1]):
                                        directC_mnl[
                                            (
                                                Int32(0),
                                                n_coord,
                                                tile_coord_mnl[2],
                                            )
                                        ] = sC[(Int32(0), n_local, epi_buffer)]
                        else:
                            if warp_idx == 0:
                                cute.copy(
                                    tma_atom_c,
                                    bSG_sD[(None, epi_buffer)],
                                    bSG_gD[(None, gmem_coord)],
                                )
                                if has_multi_epi_store:
                                    tma_store_pipeline.producer_commit()
                                    tma_store_pipeline.producer_acquire()

                # Advance to the next work tile
                if cutlass.const_expr(self.single_work_tile_per_cta):
                    work_tile = WorkTileInfo(
                        work_tile.tile_idx,
                        cutlass.Boolean(0),
                    )
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                if has_multi_epi_store:
                    tma_store_pipeline.producer_tail()

        # DMA warp group
        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                if cutlass.const_expr(not self.use_m1_non_tma_a):
                    tAgA_mkl = tAgA[
                        (None, tile_coord_mnl[0], None, tile_coord_mnl[2])
                    ]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                if cutlass.const_expr(not self.use_m1_non_tma_sfa):
                    sfa_tile_coord_m = tile_coord_mnl[0] // self.sfa_tiles_per_block
                    tAgSFA_mkl = tAgSFA[
                        (None, sfa_tile_coord_m, None, tile_coord_mnl[2])
                    ]
                sfb_tile_coord_n = tile_coord_mnl[1] // self.sfb_tiles_per_block
                tBgSFB_nkl = tBgSFB[(None, sfb_tile_coord_n, None, tile_coord_mnl[2])]

                mainloop_producer_state.reset_count()
                lane = Int32(tidx % self.num_threads_per_warp)

                if cutlass.const_expr(self.a_bf16_fused):
                    full_k = Int32(directX_bf16.shape[1])
                    nvec = full_k // Int32(8)
                    bf16_base = get_ptr_as_int64(directX_bf16, Int32(0))
                    local_amax = cutlass.Float32(0.0)
                    i_vec = lane
                    while i_vec < nvec:
                        w0, w1, w2, w3 = ld_global_v4_u32(
                            bf16_base + cutlass.Int64(i_vec) * cutlass.Int64(16)
                        )
                        for w in (w0, w1, w2, w3):
                            hi = u32_as_f32(w & Uint32(0x7FFF0000))
                            lo = u32_as_f32(
                                (w << Uint32(16)) & Uint32(0x7FFF0000)
                            )
                            local_amax = fmax_f32(
                                local_amax, fmax_f32(hi, lo)
                            )
                        i_vec += Int32(self.num_threads_per_warp)
                    fused_amax = warp_reduce(local_amax, fmax_f32)
                    fused_amax_c = fmax_f32(
                        fused_amax, cutlass.Float32(1e-6)
                    )
                    fused_gs = (
                        cutlass.Float32(self._fused_gs_num) / fused_amax_c
                    )
                    _fq_k_base = Int32(0)
                    _fq_sa_stage = Int32(0)
                    _fq_packed_scales = Uint32(0)
                    _fq_k_abs = Int32(0)
                    _fq_val = cutlass.Float32(0.0)
                    _fq_bmax = cutlass.Float32(0.0)
                    _fq_su32 = Uint32(0)
                    _fq_inv = cutlass.Float32(0.0)
                    _fq_scaled = cutlass.Float32(0.0)
                    _fq_pair = Uint32(0)
                    _fq_code = Uint8(0)
                    _fq_ssfa_stage = Int32(0)
                    _fq_lin = Int32(0)
                    _fq_m = Int32(0)
                    _fq_sg_idx = Int32(0)
                    _fq_sf_off = Int32(0)
                    _fq_sb = Uint8(0)
                    _fq_sfa_sg = self.tile_shape_mnk[2] // self.sf_vec_size
                    _fq_sfa_slots = self.sfa_tile_shape_mk[0] * _fq_sfa_sg
                    _fq_sg = 0
                    _fq_si = 0

                if cutlass.const_expr(self.use_m1_non_tma_a):
                    a_iter = 0
                    k_local = Int32(0)
                    k_coord = Int32(0)

                if cutlass.const_expr(self.use_m1_non_tma_sfa):
                    scale_groups_per_k_tile = (
                        self.tile_shape_mnk[2] // self.sf_vec_size
                    )
                    sfa_slots = (
                        self.sfa_tile_shape_mk[0] * scale_groups_per_k_tile
                    )
                    sfa_iter = 0
                    linear = Int32(0)
                    m_local = Int32(0)
                    scale_group = Int32(0)
                    k_local_sfa = Int32(0)
                    k_coord_sfa = Int32(0)

                for k_tile in range(0, k_tile_cnt, 1, unroll=2):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

                    tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                    tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]
                    if cutlass.const_expr(not self.use_m1_non_tma_a):
                        tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                        tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                    if cutlass.const_expr(not self.use_m1_non_tma_sfa):
                        tAgSFA_k = tAgSFA_mkl[
                            (None, mainloop_producer_state.count)
                        ]
                        tAsSFA_pipe = tAsSFA[(None, mainloop_producer_state.index)]

                    tBgSFB_k = tBgSFB_nkl[(None, mainloop_producer_state.count)]
                    tBsSFB_pipe = tBsSFB[(None, mainloop_producer_state.index)]

                    if cutlass.const_expr(self.a_bf16_fused):
                        _fq_k_base = (
                            mainloop_producer_state.count
                            * Int32(self.tile_shape_mnk[2])
                        )
                        _fq_sa_stage = (
                            mainloop_producer_state.index
                            * Int32(
                                self.tile_shape_mnk[0]
                                * self.tile_shape_mnk[2]
                            )
                        )
                        _fq_packed_scales = Uint32(0)

                        for _fq_sg in cutlass.range_constexpr(
                            self.tile_shape_mnk[2] // self.sf_vec_size
                        ):
                            _fq_k_abs = (
                                _fq_k_base
                                + Int32(_fq_sg * self.sf_vec_size)
                                + lane
                            )
                            _fq_val = cutlass.Float32(
                                directX_bf16[(Int32(0), _fq_k_abs)]
                            )
                            _fq_bmax = warp_reduce(
                                fabs_f32(_fq_val), fmax_f32
                            )
                            _fq_su32 = fp6_block_ue8m0_exact(
                                _fq_bmax,
                                fused_gs,
                                cutlass.Float32(self._fused_fmt_max),
                            )
                            _fq_inv = ue8m0_output_scale_exact(
                                _fq_su32, fused_gs
                            )
                            _fq_scaled = _fq_val * _fq_inv
                            if cutlass.const_expr(
                                self._fused_act_fmt == "e4m3"
                            ):
                                _fq_pair = cvt_f32_to_e4m3x2(
                                    cutlass.Float32(0.0), _fq_scaled
                                )
                            elif cutlass.const_expr(
                                self._fused_act_fmt == "e3m2"
                            ):
                                _fq_pair = cvt_f32_to_e3m2x2(
                                    cutlass.Float32(0.0), _fq_scaled
                                )
                            else:
                                _fq_pair = cvt_f32_to_e2m3x2(
                                    cutlass.Float32(0.0), _fq_scaled
                                )
                            _fq_code = Uint8(_fq_pair & Uint32(0xFF))

                            # sA row-0 store (SW128 XOR is zero for row 0)
                            st_shared_u8(
                                sa_base_addr
                                + _fq_sa_stage
                                + Int32(_fq_sg * self.sf_vec_size)
                                + lane,
                                _fq_code,
                            )

                            _fq_packed_scales = _fq_packed_scales | (
                                (_fq_su32 & Uint32(0xFF))
                                << Uint32(_fq_sg * 8)
                            )

                        # Broadcast scale bytes to all 128 SFA M-rows.
                        # SFA atom layout (Sw<3,4,3> UE8M0 block): flat offset
                        # for (m_row, sg) = (m%32)*16 + (m//32)*4 + sg.
                        _fq_sfa_sg = self.tile_shape_mnk[2] // self.sf_vec_size
                        _fq_sfa_slots = self.sfa_tile_shape_mk[0] * _fq_sfa_sg
                        _fq_ssfa_stage = (
                            mainloop_producer_state.index
                            * Int32(
                                (self.sfa_tile_shape_mk[0] // 128) * 128
                                * _fq_sfa_sg
                            )
                        )
                        for _fq_si in cutlass.range_constexpr(
                            (_fq_sfa_slots + self.num_threads_per_warp - 1)
                            // self.num_threads_per_warp
                        ):
                            _fq_lin = lane + Int32(
                                _fq_si * self.num_threads_per_warp
                            )
                            if _fq_lin < Int32(_fq_sfa_slots):
                                _fq_m = _fq_lin // Int32(_fq_sfa_sg)
                                _fq_sg_idx = _fq_lin - _fq_m * Int32(
                                    _fq_sfa_sg
                                )
                                _fq_sf_off = (
                                    (_fq_m & Int32(31)) * Int32(16)
                                    + (_fq_m >> Int32(5)) * Int32(4)
                                    + _fq_sg_idx
                                )
                                _fq_sb = Uint8(
                                    (_fq_packed_scales
                                     >> (Uint32(_fq_sg_idx) * Uint32(8)))
                                    & Uint32(0xFF)
                                )
                                st_shared_u8(
                                    ssfa_base_addr
                                    + _fq_ssfa_stage
                                    + _fq_sf_off,
                                    _fq_sb,
                                )

                        cute.arch.fence_proxy("async.shared", space="cta")

                    elif cutlass.const_expr(self.use_m1_non_tma_a):
                        for a_iter in cutlass.range_constexpr(
                            (self.tile_shape_mnk[2] + self.num_threads_per_warp - 1)
                            // self.num_threads_per_warp
                        ):
                            k_local = lane + Int32(a_iter * self.num_threads_per_warp)
                            if k_local < Int32(self.tile_shape_mnk[2]):
                                k_coord = (
                                    mainloop_producer_state.count
                                    * Int32(self.tile_shape_mnk[2])
                                    + k_local
                                )
                                sA[
                                    (
                                        Int32(0),
                                        k_local,
                                        mainloop_producer_state.index,
                                    )
                                ] = directA_mkl[
                                    (
                                        Int32(0),
                                        k_coord,
                                        tile_coord_mnl[2],
                                    )
                                ]
                    else:
                        cute.copy(
                            tma_atom_a,
                            tAgA_k,
                            tAsA_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                        )

                    if cutlass.const_expr(self.a_bf16_fused):
                        pass  # scales already written above
                    elif cutlass.const_expr(self.use_m1_non_tma_sfa):
                        scale_groups_per_k_tile = (
                            self.tile_shape_mnk[2] // self.sf_vec_size
                        )
                        sfa_slots = (
                            self.sfa_tile_shape_mk[0] * scale_groups_per_k_tile
                        )
                        for sfa_iter in cutlass.range_constexpr(
                            (sfa_slots + self.num_threads_per_warp - 1)
                            // self.num_threads_per_warp
                        ):
                            linear = lane + Int32(
                                sfa_iter * self.num_threads_per_warp
                            )
                            m_local = linear // Int32(scale_groups_per_k_tile)
                            scale_group = (
                                linear - m_local * Int32(scale_groups_per_k_tile)
                            )
                            k_local_sfa = scale_group * Int32(self.sf_vec_size)
                            k_coord_sfa = (
                                mainloop_producer_state.count
                                * Int32(self.tile_shape_mnk[2])
                                + k_local_sfa
                            )
                            if linear < Int32(sfa_slots):
                                sSFA[
                                    (
                                        m_local,
                                        k_local_sfa,
                                        mainloop_producer_state.index,
                                    )
                                ] = directSFA_mkl[
                                    (
                                        Int32(0),
                                        k_coord_sfa,
                                        tile_coord_mnl[2],
                                    )
                                ]
                        cute.arch.fence_proxy("async.shared", space="cta")
                    else:
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_k,
                            tAsSFA_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                        )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_k,
                        tBsSFB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                if cutlass.const_expr(self.single_work_tile_per_cta):
                    work_tile = WorkTileInfo(
                        work_tile.tile_idx,
                        cutlass.Boolean(0),
                    )
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple,
        a_dtype,
        b_dtype,
        sf_dtype,
        sfa_smem_layout,
        sfb_smem_layout,
        epi_tile: tuple,
        c_dtype,
        smem_capacity: int,
        occupancy: int,
        b_packed: bool = False,
    ) -> tuple:
        epi_stage_max = (tile_shape_mnk[1] // epi_tile[1]) * (
            tile_shape_mnk[0] // epi_tile[0]
        )
        epi_stage = min(epi_stage_max, 4)
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        # b_packed costs no extra smem: the packed TMA tile is aliased into the
        # bottom of each sB stage and expanded in place.
        sf_bytes_per_stage = (
            cute.size(cute.filter_zeros(sfa_smem_layout).shape) * sf_dtype.width // 8
            + cute.size(cute.filter_zeros(sfb_smem_layout).shape) * sf_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        raw_ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // (ab_bytes_per_stage + sf_bytes_per_stage)
        ab_stage = max(1, min(raw_ab_stage, 4))
        if tile_shape_mnk[0] == 64 and tile_shape_mnk[1] == 128:
            ab_stage = max(1, min(raw_ab_stage, 5))
        if b_packed:
            # In-place packed staging freed 12 KB/stage; deeper pipelines give
            # the producer the lookahead the packed consumer chain needs.
            ab_stage = max(1, min(raw_ab_stage, 5))
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple,
        epi_tile: tuple,
        a_dtype,
        a_layout,
        b_dtype,
        b_layout,
        ab_stage: int,
        c_dtype,
        c_layout,
        epi_stage: int,
        sf_vec_size: int,
        tiled_mma,
    ) -> tuple:
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.is_k_major_a()
        b_is_k_major = b_layout.is_k_major_b()
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]

        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        sfa_smem_layout_staged = sm120_make_smem_layout_sfa(
            tiled_mma,
            tile_shape_mnk,
            sf_vec_size,
            ab_stage,
        )
        sfb_smem_layout_staged = sm120_make_smem_layout_sfb(
            tiled_mma,
            tile_shape_mnk,
            sf_vec_size,
            ab_stage,
        )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                c_layout,
                c_dtype,
                c_major_mode_size,
            ),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )

        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            epi_smem_layout_staged,
        )

    @staticmethod
    def _compute_grid(
        c,
        tile_shape_mnk: tuple,
        max_active_clusters,
    ) -> tuple:
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c,
        epi_smem_layout_staged,
        epi_tile: tuple,
    ) -> tuple:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )
        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor,
        smem_layout_staged,
        smem_tile: tuple,
        mcast_dim: int,
        internal_type=None,
    ) -> tuple:
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
            internal_type=internal_type,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def can_implement(
        ab_dtype,
        sf_dtype,
        sf_vec_size: int,
        c_dtype,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        # The current target only supports cluster (1,1)
        if cluster_shape_mn != (1, 1):
            return False
        # Tile M must be divisible by 128; tile N follows 64-column warpgroup
        # quanta, while the SF paths round narrow tiles up to full 128-element
        # scale-factor blocks.
        if mma_tiler_mn in ((16, 64), (16, 128), (32, 64), (32, 128)):
            if ab_dtype not in (
                cutlass.Float8E4M3FN,
                cutlass.Float6E3M2FN,
                cutlass.Float6E2M3FN,
            ):
                return False
        elif mma_tiler_mn[0] % 64 != 0 or mma_tiler_mn[1] % 64 != 0:
            return False
        # The current target supports FP4, MXFP8, and MX-FP6 warp MMA paths.
        if ab_dtype not in (
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            cutlass.Float6E3M2FN,
            cutlass.Float6E2M3FN,
        ):
            return False
        # Current target MMA constraints:
        #   sf_vec_size=16 requires sf_dtype=Float8E4M3FN
        #   sf_vec_size=32 requires sf_dtype=Float8E8M0FNU
        if sf_vec_size == 16 and sf_dtype != cutlass.Float8E4M3FN:
            return False
        if sf_vec_size == 32 and sf_dtype != cutlass.Float8E8M0FNU:
            return False
        if ab_dtype == cutlass.Float8E4M3FN and sf_vec_size != 32:
            return False
        if is_mxfp6_ab_dtype(ab_dtype) and sf_vec_size != 32:
            return False
        # Only 16-bit output types supported for now
        if c_dtype not in (cutlass.Float16, cutlass.BFloat16):
            return False
        # A must be K-major, B must be K-major
        if a_major != "k" or b_major != "k":
            return False
        # Alignment: K must be divisible by tile_k
        if ab_dtype == cutlass.Float8E4M3FN or is_mxfp6_ab_dtype(ab_dtype):
            tile_k = mxfp6_tile_k() if is_mxfp6_ab_dtype(ab_dtype) else 128
        else:
            tile_k = sf_vec_size * 8
        if k % tile_k != 0:
            return False
        return True



class _DenseGemmLaunch:
    def __init__(
        self,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
        ab_dtype: torch.dtype,
        sf_dtype: torch.dtype,
        c_dtype: torch.dtype,
        alpha_dtype: torch.dtype,
        sf_vec_size: int,
        mma_k: int,
        tile_k: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        policy: _DenseGemmPolicy,
        sm_count: int,
        sm_version: str,
        mxfp6_fmt: Optional[str] = None,
        mxfp6_fmt_a: Optional[str] = None,
        mxfp6_fmt_b: Optional[str] = None,
        b_packed: bool = False,
    ):
        if mxfp6_fmt_a is None and mxfp6_fmt_b is None:
            mxfp6_fmt_a = mxfp6_fmt
            mxfp6_fmt_b = mxfp6_fmt
        self._mxfp6_fmt_a = mxfp6_fmt_a
        self._mxfp6_fmt_b = mxfp6_fmt_b
        self._b_packed = b_packed
        self._n = n
        self._k = k
        self._l = l
        self._a_major = a_major
        self._b_major = b_major
        self._c_major = c_major
        self._ab_dtype = ab_dtype
        self._sf_dtype = sf_dtype
        self._c_dtype = c_dtype
        self._alpha_dtype = alpha_dtype
        self._sf_vec_size = sf_vec_size
        self._mma_k = mma_k
        self._tile_k = tile_k
        self._mma_tiler_mn = mma_tiler_mn
        self._cluster_shape_mn = cluster_shape_mn
        self._policy = policy

        if not DenseGemmKernel.can_implement(
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            c_dtype,
            mma_tiler_mn,
            cluster_shape_mn,
            n,
            k,
            l,
            a_major,
            b_major,
            c_major,
        ):
            raise TypeError(
                "dense_gemm launch is unsupported with "
                f"{ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype}, "
                f"{mma_tiler_mn}, {cluster_shape_mn}, {n}, {k}, {l}, "
                f"{a_major}, {b_major}, {c_major}"
            )

        self._max_active_clusters = _max_active_clusters_for(
            self._cluster_shape_mn, sm_count
        )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        x_bf16_ptr: cute.Pointer,
        w_gscale_ptr: cute.Pointer,
        m: cutlass.Int32,
        current_stream: cuda.CUstream,
    ):
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout(
                (m, self._k, self._l),
                order=(0, 1, 2) if self._a_major == "m" else (1, 0, 2),
            ),
        )
        # Packed B carries 3 bytes per 4 codes: gmem extent is 3K/4.
        b_k_extent = self._k * 3 // 4 if self._b_packed else self._k
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout(
                (self._n, b_k_extent, self._l),
                order=(0, 1, 2) if self._b_major == "n" else (1, 0, 2),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout(
                (m, self._n, self._l),
                order=(0, 1, 2) if self._c_major == "m" else (1, 0, 2),
            ),
        )
        alpha_tensor = cute.make_tensor(
            alpha_ptr,
            layout=cute.make_ordered_layout((1,), order=(0,)),
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, layout=cute.make_layout((1,)))
        sfb_tensor = cute.make_tensor(sfb_ptr, layout=cute.make_layout((1,)))
        x_bf16_tensor = cute.make_tensor(
            x_bf16_ptr,
            layout=cute.make_ordered_layout((m, self._k), order=(0, 1)),
        )
        w_gscale_tensor = cute.make_tensor(
            w_gscale_ptr,
            layout=cute.make_ordered_layout((Int32(1),), order=(0,)),
        )
        policy = self._policy
        DenseGemmKernel(
            sf_vec_size=self._sf_vec_size,
            mma_tiler_mn=self._mma_tiler_mn,
            cluster_shape_mn=self._cluster_shape_mn,
            mma_k=self._mma_k,
            tile_k=self._tile_k,
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            use_m1_non_tma_a=policy.use_m1_non_tma,
            use_m1_non_tma_c=policy.use_m1_non_tma,
            use_m1_non_tma_sfa=policy.use_m1_non_tma,
            mxfp6_fmt_a=self._mxfp6_fmt_a,
            mxfp6_fmt_b=self._mxfp6_fmt_b,
            b_packed=self._b_packed,
        )(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            alpha_tensor,
            self._max_active_clusters,
            current_stream,
            x_bf16=x_bf16_tensor,
            w_gscale=w_gscale_tensor,
        )


@functools.cache
def _get_compiled_dense_gemm(
    n: int,
    k: int,
    l: int,
    a_major: str,
    b_major: str,
    c_major: str,
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    alpha_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    mma_k: int,
    tile_k: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    policy: _DenseGemmPolicy,
    sm_count: int,
    sm_version: str,
    mxfp6_fmt: Optional[str] = None,
    mxfp6_fmt_a: Optional[str] = None,
    mxfp6_fmt_b: Optional[str] = None,
    b_packed: bool = False,
) -> Callable:
    def _make_runtime_pointers(
        input_tensors: Optional[List[torch.Tensor]],
    ) -> List[cute.Pointer]:
        if input_tensors is None:
            (
                a_data_ptr,
                b_data_ptr,
                sfa_data_ptr,
                sfb_data_ptr,
                c_data_ptr,
                alpha_data_ptr,
                x_bf16_data_ptr,
                w_gscale_data_ptr,
            ) = [16 for _ in range(8)]
        else:
            (
                a_tensor_gpu,
                b_tensor_gpu,
                sfa_tensor_gpu,
                sfb_tensor_gpu,
                c_tensor_gpu,
                alpha_tensor_gpu,
                x_bf16_tensor_gpu,
                w_gscale_tensor_gpu,
            ) = input_tensors
            (
                a_data_ptr,
                b_data_ptr,
                sfa_data_ptr,
                sfb_data_ptr,
                c_data_ptr,
                alpha_data_ptr,
            ) = (
                a_tensor_gpu.data_ptr(),
                b_tensor_gpu.data_ptr(),
                sfa_tensor_gpu.data_ptr(),
                sfb_tensor_gpu.data_ptr(),
                c_tensor_gpu.data_ptr(),
                alpha_tensor_gpu.data_ptr(),
            )
            x_bf16_data_ptr = (
                x_bf16_tensor_gpu.data_ptr()
                if x_bf16_tensor_gpu is not None
                else 16
            )
            w_gscale_data_ptr = (
                w_gscale_tensor_gpu.data_ptr()
                if w_gscale_tensor_gpu is not None
                else 16
            )

        return [
            make_ptr(ab_dtype, a_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(ab_dtype, b_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(sf_dtype, sfa_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(sf_dtype, sfb_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(c_dtype, c_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(alpha_dtype, alpha_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, x_bf16_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, w_gscale_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
        ]

    launch = _DenseGemmLaunch(
        n=n,
        k=k,
        l=l,
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        alpha_dtype=alpha_dtype,
        sf_vec_size=sf_vec_size,
        mma_k=mma_k,
        tile_k=tile_k,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        policy=policy,
        sm_count=sm_count,
        sm_version=sm_version,
        mxfp6_fmt=mxfp6_fmt,
        mxfp6_fmt_a=mxfp6_fmt_a,
        mxfp6_fmt_b=mxfp6_fmt_b,
        b_packed=b_packed,
    )
    compile_key = (
        n,
        k,
        l,
        ab_dtype,
        sf_dtype,
        c_dtype,
        alpha_dtype,
        sf_vec_size,
        mma_k,
        tile_k,
        mma_tiler_mn,
        cluster_shape_mn,
        policy,
        sm_count,
        sm_version,
        mxfp6_fmt_a if mxfp6_fmt_a is not None else mxfp6_fmt,
        mxfp6_fmt_b if mxfp6_fmt_b is not None else mxfp6_fmt,
        b_packed,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=compile_key,
    )
    compiled_kernel = b12x_compile(
        launch,
        *_make_runtime_pointers(None),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("gemm.dense", 1, compile_key),
    )

    def tensor_api(
        a_tensor_gpu: torch.Tensor,
        b_tensor_gpu: torch.Tensor,
        sfa_tensor_gpu: torch.Tensor,
        sfb_tensor_gpu: torch.Tensor,
        c_tensor_gpu: Optional[torch.Tensor] = None,
        alpha_tensor_gpu: Optional[torch.Tensor] = None,
        x_bf16_tensor_gpu: Optional[torch.Tensor] = None,
        w_gscale_tensor_gpu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m = a_tensor_gpu.shape[0]
        if c_tensor_gpu is None:
            c_tensor_gpu = torch.empty(
                (m, n, l),
                dtype=cutlass_to_torch_dtype(c_dtype),
                device=a_tensor_gpu.device,
            )
        if alpha_tensor_gpu is None:
            alpha_tensor_gpu = torch.ones(
                (1,),
                dtype=torch.float32,
                device=a_tensor_gpu.device,
            )

        nonlocal compiled_kernel
        compiled_kernel(
            *_make_runtime_pointers(
                [
                    a_tensor_gpu,
                    b_tensor_gpu,
                    sfa_tensor_gpu,
                    sfb_tensor_gpu,
                    c_tensor_gpu,
                    alpha_tensor_gpu,
                    x_bf16_tensor_gpu,
                    w_gscale_tensor_gpu,
                ]
            ),
            m,
            current_cuda_stream(),
        )
        return c_tensor_gpu

    return tensor_api


def _select_default_mma_tiler_mn(
    m: int,
    n: int,
    sm_count: int,
    *,
    is_mxfp8: bool,
    is_mxfp6: bool = False,
) -> Tuple[int, int]:
    coarse_tile = (128, 128)
    if (is_mxfp8 or is_mxfp6) and n > 1536:
        # Small-M decode specialization (Phase 2.1). m<=16 covers BS1 decode
        # (m=1) AND the MTP spec-decode verify forward (m = 1 +
        # num_speculative_tokens, typically 5-8), which streams the full
        # weight set every step; all fit in one 16-row M-tile. Widening past
        # the historical m<=4 cap is safe ONLY because the vLLM plugin
        # warm-runs every dense shape at m in {1,2,4,5,8,16} at load time
        # (_warm_dense_decode_shapes), so no first-use JIT can hit
        # mid-serving on live prefill-tail token counts.
        if m <= 16:
            return (16, 128)
        return coarse_tile

    coarse_tiles = ((m + coarse_tile[0] - 1) // coarse_tile[0]) * (
        (n + coarse_tile[1] - 1) // coarse_tile[1]
    )
    # The coarse CTA-count heuristic misses exact-small-M, wide-N cases: a wide
    # N dimension can generate plenty of CTAs even while each 128-row M tile is
    # mostly empty. Keep using the narrower 64x128 tile while the 128x128 plan
    # still leaves the GPU below the existing half-SM occupancy proxy.
    if n > 1536:
        if m <= 64:
            return (64, 128)
        if m <= 256 and coarse_tiles < max(1, sm_count // 2):
            return (64, 128)
    if m <= 128 and coarse_tiles < max(1, sm_count // 2):
        if n > 1536:
            return (64, 128)
        medium_tile = (128, 64)
        medium_tiles = ((m + medium_tile[0] - 1) // medium_tile[0]) * (
            (n + medium_tile[1] - 1) // medium_tile[1]
        )
        if medium_tiles < max(1, sm_count // 2):
            return (64, 64)
        return (128, 64)
    return coarse_tile


def _expand_packed_mxfp6_ab(t: torch.Tensor, num_codes: int) -> torch.Tensor:
    """Expand a 3:4-packed FP6 operand ``(X, packed_k, L)`` to byte-containers.

    Returns an ``(X, num_codes, L)`` uint8 view whose underlying memory is laid out
    K-major (matching ``a_major``/``b_major == "k"``), so the compiled kernel reads
    each FP6 code from one byte. The launch only uses ``data_ptr`` plus the compiled
    ``(X, K, L)`` layout, so the returned view's strides need not be contiguous; the
    contiguous backing buffer is kept alive by the returned view.
    """
    t_lxk = t.permute(2, 0, 1).contiguous()  # (L, X, packed_k), packed_k fastest
    e_lxk = expand_mxfp6_packed_to_bytes(t_lxk, num_codes)  # (L, X, num_codes)
    return e_lxk.permute(1, 2, 0)  # (X, num_codes, L), K stride 1


def dense_gemm(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: Optional[torch.Tensor] = None,
    *,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    sm_count: Optional[int] = None,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    alpha: Optional[torch.Tensor] = None,
    alpha_dtype: Optional[str] = None,
    a_preexpanded: bool = False,
    b_preexpanded: bool = False,
    b_packed: bool = False,
    a_fmt: Optional[str] = None,
    b_fmt: Optional[str] = None,
    x_bf16: Optional[torch.Tensor] = None,
    w_gscale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute dense block-scaled GEMM for one expert-major batch stack.

    ``b_preexpanded``: when True the RHS is already in 1-byte-per-code FP8
    container layout (the output of ``_expand_packed_mxfp6_ab``) and the
    per-call weight expansion is skipped. Callers hoist the expansion to load
    time so the static weight is unpacked once instead of every token.

    ``a_preexpanded``: same for the LHS — its shape is then ``(M, K, L)`` with
    one code per byte (e.g. the quantizer's ``emit="bytes"`` output), so the
    logical K is read directly from the shape and no expansion runs.

    ``b_packed``: native packed-FP6 streaming. The RHS stays in the 3:4-packed
    wire format ``(N, 3K/4, L)``; the kernel TMA-streams the packed bytes and
    expands to byte-containers in smem. 25% less B HBM traffic than the
    byte-container layout and no expanded copy resident in VRAM. MX-FP6 only;
    mutually exclusive with ``b_preexpanded``.

    ``a_fmt`` / ``b_fmt``: optional per-operand MX sub-formats (``e2m3`` /
    ``e3m2`` / ``e4m3``). When omitted, both operands use the format implied by
    ``ab_dtype``. W6A8 dense passes ``a_fmt="e4m3"`` (activations) and
    ``b_fmt="e2m3"`` (weights) with ``ab_dtype="float6_e2m3fn"`` so the weight
    packing / expansion path stays on the MX-FP6 branch.
    """
    a_torch, sfa_torch = lhs
    b_torch, sfb_torch = rhs
    if b_packed and b_preexpanded:
        raise ValueError("b_packed and b_preexpanded are mutually exclusive")

    m, k, l = a_torch.shape
    n, _, _ = b_torch.shape
    mxfp6_fmt: Optional[str] = None
    mxfp6_fmt_a: Optional[str] = None
    mxfp6_fmt_b: Optional[str] = None
    if ab_dtype == "float4_e2m1fn":
        is_mxfp8 = False
        is_mxfp6 = False
        k *= 2
        mma_k = 64
        tile_k = sf_vec_size * 8
    elif ab_dtype == "float8_e4m3fn":
        is_mxfp8 = True
        is_mxfp6 = False
        mma_k = 32
        tile_k = 128
    elif ab_dtype in ("float6_e3m2fn", "float6_e2m3fn"):
        is_mxfp8 = False
        is_mxfp6 = True
        if sf_vec_size != 32:
            raise ValueError("MX-FP6 dense_gemm requires sf_vec_size=32")
        if sf_dtype != "float8_e8m0fnu":
            raise ValueError("MX-FP6 dense_gemm requires sf_dtype='float8_e8m0fnu'")
        if not a_preexpanded:
            k = mxfp6_logical_k_from_packed_bytes(k)
        mma_k = 32
        tile_k = mxfp6_tile_k(sf_vec_size)
        weight_fmt = "e3m2" if ab_dtype == "float6_e3m2fn" else "e2m3"
        mxfp6_fmt_b = b_fmt if b_fmt is not None else weight_fmt
        mxfp6_fmt_a = a_fmt if a_fmt is not None else weight_fmt
        mxfp6_fmt = mxfp6_fmt_a if mxfp6_fmt_a == mxfp6_fmt_b else None
        for name, fmt in (("a_fmt", mxfp6_fmt_a), ("b_fmt", mxfp6_fmt_b)):
            if fmt not in ("e2m3", "e3m2", "e4m3"):
                raise ValueError(f"unsupported {name}={fmt!r}")
    else:
        raise TypeError(f"dense_gemm unsupported ab_dtype: {ab_dtype}")
    if b_packed:
        if mxfp6_fmt_b is None:
            raise ValueError("b_packed requires an MX-FP6 ab_dtype")
        if b_torch.shape[1] * 4 != k * 3:
            raise ValueError(
                f"b_packed expects (N, 3K/4, L); got packed K bytes "
                f"{b_torch.shape[1]} for logical K {k}"
            )

    if sm_count is None:
        sm_count = get_num_sm(a_torch.device)
    ab_cutlass_dtype = get_cutlass_dtype(ab_dtype)
    if mxfp6_fmt_a is not None:
        # Stage 1: carry MX codes in Float8E4M3FN byte-containers so the kernel
        # uses cutlass's native 8-bit smem/TMA/ldmatrix path (cutlass cannot build
        # a 6-bit smem layout). Expand the 3:4-packed inputs to one code per byte
        # at this load boundary; the on-disk/wire format stays packed. E4M3
        # activations are already one-byte-per-code (a_preexpanded required).
        ab_cutlass_dtype = cutlass.Float8E4M3FN
        if not a_preexpanded:
            if mxfp6_fmt_a == "e4m3":
                raise ValueError("e4m3 activations require a_preexpanded=True")
            a_torch = _expand_packed_mxfp6_ab(a_torch, k)
        if not (b_preexpanded or b_packed):
            b_torch = _expand_packed_mxfp6_ab(b_torch, k)
    sf_cutlass_dtype = get_cutlass_dtype(sf_dtype)
    c_cutlass_dtype = get_cutlass_dtype(c_dtype)
    if mma_tiler_mn is None:
        mma_tiler_mn = _select_default_mma_tiler_mn(
            m,
            n,
            sm_count,
            is_mxfp8=is_mxfp8,
            is_mxfp6=is_mxfp6,
        )
    if alpha_dtype is None:
        alpha_dtype = "float32" if alpha is None else str(alpha.dtype).split(".")[-1]
    alpha_cutlass_dtype = get_cutlass_dtype(alpha_dtype)
    policy = _dense_gemm_policy_for(
        m=m,
        n=n,
        l=l,
        ab_dtype=ab_cutlass_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sm_count=sm_count,
    )

    t0 = time.perf_counter() if _B12X_TIMING else 0.0
    cache_before = _get_compiled_dense_gemm.cache_info() if _B12X_TIMING else None
    compiled = _get_compiled_dense_gemm(
        n=n,
        k=k,
        l=l,
        a_major="k",
        b_major="k",
        c_major="n",
        ab_dtype=ab_cutlass_dtype,
        sf_dtype=sf_cutlass_dtype,
        c_dtype=c_cutlass_dtype,
        alpha_dtype=alpha_cutlass_dtype,
        sf_vec_size=sf_vec_size,
        mma_k=mma_k,
        tile_k=tile_k,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        policy=policy,
        sm_count=sm_count,
        sm_version="sm_120",
        mxfp6_fmt=mxfp6_fmt,
        mxfp6_fmt_a=mxfp6_fmt_a,
        mxfp6_fmt_b=mxfp6_fmt_b,
        b_packed=b_packed,
    )
    t_compiled = time.perf_counter() if _B12X_TIMING else 0.0
    result = compiled(
        a_tensor_gpu=a_torch,
        b_tensor_gpu=b_torch,
        sfa_tensor_gpu=sfa_torch,
        sfb_tensor_gpu=sfb_torch,
        c_tensor_gpu=out,
        alpha_tensor_gpu=alpha,
        x_bf16_tensor_gpu=x_bf16,
        w_gscale_tensor_gpu=w_gscale,
    )
    if _B12X_TIMING:
        t_launch = time.perf_counter()
        cache_after = _get_compiled_dense_gemm.cache_info()
        assert cache_before is not None
        compile_ms = (t_compiled - t0) * 1000.0
        launch_ms = (t_launch - t_compiled) * 1000.0
        total_ms = (t_launch - t0) * 1000.0
        if total_ms >= _B12X_TIMING_THRESHOLD_MS:
            logger.warning(
                "b12x_dense_gemm timing m=%d n=%d k=%d l=%d ab=%s sf=%s c=%s "
                "tile=%s cache_hit=%s compile_or_lookup=%.3fms "
                "launch_enqueue=%.3fms total=%.3fms cache=%s",
                m,
                n,
                k,
                l,
                ab_dtype,
                sf_dtype,
                c_dtype,
                mma_tiler_mn,
                cache_after.hits > cache_before.hits,
                compile_ms,
                launch_ms,
                total_ms,
                cache_after,
            )
    return result
