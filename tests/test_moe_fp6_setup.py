"""Smoke tests for MX-FP6 fused MoE kernel setup (no GPU compile)."""

from __future__ import annotations

import cutlass

from b12x.cute.utils import mxfp6_tile_k, mxfp6_num_k_blocks
from b12x.moe.fused import mxfp6_moe
from b12x.moe.fused.dynamic import MoEDynamicKernelBackend
from b12x.moe.fused.micro import MoEMicroKernelBackend, MXFP6_BLOCK_SIZE
from b12x.moe.fused.static import _MoEStaticKernelBase


def test_mxfp6_moe_helpers_import():
    assert mxfp6_moe.moe_emit_mma_k_block is not None


def test_mxfp6_tile_k_matches_moe_expectation():
    assert mxfp6_tile_k() == 128
    assert mxfp6_num_k_blocks(128) == 4


def test_moe_static_dynamic_init_tile_k_default():
    backend = _MoEStaticKernelBase(
        sf_vec_size=32,
        mma_tiler_mn=(128, 128),
        output_tile_count_n=1,
    )
    assert backend.tile_shape_mnk[2] == 256
    dyn = MoEDynamicKernelBackend(sf_vec_size=32, mma_tiler_mn=(128, 128))
    assert dyn.tile_shape_mnk[2] == 256


def test_micro_mxfp6_not_supported():
    assert MXFP6_BLOCK_SIZE == 32
    assert MoEMicroKernelBackend.is_supported_mxfp6(1, 128, 128, 8, 8) is False


def test_fp6_element_types_exist():
    assert cutlass.Float6E3M2FN is not None
    assert cutlass.Float6E2M3FN is not None
