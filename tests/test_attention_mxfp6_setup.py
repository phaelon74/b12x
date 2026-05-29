"""Smoke tests for MX-FP6 attention MMA helpers (no GPU compile)."""

from __future__ import annotations

import cutlass

from b12x.attention import mxfp6_mma
from b12x.cute.fp6 import (
    cvt_bf16x2_to_e3m2x2,
    cvt_bf16x2x2_to_e3m2x4,
)


def test_mxfp6_mma_helpers_import():
    assert mxfp6_mma._literal_qk_mma_into_sfrag_mxfp6_raw_paged is not None
    assert mxfp6_mma._literal_pv_mma_into_ofrag_mxfp6_scaled_mla is not None
    assert mxfp6_mma._compute_mxfp6_tile_partials is not None


def test_bf16_to_fp6_cvt_helpers_exist():
    assert cvt_bf16x2_to_e3m2x2 is not None
    assert cvt_bf16x2x2_to_e3m2x4 is not None


def test_mla_kernel_accepts_kv_nope_dtype_param():
    from b12x.attention.mla.kernel import SparseMLAKernel

    kernel = SparseMLAKernel(1, kv_nope_dtype=cutlass.Float6E3M2FN)
    assert kernel.kv_nope_dtype == cutlass.Float6E3M2FN
