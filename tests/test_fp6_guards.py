from __future__ import annotations

import cutlass
import pytest

from b12x.cute.utils import mxfp6_packed_k_bytes, mxfp6_tile_k
from b12x.gemm.dense import DenseGemmKernel
from b12x.integration import tp_moe
from b12x.moe.fused.micro import MoEMicroKernelSilu


def test_dense_gemm_can_implement_mixed_fp6_requires_uniform_ab_dtype() -> None:
    """Dense GEMM uses one ab_dtype for both operands; mixed formats are MoE-only."""
    assert DenseGemmKernel.can_implement(
        cutlass.Float6E3M2FN,
        cutlass.Float8E8M0FNU,
        sf_vec_size=32,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        n=128,
        k=128,
        l=1,
        a_major="k",
        b_major="k",
        c_major="n",
    )


def test_moe_default_mxfp6_mixed_format_dtypes() -> None:
    act, wt, sf = tp_moe._mxfp6_cutlass_dtypes("mxfp6_default")
    assert act is cutlass.Float6E3M2FN
    assert wt is cutlass.Float6E2M3FN
    assert sf is cutlass.Float8E8M0FNU


def test_moe_uniform_mxfp6_e2m3_source_format() -> None:
    act, wt, _ = tp_moe._mxfp6_cutlass_dtypes("mxfp6_e2m3")
    assert act is cutlass.Float6E2M3FN
    assert wt is cutlass.Float6E2M3FN


def test_micro_kernel_rejects_mxfp6() -> None:
    assert MoEMicroKernelSilu.is_supported_mxfp6(1, 128, 128, 8, 4) is False


def test_w6a6_packed_k_bytes_matches_tile_k() -> None:
    k = mxfp6_tile_k()
    assert mxfp6_packed_k_bytes(k) == (k * 3) // 4
    assert k % 128 == 0


def test_normalize_fp6_source_format_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="source_format must be"):
        tp_moe._normalize_fp6_source_format("not_a_format")
