"""GPU tests for sparse MLA MX-FP6 QK/PV (Phase A: direct SparseMLAKernel construction)."""

from __future__ import annotations

import cutlass
import pytest
import torch

from b12x.attention.mla.kernel import (
    SparseMLAKernel,
    _MLA_HEADS_PER_TILE,
    _extract_packed_kv_runtime_views,
    _to_kernel_tensor,
    _torch_to_cutlass_dtype,
    _view_last_dim_as_u32,
    clear_sparse_mla_kernel_cache,
)
from b12x.attention.mla.reference import (
    pack_mla_kv_cache_fp6_reference,
    sparse_mla_fp6_reference,
)
from b12x.cute.compiler import KernelCompileSpec, clear_compile_cache, launch as b12x_launch
from b12x.cute.fp6 import Fp6Format
from b12x.cute.utils import current_cuda_stream

from .helpers import require_sm120

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for MLA FP6 GPU tests"
)

_MLA_HEAD_DIM = 576
_MLA_V_DIM = 512
_MLA_SM_SCALE = (_MLA_V_DIM + 64) ** -0.5


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        actual.reshape(-1).float(),
        expected.reshape(-1).float(),
        dim=0,
    ).item()


def _make_synthetic_mla_case(
    device: torch.device,
    *,
    num_heads: int = 16,
    cache_len: int = 128,
    width: int = 64,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    q_all = (
        torch.randn(
            (1, num_heads, _MLA_HEAD_DIM),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device, dtype=torch.bfloat16)
        * 0.15
    )
    k_nope = (
        torch.randn(
            (cache_len, 1, _MLA_V_DIM),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device, dtype=torch.bfloat16)
        * 0.15
    )
    k_rope = (
        torch.randn(
            (cache_len, 1, 64),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device, dtype=torch.bfloat16)
        * 0.15
    )
    page_table_1 = torch.arange(width, dtype=torch.int32, device=device).unsqueeze(0)
    active_token_counts = torch.tensor([width], dtype=torch.int32, device=device)
    return q_all, k_nope, k_rope, page_table_1, active_token_counts


def _launch_sparse_mla_kernel_fp6(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    active_token_counts: torch.Tensor,
    output: torch.Tensor,
    kv_nope_dtype: type,
    identity_page_table: bool = True,
) -> None:
    clear_sparse_mla_kernel_cache()
    clear_compile_cache()

    kv_rows_u32, kv_scales = _extract_packed_kv_runtime_views(kv_cache)
    q_u32 = _view_last_dim_as_u32(q_all)
    sm_scale_tensor = torch.tensor([_MLA_SM_SCALE], dtype=torch.float32, device=q_all.device)
    head_tiles = (int(output.shape[1]) + _MLA_HEADS_PER_TILE - 1) // _MLA_HEADS_PER_TILE
    kernel = SparseMLAKernel(
        head_tiles,
        identity_page_table=identity_page_table,
        kv_nope_dtype=kv_nope_dtype,
    )
    args = (
        _to_kernel_tensor(q_u32, cutlass.Uint32, assumed_align=16),
        _to_kernel_tensor(kv_rows_u32, cutlass.Uint32, assumed_align=16),
        _to_kernel_tensor(kv_scales, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(page_table_1, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(active_token_counts, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(sm_scale_tensor, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(output, _torch_to_cutlass_dtype(output.dtype)),
        current_cuda_stream(),
    )
    cache_key = (
        head_tiles,
        identity_page_table,
        str(kv_nope_dtype),
        str(output.dtype),
    )
    compile_spec = KernelCompileSpec.from_key(
        "attention.mla.sparse.fp6_direct",
        1,
        cache_key,
        labels=("head_tiles", "identity_page_table", "kv_nope_dtype", "output_dtype"),
    )
    b12x_launch(
        kernel,
        compile_spec=compile_spec,
        compile_args=args,
        runtime_args=args,
    )


def _run_fp6_case(
    *,
    fmt: Fp6Format,
    kv_nope_dtype: type,
    debug_qk_bf16: bool,
) -> float:
    device = require_sm120()
    q_all, k_nope, k_rope, page_table_1, active_token_counts = _make_synthetic_mla_case(
        device,
        seed=42 if fmt == "e3m2" else 43,
    )
    kv_cache = pack_mla_kv_cache_fp6_reference(k_nope, k_rope, fmt=fmt)
    output = torch.empty(
        (1, q_all.shape[1], _MLA_V_DIM),
        device=device,
        dtype=torch.bfloat16,
    )

    import os

    if debug_qk_bf16:
        os.environ["B12X_MLA_DEBUG_QK_BF16"] = "1"
    else:
        os.environ.pop("B12X_MLA_DEBUG_QK_BF16", None)
    os.environ.pop("B12X_MLA_DEBUG_PV_BF16", None)

    _launch_sparse_mla_kernel_fp6(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        output=output,
        kv_nope_dtype=kv_nope_dtype,
    )
    torch.cuda.synchronize(device)

    expected = sparse_mla_fp6_reference(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        active_token_counts=active_token_counts,
        sm_scale=_MLA_SM_SCALE,
        v_head_dim=_MLA_V_DIM,
        fmt=fmt,
    )
    cos = _cosine(output, expected)
    assert float(output.abs().max()) > 0.0
    return cos


@pytest.mark.parametrize(
    ("fmt", "kv_nope_dtype"),
    [
        ("e3m2", cutlass.Float6E3M2FN),
        ("e2m3", cutlass.Float6E2M3FN),
    ],
)
def test_sparse_pv_mxfp6(fmt: str, kv_nope_dtype: type) -> None:
    """PV-only FP6 path with BF16 QK (isolates Phase A PV helper)."""
    cos = _run_fp6_case(fmt=fmt, kv_nope_dtype=kv_nope_dtype, debug_qk_bf16=True)
    assert cos > 0.95, f"fmt={fmt} sparse_pv cos={cos:.4f}"


@pytest.mark.parametrize(
    ("fmt", "kv_nope_dtype"),
    [
        ("e3m2", cutlass.Float6E3M2FN),
        ("e2m3", cutlass.Float6E2M3FN),
    ],
)
def test_sparse_onepass_mxfp6(fmt: str, kv_nope_dtype: type) -> None:
    """End-to-end sparse MLA with FP6 QK and PV."""
    cos = _run_fp6_case(fmt=fmt, kv_nope_dtype=kv_nope_dtype, debug_qk_bf16=False)
    assert cos > 0.95, f"fmt={fmt} sparse_onepass cos={cos:.4f}"
