from __future__ import annotations

import pytest
import torch

from b12x.cute.fp4 import swizzle_block_scale
from b12x.cute.fp6 import (
    FLOAT6_E3M2_MAX,
    SF_VEC_SIZE_FP6,
    _encode_fp6_nearest,
    _ue8m0_scale_from_block_max,
    dequant_mxfp6_torch,
    quantize_grouped_mxfp6_torch,
)
from b12x.cute.utils import mxfp6_packed_k_bytes
from b12x.gemm.dense import dense_gemm
from b12x.quantization import allocate_bf16_to_fp6_tma_outputs, compile_bf16_to_fp6_tma
from tests.helpers import require_sm120

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FP6 GPU tests"
)


def _bf16_global_scale(amax: float) -> torch.Tensor:
    return torch.tensor(
        [float(torch.finfo(torch.float8_e4m3fn).max) * 28.0 / max(amax, 1e-6)],
        dtype=torch.float32,
        device="cuda",
    )


def _quantize_bf16_matrix(bf16: torch.Tensor, fmt: str = "e3m2") -> tuple[torch.Tensor, torch.Tensor]:
    m, k = bf16.shape
    row_counts = torch.tensor([m], dtype=torch.int32, device=bf16.device)
    gs = _bf16_global_scale(float(bf16.abs().max().item()))
    packed, scale_view = quantize_grouped_mxfp6_torch(
        bf16.unsqueeze(0),
        row_counts,
        gs,
        fmt=fmt,  # type: ignore[arg-type]
    )
    return packed[:, :, 0].contiguous(), scale_view


@pytest.mark.parametrize("fmt", ["e3m2", "e2m3"])
def test_bf16_to_fp6_tma_matches_torch_reference(fmt: str) -> None:
    require_sm120()
    m = k = 128
    torch.manual_seed(7)
    bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.25
    ref_packed, ref_scales = _quantize_bf16_matrix(bf16, fmt=fmt)
    out = allocate_bf16_to_fp6_tma_outputs(m, k, device=bf16.device)
    gs = _bf16_global_scale(float(bf16.abs().max().item()))
    launch = compile_bf16_to_fp6_tma(m, k)
    launch(bf16, gs, out.packed_a_flat, out.scale_flat)
    torch.cuda.synchronize()

    assert out.packed_a_storage.shape == (1, m, mxfp6_packed_k_bytes(k))
    ref_deq = dequant_mxfp6_torch(
        ref_packed,
        ref_scales,
        num_fp6=k,
        fmt=fmt,  # type: ignore[arg-type]
        global_scale=gs,
    )
    ker_deq = dequant_mxfp6_torch(
        out.packed_a_storage.squeeze(0),
        out.scale_storage.view(1, -1),
        num_fp6=k,
        fmt=fmt,  # type: ignore[arg-type]
        global_scale=gs,
    )
    torch.testing.assert_close(ker_deq, ref_deq, rtol=0.0, atol=0.15)


def test_dense_gemm_mxfp6_nonzero_vs_torch_reference() -> None:
    require_sm120()
    from b12x.integration.tp_moe import clear_tp_moe_caches

    clear_tp_moe_caches()
    m = n = k = 128
    torch.manual_seed(11)
    a_bf = torch.randn(1, m, k, device="cuda", dtype=torch.bfloat16) * 0.2
    b_bf = torch.randn(1, n, k, device="cuda", dtype=torch.bfloat16) * 0.2
    a_packed, a_sf = _quantize_bf16_matrix(a_bf.squeeze(0), fmt="e3m2")
    b_packed, b_sf = _quantize_bf16_matrix(b_bf.squeeze(0), fmt="e3m2")
    a_gs = _bf16_global_scale(float(a_bf.abs().max().item()))
    b_gs = _bf16_global_scale(float(b_bf.abs().max().item()))

    a_g = a_packed.unsqueeze(-1)
    b_g = b_packed.unsqueeze(-1)
    alpha = (1.0 / (a_gs[0] * b_gs[0])).view(1)
    out = torch.empty((m, n, 1), device="cuda", dtype=torch.bfloat16)
    dense_gemm(
        (a_g, a_sf),
        (b_g, b_sf),
        alpha=alpha,
        ab_dtype="float6_e3m2fn",
        sf_dtype="float8_e8m0fnu",
        sf_vec_size=SF_VEC_SIZE_FP6,
        c_dtype="bfloat16",
        out=out,
    )
    torch.cuda.synchronize()

    a_f = dequant_mxfp6_torch(
        a_packed,
        a_sf,
        num_fp6=k,
        fmt="e3m2",
        global_scale=a_gs,
    )
    b_f = dequant_mxfp6_torch(
        b_packed,
        b_sf,
        num_fp6=k,
        fmt="e3m2",
        global_scale=b_gs,
    )
    ref = (a_f @ b_f.T) * alpha.item()
    cos = torch.nn.functional.cosine_similarity(
        out[:, :, 0].reshape(-1).float(),
        ref.reshape(-1).float(),
        dim=0,
    ).item()
    assert cos > 0.95
    assert float(out.abs().max()) > 0.0


def _swizzled_scales_for_quantized(
    values_bf16: torch.Tensor,
    global_scales: torch.Tensor,
    *,
    fmt: str,
) -> torch.Tensor:
    """Build flat swizzled UE8M0 scale bytes matching MoE weight layout."""
    fmt_max = FLOAT6_E3M2_MAX if fmt == "e3m2" else 7.5
    groups, rows, cols = values_bf16.shape
    scales = torch.zeros(
        (groups, rows, cols // SF_VEC_SIZE_FP6),
        dtype=torch.uint8,
        device=values_bf16.device,
    )
    for group_idx in range(groups):
        valid_rows = rows
        x = values_bf16[group_idx, :valid_rows].float()
        gs = float(global_scales[group_idx].item())
        sliced = x.view(valid_rows, cols // SF_VEC_SIZE_FP6, SF_VEC_SIZE_FP6)
        block_max = sliced.abs().amax(dim=-1)
        scales[group_idx, :valid_rows] = _ue8m0_scale_from_block_max(
            block_max * gs, fmt_max
        )
    return swizzle_block_scale(scales.to(torch.float8_e8m0fnu)).view(torch.uint8)


def _synthetic_mxfp6_moe_weights(
    *,
    experts: int,
    k: int,
    n: int,
    device: torch.device,
    weight_fmt: str = "e2m3",
):
    w1_n = 2 * n
    w1_bf = torch.randn(experts, w1_n, k, device=device, dtype=torch.bfloat16) * 0.15
    w2_bf = torch.randn(experts, k, n, device=device, dtype=torch.bfloat16) * 0.15
    row_full = torch.full((experts,), w1_n, dtype=torch.int32, device=device)
    row_w2 = torch.full((experts,), k, dtype=torch.int32, device=device)
    gs1 = torch.ones(experts, device=device, dtype=torch.float32)
    gs2 = torch.ones(experts, device=device, dtype=torch.float32)
    w1_packed, _ = quantize_grouped_mxfp6_torch(
        w1_bf, row_full, gs1, fmt=weight_fmt  # type: ignore[arg-type]
    )
    w2_packed, _ = quantize_grouped_mxfp6_torch(
        w2_bf, row_w2, gs2, fmt=weight_fmt  # type: ignore[arg-type]
    )
    w1_fp6 = w1_packed.permute(2, 0, 1).contiguous()
    w2_fp6 = w2_packed.permute(2, 0, 1).contiguous()
    w1_bs = _swizzled_scales_for_quantized(w1_bf, gs1, fmt=weight_fmt)
    w2_bs = _swizzled_scales_for_quantized(w2_bf, gs2, fmt=weight_fmt)
    return w1_fp6, w1_bs, w2_fp6, w2_bs


def test_moe_fp6_synthetic_smoke() -> None:
    require_sm120()
    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace,
        b12x_moe_fp6,
        clear_tp_moe_caches,
    )

    clear_tp_moe_caches()
    device = torch.device("cuda")
    m, k, n, experts, topk = 2, 128, 128, 4, 2
    torch.manual_seed(3)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
    topk_ids = torch.randint(0, experts, (m, topk), device=device, dtype=torch.int32)
    topk_weights = torch.softmax(
        torch.randn(m, topk, device=device), dim=-1
    ).to(torch.float32)
    w1_fp6, w1_sf, w2_fp6, w2_sf = _synthetic_mxfp6_moe_weights(
        experts=experts, k=k, n=n, device=device
    )
    a1_gscale = torch.ones(1, device=device, dtype=torch.float32)
    a2_gscale = torch.ones(1, device=device, dtype=torch.float32)
    w1_alphas = torch.ones(experts, device=device, dtype=torch.float32)
    w2_alphas = torch.ones(experts, device=device, dtype=torch.float32)

    workspace = allocate_tp_moe_workspace(
        x,
        a1_gscale,
        w1_fp6,
        a2_gscale,
        w2_fp6,
        topk_ids,
        quant_mode="w6a6",
        input_scales_static=True,
    )
    out = b12x_moe_fp6(
        x,
        a1_gscale,
        w1_fp6,
        w1_sf,
        w1_alphas,
        a2_gscale,
        w2_fp6,
        w2_sf,
        w2_alphas,
        topk_weights,
        topk_ids,
        workspace=workspace,
        input_scales_static=True,
        source_format="mxfp6_default",
    )
    torch.cuda.synchronize()
    assert out.shape == (m, k)
    assert float(out.float().norm()) > 1e-4
