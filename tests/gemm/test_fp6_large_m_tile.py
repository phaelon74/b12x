"""Bit-equality and regime isolation for the MX-FP6 wide-N prefill tile.

Phase B item 4: the wide-N m>16 default tile moved from the (128,128) pin to
the sweep-measured (128,64) winner (``SPARKINFER_FP6_LARGE_M_TILE`` overrides
for A/B). Tiles only change the CTA work decomposition — the per-output-element
accumulation order is identical — so outputs must be BIT-IDENTICAL across
tiles. The m<=16 decode regime takes the wave-cliff heuristic ((16,64), or
(32,128) when ceil(N/64) leaves a tiny tail wave); `SPARKINFER_FP6_DECODE_TILE`
forces a fixed tile for A/B.
"""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@cuda_required
@pytest.mark.parametrize("m", [32, 128, 256])
def test_fp6_large_m_tile_bit_exact_vs_old_pin(m, monkeypatch):
    import sparkinfer._lib.dense_gemm as dg
    import sparkinfer.quantization.mxfp6.fp6_dense_weights as fdw

    torch.manual_seed(0)
    n, k = 6144, 512
    fp6w = fdw.quantize_dense_weight_to_fp6(
        (torch.randn(n, k, device="cuda") * 0.1).to(torch.bfloat16)
    )
    x = (torch.randn(m, k, device="cuda") * 0.1).to(torch.bfloat16)
    args = (fp6w.scale_storage, fp6w.global_scale, fp6w.fmt, n, k)
    weight = fp6w.expanded_weight()

    monkeypatch.setattr(dg, "_SPARKINFER_FP6_LARGE_M_TILE", (128, 128))
    y_old = fdw.dense_fp6_linear_expanded(x, weight, *args)
    monkeypatch.setattr(dg, "_SPARKINFER_FP6_LARGE_M_TILE", (128, 64))
    y_new = fdw.dense_fp6_linear_expanded(x, weight, *args)

    torch.testing.assert_close(y_new, y_old, rtol=0.0, atol=0.0)


def test_fp6_tile_regime_selection():
    from sparkinfer._lib.dense_gemm import _select_default_mma_tiler_mn

    common = dict(sm_count=188, is_mxfp8=False, is_mxfp6=True)
    # Decode regime: width-64 unless ceil(N/64) leaves a tiny tail wave.
    for m in (1, 2, 8, 16):
        # qkv N=7168: 112 CTAs, single wave -> (16,64).
        assert _select_default_mma_tiler_mn(m, 7168, **common) == (16, 64)
        # gate_up N=28672: 448 CTAs, healthy 72-CTA tail -> (16,64).
        assert _select_default_mma_tiler_mn(m, 28672, **common) == (16, 64)
        # o/down N=12288 was exempted to (32,128) to dodge a 4-CTA tail wave on
        # 188 SMs. Retired Jul 26 2026: that tail is an occupancy-1 artifact,
        # and _dense_gemm_target_occupancy now returns 2 for exactly these
        # shapes, so width-64 wins outright. Decode is width-64 everywhere.
        assert _select_default_mma_tiler_mn(m, 12288, **common) == (16, 64)
    # Wide-N prefill regime takes the sweep winner for every m > 16.
    for m in (17, 32, 512, 8192):
        assert _select_default_mma_tiler_mn(m, 7168, **common) == (128, 64)
    # Narrow-N keeps the unmeasured coarse default.
    assert _select_default_mma_tiler_mn(8192, 1024, **common) == (128, 128)
    # A declared expected_m regime hint owns the decision.
    assert _select_default_mma_tiler_mn(
        1, 7168, expected_m=8192, **common
    ) == (128, 64)
    assert _select_default_mma_tiler_mn(
        8192, 7168, expected_m=8, **common
    ) == (16, 64)


@cuda_required
@pytest.mark.parametrize("m", [1, 2, 8, 16])
def test_fp6_decode_tile_bit_exact_vs_old_pin(m, monkeypatch):
    """Heuristic decode tiles are bit-identical to the old (16,128) pin.

    Runs the PACKED weight stream (the production decode path) through both
    the forced old tile and the heuristic default.
    """
    import sparkinfer._lib.dense_gemm as dg
    import sparkinfer.quantization.mxfp6.fp6_dense_weights as fdw

    torch.manual_seed(1)
    n, k = 6144, 512
    fp6w = fdw.quantize_dense_weight_to_fp6(
        (torch.randn(n, k, device="cuda") * 0.1).to(torch.bfloat16)
    )
    x = (torch.randn(m, k, device="cuda") * 0.1).to(torch.bfloat16)
    args = (fp6w.scale_storage, fp6w.global_scale, fp6w.fmt, n, k)

    monkeypatch.setattr(dg, "_SPARKINFER_FP6_DECODE_TILE", (16, 128))
    y_old = fdw.dense_fp6_linear_expanded(x, fp6w.packed, *args)
    monkeypatch.setattr(dg, "_SPARKINFER_FP6_DECODE_TILE", None)
    y_new = fdw.dense_fp6_linear_expanded(x, fp6w.packed, *args)

    torch.testing.assert_close(y_new, y_old, rtol=0.0, atol=0.0)


def test_parse_tile_env_guard(monkeypatch):
    from sparkinfer._lib.dense_gemm import _parse_tile_env

    monkeypatch.setenv("SPARKINFER_FP6_LARGE_M_TILE", "64x128")
    assert _parse_tile_env("SPARKINFER_FP6_LARGE_M_TILE", (128, 64)) == (64, 128)
    monkeypatch.setenv("SPARKINFER_FP6_LARGE_M_TILE", "bogus")
    with pytest.raises(ValueError, match="128x64"):
        _parse_tile_env("SPARKINFER_FP6_LARGE_M_TILE", (128, 64))
    monkeypatch.delenv("SPARKINFER_FP6_LARGE_M_TILE")
    assert _parse_tile_env("SPARKINFER_FP6_LARGE_M_TILE", (128, 64)) == (128, 64)
