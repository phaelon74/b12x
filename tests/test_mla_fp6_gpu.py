"""GPU tests for sparse MLA MX-FP6 QK/PV (Phase A: direct SparseMLAKernel construction)."""

from __future__ import annotations

import math
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Float32, Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x.attention.mla.kernel import (
    SparseMLAKernel,
    _MLA_HEADS_PER_TILE,
    _MLA_NOPE_GROUP_KV_VECS,
    _MLA_NUM_MMA_KV,
    _MLA_TOKEN_TILE,
    _MLA_VO_NUM_MMA_D,
    _extract_packed_kv_runtime_views,
    _stage_kv_u32_block,
    _store_output_group,
    _to_kernel_tensor,
    _torch_to_cutlass_dtype,
    _view_last_dim_as_u32,
    _zero_output_frag,
    clear_sparse_mla_kernel_cache,
    get_sparse_mla_shared_storage_cls,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
)
from b12x.attention.mla.reference import (
    pack_mla_kv_cache_fp6_reference,
    sparse_mla_fp6_reference,
    unpack_mla_kv_cache_fp6_reference,
)
from b12x.attention.mxfp6_mma import _literal_pv_mma_into_ofrag_mxfp6_scaled_mla
from b12x.cute.compiler import KernelCompileSpec, clear_compile_cache, launch as b12x_launch
from b12x.cute.fp6 import (
    Fp6Format,
    _decode_fp6_e2m3,
    _decode_fp6_e3m2,
    _encode_fp6_nearest,
)
from b12x.cute.utils import current_cuda_stream

from .helpers import require_sm120

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for MLA FP6 GPU tests"
)

_MLA_HEAD_DIM = 576
_MLA_V_DIM = 512
_MLA_GROUP_SIZE = 128
_MLA_SM_SCALE = (_MLA_V_DIM + 64) ** -0.5


def _to_cute_tensor(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor


@cute.jit
def _fill_probe_p_frag(
    mP: cute.Tensor,
    p_frag: cute.Tensor,
    lane: Int32,
):
    lane_group = lane // Int32(4)
    lane_pair_base = Int32(2) * (lane % Int32(4))
    row0 = lane_group
    row1 = lane_group + Int32(8)
    for mma_kv in cutlass.range_constexpr(_MLA_NUM_MMA_KV):
        k_base = Int32(mma_kv * 16) + lane_pair_base
        p_frag[0, mma_kv, 0] = pack_f32x2_to_bfloat2(
            Float32(mP[row0, k_base + Int32(0)]),
            Float32(mP[row0, k_base + Int32(1)]),
        )
        p_frag[0, mma_kv, 1] = pack_f32x2_to_bfloat2(
            Float32(mP[row1, k_base + Int32(0)]),
            Float32(mP[row1, k_base + Int32(1)]),
        )
        p_frag[0, mma_kv, 2] = pack_f32x2_to_bfloat2(
            Float32(mP[row0, k_base + Int32(8)]),
            Float32(mP[row0, k_base + Int32(9)]),
        )
        p_frag[0, mma_kv, 3] = pack_f32x2_to_bfloat2(
            Float32(mP[row1, k_base + Int32(8)]),
            Float32(mP[row1, k_base + Int32(9)]),
        )


class MlaMxfp6PvProbeKernel:
    """Isolate ``_literal_pv_mma_into_ofrag_mxfp6_scaled_mla`` (one nope group, 64 tokens)."""

    num_threads = 32

    def __init__(self, kv_nope_dtype: type):
        self.kv_nope_dtype = kv_nope_dtype

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mVWords: cute.Tensor,
        mScale: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mP, mVWords, mScale, mOut).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mVWords: cute.Tensor,
        mScale: cute.Tensor,
        mOut: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(get_sparse_mla_shared_storage_cls())
        sTokenIdx = storage.token_idx.get_tensor(cute.make_layout((_MLA_TOKEN_TILE,), stride=(1,)))
        sScale = storage.token_scale_a.get_tensor(cute.make_layout((_MLA_TOKEN_TILE,), stride=(1,)))
        kv_base_addr = shared_ptr_to_u32(storage.kv_stage_a.data_ptr())

        token_local = lane
        while token_local < Int32(_MLA_TOKEN_TILE):
            sTokenIdx[token_local] = token_local
            sScale[token_local] = Float32(mScale[token_local])
            token_local += Int32(self.num_threads)
        cute.arch.sync_threads()

        _stage_kv_u32_block(
            mVWords,
            sTokenIdx,
            Int32(0),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
            Int32(_MLA_NOPE_GROUP_KV_VECS),
            kv_base_addr,
            Int32(mVWords.shape[0]),
            lane,
        )
        cute.arch.sync_threads()

        p_layout = cute.make_layout((1, _MLA_NUM_MMA_KV, 4), stride=(8, 4, 1))
        o_layout = cute.make_layout((1, _MLA_VO_NUM_MMA_D, 8), stride=(_MLA_VO_NUM_MMA_D * 8, 8, 1))
        md_layout = cute.make_layout((1, 2), stride=(2, 1))

        p_frag = cute.make_rmem_tensor(p_layout, Uint32)
        _fill_probe_p_frag(mP, p_frag, lane)

        o_frag = cute.make_rmem_tensor(o_layout, Float32)
        _zero_output_frag(o_frag)

        _literal_pv_mma_into_ofrag_mxfp6_scaled_mla(
            o_frag,
            p_frag,
            kv_base_addr,
            sScale,
            Int32(0),
            Float32(1.0),
            lane,
            self.kv_nope_dtype,
        )

        d_frag = cute.make_rmem_tensor(md_layout, Float32)
        d_frag[0, 0] = Float32(1.0)
        d_frag[0, 1] = Float32(1.0)
        _store_output_group(
            mOut,
            o_frag,
            d_frag,
            Int32(0),
            Int32(0),
            Int32(0),
            lane,
        )


def _run_mxfp6_pv_probe_arrays(
    *,
    fmt: Fp6Format,
    kv_nope_dtype: type,
    seed: int,
    p_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the isolated FP6 PV probe; return (out[16,128], V_true[64,128], P[16,64], scales[64])."""
    device = require_sm120()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    if p_override is not None:
        p = p_override.to(device=device, dtype=torch.bfloat16)
    else:
        # Model real attention weights: a softmax over tokens (non-negative, sums
        # to 1, peaked). Signed Gaussian P is a pathological worst case for FP6
        # cosine (sign cancellation + many near-zero values that quantize poorly).
        logits = torch.randn(
            (_MLA_HEADS_PER_TILE, _MLA_TOKEN_TILE),
            generator=gen,
            dtype=torch.float32,
        )
        p = (
            torch.softmax(logits, dim=1)
            .to(device=device, dtype=torch.bfloat16)
        )
    k_nope = (
        torch.randn(
            (_MLA_TOKEN_TILE, 1, _MLA_V_DIM),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device, dtype=torch.bfloat16)
        / 4
    )
    k_rope = (
        torch.randn(
            (_MLA_TOKEN_TILE, 1, 64),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device, dtype=torch.bfloat16)
        / 4
    )
    kv_packed = pack_mla_kv_cache_fp6_reference(k_nope, k_rope, fmt=fmt)
    packed_2d = kv_packed[:, 0, :].contiguous().view(torch.uint8)
    v_words = (
        packed_2d[:, :_MLA_GROUP_SIZE]
        .contiguous()
        .view(torch.uint32)
        .reshape(_MLA_TOKEN_TILE, _MLA_NOPE_GROUP_KV_VECS * 4)
    )
    # Per-token group-0 dequant scale (NSA layout: 4-byte float32 per group after nope codes).
    scales = (
        packed_2d[:, _MLA_V_DIM : _MLA_V_DIM + 4]
        .contiguous()
        .view(torch.float32)
        .reshape(_MLA_TOKEN_TILE)
        .to(device)
    )
    # V_true = fp6_decode(codes) * dequant_scale for group 0.
    v_dequant = (
        unpack_mla_kv_cache_fp6_reference(kv_packed, fmt=fmt)[:_MLA_TOKEN_TILE, 0, :_MLA_GROUP_SIZE]
        .to(torch.float32)
    )

    out = torch.empty((1, _MLA_HEADS_PER_TILE, _MLA_GROUP_SIZE), device=device, dtype=torch.float32)
    kernel = MlaMxfp6PvProbeKernel(kv_nope_dtype=kv_nope_dtype)
    stream = cuda.CUstream(torch.cuda.current_stream(device=device).cuda_stream)
    compile_args = (
        _to_cute_tensor(p, cutlass.BFloat16),
        _to_cute_tensor(v_words, cutlass.Uint32),
        _to_cute_tensor(scales, cutlass.Float32),
        _to_cute_tensor(out, cutlass.Float32),
        stream,
    )
    compiled = cute.compile(kernel, *compile_args)
    compiled(*compile_args)
    torch.cuda.synchronize(device)

    return out.squeeze(0), v_dequant, p.to(torch.float32), scales


def _quantize_p_mxfp6_like_kernel(
    p: torch.Tensor,
    scales: torch.Tensor,
    fmt: Fp6Format,
) -> torch.Tensor:
    """Apply the kernel's MX-FP6 quantization of ``P*scale`` to the reference.

    The kernel folds the per-token V dequant scale into P, then quantizes to FP6
    using one dynamic UE8M0 scale per MMA A K=32 block. The scale is reduced to a
    single warp-wide max, so it is uniform across all 16 heads and the 32 tokens
    of each ``mma_pair`` (tokens 0-31, then 32-63). The UE8M0 exponent is shifted
    down by the format's max exponent so the block max lands near the format max.
    Modelling this lets the probe verify the PV *math* (cos ~ 1) rather than the
    FP6 quant noise.
    """
    decode = _decode_fp6_e3m2 if fmt == "e3m2" else _decode_fp6_e2m3
    emax = 4 if fmt == "e3m2" else 2
    # Match the kernel input precision (P arrives as BF16, scale as FP32).
    p_f = p.to(torch.bfloat16).to(torch.float32)
    s_f = scales.to(torch.float32)
    q = torch.zeros_like(p_f)
    for pair in range(2):
        toks = list(range(pair * 32, pair * 32 + 32))
        block = p_f[:, toks] * s_f[toks].unsqueeze(0)  # [16, 32]
        amax = float(block.abs().max())
        if amax <= 0.0:
            continue
        ue8m0 = math.ceil(math.log2(amax)) + 127 - emax
        ue8m0 = max(0, min(255, ue8m0))
        sfa_val = 2.0 ** (ue8m0 - 127)
        inv_sfa = 1.0 / sfa_val
        for h in range(_MLA_HEADS_PER_TILE):
            for t in toks:
                v = float(p_f[h, t]) * float(s_f[t])
                code = _encode_fp6_nearest(v * inv_sfa, fmt)
                q[h, t] = decode(code) * sfa_val
    return q


def _run_mxfp6_pv_probe(
    *,
    fmt: Fp6Format,
    kv_nope_dtype: type,
    seed: int,
) -> float:
    actual, v_dequant, p, scales = _run_mxfp6_pv_probe_arrays(
        fmt=fmt, kv_nope_dtype=kv_nope_dtype, seed=seed
    )
    # Kernel multiplies the FP6-quantized (P*scale) by the raw decoded V codes:
    #   O[h,d] = sum_t Q(P[h,t]*scale[t]) * (V_true[t,d] / scale[t]).
    # Model the same MX-FP6 quantization of P so the probe checks the PV math,
    # not the (large, expected) FP6 quantization floor of representing P.
    q_scaled_p = _quantize_p_mxfp6_like_kernel(p, scales, fmt)
    v_codes = v_dequant / scales.to(torch.float32).unsqueeze(1)
    ref = torch.matmul(q_scaled_p, v_codes)
    return _cosine(actual, ref)


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
    identity_page_table: bool = False,
    debug_qk_bf16: bool = False,
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
        "1" if debug_qk_bf16 else "0",
        os.environ.get("B12X_MLA_DEBUG_PV_BF16", "0"),
    )
    compile_spec = KernelCompileSpec.from_key(
        "attention.mla.sparse.fp6_direct",
        4,
        cache_key,
        labels=(
            "head_tiles",
            "identity_page_table",
            "kv_nope_dtype",
            "output_dtype",
            "debug_qk_bf16",
            "debug_pv_bf16",
        ),
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
    kv_cache = pack_mla_kv_cache_fp6_reference(k_nope, k_rope, fmt=fmt).view(
        torch.float8_e4m3fn
    )
    output = torch.empty(
        (1, q_all.shape[1], _MLA_V_DIM),
        device=device,
        dtype=torch.bfloat16,
    )

    if debug_qk_bf16:
        os.environ["B12X_MLA_DEBUG_QK_BF16"] = "1"
    else:
        os.environ.pop("B12X_MLA_DEBUG_QK_BF16", None)
    os.environ.pop("B12X_MLA_DEBUG_PV_BF16", None)

    try:
        _launch_sparse_mla_kernel_fp6(
            q_all=q_all,
            kv_cache=kv_cache,
            page_table_1=page_table_1,
            active_token_counts=active_token_counts,
            output=output,
            kv_nope_dtype=kv_nope_dtype,
            debug_qk_bf16=debug_qk_bf16,
        )
        torch.cuda.synchronize(device)
    except Exception:
        torch.cuda.synchronize()
        raise

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
    """Isolated MLA MX-FP6 PV MMA (one 128-dim nope group, 64 tokens)."""
    seed = 42 if fmt == "e3m2" else 43
    cos = _run_mxfp6_pv_probe(fmt=fmt, kv_nope_dtype=kv_nope_dtype, seed=seed)
    assert cos > 0.95, f"fmt={fmt} mxfp6_pv_probe cos={cos:.4f}"


@pytest.mark.parametrize("token0", [0, 1, 16, 17, 32])
def test_mxfp6_pv_single_token_diag(token0: int) -> None:
    """Diagnostic: one-hot P over token0 -> O[h,:] must equal V_true[token0,:] for all heads.

    Localizes the FP6 PV layout bug:
      - dims permuted but values present  -> byte_perm / V-column mapping wrong
      - wrong token's V appears           -> K-pairing / token mapping wrong
    """
    fmt = "e3m2"
    p = torch.zeros((_MLA_HEADS_PER_TILE, _MLA_TOKEN_TILE), dtype=torch.float32)
    p[:, token0] = 1.0
    out, v_dequant, _p, _scales = _run_mxfp6_pv_probe_arrays(
        fmt=fmt, kv_nope_dtype=cutlass.Float6E3M2FN, seed=42, p_override=p
    )
    expected = v_dequant[token0]  # [128]

    head0 = out[0]
    cos_head0 = _cosine(head0, expected)

    # Does the produced output match some OTHER token's V (token mismap)?
    token_cos = torch.tensor(
        [_cosine(head0, v_dequant[t]) for t in range(_MLA_TOKEN_TILE)]
    )
    best_token = int(token_cos.argmax().item())

    # Are the values present but permuted across dims? Compare sorted magnitudes.
    sorted_cos = _cosine(
        torch.sort(head0.abs())[0], torch.sort(expected.abs())[0]
    )

    msg = (
        f"\ntoken0={token0} cos(head0, V_true[token0])={cos_head0:.4f}"
        f"\n  best matching token={best_token} (cos={token_cos[best_token]:.4f})"
        f"\n  sorted-magnitude cos={sorted_cos:.4f} "
        f"(high => values correct but dims permuted)"
        f"\n  head0[:8]={head0[:8].tolist()}"
        f"\n  expect[:8]={expected[:8].tolist()}"
    )
    assert cos_head0 > 0.95, msg


def test_mxfp6_pv_all_token_sweep() -> None:
    """Sweep every one-hot token; O[h,:] must equal V_true[token,:] for all 64 tokens.

    The 5-token single-token diag can pass while multi-token sums fail if some
    K-positions (e.g. tokens 2-15, 18-31) are mismapped by the B-operand path.
    This compiles the probe once and checks all 64 tokens to localize such bugs.
    """
    fmt = "e3m2"
    device = require_sm120()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)

    p = torch.zeros((_MLA_HEADS_PER_TILE, _MLA_TOKEN_TILE), device=device, dtype=torch.bfloat16)
    k_nope = (
        torch.randn((_MLA_TOKEN_TILE, 1, _MLA_V_DIM), generator=gen, dtype=torch.float32)
        .to(device=device, dtype=torch.bfloat16)
        / 4
    )
    k_rope = (
        torch.randn((_MLA_TOKEN_TILE, 1, 64), generator=gen, dtype=torch.float32)
        .to(device=device, dtype=torch.bfloat16)
        / 4
    )
    kv_packed = pack_mla_kv_cache_fp6_reference(k_nope, k_rope, fmt=fmt)
    packed_2d = kv_packed[:, 0, :].contiguous().view(torch.uint8)
    v_words = (
        packed_2d[:, :_MLA_GROUP_SIZE]
        .contiguous()
        .view(torch.uint32)
        .reshape(_MLA_TOKEN_TILE, _MLA_NOPE_GROUP_KV_VECS * 4)
    )
    scales = (
        packed_2d[:, _MLA_V_DIM : _MLA_V_DIM + 4]
        .contiguous()
        .view(torch.float32)
        .reshape(_MLA_TOKEN_TILE)
        .to(device)
    )
    v_dequant = (
        unpack_mla_kv_cache_fp6_reference(kv_packed, fmt=fmt)[:_MLA_TOKEN_TILE, 0, :_MLA_GROUP_SIZE]
        .to(torch.float32)
    )

    out = torch.empty((1, _MLA_HEADS_PER_TILE, _MLA_GROUP_SIZE), device=device, dtype=torch.float32)
    kernel = MlaMxfp6PvProbeKernel(kv_nope_dtype=cutlass.Float6E3M2FN)
    stream = cuda.CUstream(torch.cuda.current_stream(device=device).cuda_stream)
    args = (
        _to_cute_tensor(p, cutlass.BFloat16),
        _to_cute_tensor(v_words, cutlass.Uint32),
        _to_cute_tensor(scales, cutlass.Float32),
        _to_cute_tensor(out, cutlass.Float32),
        stream,
    )
    compiled = cute.compile(kernel, *args)

    failures: list[tuple[int, float, int]] = []
    for token0 in range(_MLA_TOKEN_TILE):
        p.zero_()
        p[:, token0] = 1.0
        compiled(*args)
        torch.cuda.synchronize(device)
        head0 = out.squeeze(0)[0]
        cos = _cosine(head0, v_dequant[token0])
        if cos <= 0.95:
            token_cos = torch.tensor(
                [_cosine(head0, v_dequant[t]) for t in range(_MLA_TOKEN_TILE)]
            )
            failures.append((token0, cos, int(token_cos.argmax().item())))

    msg = "failing tokens (token0, cos, best_match): " + ", ".join(
        f"({t},{c:.3f},{b})" for t, c, b in failures
    )
    assert not failures, msg


@pytest.mark.parametrize(
    ("fmt", "kv_nope_dtype"),
    [
        ("e3m2", cutlass.Float6E3M2FN),
        ("e2m3", cutlass.Float6E2M3FN),
    ],
)
def test_mxfp6_pv_uniform_scale_probe(fmt: str, kv_nope_dtype: type) -> None:
    """Constant P -> every lane computes the SAME sfa. If the dynamic per-row
    scale-operand selection is the bug, a uniform sfa should restore norm ratio ~1.
    A clean per-format power-of-two norm ratio instead points at the scale value.
    """
    p = torch.full((_MLA_HEADS_PER_TILE, _MLA_TOKEN_TILE), 0.1, dtype=torch.float32)
    actual, v_dequant, p_f, scales = _run_mxfp6_pv_probe_arrays(
        fmt=fmt, kv_nope_dtype=kv_nope_dtype, seed=42, p_override=p
    )
    ref_full = torch.matmul(p_f, v_dequant)
    ratio = float(actual[0].norm()) / float(ref_full[0].norm())
    print(f"\n{fmt}: uniform-P cos = {_cosine(actual, ref_full):.4f}  norm_ratio = {ratio:.4f}")


def test_mxfp6_pv_softmax_breakdown() -> None:
    """Print cosine breakdown for softmax P to localize the multi-token error."""
    fmt = "e3m2"
    actual, v_dequant, p, scales = _run_mxfp6_pv_probe_arrays(
        fmt=fmt, kv_nope_dtype=cutlass.Float6E3M2FN, seed=42
    )
    ref_full = torch.matmul(p, v_dequant)
    q_scaled_p = _quantize_p_mxfp6_like_kernel(p, scales, fmt)
    v_codes = v_dequant / scales.to(torch.float32).unsqueeze(1)
    ref_quant = torch.matmul(q_scaled_p, v_codes)

    print(f"\ncos(actual, ref_full)  = {_cosine(actual, ref_full):.4f}")
    print(f"cos(actual, ref_quant) = {_cosine(actual, ref_quant):.4f}")
    print(f"cos(ref_quant, ref_full) = {_cosine(ref_quant, ref_full):.4f}")
    per_head = [_cosine(actual[h], ref_full[h]) for h in range(_MLA_HEADS_PER_TILE)]
    print("per-head cos(actual,ref_full):")
    for h in range(_MLA_HEADS_PER_TILE):
        print(f"  head {h:2d}: {per_head[h]:.4f}  (row{'0' if h < 8 else '1'})")
    # Also: norm ratio per head (detects scale errors).
    print("per-head ||actual||/||ref_full||:")
    for h in range(_MLA_HEADS_PER_TILE):
        na = float(actual[h].norm())
        nr = float(ref_full[h].norm())
        print(f"  head {h:2d}: {na / nr if nr > 0 else 0.0:.4f}")


def test_mxfp6_pv_per_head_distinct() -> None:
    """Distinct one-hot token per head: head h -> token h, so O[h,:] == V_true[h,:].

    All prior diagnostics used head-uniform P (p[:,t]=1 for every head), which
    masks cross-head contamination. The fragment packs heads {h, h+8} into one
    lane sharing a single dynamic sfa; a row-mixing or shared-scale bug only
    surfaces with per-head-distinct P.
    """
    fmt = "e3m2"
    out, v_dequant, _p, _scales = _run_mxfp6_pv_probe_arrays(
        fmt=fmt,
        kv_nope_dtype=cutlass.Float6E3M2FN,
        seed=42,
        p_override=torch.eye(_MLA_HEADS_PER_TILE, _MLA_TOKEN_TILE, dtype=torch.float32),
    )
    cos_per_head = [
        (h, _cosine(out[h], v_dequant[h]), int(
            torch.tensor([_cosine(out[h], v_dequant[t]) for t in range(_MLA_TOKEN_TILE)]).argmax().item()
        ))
        for h in range(_MLA_HEADS_PER_TILE)
    ]
    failures = [(h, c, b) for h, c, b in cos_per_head if c <= 0.95]
    msg = "failing heads (head, cos, best_token): " + ", ".join(
        f"({h},{c:.3f},{b})" for h, c, b in failures
    )
    assert not failures, msg


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
