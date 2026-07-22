"""MX-FP6 (W6A6) BS1 decode micro-kernel for small-batch routed MoE on SM120.

Decode-focused analogue of the NVFP4 ``MoEMicroKernelBackend``: a no-tensor-core,
software-GEMV fused MoE for small token counts (``m in {1,2,4,8}``). One CTA per
token runs FC1 -> SiLU(gate)*up -> in-kernel FP6 re-quant -> FC2 with per-expert
router-weighted accumulation, decoding FP6 codes through a shared 64-entry LUT and
reading the production packed weight ``(E, rows, 3*cols/4)`` + swizzled UE8M0 scale
storage directly (no host re-layout). Gated behind ``B12X_ENABLE_FP6_MICRO``.

The compute path is validated bit-faithfully against ``b12x_moe_fp6`` (the static
fused FP6 path) in ``tests/test_fp6_micro.py``.
"""
from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x.cute.fp4 import cvt_e8m0_to_f32, fabs_f32, fmax_f32
from b12x.cute.fp6 import (
    build_fp6_decode_lut,
    fp6_decode_code,
    fp6_unpack4_codes,
    mxfp6_swizzled_scale_offset,
)
from b12x.moe.fused.mxfp6_moe import moe_mxfp6_quantize_input_block_containers

# Threads per CTA. Rows of each phase are processed in BLK-strides so the kernel
# supports FC1 (2N) / FC2 (K) dimensions larger than one thread block.
_BLK = 256


def _fmt_pair(source_format: str) -> tuple[str, str]:
    """(activation_fmt, weight_fmt) for a MX-FP6 source format."""
    sf = source_format.lower()
    if sf in ("mxfp6_e3m2", "e3m2"):
        return "e3m2", "e3m2"
    if sf in ("mxfp6_e2m3", "e2m3"):
        return "e2m3", "e2m3"
    if sf == "mxfp6_w6a8":
        # The micro kernel's activation decode LUT is FP6-only; b12x_moe_fp6
        # routes W6A8 to the static fused path before reaching here.
        raise ValueError("mxfp6_w6a8 is not supported by the FP6 micro kernel")
    # mxfp6_default / mxfp6_mixed: E3M2 activations, E2M3 weights.
    return "e3m2", "e2m3"


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _align_up(v: int, a: int) -> int:
    return ((int(v) + int(a) - 1) // int(a)) * int(a)


class MoEMicroFp6Kernel:
    """Compiled per (n, k, topk, formats); see module docstring for the algorithm."""

    def __init__(self, n: int, k: int, topk: int, act_fmt: str, weight_fmt: str):
        self.n = int(n)
        self.two_n = 2 * int(n)
        self.k = int(k)
        self.topk = int(topk)
        self.act_fmt = str(act_fmt)
        self.weight_fmt = str(weight_fmt)
        self.k_blocks = int(k) // 32
        self.n_blocks = int(n) // 32
        self.cpd4_w1 = _align_up(self.k_blocks, 4) // 4
        self.cpd4_w2 = _align_up(self.n_blocks, 4) // 4
        # Compile-time strided-loop trip counts (rows processed in _BLK strides).
        self.it_kb = _ceil_div(self.k_blocks, _BLK)
        self.it_n = _ceil_div(self.n, _BLK)
        self.it_nb = _ceil_div(self.n_blocks, _BLK)
        self.it_k = _ceil_div(self.k, _BLK)

    @property
    def __cache_key__(self):
        return (
            self.n, self.k, self.topk, self.act_fmt, self.weight_fmt,
            self.cpd4_w1, self.cpd4_w2,
        )

    @cute.jit
    def __call__(
        self,
        mLutW: cute.Tensor, mLutA: cute.Tensor, mX: cute.Tensor,
        mW1p: cute.Tensor, mW1s: cute.Tensor, mW2p: cute.Tensor, mW2s: cute.Tensor,
        mIds: cute.Tensor, mTw: cute.Tensor, mGs1: cute.Tensor, mGs2: cute.Tensor,
        mOut: cute.Tensor, m: Int32, stream: cuda.CUstream,
    ):
        self.kernel(
            mLutW, mLutA, mX, mW1p, mW1s, mW2p, mW2s, mIds, mTw, mGs1, mGs2, mOut
        ).launch(grid=(m, 1, 1), block=[_BLK, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mLutW: cute.Tensor, mLutA: cute.Tensor, mX: cute.Tensor,
        mW1p: cute.Tensor, mW1s: cute.Tensor, mW2p: cute.Tensor, mW2s: cute.Tensor,
        mIds: cute.Tensor, mTw: cute.Tensor, mGs1: cute.Tensor, mGs2: cute.Tensor,
        mOut: cute.Tensor,
    ):
        import cutlass.utils as cutlass_utils

        t = cute.arch.block_idx()[0]
        r = cute.arch.thread_idx()[0]
        smem = cutlass_utils.SmemAllocator()
        # Smem footprint: sa_dq[K] + sinter[N] + sout[K] = (2K + N) f32. FC1+gate
        # are fused per-thread (no [2N] buffer) and the intermediate is requantized
        # in place, to keep the footprint small enough for opt-in smem.
        sa_dq = smem.allocate_tensor(
            element_type=Float32, layout=cute.make_layout(self.k), byte_alignment=128
        )
        sinter = smem.allocate_tensor(
            element_type=Float32, layout=cute.make_layout(self.n), byte_alignment=128
        )
        sout = smem.allocate_tensor(
            element_type=Float32, layout=cute.make_layout(self.k), byte_alignment=128
        )
        gs1 = Float32(mGs1[0])
        gs2 = Float32(mGs2[0])

        # ---- phase 0: quantize activation x[t,:] -> sa_dq (FP6, per 32-block) ----
        for it in cutlass.range_constexpr(self.it_kb):
            b = r + it * _BLK
            if b < self.k_blocks:
                vals = cute.make_rmem_tensor((32,), Float32)
                bmax = Float32(0.0)
                for j in cutlass.range_constexpr(32):
                    v = Float32(mX[t, b * 32 + j])
                    vals[j] = v
                    bmax = fmax_f32(bmax, fabs_f32(v))
                containers, sb = moe_mxfp6_quantize_input_block_containers(
                    vals, bmax, gs1, self.act_fmt
                )
                bsc = cvt_e8m0_to_f32(Uint32(sb))
                for j in cutlass.range_constexpr(32):
                    sa_dq[b * 32 + j] = fp6_decode_code(mLutA, Uint32(containers[j])) * bsc / gs1

        for it in cutlass.range_constexpr(self.it_k):
            row = r + it * _BLK
            if row < self.k:
                sout[row] = Float32(0.0)
        cute.arch.sync_threads()

        for slot in cutlass.range_constexpr(self.topk):
            e = Int32(mIds[t, slot])
            # ---- FC1 + gate fused: inter[row] = silu(gate)*up, one thread/row ----
            # up   = W1[e, row, :]   (rows [0, N))
            # gate = W1[e, N+row, :] (rows [N, 2N))  -- w13 layout
            for it in cutlass.range_constexpr(self.it_n):
                row = r + it * _BLK
                if row < self.n:
                    acc_up = Float32(0.0)
                    acc_gate = Float32(0.0)
                    rg = self.n + row
                    # Runtime loop (not unrolled) so large K stays compilable.
                    for blk in cutlass.range(self.k_blocks):
                        ou = mxfp6_swizzled_scale_offset(
                            Int32(row), Int32(blk), Int32(self.cpd4_w1)
                        )
                        og = mxfp6_swizzled_scale_offset(
                            Int32(rg), Int32(blk), Int32(self.cpd4_w1)
                        )
                        wu = cvt_e8m0_to_f32(Uint32(mW1s[e, ou]))
                        wg = cvt_e8m0_to_f32(Uint32(mW1s[e, og]))
                        for gg in cutlass.range_constexpr(8):
                            g = blk * 8 + gg
                            bk = blk * 32 + gg * 4
                            u0, u1, u2, u3 = fp6_unpack4_codes(
                                Uint32(mW1p[e, row, g * 3 + 0]),
                                Uint32(mW1p[e, row, g * 3 + 1]),
                                Uint32(mW1p[e, row, g * 3 + 2]),
                            )
                            acc_up = acc_up + fp6_decode_code(mLutW, u0) * wu * sa_dq[bk + 0]
                            acc_up = acc_up + fp6_decode_code(mLutW, u1) * wu * sa_dq[bk + 1]
                            acc_up = acc_up + fp6_decode_code(mLutW, u2) * wu * sa_dq[bk + 2]
                            acc_up = acc_up + fp6_decode_code(mLutW, u3) * wu * sa_dq[bk + 3]
                            v0, v1, v2, v3 = fp6_unpack4_codes(
                                Uint32(mW1p[e, rg, g * 3 + 0]),
                                Uint32(mW1p[e, rg, g * 3 + 1]),
                                Uint32(mW1p[e, rg, g * 3 + 2]),
                            )
                            acc_gate = acc_gate + fp6_decode_code(mLutW, v0) * wg * sa_dq[bk + 0]
                            acc_gate = acc_gate + fp6_decode_code(mLutW, v1) * wg * sa_dq[bk + 1]
                            acc_gate = acc_gate + fp6_decode_code(mLutW, v2) * wg * sa_dq[bk + 2]
                            acc_gate = acc_gate + fp6_decode_code(mLutW, v3) * wg * sa_dq[bk + 3]
                    sig = cute.arch.rcp_approx(
                        Float32(1.0) + cute.math.exp(-acc_gate, fastmath=False)
                    )
                    sinter[row] = sig * acc_gate * acc_up
            cute.arch.sync_threads()
            # ---- re-quant intermediate to FP6 in place (one thread per 32-block) ----
            for it in cutlass.range_constexpr(self.it_nb):
                b = r + it * _BLK
                if b < self.n_blocks:
                    vals = cute.make_rmem_tensor((32,), Float32)
                    bmax = Float32(0.0)
                    for j in cutlass.range_constexpr(32):
                        v = sinter[b * 32 + j]
                        vals[j] = v
                        bmax = fmax_f32(bmax, fabs_f32(v))
                    containers, sb = moe_mxfp6_quantize_input_block_containers(
                        vals, bmax, gs2, self.act_fmt
                    )
                    bsc = cvt_e8m0_to_f32(Uint32(sb))
                    for j in cutlass.range_constexpr(32):
                        sinter[b * 32 + j] = (
                            fp6_decode_code(mLutA, Uint32(containers[j])) * bsc / gs2
                        )
            cute.arch.sync_threads()
            # ---- FC2: y[row] = sum_n decode(W2[e,row,n]) * wscale * inter_dq[n] ----
            for it in cutlass.range_constexpr(self.it_k):
                row = r + it * _BLK
                if row < self.k:
                    acc2 = Float32(0.0)
                    # Runtime loop (not unrolled) so large N stays compilable.
                    for blk in cutlass.range(self.n_blocks):
                        off = mxfp6_swizzled_scale_offset(
                            Int32(row), Int32(blk), Int32(self.cpd4_w2)
                        )
                        wsc = cvt_e8m0_to_f32(Uint32(mW2s[e, off]))
                        for gg in cutlass.range_constexpr(8):
                            g = blk * 8 + gg
                            c0, c1, c2, c3 = fp6_unpack4_codes(
                                Uint32(mW2p[e, row, g * 3 + 0]),
                                Uint32(mW2p[e, row, g * 3 + 1]),
                                Uint32(mW2p[e, row, g * 3 + 2]),
                            )
                            bn = blk * 32 + gg * 4
                            acc2 = acc2 + fp6_decode_code(mLutW, c0) * wsc * sinter[bn + 0]
                            acc2 = acc2 + fp6_decode_code(mLutW, c1) * wsc * sinter[bn + 1]
                            acc2 = acc2 + fp6_decode_code(mLutW, c2) * wsc * sinter[bn + 2]
                            acc2 = acc2 + fp6_decode_code(mLutW, c3) * wsc * sinter[bn + 3]
                    sout[row] = sout[row] + Float32(mTw[t, slot]) * acc2
            cute.arch.sync_threads()

        for it in cutlass.range_constexpr(self.it_k):
            row = r + it * _BLK
            if row < self.k:
                mOut[t, row] = sout[row].to(cutlass.BFloat16)


# Per-CTA static smem the kernel allocates: sa_dq[K] + sinter[N] + sout[K] f32,
# plus a small allocator/alignment margin.
_SMEM_MARGIN_BYTES = 2048
# Fallback opt-in smem ceiling if the device cannot report one (SM80+ baseline).
_DEFAULT_MAX_SMEM = 48 * 1024


def fp6_micro_smem_bytes(k: int, n: int) -> int:
    """Static shared-memory footprint (bytes) of the FP6 micro kernel."""
    return (2 * int(k) + int(n)) * 4 + _SMEM_MARGIN_BYTES


def fp6_micro_fits_smem(k: int, n: int, device: torch.device) -> bool:
    """Whether the FP6 micro kernel's smem fits this device's opt-in budget."""
    try:
        props = torch.cuda.get_device_properties(device)
        max_smem = int(
            getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SMEM)
        )
    except Exception:
        max_smem = _DEFAULT_MAX_SMEM
    return fp6_micro_smem_bytes(k, n) <= max_smem


@functools.lru_cache(maxsize=8)
def _decode_lut(fmt: str, device_index: int) -> torch.Tensor:
    return build_fp6_decode_lut(fmt, device=torch.device("cuda", device_index))


# Compiled-kernel cache: keyed by shape/format/m/device so the (expensive) JIT
# compile happens once, then each decode call just relaunches.
_COMPILED: dict = {}


def _ct(x: torch.Tensor, dtype) -> cute.Tensor:
    t = from_dlpack(x, assumed_align=16)
    t.element_type = dtype
    return t


def fp6_micro_moe(
    a: torch.Tensor,
    a1_gscale: torch.Tensor,
    w1_fp6: torch.Tensor,
    w1_blockscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp6: torch.Tensor,
    w2_blockscale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    source_format: str = "mxfp6_default",
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the FP6 BS1 decode micro-kernel. Returns ``[m, k]`` bf16.

    Consumes the production ``FP6MoEWeights`` storage directly: ``w1_fp6``
    ``(E, 2N, 3K/4)``, ``w2_fp6`` ``(E, K, 3N/4)`` packed codes, with swizzled
    UE8M0 ``*_blockscale``. Activations are quantized to FP6 inside the kernel.
    """
    act_fmt, weight_fmt = _fmt_pair(source_format)
    m, k = int(a.shape[0]), int(a.shape[1])
    experts, two_n, _ = w1_fp6.shape
    n = int(two_n) // 2
    topk = int(topk_ids.shape[1])
    dev_index = a.device.index if a.device.index is not None else 0

    lut_w = _decode_lut(weight_fmt, dev_index)
    lut_a = _decode_lut(act_fmt, dev_index)
    w1s = w1_blockscale.reshape(int(experts), -1).contiguous()
    w2s = w2_blockscale.reshape(int(experts), -1).contiguous()
    if output is None:
        output = torch.empty(m, k, dtype=torch.bfloat16, device=a.device)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        _ct(lut_w, cutlass.Float32),
        _ct(lut_a, cutlass.Float32),
        _ct(a.contiguous(), cutlass.BFloat16),
        _ct(w1_fp6.contiguous(), cutlass.Uint8),
        _ct(w1s, cutlass.Uint8),
        _ct(w2_fp6.contiguous(), cutlass.Uint8),
        _ct(w2s, cutlass.Uint8),
        _ct(topk_ids.contiguous().to(torch.int32), cutlass.Int32),
        _ct(topk_weights.contiguous().to(torch.float32), cutlass.Float32),
        _ct(a1_gscale.to(torch.float32), cutlass.Float32),
        _ct(a2_gscale.to(torch.float32), cutlass.Float32),
        _ct(output, cutlass.BFloat16),
        Int32(m),
        stream,
    )

    key = (n, k, topk, act_fmt, weight_fmt, m, dev_index)
    compiled = _COMPILED.get(key)
    if compiled is None:
        kernel = MoEMicroFp6Kernel(n, k, topk, act_fmt, weight_fmt)
        compiled = cute.compile(kernel, *args)
        _COMPILED[key] = compiled
    compiled(*args)
    return output
