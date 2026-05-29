"""BF16 → NVFP4 / MX-FP6 TMA quantization kernel APIs."""
from dataclasses import dataclass
from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.typing import AddressSpace

from b12x.cute.compiler import KernelCompileSpec, compile as b12x_compile
from b12x.cute.fp4 import align_up
from b12x.cute.utils import (
    MXFP6_SF_VEC_SIZE,
    current_cuda_stream,
    get_max_active_clusters,
    get_num_sm,
    make_ptr,
    mxfp6_packed_k_bytes,
)
from b12x.quantization.bf16_to_fp4_tma import TestKernel as FP4TestKernel
from b12x.quantization.bf16_to_fp4_tma import make_ptr as _fp4_make_ptr
from b12x.quantization.bf16_to_fp6_tma import TestKernel as FP6TestKernel
from b12x.quantization.bf16_to_fp6_tma import make_ptr as _fp6_make_ptr
from b12x.cute.runtime_control import raise_if_kernel_resolution_frozen

_TILE_M = 128
_TILE_K = 128
_SF_VEC_SIZE_FP4 = 16
_SF_VEC_SIZE_FP6 = MXFP6_SF_VEC_SIZE
_KERNEL_CACHE_FP4: Dict[Tuple, object] = {}
_KERNEL_CACHE_FP6: Dict[Tuple, object] = {}


@dataclass
class BF16ToFP4TMAOutputs:
    packed_a_storage: torch.Tensor
    scale_storage: torch.Tensor
    packed_a_view: object
    sfa_ptr: object

    @property
    def packed_a_flat(self) -> torch.Tensor:
        return self.packed_a_storage.view(-1)

    @property
    def scale_flat(self) -> torch.Tensor:
        return self.scale_storage.view(-1)


def allocate_bf16_to_fp4_tma_outputs(
    M: int, K: int, *, device: torch.device = torch.device("cuda"),
) -> BF16ToFP4TMAOutputs:
    rows_pad = align_up(M, _TILE_M)
    cols_pad_sf = align_up(K // _SF_VEC_SIZE_FP4, 4)
    packed_a_storage = torch.zeros(1, M, K // 2, dtype=torch.uint8, device=device)
    scale_storage = torch.zeros(rows_pad * cols_pad_sf, dtype=torch.uint8, device=device)
    packed_a_view = packed_a_storage.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
    sfa_ptr = make_ptr(
        cutlass.Float8E4M3FN, scale_storage.data_ptr(),
        cute.AddressSpace.gmem, assumed_align=16,
    )
    return BF16ToFP4TMAOutputs(
        packed_a_storage=packed_a_storage,
        scale_storage=scale_storage,
        packed_a_view=packed_a_view,
        sfa_ptr=sfa_ptr,
    )


def compile_bf16_to_fp4_tma(M: int, K: int):
    """Compile the BF16→FP4 TMA kernel for (M, K). Returns a launch callable.

    The callable signature is: ``launch(bf16_input, global_scale, packed_a_flat, scale_flat)``
    where packed_a_flat and scale_flat come from ``BF16ToFP4TMAOutputs``.
    """
    assert M % _TILE_M == 0 and K % _TILE_K == 0
    cache_key = (M, K)
    cached = _KERNEL_CACHE_FP4.get(cache_key)
    if cached is not None:
        return cached

    ab = cutlass.Float4E2M1FN
    sf = cutlass.Float8E4M3FN
    bf = cutlass.BFloat16
    bf16_fake = cute.runtime.make_fake_compact_tensor(bf, (M, K), stride_order=(1, 0), assumed_align=16)
    gs_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
    pa_fake = cute.runtime.make_fake_compact_tensor(ab, (M, K, 1), stride_order=(1, 0, 2), assumed_align=16)
    sfa_fake = _fp4_make_ptr(sf, 16, AddressSpace.gmem, assumed_align=16)
    mac = min(get_max_active_clusters(1), get_num_sm(torch.device("cuda")))
    kernel = FP4TestKernel()
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=cache_key)
    raw = b12x_compile(
        kernel,
        bf16_fake,
        gs_fake,
        pa_fake,
        sfa_fake,
        mac,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "quantization.bf16_to_fp4_tma",
            1,
            cache_key,
        ),
    )

    def launch(bf16_input, global_scale, packed_a_flat, scale_flat):
        pa_view = packed_a_flat.view(1, M, K // 2).permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
        sfa_p = _fp4_make_ptr(sf, scale_flat.data_ptr(), AddressSpace.gmem, assumed_align=16)
        raw(bf16_input, global_scale, pa_view, sfa_p, current_cuda_stream())

    _KERNEL_CACHE_FP4[cache_key] = launch
    return launch


@dataclass
class BF16ToFP6TMAOutputs:
    packed_a_storage: torch.Tensor
    scale_storage: torch.Tensor
    packed_a_view: object
    sfa_ptr: object

    @property
    def packed_a_flat(self) -> torch.Tensor:
        return self.packed_a_storage.view(-1)

    @property
    def scale_flat(self) -> torch.Tensor:
        return self.scale_storage.view(-1)


def allocate_bf16_to_fp6_tma_outputs(
    M: int,
    K: int,
    *,
    device: torch.device = torch.device("cuda"),
) -> BF16ToFP6TMAOutputs:
    """Allocate uint8 packed activations (``3K/4`` bytes per row) and swizzled UE8M0 scales."""
    rows_pad = align_up(M, _TILE_M)
    cols_pad_sf = align_up(K // _SF_VEC_SIZE_FP6, 4)
    packed_k_bytes = mxfp6_packed_k_bytes(K)
    packed_a_storage = torch.zeros(1, M, packed_k_bytes, dtype=torch.uint8, device=device)
    scale_storage = torch.zeros(rows_pad * cols_pad_sf, dtype=torch.uint8, device=device)
    packed_a_view = packed_a_storage.permute(1, 2, 0)
    sfa_ptr = make_ptr(
        cutlass.Float8E8M0FNU,
        scale_storage.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return BF16ToFP6TMAOutputs(
        packed_a_storage=packed_a_storage,
        scale_storage=scale_storage,
        packed_a_view=packed_a_view,
        sfa_ptr=sfa_ptr,
    )


def compile_bf16_to_fp6_tma(M: int, K: int):
    """Compile the BF16→MX-FP6 (E3M2) TMA kernel for (M, K).

    The callable signature is: ``launch(bf16_input, global_scale, packed_a_flat, scale_flat)``
    where packed_a_flat and scale_flat come from ``BF16ToFP6TMAOutputs``.
    """
    assert M % _TILE_M == 0 and K % _TILE_K == 0
    cache_key = (M, K)
    cached = _KERNEL_CACHE_FP6.get(cache_key)
    if cached is not None:
        return cached

    ab = cutlass.Float6E3M2FN
    sf = cutlass.Float8E8M0FNU
    bf = cutlass.BFloat16
    packed_k_bytes = mxfp6_packed_k_bytes(K)
    bf16_fake = cute.runtime.make_fake_compact_tensor(
        bf, (M, K), stride_order=(1, 0), assumed_align=16
    )
    gs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    pa_fake = cute.runtime.make_fake_compact_tensor(
        ab, (M, packed_k_bytes, 1), stride_order=(1, 0, 2), assumed_align=16
    )
    sfa_fake = _fp6_make_ptr(sf, 16, AddressSpace.gmem, assumed_align=16)
    mac = min(get_max_active_clusters(1), get_num_sm(torch.device("cuda")))
    kernel = FP6TestKernel()
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=cache_key)
    raw = b12x_compile(
        kernel,
        bf16_fake,
        gs_fake,
        pa_fake,
        sfa_fake,
        mac,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "quantization.bf16_to_fp6_tma",
            1,
            cache_key,
        ),
    )
    def launch(bf16_input, global_scale, packed_a_flat, scale_flat):
        pa_storage = packed_a_flat.view(1, M, packed_k_bytes).permute(1, 2, 0)
        sfa_p = _fp6_make_ptr(
            sf, scale_flat.data_ptr(), AddressSpace.gmem, assumed_align=16
        )
        raw(bf16_input, global_scale, pa_storage, sfa_p, current_cuda_stream())

    _KERNEL_CACHE_FP6[cache_key] = launch
    return launch
