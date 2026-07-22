"""Opaque ``torch.custom_op`` wrapper for the small-N bf16 GEMV.

Registers ``b12x::bf16_gemv_small_n`` so vLLM's Dynamo/CUDA-graph path treats
the GEMV (compile-cache lookup + CUTE launch) as a single opaque node, exactly
like ``b12x::fp6_dense_linear``. The fallback for shapes the kernel does not
cover (prefill m > SMALL_M_MAX, odd K, misalignment) lives INSIDE the op, so
the calling graph never branches on data-dependent shapes.

Importing this module performs the registration; the vLLM plugin imports it in
``process_weights_after_loading`` so the op exists before model compilation.
"""
from __future__ import annotations

import torch

from b12x.gemm.bf16_gemv import (
    SMALL_M_MAX,
    compile_bf16_gemv_small_n,
    get_cached_bf16_gemv_small_n,
)


def _use_kernel(x: torch.Tensor, weight: torch.Tensor) -> bool:
    m, k = x.shape
    return (
        1 <= m <= SMALL_M_MAX
        and k % 8 == 0
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
    )


@torch.library.custom_op("b12x::bf16_gemv_small_n", mutates_args=())
def bf16_gemv_small_n(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``y = x @ weight.T`` for bf16 ``x (m, K)`` / ``weight (N, K)``.

    Routes small decode shapes (``m <= SMALL_M_MAX``) through the CUTE GEMV;
    anything else falls back to ``F.linear`` (cuBLAS) inside the op.
    """
    if not _use_kernel(x, weight):
        return torch.nn.functional.linear(x, weight)
    if not x.is_contiguous() or x.data_ptr() % 16 != 0:
        x = x.contiguous()
        if x.data_ptr() % 16 != 0:
            return torch.nn.functional.linear(x, weight)
    m, k = x.shape
    n = weight.shape[0]
    launch = get_cached_bf16_gemv_small_n(m, n, k)
    if launch is None:
        # Never JIT while a stream is capturing: cute.compile (ptxas +
        # cuModuleLoad + syncs) corrupts an in-flight CUDA-graph capture.
        # Serving precompiles all m at weight load, so this only triggers
        # for shapes precompile never saw — cuBLAS is the safe answer there.
        if torch.cuda.is_current_stream_capturing():
            return torch.nn.functional.linear(x, weight)
        launch = compile_bf16_gemv_small_n(m, n, k)
    y = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    launch(x, weight, y)
    return y


@bf16_gemv_small_n.register_fake
def _bf16_gemv_small_n_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)
