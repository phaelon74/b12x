from __future__ import annotations

import cutlass
import pytest
import torch

from b12x.cute.utils import mxfp6_packed_k_bytes
from b12x.quantization import (
    allocate_bf16_to_fp6_tma_outputs,
    compile_bf16_to_fp6_tma,
)
from b12x.quantization.bf16_to_fp6_tma import TestKernel


def test_bf16_to_fp6_tma_kernel_import() -> None:
    assert TestKernel is not None
    k = TestKernel()
    assert k.tile_shape_mnk == (128, 128, 128)
    assert k.threads_per_cta == 160


def test_allocate_bf16_to_fp6_tma_outputs_shapes() -> None:
    m, k = 128, 128
    out = allocate_bf16_to_fp6_tma_outputs(m, k, device=torch.device("cpu"))
    assert out.packed_a_storage.shape == (1, m, mxfp6_packed_k_bytes(k))
    assert out.packed_a_storage.dtype == torch.uint8
    rows_pad = 128
    cols_pad_sf = 4
    assert out.scale_storage.shape == (rows_pad * cols_pad_sf,)
    assert out.scale_storage.dtype == torch.uint8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile")
def test_compile_bf16_to_fp6_tma_returns_callable() -> None:
    launch = compile_bf16_to_fp6_tma(128, 128)
    assert callable(launch)
    launch2 = compile_bf16_to_fp6_tma(128, 128)
    assert launch2 is launch


def test_compile_bf16_to_fp6_tma_rejects_bad_k() -> None:
    with pytest.raises(AssertionError):
        compile_bf16_to_fp6_tma(128, 129)
