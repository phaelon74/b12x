"""Direct routed W4A16 micro MoE kernel for SM120.

The compact W4A16 path dequantizes whole FP4 B tiles into BF16 shared memory
and then runs BF16 MMA.  For tiny routed batches that structure is dominated
by CTA barriers and underfilled producer/consumer phases.  The direct micro
kernel keeps the W4A4 scheduler, but stages 16-bit activations and activated
intermediates directly before dotting with FP4 weights.
"""

from __future__ import annotations

from typing import Tuple

from b12x.moe.fused.micro import (
    _MAX_DIRECT_K_SEGMENTS,
    _direct_k_segments_for_k,
    _direct_k_segments_supported,
    _fc1_chunks_for_m,
    MoEMicroKernelBackend as _DirectMoEMicroKernelBackend,
)

_MAX_DIRECT_MICRO_TOKENS = 32


class MoEMicroKernelBackend(_DirectMoEMicroKernelBackend):
    """Low-latency direct W4A16 path for micro routed batches."""

    @classmethod
    def is_supported(
        cls,
        m: int,
        k: int,
        n: int,
        num_topk: int,
        weight_E: int,
    ) -> bool:
        if m <= 0 or m > _MAX_DIRECT_MICRO_TOKENS:
            return False
        if k <= 0 or k % 16 != 0 or k % 128 != 0:
            return False
        if _direct_k_segments_for_k(k) > _MAX_DIRECT_K_SEGMENTS:
            return False
        if n <= 0 or n % 16 != 0:
            return False
        if m >= 4 and n >= 4096:
            # CUTLASS 4.5 can report this family as launchable by thread-count
            # metadata, but the direct W4A16 kernel is outside its safe
            # resource envelope for multi-token, very-wide FC1 shapes.
            return False
        fc1_chunks = _fc1_chunks_for_m(m, n)
        if m > 1:
            fc1_chunks = max(fc1_chunks, n // 16)
        if n % fc1_chunks != 0:
            return False
        i_chunk = n // fc1_chunks
        if i_chunk % 16 != 0:
            return False
        k_segments = _direct_k_segments_for_k(k)
        return (
            _direct_k_segments_supported(k_segments)
            and 0 < num_topk <= 32
            and weight_E > 0
        )

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        fast_math: bool = False,
        activation: str = "silu",
        share_input_across_experts: bool = False,
        share_expert_scales: bool = False,
        single_token: bool = False,
        dynamic_down_scale: bool = False,
        compile_time_phase: int = 0,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            output_tile_count_n,
            fast_math=fast_math,
            activation=activation,
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            single_token=single_token,
            dynamic_down_scale=dynamic_down_scale,
            compile_time_phase=compile_time_phase,
            w4a16_mode=True,
        )


__all__ = ["MoEMicroKernelBackend"]
