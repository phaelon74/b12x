"""Compatibility shim for the warp-level MXF8 (``mxf8f6f4`` ``m16n8k32``)
block-scaled MMA op.

This project pins cutlass-dsl 4.5.2, which ships native SM120 MXF8/MXF6
block-scaled warp ops (including ``MmaMXF8Op``), so the resolver below uses the
native op directly. This shim is retained only as a transparent fallback for
older cutlass-dsl releases that shipped the FP4 block-scaled warp ops
(``MmaMXF4Op`` / ``MmaMXF4NVF4Op``) but not ``MmaMXF8Op``. The FP8/FP6 dense +
MoE GEMM paths need the MXF8 atom purely to derive the ``m16n8k32`` smem /
scale-factor layout geometry (the actual MMA in the mainloop is emitted as
inline PTX), so we reconstruct it here when a native op is unavailable.

On those older releases ``MmaSM120BlockScaledOp.__post_init__`` only validates
the FP4 case, but the underlying MLIR builder (``MmaAtomSM120BlockScaledType.get``)
is generic over the operand dtype. We therefore subclass the base op, bypass the
FP4-only guard, and replicate the upstream ``_make_trait`` with
``shape=(16, 8, 32)``.

Because ``_resolve_mxf8_op`` prefers a native ``MmaMXF8Op`` when present, the
shim is transparent across versions and stays dormant on the pinned 4.5.2.
"""

from __future__ import annotations

from typing import Any, Optional, Type

import cutlass
from cutlass.cute.nvgpu.warp import mma as _warp_mma


def _build_shim_class() -> type:
    base = _warp_mma.MmaSM120BlockScaledOp
    block_scaled_trait = _warp_mma.MmaBlockScaledTrait
    _pack_shape = _warp_mma._pack_shape
    make_atom = _warp_mma.make_atom
    nvgpu_ir = _warp_mma._cute_nvgpu_ir

    class _MmaMXF8Trait(block_scaled_trait):
        pass

    class _MmaMXF8OpShim(base):
        """Warp-level MXF8 (``m16n8k32``) block-scaled MMA op fallback for
        cutlass-dsl releases without a native ``MmaMXF8Op`` (pre-4.5.2)."""

        descriptive_name = "warp-level MXF8 MMA Operation (b12x legacy shim)"

        def __init__(
            self,
            ab_dtype: Type[cutlass.Numeric],
            acc_dtype: Type[cutlass.Numeric],
            sf_type: Type[cutlass.Numeric],
        ) -> None:
            # Set the frozen-dataclass fields directly to skip the base
            # ``__post_init__`` (which rejects non-FP4 operands on pre-4.5.2
            # cutlass-dsl releases).
            object.__setattr__(self, "ab_dtype", ab_dtype)
            object.__setattr__(self, "acc_dtype", acc_dtype)
            object.__setattr__(self, "shape_mnk", (16, 8, 32))
            object.__setattr__(self, "sf_type", sf_type)
            object.__setattr__(self, "sf_vec_size", 32)
            object.__setattr__(self, "use_sf_layout_TV", False)
            if acc_dtype is not cutlass.Float32:
                raise ValueError(
                    "MmaMXF8Op shim expects acc_dtype=Float32, "
                    f"got {acc_dtype}"
                )
            if sf_type is not cutlass.Float8E8M0FNU:
                raise ValueError(
                    "MmaMXF8Op shim expects sf_type=Float8E8M0FNU, "
                    f"got {sf_type}"
                )

        def _make_trait(
            self,
            *,
            loc: Optional[Any] = None,
            ip: Optional[Any] = None,
            **kwargs: Any,
        ) -> "_MmaMXF8Trait":
            shape_mnk = _pack_shape(self.shape_mnk, loc=loc, ip=ip)
            ty = nvgpu_ir.MmaAtomSM120BlockScaledType.get(
                shape_mnk.type.attribute,
                32,
                False,
                self.ab_dtype.mlir_type,
                self.ab_dtype.mlir_type,
                self.acc_dtype.mlir_type,
                self.sf_type.mlir_type,
            )
            return _MmaMXF8Trait(make_atom(ty, loc=loc, ip=ip))

    return _MmaMXF8OpShim


def _resolve_mxf8_op() -> type:
    native = getattr(_warp_mma, "MmaMXF8Op", None)
    if native is not None:
        return native
    return _build_shim_class()


# Drop-in replacement for ``cute.nvgpu.warp.MmaMXF8Op``: native when available
# (the pinned cutlass-dsl 4.5.2 ships it), otherwise the legacy shim.
MmaMXF8Op = _resolve_mxf8_op()
