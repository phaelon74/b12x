"""Drop-in replacement for ``cutlass.utils.blackwell_helpers``.

b12x's SM120 block-scaled GEMM mainloop uses a handful of scale-factor partition
helpers (``partition_fragment_SFA``/``SFB``, ``thrfrg_SFA``/``SFB``,
``get_layoutSFA_TV``/``SFB_TV``). The public ``nvidia-cutlass-dsl`` 4.5.2 wheel
ships only the FP4 path and omits these, so a fresh ``pip install`` would crash
at GEMM compile time with ``AttributeError`` on the missing symbol.

This module exposes the same surface as ``cutlass.utils.blackwell_helpers`` (via
attribute forwarding) and transparently supplies the missing SM120 MXF8/MXF6
helpers from :mod:`b12x.cute.sm120_blockscaled_helpers`. A cutlass build that
already provides a symbol natively always wins; only genuinely-missing symbols
fall back to the vendored implementation. Import this as ``sm120_utils`` instead
of ``cutlass.utils.blackwell_helpers``.
"""

from __future__ import annotations

import cutlass.utils.blackwell_helpers as _blackwell_helpers

from b12x.cute import sm120_blockscaled_helpers as _vendored

# Helpers b12x may need that the stock PyPI cutlass-dsl wheel does not export.
_VENDORED_FALLBACKS = (
    "partition_fragment_SFA",
    "partition_fragment_SFB",
    "thrfrg_SFA",
    "thrfrg_SFB",
    "get_layoutSFA_TV",
    "get_layoutSFB_TV",
)


def __getattr__(name: str):
    # Prefer the cutlass-native symbol when the installed build provides it; a
    # patched/newer cutlass stays authoritative. Only fall back to the vendored
    # b12x implementation when the native module is missing the symbol.
    native = getattr(_blackwell_helpers, name, None)
    if native is not None:
        return native
    if name in _VENDORED_FALLBACKS:
        return getattr(_vendored, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r} "
        f"(not found in cutlass.utils.blackwell_helpers nor vendored fallbacks)"
    )


def __dir__() -> list[str]:
    return sorted(set(dir(_blackwell_helpers)) | set(_VENDORED_FALLBACKS))
