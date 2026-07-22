"""Installable vLLM plugin that registers the B12X MX-FP6 (W6A6) quantization.

This is wired through the ``vllm.general_plugins`` entry point (see
``pyproject.toml``), which vLLM loads in *every* process -- the front process,
the engine-core process, and each tensor-parallel worker. That is what makes the
registration survive worker spawn under TP, unlike an in-process shim.

Flow
----
1. Entry point ``register_b12x_fp6()`` runs in each process and calls
   ``register_quantization_config("b12x_fp6")`` once (idempotent).
2. ``B12XFp6Config.override_quantization_method`` claims a ``modelopt`` +
   ``quant_algo=W6A6`` checkpoint when ``B12X_ENABLE_FP6=1``, so the launch
   script does not strictly need ``--quantization b12x_fp6`` (passing it is also
   fine and forces selection).
3. The checkpoint directory is resolved from ``B12X_FP6_MODEL_DIR`` (exported by
   the launch script and inherited by spawned workers) or, as a fallback, from
   ``maybe_update_config(model_name=...)``.
4. ``get_quant_method`` dispatches each layer to the tested B12X kernels in
   :mod:`b12x.integration.fp6_serving`:
     * ``FusedMoE``  -> ``_VllmMoEMethod`` -> :class:`B12XFP6MoEMethod`
       -> ``b12x_moe_fp6``
     * ``LinearBase``-> ``_VllmLinearMethod`` -> ``dense_fp6_linear``

Weight binding (READ THIS)
--------------------------
Both methods register **real** vLLM params in ``create_weights``
(mirroring vLLM's own ModelOpt NVFP4 methods) so the framework loader places
the packed FP6 tensors itself.

The dense linear method registers:

* ``weight``        ``(out, 3*in/4)`` uint8   -- packed MX-FP6 codes
* ``weight_scale``  ``(out, in/32)``  uint8   -- UE8M0 block scales (unswizzled)
* ``weight_scale_2``/``input_scale`` per-shard f32 globals (unit, pure-MX)

Because vLLM fuses ``q/k/v_proj -> qkv_proj`` and ``gate/up_proj ->
gate_up_proj`` via ``stacked_params_mapping``, the standard merged/QKV weight
loaders split the separate on-disk matrices into our packed params with no
custom fusion code (FP6 row-packing + per-block scales are independent per row,
so a row-concat is bit-identical to a single-matrix quant).
``process_weights_after_loading`` swizzles the scales once and builds the
kernel-ready :class:`FP6DenseWeight`; ``apply`` runs ``dense_fp6_linear``.

The MoE method registers stacked per-expert params (``w13_weight`` ``(E, 2N,
3K/4)``, ``w2_weight`` ``(E, K, 3N/4)``, block scales, per-tensor globals) that
vLLM's FusedMoE expert loader fills via the standard ``(expert_id, shard_id)``
convention — model-agnostic across expert counts and checkpoint naming.
``process_weights_after_loading`` reorders the FC1 rows from vLLM's ``[gate;
up]`` to the kernel's ``[up; gate]`` contract, swizzles the block scales,
builds :class:`FP6MoEWeights`, and warm-runs the decode token counts so the
fused kernels are compiled and workspaces allocated before CUDA-graph capture.
Shared experts and routing stay with vLLM (the runner hands ``apply`` finished
``topk_weights``/``topk_ids``).

Non-FP6 ``LinearBase`` layers (``lm_head``, norms, any checkpoint module left
BF16, ...) get vLLM's stock ``UnquantizedLinearMethod`` — except *small-N*
bf16 linears (e.g. the GDN ``in_proj_ba``, N <= 1024 with K >= 1024), which
get a subclass that routes decode-sized calls through the b12x small-N GEMV
(``b12x::bf16_gemv_small_n``): cuBLAS picks a ~28 us WMMA tile kernel for
those shapes, the GEMV does the same job in ~3 us (~1.35 ms/step across the
48 GDN layers).

Full multimodal FP6 is supported:

* **``linear_attn`` (GDN).** Qwen3.6 builds ``in_proj_qkvz`` /``in_proj_ba`` as
  ``MergedColumnParallelLinear`` *with* ``quant_config``, so they route through
  this method. ``in_proj_qkvz`` fuses the on-disk ``in_proj_qkv`` (q/k/v) +
  ``in_proj_z`` (mapped to shards ``(0,1,2)`` and ``3``); the packed FP6 shards
  load by output-dim concat, bit-identical to a single-matrix quant. ``out_proj``
  is a standalone FP6 linear.
* **Vision tower.** ``Qwen3_VisionTransformer`` is constructed *with*
  ``quant_config`` too, so ``attn.qkv`` / ``attn.proj`` / ``mlp.linear_fc1`` /
  ``mlp.linear_fc2`` are FP6 when present on disk (``--include-vision``).

.. note::
   This targets the ``in_proj_qkvz`` + ``in_proj_ba`` GDN layout (what current
   nightlies emit). If a future vLLM (post the 6-way fused ``in_proj`` perf PR)
   instantiates a single fused ``in_proj`` under a *uniform-precision* quant
   config, we must expose a skip-list attribute so vLLM falls back to the
   per-module qkvz+ba layout (our quant is genuinely mixed-precision).

The module imports without vLLM installed (all vLLM imports are deferred); the
``QuantizationConfig`` subclass is defined inside :func:`register_b12x_fp6` so it
can subclass the framework base only when vLLM is present.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import torch

from b12x.integration.fp6_serving import (
    QUANT_ALGO,
    QUANT_METHOD,
    B12XFP6MoEMethod,
    is_b12x_fp6_enabled,
)

MODEL_DIR_ENV = "B12X_FP6_MODEL_DIR"
QUANT_NAME = "b12x_fp6"

# Unquantized bf16 linears with N <= MAX_OUT and K >= MIN_IN are routed
# through the b12x small-N GEMV (see module docstring). Catches the GDN
# ``in_proj_ba`` while excluding lm_head (N = vocab) and anything wide
# enough that cuBLAS tiles efficiently. B12X_DISABLE_BF16_GEMV=1 turns the
# routing off entirely (debug isolation switch).
SMALL_N_GEMV_MAX_OUT = 1024
SMALL_N_GEMV_MIN_IN = 1024


def _bf16_gemv_disabled() -> bool:
    return os.environ.get("B12X_DISABLE_BF16_GEMV", "").lower() in (
        "1",
        "true",
        "yes",
    )


# Fallback decode token counts warm-run per MoE layer at load time so the
# fused kernels are JIT-compiled and the (M, topk) workspaces exist before
# CUDA-graph capture. The actual warm list is derived from vLLM's resolved
# cudagraph_capture_sizes at load time (see _moe_warm_decode_ms) — vLLM pads
# every decode batch to a captured size, so warming exactly the capture sizes
# is both necessary and sufficient. This tuple is only the safety net when
# the config is unavailable; it covers the common MTP configs (verify batch
# m = 1+k for k in {1,2,3,4,7}).
_MOE_WARM_DECODE_MS = (1, 2, 3, 4, 5, 8)

# Never warm (or trust) capture sizes above this: decode/verify batches are
# small by construction, and warming huge Ms would waste load time and pool
# memory. Anything larger falls outside full_decode_only capture anyway.
_MOE_WARM_MAX_M = 64


def _moe_warm_decode_ms() -> tuple[int, ...]:
    """Decode Ms to warm-run per MoE layer before CUDA-graph capture.

    Priority: B12X_MOE_WARM_MS env override > vLLM's resolved
    cudagraph_capture_sizes (unioned with the fallback tuple) > fallback
    tuple. Derived dynamically so launch scripts do NOT have to keep their
    capture sizes in sync with a constant here.
    """
    env = os.environ.get("B12X_MOE_WARM_MS", "").strip()
    if env:
        try:
            ms = {int(v) for v in env.replace(",", " ").split()}
            return tuple(sorted(m for m in ms if 1 <= m <= _MOE_WARM_MAX_M))
        except ValueError:
            logger.warning("Ignoring malformed B12X_MOE_WARM_MS=%r", env)
    sizes = set(_MOE_WARM_DECODE_MS)
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        cap = getattr(cfg.compilation_config, "cudagraph_capture_sizes", None)
        if cap:
            sizes.update(int(s) for s in cap)
    except Exception:  # config not resolvable here -> fallback tuple only
        logger.debug("vLLM config unavailable for MoE warm sizes", exc_info=True)
    return tuple(sorted(s for s in sizes if 1 <= s <= _MOE_WARM_MAX_M))

# Dense decode/verify token counts warm-run at load time (Phase 2.1). The
# (16,128) small-M tile now covers m<=16 (BS1 decode + MTP verify m=1+k);
# widening past the old m<=4 cap is safe only because every m-bucket of every
# unique dense shape is compiled and first-launched HERE — never mid-serving
# on a live prefill-tail token count. {1,2,4,5,8,16} covers both scheduler
# policies (direct one-M-tile for m<16, standard at 16) and the common
# cudagraph capture sizes.
_DENSE_WARM_DECODE_MS = (1, 2, 4, 5, 8, 16)

# Unique (out, in, fmt, act_fmt, packed) dense shapes already warmed — the
# compiled-kernel cache is keyed on shape, not layer, so one warm-run per
# shape covers every layer sharing it.
_DENSE_WARMED_SHAPES: set[tuple] = set()

# Fused vLLM module suffix -> ordered HF constituent projection names. The order
# matches how vLLM concatenates the output (row) dimension of the fused weight
# (see each model's ``stacked_params_mapping`` / ``packed_modules_mapping``).
#
# A fused module is treated as FP6 iff *every* constituent is FP6 on disk, so a
# partially-BF16 fusion (e.g. Qwen3.6 GDN ``in_proj_ba``, whose ``num_v_heads``
# rows are too small to quantize) cleanly falls back to BF16.
_FUSED_PARTS: dict[str, tuple[str, ...]] = {
    # Standard attention / MLP fusions.
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    # Qwen3.5/3.6 Gated-DeltaNet (linear_attn) input projections. vLLM fuses the
    # on-disk ``in_proj_qkv`` (pre-merged q/k/v) + ``in_proj_z`` into a single
    # ``in_proj_qkvz`` MergedColumnParallelLinear; the FP6 packed shards load by
    # output-dim concat exactly like qkv_proj. ``in_proj_ba`` is listed for
    # completeness but its constituents are tiny and stay BF16.
    "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
    "in_proj_ba": ("in_proj_b", "in_proj_a"),
}

_registered = False
# Set by register_b12x_fp6(); lets _rebuild_b12x_fp6_config reconstruct the
# dynamically-defined config class in a freshly spawned process.
_CONFIG_CLS: Optional[type] = None

import logging  # noqa: E402

logger = logging.getLogger("b12x.vllm_fp6")


def _rebuild_b12x_fp6_config(model_dir: Optional[str]) -> Any:
    """Pickle factory for :class:`B12XFp6Config` (see its ``__reduce__``).

    The config class is defined inside :func:`register_b12x_fp6` (it can only
    subclass vLLM's base once vLLM is importable), so pickle cannot reference it
    by name and cloudpickle would fall back to serializing the class *by value*
    -- dragging the whole closure graph (including unpicklable torch internals
    like ``torch.ops``) into vLLM's spawn-time ``VllmConfig`` pickle. This
    module-level factory is picklable by reference; the spawned process re-runs
    the (idempotent) registration and rebuilds a fresh config, which re-resolves
    its state lazily from ``B12X_FP6_MODEL_DIR``.
    """
    register_b12x_fp6()
    assert _CONFIG_CLS is not None
    return _CONFIG_CLS(model_dir)


def _resolve_model_dir(explicit: Optional[str] = None) -> Optional[str]:
    """Checkpoint dir from an explicit value, the env override, else None."""
    return explicit or os.environ.get(MODEL_DIR_ENV) or None


def _norm_key(name: str) -> str:
    """Canonicalize a module path so vLLM prefixes match checkpoint names.

    vLLM and the on-disk checkpoint disagree on namespace ordering (e.g. vLLM
    ``language_model.layers.0...`` vs checkpoint ``model.language_model.layers.0...``).
    Strip a leading ``model.`` and collapse the two ``language_model``/``model``
    orderings to a single canonical form so a dict lookup lines up.
    """
    name = name.removeprefix("model.")
    name = name.replace("language_model.model.", "language_model.")
    return name


def register_b12x_fp6() -> None:
    """vLLM ``general_plugins`` entry point: register the ``b12x_fp6`` config.

    Safe to call multiple times and in multiple processes (vLLM loads general
    plugins in every process); registration is guarded so it only happens once
    per process.
    """
    global _registered, _CONFIG_CLS, logger
    if _registered:
        return

    # Route plugin logging through vLLM's logger config: vLLM only wires
    # handlers for the "vllm" namespace, so plain "b12x.*" loggers are
    # invisible in server output (this is why earlier `grep "B12X FP6"`
    # runs came back empty).
    try:
        from vllm.logger import init_logger

        logger = init_logger("vllm.b12x_fp6")
    except Exception:
        pass

    try:
        # vLLM < 2026-06 MoE refactor (pre #41184).
        from vllm.model_executor.layers.fused_moe.layer import FusedMoEMethodBase
    except ImportError:
        try:
            from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
        except ImportError:
            from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
                FusedMoEMethodBase,
            )
    from vllm.model_executor.layers.linear import (
        LinearMethodBase,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    from vllm.model_executor.parameter import (
        ModelWeightParameter,
        PerTensorScaleParameter,
    )

    # vLLM method wrappers (defined first so the config closes over them).
    class _VllmLinearMethod(LinearMethodBase):  # type: ignore[misc]
        """Real vLLM linear method for B12X MX-FP6 (mirrors ModelOpt NVFP4).

        Registers the packed FP6 params so vLLM's loader (incl. the merged
        qkv/gate_up shard loaders) places the on-disk tensors itself, then
        swizzles the block scales once and runs ``dense_fp6_linear``.
        """

        def __init__(
            self,
            source_format: str,
            *,
            default_act_fmt: str,
            act_fmt_overrides: list[tuple[str, str]] | None = None,
        ):
            self.source_format = source_format
            self.default_act_fmt = default_act_fmt
            self.act_fmt_overrides = list(act_fmt_overrides or [])

        def create_weights(
            self,
            layer: Any,
            input_size_per_partition: int,
            output_partition_sizes: list[int],
            input_size: int,
            output_size: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs: Any,
        ) -> None:
            del input_size, params_dtype
            weight_loader = extra_weight_attrs.get("weight_loader")
            out_total = sum(output_partition_sizes)
            in_p = int(input_size_per_partition)
            if in_p % 32 != 0:
                raise ValueError(
                    f"B12X FP6 requires in_features % 32 == 0, got {in_p}"
                )
            layer.logical_widths = output_partition_sizes
            # Phase 3.1: full (unsharded) out_features for the packed-B
            # decision — use_packed_gemm must compare against the global N,
            # not the per-GPU N/tp, to avoid flipping gate_up_proj off the
            # packed path at TP>1.
            layer.b12x_out_features_unsharded = int(output_size)

            # Packed FP6 codes: 4 values per 3 bytes along the input dim.
            weight = ModelWeightParameter(
                data=torch.empty(out_total, (3 * in_p) // 4, dtype=torch.uint8),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)

            # Per-32 UE8M0 block scales, unswizzled as stored on disk.
            weight_scale = ModelWeightParameter(
                data=torch.empty(out_total, in_p // 32, dtype=torch.uint8),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale", weight_scale)

            # Per-shard global scales (unit for pure-MX W6A6).
            n_shards = len(output_partition_sizes)
            weight_scale_2 = PerTensorScaleParameter(
                data=torch.ones(n_shards, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale_2", weight_scale_2)
            input_scale = PerTensorScaleParameter(
                data=torch.ones(n_shards, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_scale", input_scale)

        def process_weights_after_loading(self, layer: Any) -> None:
            from b12x.quantization.fp6_checkpoint import resolve_activation_format
            from b12x.quantization.fp6_dense_weights import PACKED_GEMM_MIN_N

            ws2 = layer.weight_scale_2.data.float()
            if not torch.allclose(ws2, torch.ones_like(ws2), atol=1e-3):
                raise NotImplementedError(
                    "B12X FP6 vLLM path requires unit weight_scale_2 "
                    "(pure-MX W6A6 contract)."
                )
            # Model-agnostic per-module act format: fnmatch overrides against
            # layer.prefix (e.g. "*.linear_attn.*" -> e3m2). Default is the
            # checkpoint-wide activation_format (e4m3 for mxfp6_w6a8).
            prefix = str(getattr(layer, "prefix", "") or "")
            act_fmt = resolve_activation_format(
                prefix,
                default_fmt=self.default_act_fmt,
                overrides=self.act_fmt_overrides,
            )
            w = _dense_weight_from_loaded(
                layer.weight.data,
                layer.weight_scale.data,
                self.source_format,
                act_fmt=act_fmt,
                out_features_unsharded=getattr(
                    layer, "b12x_out_features_unsharded", 0
                ),
            )
            dev, dt = layer.weight.device, layer.weight.dtype
            # Per-layer weight format (see FP6DenseWeight.use_packed_gemm):
            # * Wide-N layers (e.g. gate_up_proj) keep the 3:4-packed codes and
            #   the GEMM streams them natively (b_packed) — fastest there AND
            #   25% less weight VRAM (no expanded copy exists at all).
            # * Other layers materialize the 1-byte/code expansion once and
            #   release the packed source: at low CTA counts the in-smem
            #   expansion chain loses to pre-expanded streaming, and keeping
            #   the packed codes resident would waste ~0.75 byte/code on top.
            gemm_w = w.gemm_weight()
            if w.use_packed_gemm and w.out_features < PACKED_GEMM_MIN_N:
                logger.info(
                    "B12X FP6: %s kept packed-B (per-GPU N=%d < %d, "
                    "full N=%d >= threshold)",
                    prefix,
                    w.out_features,
                    PACKED_GEMM_MIN_N,
                    w.out_features_unsharded,
                )
            if not w.use_packed_gemm:
                # The packed view aliases ``layer.weight``'s storage, so both
                # references are dropped to actually reclaim it.
                w.packed = torch.empty(0, dtype=torch.uint8, device=dev)
            layer.b12x_fp6_weight = w
            layer.weight.data = torch.empty(0, dtype=dt, device=dev)
            layer.weight_scale.data = torch.empty(
                0, dtype=layer.weight_scale.dtype, device=dev
            )
            # Registers torch.ops.b12x.fp6_dense_linear before vLLM compiles the
            # model; apply() goes through the opaque op so Dynamo/cudagraphs never
            # trace into the CUTE JIT machinery.
            import b12x.quantization.fp6_dense_op  # noqa: F401

            # Flat attributes for apply(): plain tensors/ints/strings trace
            # cleanly under Dynamo (no dataclass method calls in the hot path).
            layer.b12x_fp6_gemm_weight = gemm_w
            layer.b12x_fp6_scales = w.scale_storage
            layer.b12x_fp6_gscale = w.global_scale
            layer.b12x_fp6_fmt = w.fmt
            layer.b12x_fp6_act_fmt = w.act_fmt
            layer.b12x_fp6_out_features = w.out_features
            layer.b12x_fp6_in_features = w.in_features
            if self.act_fmt_overrides and act_fmt != self.default_act_fmt:
                logger.info(
                    "B12X FP6: act_fmt override %s -> %s (default %s)",
                    prefix,
                    act_fmt,
                    self.default_act_fmt,
                )
            self._warm_dense_decode(layer)

        def _warm_dense_decode(self, layer: Any) -> None:
            """Compile + first-launch every decode-m variant at load time.

            Phase 2.1 widened the (16,128) small-M dense tile from m<=4 to
            m<=16. That is safe only because each (m-bucket, N, K) kernel is
            compiled and launched here, before serving and before any
            CUDA-graph capture — never at first sight of a live token count.
            One warm-run per unique shape (the compile cache is shape-keyed).
            """
            gemm_w = layer.b12x_fp6_gemm_weight
            if not gemm_w.is_cuda:
                return
            key = (
                layer.b12x_fp6_out_features,
                layer.b12x_fp6_in_features,
                layer.b12x_fp6_fmt,
                layer.b12x_fp6_act_fmt,
                tuple(gemm_w.shape),  # packed-B vs expanded variant
            )
            if key in _DENSE_WARMED_SHAPES:
                return
            _DENSE_WARMED_SHAPES.add(key)
            for m in _DENSE_WARM_DECODE_MS:
                x = torch.zeros(
                    m,
                    layer.b12x_fp6_in_features,
                    dtype=torch.bfloat16,
                    device=gemm_w.device,
                )
                self.apply(layer, x)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        def apply(
            self, layer: Any, x: torch.Tensor, bias: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            x2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
            y = torch.ops.b12x.fp6_dense_linear(
                x2d.to(torch.bfloat16),
                layer.b12x_fp6_gemm_weight,
                layer.b12x_fp6_scales,
                layer.b12x_fp6_gscale,
                layer.b12x_fp6_fmt,
                layer.b12x_fp6_out_features,
                layer.b12x_fp6_in_features,
                layer.b12x_fp6_act_fmt,
            )
            if x.dim() > 2:
                y = y.reshape(*x.shape[:-1], y.shape[-1])
            return y + bias if bias is not None else y

    # vLLM routes a layer through the packed-aware v2 weight loader only if the
    # quant-method *class name* is in ``WEIGHT_LOADER_V2_SUPPORTED`` (linear.py).
    # Without this our ModelWeightParameter falls back to the v1 loader, which
    # can't place packed FP6 weights or the GDN tuple shard-ids (q/k/v -> (0,1,2)).
    import vllm.model_executor.layers.linear as _vllm_linear

    _register_v2 = getattr(
        _vllm_linear, "register_weight_loader_v2_supported_method", None
    )
    if _register_v2 is not None:
        _register_v2(_VllmLinearMethod)
    elif _VllmLinearMethod.__name__ not in _vllm_linear.WEIGHT_LOADER_V2_SUPPORTED:
        _vllm_linear.WEIGHT_LOADER_V2_SUPPORTED.append(_VllmLinearMethod.__name__)

    class _VllmSmallNBF16Method(UnquantizedLinearMethod):  # type: ignore[misc]
        """Unquantized bf16 linear with the decode path routed to the b12x
        small-N GEMV (``b12x::bf16_gemv_small_n``).

        Weight creation/loading is stock vLLM; only ``apply`` changes. The op
        itself falls back to ``F.linear`` for shapes the kernel does not
        cover (prefill m > 16, odd K), so this is always-correct routing,
        not a shape-dependent graph branch.
        """

        def process_weights_after_loading(self, layer: Any) -> None:
            super().process_weights_after_loading(layer)
            # Register torch.ops.b12x.bf16_gemv_small_n before model compile.
            import b12x.gemm.bf16_gemv_op  # noqa: F401
            from b12x.gemm.bf16_gemv import precompile_bf16_gemv_small_n

            # The GEMV reads through a raw cuLaunchKernel, and the
            # loader-placed weight storage faults under it (IMA at first
            # touch) while fresh allocations from the same process run clean
            # — vLLM nightly's loader/allocator places weights in memory our
            # out-of-band launches can't address. Keep a private clone
            # (~1 MB/layer) and use it for both the warm-run and apply().
            #
            # Compile + warm-run every decode-m variant NOW (load time): the
            # op must never JIT or first-launch mid-serving, where CUDA-graph
            # capture may be active.
            w = layer.weight
            if w.dim() == 2 and w.is_cuda:
                layer.b12x_gemv_weight = w.data.detach().clone().contiguous()
                precompile_bf16_gemv_small_n(
                    layer.b12x_gemv_weight, log=logger
                )

        def apply(
            self, layer: Any, x: torch.Tensor, bias: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            # The private clone (see process_weights_after_loading); never
            # point the GEMV at the loader-placed layer.weight storage.
            w = getattr(layer, "b12x_gemv_weight", None)
            if (
                w is not None
                and bias is None
                and x.dtype == torch.bfloat16
                and w.dtype == torch.bfloat16
            ):
                x2d = x.reshape(-1, x.shape[-1])
                y = torch.ops.b12x.bf16_gemv_small_n(x2d, w)
                return y.reshape(*x.shape[:-1], w.shape[0])
            return super().apply(layer, x, bias)

    try:
        from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
    except ImportError:
        # Post-refactor vLLM defines the enum in routed_experts.py.
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            FusedMoeWeightScaleSupported,
        )
    from vllm.model_executor.utils import set_weight_attrs

    class _VllmMoEMethod(FusedMoEMethodBase):  # type: ignore[misc]
        """Real vLLM FusedMoE method for B12X MX-FP6 (mirrors the dense parity).

        Registers per-expert packed FP6 params so vLLM's standard expert
        weight loader (``make_expert_params_mapping`` + ``(expert_id,
        shard_id)`` convention) places the on-disk tensors itself, then
        builds the kernel-ready :class:`FP6MoEWeights` once and runs
        ``b12x_moe_fp6`` through the proven :class:`B12XFP6MoEMethod` core.

        Routing is the framework's job in this vLLM: ``MoERunner`` calls
        ``router.select_experts`` and hands ``apply`` finished
        ``topk_weights``/``topk_ids``.
        """

        # Decode/capture token counts get persistent per-M scatter-output
        # buffers (the kernel accumulates into a caller-owned zeroed buffer
        # and refuses to allocate one during CUDA-graph capture). Covers
        # vLLM's largest default cudagraph capture size (512) — serving
        # configs that pin smaller capture lists (e.g. [1,2,4]) simply
        # create fewer buffers. Prefill Ms above this let the kernel
        # allocate per call (eager-only); if capture is somehow active at
        # an uncovered M, a buffer is still created (it lands in the
        # graph's private pool, which is exactly where per-graph buffers
        # live) rather than letting the kernel raise mid-capture.
        _OUT_BUF_MAX_M = 512

        def __init__(self, moe_config: Any, source_format: str):
            super().__init__(moe_config)
            self.source_format = source_format
            self.core: Optional[B12XFP6MoEMethod] = None
            self._out_bufs: dict[int, torch.Tensor] = {}

        def _output_for(self, x: torch.Tensor) -> Optional[torch.Tensor]:
            """Zeroed persistent ``(M, K)`` scatter buffer for small M.

            Created on first sight of each M — vLLM eagerly warm-runs every
            capture size before capturing, so the buffer normally exists
            (and ``zero_`` is the only op that lands in the graph) by
            capture time.
            """
            m = int(x.shape[0])
            if m > self._OUT_BUF_MAX_M:
                if not torch.cuda.is_current_stream_capturing():
                    return None
                # Capture at an uncovered M: allocate into the graph pool
                # (zeroing is captured, so replays re-zero) but do not
                # retain it — pool memory belongs to the graph.
                return torch.zeros(m, x.shape[1], dtype=x.dtype, device=x.device)
            buf = self._out_bufs.get(m)
            if buf is None:
                buf = torch.zeros(
                    m, x.shape[1], dtype=x.dtype, device=x.device
                )
                self._out_bufs[m] = buf
            else:
                buf.zero_()
            return buf

        def create_weights(
            self,
            layer: Any,
            num_experts: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs: Any,
        ) -> None:
            del params_dtype
            weight_loader = extra_weight_attrs.get("weight_loader")
            e = int(num_experts)
            k = int(hidden_size)
            n = int(intermediate_size_per_partition)
            if k % 32 != 0 or n % 32 != 0:
                raise ValueError(
                    f"B12X FP6 MoE requires hidden/intermediate % 32 == 0, "
                    f"got K={k}, N={n}"
                )
            if not getattr(self.moe, "is_act_and_mul", True):
                raise NotImplementedError(
                    "B12X FP6 MoE supports gated (act-and-mul) experts only"
                )

            def _param(shape: tuple[int, ...], dtype: torch.dtype, **attrs: Any):
                p = torch.nn.Parameter(
                    torch.empty(*shape, dtype=dtype), requires_grad=False
                )
                set_weight_attrs(p, {"weight_loader": weight_loader, **attrs})
                return p

            group = FusedMoeWeightScaleSupported.GROUP.value
            tensor = FusedMoeWeightScaleSupported.TENSOR.value
            # Packed FP6 codes (4 values per 3 bytes along the input dim).
            # vLLM's loader fills w13 as [gate; up] (w1 -> rows 0:N,
            # w3 -> rows N:2N); rows are reordered to the kernel's
            # [up; gate] contract in process_weights_after_loading.
            layer.register_parameter(
                "w13_weight", _param((e, 2 * n, (3 * k) // 4), torch.uint8)
            )
            layer.register_parameter(
                "w2_weight", _param((e, k, (3 * n) // 4), torch.uint8)
            )
            # Per-32 UE8M0 block scales, unswizzled as stored on disk.
            layer.register_parameter(
                "w13_weight_scale",
                _param((e, 2 * n, k // 32), torch.uint8, quant_method=group),
            )
            layer.register_parameter(
                "w2_weight_scale",
                _param((e, k, n // 32), torch.uint8, quant_method=group),
            )
            # Per-tensor global scales (unit under the pure-MX W6A6 contract;
            # validated after load). w13 keeps the w1/w3 pair per expert.
            layer.register_parameter(
                "w13_weight_scale_2",
                _param((e, 2), torch.float32, quant_method=tensor),
            )
            layer.register_parameter(
                "w2_weight_scale_2",
                _param((e,), torch.float32, quant_method=tensor),
            )
            layer.register_parameter(
                "w13_input_scale", _param((e,), torch.float32)
            )
            layer.register_parameter(
                "w2_input_scale", _param((e,), torch.float32)
            )

        def uses_weight_scale_2_pattern(self) -> bool:
            return True

        def get_fused_moe_quant_config(self, layer: Any) -> Any:
            return None  # legacy apply() path; no modular-kernel config

        def process_weights_after_loading(self, layer: Any) -> None:
            from b12x.quantization.fp6_checkpoint import weight_format_for_source
            from b12x.quantization.fp6_moe_weights import FP6MoEWeights
            from b12x.quantization.fp6_safetensors_load import _swizzle_stacked

            for name in (
                "w13_weight_scale_2",
                "w2_weight_scale_2",
                "w13_input_scale",
                "w2_input_scale",
            ):
                t = getattr(layer, name).data.float()
                if not torch.allclose(t, torch.ones_like(t), atol=1e-3):
                    raise NotImplementedError(
                        f"B12X FP6 MoE requires unit {name} "
                        "(pure-MX W6A6 contract)."
                    )

            # vLLM stores this as the string-valued MoEActivation enum
            # (str() would give "MoEActivation.SILU"; .value gives "silu").
            act = getattr(layer, "activation", None) or getattr(
                self.moe, "activation", "silu"
            )
            activation = str(getattr(act, "value", act)).lower()
            e = int(layer.w13_weight.shape[0])
            k = int(layer.w2_weight.shape[1])
            n = int(layer.w13_weight.shape[1]) // 2
            dev = layer.w13_weight.device

            # vLLM convention is [gate; up]; the kernel contract is
            # [up; gate] (and the BS1 micro path requires the physical
            # order, so reorder rows instead of using w13_layout). The swap
            # is IN PLACE on the loader-placed storage: a cat-into-fresh
            # copy doubles the routed-expert footprint (~25 GiB on a 256-
            # expert 40-layer model) and the freed originals don't reliably
            # return to the pool before the framework sizes its caches.
            def _swap_halves(t: torch.Tensor) -> torch.Tensor:
                gate = t[:, :n].clone()  # transient: half of one tensor
                t[:, :n] = t[:, n:]
                t[:, n:] = gate
                return t

            w1_fp6 = _swap_halves(layer.w13_weight.data)
            w1_scale = _swap_halves(layer.w13_weight_scale.data)
            w2_fp6 = layer.w2_weight.data
            w2_scale = layer.w2_weight_scale.data

            ones_e = torch.ones(e, dtype=torch.float32, device=dev)
            ones_1 = torch.ones(1, dtype=torch.float32, device=dev)
            weights = FP6MoEWeights(
                w1_fp6=w1_fp6,
                w1_blockscale=_swizzle_stacked(w1_scale).to(dev).contiguous(),
                w1_alphas=ones_e,
                w2_fp6=w2_fp6,
                w2_blockscale=_swizzle_stacked(w2_scale).to(dev).contiguous(),
                w2_alphas=ones_e.clone(),
                a1_gscale=ones_1,
                a2_gscale=ones_1.clone(),
                num_experts=e,
                k=k,
                n=n,
                weight_fmt=weight_format_for_source(self.source_format),
                source_format=self.source_format,
                activation=activation,
            )

            # Detach the loader params. The packed code storages live on,
            # owned by FP6MoEWeights (w1_fp6/w2_fp6 ARE the loader
            # storage); the unswizzled scale storages drop their last
            # reference at function exit (the swizzled copies are fresh).
            # empty_cache returns the per-layer transients (swap halves,
            # swizzle sources) so they don't accumulate across layers.
            for name in ("w13_weight", "w2_weight", "w13_weight_scale",
                         "w2_weight_scale"):
                p = getattr(layer, name)
                p.data = torch.empty(0, dtype=p.dtype, device=dev)
            if dev.type == "cuda":
                torch.cuda.empty_cache()

            self.core = B12XFP6MoEMethod(weights)
            # JIT-compile the fused kernels and pre-fill the (M, topk)
            # workspace cache for decode shapes NOW: nothing may compile or
            # allocate during CUDA-graph capture (full_decode_only sizes).
            if dev.type == "cuda":
                topk = int(getattr(self.moe, "experts_per_token", 0) or 8)
                router_on_input = bool(
                    getattr(layer, "apply_router_weight_on_input", False)
                )
                for m in _moe_warm_decode_ms():
                    x = torch.zeros(m, k, dtype=torch.bfloat16, device=dev)
                    ids = (
                        torch.arange(m * topk, dtype=torch.int32, device=dev)
                        .remainder(e)
                        .reshape(m, topk)
                    )
                    w = torch.full(
                        (m, topk), 1.0 / topk, dtype=torch.float32, device=dev
                    )
                    self.core.apply(
                        x,
                        w,
                        ids,
                        apply_router_weight_on_input=router_on_input,
                        output=self._output_for(x),
                    )
                torch.cuda.synchronize()
                # The first warm-run call expanded the packed FP6 codes to the
                # kernel's byte-container form and released the packed storage
                # (release_packed_weights). Return those blocks (plus the
                # expansion transients) to the driver per layer so they don't
                # accumulate and inflate vLLM's measured load memory.
                torch.cuda.empty_cache()

        def apply(
            self,
            layer: Any,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts: Any = None,
            shared_experts_input: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            # shared_experts/shared_experts_input are runner-managed: the
            # MoERunner executes them itself (NO_OVERLAP before apply, or
            # MULTI_STREAM_OVERLAPPED after) and adds shared + fused outputs.
            # They are only passed here for methods that advertise
            # mk_can_overlap_shared_experts (modular-kernel internal
            # overlap), which this method does not — so ignore them. The
            # shared-expert MLP itself binds through the dense FP6 path.
            del shared_experts, shared_experts_input
            assert self.core is not None, "process_weights_after_loading not run"
            return self.core.apply(
                x,
                topk_weights,
                topk_ids.to(torch.int32),
                apply_router_weight_on_input=bool(
                    getattr(layer, "apply_router_weight_on_input", False)
                ),
                output=self._output_for(x),
            )

    @register_quantization_config(QUANT_NAME)
    class B12XFp6Config(QuantizationConfig):  # type: ignore[misc]
        """vLLM quantization config backed by the B12X MX-FP6 static kernel."""

        def __init__(self, model_dir: Optional[str] = None) -> None:
            super().__init__()
            self._model_dir = _resolve_model_dir(model_dir)
            # Normalized names of every FP6-quantized module on disk.
            self._fp6_modules: set[str] = set()
            self._source_format = "mxfp6_default"
            self._linear_method: Optional[Any] = None
            self._loaded = False
            self._match_hits = 0
            self._match_miss: list[str] = []
            self._miss_count = 0
            self._warned_zero_overlap = False

        def __reduce__(self):
            # Spawn-safe pickling: reconstruct via the module-level factory
            # instead of letting cloudpickle serialize this dynamic class (and
            # its closure over torch internals) by value. Lazy state reloads
            # from B12X_FP6_MODEL_DIR in the receiving process.
            return (_rebuild_b12x_fp6_config, (self._model_dir,))

        @classmethod
        def get_name(cls) -> str:
            return QUANT_NAME

        @classmethod
        def get_supported_act_dtypes(cls) -> list[torch.dtype]:
            return [torch.bfloat16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 100  # SM 10.0+ (B12X targets SM120 Blackwell)

        @staticmethod
        def get_config_filenames() -> list[str]:
            return []

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "B12XFp6Config":
            # No model path here; resolved lazily from B12X_FP6_MODEL_DIR or
            # maybe_update_config. `config` is the checkpoint quant block.
            return cls()

        @classmethod
        def override_quantization_method(
            cls,
            hf_quant_cfg: dict[str, Any],
            user_quant: Optional[str],
            hf_config: Any = None,
        ) -> Optional[str]:
            if not hf_quant_cfg:
                return None
            if not is_b12x_fp6_enabled():
                return None
            method = str(hf_quant_cfg.get("quant_method", "")).lower()
            algo = str(hf_quant_cfg.get("quant_algo", "")).upper()
            if method == QUANT_METHOD and algo == QUANT_ALGO:
                return QUANT_NAME
            return None

        def maybe_update_config(
            self, model_name: str, hf_config: Any = None, revision: Any = None
        ) -> None:
            # Front process gives us the model path here; prefer the env (which
            # workers also see) but fall back to this if unset.
            if self._model_dir is None and model_name:
                self._model_dir = model_name

        # -- lazy metadata loading ----------------------------------------
        def _ensure_loaded(self) -> None:
            if self._loaded:
                return
            model_dir = _resolve_model_dir(self._model_dir)
            if not model_dir:
                raise RuntimeError(
                    "B12X FP6: model directory unknown. Set B12X_FP6_MODEL_DIR to "
                    "the FP6 checkpoint path (the launch script does this)."
                )
            from b12x.quantization.fp6_checkpoint import WEIGHT_SCALE_SUFFIX
            from b12x.quantization.fp6_safetensors_load import (
                _source_format_from_config,
            )
            from b12x.quantization.model_fp6 import SafetensorsModel

            # Read the index (cheap; no tensor data) to learn which modules are
            # FP6-quantized. A module is FP6 iff it has a ``.weight_scale`` key.
            # MoE expert weights bind through the same index: vLLM's expert
            # loader streams them into the params _VllmMoEMethod registers (no
            # separate bulk preload).
            from b12x.quantization.fp6_checkpoint import (
                activation_format_for_source,
                load_act_fmt_overrides,
            )

            st = SafetensorsModel(model_dir)
            qcfg = st.config.get("quantization_config", {}) or {}
            self._source_format = _source_format_from_config(qcfg)
            self._default_act_fmt = activation_format_for_source(self._source_format)
            self._act_fmt_overrides = load_act_fmt_overrides(qcfg)
            self._fp6_modules = {
                _norm_key(k[: -len(WEIGHT_SCALE_SUFFIX)])
                for k in st.keys()
                if k.endswith(WEIGHT_SCALE_SUFFIX)
            }
            self._linear_method = _VllmLinearMethod(
                self._source_format,
                default_act_fmt=self._default_act_fmt,
                act_fmt_overrides=self._act_fmt_overrides,
            )
            self._loaded = True
            logger.info(
                "B12X FP6: %d FP6 modules discovered in %s "
                "(source_format=%s act_fmt=%s overrides=%d)",
                len(self._fp6_modules),
                model_dir,
                self._source_format,
                self._default_act_fmt,
                len(self._act_fmt_overrides),
            )

        def _is_fp6_linear(self, prefix: str) -> bool:
            """True iff ``prefix`` maps to FP6 weights on disk (incl. fused)."""
            if _norm_key(prefix) in self._fp6_modules:
                return True
            parts = _FUSED_PARTS.get(prefix.rsplit(".", 1)[-1])
            if parts is None:
                return False
            parent = _norm_key(prefix.rsplit(".", 1)[0])
            return all(f"{parent}.{p}" in self._fp6_modules for p in parts)

        def _has_fp6_experts(self, prefix: str) -> bool:
            """True iff the FusedMoE at ``prefix`` has FP6 experts on disk.

            Expert modules live *under* the FusedMoE prefix (e.g.
            ``...mlp.experts.7.gate_proj``), so this is a prefix scan rather
            than an exact lookup. Name-agnostic by design: any per-expert
            projection naming (gate/up/down, w1/w2/w3, ...) matches.
            """
            # Post-refactor vLLM appends a ``routed_experts`` module level
            # (``...mlp.experts.routed_experts``) that does not exist in the
            # on-disk checkpoint names; strip it before the prefix scan.
            root = _norm_key(prefix).removesuffix(".routed_experts") + "."
            return any(m.startswith(root) for m in self._fp6_modules)

        def _maybe_warn_zero_overlap(self, prefix: str) -> None:
            """Loudly flag an index that shares nothing with this model.

            A stale ``B12X_FP6_MODEL_DIR`` (pointing at a different model's
            checkpoint) makes every layer silently fall back to BF16 and then
            fail later in the loader with an opaque KeyError; warn at the
            first sign instead.
            """
            if self._warned_zero_overlap or self._match_hits:
                return
            # Threshold sits above any realistic deliberately-BF16 prelude
            # (vision towers bind first and are ~110-150 modules); a stale
            # index misses on EVERY module so it blows past this quickly.
            if self._miss_count < 200:
                return
            self._warned_zero_overlap = True
            logger.warning(
                "B12X FP6: %d layers checked and NONE matched the FP6 index "
                "(%d modules from %s). B12X_FP6_MODEL_DIR likely points at a "
                "different model's checkpoint (first miss: %s).",
                self._miss_count,
                len(self._fp6_modules),
                _resolve_model_dir(self._model_dir),
                self._match_miss[0],
            )

        def get_quant_method(self, layer: Any, prefix: str):
            from vllm.model_executor.layers.linear import (
                LinearBase,
                UnquantizedLinearMethod,
            )
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                ParallelLMHead,
            )

            self._ensure_loaded()

            # The MoE refactor (vLLM #41184, June 2026) replaced the FusedMoE
            # layer class with RoutedExperts; support both spellings. On some
            # nightlies ``FusedMoE`` still imports but is a factory *function*
            # (post-#44941 alias), so only isinstance against actual classes.
            _moe_types: list[type] = []
            try:
                from vllm.model_executor.layers.fused_moe.routed_experts import (
                    RoutedExperts,
                )

                _moe_types.append(RoutedExperts)
            except ImportError:
                pass
            try:
                from vllm.model_executor.layers.fused_moe import FusedMoE

                if isinstance(FusedMoE, type):
                    _moe_types.append(FusedMoE)
            except ImportError:
                pass
            is_moe = bool(_moe_types) and isinstance(layer, tuple(_moe_types))

            if is_moe:
                if self._has_fp6_experts(prefix):
                    logger.info("B12X FP6: bound FP6 MoE %s", prefix)
                    return _VllmMoEMethod(layer.moe_config, self._source_format)
                # No B12X experts for this layer -> let vLLM use its own path.
                logger.info("B12X FP6: vLLM-native MoE fallback for %s", prefix)
                return None

            if isinstance(layer, (LinearBase, ParallelLMHead)):
                if self._is_fp6_linear(prefix):
                    self._match_hits += 1
                    if self._match_hits <= 8:
                        logger.info("B12X FP6: bound FP6 linear %s", prefix)
                    return self._linear_method
                # Deliberately-BF16 layer (vision, lm_head, norms, unquantized
                # linear_attn, ...). vLLM forbids None here, so hand back the
                # stock unquantized method — or the small-N GEMV variant for
                # narrow projections like the GDN in_proj_ba.
                self._miss_count += 1
                if len(self._match_miss) < 24:
                    self._match_miss.append(prefix)
                    logger.info("B12X FP6: BF16 fallback for %s", prefix)
                self._maybe_warn_zero_overlap(prefix)
                out_size = int(getattr(layer, "output_size", 0) or 0)
                in_size = int(getattr(layer, "input_size", 0) or 0)
                if (
                    not _bf16_gemv_disabled()
                    and 0 < out_size <= SMALL_N_GEMV_MAX_OUT
                    and in_size >= SMALL_N_GEMV_MIN_IN
                ):
                    logger.debug(
                        "B12X FP6: small-N bf16 GEMV for %s (N=%d, K=%d)",
                        prefix,
                        out_size,
                        in_size,
                    )
                    return _VllmSmallNBF16Method()
                return UnquantizedLinearMethod()

            return None  # non-linear (attention/embedding) -> vLLM default

    _CONFIG_CLS = B12XFp6Config
    _registered = True


def _dense_weight_from_loaded(
    weight_u8: torch.Tensor,
    scale_u8: torch.Tensor,
    source_format: str,
    *,
    act_fmt: str | None = None,
    out_features_unsharded: int = 0,
):
    """Build a kernel-ready ``FP6DenseWeight`` from vLLM-loaded params.

    ``weight_u8`` is the packed ``(out, 3*in/4)`` uint8 codes and ``scale_u8`` the
    unswizzled ``(out, in/32)`` UE8M0 block scales, exactly as vLLM's loader
    placed them (already on the GPU, with any fused qkv/gate_up shards merged
    along the output dim). The block scales are swizzled once into the cutlass
    layout :meth:`FP6DenseWeight.scale_view` expects; the global scale is unit
    (pure-MX W6A6), so ``dense_fp6_linear``'s ``alpha = 1/(a_gscale * 1)`` holds.

    ``act_fmt`` overrides the checkpoint-wide activation format for this module
    (pattern-based ablation); ``None`` uses :func:`activation_format_for_source`.

    ``out_features_unsharded`` is the full (pre-TP-shard) out dim used for the
    packed-B decision; 0 falls back to the per-GPU out_features.
    """
    from b12x.cute.fp4 import swizzle_block_scale
    from b12x.quantization.fp6_checkpoint import (
        activation_format_for_source,
        weight_format_for_source,
    )
    from b12x.quantization.fp6_dense_weights import FP6DenseWeight

    out_f = int(weight_u8.shape[0])
    in_f = int(weight_u8.shape[1]) * 4 // 3
    scale_storage = (
        swizzle_block_scale(scale_u8.view(torch.float8_e8m0fnu))
        .reshape(-1)
        .view(torch.uint8)
        .contiguous()
    )
    return FP6DenseWeight(
        packed=weight_u8.contiguous(),
        scale_storage=scale_storage,
        global_scale=torch.ones(1, dtype=torch.float32, device=weight_u8.device),
        out_features=out_f,
        in_features=in_f,
        fmt=weight_format_for_source(source_format),
        act_fmt=(
            act_fmt
            if act_fmt is not None
            else activation_format_for_source(source_format)
        ),
        out_features_unsharded=out_features_unsharded,
    )
