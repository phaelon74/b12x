"""Framework enablement layer for serving B12X MX-FP6 (W6A6) checkpoints.

This is the reference surface a vLLM / SGLang quantization backend calls to run a
B12X FP6 checkpoint produced by :mod:`b12x.quantization.fp6_safetensors_export`.
It is framework-agnostic (no vLLM/SGLang imports) so it can be wired into any
fork; the ``examples/`` adapters show the concrete registration.

Enablement is twofold (mirrors how the FP4 b12x path is selected framework-side):

1. **Env gate** — ``B12X_ENABLE_FP6=1`` must be set. If unset, ``should_use_b12x_fp6``
   returns ``False`` and the framework falls back to its native path.
2. **Checkpoint detection** — ``config.json`` carries
   ``quantization_config={"quant_method": "modelopt", "quant_algo": "W6A6", ...}``.

Exact B12X FP6 call contract
----------------------------

**MoE** (gated SiLU). All tensors on CUDA; ``E`` experts, hidden ``K``, intermediate
``N``; FP6 codes pack 4 values into 3 bytes (``3*dim/4``); block scales are swizzled
UE8M0 (``float8_e8m0fnu`` bytes) at ``sf_vec_size=32``:

* ``hidden_states``  ``(M, K)``      bfloat16 activations (quantized to FP6 in-kernel)
* ``topk_weights``   ``(M, topk)``   float32 router weights
* ``topk_ids``       ``(M, topk)``   int32 expert ids
* weights come from :func:`b12x.quantization.load_fp6_moe_checkpoint` as a
  :class:`~b12x.quantization.fp6_moe_weights.FP6MoEWeights`:
    - ``w1_fp6``        ``(E, 2N, 3K/4)`` uint8     (FC1, rows ``[up; gate]``)
    - ``w1_blockscale`` swizzled UE8M0 bytes (FC1)
    - ``w2_fp6``        ``(E, K, 3N/4)``  uint8     (FC2 / down)
    - ``w2_blockscale`` swizzled UE8M0 bytes (FC2)
    - ``w1_alphas``/``w2_alphas`` ``(E,)`` f32 (1.0 for pure-MX W6A6)
    - ``a1_gscale``/``a2_gscale`` ``(1,)`` f32 (1.0; activations quantized live)
* output ``(M, K)`` bfloat16.

Routing is the framework's responsibility (B12X assumes Qwen3.6's MoE mapping is
already wired). ``B12XFP6MoEMethod.apply`` consumes ``topk_ids``/``topk_weights``
and returns the routed-and-combined output.

**Dense linear** ``y = x @ W.T``:

* ``x`` ``(M, in_features)`` bfloat16 -> ``y`` ``(M, out_features)`` bfloat16
* weight from :func:`b12x.quantization.load_fp6_dense_checkpoint` as an
  :class:`~b12x.quantization.fp6_dense_weights.FP6DenseWeight`.

End-to-end flow
---------------

    pip install b12x
    python scripts/quantize_model_fp6.py --model <bf16 model> --out <fp6 model> --arch auto
    export B12X_ENABLE_FP6=1
    vllm serve <fp6 model>      # framework adapter detects W6A6 and calls B12X
"""
from __future__ import annotations

import os
from typing import Any, Optional

import torch

ENABLE_ENV = "B12X_ENABLE_FP6"
QUANT_METHOD = "modelopt"
QUANT_ALGO = "W6A6"


def is_b12x_fp6_enabled() -> bool:
    """True iff ``B12X_ENABLE_FP6`` is set to a truthy value."""
    return os.environ.get(ENABLE_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def _quant_config(config: Any) -> Optional[dict]:
    qc = (
        config.get("quantization_config")
        if isinstance(config, dict)
        else getattr(config, "quantization_config", None)
    )
    if qc is None:
        return None
    if isinstance(qc, dict):
        return qc
    if hasattr(qc, "to_dict"):
        return qc.to_dict()
    try:
        return dict(vars(qc))
    except TypeError:
        return None


def is_b12x_fp6_checkpoint(config: Any) -> bool:
    """True iff ``config`` declares a B12X FP6 (modelopt + W6A6) quantization."""
    qc = _quant_config(config)
    if not qc:
        return False
    return (
        str(qc.get("quant_method", "")).lower() == QUANT_METHOD
        and str(qc.get("quant_algo", "")).upper() == QUANT_ALGO
    )


def should_use_b12x_fp6(config: Any) -> bool:
    """Gate: env enabled AND the checkpoint is a B12X FP6 checkpoint."""
    return is_b12x_fp6_enabled() and is_b12x_fp6_checkpoint(config)


# Kernel workspaces shared across ALL MoE layers of a model (module-level,
# keyed by token count + geometry). Layers run sequentially on one stream —
# eager and inside CUDA graphs alike — so scratch reuse is safe, and every
# layer of a model has identical (E, K, N, topk) geometry. Per-layer caching
# multiplies the footprint by the layer count (~40x): fatal under vLLM's
# CUDA-graph memory estimator, which measures the retained delta per capture
# size and extrapolates it across every planned graph (seen as a 155 GiB
# "estimate" -> negative KV budget on a 40-layer 256-expert model).
#
# Sharing is sound because the workspace holds only scratch buffers and
# static *unit* input-scale expansions — the pure-MX W6A6 contract (unit
# a1/a2 gscales, validated at load) makes those identical for every layer.
_SHARED_WORKSPACE_CACHE: dict[tuple, Any] = {}


class B12XFP6MoEMethod:
    """Reference routed-MoE method backed by :func:`b12x_moe_fp6`.

    Holds one layer's :class:`FP6MoEWeights`; kernel workspaces come from the
    process-wide shared cache (see ``_SHARED_WORKSPACE_CACHE``).
    """

    def __init__(self, weights, *, input_scales_static: bool = True):
        self.weights = weights
        self.input_scales_static = input_scales_static

    def _workspace(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor):
        from b12x.integration.tp_moe import allocate_tp_moe_workspace

        w = self.weights
        key = (
            int(hidden_states.shape[0]),
            int(topk_ids.shape[1]),
            int(w.num_experts),
            int(w.k),
            int(w.n),
            hidden_states.dtype,
            hidden_states.device,
            w.activation,
            self.input_scales_static,
        )
        ws = _SHARED_WORKSPACE_CACHE.get(key)
        if ws is None:
            ws = allocate_tp_moe_workspace(
                hidden_states,
                w.a1_gscale,
                w.w1_fp6,
                w.a2_gscale,
                w.w2_fp6,
                topk_ids,
                quant_mode="w6a6",
                input_scales_static=self.input_scales_static,
                activation=w.activation,
            )
            _SHARED_WORKSPACE_CACHE[key] = ws
        return ws

    def apply(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        apply_router_weight_on_input: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the FP6 fused MoE for one layer; returns ``(M, K)`` bf16.

        ``output`` (zeroed, ``(M, K)`` bf16) is required under CUDA-graph
        capture: the kernel scatter-accumulates into it and refuses to
        allocate internally while a capture is active.
        """
        from b12x.integration.tp_moe import b12x_moe_fp6

        w = self.weights
        return b12x_moe_fp6(
            hidden_states,
            w.a1_gscale,
            w.w1_fp6,
            w.w1_blockscale,
            w.w1_alphas,
            w.a2_gscale,
            w.w2_fp6,
            w.w2_blockscale,
            w.w2_alphas,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            workspace=self._workspace(hidden_states, topk_ids),
            output=output,
            input_scales_static=self.input_scales_static,
            activation=w.activation,
            source_format=w.source_format,
            # Free the 3:4-packed codes once the kernel's byte-container
            # expansions exist (first call, i.e. the load-time warm-run):
            # keeping both copies doubles expert weight memory (~24 GiB on
            # Qwen3.6-35B-A3B). Ignored when B12X_ENABLE_FP6_MICRO is set,
            # since the micro decode path streams the packed bytes.
            release_packed_weights=True,
        )


class B12XFP6LinearMethod:
    """Reference dense-linear method backed by :func:`dense_fp6_linear`."""

    def __init__(self, weight):
        self.weight = weight

    def apply(
        self, x: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute ``y = x @ W.T`` in MX-FP6; returns ``(M, out_features)`` bf16."""
        from b12x.quantization import dense_fp6_linear

        return dense_fp6_linear(x, self.weight, out=out)


def load_b12x_fp6_moe_methods(
    model_path: str,
    *,
    activation: str = "silu",
    device: torch.device | str = "cuda",
    limit_layers: Optional[int] = None,
) -> dict[int, B12XFP6MoEMethod]:
    """Load every routed-MoE layer as ``{layer_index: B12XFP6MoEMethod}``."""
    from b12x.quantization import load_fp6_moe_checkpoint

    layers = load_fp6_moe_checkpoint(
        model_path, activation=activation, device=device, limit_layers=limit_layers
    )
    return {layer: B12XFP6MoEMethod(weights) for layer, weights in layers.items()}


def load_b12x_fp6_linear_methods(
    model_path: str,
    *,
    device: torch.device | str = "cuda",
) -> dict[str, B12XFP6LinearMethod]:
    """Load every FP6 dense linear as ``{module_name: B12XFP6LinearMethod}``."""
    from b12x.quantization import load_fp6_dense_checkpoint

    weights = load_fp6_dense_checkpoint(model_path, device=device)
    return {name: B12XFP6LinearMethod(w) for name, w in weights.items()}
