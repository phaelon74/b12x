"""Reference vLLM adapter for B12X MX-FP6 (W6A6) checkpoints.

This is *reference* wiring, not a drop-in plugin: vLLM's quantization API moves
between versions, so copy the pieces you need into your fork and adapt the base
classes. It shows the three hooks that route a B12X FP6 checkpoint to the B12X
kernel:

1. **Detection** — recognize ``quantization_config={"quant_method":"modelopt",
   "quant_algo":"W6A6"}`` plus the ``B12X_ENABLE_FP6=1`` env gate
   (:func:`b12x_fp6_quant_config_match`).
2. **MoE** — a ``FusedMoEMethodBase.apply`` that calls B12X's ``b12x_moe_fp6`` via
   :class:`b12x.integration.fp6_serving.B12XFP6MoEMethod`.
3. **Linear** — a ``LinearMethodBase.apply`` calling ``dense_fp6_linear`` via
   :class:`b12x.integration.fp6_serving.B12XFP6LinearMethod`.

vLLM keeps ownership of routing (top-k gating) and the Qwen3.6 module map; B12X
only provides the quantized matmuls. All B12X pieces below are exercised by the
test-suite; the vLLM base-class glue is illustrative (marked ``vLLM glue``).

Launch:
    export B12X_ENABLE_FP6=1
    vllm serve /path/to/qwen_fp6   # with B12XFp6Config registered

vLLM imports are deferred so this file imports without vLLM installed.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from b12x.integration.fp6_serving import (
    QUANT_ALGO,
    QUANT_METHOD,
    B12XFP6LinearMethod,
    B12XFP6MoEMethod,
    is_b12x_fp6_enabled,
    load_b12x_fp6_linear_methods,
    load_b12x_fp6_moe_methods,
)


def b12x_fp6_quant_config_match(hf_quant_cfg: dict) -> bool:
    """Detection hook: claim the checkpoint iff env-enabled and modelopt+W6A6.

    Wire into vLLM's ``QuantizationConfig.override_quantization_method`` (or the
    equivalent registry check) so B12X wins over the stock ModelOpt path.
    """
    return (
        is_b12x_fp6_enabled()
        and str(hf_quant_cfg.get("quant_method", "")).lower() == QUANT_METHOD
        and str(hf_quant_cfg.get("quant_algo", "")).upper() == QUANT_ALGO
    )


def layer_index_from_prefix(prefix: str) -> int:
    """e.g. ``model.layers.3.mlp`` -> 3 (first integer path segment)."""
    for part in prefix.split("."):
        if part.isdigit():
            return int(part)
    raise ValueError(f"no layer index in prefix {prefix!r}")


def build_b12x_fp6_vllm_config(model_path: str):
    """Construct a vLLM ``QuantizationConfig`` subclass wired to B12X (deferred import).

    Returns the *class*; vLLM instantiates it via ``from_config``. The per-layer
    B12X methods are loaded once from ``model_path`` and dispatched in
    ``get_quant_method`` by layer index (MoE) or module name (linear).
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEMethodBase
    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    # --- vLLM glue: thin method wrappers delegating to the B12X core ----------
    class _MoEMethod(FusedMoEMethodBase):
        def __init__(self, core: B12XFP6MoEMethod):
            self.core = core

        def create_weights(self, *args, **kwargs):  # weights pre-loaded by B12X
            return

        def apply(
            self,
            layer: Any,
            x: torch.Tensor,
            router_logits: torch.Tensor,
            top_k: int,
            renormalize: bool,
            **kwargs,
        ) -> torch.Tensor:
            topk_weights, topk_ids = select_experts(router_logits, top_k, renormalize)
            return self.core.apply(x, topk_weights, topk_ids.to(torch.int32))

    class _LinearMethod(LinearMethodBase):
        def __init__(self, core: B12XFP6LinearMethod):
            self.core = core

        def create_weights(self, *args, **kwargs):
            return

        def apply(
            self, layer: Any, x: torch.Tensor, bias: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            y = self.core.apply(x)
            return y + bias if bias is not None else y

    # --- the config that vLLM looks up per layer -----------------------------
    class B12XFp6Config(QuantizationConfig):
        def __init__(self) -> None:
            super().__init__()
            self.model_path = model_path
            self._moe = load_b12x_fp6_moe_methods(model_path)
            self._linear = load_b12x_fp6_linear_methods(model_path)

        @classmethod
        def get_name(cls) -> str:
            return "b12x_fp6"

        @classmethod
        def get_supported_act_dtypes(cls) -> list[torch.dtype]:
            return [torch.bfloat16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 120  # SM 12.0

        @classmethod
        def get_config_filenames(cls) -> list[str]:
            return []

        @classmethod
        def from_config(cls, config: dict) -> "B12XFp6Config":
            return cls()

        def get_quant_method(self, layer: Any, prefix: str):
            from vllm.model_executor.layers.fused_moe import FusedMoE
            from vllm.model_executor.layers.linear import LinearBase

            if isinstance(layer, FusedMoE):
                return _MoEMethod(self._moe[layer_index_from_prefix(prefix)])
            if isinstance(layer, LinearBase):
                core = self._linear.get(prefix)
                if core is not None:
                    return _LinearMethod(core)
            return None  # not quantized -> vLLM's default BF16 path

    return B12XFp6Config


def select_experts(
    router_logits: torch.Tensor, top_k: int, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal top-k gating; in practice reuse vLLM's ``FusedMoE.select_experts``."""
    probs = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(probs, top_k, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_ids
