"""Reference SGLang adapter for B12X MX-FP6 (W6A6) checkpoints.

Same contract as the vLLM reference (:mod:`examples.vllm_fp6_adapter`); only the
framework glue differs. SGLang's quantization layer mirrors vLLM's
``QuantizationConfig`` / ``QuantizeMethodBase`` shape, so the wiring is analogous:

1. **Detection** — :func:`b12x_fp6_quant_config_match` (env gate + modelopt/W6A6).
2. **MoE** — call :class:`b12x.integration.fp6_serving.B12XFP6MoEMethod.apply`
   from SGLang's fused-MoE quant method.
3. **Linear** — call :class:`B12XFP6LinearMethod.apply` from SGLang's linear method.

SGLang owns routing and the Qwen3.6 module map. SGLang imports are deferred so
this file imports without SGLang installed.

Launch:
    export B12X_ENABLE_FP6=1
    python -m sglang.launch_server --model-path /path/to/qwen_fp6  # adapter registered
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
    """Detection hook for SGLang's quant-config registry."""
    return (
        is_b12x_fp6_enabled()
        and str(hf_quant_cfg.get("quant_method", "")).lower() == QUANT_METHOD
        and str(hf_quant_cfg.get("quant_algo", "")).upper() == QUANT_ALGO
    )


class B12XFp6Runtime:
    """Holds the loaded per-layer B12X methods for one model.

    Construct once at model load, then dispatch from SGLang's quant methods:
        rt = B12XFp6Runtime(model_path)
        # in the fused-MoE forward:
        out = rt.moe(layer_idx).apply(hidden, topk_weights, topk_ids)
        # in a linear forward:
        y = rt.linear(module_name).apply(x)
    """

    def __init__(
        self,
        model_path: str,
        *,
        activation: str = "silu",
        device: torch.device | str = "cuda",
    ):
        self.model_path = model_path
        self._moe = load_b12x_fp6_moe_methods(
            model_path, activation=activation, device=device
        )
        self._linear = load_b12x_fp6_linear_methods(model_path, device=device)

    def moe(self, layer_index: int) -> B12XFP6MoEMethod:
        return self._moe[layer_index]

    def linear(self, module_name: str) -> Optional[B12XFP6LinearMethod]:
        return self._linear.get(module_name)


def apply_moe(
    runtime: B12XFp6Runtime,
    layer_index: int,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool = True,
) -> torch.Tensor:
    """Reference fused-MoE forward: gate -> B12X FP6 kernel -> combined output."""
    probs = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(probs, top_k, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return runtime.moe(layer_index).apply(
        hidden_states, topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    )


def apply_linear(
    runtime: B12XFp6Runtime,
    module_name: str,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference dense-linear forward via the B12X FP6 GEMM."""
    method = runtime.linear(module_name)
    if method is None:
        raise KeyError(f"{module_name!r} is not a B12X FP6 linear")
    y = method.apply(x)
    return y + bias if bias is not None else y
