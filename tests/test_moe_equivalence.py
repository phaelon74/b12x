"""Smoke tests for the auto-dispatched TP MoE path."""

from __future__ import annotations

import functools
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_moe import (
    GRAPH_REPLAY_TOLERANCES,
    MODEL_PATH,
    TP_RANK,
    TP_SIZE,
    allocate_layer_chain_workspace,
    ModelSpec,
    compare_graph_replay_outputs,
    capture_moe_layer_chain,
    get_scale_contract_params,
    load_expert_weight_stack,
    load_expert_weights,
    make_input_activations,
    make_multilayer_routing_case,
    make_routed_inputs,
    run_moe_layer_chain,
)


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")


def _make_spec() -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )


@functools.lru_cache(maxsize=1)
def _load_multilayer_weights() -> tuple:
    return tuple(
        load_expert_weight_stack(
            MODEL_PATH,
            _make_spec(),
            layer_start=0,
            num_layers=4,
        )
    )


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_moe_nonzero(m):
    """Validate `b12x_moe_fp4` produces non-zero output with real weights."""
    _skip_if_unavailable()

    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace,
        b12x_moe_fp4,
        clear_tp_moe_caches,
    )

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec)

    x, topk_ids, topk_weights = make_routed_inputs(spec, m, seed=99, device=device)
    workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_quant_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_quant_per_expert,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
    )

    out = b12x_moe_fp4(
        x,
        weights.w13_input_scale_quant_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_quant_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights, topk_ids,
        workspace=workspace,
        input_scales_static=True,
    )
    torch.cuda.synchronize()

    out_norm = out.float().norm().item()
    print(f"\nm={m}: out_norm={out_norm:.4f}, shape={out.shape}")
    assert out_norm > 0.01, f"m={m}: output is near-zero (norm={out_norm})"
    assert out.shape == (m, spec.hidden_size)


@pytest.mark.parametrize("m", [1, 2])
def test_moe_cuda_graph_replay_tracks_routing_updates(m):
    """Validate graph replay stays correct when routing contents change."""
    _skip_if_unavailable()

    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace,
        b12x_moe_fp4,
        clear_tp_moe_caches,
    )

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec)

    x0, topk_ids0, topk_weights0 = make_routed_inputs(spec, m, seed=123, device=device)
    x_buf = x0.clone()
    topk_ids_buf = topk_ids0.clone()
    topk_weights_buf = topk_weights0.clone()
    graph_output = torch.empty_like(x_buf)
    workspace = allocate_tp_moe_workspace(
        x_buf,
        weights.w13_input_scale_quant_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_quant_per_expert,
        weights.w2_weight,
        topk_ids_buf,
        input_scales_static=True,
    )

    # Compile once before capture; the replay check below is about routing safety.
    b12x_moe_fp4(
        x_buf,
        weights.w13_input_scale_quant_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_quant_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights_buf,
        topk_ids_buf,
        output=graph_output,
        workspace=workspace,
        input_scales_static=True,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        b12x_moe_fp4(
            x_buf,
            weights.w13_input_scale_quant_per_expert,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            weights.g1_alphas_per_expert,
            weights.w2_input_scale_quant_per_expert,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            weights.g2_alphas_per_expert,
            topk_weights_buf,
            topk_ids_buf,
            output=graph_output,
            workspace=workspace,
            input_scales_static=True,
        )

    for seed in (123, 456):
        x, topk_ids, topk_weights = make_routed_inputs(spec, m, seed=seed, device=device)
        x_buf.copy_(x)
        topk_ids_buf.copy_(topk_ids)
        topk_weights_buf.copy_(topk_weights)

        graph.replay()
        torch.cuda.synchronize()
        replay_out = graph_output.clone()

        eager_out = b12x_moe_fp4(
            x,
            weights.w13_input_scale_quant_per_expert,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            weights.g1_alphas_per_expert,
            weights.w2_input_scale_quant_per_expert,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            weights.g2_alphas_per_expert,
            topk_weights,
            topk_ids,
            workspace=workspace,
            input_scales_static=True,
        )
        torch.cuda.synchronize()

        metrics = compare_graph_replay_outputs(replay_out, eager_out)
        max_abs = metrics.max_abs
        cos = metrics.cos

        assert max_abs < 5e-4, f"m={m} seed={seed}: max_abs={max_abs:.6f}"
        assert cos > 0.9999, f"m={m} seed={seed}: cos={cos:.6f}"


@pytest.mark.parametrize("m", [1, 2, 4])
def test_moe_cuda_graph_replay_multilayer_tracks_routing_updates(m):
    """Validate a captured multi-layer graph stays correct under routing churn."""
    _skip_if_unavailable()

    from b12x.integration.tp_moe import clear_tp_moe_caches

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights_stack = list(_load_multilayer_weights())
    params_stack = [get_scale_contract_params(weights, "shared") for weights in weights_stack]
    num_layers = len(weights_stack)

    x_buf = make_input_activations(spec, m, seed=10_000 + m, device=device)
    initial_case = make_multilayer_routing_case(
        spec,
        m,
        num_layers,
        device,
        pattern="disjoint",
        seed=20_000 + m,
    )
    topk_ids_bufs = [topk_ids.clone() for topk_ids, _ in initial_case]
    topk_weights_bufs = [topk_weights.clone() for _, topk_weights in initial_case]
    graph_output_bufs = [torch.empty_like(x_buf) for _ in range(num_layers)]
    eager_output_bufs = [torch.empty_like(x_buf) for _ in range(num_layers)]
    shared_workspace = allocate_layer_chain_workspace(
        weights_stack,
        params_stack,
        x_buf,
        topk_ids_bufs,
    )

    run_moe_layer_chain(
        weights_stack,
        params_stack,
        x_buf,
        topk_ids_bufs,
        topk_weights_bufs,
        activation="silu",
        fast_math=True,
        output_buffers=graph_output_bufs,
        workspace=shared_workspace,
    )
    torch.cuda.synchronize()
    graph = capture_moe_layer_chain(
        weights_stack,
        params_stack,
        x_buf,
        topk_ids_bufs,
        topk_weights_bufs,
        activation="silu",
        fast_math=True,
        output_buffers=graph_output_bufs,
        workspace=shared_workspace,
    )

    scenario_specs = [
        ("disjoint", "disjoint", 1100),
        ("overlap", "overlap", 2200),
        ("random-a", "random", 3300),
        ("random-b", "random", 4400),
    ]
    max_abs_tol = GRAPH_REPLAY_TOLERANCES["max_abs"]
    cos_tol = GRAPH_REPLAY_TOLERANCES["cos_min"]

    for scenario_name, pattern, seed in scenario_specs:
        x_case = make_input_activations(
            spec,
            m,
            seed=30_000 + m + seed,
            device=device,
        )
        routing_case = make_multilayer_routing_case(
            spec,
            m,
            num_layers,
            device,
            pattern=pattern,
            seed=40_000 + m + seed,
        )

        x_buf.copy_(x_case)
        for layer_idx, (topk_ids, topk_weights) in enumerate(routing_case):
            topk_ids_bufs[layer_idx].copy_(topk_ids)
            topk_weights_bufs[layer_idx].copy_(topk_weights)

        graph.replay()
        torch.cuda.synchronize()

        run_moe_layer_chain(
            weights_stack,
            params_stack,
            x_buf,
            topk_ids_bufs,
            topk_weights_bufs,
            activation="silu",
            fast_math=True,
            output_buffers=eager_output_bufs,
            workspace=shared_workspace,
        )
        torch.cuda.synchronize()

        for layer_idx, (replay_out, eager_out) in enumerate(
            zip(graph_output_bufs, eager_output_bufs, strict=True)
        ):
            metrics = compare_graph_replay_outputs(replay_out, eager_out)
            max_abs = metrics.max_abs
            cos = metrics.cos

            assert (
                max_abs < max_abs_tol
            ), f"m={m} scenario={scenario_name} layer={layer_idx}: max_abs={max_abs:.6f}"
            assert (
                cos > cos_tol
            ), f"m={m} scenario={scenario_name} layer={layer_idx}: cos={cos:.6f}"


@pytest.mark.parametrize("m", [1, 4, 17, 64])
def test_moe_bitwise_determinism(m, monkeypatch):
    """Call b12x_moe_fp4 twice with identical inputs and verify bit-identical output.

    Self-contained test with synthetic weights — no model files required.
    With B12X_MOE_DETERMINISTIC=1 (opt-in; default is the fast atomic-scatter
    path), each (routed pair, intermediate
    slice) FC2 partial scatters to its own unique staging row (exactly one
    atomic add per location — order-independent values), the kernel records
    each physical row's pair index, and the host sums slices then topk slots
    in canonical order.  This must produce bit-identical results across
    calls.
    """
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")

    monkeypatch.setenv("B12X_MOE_DETERMINISTIC", "1")

    from b12x.integration.tp_moe import (
        allocate_tp_moe_workspace_pool,
        b12x_moe_fp4,
        clear_tp_moe_caches,
    )
    from tests.test_tp_moe_relu2_reference import _quantize_moe_weight_storage

    clear_tp_moe_caches()

    device = torch.device("cuda")
    torch.manual_seed(42)

    hidden_size = 4096
    intermediate_size = 1024
    num_experts = 8
    top_k = 8

    x = torch.randn(m, hidden_size, device=device, dtype=torch.bfloat16) / 10
    topk_ids = torch.stack(
        [
            (torch.arange(top_k, device=device, dtype=torch.int32) + tok) % num_experts
            for tok in range(m)
        ]
    ).contiguous()
    topk_logits = torch.randn(m, top_k, device=device, dtype=torch.float32)
    topk_weights = torch.softmax(topk_logits, dim=-1).contiguous()

    weight_scale = torch.ones(num_experts, device=device, dtype=torch.float32)
    w13 = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size,
        device=device, dtype=torch.bfloat16,
    ) / 50
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size,
        device=device, dtype=torch.bfloat16,
    ) / 50
    w13_fp4, w13_bs = _quantize_moe_weight_storage(w13, weight_scale)
    w2_fp4, w2_bs = _quantize_moe_weight_storage(w2, weight_scale)
    w13_input_scale = torch.linspace(
        0.00045, 0.00075, steps=num_experts, device=device, dtype=torch.float32,
    )
    w2_input_scale = torch.linspace(
        0.00030, 0.00055, steps=num_experts, device=device, dtype=torch.float32,
    )
    w13_alphas = torch.ones(num_experts, device=device, dtype=torch.float32)
    w2_alphas = torch.ones(num_experts, device=device, dtype=torch.float32)

    workspace = allocate_tp_moe_workspace_pool()

    def _run():
        return b12x_moe_fp4(
            x,
            1.0 / w13_input_scale,
            w13_fp4, w13_bs, w13_alphas,
            1.0 / w2_input_scale,
            w2_fp4, w2_bs, w2_alphas,
            topk_weights, topk_ids,
            workspace=workspace,
            input_scales_static=True,
        )

    out1 = _run().clone()
    torch.cuda.synchronize()
    out2 = _run().clone()
    torch.cuda.synchronize()

    mismatches = (out1 != out2).sum().item()
    total = out1.numel()
    assert mismatches == 0, (
        f"m={m}: {mismatches}/{total} elements differ between two identical calls"
    )
