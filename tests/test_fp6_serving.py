from __future__ import annotations

import json
import pathlib

import pytest
import torch

from b12x.integration.fp6_serving import (
    B12XFP6MoEMethod,
    is_b12x_fp6_checkpoint,
    is_b12x_fp6_enabled,
    load_b12x_fp6_moe_methods,
    should_use_b12x_fp6,
)
from b12x.quantization.fp6_checkpoint import build_quantization_config

safetensors_torch = pytest.importorskip("safetensors.torch")


def test_env_gate(monkeypatch) -> None:
    monkeypatch.delenv("B12X_ENABLE_FP6", raising=False)
    assert not is_b12x_fp6_enabled()
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv("B12X_ENABLE_FP6", truthy)
        assert is_b12x_fp6_enabled()
    monkeypatch.setenv("B12X_ENABLE_FP6", "0")
    assert not is_b12x_fp6_enabled()


def test_checkpoint_detection() -> None:
    good = {"quantization_config": {"quant_method": "modelopt", "quant_algo": "W6A6"}}
    assert is_b12x_fp6_checkpoint(good)
    # Wrong algo (NVFP4) / wrong method / missing -> not ours.
    assert not is_b12x_fp6_checkpoint(
        {"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}
    )
    assert not is_b12x_fp6_checkpoint(
        {"quantization_config": {"quant_method": "awq", "quant_algo": "W6A6"}}
    )
    assert not is_b12x_fp6_checkpoint({})


def test_checkpoint_detection_object_config() -> None:
    class _Cfg:
        quantization_config = {"quant_method": "modelopt", "quant_algo": "W6A6"}

    assert is_b12x_fp6_checkpoint(_Cfg())


def test_exporter_config_is_detectable() -> None:
    """The quantization_config the exporter writes must be recognized by the adapter."""
    cfg = {"quantization_config": build_quantization_config()}
    assert is_b12x_fp6_checkpoint(cfg)
    assert cfg["quantization_config"]["weight_format"] == "e2m3"
    assert cfg["quantization_config"]["activation_format"] == "e4m3"


def test_should_use_combines_gate_and_detection(monkeypatch) -> None:
    cfg = {"quantization_config": {"quant_method": "modelopt", "quant_algo": "W6A6"}}
    monkeypatch.setenv("B12X_ENABLE_FP6", "1")
    assert should_use_b12x_fp6(cfg)
    monkeypatch.setenv("B12X_ENABLE_FP6", "0")
    assert not should_use_b12x_fp6(cfg)  # gate off
    monkeypatch.setenv("B12X_ENABLE_FP6", "1")
    assert not should_use_b12x_fp6({})  # not an FP6 checkpoint


def _fake_moe_ckpt(path: pathlib.Path, *, e: int, k: int, n: int) -> None:
    t: dict[str, torch.Tensor] = {}
    base = "model.language_model.layers.0.mlp"
    for ei in range(e):
        t[f"{base}.experts.{ei}.gate_proj.weight"] = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
        t[f"{base}.experts.{ei}.up_proj.weight"] = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
        t[f"{base}.experts.{ei}.down_proj.weight"] = torch.randn(k, n, dtype=torch.bfloat16) * 0.1
    t[f"{base}.gate.weight"] = torch.randn(e, k, dtype=torch.bfloat16)
    path.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(
        {k2: v.contiguous() for k2, v in t.items()}, str(path / "model.safetensors")
    )
    (path / "config.json").write_text(json.dumps({"model_type": "test"}))


def test_load_moe_methods_cpu(tmp_path) -> None:
    """Export -> load_b12x_fp6_moe_methods returns per-layer methods (no GPU)."""
    from b12x.quantization.fp6_safetensors_export import (
        export_moe_model_to_fp6_safetensors,
    )

    _fake_moe_ckpt(tmp_path / "m", e=2, k=64, n=32)
    out = tmp_path / "fp6"
    export_moe_model_to_fp6_safetensors(
        tmp_path / "m", out, device="cpu", use_gpu=False, verbose=False
    )
    methods = load_b12x_fp6_moe_methods(str(out), device="cpu")
    assert set(methods) == {0}
    assert isinstance(methods[0], B12XFP6MoEMethod)
    assert methods[0].weights.num_experts == 2
