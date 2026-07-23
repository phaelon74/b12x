# Test_Rebase_FP6 — Validating the FP6 forward-port onto sparkinfer

This document validates that the FP6 (W6A6/W6A8) work forward-ported from the
old `b12x` layout onto the new **sparkinfer** codebase compiles, runs, and is
numerically correct on real hardware.

**Scope of this pass (Phase 1 — kernel validation):**

- Import/registry smoke of every ported module
- FP6 unit tests (quantization, dense GEMM, MoE setup, BF16 GEMV)
- Dense W6A8 GEMM correctness + performance
- MoE `w6a8_mx` end-to-end correctness vs a BF16 reference + performance
- Bit-determinism spot checks

**NOT covered by this pass (Phase 2 — requires the vLLM plugin rewrite):**

- KLD scoring vs BF16 (dense 0.033389 / MoE 0.015043 targets)
- `vllm serve` throughput/TPOT benchmarks
- The plugin lives on a separate branch and is rewritten *after* this pass
  confirms the kernels are healthy.

---

## 1. Prerequisites

| Requirement | Value |
|---|---|
| GPU | SM120/SM121 Blackwell (RTX PRO 6000) |
| Python | >= 3.10 |
| torch | >= 2.12.0 (upstream sparkinfer requirement — **newer than the 2.11 the FP6 branch was developed on**) |
| CUTLASS DSL | nvidia-cutlass-dsl == 4.6.0 (pinned by pyproject) |

## 2. Setup

```bash
git clone https://github.com/phaelon74/b12x.git sparkinfer-fp6
cd sparkinfer-fp6
git checkout fp6-sparkinfer

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Sanity-check the environment:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# expect: 2.12.x, RTX PRO 6000, (12, 0)
python -c "import cutlass; print(cutlass.__version__)"
# expect: 4.6.0
```

## 3. Step 1 — Import smoke (no GPU work yet)

```bash
python - <<'EOF'
import sparkinfer._lib.fp6
import sparkinfer._lib.dense_gemm
import sparkinfer._lib.dense_gemm_mxfp6
import sparkinfer.quantization.mxfp6
import sparkinfer.gemm.bf16_gemv
import sparkinfer.moe._shared.kernels.mxfp6_moe
import sparkinfer.moe._shared.kernels.w6a8.weights
from sparkinfer.moe._shared.execution import OperandEncoding, GemmEngine
assert hasattr(OperandEncoding, "FP6_E2M3")
assert hasattr(GemmEngine, "MXFP6_QMMA")
print("import smoke OK")
EOF
```

**PASS:** prints `import smoke OK` with no tracebacks.

## 4. Step 2 — FP6 unit tests

```bash
pytest tests/quantization/ -k "fp6 or calib" -v
pytest tests/gemm/test_dense_gemm_fp6.py tests/gemm/test_fp6_dense_op.py \
       tests/gemm/test_fp6_dense_w6a8.py tests/gemm/test_fp6_guards.py \
       tests/gemm/test_fp6_packed_b.py tests/gemm/test_bf16_gemv.py -v
pytest tests/moe/test_moe_fp6_setup.py tests/moe/test_fp6_micro.py -v
```

**PASS:** all tests pass. A handful of `tests/moe/test_moe_fp6_setup.py` tests
are expected **skips** (static-backend and micro `is_supported_mxfp6` gates
that were intentionally not ported). Any **failure or error** is a bug in the
port — report the full pytest output.

Key tests to watch:

- `test_fp6_dense_w6a8.py::test_w6a8_dense_linear_numeric` — first real GPU
  compile of the ported dense FP6 kernel. If the CUTLASS DSL trace fails,
  everything downstream will too.
- `test_fp6_dense_w6a8.py::test_dense_fp6_linear_deterministic` — dense path
  bit-determinism.
- `test_fp6_dense_w6a8.py::test_fused_quant_matches_unfused` — the fused-quant
  prologue must be bit-identical to the separate quantizer.

## 5. Step 3 — Regression check on non-FP6 paths

The port touched shared files (`_lib/dense_gemm.py`, `moe/fused_moe/_impl.py`,
`moe/_shared/kernels/dynamic.py`, `silu.py`, `relu2.py`, `micro.py`,
`execution.py`). Confirm upstream's own suites still pass:

```bash
pytest tests/gemm/ tests/moe/ tests/quantization/ -x -q
```

**PASS:** same pass/skip results as a clean `upstream/master` checkout. If you
see a failure, first check whether it also fails on `upstream/master` (i.e.
pre-existing) before attributing it to the FP6 port.

## 6. Step 4 — Dense GEMM correctness + performance

```bash
# Correctness gate at a small shape: the reference check dequantizes with a
# pure-Python per-block loop, which is minutes-to-hours at 5120x5120. Run the
# check small, then the real-shape sweep with --no-check (real-shape
# correctness is already covered by tests/gemm/test_fp6_dense_w6a8.py).
python benchmarks/benchmark_dense_gemm_fp6.py --m 1 16 128 --n 256 --k 256

# Perf across M sweep at a Qwen3.6-27B-like layer shape
python benchmarks/benchmark_dense_gemm_fp6.py --m 1 16 128 1024 --n 5120 --k 5120 --no-check

# Packed-B streaming smoke (3:4 packed weight expansion path)
python scripts/smoke_packed_b.py

# Small-N GEMV path
python scripts/smoke_bf16_gemv.py --n 96 --k 5120
python scripts/bench_bf16_gemv.py
```

**PASS:** built-in correctness checks pass (do NOT pass `--no-check`); no
launch failures across the M sweep. Perf expectation: at M=1 decode the FP6
GEMM should be clearly faster than a BF16 GEMM of the same shape (it was
~1.5-2x on the pre-rebase branch thanks to 6-bit weight traffic); exact
numbers will differ slightly under torch 2.12 + DSL 4.6.0 — record them.

## 7. Step 5 — MoE w6a8_mx end-to-end vs BF16 reference

This is the acceptance gate for the MoE port (there is no self-contained GPU
unit test for the full `w6a8_mx` pipeline; this script IS that test — it runs
prepare-weights + activation quant + dynamic kernel + combine and compares
against a BF16 torch reference):

```bash
# Qwen3.6-35B-A3B-like shape: 256 experts, hidden 2048, intermediate 512, topk 8
python scripts/bench_fp6_moe.py --experts 256 --k 2048 --n 512 --topk 8 \
    --tokens 1,8,128,512,4096

# Small-shape kernel bench (fast iteration if the big one fails)
python benchmarks/benchmark_moe_fp6.py --m 128 --k 2048 --n 512 --experts 64 --topk 8
```

**PASS:** relative error / cosine similarity vs the BF16 baseline in the same
band as the pre-rebase branch (cosine > 0.99 per token batch), and no CUDA
errors at any token count. The 4096-token point exercises the workspace
recycling path — watch for illegal-address errors there specifically.

## 8. Step 6 — MoE determinism spot check

The deterministic combine is opt-in and only needed for scoring:

```bash
SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=1 \
python scripts/bench_fp6_moe.py --experts 256 --k 2048 --n 512 --topk 8 --tokens 128 --iters 5
```

Run it twice. **PASS:** both runs execute without error, and the reported
error metrics are bit-identical between the two runs. Expect the combine to
be measurably slower with the flag on (1.3-4.4x on the combine alone) — that
is by design; the flag is off by default for serving.

## 9. Step 7 — Quantization tooling smoke

```bash
# Synthetic MoE artifact end-to-end (no model download needed)
python scripts/quantize_moe_fp6.py --demo --experts 8 --k 256 --n 256 --output /tmp/demo_fp6.pt
python scripts/validate_fp6_moe_artifact.py --artifact /tmp/demo_fp6.pt --tokens 8,128

# If the real models are on the rig, a dry-run costs nothing:
python scripts/quantize_model_fp6.py --model /path/to/Qwen3.6-35B-A3B --out /tmp/probe --arch auto --dry-run
```

**PASS:** validate script reports cosine above its 0.90 threshold at every
token count; dry-run prints the discovered architecture/layer plan without
errors.

## 10. Reporting back

For each step, report PASS/FAIL plus:

1. Full console output of any failure (the first failure matters most —
   later ones usually cascade).
2. The dense M-sweep timings from Step 4 and MoE timings from Step 5, so we
   can compare against the pre-rebase numbers before opening the PR.
3. `pip freeze | grep -E "torch|cutlass"` — exact versions the pass ran on.

Known risk areas (first GPU exercise of code that has only been byte-compiled
on the dev box):

- First CUTLASS DSL 4.6.0 trace of the FP6 dense and MoE kernels (was 4.x
  before; DSL version bumps have caused MLIR ICEs in the past).
- torch 2.11 -> 2.12 behavior changes in the quantizer/host-side code.
- The `w6a8_mx` dynamic-kernel launch wrapper (fake-tensor arity/dtypes were
  only validated at compile time).

## 11. What happens after this pass

1. **Phase 2 — vLLM plugin rewrite (Branch B):** rewrite the plugin against
   sparkinfer's new op APIs (`sparkinfer::` namespace, plan/bind/run facade).
2. **Phase 3 — KLD re-validation on the rig:** eager + deterministic scoring,
   same `score_mode_kld.py` command as before. Targets: dense W6A8
   **0.033389**, MoE W6A8 **0.015043**, bit-identical across repeat runs.
3. **Phase 4 — serve benchmarks:** TPOT/TTFT/MTP-acceptance parity with the
   pre-rebase branch.
4. **PR submission** to `local-inference-lab/sparkinfer` once 1-3 are green.
