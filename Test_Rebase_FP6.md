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

**Phase 2 (install + KLD + serve + benchmarks)** starts at
[Section 11](#11-phase-2--what-to-do-now).

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
# Synthetic MoE artifact end-to-end (no model download needed; safetensors only)
python scripts/quantize_moe_fp6.py --demo --experts 8 --k 256 --n 256 --output /tmp/demo_fp6.safetensors
python scripts/validate_fp6_moe_artifact.py --artifact /tmp/demo_fp6.safetensors --tokens 8,128 --reference

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

## 11. Phase 2 — what to do now

Phase 1 is green.  The FP6 vLLM shim now lives at
`sparkinfer/integration/vllm/` (see `docs/mxfp6-vllm-integration.md`) and
registers itself through the `vllm.general_plugins` entry point in
sparkinfer's own `pyproject.toml` — same pattern as the maintainer's NVFP4
glue, no separate plugin package anymore.

**No requantization is needed.** The on-disk checkpoint contract is unchanged
(`quant_method=modelopt`, `quant_algo=W6A6`, unswizzled UE8M0 scales), so the
existing Qwen3.6-27B-FP6 and Qwen3.6-35B-A3B-FP6 checkpoints load as-is.

Order of operations (sections 12-15):

1. Replace `b12x` with `sparkinfer` in the KLD vLLM venv (Section 12)
2. KLD-score the dense and MoE models — twice each, bit-identical (Section 13)
3. Serve both models for a live smoke (Section 14)
4. Benchmark TPOT/TTFT/acceptance vs pre-rebase (Section 15)

Env-var renames vs the old plugin (legacy `B12X_*` names still work, but use
the new ones going forward):

| Old (b12x) | New (sparkinfer) |
|---|---|
| `B12X_ENABLE_FP6` | `SPARKINFER_ENABLE_FP6` |
| `B12X_FP6_MODEL_DIR` | `SPARKINFER_FP6_MODEL_DIR` |
| `B12X_MOE_DETERMINISTIC` | `SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT` |
| `B12X_MOE_WARM_MS` | `SPARKINFER_MOE_WARM_MS` |
| `B12X_DISABLE_BF16_GEMV` | `SPARKINFER_DISABLE_BF16_GEMV` |

---

## 12. Install sparkinfer into the KLD vLLM venv

```bash
# 1. Activate the KLD fork's venv
cd ~/kld-nightly-vllm && source venv/bin/activate

# 2. Uninstall the old b12x package (its entry point would double-claim
#    the checkpoint alongside the new sparkinfer_fp6 plugin)
pip uninstall -y b12x

# 3. Get the fp6-sparkinfer branch (fresh clone shown; a pull of the
#    existing ~/fp6-sparkinfer/sparkinfer-fp6 checkout works too)
cd ~
git clone https://github.com/phaelon74/b12x.git sparkinfer-fp6-serve
cd sparkinfer-fp6-serve
git checkout fp6-sparkinfer

# 4. Install editable INTO THE KLD VENV (entry-point metadata lands here)
pip install -e .

# 5. Verify the plugin registers and the entry point is visible
python -c "from sparkinfer.integration.vllm.plugin import register_sparkinfer_fp6; register_sparkinfer_fp6(); print('plugin OK')"
python -c "
from importlib.metadata import entry_points
eps = [e for e in entry_points(group='vllm.general_plugins') if e.name == 'sparkinfer_fp6']
print('entry point OK' if eps else 'ENTRY POINT MISSING')
"
```

**PASS:** both checks print OK.  If the entry point is missing, re-run
`pip install -e .` (an editable refresh alone does not rewrite entry-point
metadata).

---

## 13. KLD scoring (run each model TWICE — must be bit-identical)

KLD runs offline through `score_mode_kld.py` (no server needed).  Determinism
requirements (unchanged from pre-rebase, see old §4.0.5): eager enforce for
all models, plus the deterministic MoE combine for the MoE model only.

### 13.1 Dense — Qwen3.6-27B-FP6

```bash
cd ~/kld-nightly-vllm && source venv/bin/activate
export SPARKINFER_ENABLE_FP6=1
export SPARKINFER_FP6_MODEL_DIR=/media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P12
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1   # eager enforce — mandatory for reproducible KLD

python examples/offline_inference/score_mode_kld.py \
  --model "$SPARKINFER_FP6_MODEL_DIR" \
  --reference-logits ~/kld-nightly-vllm/kld-vllm/ref_logits_Qwen3.6-27B_ctx2048_s512 \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --context-length 2048 --stride 512 \
  --gpu-memory-utilization 0.90 --max-num-seqs 128 \
  2>&1 | tee /tmp/phase2_kld_dense_run1.log

# Run the IDENTICAL command again:
#   ... | tee /tmp/phase2_kld_dense_run2.log
```

**PASS:** Mean KLD ≈ **0.033389** (pre-rebase eager dense baseline;
band-stable ±1e-4 is acceptable given torch 2.11→2.13) AND run1 == run2 to
the last bit.  Any run-to-run drift is a bug — report it, do not average.

### 13.2 MoE — Qwen3.6-35B-A3B-FP6

Adds the deterministic-combine flag (MoE only; dense has no atomics):

```bash
export SPARKINFER_ENABLE_FP6=1
export SPARKINFER_FP6_MODEL_DIR=/media/fmodels/TheHouseOfTheDude/qwen3-6_35B-A3B_moe_fp6
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
export SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=1   # deterministic MoE combine

python examples/offline_inference/score_mode_kld.py \
  --model "$SPARKINFER_FP6_MODEL_DIR" \
  --reference-logits /media/fmodels/kld-refs/qwen3.6-35b-a3b_ctx2048_s512 \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --context-length 2048 --stride 512 \
  --gpu-memory-utilization 0.90 --max-num-seqs 128 \
  2>&1 | tee /tmp/phase2_kld_moe_run1.log

# Run the IDENTICAL command again -> /tmp/phase2_kld_moe_run2.log
```

**PASS:** Mean KLD ≈ **0.015043** (pre-rebase eager MoE baseline) AND
run1 == run2 to the last bit.

Adjust `--model` / `--reference-logits` paths to wherever the checkpoints and
pinned eager reference logits actually live on the rig; the flags themselves
must match the pre-rebase scoring command exactly.

---

## 14. Serve the models

Serving is done through the two launch scripts (updated for the sparkinfer
rebase — they now export `SPARKINFER_ENABLE_FP6` / `SPARKINFER_FP6_MODEL_DIR`,
pass `--quantization sparkinfer_fp6`, unset the legacy `B12X_*` names, and
unset the KLD-only vars `TORCH_COMPILE_DISABLE` /
`SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT`):

- `VLLM-Launch_Scripts/qwen3.6-27b-fp6.sh`
- `VLLM-Launch_Scripts/qwen3.6-35b-a3b-fp6.sh`

Copy the current versions to the rig and run them from the serving venv
(`~/kld-nightly-vllm && source venv/bin/activate`). No FP6 env setup is needed
beforehand — the scripts set everything, and stale vars from a KLD shell
cannot leak in.

### 14.1 Dense — Qwen3.6-27B-FP6, TP=1 with MTP

```bash
MODEL_DIR=/media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P12 \
MTP_SPEC='{"method":"qwen3_next_mtp","num_speculative_tokens":4}' \
./qwen3.6-27b-fp6.sh API-KEY-HERE
```

`MODEL_DIR` must point at the checkpoint you scored in Section 13.
`MTP_SPEC` with k=4 matches the pre-rebase benchmark baseline (the script's
default is k=2); the script derives the cudagraph capture sizes from it
automatically.

### 14.2 Dense TP=2 variant

```bash
CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 \
MODEL_DIR=/media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P12 \
MTP_SPEC='{"method":"qwen3_next_mtp","num_speculative_tokens":4}' \
./qwen3.6-27b-fp6.sh API-KEY-HERE
```

The script adds `--disable-custom-all-reduce` automatically whenever
`TP_SIZE > 1` (Blackwell sm_120 custom all-reduce crashes during graph
capture, so this is mandatory).

### 14.3 MoE — Qwen3.6-35B-A3B-FP6

```bash
MODEL_DIR=/media/fmodels/TheHouseOfTheDude/qwen3-6_35B-A3B_moe_fp6 \
MTP_SPEC='{"method":"qwen3_next_mtp","num_speculative_tokens":4}' \
./qwen3.6-35b-a3b-fp6.sh API-KEY-HERE
```

For a TP=2 MoE serve, prefix with `CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2` as in
14.2. Unlike the old b12x binding (which loaded whole experts from the sidecar
and was TP=1-only), the sparkinfer shim registers per-partition-sized expert
weights (`intermediate_size_per_partition`) and lets vLLM's standard FusedMoE
weight loader shard w13 along the intermediate (row) dim and w2 along its
packed input dim, with the usual TP all-reduce combining the partial sums.
Shard sizes must stay divisible by 32 (intermediate 512 → TP=2/4/8 all fine).
Expect sub-linear latency gains at BS1 — the per-rank expert GEMMs are tiny,
so router/DeltaNet/all-reduce overhead dominates; TP mainly buys throughput
under concurrency, not single-stream latency.

**PASS (all serves):** server reaches "Application startup complete"; startup
logs show `SparkInfer FP6: N FP6 modules discovered ...` and
`SparkInfer FP6: bound FP6 linear/MoE ...` lines; a test completion returns
coherent text:

```bash
curl -s http://localhost:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer API-KEY-HERE' \
  -d '{"model":"Qwen3.6-27B-FP6-W6A6","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```

---

## 15. Serve benchmarks (compare vs pre-rebase)

With each server from Section 14 running, benchmark from a second shell.
The server is API-key protected (the launch scripts pass `--api-key`), and the
bench client sends the key as a Bearer token from the `OPENAI_API_KEY` env var
— without it every request comes back 401 and the benchmark reports garbage:

```bash
export OPENAI_API_KEY=API-KEY-HERE   # same key passed to the launch script

vllm bench serve --base-url http://localhost:8001 \
  --model Qwen3.6-27B-FP6-W6A6 \
  --tokenizer /media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P12 \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 8 --max-concurrency 1 --temperature 0
```

(Swap `--model` / `--tokenizer` for the MoE run.)

Pre-rebase reference numbers (dense 27B, k=4 MTP, Jul 12):

| metric | TP=1 | TP=2 |
|---|---|---|
| Output tok/s | 108.49 | 134.89 |
| Mean TPOT (ms) | 7.80 | 5.97 |
| Mean ITL (ms) | 28.28 | 20.26 |
| Acceptance rate (%) | 65.99 | 60.48 |

**PASS:** parity (±5%) with the table above; server logs at TP=2 still show
`kept packed-B` for the wide fused projections.

---

## 16. Reporting back + PR

1. Dense + MoE KLD values — both runs each; state explicitly whether the two
   runs matched bit-for-bit.
2. Serve benchmark table (TP=1/TP=2 dense, TP=2 MoE) vs the Section 15
   reference numbers.
3. Any launch flags that had to differ from this document.
4. Full console output of any failure.

Once Sections 12-15 are green, `fp6-sparkinfer` is PR-ready:

- Kernel port (Phase 1, Sections 3-9)
- `sparkinfer/integration/vllm/` shim (the files the maintainer drops into
  his private `sparkinfer/integration/` tree)
- `docs/mxfp6-vllm-integration.md` + `docs/mxfp6-w6a8.md`

Open the PR against `local-inference-lab/sparkinfer` `master`.

---

## 17. In-kernel per-row quant fix (decode-latency regression)

Profiling the Jul 23 serve benches showed decode steps ~8.5 ms slower than
the pre-rebase baseline at both TP=1 and TP=2. Root cause: the per-row
activation-scaling recipe (added during the rebase for m=1 bit-exactness)
ran as ~12 eager torch kernels around every FP6 linear on the decode path.
The fix fuses the whole recipe into `SmallMQuantKernel`
(`per_row=True`): row amax, bf16 pre-scale, unit-gs quantization and the
per-row output correction all happen in the one quant launch. Numerics are
bit-identical by construction; `SPARKINFER_DENSE_PER_ROW_IN_KERNEL=0`
restores the host chain for A/B.

**Divide-rounding follow-up (Jul 23):** bring-up of the fused kernel
uncovered that torch's CUDA f32 scalar/tensor division is not always
correctly rounded (e.g. `200704 / 2.625` lands 1 ulp high), while the
kernel's `div.rn.f32` is. The host chain now divides in f64 and casts to
f32 — provably the same bits as `div.rn.f32` — so both paths agree on every
operand. This slightly changes the pre-scale on rare boundary rows, so the
KLD constants below are RE-BASELINED by this run: record the new values,
then verify they repeat bit-exactly.

Validation sequence on the rig. Everything (unit tests, KLD, serving,
benches) runs in the ONE shared venv (`~/sparkinfer-kld-nightly/venv`).
First `git pull` the `fp6-sparkinfer` checkout, then reinstall:

```bash
cd ~/sparkinfer-kld-nightly && source venv/bin/activate
# --no-deps is MANDATORY in this venv: vLLM pins torch==2.11.0 and a plain
# install lets sparkinfer's torch>=2.12 requirement upgrade the whole stack.
uv pip install -e ~/sparkinfer-kld-nightly/fp6-sparkinfer --no-deps
```

If `--no-deps` is ever forgotten and torch gets upgraded (torch 2.13.0
appears in the install log), recover with:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
cd ~/sparkinfer-kld-nightly/kld-nightly && VLLM_USE_PRECOMPILED=1 uv pip install -e .
uv pip install -e ~/sparkinfer-kld-nightly/fp6-sparkinfer --no-deps
python -c "import torchvision; import vllm; print('stack OK')"
```

### 17.1 Unit tests

Run from the sparkinfer checkout (same venv). Use `python -m pytest` — the
venv does not ship pytest, and a bare `pytest` silently falls through to the
SYSTEM interpreter (no torch) with a confusing `ModuleNotFoundError`:

```bash
cd ~/sparkinfer-kld-nightly/fp6-sparkinfer
uv pip install pytest   # once, if not already present
python -m pytest tests/quantization/test_fp6_small_m_quant.py -v
python -m pytest tests/quantization/test_fp6_dense_weights_pipeline.py -v
```

**PASS:** all green, in particular `test_small_m_per_row_matches_host_chain`
(kernel vs host recipe, incl. the all-zero-row clamp edge) and
`test_small_m_per_row_linear_ab_bit_exact` (fused vs host path, bitwise).

### 17.2 Dense KLD — the hard gate

Re-run Section 13.1 exactly, TWICE, plus once with
`SPARKINFER_DENSE_PER_ROW_IN_KERNEL=0`. **PASS:** all three runs print the
IDENTICAL mean KLD, close to (but not necessarily exactly) the old
0.034423 — the correctly-rounded divide re-baselines the constant. Record
the new value; any run-to-run or fused-vs-fallback difference is a bug.

### 17.3 MoE KLD

Re-run Section 13.2 exactly, twice. **PASS:** both runs print the
IDENTICAL mean KLD, close to the old 0.011016 (MoE attention/dense
projections share this decode path, so this constant re-baselines too).
Record the new value.

### 17.4 Serve bench

Re-run Section 15 for dense TP=1 (bench twice, keep the warm run).
**Expected:** mean ITL drops from ~36.8 ms to ~29 ms and single-stream
output tok/s recovers to roughly the 100+ range, restoring parity with the
pre-rebase 108.5 tok/s baseline (residual gap, if any, is the nightly
vLLM/harness delta — compare ITL, not just tok/s, and note the MTP
acceptance rate, which varies with the random prompts).
