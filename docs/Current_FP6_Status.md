# B12X FP6 — Current Status (July 2026)

Handoff document for developers joining the B12X MX-FP6 effort. Captures where the
project stands on **performance**, **accuracy**, **vLLM integration**, and **open
work**. Last full revision: primetime close-out, July 22, 2026.

## 0. Primetime close-out (July 22, 2026)

The FP6 W6A8 stack is **primetime-ready**. Final state:

- **Accuracy (eager-banked, bit-reproducible):** dense 27B 0.033389; MoE
  35B-A3B 0.015043 (FP8 band). Every accuracy lever was measured or
  re-adjudicated under deterministic eager scoring — no open "was it noise?"
  questions. Scoring recipe: §4.0.5 (MANDATORY).
- **Determinism:** dense is bit-deterministic under eager scoring; MoE is
  bit-deterministic with the opt-in `B12X_MOE_DETERMINISTIC=1` combine
  (§4.0.6 Phase C). Production serving is untouched by either.
- **Serving (Jul 22, MTP k=4, BS1):** dense 100 tok/s TP1 / 135 TP2 @1K ctx;
  MoE 272 tok/s TP1 / 279 TP2 @1K ctx. Full context sweep in §4.0.6.
- **Final defaults:** per-row activation scaling ON; fused quant OFF (~10%
  slower, kept as gate); MoE deterministic combine OFF (opt-in for scoring);
  persistent scratch ON. Env reference: §3.4.
- **Deferred:** Phase 4.2 W6A8 MoE micro-kernel — the fused kernel already
  exceeds targets by ~40% and MoE decode is no longer expert-GEMM-bound at
  BS1; estimated end-to-end gain is single-digit percent.
- Investigation-only tooling (dense self-check/trace hooks, the
  zero-output diagnostic, the NO-GO'd Hadamard rotation module) has been
  removed from the tree; this document remains the historical record.

---

## 1. Project goal

Build a **fully functional Phase-A FP6 kernel** inside B12X that can serve real
models through vLLM with:

- **Native FP6 speeds** comparable to FP4/FP8 serving paths (not a dequant-to-BF16 fallback).
- **Accuracy between FP4 and FP8** on downstream KLD (target band roughly **0.016–0.020** for dense; MoE **0.015043 eager-banked — in the FP8 band**; dense settled at 0.033389, the measured W6A8 dense floor — see §4.2).
- **Model-agnostic** MoE and dense paths (Qwen3.6 family first; architecture discovery in the quant script).

Primary reference models:

| Model | Role | Notes |
|-------|------|-------|
| **Qwen3.6-27B** (dense) | Dense FP6 baseline | `TheHouseOfTheDude/Qwen3.6-27B-FP6` on HF |
| **Qwen3.6-35B-A3B** (MoE) | MoE FP6 target | `TheHouseOfTheDude/Qwen3.6-35B-A3B-FP6` |
| **Cydonia-24B-v4.3** | Non-MTP dense validation | Used to prove generality beyond MTP |

Hardware target: **4× NVIDIA RTX PRO 6000 Blackwell** (`sm_120` / `sm_120a`), CUDA 13.x, torch 2.11+cu130, `nvidia-cutlass-dsl` 4.5.0+.

---

## 2. Terminology: weights on disk vs runtime kernel

There is **no W6A16** format in this stack. Use this mapping:

| Layer | What it means in B12X |
|-------|----------------------|
| **On-disk checkpoint** | **W6 weights** — E2M3 packed MX-FP6 codes (`uint8`, 3:4 packed layout) + UE8M0 block scales. Stored under a **ModelOpt-mirror `W6A6` schema** in `config.json` (`quant_method=modelopt`, `quant_algo=W6A6`). Activations are **not** stored; `input_scale` is a placeholder scalar (1.0). |
| **Runtime dense kernel** | **W6A6 or W6A8** — E2M3 weights (from disk) + activations quantized live each forward. Activation format is taken from `quantization_config.activation_format` (`e2m3`/`e3m2` for W6A6, **`e4m3` for W6A8**). Checkpoints exported with `--source-format mxfp6_w6a8` (script default) run W6A8 on dense as of Phase 1.1. |
| **Runtime MoE kernel (default export)** | **W6A8** — E2M3 weights (same on-disk bytes) + **E4M3 (FP8) activations** quantized live. Selected via `--source-format mxfp6_w6a8` at export (current script default). |
| **Runtime MoE kernel (legacy)** | **W6A6** — E2M3 weights + E3M2 activations (`mxfp6_default`). |

**Important:** `mxfp6_w6a8` vs `mxfp6_e2m3` at export time produces **identical weight bytes** for dense; only `config.json` metadata (`activation_format`) differs. Dense runtime **honors** `activation_format` (Phase 1.1): `e4m3` → W6A8 MMA, otherwise W6A6.

Weight format has been **E2M3 on disk since day one** (May 29, 2026). The MoE accuracy work changed **activation format only** (E3M2 → E2M3 → E4M3), not weight codes.

---

## 3. Performance status

### 3.1 Measured throughput (decode, single-GPU unless noted)

| Model / path | Throughput | Conditions |
|--------------|------------|------------|
| **MoE Qwen3.6-35B-A3B FP6** | **~190 tok/s** | BS1 decode, W6A8 activations, full B12X MoE kernel |
| **Dense Qwen3.6-27B FP6 TP=1** | **108–128 tok/s** | MTP k=4, 1K ctx, greedy; Phase 2 tile+warmup (128 tok/s best) |
| **Dense Qwen3.6-27B FP6 TP=2** | **135 tok/s** | MTP k=4, 1K ctx, greedy; Phase 3.1 packed-B fix + NCCL all-reduce |
| **Dense target** | **120–140 tok/s** | **Hit at TP=1 (128) and TP=2 (135)**; next ceiling is Phase 4.1 fusion |

MoE is **performance-competitive** for the activated-expert workload. Dense decode
now meets the 120–140 tok/s target at both TP=1 and TP=2. Phase 4 (quant→GEMM
fusion) is the next lever for pushing beyond ~135 tok/s.

### 3.2 What was built for performance

**Dense decode path**

- **Small-M activation quantizer** (`bf16_to_fp6_small_m.py`) — for `M ≤ 16`, quantizes only real rows (no 128-row pad copy); critical for BS1 decode.
- **Byte-container activations** — quantizer emits `emit="bytes"` layout so GEMM skips per-call packed→byte expansion.
- **Packed-B streaming** (`b_packed=True`) — for wide-N weights (`out_features ≥ B12X_PACKED_B_MIN_N`, default **12288**), streams 3:4-packed weights from HBM (~25% less B traffic). Narrow-N layers keep expanded byte cache.
- **Opaque `torch.library.custom_op`** (`b12x::fp6_dense_linear`) — CUDA-graph safe; avoids Dynamo tracing into JIT.
- **Small-N BF16 GEMV** (`b12x::bf16_gemv_small_n`) — routes tiny bf16 linears (e.g. GDN `in_proj_ba`, N≤1024) around slow cuBLAS WMMA tiles.

**MoE decode path**

- Fused MX-FP6 MoE kernels (`b12x_moe_fp6` in `tp_moe.py`) with static/dynamic dispatch.
- Optional **BS1 micro-kernel** (`B12X_ENABLE_FP6_MICRO`) — bypassed for `mxfp6_w6a8` (E4M3 acts use main fused path).
- **vLLM-native weight binding** — packed expert weights loaded by vLLM's standard MoE loaders; FC1 row reorder `[gate;up]` → `[up;gate]` in `process_weights_after_loading`.
- **Shared MoE workspace cache** (`_SHARED_WORKSPACE_CACHE`) — reduces per-layer allocation during warm-run / CUDA-graph capture.
- **Warm-run at load** for decode token counts `(1, 2, 4)` so kernels are compiled and workspaces exist before graph capture.

### 3.2.1 Phase 2 — dense decode (Jul 8, code done — pending rig)

Accuracy (Phase 1) is closed at the format's best achievable capability
(§4.0); performance is now the active workstream. Two changes, individually
measurable:

**Phase 2.1 — small-M tile widened to m≤16 + load-time dense warm-run.**

- `_select_default_mma_tiler_mn` (`b12x/gemm/dense.py`): the (16,128) decode
  tile now covers **m ≤ 16** (was m ≤ 4). This removes the MTP-verify tile
  cliff: the verify forward (m = 1 + num_speculative_tokens, typically 5–8)
  streams the full weight set every step and previously fell onto the
  (128,128) tile, wasting ~97% of each M-tile.
- Widening is safe ONLY because the vLLM plugin now **warm-runs every unique
  dense shape at m ∈ {1,2,4,5,8,16} at load time**
  (`_warm_dense_decode` in `vllm_plugin.py`) — every m-bucket kernel is
  compiled and first-launched before serving, so live prefill-tail token
  counts can never hit a mid-serving JIT (the reason the cap was 4).
  One warm-run per unique (N, K, fmt, act_fmt, packed) shape; the compile
  cache is shape-keyed, so this adds seconds, not minutes, to load.

**Phase 2.2 — persistent decode-quant scratch (allocation-free hot path).**

- `_small_m_quant_scratch` (`fp6_dense_weights.py`): the small-M activation
  quantizer's codes + swizzled-scale + alpha buffers are now persistent per
  (device, stream, m_pad, K) bucket instead of 3 fresh `torch.empty` per
  linear per step (hundreds of allocations/step at 27B scale). Buffers are
  consumed by the GEMM before the call returns, so cross-layer reuse is
  stream-ordered safe; buckets are keyed by stream because vLLM can run the
  shared-expert MLP (bound through this dense path) on a side stream.
- The GEMM **output** stays freshly allocated by design: it escapes to the
  framework, and during CUDA-graph capture a fresh allocation lands in the
  graph's private pool — where per-graph buffers belong. During capture the
  scratch cache is bypassed the same way (capture-pool allocations are never
  retained in the global cache); capture normally finds the load-time
  buckets already in place.
- Kill-switch for A/B isolation: `B12X_DENSE_PERSISTENT_SCRATCH=0`.

**Rig validation (Phase 2):**

```bash
# Tile selector units (CPU-safe) + the dense GEMM/CUDA-graph suite
python -m pytest tests/test_gemm_stack.py tests/test_fp6_dense_w6a8.py -v

# Tile cliff quantified per m (t16/t128 < 1.0 = the Phase 2.1 win at that m)
python scripts/bench_fp6_decode.py --hbm --ms 1 4 5 8 16

# A/B the persistent scratch in isolation
B12X_DENSE_PERSISTENT_SCRATCH=0 python scripts/bench_fp6_decode.py --hbm
B12X_DENSE_PERSISTENT_SCRATCH=1 python scripts/bench_fp6_decode.py --hbm

# End-to-end: dense serving decode tok/s (target 120-140, baseline ~107),
# then a KLD spot-check (expect BIT-IDENTICAL 0.033697 — Phase 2 changes
# scheduling and allocation only, never math).
```

Expected: 8–15% decode gain from 2.1 (largest with MTP verify enabled),
5–10% from 2.2; KLD unchanged.

**Rig results (Jul 9): both changes validated at the kernel level.**

- 35/35 tests green (incl. the new tile-selector cases).
- **Tile cliff (2.1): the (16,128) tile is 3–5x faster than (128,128) at
  every m ≤ 16** (`t16/t128` 0.21–0.35x across all four shapes). Before the
  widening, every m in 5..16 — the MTP-verify range — ran on the slow side
  of that ratio. Worst case fixed: 5120x13824 at m=5 was 0.175 ms on the
  coarse tile, now 0.037 ms.
- **Persistent scratch (2.2): −9% on the decode-path sum** (0.3294 →
  0.2999 ms over the 4 representative linears, `--hbm`), quantizer alone
  −15% (0.0370 → 0.0315 ms). Within the predicted 5–10% band.
- **KLD spot-check (Jul 9): PASS — 0.032960** (P12, same refs; scratch-off
  control 0.033018).
- **KLD determinism (Jul 13): ROOT-CAUSED — NOT A B12X BUG.**  Run-to-run
  KLD varied by ~0.001 (e.g. 0.032821 → 0.034323).  A full forensic trace
  (env-gated per-call checksum/fingerprint self-check hooks in
  `fp6_dense_weights.py`, since removed after the investigation closed)
  established:
  1. `dense_fp6_linear` is bit-deterministic in-situ: 79,380 self-checked
     calls per KLD run, zero recompute mismatches.
  2. vLLM batching/chunking is deterministic (identical M histograms and
     per-forward compositions across runs).
  3. Drift first appears in the *input* of a linear (i.e. the output of the
     residual+RMSNorm between layers) while every preceding captured value
     is bit-identical — compiled-kernel territory (`custom_ops: ['none']`).
  4. With `TORCH_COMPILE_DISABLE=1` (vLLM native ops, no Inductor, no CUDA
     graphs) and b12x fully active, KLD is **bit-identical** across runs
     (0.032782 / 0.032782).
  5. Stock **BF16** (no b12x) with compile on ALSO wobbles across runs
     (0.004264 / 0.004092) on the same vLLM build.
  **Conclusion:** the non-determinism is in the vLLM fork's torch.compile /
  Inductor stack (the build enables `combo_kernels` +
  `benchmark_combo_kernel`, i.e. timing-based kernel selection); it affects
  every model and every quant method.  FP6 amplifies its visibility because
  quantization boundaries occasionally flip a code under tiny input drift.
  **Deterministic-KLD recipe:** export `TORCH_COMPILE_DISABLE=1` for scoring
  runs (applies equally to all kernels; script unchanged).  For MoE models
  `B12X_MOE_DETERMINISTIC=1` is additionally required — see §4.0.5 for the
  full mandatory scoring recipe.  Permanent fix
  belongs in the vLLM fork: disable `benchmark_combo_kernel` /
  `combo_kernels` (stock vLLM defaults) or pin norm ops to native kernels.
  **Kept in b12x from this investigation:** (a) per-row activation global
  scale for m > 1 — each row quantizes with its own amax, making the FP6
  linear batch-composition-independent (robustness + slightly finer-grained
  scaling; decode m=1 unchanged); (b) the self-check diagnostics (zero
  overhead when env vars unset).  Per-kernel math is bit-stable (51/51
  unit tests).
- **End-to-end serving bench (Jul 11, P12, BS1, MTP k=2, vision loaded,
  `vllm bench serve` random dataset, 8 prompts x 256 out):**

| ctx | out tok/s (incl. prefill) | mean TPOT | effective decode tok/s (1000/TPOT) | mean ITL (step) | MTP accept |
|-----|--------------------------|-----------|-----------------------------------|-----------------|-----------|
| 1K   | 82.4 | 10.34 ms | **~97** | 25.1 ms | 71.8% |
| 8K   | 51.3 | 13.60 ms | ~74 | 29.5 ms | 59.0% |
| 32K  | 25.2 | 20.20 ms | ~50 | 44.7 ms | 61.1% |
| 64K  | 15.8 | 32.36 ms | ~31 | 65.0 ms | 50.6% |
| 120K | 7.4  | 43.70 ms | ~23 | 98.8 ms | 63.5% |

  Caveats recorded: (a) the headline "output tok/s" column includes prefill
  time in the denominator — TPOT is the decode-rate metric; (b) `vllm bench
  serve` no longer defaults to greedy, so MTP acceptance (and thus tok/s)
  varies with sampling — pass `--temperature 0` for comparable runs;
  (c) **this config does NOT exercise the Phase 2.1 tile widening**: MTP
  k=2 → verify m=3, which was already under the old m≤4 cap. The k=4
  (verify m=5) pass is the one that measures 2.1 end-to-end; (d) the ~107
  tok/s baseline predates this bench methodology and is not directly
  comparable — no same-methodology pre-P2 number exists.
- **Greedy A/B: k=2 vs k=4 at 1K ctx (Jul 12, `--temperature 0`):**

| metric | k=2 (verify m=3, old tile) | k=4 (verify m=5, Phase 2.1 tile) | delta |
|---|---|---|---|
| TPOT | 9.18 ms | **7.81 ms** | **−15%** |
| Effective decode (1000/TPOT) | ~109 tok/s | **~128 tok/s** | **+17%** |
| Acceptance length | 2.64 | **3.64** | +38% |
| Acceptance rate | 81.94% | 65.91% | −16 pp (expected: more positions to fail) |
| ITL (step time) | 24.07 ms | **28.26 ms** | +17% (verify step is heavier) |
| Output tok/s (incl. prefill) | 100.28 | **110.10** | +10% |

  Read: the verify step is slower (4 draft tokens vs 2 → more work per
  step → ITL up), but each step yields 38% more accepted tokens, so the
  per-token cost TPOT drops 15%. **128 tok/s at greedy, 1K ctx, inside the
  120–140 target band.** This is the Phase 2.1 tile widening paying off:
  verify m=5 now runs on the (16,128) tile instead of (128,128), and the
  kernel bench confirmed a 4x GEMM speedup at that shape.

- Decode-step arithmetic at 1K ctx: ITL ~24–28 ms/step ≈ 64 layers ×
  ~0.31 ms of dense FP6 linear time (kernel bench sum) + draft/verify
  overhead — the step is **dense-linear-bound, and each linear is
  quant+launch-bound** (fp6 M=1 ~0.075 ms vs bf16 ~0.025 ms on narrow
  shapes; the GEMM itself is only ~0.02 ms). Pushing beyond 128 tok/s at
  k=4 requires Phase 4.1 (quant→GEMM prologue fusion), not further
  Phase 2-style scheduling.

### 3.2.2 Phase 3 — TP-aware packed-B threshold (Jul 12, **validated**)

**Problem:** At TP=2, vLLM shards weights along the output dimension.  Fused
`gate_up_proj` (full N=13824) becomes 6912 per GPU — below the
`PACKED_GEMM_MIN_N=12288` threshold.  `use_packed_gemm` flips to the expanded
path, which is 33% heavier in weight VRAM *and* slower on wide-N shapes.
This is the primary cause of TP=2 ≤ TP=1 regression on dense decode.

**Fix (Phase 3.1):** `FP6DenseWeight` now carries `out_features_unsharded`
(the full, pre-shard N).  `use_packed_gemm` compares the unsharded N against
the threshold instead of the per-GPU N.  The vLLM plugin threads
`output_size` (available in `create_weights`) into the weight at bind time.

Files changed:
- `b12x/quantization/fp6_dense_weights.py`: new `out_features_unsharded`
  field on `FP6DenseWeight`, `use_packed_gemm` uses it, `to()` preserves it.
- `b12x/integration/vllm_plugin.py`: `create_weights` stashes
  `layer.b12x_out_features_unsharded`, `process_weights_after_loading` passes
  it through `_dense_weight_from_loaded`, info log when the fix activates.
- `tests/test_fp6_dense_w6a8.py`: 6 new `TestPackedGemmTPAware` tests
  (TP=1 packed, TP=1 expanded, TP=2 packed-stays, TP=2 expanded-stays,
  fallback-zero, `to()` preservation).

**Rig validation (Phase 3.1):**

```bash
# 1. Unit tests (no GPU needed for the TP-aware tests, GPU for the rest)
python -m pytest tests/test_fp6_dense_w6a8.py -v

# 2. KLD spot-check — must be band-stable vs Phase 2 (0.032960-0.033018)
#    (Phase 3.1 only changes the packed-B decision at TP>1; at TP=1 the
#    code path is identical, so KLD MUST be unchanged)
python examples/offline_inference/score_mode_kld.py \
  --model /media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P12 \
  --reference-model /media/fmodels/Qwen/Qwen3.6-27B \
  --max-tokens 200 --temperature 0

# 3. TP=1 decode benchmark (regression check — should match Phase 2 numbers)
#    Launch server with TP=1 as before (Phase 2 config), then:
vllm bench serve --base-url http://localhost:8001 \
  --model Qwen3.6-27B-FP6-W6A6 \
  --tokenizer /media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P12 \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 8 --max-concurrency 1 --temperature 0

# 4. TP=2 decode benchmark (THE validation — should improve vs pre-fix TP=2)
#    Launch with TP=2.  Serve script auto-adds --disable-custom-all-reduce at
#    TP>1 (vLLM custom all-reduce crashes on Blackwell sm_120 during graph
#    capture; NCCL fallback is ~1-3% slower at TP=2).
CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 \
  MODEL_DIR=/media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P12 \
  MTP_SPEC='{"method":"qwen3_next_mtp","num_speculative_tokens":4}' \
  CUDAGRAPH_CAPTURE_SIZES='[1,2,3,4,5,8]' \
  ./qwen3.6-27b-fp6.sh

#    Then benchmark:
vllm bench serve --base-url http://localhost:8001 \
  --model Qwen3.6-27B-FP6-W6A6 \
  --tokenizer /media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P12 \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 8 --max-concurrency 1 --temperature 0

# Expected: TP=2 TPOT drops (more tok/s), matching or exceeding TP=1;
# server logs should show "kept packed-B" for gate_up_proj layers.
```

**Phase 3.1 rig results (Jul 12, validated):**

Unit tests: 12/12 passed. KLD spot-check: **0.032943** (band-stable vs Phase 2).

| metric | TP=1 k=4 (baseline) | TP=2 k=4 (Phase 3.1) | delta |
|---|---|---|---|
| Output tok/s | 108.49 | **134.89** | **+24%** |
| Mean TPOT (ms) | 7.80 | **5.97** | **-23%** |
| Mean ITL (ms) | 28.28 | **20.26** | **-28%** |
| Acceptance rate (%) | 65.99 | 60.48 | -5.5pp |
| Acceptance length | 3.64 | 3.42 | -0.22 |

Notes:
- **TP=2 now exceeds TP=1 by 24%.** The packed-B fix is the dominant lever:
  server logs confirmed `"kept packed-B"` for `in_proj_qkvz` (N=16384→8192/GPU)
  and `qkv_proj` (N=14336→7168/GPU) across all 64 layers on both TP ranks.
- **Custom all-reduce crashes on Blackwell sm_120** during CUDA graph capture
  (`custom_all_reduce.cuh:455 'invalid argument'`). Workaround:
  `--disable-custom-all-reduce` (auto-enabled in serve script at TP>1). NCCL
  fallback costs ~1-3% at TP=2; revisit when vLLM fixes upstream.
- Acceptance rate drops 5.5pp at TP=2 — expected from per-GPU weight shard
  rounding differences; does not negate the throughput win.

### 3.2.3 Phase 4 — quant→GEMM prologue fusion (IMPLEMENTED — awaiting rig validation)

**Goal:** eliminate the separate activation-quant kernel and the HBM round-trip
for activation codes+scales on the decode hot path (m≤16). MoE already fuses
quantization into its GEMM prologue; dense still ran
`bf16_to_fp6_small_m` → `dense_gemm` as two launches.

**Implementation (Jul 12):**

- `B12X_DENSE_FUSED_QUANT=1` enables `a_bf16_fused` on m≤16 non-TMA-A tiles.
- Producer warp does a full-row amax scan from BF16, computes `fused_gs` and
  writes `alpha` into smem (existing scaffold).
- **NEW: per-K-tile BF16→FP6/FP8 quant** directly into sA + sSFA smem.
  For each K-tile (128 elements, 4 scale groups of 32):
  - 32 warp lanes each load one BF16 value from `directX_bf16`.
  - Warp-reduce `fabs` for block_max; compute UE8M0 scale via
    `fp6_block_ue8m0_exact`, then `ue8m0_output_scale_exact` for inv_scale.
  - Per-lane quantisation via `cvt_f32_to_{e4m3,e3m2,e2m3}x2` (PTX
    `cvt.rn.satfinite`), low byte extracted as the byte-container code.
  - Codes stored to sA via `st_shared_u8` (row-0 SW128 XOR is identity).
  - Scale bytes packed into a Uint32 register, then broadcast to all 128 SFA
    M-rows (512 slots) using the CUTLASS SFA tile-atom flat addressing
    `(m%32)*16 + (m//32)*4 + sg`.
- **`_DenseGemmLaunch.__call__`** now accepts `x_bf16_ptr` / `w_gscale_ptr`,
  creates cute tensors `(m, K)` BF16 and `(1,)` F32, and passes them through
  to `DenseGemmKernel.__call__`.
- **`dense_gemm()`** accepts optional `x_bf16` / `w_gscale` torch tensors and
  passes them through the `tensor_api` closure to the compiled kernel.
- **`dense_fp6_linear_expanded()`** skips the standalone
  `_quantize_matrix_fp6_bytes_small_m` when fused, allocating only dummy
  A-code/SFA buffers (for TMA descriptor validity) and passing the raw BF16
  `x` and `global_scale` to the GEMM.
- `bench_fp6_decode.py` now includes an `fp6 fused` column that runs the
  fused path side-by-side with the unfused path.
- New test `test_fused_quant_matches_unfused` verifies numeric parity between
  fused and unfused paths at m=1,4,8.

**Expected gain:** 10–20% on decode (removes quant kernel launch + activation
code/scale HBM write+read round-trip).

**Rig validation steps:**

```bash
# 1. Unit tests (numeric parity vs unfused path)
python -m pytest tests/test_fp6_dense_w6a8.py -v -k fused
python -m pytest tests/test_fp6_dense_w6a8.py tests/test_gemm_stack.py -v

# 2. Kernel micro-bench (compare fp6 M=1 vs fp6 fused column)
python scripts/bench_fp6_decode.py --hbm

# 3. KLD spot-check (must stay band-stable vs unfused 0.032943)
B12X_DENSE_FUSED_QUANT=1 python3 examples/offline_inference/score_mode_kld.py \
  --model /media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P12 \
  --reference-logits ./ref_logits_Qwen3.6-27B_ctx2048_s512 \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85

# 4. End-to-end decode (target: >135 tok/s TP=2, >128 tok/s TP=1 at k=4)
B12X_DENSE_FUSED_QUANT=1 ./qwen3.6-27b-fp6.sh
vllm bench serve --base-url http://localhost:8001 \
  --model Qwen3.6-27B-FP6-W6A6 \
  --tokenizer /media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P12 \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 8 --max-concurrency 1 --temperature 0
```

### 3.3 Performance work still open

| Area | Notes |
|------|-------|
| **Phase 4.1: quant→GEMM prologue fusion** | **CLOSED (Jul 21)** — implemented and numerically bit-identical, but ~10% slower at M=1 on all shapes; default stays unfused, gate kept |
| **Phase 4.2: W6A8 MoE micro-kernel** | **DEFERRED (Jul 22)** — fused kernel already at 272 tok/s BS1 (target was ~190); MoE decode no longer expert-GEMM-bound, est. single-digit % end-to-end |
| **vLLM custom all-reduce on Blackwell** | Upstream fix pending; launch scripts auto-add `--disable-custom-all-reduce` at TP>1 (crashes during graph capture on sm_120) |
| **Prefill benchmarking** | **DONE (Jul 22)** — context sweep 1K–32K banked in §4.0.6 Phase D |

### 3.4 Key performance environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `B12X_PACKED_B_MIN_N` | `12288` | Min `out_features` for packed weight streaming in dense GEMM |
| `B12X_DENSE_PERSISTENT_SCRATCH` | on | Set `0` to disable the Phase 2.2 persistent decode-quant scratch (A/B isolation) |
| `B12X_DENSE_PER_ROW_GS` | on | Set `0` to disable per-row activation global scale (m>1 unfused); A/B vs legacy per-tensor |
| `B12X_DENSE_FUSED_QUANT` | off | Set `1` to enable Phase 4.1 quant→GEMM prologue fusion (m≤16 decode). **Stays off: Jul 21 bench showed fused ~10% slower at M=1 on all shapes** (in-kernel amax costs more than the launch it saves); numerically vindicated (bit-identical KLD), kept as a gate only |
| `B12X_MOE_DETERMINISTIC` | off | Set `1` for the bit-deterministic MoE combine (KLD scoring only — 1.3–4.4x slower; see §4.0.5). Auto-disabled under CUDA graph capture |
| `B12X_MOE_WARM_MS` | auto | Override the MoE decode Ms warm-run before CUDA-graph capture (e.g. `1,2,4,5,8`). Default: vLLM's resolved `cudagraph_capture_sizes` (≤64) unioned with a small fallback set — no manual sync with launch scripts needed |
| `B12X_DISABLE_BF16_GEMV` | off | Set `1` to disable small-N BF16 GEMV routing (debug) |
| `B12X_ENABLE_FP6_MICRO` | off | BS1 MoE micro-kernel opt-in |
| `B12X_MOE_FORCE_A16` | off | Force A16 path in MoE (debug) |
| `B12X_FP6_ACT_FMT_OVERRIDES` | off | `pat=fmt,pat=fmt` fnmatch overrides for dense activation format (e.g. `*.linear_attn.*=e3m2`); model-agnostic; see §4.0.3 |
| `B12X_TIMING` / `VLLM_B12X_TIMING` | off | Per-kernel timing logs in `tp_moe.py` |
| `B12X_FAST_MATH` | on | Fast-math toggles in MoE dispatch |

---

## 4. Accuracy status (KLD)

KLD is measured against a **BF16 reference** on **Wikitext** (~204,700 positions) unless noted. Goal: **between FP4 and FP8** (FP8 reference ~0.0158; INT4 ~0.02–0.025).

**Reference-logit determinism (Jul 14, 2026):** scoring uses **eager mode by default**
in the KLD fork (`score_mode_kld.py`); reference and quant runs must use the
**same execution stack**. Pinned reference dirs live under
`/media/fmodels/kld-refs/` (see §4.0.6 Phase A). Historical compiled-era KLD
numbers carry ±4e-4 run noise and are not directly comparable to eager baselines.

### 4.0.5 KLD scoring requirements — MANDATORY for reproducible numbers

Anyone running `score_mode_kld.py` against the b12x FP6 kernel MUST set both
of the following, or the reported Mean KLD is not bit-reproducible and MUST
NOT be recorded or compared:

1. **Eager enforce** — `TORCH_COMPILE_DISABLE=1` (or a fork build where the
   score script defaults to eager). Required for **every** model and **every**
   quant method (BF16 included): the fork's Inductor stack selects kernels by
   compile-time *timing* (`combo_kernels` + `benchmark_combo_kernel`), so the
   compiled graph itself differs run to run.
2. **MoE deterministic combine** — `B12X_MOE_DETERMINISTIC=1` (**opt-in,
   default OFF** — the default is the fast atomic path, measured 1.3–4.4x
   faster on the MoE kernel bench),
   required **in addition to eager** whenever the model is MoE and the b12x
   fused MoE kernel is active. The fused kernel scatters FC2 partials with
   BF16 atomic adds whose arrival order is hardware-scheduled; eager mode does
   nothing about that. Dense models need only item 1 — the dense FP6 path has
   no atomics and is bit-deterministic under eager alone.

**Why only b12x FP6 needs flag 2:** stock vLLM MoE backends (Triton
`fused_moe`, Marlin, machete/INT4-INT8, CUTLASS FP8/NVFP4) compute per-expert
GEMMs into separate buffers and reduce over top-k in a **fixed order** — no
global-memory atomic accumulation — so under eager they are already
run-to-run deterministic with no extra configuration. b12x's fused kernel is
the outlier because it fuses the expert scatter into the FC2 epilogue with
atomic adds for speed; `B12X_MOE_DETERMINISTIC=1` is the switch that trades
that speed back for bit-exactness (see §4.0.6 Phase C).

**Scope of the guarantee:** run-to-run determinism — the same score command
produces the same Mean KLD, every run, to the last bit. It does not claim
batch-invariance (the same prompt embedded in a different batch composition
can differ in final bits); that caveat applies to every backend including
BF16 and is moot for score mode's fixed dataset and order.

### 4.0.6 Phase A — deterministic re-baseline (primetime, Jul 14)

**Goal:** bank new eager KLD numbers under a single reproducible harness. Code
change: `B12X_DENSE_PER_ROW_GS` env gate (default on) in
`fp6_dense_weights.py`.

**Reference logits (pinned on disk):**

| Model | BF16 source | Reference dir (use after first generate) |
|-------|-------------|------------------------------------------|
| Dense 27B | `/media/fmodels/Qwen/Qwen3.6-27B` | `/media/fmodels/kld-refs/ref_logits_Qwen3.6-27B_ctx2048_s512` |
| MoE 35B-A3B | `/media/fmodels/Qwen/Qwen3.6-35B-A3B` | `/media/fmodels/kld-refs/ref_logits_Qwen3.6-35B-A3B_ctx2048_s512` |

First `--reference-model` run writes logits into the **cwd**
(`~/kld-nightly-vllm/kld-vllm/`); move them to `kld-refs/` before subsequent
`--reference-logits` runs.

**Dense bank (record in table below):**

| Config | Env | Quant model |
|--------|-----|-------------|
| A2 per-row ON | default | `/media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P12` |
| A3 per-row OFF | `B12X_DENSE_PER_ROW_GS=0` | same |

**MoE wobble quant (pre Phase C fix):** score
`/media/fmodels/TheHouseOfTheDude/Qwen3.6-35B-A3B-FP6-P0` **twice** against MoE
refs; record both `Mean KLD` values (expect slight wobble from atomic scatter).

| Run | Dense per-row ON | Dense per-row OFF | MoE run 1 | MoE run 2 |
|-----|------------------|-------------------|-----------|-----------|
| Jul 14 (2x each) | **0.033389 / 0.033389** | 0.034211 / 0.034211 | 0.014777 | 0.014864 |

**Phase A results (Jul 14, 2026):**

- **Dense is bit-deterministic under eager scoring** — both configs reproduced
  exactly across consecutive runs. The eager scoring standard works.
- **Per-row activation scaling KEPT (Phase B.1 decision):** per-row ON scores
  0.033389 vs per-tensor 0.034211 — a real **−0.00082** accuracy win on top of
  its batch-composition invariance. `B12X_DENSE_PER_ROW_GS` stays default-on.
- **MoE atomic-scatter wobble confirmed:** 0.014777 vs 0.014864 (Δ 8.7e-5)
  across identical eager runs — the `scatter_add_v4_bf16x2` BF16 atomic combine
  is the only remaining non-determinism; Phase C fixes it.
- **MoE eager baseline ~0.0148** — deeper into the FP8 band than the compiled-era
  0.0165 number (different stack, not directly comparable; 0.0148 is the new
  eager reference point, final bank after Phase C).

**Phase B results (Jul 21, 2026):**

- **B2 — Fused quant VINDICATED:** `B12X_DENSE_FUSED_QUANT=1` scores **0.033389**,
  exact bit-match with unfused. The old compiled-era "mismatch" (0.033071 vs
  0.032923) was purely compiled-stack noise. Fused path is ready for default
  decision in Phase D (bench-gated).
- **B3 — Rotation permanently NO-GO:** P12 weight-only **0.023756** vs P14-RotSim
  **0.023876** — rotation is actually *worse* by +0.00012. Phase 1.4 closed
  definitively under trustworthy deterministic scoring.
- **B4 — MSE exponent NEUTRAL:** P0 ceil **0.023733** vs P12 MSE **0.023756** —
  delta +0.000023, effectively equal. The old compiled-era claim of MSE being
  0.00073 better was noise. Either rule is acceptable; MSE stays default for
  continuity (no code change justified by a 23-ppm delta).

**Accuracy is fully adjudicated** — every decision from the compiled era has been
re-measured under deterministic scoring. No open accuracy questions remain for dense
or weight-only paths. Remaining accuracy work: MoE deterministic combine (Phase C).

**Phase C — MoE deterministic combine (implemented):**

Root cause: the fused MoE kernels (both static and dynamic) use GPU atomic BF16
scatter-adds (`red.global.add.noftz.bf16x2`) to combine top-k expert outputs per
token. With non-deterministic warp arrival order and non-associative FP addition,
the results vary across invocations.

Fix (env-gated, `B12X_MOE_DETERMINISTIC=1`, **opt-in — default OFF** after the
Jul 21 bench quantified the cost at 1.3–4.4x on the MoE kernel; auto-disabled
during CUDA graph capture):
1. **Kernel**: the scatter epilogue is byte-identical to the baseline (same
   atomic `scatter_add_*`). Only the cached destination row changes: in
   deterministic mode each `(routed pair, intermediate slice)` FC2 **partial**
   scatters to its own **unique** staging row instead of the shared per-token
   row. The slice split matters: FC2's K dimension (the intermediate) is
   parallelized across work tiles/tasks — one slice each — so a per-pair row
   alone still receives one atomic add *per slice* in arbitrary CTA order.
   With one location per (pair, slice) there is exactly one atomic add per
   element, so addition order is irrelevant — every staging value is
   bit-deterministic. (Earlier attempts that swapped the epilogue
   instruction itself triggered a CUTLASS-DSL MLIR dominance ICE, because the
   changed trace shape corrupted the MMA-atom region threading; redirecting
   the row index at the routing-metadata cache site avoids that entirely.)
2. **Host combine (canonical order)**: the staging buffer is host-zeroed
   (`torch.zeros`) and `token_map` is prefilled with -1; in deterministic mode
   the kernel's pack phase records each physical row's routed **pair index**
   (token * num_topk + slot) instead of the token. The host then (a) sums the
   per-slice partials of each physical row in fixed slice order, (b) places
   each valid row total at its canonical (token, slot) position via
   `index_copy_` (unique indices — deterministic), and (c) sums the topk
   slots in fixed order in FP32, then casts to BF16. A plain `index_add_`
   over physical rows is NOT enough: the pack phase assigns physical rows
   through atomic counters, so row order permutes run-to-run and FP32
   addition is non-associative — the sum order must be canonicalized, not
   just made collision-free. Cost: staging is `phys_rows × slices × k` BF16,
   allocated per launch in deterministic mode only.
3. **Flag threading**: the deterministic flag is computed once per launch
   (env + capture check) and threaded explicitly into kernel compilation and
   the kernel cache key, so a det-compiled kernel is never launched without a
   staging buffer (and vice versa).

Files changed: `b12x/moe/fused/dynamic.py`, `b12x/moe/fused/static.py`,
`b12x/moe/fused/silu.py`, `b12x/moe/fused/relu2.py`,
`b12x/integration/tp_moe.py`.

Validation: `test_moe_bitwise_determinism` in `tests/test_moe_equivalence.py`
calls `b12x_moe_fp4` twice with identical inputs and asserts zero mismatches
(sets `B12X_MOE_DETERMINISTIC=1` itself — the flag is opt-in).

**Phase C sign-off (Jul 21, 2026):** determinism test 4/4 PASS (m=1/4/17/64);
end-to-end MoE KLD bit-identical across two eager scoring runs:
**0.015043 / 0.015043**. (Slightly above the pre-fix atomic wobble band
0.014777–0.014864 because the FP32 canonical combine rounds differently than
BF16 atomic accumulation — a stack change, not a regression.)

**Deterministic-mode cost (Jul 21 `bench_fp6_moe.py`, E=256 K=2048 N=512
topk=8):** det ON vs OFF — 1 tok 0.207/0.161 ms (+29%), 8 tok 0.662/0.202
(3.3x), 128 tok 3.200/0.909 (3.5x), 512 tok 3.438/0.921 (3.7x), 4096 tok
7.232/1.637 (4.4x). This cost is why the flag defaults OFF: production serving
always runs the untouched atomic path; scoring opts in per §4.0.5.

**Phase D serve revalidation (Jul 22, 2026 — PASS, Phase D closed):**
`vllm bench serve` random dataset, 8 prompts x 256 out, BS1, greedy, MTP k=4,
capture sizes incl. the m=5 verify batch. Decode tok/s = 1000 / mean TPOT:

| ctx | Dense TP1 | Dense TP2 | MoE TP1 | MoE TP2 |
|-----|-----------|-----------|---------|---------|
| 1K  | 100 | **135** | **272** | **279** |
| 4K  | 100 | 122 | 253 | 251 |
| 8K  | 87  | 118 | 213 | 242 |
| 16K | 77  | 91  | 187 | 203 |
| 32K | 72  | 81  | 142 | 146 |

- Dense TP1 100 tok/s @1K vs banked ~97 (Jul 11, k=2) — flat/slightly better;
  per-row scaling + Phases A-C cost nothing on dense decode or prefill (32K
  TTFT 4.1 s TP1 / 3.0 s TP2, ~8-11k tok/s prefill).
- Dense TP2 135 tok/s hits the ~135 target (NCCL-fallback all-reduce; the
  launch scripts now auto-add --disable-custom-all-reduce at TP>1 — the
  custom kernel crashes during graph capture on Blackwell sm_120).
- MoE far above the ~190 reference (272 @1K TP1); TP2 adds only ~3-15% decode
  (small activations + NCCL fallback) but cuts TTFT ~30% — TP1 is the
  efficient deployment for the 35B-A3B.
- MTP k=4 acceptance 60-67% (acceptance length ~3.4-3.7) on both models.

**Pre-existing compiler bug found by this test (July 21, 2026)**: the FP4 MMA
path (`moe_emit_fp4_mma` in `mxfp6_moe.py`) calls `mma_atom.set(...)`, which
produces a *new* MLIR atom value. When that happens inside a `@cute.jit`
*callee*, the caller's dynamic loops (the k-tile mainloop) cannot thread the
updated atom through their scf regions, and the IR verifier fails with
"operand does not dominate this use". It never fired before because
(a) production FP6 uses the inline `mxf8f6f4` MMA which does not call
`mma_atom.set`, and (b) the gated FP4 static kernel had never been compiled
with >1 M-tile; the determinism test's FP4 + silu + top_k=8 + m=17/64 shape
was the first. Fix: `moe_emit_fp4_mma` now takes the MMA *op descriptor*
(`self.mma_op`) and creates a **fresh `cute.make_mma_atom` per emission**
instead of chaining `set` calls on one kernel-lifetime atom. Every atom SSA
value is then local to one straight-line block, so dominance holds by
construction; the atom is trace-time metadata with zero runtime cost. The
helpers were also made plain Python (no `@cute.jit`) and the static gated MMA
tile loops aligned to `cutlass.range_constexpr` like the non-gated path.
Independent of the determinism work — it ICEd identically with
`B12X_MOE_DETERMINISTIC=0`.

Exact commands: §4.0.6.1 below.

#### 4.0.6.1 Phase A commands (rig)

Run from `~/kld-nightly-vllm/kld-vllm` with the venv active. Scoring is eager by
default (no extra flags). See primetime plan for full copy-paste blocks.

### 4.0 Locked Phase 0 baseline (July 8, 2026)

Reproducible baseline on rig HEAD `52c27d7` (Phase 0 guardrail changes included).
All later phases measure against these numbers. Exact commands in §4.0.1.

**This baseline is a measurement, NOT an acceptance.** Dense full-pipeline
**0.0400 is not shippable** — it is the honest starting point of the
full-FP6 (Profile A) serving path, which has measured 0.037–0.039 since June
(§4.2); Phase 0 changed nothing material (+0.0005 tie-break shift). The
"0.02x dense" numbers people remember are the **weight-only floor (0.0236)**
and the **Profile B full pipeline (0.029)** — Profile B keeps GDN linear_attn
in BF16, which violates the 100%-FP6 production constraint. Recovery is the
explicit job of Phase 1, with hard gates:

| Gate | Metric | Target | Mechanism | Status |
|------|--------|--------|-----------|--------|
| Phase 1.1 exit | Dense full-pipeline | **≤ 0.026** | W6A8 dense (E4M3 acts) | **MISSED — 0.033731**; escalate exhausted (§4.0.3) — bank global e4m3 |
| Phase 1.2 exit | Dense weight-only | **≤ 0.022** | MSE-optimal per-block exponent | **MISSED — 0.022881** (§4.0.4); bank mse default |
| Phase 1.3 exit | Dense full-pipeline | **≤ 0.020** | GPTQ-style offline error feedback | **SKIPPED by decision** (§4.0.5) — cannot reach gate from weight side |
| Phase 1.4 | Dense full-pipeline | sim go/no-go first | Group-32 block Hadamard rotation | **NO-GO — measured** (§4.0.5): weight-side KLD delta -0.00012; kernel work unjustified |

**Phase 1 CLOSED (Jul 8, 2026) — accuracy is at best achievable capability
for the W6A8 MX-FP6 format.** Every lever was measured or bounded: activation
formats (1.1 — global E4M3 optimal), MSE block scales (1.2 — banked as
default, −0.0007 weight-only), GPTQ (1.3 — bounded, decisive only at FP4
resolution), block Hadamard rotation (1.4 — measured no-go, error is spread
across all projections, not concentrated). **Banked production defaults:
W6A8 (E4M3 activations) + MSE per-block exponents.** The dense format floor
is **full ≈ 0.0337 = weight 0.0229 + runtime ≈ 0.0108**; MoE sits at
**0.0165, inside the FP8 band**. The original ≤0.020 dense gate is not
reachable within per-32-block UE8M0 W6A8 quantization — it would need a
format change, not better parameter selection. Focus moves to Phase 2
(performance).

| Measurement | Value | Artifact |
|-------------|-------|----------|
| Dense full-pipeline KLD (Profile A: MLP + attn + linear_attn FP6) | **0.039980** | `Qwen3.6-27B-FP6-P0` |
| Dense weight-only KLD (dequant→BF16, **same export**) | **0.023612** | `Qwen3.6-27B-FP6-P0-DequantBF16` |
| Dense runtime slice (full − weight-only, same export) | **≈0.0164** | — |
| ...of which GDN `linear_attn` runtime (vs Profile B full 0.029, June) | **≈0.011** | see §4.2 |
| MoE full-pipeline KLD (W6A8) | **0.016509** | `Qwen3.6-35B-A3B-FP6-P0` |
| Dense export rel-RMSE (`--report-error`) | **2.82–2.83%** | uniform across MLP groups |
| Decode bench, sum fp6 M=1 over 4 linears (`--hbm`) | **0.316–0.320 ms** | pre/post Phase 0 identical |
| MoE bench BS1 static | **0.159 ms** (~6289 tok/s kernel-level) | defaults E=256 K=2048 N=512 topk=8 |

Notes:

- Dense full KLD was 0.039520 immediately before the Phase 0 fmt-aware-numerator
  change and 0.039980 after — the +0.0005 is the expected power-of-two tie-break
  shift in runtime e2m3 activation codes (§7 block-scale rule; gs cancels in
  ceil-containment).
- **MoE improved 0.0228 → 0.0165** vs the June measurement (same-by-determinism
  reference logits) — fresh checkpoint + current runtime; at the FP8 band (~0.0158).
- Unit suite: 61 tests, all green (one stale expectation fixed:
  `test_load_fp6_moe_checkpoint_cpu` now expects the `mxfp6_w6a8` export default).
  Run with `python -m pytest` from the repo venv — a bare `pytest` can resolve to
  `/usr/bin/python3` (no torch) and fail collection.

#### 4.0.2 Phase 1.1 result (Jul 8, 2026) — gate MISSED

| Measurement | Value | Notes |
|-------------|-------|-------|
| Dense full-pipeline after W6A8 | **0.033731** | Same `-P0` checkpoint + same ref logits |
| Delta vs P0 baseline (0.039980) | **−0.00625** | Real recovery; not enough for ≤0.026 |
| Remaining above weight floor | **≈0.0101** (0.0337 − 0.0236) | Matches the GDN `linear_attn` runtime slice (~0.011) |
| Decode bench sum fp6 M=1 | **0.3233 ms** | vs P0 0.316–0.320; no material regression |
| Unit tests | **40/40 passed** | incl. e4m3 small-M bit-exact + W6A8 smoke |

**Interpretation:** global E4M3 activations recovered roughly the old MLP+attention
activation estimate (~0.005–0.006). The leftover ~0.010 is concentrated in GDN
`linear_attn` FP6 projections — W6A8 alone does not fix that path.

**Escalation (model-agnostic, not Qwen-specific):** serve-time
`activation_format_overrides` / `B12X_FP6_ACT_FMT_OVERRIDES` apply fnmatch
patterns against the module path (e.g. `*.linear_attn.*=e3m2`) so *any* dense
architecture can tune recurrent / linear-attn vs MLP activation formats without
re-export or architecture branches. See §4.0.3.

#### 4.0.3 Phase 1.1 escalate — per-module act_fmt ablation (rig)

Same `-P0` checkpoint + same ref logits. Env overrides win over `config.json`.
Expect log lines `B12X FP6: act_fmt override ...` for matched modules.

```bash
cd ~/kld-nightly-vllm && source venv/bin/activate
export B12X_ENABLE_FP6=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export B12X_FP6_MODEL_DIR=/media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P0

# A) linear_attn activations -> e3m2 (more range); MLP/attn stay e4m3
export B12X_FP6_ACT_FMT_OVERRIDES='*.linear_attn.*=e3m2'
python examples/offline_inference/score_mode_kld.py \
  --model "$B12X_FP6_MODEL_DIR" \
  --reference-logits ~/kld-nightly-vllm/kld-vllm/ref_logits_Qwen3.6-27B_ctx2048_s512 \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --context-length 2048 --stride 512 \
  --gpu-memory-utilization 0.90 --max-num-seqs 128 \
  2>&1 | tee /tmp/p11_kld_la_e3m2.log

# B) linear_attn -> e2m3 (same as pre-W6A8 dense acts on those modules only)
export B12X_FP6_ACT_FMT_OVERRIDES='*.linear_attn.*=e2m3'
# ... same score_mode_kld.py ... | tee /tmp/p11_kld_la_e2m3.log

# C) Control: clear overrides (global e4m3) — should reproduce 0.033731
unset B12X_FP6_ACT_FMT_OVERRIDES
```

**Ablation results (Jul 8, same `-P0` + same refs):**

| Config | Mean KLD | vs global e4m3 (0.033731) |
|--------|----------|---------------------------|
| Global e4m3 (no override) | **0.033731** | baseline (best) |
| `*.linear_attn.*=e3m2` | **0.039291** | **+0.0056 worse** |
| `*.linear_attn.*=e2m3` | **0.038316** | **+0.0046 worse** |

**Conclusion — Phase 1.1 activation levers EXHAUSTED:** E4M3 is already the
best activation format for `linear_attn` as well as MLP/attention. Narrowing
those modules to e3m2/e2m3 regresses toward the pre-W6A8 band. Keep **global
e4m3**, leave `B12X_FP6_ACT_FMT_OVERRIDES` unset. The remaining
~0.010 above the weight floor (0.0337 − 0.0236) is **not** an activation-format
problem → Phase 1.2 (MSE-optimal block scale) + 1.3 (GPTQ) on the weight path,
with linear_attn still in the FP6 export. Do not add BF16 special-cases.

#### 4.0.4 Phase 1.2 — MSE-optimal per-block exponent (code done; rig pending)

**What changed (model-agnostic, offline weights only):**
`quantize_linear_to_fp6(..., block_scale_rule="mse"|"ceil")` jointly chooses
per-32-block UE8M0 exponent **and** FP6 codes. For each block it tries the
amax-ceil containment scale and the one-finer exponent (ceil−1, may clip the
block max), keeps the lower reconstruction MSE. Never blind-floors (unlike the
reverted round-nearest experiment in §4.3). Default is **`mse`**. Runtime
activation quantizers stay ceil. Provenance is written to
`quantization_config.block_scale_rule`.

CLI: `--block-scale-rule mse|ceil` on `scripts/quantize_model_fp6.py`.

Unit tests: `tests/test_fp6_mse_block_scale.py`.

**Gate:** dense weight-only KLD **≤ 0.022** (P0 ceil floor was **0.023612**).
Also expect `--report-error` rel-RMSE to drop from ~2.82% toward ~2.0–2.5%.

**Rig validation (same refs as P0 / W6A8):**

```bash
# Unit tests (venv)
python -m pytest tests/test_fp6_mse_block_scale.py -v

# Dense Profile A re-export with MSE block scales
python scripts/quantize_model_fp6.py \
  --model /media/fmodels/Qwen/Qwen3.6-27B \
  --out   /media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P12 \
  --arch dense --source-format mxfp6_w6a8 --include-linear-attn \
  --block-scale-rule mse --report-error

python scripts/dequantize_fp6_to_bf16.py \
  --model /media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P12 \
  --out   /media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P12-DequantBF16

# Weight-only KLD (gate ≤ 0.022) + full-pipeline (informational)
# ... same score_mode_kld harness as P0, pointing at -P12 / -P12-DequantBF16
```

**Rig results (Jul 8, same refs as P0 / W6A8):**

| Metric | Target | Result |
|--------|--------|--------|
| Dense weight-only KLD (`-P12-DequantBF16`) | **≤ 0.022** | **0.022881 — MISSED** by ~0.0009 (Δ −0.00073 vs P0 0.023612) |
| Dense full-pipeline KLD (`-P12`) | informational | **0.033697** (≈ flat vs W6A8 0.033731) |

**Conclusion — bank `block_scale_rule=mse` as default:** small, real weight-floor
improvement; not enough to clear ≤0.022 and no material full-pipeline move (the
~0.011 runtime slice dominates).

#### 4.0.5 Phase 1.3 SKIPPED (decision) → Phase 1.4 Hadamard active

**Phase 1.3 (GPTQ) skipped by decision (Jul 8), implementation rolled back.**
Rationale, recorded so it is not relitigated:

1. **GPTQ's value shrinks with format resolution.** It reallocates rounding
   error via column correlations — decisive at FP4 (16 levels), marginal at
   FP6 E2M3 (64 levels) where RTN + MSE scales already sit near the format
   floor. Our own evidence agrees: MoE MSE-scale ablation "no benefit" (§4.1),
   Phase 1.2 bought only −0.0007 on the dense weight floor.
2. **Gate arithmetic:** full 0.033697 ≈ weight 0.022881 + runtime slice
   ~0.0108. Even a *perfect* weight quantizer leaves ≈ 0.011 — Gate 1.3
   (full ≤ 0.020) is unreachable from the weight side alone.
3. Practical cost was high (per-linear `(in,in)` Hessian calibration capture,
   ~70–80 GB for a 27B, plus slow sequential export) for an expected ~0.002.

GPTQ remains a *possible* later stacking lever (it composes with rotated
weights) but is not on the critical path.

**Phase 1.4 — group-32 block Hadamard rotation (ACTIVE, simulation first).**
The only remaining lever that attacks the dominant ~0.011 runtime activation
slice: rotate within each 32-element MX block (never across — global rotations
fight the group scale), `y = Σ_b (W_b H)(Hᵀ x_b)`. Outliers spread across the
block → smaller block amax → finer UE8M0 steps for both operands.

**Honesty note:** production serving needs `Hᵀ` on activations inside the
runtime quantizer (kernel work — this is why 1.4 was held back). Weights
rotate offline. The simulation harness measures the expected gain BEFORE any
kernel investment:

> **Historical note (Jul 22):** Phase 1.4 closed permanently NO-GO (§4.0.6
> Phase B.3: rotation measured WORSE by +0.00012 under eager scoring). The
> rotation module (`b12x/quantization/hadamard.py`), simulation harnesses
> (`scripts/ablate_hadamard_fp6.py`, `scripts/simulate_hadamard_dequant.py`)
> and tests were removed from the tree; the section below is kept as the
> record of what was measured and why it died.

**Go/no-go (rig):**

```bash
python -m pytest tests/test_fp6_hadamard.py -v

# Real hidden states for the activation side (dense: hooks layers.N.mlp input)
# Output is safetensors — pickle .pt paths are rejected.
python scripts/capture_hidden_states.py \
  --model /media/fmodels/Qwen/Qwen3.6-27B --layer 20 \
  --out /tmp/qwen27_l20_hidden.safetensors --tokens 4096

# Per-projection baseline vs Hadamard on real weights (incl. linear_attn)
python scripts/ablate_hadamard_fp6.py \
  --model /media/fmodels/Qwen/Qwen3.6-27B --layer 20 \
  --include-linear-attn --tokens 512 \
  --x-from /tmp/qwen27_l20_hidden.safetensors
# Repeat for 2-3 layers (e.g. --layer 5 / 20 / 40) before deciding.
```

Decision rule: **negative `full-Δ%`** consistently (esp. on `linear_attn.*`)
→ proceed to kernel design (rotated export + `Hᵀ` in the activation
quantizer). **~0% or positive** → Phase 1.4 is dead too; remaining options are
per-row global scale (weight side, §4.4) and revisiting the accuracy target.
Caveat: on pure gaussian inputs the activation side is rotation-invariant, so
run with `--x-from` real captures for the meaningful number.

**Layer-20 sim results (Qwen3.6-27B, real x, 512 tokens):**

| projection | full-base | full-had | full-Δ% |
|---|---|---|---|
| linear_attn.in_proj_a | 0.03230 | 0.02851 | **-11.7%** |
| linear_attn.in_proj_b | 0.03698 | 0.02756 | **-25.5%** |
| linear_attn.in_proj_qkv | 0.02971 | 0.02966 | -0.2% |
| linear_attn.in_proj_z | 0.02874 | 0.02803 | -2.5% |
| linear_attn.out_proj | 0.03889 | 0.03872 | -0.4% |
| mlp.down_proj | 0.03865 | 0.03853 | -0.3% |
| mlp.gate_proj | 0.02903 | 0.02972 | +2.4% |
| mlp.up_proj | 0.03627 | 0.03862 | **+6.5%** |

Group means: `linear_attn` **-8.5%**, `mlp` **+2.8%**. Read: rotation helps
exactly where per-block outlier structure exists (`in_proj_a/b`) and *hurts*
blocks with correlated same-sign content (MLP inputs post-SiLU-adjacent
residual — the all-ones Hadamard row concentrates the block mean into one
element, raising amax). Implication: if 1.4 proceeds, rotation must be
**selective per-module** (flagged per linear in the checkpoint config, e.g.
`linear_attn.in_proj_*` only), never blanket.

**Depth-stability check (layers 5 / 20 / 40, full-Δ%):**

| projection | L5 | L20 | L40 | verdict |
|---|---|---|---|---|
| linear_attn.in_proj_a | -11.3% | -11.7% | -13.5% | **stable, double-digit** |
| linear_attn.in_proj_b | -13.5% | -25.5% | -27.2% | **stable, large** |
| linear_attn.in_proj_qkv | -1.3% | -0.2% | +3.8% | noise |
| linear_attn.in_proj_z | -1.4% | -2.5% | +0.8% | noise |
| linear_attn.out_proj | -0.5% | -0.4% | -0.3% | noise |
| mlp.down_proj | -0.3% | -0.3% | -0.4% | noise |
| mlp.gate_proj | 0.0% | +2.4% | +1.3% | slight hurt |
| mlp.up_proj | +0.8% | +6.5% | +14.5% | **hurt, grows with depth** |

**Sim verdict: CONDITIONAL GO.** Rotate `linear_attn.in_proj_a` +
`in_proj_b` ONLY; blanket rotation is dead (`up_proj` regression grows with
depth). Before any kernel commitment, measure the weight-side gain
end-to-end for free: fold the rotation into a BF16 dequant control
(`W_sim = QDQ(W_b H) Hᵀ`, exactly the rotated serving path under exact
activations — stock-vLLM servable) and diff its KLD against
`-P12-DequantBF16` (0.022881). See `scripts/simulate_hadamard_dequant.py`
below. Kernel work is justified only if that KLD delta plus the projected
activation-side share moves materially toward the 0.020 gate.

**Weight-side KLD measurement (rig, no kernel needed):**

```bash
python -m pytest tests/test_fp6_hadamard.py -v   # incl. fold-back identity

# BF16 checkpoint: rotated QDQ on in_proj_a/b, plain P12 QDQ elsewhere.
# Stock-vLLM loadable; NOT an FP6 artifact (see hadamard_sim.json marker).
python scripts/simulate_hadamard_dequant.py \
  --model /media/fmodels/Qwen/Qwen3.6-27B \
  --out   /media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P14-RotSim-DequantBF16 \
  --include-linear-attn

# Same score_mode_kld harness as P12-DequantBF16, same --reference-logits.
```

Reading the result: `P14-RotSim` KLD vs `P12-DequantBF16` 0.022881 is the
pure weight-side rotation gain on exactly the two flagged projections. The
activation-side share (the larger part of the sim's full-Δ%) stacks on top
of this in a real rotated kernel — but only measure that after the weight
side proves it moves the needle at the KLD level.

**Result (Jul 8): `P14-RotSim` KLD = 0.022758** — a delta of **-0.00012**
vs the P12 control (0.022881). Note the artifact is BF16 (~52G) by design:
it is a weight-only KLD isolation control like the other `-DequantBF16`
dirs, NOT an FP6/FP8 export.

**Phase 1.4 verdict: NO-GO for kernel work.** Despite -32% rel-RMSE on
`in_proj_b` weights, the KLD moved 0.0001 — those two projections carry a
tiny share of the model's total quantization error. Scaling the same logic
to the activation side (similar per-projection rel-RMSE gains, same two
projections) projects a full-pipeline gain of order 0.0002–0.0005, against
a 0.0337 → 0.020 gap. The `Hᵀ`-in-quantizer kernel investment cannot be
justified by that. Rotation is banked as understood-and-rejected; the
harness stays for future architectures where outlier projections dominate.

**Where this leaves Phase 1:** full = weight 0.0229 + runtime ~0.0108. Every
per-block lever (act formats, MSE scales, GPTQ analysis, rotation) is now
measured or bounded. Reaching ≤0.020 needs either (a) per-row weight global
scales (`weight_scale_2` per-row is already dequant-supported, §4.4 —
attacks the 0.0229 weight floor across ALL projections, not two), or (b) a
revised accuracy target acknowledging the W6A8 dense floor sits near 0.033.

#### 4.0.1 Baseline commands (rig)

```bash
# Dense export (Profile A coverage) + weight-only control
python scripts/quantize_model_fp6.py \
  --model /media/fmodels/Qwen/Qwen3.6-27B \
  --out   /media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P0 \
  --arch dense --source-format mxfp6_w6a8 --include-linear-attn --report-error
python scripts/dequantize_fp6_to_bf16.py \
  --model /media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P0 \
  --out   /media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P0-DequantBF16

# MoE export (script defaults, W6A8)
python scripts/quantize_model_fp6.py \
  --model /media/fmodels/Qwen/Qwen3.6-35B-A3B \
  --out   /media/fmodels/TheHouseOfTheDude/Qwen3.6-35B-A3B-FP6-P0 \
  --arch moe --source-format mxfp6_w6a8 --report-error

# KLD (vLLM fork dir; drop --reference-model to reuse saved reference logits)
#
# DETERMINISTIC KLD (see §4.0.5 — mandatory): eager enforce
# (TORCH_COMPILE_DISABLE=1, or a fork where score mode defaults to eager) for
# ALL models, plus B12X_MOE_DETERMINISTIC=1 (opt-in, default OFF) for MoE.
# Without both, the Mean KLD is not bit-reproducible and must not be recorded.
#
export B12X_ENABLE_FP6=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export B12X_FP6_MODEL_DIR=/media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-P0
python examples/offline_inference/score_mode_kld.py \
  --model "$B12X_FP6_MODEL_DIR" \
  --reference-logits ~/kld-nightly-vllm/kld-vllm/ref_logits_Qwen3.6-27B_ctx2048_s512 \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --context-length 2048 --stride 512 \
  --gpu-memory-utilization 0.90 --max-num-seqs 128
# MoE refs: /media/fmodels/kld-refs/qwen3.6-35b-a3b_ctx2048_s512
# Weight-only control: unset B12X_ENABLE_FP6 B12X_FP6_MODEL_DIR, point --model
# at the -DequantBF16 dir, same --reference-logits.

# Benches
python scripts/bench_fp6_decode.py --source-format mxfp6_w6a8 --hbm
python scripts/bench_fp6_moe.py
```

### 4.1 MoE (Qwen3.6-35B-A3B) — progression

| Configuration | Mean KLD | Notes |
|---------------|----------|-------|
| Original W6A6 (E3M2 activations) | **0.0268** | Worse than calibrated INT4 |
| E2M3 activation flip (config) | **0.0237** | Tied INT4 |
| W6A8 (E4M3 activations), June checkpoint | **0.0228** | ~190 tok/s unchanged |
| MSE-optimal block scale (ablation) | no benefit | Cancelled after sweep showed no gain |
| **W6A8 fresh re-quant (P0 baseline, Jul 8)** | **0.0165** | **At the FP8 band (~0.0158)**; checkpoint + runtime evolution since June — refs are deterministic, so the gain is real |

MoE activation ablation (`scripts/ablate_moe_act_quant.py`) showed FC1 vs FC2 hop contributions; W6A8 was the winning activation format without speed regression.

### 4.2 Dense (Qwen3.6-27B) — current understanding

| Run | Mean KLD | What it measures |
|-----|----------|------------------|
| **Original** (`TheHouseOfTheDude/Qwen3.6-27B-FP6`, Jun 11 weights) | **~0.016** | Historical baseline (full B12X pipeline) — **see §4.2.1** |
| After MoE code changes + config edits | **0.037–0.039** | Misattributed to "MoE broke dense" initially |
| Test: disable BF16 GEMV + e2m3 config | **0.039834** | Runtime optimizations **not** the cause |
| Test: disable GEMV + disable packed-B | **0.037517** | Same conclusion |
| **Dequant→BF16** (stock vLLM, no B12X) | **0.024112** | **Weight error only** |
| **Fresh re-quant** (current script, full pipeline) | **0.029002** | **Profile B** (linear_attn BF16): weight + activation + kernel |
| `--report-error` per-layer RMSE (4 layers) | **~2.8% rel-RMSE** uniform | Quantizer working correctly; E2M3 floor |
| **P0 baseline full pipeline** (Profile A, Jul 8) | **0.039980** | linear_attn FP6 at runtime; see §4.0 |
| **P0 baseline weight-only** (same export) | **0.023612** | First same-export full/weight-only pair |

#### 4.2.1 Historical 0.016 baseline — unreproduced (possible fluke)

The **~0.016** number recorded for the first `Qwen3.6-27B-FP6` upload is **not reproducible** with any configuration tested since the dense investigation (June 2026):

- The **original quant command is lost** (no shell history).
- Config bundle analysis (July 2026) shows the HF checkpoint is **Profile A** (~24G):
  MLP + self_attn + **linear_attn** (`--include-linear-attn`), not a minimal export.
- **Dequant→BF16 KLD = 0.024** on that same HF checkpoint proves weight error alone
  cannot reach 0.016 — even a perfect runtime kernel cannot beat the weight floor.
- **Full-pipeline FRESH** (Profile B, 31G) = **0.029**; Profile B is a *larger* export
  (more BF16 GDN weights), not the HF recipe.

**Working conclusion:** treat **0.016 as an unreliable outlier / likely fluke** — most
plausibly a **KLD harness or reference mismatch** (different model dir, reference checkpoint,
or score-mode settings), not a secret better quant recipe. The reproducible floors are
**~0.024 weight-only** and **~0.029 full pipeline** (Profile B).

**Conclusions from the dense investigation** (updated with the P0 baseline)

1. **Runtime kernel is exonerated** — env toggles and MoE/W6A8 code paths are additive for dense; they do not explain 0.016 → 0.037.
2. **Weight quantization floor is ~0.024 KLD** (~2.8% rel-RMSE) for **full-coverage symmetric E2M3** with per-tensor global scale and **ceil** UE8M0 block scales.
3. **Runtime (activation + kernel) overhead is ~0.016 KLD on Profile A** — the P0
   baseline finally measures full (0.0400) and weight-only (0.0236) on the SAME
   export. The previously documented ~0.005 came from subtracting across
   different exports (Profile B full 0.029 − Profile A dequant 0.024) and
   understated the slice by ~3x.
4. **GDN `linear_attn` runtime accounts for ~0.011 of that slice**: Profile B
   (linear_attn BF16 at runtime) full-pipeline = 0.029 vs Profile A = 0.0400,
   with near-identical weight floors. The remaining ~0.005 matches the old
   MLP+attention-only estimate. This makes linear_attn activation quantization
   the top dense accuracy target for Phase 1 (e.g., W6A8/E4M3 activations on
   the GDN projections first).
5. Comparing `config.json` across historical exports (§6.4) is the best remaining lead on what made 0.016 different.

### 4.3 Failed accuracy experiment: round-to-nearest block scale

Attempted changing UE8M0 exponent from **ceil** to **round-to-nearest** in `_ue8m0_scale_from_block_max` and `fp6_block_ue8m0_exact`.

**Result: catastrophic — ~27% rel-RMSE** (vs ~2.8% baseline). **Reverted immediately.**

**Why:** The block scale is a **containment** exponent, not a value quantizer. Rounding down leaves the scale too small; most elements in the block saturate. **Ceil is required** for correctness. Do not revisit naive round-nearest on the exponent.

### 4.4 Accuracy work still open

| Approach | Expected impact | Effort | Stays in FP6 kernel? |
|----------|-----------------|--------|---------------------|
| **MSE-optimal block scale** (per block: try ceil vs ceil−1, pick lower MSE) | ~10–15% RMSE → KLD ~0.021–0.022 weight-only | **Phase 1.2 code done** (`block_scale_rule=mse`); pending rig | Yes |
| **Per-row global scale** (one `gs` per output channel vs per-tensor) | ~30–40% RMSE reduction | High; format + epilogue change | Yes |
| **W6A8 on dense** (E4M3 activations) | Part of the **~0.016** runtime slice (§4.2 #3-4); GDN linear_attn is ~0.011 of it. **Phase 1.1 implemented** — dense honors `activation_format=e4m3`. Gate: full-pipeline **≤ 0.026**. | Done (code); pending rig KLD | Yes |
| **Mixed precision** (`--no-attention`, sensitive layers BF16) | Can reach ~0.016–0.018 KLD | Low | **No** — BF16 fallback, loses FP6 speed on those layers |
| **Recover original 0.016 recipe** | Unknown until original quant command / HF config verified | Investigation | Depends |

**User constraint:** Stay **100% in the FP6 kernel path** for performance — mixed-precision BF16 fallback is not acceptable for production.

---

## 5. vLLM integration

### 5.0 Two-repo setup (important for new developers)

B12X FP6 serving and KLD measurement use **two different vLLM-related pieces**:

| Piece | Where it lives | Purpose |
|-------|----------------|---------|
| **Score-mode KLD / PPL vLLM** | [phaelon74/vllm `feature/score-mode-ppl-kld`](https://github.com/phaelon74/vllm/tree/feature/score-mode-ppl-kld) | Starting vLLM fork for **Wikitext KLD** and score-mode evaluation. **Does not include B12X FP6** — install this fork (or equivalent) for the KLD harness only. |
| **B12X FP6 plugin** | **This repo** (`b12x`) via `pip install -e .` | Registers `b12x_fp6` quantization, loads FP6 weights, runs `b12x_moe_fp6` / `dense_fp6_linear`. **No vLLM source patches required** for FP6 matmuls if the installed vLLM supports `general_plugins` + the Qwen3.6 module layout. |

**Production stack on the main rig:** vLLM built from the **score-mode-ppl-kld** branch **plus** `pip install -e` of this **b12x** repo into the same venv. The team does not maintain a single merged PR; integration is **plugin + env vars**.

KLD workflow: run the score-mode KLD entrypoint from the vLLM fork against BF16 reference and FP6 checkpoint with `B12X_ENABLE_FP6=1`. `scripts/capture_hidden_states.py` uses the same Wikitext source as the KLD script (`wikitext-2-raw-v1`).

**Possible local vLLM model-definition changes** (on the serving box, not in b12x): Qwen3.6 multimodal builds may need `quant_config` passed into `linear_attn` and vision tower modules so FP6 shards on disk actually bind. The b12x plugin doc (`vllm_plugin.py`) assumes vLLM constructs those layers with `quant_config`. If a vLLM nightly omits that, layers silently stay BF16. Check the local vLLM tree on the main rig for any Qwen3.6-specific diffs beyond upstream.

### 5.1 How B12X FP6 loads into vLLM

B12X registers as an **installable vLLM plugin** (not a fork patch):

```toml
# pyproject.toml
[project.entry-points."vllm.general_plugins"]
b12x_fp6 = "b12x.integration.vllm_plugin:register_b12x_fp6"
```

**Flow**

1. `pip install b12x` (editable: `pip install -e ".[dev]"`).
2. Set **`B12X_ENABLE_FP6=1`** — master gate; without it, vLLM uses stock ModelOpt path.
3. Optionally set **`B12X_FP6_MODEL_DIR=/path/to/checkpoint`** — required for spawned TP workers to resolve weights.
4. `vllm serve /path/to/fp6-model` — plugin claims checkpoints with `quantization_config.quant_method=modelopt` and `quant_algo=W6A6`.

Reference adapter (for forks): `examples/vllm_fp6_adapter.py`. Production path: `b12x/integration/vllm_plugin.py`.

### 5.2 Checkpoint detection

```json
"quantization_config": {
  "quant_method": "modelopt",
  "quant_algo": "W6A6",
  "weight_format": "e2m3",
  "activation_format": "e4m3",
  "group_size": 32,
  ...
}
```

`B12XFp6Config.override_quantization_method` wins over stock ModelOpt when env gate is on.

### 5.3 Weight binding (critical design)

**Dense (`B12XFP6LinearMethod`)**

- Registers real vLLM params in `create_weights`:
  - `weight` — `(out, 3*in/4)` uint8 packed FP6
  - `weight_scale` — `(out, in/32)` uint8 UE8M0 (unswizzled on disk)
  - `weight_scale_2` / `input_scale` — per-shard f32 globals
- vLLM's **QKV** and **gate_up** fused loaders split on-disk matrices into packed params (row-independent packing).
- `process_weights_after_loading`: swizzle scales, build `FP6DenseWeight`, register opaque op.
- `apply`: `b12x::fp6_dense_linear` custom op → `dense_fp6_linear_expanded`.

**MoE (`B12XFP6MoEMethod`)**

- Registers stacked per-expert params: `w13_weight`, `w2_weight`, scales, alphas.
- vLLM expert loader fills via `(expert_id, shard_id)` convention.
- `process_weights_after_loading`: **in-place FC1 row swap** `[gate;up]` → `[up;gate]`, swizzle scales, build `FP6MoEWeights`, warm-run decode sizes, `empty_cache`.
- `apply`: `b12x_moe_fp6` with vLLM-supplied `topk_weights` / `topk_ids`.

**Non-FP6 layers** (norms, router, `lm_head`, partial GDN) → `UnquantizedLinearMethod` or small-N GEMV subclass.

### 5.4 CUDA graphs and multiprocessing

- **Opaque custom ops** prevent Dynamo from tracing into CUTLASS JIT (avoids `--enforce-eager`).
- **Pickle-safe config factory** (`_rebuild_b12x_fp6_config`) — vLLM spawn must not pickle unpicklable torch internals.
- **`VLLM_WORKER_MULTIPROC_METHOD=spawn`** — recommended on multi-GPU to avoid CUDA context issues (used during KLD debugging).
- MoE warm-run covers vLLM's **resolved** `cudagraph_capture_sizes` automatically (read from the live config at load; `B12X_MOE_WARM_MS` overrides); shared workspace cache keyed by tensor `id()`.

### 5.5 Multimodal / GDN / vision

- **Linear attention (GDN):** `in_proj_qkvz` fuses qkv+z FP6 shards; `in_proj_ba` often stays BF16 (small N); `out_proj` FP6 when on disk.
- **Vision tower:** FP6 when exported with `--include-vision`.
- Fused-module rule: FP6 only if **every** constituent projection is FP6 on disk.

### 5.6 Launch checklist

```bash
export B12X_ENABLE_FP6=1
export B12X_FP6_MODEL_DIR=/media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6
# Optional for multi-GPU:
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve "$B12X_FP6_MODEL_DIR" \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 512   # lower if Mamba cache block limit errors
```

### 5.7 What B12X adds to vLLM (plugin behavior — all in `b12x/integration/vllm_plugin.py`)

This is the checklist of **behavioral changes** B12X imposes on vLLM when the plugin is active. None of this is in the score-mode KLD fork; it ships from **this repo**.

**Registration & detection**

- `register_b12x_fp6()` entry point via `pyproject.toml` → `vllm.general_plugins`.
- Registers `B12XFp6Config` as quantization name `b12x_fp6`.
- Claims `quant_method=modelopt` + `quant_algo=W6A6` when `B12X_ENABLE_FP6=1`.
- Resolves checkpoint path from `B12X_FP6_MODEL_DIR` (critical for TP worker spawn).
- Pickle-safe `__reduce__` / `_rebuild_b12x_fp6_config` for multiprocess spawn.

**Dense linear (`_VllmLinearMethod`)**

- Registers ModelOpt-mirror params: packed `weight`, `weight_scale`, `weight_scale_2`, `input_scale`.
- Hooks **v2 weight loader** (`WEIGHT_LOADER_V2_SUPPORTED`) for packed FP6 + GDN QKV shard ids.
- `process_weights_after_loading`: swizzle UE8M0 scales, build `FP6DenseWeight`, optionally drop packed storage for narrow-N layers, register `b12x::fp6_dense_linear`.
- `apply`: opaque custom op (CUDA-graph / Dynamo safe) → full W6A6 dense path.
- Fused modules (`qkv_proj`, `gate_up_proj`, `in_proj_qkvz`): FP6 only if **all** constituents are FP6 on disk.

**Small-N BF16 GEMV (`_VllmSmallNBF16Method`)**

- Subclass of `UnquantizedLinearMethod` for bf16 layers with N≤1024, K≥1024 (GDN `in_proj_ba`).
- Private weight **clone** (vLLM loader memory incompatible with raw cuLaunchKernel).
- Precompile + warm-run `b12x::bf16_gemv_small_n` at load time.
- Disable via `B12X_DISABLE_BF16_GEMV=1`.

**MoE (`_VllmMoEMethod`)**

- Registers per-expert stacked params (`w13_weight`, `w2_weight`, scales, globals) for vLLM's expert loader.
- `process_weights_after_loading`: in-place **FC1 row reorder** vLLM `[gate;up]` → kernel `[up;gate]`; swizzle scales; build `FP6MoEWeights`; warm-run the resolved cudagraph capture sizes; `torch.cuda.empty_cache()`.
- Persistent CUDA-graph scatter output buffers for M≤512; zero-in-place per forward.
- `apply`: delegate to `B12XFP6MoEMethod` → `b12x_moe_fp6` with vLLM's `topk_weights` / `topk_ids`.
- Shared expert workspace cache across layers (`fp6_serving._SHARED_WORKSPACE_CACHE`).

**Export (offline, not vLLM)**

- `scripts/quantize_model_fp6.py` → full HF safetensors checkpoint (§6).
- vLLM only **consumes** the export; it does not run quantization.

**Logging**

- Plugin logger routed to `vllm.b12x_fp6` so messages appear in vLLM server logs.

---

## 6. Quantization pipeline

### 6.1 Primary script

```bash
python scripts/quantize_model_fp6.py \
  --model /path/to/bf16-hf \
  --out   /path/to/fp6-hf \
  --arch  auto          # or dense | moe
  --format safetensors  # default
  --source-format mxfp6_w6a8   # MoE default; use mxfp6_e2m3 for explicit W6A6 metadata
```

**Flags**

| Flag | Effect |
|------|--------|
| `--no-attention` | Dense: MLP only, attention stays BF16 |
| `--include-linear-attn` | Quantize GDN `in_proj_*` / `out_proj` |
| `--include-vision` | Quantize vision tower linears |
| `--report-error` | Per-group rel-RMSE table after export |
| `--limit-layers N` | Fast partial export for debugging |
| `--dry-run` | Plan only, no GPU |
| `--skip-experts` | MoE sensitivity: experts BF16 |

**Default coverage (dense):** MLP + attention (`include_attention=True`). Norms, embeddings, `lm_head`, router, SSM/GDN (unless `--include-linear-attn`), vision (unless `--include-vision`) stay BF16.

### 6.2 Diagnostic scripts

| Script | Purpose |
|--------|---------|
| `scripts/dequantize_fp6_to_bf16.py` | Weight-only KLD isolation (stock vLLM) |
| `scripts/collect_fp6_configs.py` | Bundle `config.json` from multiple FP6 dirs for coverage comparison |
| `scripts/ablate_moe_act_quant.py` | Per-hop MoE activation format ablation |
| `scripts/capture_hidden_states.py` | Real hidden states for ablation (Wikitext) |
| `scripts/bench_fp6_decode.py` | Dense decode micro-bench (quant vs GEMM) |
| `scripts/bench_fp6_moe.py` | MoE kernel bench |
| `scripts/validate_fp6_moe_artifact.py` | MoE artifact validation |

### 6.3 On-disk tensor layout (per linear)

Mirrors ModelOpt NVFP4 key names:

- `<name>.weight` — `(out, 3*in//4)` uint8 packed E2M3
- `<name>.weight_scale` — `(out, in//32)` uint8 UE8M0 (unswizzled)
- `<name>.weight_scale_2` — f32 global weight scale
- `<name>.input_scale` — f32 placeholder (1.0)

Swizzle applied at **load time** in vLLM `process_weights_after_loading`.

### 6.4 On-disk FP6 checkpoint inventory (27B dense experiments)

Multiple exports exist on the main rig under `/media/fmodels/TheHouseOfTheDude/`.
**Config bundle analysis** (`fp6_configs_bundle.json`, July 2026) shows **two distinct
export profiles** — not seven unique recipes.

#### Profile A — ~24 GB (`exclude_modules_count: 195`)

| Directory | Size | Config mtime | `activation_format` |
|-----------|------|--------------|---------------------|
| `qwen3-6_27B_dense_fp6_full` | 23.9G | Jun 9 | e3m2 |
| `qwen3-6_27B_dense_fp6_la` | 24.2G | Jun 8 | e3m2 |
| **`Qwen3.6-27B-FP6`** (HF upload) | 24.2G | Jun 14 | **e2m3** |

**Quantizes (FP6 on disk):** MLP + `self_attn` q/k/v/o + **GDN linear_attn**
`in_proj_qkv`, `in_proj_z`, `out_proj` (i.e. export with `--include-linear-attn`).

**Stays BF16:** `in_proj_a` / `in_proj_b`, norms, vision, MTP, `lm_head`, etc.

#### Profile B — ~31 GB (`exclude_modules_count: 198`)

| Directory | Size | Config mtime | `activation_format` |
|-----------|------|--------------|---------------------|
| `qwen3-6_27B_dense_fp6` | 30.4G | Jun 8 | e3m2 |
| `qwen3-6_27B_dense_fp6_gr` | 30.4G | Jun 8 | e3m2 |
| `Qwen3.6-27B-FP6-FRESH` | 30.4G | Jun 14 | e2m3 |

**Same as Profile A, except** GDN `in_proj_qkv`, `in_proj_z`, `out_proj` are listed in
`exclude_modules` → those large projections stay **BF16** (~6.5 GB extra on disk).

`dense_fp6` and `dense_fp6_gr` configs are **identical** in coverage; `_gr` suffix
does not change `exclude_modules` in the recorded configs.

#### Phase 0 baseline exports (Jul 8, 2026 — quant commands in §4.0.1)

| Directory | Profile / coverage | Role |
|-----------|--------------------|------|
| `Qwen3.6-27B-FP6-P0` | Profile A (MLP + self_attn + linear_attn, `mxfp6_w6a8`) | Dense baseline (full 0.0400) |
| `Qwen3.6-27B-FP6-P0-DequantBF16` | dequant of `-P0` | Weight-only control (0.0236) |
| `Qwen3.6-35B-A3B-FP6-P0` | MoE script defaults (`mxfp6_w6a8`) | MoE baseline (0.0165) |

#### Dequant control

| Directory | Size | Notes |
|-----------|------|-------|
| `Qwen3.6-27B-FP6-DequantBF16` | 51.8G | No `quantization_config`; BF16 weights from Profile A HF checkpoint |

#### Implications for the 0.016 KLD claim

The HF model (`Qwen3.6-27B-FP6`) is **Profile A** (includes linear-attn FP6), not the
narrower Profile B. So 0.016 was **not** from “MLP-only / no attention” coverage.

Yet **dequant→BF16 KLD on that same HF checkpoint = 0.024** (weight error only).
That is **better** than full-pipeline FRESH (0.029) but still **worse** than the
claimed 0.016 full-pipeline number — which is **arithmetically inconsistent** with
the measured weight floor unless the original 0.016 used a different checkpoint,
reference model, or KLD harness. **Treat 0.016 as unreproduced / likely fluke** (§4.2.1).

Collect configs:

```bash
python scripts/collect_fp6_configs.py \
  --out /tmp/fp6_configs_bundle.json

# or human-readable for paste into chat:
python scripts/collect_fp6_configs.py \
  --out /tmp/fp6_configs_bundle.txt --format text
```

---

## 7. Code map (start here)

| Area | Path |
|------|------|
| FP6 intrinsics, quantizers, MMA | `b12x/cute/fp6.py` |
| Dense GEMM | `b12x/gemm/dense.py`, `b12x/gemm/dense_mxfp6.py` |
| MoE fused kernels | `b12x/moe/fused/mxfp6_moe.py`, `micro_fp6.py` |
| MoE integration / dispatch | `b12x/integration/tp_moe.py` |
| vLLM plugin | `b12x/integration/vllm_plugin.py` |
| Framework-agnostic serving API | `b12x/integration/fp6_serving.py` |
| Offline weight quant | `b12x/quantization/fp6_dense_weights.py`, `fp6_moe_weights.py` |
| Safetensors export | `b12x/quantization/fp6_safetensors_export.py` |
| Checkpoint schema | `b12x/quantization/fp6_checkpoint.py` |
| CUDA-graph opaque dense op | `b12x/quantization/fp6_dense_op.py` |
| GPU weight quantizer (TMA) | `b12x/quantization/bf16_to_fp6_tma.py` |
| Small-M activation quantizer | `b12x/quantization/bf16_to_fp6_small_m.py` |

**Block-scale rule (do not change without proof):** `_ue8m0_scale_from_block_max` and `fp6_block_ue8m0_exact` use **`ceil(log2(block_max * gs / fmt_max)) + 127`**. Required for containment.

---

## 8. Testing & onboarding

GPU tests require **Blackwell sm_120** (tests use `require_sm120()`). Run on the **main Linux rig**, not the Windows dev workstation.

**New developers** have access to the main rig and its venv — use the same environment the team uses for quant, serve, and KLD (no separate sandbox documented here).

```bash
cd ~/kld-nightly-vllm/b12x    # rig clone
source ~/kld-nightly-vllm/venv/bin/activate   # team's existing venv
pip install -e ".[dev]"
# ALWAYS python -m pytest: a bare `pytest` can resolve to /usr/bin/python3
# (no torch) and fail collection with ModuleNotFoundError.
python -m pytest tests/test_fp6_serving.py tests/test_fp6_gpu.py tests/test_fp6_dense_op.py -v
```

Key test areas: serving detection, GPU quantizer round-trip, MoE numeric vs reference, vLLM plugin weight binding.

**Suggested first-day reads:** this doc, `b12x/integration/vllm_plugin.py` module docstring, `examples/vllm_fp6_adapter.py`, `scripts/quantize_model_fp6.py --help`.

---

## 9. Areas where additional ground can be gained

No fixed priority order — each area has independent upside for **accuracy**, **performance**, or **operability**.

### 9.1 Accuracy (stay 100% in FP6 kernel)

| Area | Current | Potential | Mechanism |
|------|---------|-----------|-----------|
| Weight quant floor | ~0.024 KLD | ~0.020–0.022 | MSE-optimal per-block scale (ceil vs floor per 32-elem block, offline only) |
| Weight quant floor | ~2.8% RMSE | ~2.0% RMSE (est.) | Per-row global scale instead of per-tensor (format + epilogue work) |
| Activation slice | ~0.005 KLD overhead | smaller | W6A8 (E4M3) on dense if kernel path enabled without regression |
| Historical mystery | 0.016 claimed | explain or dismiss | Diff `config.json` across §6.4 exports; KLD the smallest (24G) variant |
| MoE | 0.0228 KLD | lower | Further activation-hop tuning; real hidden-state ablation (`capture_hidden_states.py`) |

**Do not use:** mixed-precision BF16 attention (`--no-attention`) for production — violates full-FP6 performance goal.

**Do not use:** round-to-nearest on UE8M0 **containment** exponent — tried, **~27% RMSE**, reverted (§4.3).

### 9.2 Performance

| Area | Current (Jul 22) | Potential | Mechanism |
|------|---------|-----------|-----------|
| Dense decode | 100 tok/s TP1 / 135 TP2 (@1K, MTP k=4) | incremental | `bench_fp6_decode.py` → optimize quant vs GEMM split; tune `B12X_PACKED_B_MIN_N` |
| MoE decode | 272 tok/s TP1 (@1K, MTP k=4) | single-digit % | Phase 4.2 micro-kernel (deferred — decode no longer expert-GEMM-bound at BS1) |
| Prefill | banked (§4.0.6 Phase D, 1K–32K) | — | `vllm bench serve` sweep; prefill ~8–11k tok/s dense |
| Memory at serve | OOM on high `max_num_seqs` | stable capture | Mamba cache blocks vs `gpu_memory_utilization`; tune `max_num_seqs` |

### 9.3 Integration & operability

| Area | Notes |
|------|-------|
| Config bundle tool | `scripts/collect_fp6_configs.py` — compare exports |
| Arch-aware export defaults | Dense `mxfp6_e2m3` metadata vs MoE `mxfp6_w6a8` (cosmetic + less confusion) |
| Document local vLLM diffs | Any Qwen3.6 `quant_config` wiring on main rig not in b12x |
| HF publish pipeline | Re-quant → KLD → upload after quant improvements |

### 9.4 Size vs quality

| Area | Notes |
|------|-------|
| Phase 3 reallocation | BF16 for measured-sensitive small groups + vision FP6 based on KLD sweep |
| `--include-linear-attn` / `--include-vision` | Shrinks multimodal models below FP8 size; validate KLD per flag |

---

## 10. Changelog anchor (git)

Recent accuracy-relevant commits:

| Date | Commit area | Summary |
|------|-------------|---------|
| Jul 22 | Primetime close-out | Phase D serve validation banked; launch-script fixes (brace bug, TP>1 all-reduce guard, MTP-derived capture sizes); dynamic MoE warm list; debug tooling removed |
| Jul 21 | Phase C MoE determinism | Per-(pair,slice) staging + canonical combine (`B12X_MOE_DETERMINISTIC`, opt-in); MLIR ICE fix in `mxfp6_moe.py`; MoE KLD bit-identical 0.015043 x2 |
| Jul 14–21 | Phases A/B eager re-baseline | Eager-only scoring standard; per-row activation scaling kept (−0.00082); fused quant vindicated; rotation permanently NO-GO; MSE-vs-ceil tie |
| Jul 12 | Phase 3.1 TP validated | TP=2 134.89 tok/s (+24% vs TP=1 108.49); packed-B fix + NCCL all-reduce; KLD 0.032943 |
| Jul 12 | Phase 3.1 TP-aware packed-B | `FP6DenseWeight.out_features_unsharded` + serve script `--disable-custom-all-reduce` at TP>1 |
| Jul 12 | Phase 2.1+2.2 perf | Tile+warmup+persistent scratch validated e2e: 128 tok/s at greedy k=4 1K ctx; KLD 0.032960 band-stable |
| Jul 8 | Phase 0 guardrails (`52c27d7`) | Unit-gs dense weight contract (matches export), `weight_scale_2`-aware dequant, fmt-aware global-scale numerator; baseline locked (§4.0) |
| Jun 13 | W6A8 | E4M3 activation MMA variants; `mxfp6_w6a8` default export |
| Jun 12 | Fine-tuning quant | `dequantize_fp6_to_bf16`, error reporting |
| Jun 11 | User re-quant | 27B weights timestamp Jun 11 16:46 — same E2M3 math, ~0.024 weight KLD |
| Jun 10 | Phase 2b | Packed stride fix; byte-container activations; dense speed path |
| May 29 | W6A6 kernels | Initial FP6; ceil block scale from day one |

---

*Document updated July 22, 2026 (primetime close-out §0: eager-banked accuracy, MoE bit-determinism, Phase D serve validation, final defaults, debug tooling removed).*
