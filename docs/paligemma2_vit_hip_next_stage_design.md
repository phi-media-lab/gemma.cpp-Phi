# PaliGemma2 ViT HIP Next-Stage Kernel Design

This note is the working design for the next acceleration phase after the
stable F32-attention + BF16-pack + custom projection/MLP path.

## Current Baseline

Recommended runtime path:

```bash
--paligemma_vit_backend hip_probe
--paligemma_vit_hip_return_image_tokens 1
--paligemma_vit_hip_attention_pack_bf16 1
--paligemma_vit_hip_attention_direct_qkv 1
--paligemma_vit_hip_qkv_wmma2d 1
--paligemma_vit_hip_attn_out_wmma2d 1
--paligemma_vit_hip_mlp_up_wmma2d 1
--paligemma_vit_hip_mlp_down_wmma8 1
--paligemma_vit_hip_mlp_down_fused_residual 1
```

For 448px, the recommended scripts additionally pass:

```bash
--paligemma_vit_hip_mlp_down_wmma_waves 12
```

Latest 448px profile shape after the roofline pass:

| Phase | Time |
| --- | ---: |
| MLP total | ~191.7 ms |
| Attention | ~169.4 ms |
| QKV | ~84.2 ms |
| Attention output | ~30.9 ms |
| Total device sum | ~487.3 ms |

The detailed profiling/roofline breakdown is in
`docs/paligemma2_vit_hip_roofline_profile.md`.

The integrated custom kernels already replace:

- QKV projection with grouped-2D BF16 WMMA 8x4.
- Attention output projection with grouped-2D BF16 WMMA 8x4.
- MLP up projection with grouped-2D BF16 WMMA 8x4.
- MLP down projection with grouped-N BF16 WMMA 8 by default, and WMMA 12 for
  the 448px recommended path.

The stable attention route keeps QK, softmax, and AV in F32, then packs the
attention output to BF16. BF16 QK/WMMA and deferred softmax-scale remain
explicit experiments because stress prompts showed decoder-visible drift.

## Goal

Reduce 448px ViT device time without weakening the current correctness
contract:

- no default semantic drift against the current recommended F32 attention path;
- every replacement GEMM must have a local rocBLAS comparison;
- any candidate promoted into the generation path must pass deterministic smoke
  and extended A/B guardrails.

## Priority 1: MLP-Down Grouped-N Retune

MLP-down is still about 90 ms of the 448px device profile. It is a good first
target because:

- the boundary is a pure BF16 x BF16 -> F32 GEMM:
  `up_bf16[rows, mlp] * linear_1[mlp, model_dim]`;
- the current WMMA8 kernel is already proven against rocBLAS on real tensors;
- failed 2D down sweeps showed that staging both A and B was not automatically
  better for this narrower-output shape;
- grouped-N variants can be tested in isolation before touching generation
  code.

Existing grouped-N coverage only measured 2, 4, and 8 waves per workgroup.
For PaliGemma2 down-projection, `model_dim = 1152`, so additional macro-tile
widths are worth measuring:

| Waves N | Columns per workgroup | N blocks for model_dim=1152 | Expected tradeoff |
| ---: | ---: | ---: | --- |
| 6 | 96 | 12 | Lower occupancy pressure than 8/12, more workgroups |
| 8 | 128 | 9 | Current integrated baseline |
| 12 | 192 | 6 | Exact N coverage, fewer workgroups |
| 16 | 256 | 5 | Fewest workgroups, some tail slack, higher block pressure |

The hypothesis is that 12 waves may be a better shape-specialized point for
`N=1152`: it removes N-tail slack and reduces workgroup count by one third
relative to 8 waves, while avoiding the 512-thread block pressure of 16 waves.

Execution plan:

1. Extend the standalone MLP grouped-N sweep to include 6, 12, and 16 waves.
2. Benchmark `mlp_down_224` and `mlp_down_448` first, because those are the only
   shapes that should be eligible for replacing the integrated down kernel.
3. Promote the best stable single schedule into `MlpDownWmma8Kernel` only if it
   improves 448 and does not materially regress 224.
4. Run the real-tensor down check, smoke script, and recommended profile.

Promotion rule:

- If 12 or 16 wins only in standalone but loses in full-profile timing, keep it
  as a documented negative result.
- If 12 wins 448 and is neutral on 224, update the runtime label from `wmma8`
  to the actual schedule so profile output stays honest.

## Priority 2: F32-Safe Attention Work

Attention is the largest non-MLP hotspot, but it is more fragile. Previous
attention experiments showed that small numerical rearrangements can alter
stress-prompt output. The next attention design must keep softmax semantics
close to the current F32 route.

Candidate directions:

- specialize the current F32 softmax and pack path without moving normalization
  across AV;
- reduce split/pack overhead with vectorized kernels where the tensor layout
  allows aligned `float4` or BF16-pair operations;
- only revisit fused QK/softmax/AV after the MLP-down schedule space is closed.

Non-goals for this phase:

- promote BF16 QK/WMMA into the recommended path;
- promote deferred softmax-scale into the recommended path;
- rewrite the whole attention stack before proving a smaller isolated win.

## Validation Checklist

For every promoted runtime change:

```bash
bash scripts/build_paligemma2_vit_hip_bench.sh
bash scripts/build_paligemma2_vit_hip_backend_probe.sh
bash scripts/build_gemma_paligemma2_vit_hip.sh
REBUILD=0 scripts/smoke_paligemma2_vit_hip_recommended.sh
git diff --check
find scripts -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

For attention changes, also run the extended A/B stress set before promotion.

## Execution Log: Grouped-N Down Sweep

Implemented the first design step by extending `--mlp_wmma_group_sweep_only`
from `2/4/8` waves to `2/4/6/8/12/16` waves per workgroup.

Short 20-sample sweep:

| Shape | Best existing | Best new | Decision |
| --- | ---: | ---: | --- |
| `mlp_down_224` | `8`: 0.891 ms | `12`: 0.896 ms | Keep 8 |
| `mlp_down_448` | `8`: 3.107 ms | `16`: 3.260 ms | Keep 8 |

Long 80-sample sweep:

| Shape | `8` | `12` | `16` | Standalone reading |
| --- | ---: | ---: | ---: | --- |
| `mlp_down_224` | 0.903 ms | 0.891 ms | 0.907 ms | 12 was only ~1.3% faster |
| `mlp_down_448` | 3.246 ms | 3.203 ms | 3.426 ms | 12 was only ~1.3% faster |

The standalone result was too small to promote directly, so `12` was tested in
the real backend MLP-down check and then compared against the restored `8`
schedule in the same session:

| Model | Real-tensor `12` check | Real-tensor `8` check | Decision |
| --- | ---: | ---: | --- |
| 224 | 0.922 ms | Not rerun after revert | Not enough by itself |
| 448 | 3.614 ms | 3.668 ms | Noise-level gain |

The important result is that the shape-specialized 12-wave hypothesis did not
produce a large or robust enough transfer from synthetic standalone buffers to
the real layer0 tensor path on 448px. The backend remains on grouped-N WMMA8
because changing the production schedule for a ~1-2% isolated gain would make
the recommended profile more fragile without materially moving end-to-end ViT
time. The expanded sweep stays useful for future retuning, but no runtime
change is promoted from this step.

## Execution Log: F32-Safe QKV Split Vectorization

Implemented the first attention-side local optimization by adding a vectorized
`float4` Q/K/V split kernel to the backend. This only changes the layout-copy
phase before F32 QK; QK, softmax, AV, and pack-to-BF16 keep the same precision
and ordering as the recommended path.

Standalone split microbench:

| Shape | Scalar | Vector4 | Relative | Error |
| --- | ---: | ---: | ---: | --- |
| 224, 80 samples | 0.103 ms | 0.097 ms | 0.940 | exact |
| 448, 80 samples | 0.436 ms | 0.382 ms | 0.875 | exact |
| 224, 120 samples | 0.099 ms | 0.094 ms | 0.944 | exact |
| 448, 120 samples | 0.429 ms | 0.469 ms | 1.094 | exact |

Decision:

- Use vector4 split only for `rows == 256` and `qkv_dim % 4 == 0`.
- Keep 448px on the scalar split because its vector4 result was not stable
  across longer samples.
- Do not add a user-facing flag; this is a shape-gated implementation detail
  with a scalar fallback.

Validation:

- `scripts/build_paligemma2_vit_hip_backend_probe.sh`: pass.
- `scripts/build_gemma_paligemma2_vit_hip.sh`: pass.
- `REBUILD=0 scripts/smoke_paligemma2_vit_hip_recommended.sh`: pass.
- `KEEP_GOING=1 scripts/check_paligemma2_attention_pack_bf16_ab.sh`: pass.

Recommended profile after the shape-gated split change:

| Shape | Attention | Device sum |
| --- | ---: | ---: |
| 224 | 16.861 ms | 107.193 ms |
| 448 | 172.093 ms | 526.175 ms |

The 448 improvement in this profile should be treated as run-to-run variance
because the 448 split path remains scalar. The real promoted change is the
small, exact 224px split improvement.

## Execution Log: 448px Fixed Softmax Thread Retune

After split vectorization, the next F32-safe attention retune was the fixed
softmax block size for the 448px shape. This still leaves the attention contract
unchanged:

- QK remains F32 rocBLAS.
- Scores are still normalized in F32 before AV.
- AV remains F32 rocBLAS.
- Pack-to-BF16 remains after AV.

Sequential softmax sweep:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_softmax_fixed_profile_only \
  --samples 120 \
  --warmup 30 \
  --iters 1
```

| Shape | Dynamic | Fixed64 | Fixed128 | Fixed256 | Fixed512 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 224 | 0.131 ms | 0.126 ms | 0.130 ms | 0.144 ms | n/a | keep fixed64 |
| 448 | 1.840 ms | 1.897 ms | 1.819 ms | 1.849 ms | 1.939 ms | switch to fixed128 |

Sequential recommended attention phase profile after the change:

| Shape | Split | QK | Softmax | AV | Pack BF16 | Sum | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.099 | 0.159 | 0.127 | 0.263 | 0.035 | 0.684 | 0.661 |
| 448 | 0.421 | 1.735 | 1.827 | 2.269 | 0.096 | 6.348 | 6.509 |

Validation:

- `scripts/build_paligemma2_vit_hip_backend_probe.sh`: pass.
- `scripts/build_gemma_paligemma2_vit_hip.sh`: pass.
- `REBUILD=0 scripts/smoke_paligemma2_vit_hip_recommended.sh`: pass.
- `KEEP_GOING=1 scripts/check_paligemma2_attention_pack_bf16_ab.sh`: pass.
- Extended 448 stress set with `dark_radial.ppm`: pass.

Decision:

- Promote fixed128 for 448px softmax.
- Keep fixed64 for 224px.
- Treat this as a small F32-safe attention cleanup. It does not solve the main
  448px attention cost, which remains AV/QK/softmax as a group.

## Execution Log: Attention rocBLAS Solution Resweep

After the 448px softmax retune, rebuilt the attention solution cache with a
larger candidate cap and more samples:

```bash
REBUILD=0 \
MAX_SOLUTIONS=128 \
SAMPLES=160 \
WARMUP=40 \
ITERS=1 \
scripts/build_paligemma2_vit_attention_solution_cache.sh
```

Generated cache:

| Shape | QK solution | AV solution |
| --- | ---: | ---: |
| 224 | `-111` | `-1161823143` |
| 448 | `-110` | `-451` |

Sweep details:

| Shape | Phase | Selected | Closest alternative | Interpretation |
| --- | --- | ---: | ---: | --- |
| 224 | QK | `-111` at 0.157 ms | `-110` at 0.158 ms | Near-tie; current run flipped by ~0.001 ms. |
| 224 | AV | `-1161823143` at 0.248 ms | default `0` at 0.250 ms | Near default, still selected by the harness. |
| 448 | QK | `-110` at 1.924 ms | `-111` at 1.925 ms | Near-tie, same practical performance. |
| 448 | AV | `-451` at 2.322 ms | `-450` at 2.329 ms | Near-tie, generated cache stays on `-451`. |

Validation with the regenerated cache:

- Recommended smoke: pass.
- Default `check_paligemma2_attention_pack_bf16_ab.sh`: pass.
- Recommended return profile loads `qk=-111,av=-1161823143` for 224 and
  `qk=-110,av=-451` for 448.

Decision:

- Keep the regenerated cache.
- Treat the 224 `-110`/`-111` and 448 `-450`/`-451` differences as near-ties,
  not as durable algorithmic wins.
- The next attention work should not spend more time on rocBLAS solution-index
  micro-selection; larger gains require changing how QK/softmax/AV share data.

## Execution Log: Corrected Attention Phase Bench

The standalone `--attention_recommended_phase_profile_only` benchmark was
updated to mirror the current runtime backend:

- 224px split uses the promoted vector4 split.
- 448px split remains scalar.
- 224px QK uses the regenerated cache entry `-111`.
- 448px softmax uses fixed128.
- AV uses the regenerated cache entries.

Corrected profile:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_recommended_phase_profile_only \
  --samples 120 \
  --warmup 30 \
  --iters 1
```

| Shape | Split | QK | Softmax | AV | Pack BF16 | Sum | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.105 | 0.182 | 0.142 | 0.293 | 0.039 | 0.761 | 0.659 |
| 448 | 0.398 | 1.606 | 1.807 | 2.504 | 0.096 | 6.412 | 6.762 |

The phase sum is more useful than `total` for identifying work because HIP event
timing around very small kernels and rocBLAS launches can overlap/noise at this
scale. The 448px costs still point to the same target: AV, softmax, and QK must
be improved together to get a meaningful win.

## Execution Log: Online Head-Major Attention Prototype

Implemented a bench-only structural prototype:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_online_head_major_profile_only \
  --samples 40 \
  --warmup 10 \
  --iters 1
```

Design:

- Split Q/K/V to the same head-major layout used by the recommended path.
- One workgroup owns one `[head, query]` row.
- It computes QK, stable F32 softmax, and AV directly.
- It does not write the large `[heads, rows, rows]` score matrix to global
  memory.

Result:

| Shape | Baseline QK+softmax+AV | Online core | Relative | Max abs | RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.797 ms | 1.607 ms | 2.015x slower | 0.000001 | 0.000000 |
| 448 | 6.080 ms | 53.582 ms | 8.812x slower | 0.224916 | 0.000706 |

Decision:

- Do not promote this kernel.
- A per-query-row scalar online kernel is the wrong shape for 448px. It avoids
  score-matrix global writes, but it loses the matrix-instruction throughput of
  rocBLAS QK/AV and rereads V too much.
- The next structural question is whether a multi-query scalar tile can recover
  enough K/V reuse to matter before moving to matrix-instruction QK/AV.

## Execution Log: Online Multi-Query Attention Prototype

Implemented a second bench-only online attention prototype:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_online_multiquery_profile_only \
  --samples 40 \
  --warmup 10 \
  --iters 1
```

Design:

- Split Q/K/V to the same head-major layout as the recommended path.
- One workgroup owns one `[head, query tile]` instead of one `[head, query]`.
- Tested `mq2`, `mq4`, and `mq8`, where the suffix is the number of query rows
  handled by one workgroup.
- QK scores stay in LDS, softmax is F32, and AV reuses each loaded `V[key, dim]`
  across the query rows in the tile.

Result:

| Shape | Variant | Baseline core | Online core | Relative | Baseline total | Online total | Max abs | RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | mq2 | 0.736 ms | 2.529 ms | 3.434x slower | 0.658 ms | 2.627 ms | 0.000001 | 0.000000 |
| 224 | mq4 | 0.736 ms | 2.144 ms | 2.912x slower | 0.658 ms | 2.315 ms | 0.228223 | 0.000814 |
| 224 | mq8 | 0.736 ms | 1.769 ms | 2.403x slower | 0.658 ms | 2.004 ms | 0.349439 | 0.003424 |
| 448 | mq2 | 6.063 ms | 44.866 ms | 7.399x slower | 6.191 ms | 47.096 ms | 0.157954 | 0.001725 |
| 448 | mq4 | 6.063 ms | 28.033 ms | 4.623x slower | 6.191 ms | 29.378 ms | 0.138788 | 0.001641 |
| 448 | mq8 | 6.063 ms | 38.716 ms | 6.385x slower | 6.191 ms | 39.789 ms | 0.125536 | 0.001903 |

Decision:

- Do not promote any scalar online multi-query variant.
- Multi-query reuse helped the 448px online prototype: best online core dropped
  from roughly 52 ms for one query row to 28 ms for `mq4`.
- That is still about 4.6x slower than the current rocBLAS QK + fixed softmax +
  rocBLAS AV core, so removing global score writes is not enough.
- 224px became slower than the simpler one-query online kernel because the query
  tile adds register/LDS pressure without enough reuse to offset it.
- The mq4/mq8 paths also show softmax-sensitive max-abs outliers on the
  deterministic stress input, so this family is not runtime-safe without more
  numerical work.

Implication:

- The useful lesson is structural, not that this kernel should survive. Scalar
  online attention cannot compete with rocBLAS on this iGPU unless it also gets
  matrix-instruction-level QK/AV throughput.
- The next serious attention direction should stop expanding scalar online
  kernels and instead prototype a 16x16/16xN MFMA-style QK or AV tile, or reuse
  rocBLAS for QK/AV while only fusing the parts around softmax when the data
  movement win is measurable.

## Execution Log: QK BF16 WMMA Tile Retune

After the scalar online attention experiments, the next useful direction was to
retune the existing matrix-instruction QK path instead of adding more scalar
fusion. The benchmark-only `--attention_qk_bf16_wmma_profile_only` path now
sweeps multiple grouped-2D WMMA macro-tiles and uses the current recommended
F32 QK solution as the baseline.

Latest serial retune command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_qk_bf16_wmma_profile_only \
  --samples 60 \
  --warmup 20 \
  --iters 1
```

Key result:

| Shape | F32 QK | BF16 rocBLAS QK | Best WMMA tile | Best WMMA QK | Relative to F32 | 4x4 WMMA QK |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 224 | 0.159 ms | 0.213 ms | 2x8 | 0.107 ms | 0.672x | 0.109 ms |
| 448 | 1.684 ms | 2.727 ms | 2x8 | 1.369 ms | 0.813x | 1.487 ms |

All tested WMMA tiles match BF16 rocBLAS QK to about `1e-6`, so the tile change
does not add a new precision boundary. The only precision boundary remains the
existing opt-in BF16 QK choice versus F32 QK.

Runtime change:

- Use the 2x8 macro-tile for both 224px and 448px QK BF16 WMMA.
- Both packed and no-pack WMMA-QK attention paths now route through the same
  launch helper.

Return-profile check with `attention_pack_bf16`, QKV WMMA, attn-out WMMA, MLP-up
WMMA, MLP-down WMMA, and current attention/MLP caches:

| Shape | Attention mode | Attention | Device sum |
| --- | --- | ---: | ---: |
| 224 | F32 attention + pack | 17.135 ms | 104.582 ms |
| 224 | QK BF16 WMMA 2x8 + pack | 15.771 ms | 104.686 ms |
| 448 | F32 attention + pack | 164.348 ms | 479.260 ms |
| 448 | QK BF16 WMMA 2x8 + pack | 163.828 ms | 496.382 ms |

Validation:

- Default recommended generation smoke passed with QK BF16 WMMA disabled.
- 224 image-token validation reports `qk_impl=wmma2d2x8`,
  `max_abs_vs_f32_attn=0.289477`, `rms_vs_f32_attn=0.005729`.
- 448 image-token validation reports `qk_impl=wmma2d2x8`,
  `max_abs_vs_f32_attn=0.476363`, `rms_vs_f32_attn=0.004517`.
- These values are the known BF16-QK opt-in drift against F32 attention, not a
  tile-specific mismatch.

Decision:

- Use 2x8 for the opt-in `qk_bf16_wmma` path on both supported ViT image
  resolutions.
- Keep the path opt-in because the BF16 QK precision boundary is still
  model-visible on sensitive cases.
- For further attention work, AV/softmax still matter more than another QK-only
  retune: QK microbench improves, but the full image-token path does not show a
  reliable device-sum win in the current profile.

## Execution Log: AV BF16 WMMA Tile Retune

Retuned the existing opt-in AV BF16 WMMA path with the same grouped-2D
macro-tile sweep used for QK. This path keeps QK in F32, stores normalized
softmax scores as BF16, converts V to BF16, and computes AV with local WMMA.

Command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_av_bf16_wmma_profile_only \
  --samples 120 \
  --warmup 30 \
  --iters 1
```

Key result:

| Shape | F32 softmax + F32 AV | Existing 4x4 direct total | Best tile | Best direct total | Relative to F32 |
| --- | ---: | ---: | --- | ---: | ---: |
| 224 | 0.421 ms | 0.273 ms | 8x2 | 0.264 ms | 0.628x |
| 448 | 4.368 ms | 4.862 ms | 8x2 | 3.685 ms | 0.844x |

The 448px result is the important correction: the old 4x4 tile was worse than
the F32 softmax+AV baseline in the 120-sample run, while 8x2 is clearly faster.
All WMMA tiles match BF16 rocBLAS AV within about `2e-6`, so the tile change
does not add a new numerical boundary beyond the existing BF16 score/V storage.

Runtime change:

- Change the opt-in AV BF16 WMMA backend launch from 4x4 to 8x2.
- Profile output now reports `attn=av_bf16_wmma2d8x2`.

Return-profile check without `attention_pack_bf16` because the AV BF16 WMMA path
owns that precision boundary:

| Shape | Attention mode | Attention | Device sum |
| --- | --- | ---: | ---: |
| 224 | F32 attention | 19.335 ms | 118.558 ms |
| 224 | AV BF16 WMMA 8x2 | 14.874 ms | 112.195 ms |
| 448 | F32 attention | 187.462 ms | 563.975 ms |
| 448 | AV BF16 WMMA 8x2 | 178.199 ms | 562.179 ms |

Validation:

- Default 224/448 generation A/B passed.
- Extended 448 stress guardrail still fails the known-sensitive case:
  `dark_radial.ppm` with `Describe the image.`
  - F32 attention: `a close up of a circle pattern`
  - AV BF16 WMMA: `a blue swirl`

Decision:

- Keep the 8x2 runtime tile for the explicit `av_bf16_wmma` experiment.
- Do not add this path to recommended scripts or defaults. The speedup is real,
  but the BF16 normalized-score/V precision boundary still changes generation
  for a deterministic 448 stress case.
- The next safe attention direction should preserve F32 attention probabilities
  and F32 AV semantics, or isolate a fused data-movement win that passes the same
  `dark_radial.ppm` guardrail.

## Execution Log: F32 Grouped Softmax/AV Retune

Tested the safe F32-preserving route after the BF16 AV precision guardrail
failed. This keeps the current numeric contract:

- F32 QK scores
- F32 softmax probabilities
- F32 AV accumulation against F32 V

The existing grouped scalar softmax/AV prototype was retuned to sweep multiple
lane-group counts and updated to compare against the current recommended QK
solution, AV solution, and fixed softmax choices.

Command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_softmax_av_grouped_profile_only \
  --samples 80 \
  --warmup 20 \
  --iters 1
```

Result:

| Shape | Baseline QK+softmax+AV | Best grouped config | Best grouped total | Relative | Max abs | RMS |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 224 | 0.599 ms | groups=8 | 0.985 ms | 1.646x slower | 0.000001 | 0.000000 |
| 448 | 6.224 ms | groups=16 | 15.989 ms | 2.569x slower | 0.000001 | 0.000000 |

Decision:

- Do not promote the F32 grouped softmax/AV kernel.
- It is numerically safe, but still replaces rocBLAS AV with scalar weighted
  sums. The saved score write/read boundary is not large enough to offset that
  throughput loss.
- This closes the simple safe scalar-fusion path. Further safe attention work
  should either keep rocBLAS AV and reduce surrounding overhead, or require a
  true F32/preserved-precision matrix-instruction AV design rather than scalar
  loops.

## Execution Log: BF16 Attention Pack Vectorization

After the scalar F32 softmax/AV fusion path was exhausted, the next safe target
was the already-recommended `attention_pack_bf16` boundary. This path preserves
the current attention numeric contract:

- F32 QK
- F32 softmax probabilities
- F32 AV accumulation
- F32 attention output rounded once to BF16 for `attn_out`

The implementation adds a float4-load pack kernel for the fixed PaliGemma2
`qkv_dim=72` layout. It still performs four scalar `rocblas_bfloat16(float)`
conversions, so the BF16 rounding boundary is identical to the scalar pack.

Command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_pack_vectorized_profile_only \
  --samples 120 \
  --warmup 30 \
  --iters 1
```

Result:

| Shape | Scalar pack | Float4 pack | Relative | Max abs | RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.068 ms | 0.037 ms | 0.541x | 0.000000 | 0.000000 |
| 448 | 0.304 ms | 0.196 ms | 0.644x | 0.000000 | 0.000000 |

Recommended phase/profile checks after runtime integration:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_recommended_phase_profile_only \
  --samples 80 \
  --warmup 20 \
  --iters 1

REBUILD=0 SAMPLES=8 WARMUP=3 \
  scripts/profile_paligemma2_vit_hip_recommended.sh
```

| Shape | Recommended attention | Recommended device sum |
| --- | ---: | ---: |
| 224 | 17.911 ms | 114.798 ms |
| 448 | 176.237 ms | 522.228 ms |

Validation:

```bash
KEEP_GOING=1 scripts/check_paligemma2_attention_pack_bf16_ab.sh
```

- Default 224/448 generation A/B prompts passed.

Decision:

- Promote the float4 pack-to-BF16 helper into the recommended runtime path.
- Keep the scalar kernel as a fallback for non-multiple-of-4 head dimensions.
- Do not apply this to the deferred-softmax-scale experiment; that path has a
  different normalization boundary and remains opt-in.

## Execution Log: F32 AV Dim4 Matrix-Tile Prototype

Entered the higher-risk F32-preserving matrix-kernel direction after the simple
scalar softmax/AV fusion path failed. The first prototype targets AV only:

- QK remains F32 rocBLAS.
- Softmax remains the existing F32 fixed-row kernel.
- AV is replaced by a local F32 tiled HIP kernel for the 224px `rows=256`,
  `qkv_dim=72` shape.
- 448px falls back to rocBLAS because the custom kernel is slower there.

The dim4 kernel owns all 72 head dimensions in one row tile. Each thread
accumulates four adjacent dimensions using `float4` V loads, so each staged F32
score is reused across four output channels without changing the score or V
precision boundary.

Microbench command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_av_tiled_profile_only \
  --samples 80 \
  --warmup 20 \
  --iters 1
```

Best results:

| Shape | rocBLAS AV | Old scalar tile | Best dim4 tile | Relative |
| --- | ---: | ---: | ---: | ---: |
| 224 | 0.287 ms | 0.488 ms | 0.272 ms (`bm32/bk16`) | 0.950x |
| 448 | 2.499 ms | 8.389 ms | 3.816 ms (`bm32/bk16`) | 1.527x |

Runtime integration:

- Added `--hip_attention_av_f32_dim4 1` to the backend probe.
- Added `--paligemma_vit_hip_attention_av_f32_dim4 1` to the generation CLI.
- The runtime path is shape-gated to `rows=256, qkv_dim=72`; unsupported shapes
  keep the existing rocBLAS AV path.

Runtime profile on 224 with the recommended pack/QKV/attn-out/MLP flags:

| Mode | Attention | Device sum |
| --- | ---: | ---: |
| Recommended F32 pack | 17.978 ms | 113.987 ms |
| AV F32 dim4 + pack | 18.530 ms | 115.064 ms |

448 fallback check:

| Shape | Reported attention mode | Attention | Device sum |
| --- | --- | ---: | ---: |
| 448 | `f32_pack_bf16` | 186.586 ms | 574.069 ms |

Smoke:

```bash
./build/gemma_paligemma2_vit_hip \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_return_image_tokens 1 \
  --paligemma_vit_hip_attention_pack_bf16 1 \
  --paligemma_vit_hip_attention_av_f32_dim4 1 \
  ...
```

- 224 sample prompt still returned the expected church description.

Decision:

- Keep the F32 dim4 AV kernel as an opt-in diagnostic and a useful lower-bound
  for hand-written F32 AV throughput.
- Do not promote it to the recommended path. The microbench win at 224 does not
  survive full-layer scheduling, and 448 remains much slower than rocBLAS.
- The next high-risk/high-gain work should not be another scalar F32 AV tile.
  It needs either a real fused QK/softmax/AV design that avoids the global score
  matrix while preserving matrix-level throughput, or a return to MLP/QKV where
  BF16 WMMA already maps to hardware matrix instructions.

## Execution Log: BF16 WMMA Output Fusion

Returned to the BF16 WMMA-heavy projection path after the F32 AV dim4 runtime
prototype failed to improve the complete layer. This pass fuses post-processing
directly into local WMMA store sites:

- QKV WMMA now adds `qkv_bias[col]` before writing the F32 QKV tensor.
- MLP-up WMMA now applies `linear_0_bias`, the existing BF16 rounding boundary,
  GELU, and the final BF16 store before MLP-down.

The MLP-up fusion preserves the previous BF16 route exactly at the important
precision boundary:

```text
old:  F32 accum -> AddBiasGeluToBF16Kernel:
      BF16(acc + bias) -> GELU -> BF16

new:  F32 accum inside WMMA store:
      BF16(acc + bias) -> GELU -> BF16
```

Recommended profile command:

```bash
REBUILD=0 SAMPLES=12 WARMUP=4 \
  scripts/profile_paligemma2_vit_hip_recommended.sh
```

Result after fusion:

| Shape | QKV | Attention | MLP total | Device sum |
| --- | ---: | ---: | ---: | ---: |
| 224 | 20.907 ms | 16.802 ms | 51.766 ms | 104.251 ms |
| 448 | 79.641 ms | 171.861 ms | 195.867 ms | 500.750 ms |

Previous nearby recommended profile before this fusion pass:

| Shape | QKV | Attention | MLP total | Device sum |
| --- | ---: | ---: | ---: | ---: |
| 224 | 23.963 ms | 17.978 ms | 56.434 ms | 113.987 ms |
| 448 | 88.011 ms | 176.237 ms | 203.207 ms | 522.228 ms |

Validation:

```bash
bash scripts/build_paligemma2_vit_hip_backend_probe.sh
bash scripts/build_gemma_paligemma2_vit_hip.sh
KEEP_GOING=1 scripts/check_paligemma2_attention_pack_bf16_ab.sh
```

- Default 224/448 generation A/B prompts passed.

Decision:

- Keep these fusions in the recommended WMMA path.
- This is the right kind of high-risk work for the current codebase: it targets
  large matrix stages, removes whole tensor passes, and preserves the established
  BF16/F32 precision contracts.
- Next candidates in the same style are attention-output bias/residual fusion
  or MLP-down bias/residual fusion. Both should be measured because residual
  writes are smaller than QKV/MLP-up but happen in every layer.

## Execution Log: Attention-Output Residual Fusion

Continued the same WMMA-store fusion strategy on the next per-layer boundary.
The attention-output projection previously did:

```text
attn_out WMMA/rocBLAS -> AddBiasKernel -> AddResidualKernel
```

The recommended WMMA path now writes the final residual output directly:

```text
out = (WMMA_acc + attn_out_bias[col]) + residual[row, col]
```

This preserves the old F32 addition order. The non-WMMA fallback still uses the
separate bias and residual kernels.

MLP-down bias/residual fusion was also tested, but it was not retained. Adding
bias/residual reads to the MLP-down WMMA store increased the down kernel cost
more than it saved from removing the residual kernel.

Recommended profile after keeping attention-output fusion and leaving MLP-down
on the previous path:

```bash
REBUILD=0 SAMPLES=12 WARMUP=4 \
  scripts/profile_paligemma2_vit_hip_recommended.sh
```

| Shape | QKV | Attention | Attn out | MLP total | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 | 19.258 ms | 17.666 ms | 9.011 ms | 50.991 ms | 100.843 ms |
| 448 | 79.326 ms | 173.004 ms | 29.150 ms | 196.235 ms | 495.650 ms |

Previous profile after QKV and MLP-up output fusion:

| Shape | QKV | Attention | Attn out | MLP total | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 | 20.907 ms | 16.802 ms | 10.427 ms | 51.766 ms | 104.251 ms |
| 448 | 79.641 ms | 171.861 ms | 35.471 ms | 195.867 ms | 500.750 ms |

Validation:

```bash
bash scripts/build_paligemma2_vit_hip_backend_probe.sh
bash scripts/build_gemma_paligemma2_vit_hip.sh
KEEP_GOING=1 scripts/check_paligemma2_attention_pack_bf16_ab.sh
```

- Default 224/448 generation A/B prompts passed.

Decision:

- Keep attention-output bias/residual fusion in the recommended WMMA path.
- Do not keep MLP-down bias/residual fusion in the recommended path.
- The next similar candidate should be measured before integration; adding extra
  reads to a WMMA store is only worthwhile when the removed tensor pass is large
  enough to offset the added pressure.

## Execution Log: LayerNorm BF16 Store Fusion

Implemented the next low-risk memory-traffic fusion around LayerNorm BF16
boundaries. The previous BF16 route normalized into an F32 staging matrix, then
launched a full-tensor conversion:

```text
LayerNormKernel -> F32 pre_* matrix -> F32ToBF16Kernel -> BF16 GEMM input
```

The new `LayerNormToBF16Kernel` keeps the same F32 row-statistics and
normalization math, then stores the final normalized value directly as BF16:

```text
LayerNormToBF16Kernel -> BF16 GEMM input
```

This was wired into:

- ViT layer LN0 before QKV.
- ViT layer LN1 before MLP-up.
- Final encoder norm before `img_head`.
- The BF16-only return workspace, which no longer allocates the LN0/LN1 F32
  staging matrices.

Validation:

```bash
bash scripts/build_paligemma2_vit_hip_backend_probe.sh
bash scripts/build_gemma_paligemma2_vit_hip.sh
KEEP_GOING=1 scripts/check_paligemma2_attention_pack_bf16_ab.sh
```

- Default 224/448 generation A/B prompts passed.

Two recommended profile runs after the fusion:

| Shape | LN0 | LN1 | Final norm | Head | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 run 1 | 0.895 ms | 0.890 ms | 0.034 ms | 1.535 ms | 106.241 ms |
| 224 run 2 | 0.918 ms | 0.920 ms | 0.034 ms | 1.528 ms | 108.362 ms |
| 448 run 1 | 2.766 ms | 2.721 ms | 0.105 ms | 3.721 ms | 502.582 ms |
| 448 run 2 | 2.791 ms | 2.737 ms | 0.102 ms | 3.665 ms | 506.656 ms |

Previous nearby profile before this fusion:

| Shape | LN0 | LN1 | Device sum |
| --- | ---: | ---: | ---: |
| 224 | 1.346 ms | 1.335 ms | 100.843 ms |
| 448 | 6.291 ms | 6.262 ms | 495.650 ms |

Decision:

- The local LayerNorm stages improved as intended and scratch memory dropped by
  two F32 `[rows, model_dim]` matrices in the return workspace.
- Do not count this as an end-to-end performance win yet: in the same profile
  runs, QKV/MLP timing moved upward enough to hide the LayerNorm savings.
- Keep the implementation because it preserves the precision boundary, removes
  launches and workspace, and passed generation guardrails. Future reporting
  should separate this local cleanup from larger matrix-kernel wins.

## Execution Log: MLP-Up Shape-Specific WMMA Tile

Re-ran the existing matrix-level sweeps before changing another hot path.
Results on this `gfx1150` iGPU:

- MLP-down grouped-N: `wmma8` remains the best 448px tile; 224px `wmma12` is
  only about 1% faster than `wmma8`, so it is not worth a runtime branch.
- QKV and attention-output projection: `wmma2d8x4` remains best for 224px in the
  local sweep.
- MLP-up: 224px prefers `wmma2d4x8`, while 448px still prefers `wmma2d8x4`.

Implemented a shape-specific MLP-up launcher:

```text
rows=256  -> Bf16GemmWmma2DBiasGeluBF16Kernel<4, 8>
otherwise -> Bf16GemmWmma2DBiasGeluBF16Kernel<8, 4>
```

The precision contract is unchanged:

```text
F32 WMMA accum -> BF16(acc + linear_0_bias) -> GELU -> BF16
```

Recommended profile after this retune:

| Shape | MLP-up | MLP total | Device sum |
| --- | ---: | ---: | ---: |
| 224 | 27.674 ms | 54.057 ms | 106.037 ms |
| 448 | 103.808 ms | 200.389 ms | 495.546 ms |

Validation:

```bash
bash scripts/build_paligemma2_vit_hip_backend_probe.sh
bash scripts/build_gemma_paligemma2_vit_hip.sh
KEEP_GOING=1 scripts/check_paligemma2_attention_pack_bf16_ab.sh
```

- Default 224/448 generation A/B prompts passed.

Decision:

- Keep the MLP-up shape-specific tile. It is a small 224px matrix-level win,
  leaves 448px on the previously best tile, and does not change numerical
  boundaries.
- This also confirms that further gains need stage-specific tuning. One global
  WMMA macro-tile is not optimal across PaliGemma2's projection shapes.

## Execution Log: Attention Softmax Float4 Pass

Tested the next F32-safe attention cleanup: keep QK, softmax, and AV in the
same order, but vectorize the fixed-shape in-place softmax row passes with
`float4` loads/stores. This does not move normalization across AV and does not
change the BF16 pack boundary.

Standalone fixed-softmax bench:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_softmax_fixed_profile_only \
  --samples 60 \
  --warmup 15 \
  --iters 1
```

| Shape | Previous fixed softmax | Float4 fixed softmax |
| --- | ---: | ---: |
| 224 | 0.133 ms | 0.111 ms |
| 448 | 1.810 ms | 1.779 ms |

Runtime integration:

- `rows=256`: `SoftmaxRowsFixedFloat4Kernel<256, 64>`
- `rows=1024`: `SoftmaxRowsFixedFloat4Kernel<1024, 128>`
- fallback rows keep the generic scalar `SoftmaxRowsKernel`

Recommended profile after integration:

| Shape | Attention | MLP total | Device sum |
| --- | ---: | ---: | ---: |
| 224 | 17.378 ms | 54.413 ms | 106.413 ms |
| 448 | 171.242 ms | 205.498 ms | 500.843 ms |

Validation:

```bash
bash scripts/build_paligemma2_vit_hip_backend_probe.sh
bash scripts/build_gemma_paligemma2_vit_hip.sh
KEEP_GOING=1 scripts/check_paligemma2_attention_pack_bf16_ab.sh
```

- Default 224/448 generation A/B prompts passed.

Decision:

- Keep the float4 softmax pass. It is a small local attention win that preserves
  the F32 attention contract.
- Do not treat the end-to-end profile as a regression signal for this change:
  attention moved slightly down, while MLP-down varied upward in the same run.
- Further attention work still needs to target the expensive QK/AV GEMMs or a
  numerically conservative softmax+AV design; scalar cleanup is nearly
  exhausted.

## Execution Log: MLP-Down Wave-Count Runtime Probe

Added an explicit MLP-down grouped-N wave-count selector:

```text
probe:      --hip_mlp_down_wmma_waves N
generation: --paligemma_vit_hip_mlp_down_wmma_waves N
supported:  2, 4, 6, 8, 12, 16
default:    8
```

The existing `--hip_mlp_down_wmma8 1` / `--paligemma_vit_hip_mlp_down_wmma8 1`
flags still control whether the local MLP-down WMMA kernel is used. The new
integer only selects the grouped-N macro-tile width once that path is enabled.

Rechecked the current F32 softmax+AV fusion line first:

| Shape | Recommended QK+softmax+AV | Best grouped softmax+AV |
| --- | ---: | ---: |
| 224 | 0.563 ms | 0.962 ms (`g8`, with output drift) |
| 448 | 6.188 ms | 15.162 ms (`g16`) |

Decision: keep softmax+AV fusion out of runtime. It still loses badly against
rocBLAS AV plus the optimized softmax path.

Full return-profile MLP-down wave tests:

| Shape | Waves | MLP-down | MLP total | Device sum |
| --- | ---: | ---: | ---: | ---: |
| 448 | 8 | 86.186 ms | 195.913 ms | 491.544 ms |
| 448 | 12 | 88.338 ms | 201.012 ms | 495.845 ms |
| 448 | 16 | 89.019 ms | 201.300 ms | 498.778 ms |
| 224 | 8 | 25.058 ms | 54.739 ms | 109.375 ms |
| 224 | 12 | 24.275 ms | 54.010 ms | 105.366 ms |

Default recommended profile after parameterization:

| Shape | MLP-down | MLP total | Device sum |
| --- | ---: | ---: | ---: |
| 224 | 24.471 ms | 53.897 ms | 106.600 ms |
| 448 | 86.320 ms | 196.135 ms | 494.321 ms |

Validation:

```bash
bash scripts/build_paligemma2_vit_hip_backend_probe.sh
bash scripts/build_gemma_paligemma2_vit_hip.sh
REBUILD=0 SAMPLES=8 WARMUP=3 scripts/profile_paligemma2_vit_hip_recommended.sh
KEEP_GOING=1 scripts/check_paligemma2_attention_pack_bf16_ab.sh
```

- Default 224/448 generation A/B prompts passed.

Decision:

- Keep the new wave-count selector for controlled runtime experiments.
- Do not change the recommended default away from `wmma8`: 448 is the priority
  shape and clearly regresses at 12/16 waves.
- The 224 `wmma12` result is interesting but not enough to justify changing the
  shared default while the project is focused on 448.

## Execution Log: MLP-Down Fused Epilogue Retry

Revisited MLP-down bias/residual fusion after the earlier failed attempt. The
new implementation keeps the plain MLP-down launcher intact and adds a separate
fused epilogue path:

```text
acc = up_bf16 * linear_1
value = acc + linear_1_bias[col]
value = value + residual[row, col]
out[row, col] = value
```

The addition order intentionally matches the old two-kernel epilogue:
`AddBiasKernel` followed by `AddResidualKernel`. The only scheduling change is
that the WMMA kernel writes the final layer output once instead of writing the
raw down-projection, reading/writing it in bias, then reading/writing it again in
residual.

Added runtime flags:

```text
probe:      --hip_mlp_down_fused_residual 0|1
generation: --paligemma_vit_hip_mlp_down_fused_residual 0|1
```

Current recommended scripts enable it together with `wmma8`.

Paired profile, `samples=8`, `warmup=3`:

| Shape | Mode | MLP-down | Residual | MLP total | Device sum |
| --- | --- | ---: | ---: | ---: | ---: |
| 224 | unfused `wmma8` | 24.764 ms | 1.701 ms | 54.398 ms | 106.694 ms |
| 224 | fused `wmma8+residual` | 25.355 ms | 0.000 ms | 53.735 ms | 106.603 ms |
| 448 | unfused `wmma8` | 84.161 ms | 8.319 ms | 191.852 ms | 481.315 ms |
| 448 | fused `wmma8+residual` | 90.109 ms | 0.000 ms | 189.349 ms | 479.517 ms |

Validation:

```bash
bash scripts/build_paligemma2_vit_hip_backend_probe.sh
bash scripts/build_gemma_paligemma2_vit_hip.sh
REBUILD=0 MAX_GENERATED_TOKENS=8 scripts/smoke_paligemma2_vit_hip_recommended.sh
REBUILD=0 KEEP_GOING=1 MAX_GENERATED_TOKENS=8 \
  scripts/check_paligemma2_attention_pack_bf16_ab.sh
```

- Recommended 224/448 smoke passed.
- Attention pack A/B passed for 224/448 and the default three prompts.
- Direct fused/unfused generation spot-check passed for 224/448 on
  `Describe the image.`.

Decision:

- Keep the fused MLP-down epilogue in the recommended scripts.
- The win is modest because the fused WMMA store now performs the bias/residual
  loads, but it still removes two full activation-memory passes and saves about
  2.5 ms in the 448 paired profile.
- This is a conservative defaultable optimization: no BF16/F32 boundary moves,
  no attention numerics change, and the old unfused path remains available for
  A/B by setting the new flag to `0`.

## Execution Log: MLP-Down Schedule Recheck And F32 Attention Triage

After the fused MLP-down epilogue, rechecked whether a deeper MLP-down memory
schedule should replace grouped-N `wmma8`. The first sweep was run in parallel
only as a candidate filter; promising paths were then rerun serially.

Serial grouped-N recheck on 448:

| Waves | Time |
| ---: | ---: |
| 2 | 6.358 ms |
| 4 | 6.393 ms |
| 6 | 6.506 ms |
| 8 | 3.194 ms |
| 12 | 3.208 ms |
| 16 | 3.363 ms |

Serial grouped-2D recheck on 448:

| Tile | Time |
| --- | ---: |
| 2x2 | 6.544 ms |
| 2x4 | 4.638 ms |
| 2x8 | 4.426 ms |
| 4x2 | 5.107 ms |
| 4x4 | 4.567 ms |
| 4x8 | 4.223 ms |
| 8x2 | 5.031 ms |
| 8x4 | 4.533 ms |

Decision: keep grouped-N `wmma8`. The best grouped-2D tile is still about
32% slower than grouped-N on 448, and grouped-M remains far slower. MLP-down is
not worth more tile-shape retuning unless a genuinely different memory layout or
split-K design is introduced.

Attention triage with the current F32 recommended phase profile:

| Shape | Split | QK | Softmax | AV | Pack BF16 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.096 ms | 0.150 ms | 0.128 ms | 0.266 ms | 0.026 ms | 0.664 ms |
| 448 | 0.418 ms | 1.758 ms | 1.811 ms | 2.530 ms | 0.095 ms | 6.856 ms |

Rechecked the existing F32 tiled prototypes:

| Prototype | 224 | 448 | Decision |
| --- | ---: | ---: | --- |
| QK tiled `8x16x32` | 1.299 ms vs 0.149 ms rocBLAS | 22.356 ms vs 1.879 ms rocBLAS | Reject |
| AV dim4 best | 0.280 ms vs 0.252 ms rocBLAS | 3.818 ms vs 2.413 ms rocBLAS | Reject |

Also fixed the QK tiled benchmark baseline to use `RecommendedQkSolution(rows)`
instead of hardcoding `-110`, so 224px now reports the current cache entry
`-111`.

Decision:

- Close the current scalar/tiled F32 QK and AV prototypes.
- The next attention attempt needs either rocBLAS-level matrix throughput with a
  new algorithmic reason to win, or a numerically conservative fusion that
  removes memory traffic around softmax/AV without replacing the GEMM with a
  slower scalar tile.

## Execution Log: Softmax Recompute And Direct-QKV Attention

Tested a softmax variant that avoids writing intermediate unnormalized
`exp(score - max)` values back to the score matrix. It recomputes `exp` for the
final normalized store instead. This keeps the same F32 max/sum/normalize
structure but trades extra SFU work for one less large read/write pass.

Fixed-softmax profile:

| Shape | Current float4 | Best recompute | Error |
| --- | ---: | ---: | --- |
| 224 | 0.110 ms | 0.112 ms | exact |
| 448 | 1.787 ms | 1.787 ms | exact |

Decision: do not promote recompute softmax. It is exact in the benchmark but
does not beat the current float4 softmax kernel.

The more useful F32-preserving attention optimization is direct-QKV rocBLAS
input. Instead of launching `SplitQKVForAttention` and materializing contiguous
head-major Q/K/V buffers, rocBLAS reads each head directly from the interleaved
QKV projection:

```text
qkv[row, head, q/k/v, dim]
lda = heads * 3 * qkv_dim
batch_stride = 3 * qkv_dim
```

This keeps QK, softmax, and AV in F32. The tradeoff is explicit: QK/AV reads
are less contiguous, but the split kernel and three scratch writes disappear.

Benchmark phase profile:

| Shape | Split path total | Direct-QKV total | Direct output error |
| --- | ---: | ---: | --- |
| 224 | 0.699 ms | 0.587 ms | exact |
| 448 | 6.435 ms | 6.306 ms | exact |

Real backend paired profile, `samples=8`, `warmup=3`:

| Shape | Mode | Attention | Device sum |
| --- | --- | ---: | ---: |
| 224 | split QKV + pack | 16.616 ms | 101.843 ms |
| 224 | direct QKV + pack | 14.695 ms | 102.581 ms |
| 448 | split QKV + pack | 177.827 ms | 496.601 ms |
| 448 | direct QKV + pack | 166.373 ms | 482.628 ms |

The 224 device sum is dominated by unrelated QKV/MLP variation in this paired
run, but the attention slice itself improves. The 448 profile shows both
attention and total-device improvement.

Runtime flags:

```text
probe:      --hip_attention_direct_qkv 0|1
generation: --paligemma_vit_hip_attention_direct_qkv 0|1
```

Validation:

```bash
bash scripts/build_paligemma2_vit_hip_backend_probe.sh
bash scripts/build_gemma_paligemma2_vit_hip.sh
REBUILD=0 MAX_GENERATED_TOKENS=8 scripts/smoke_paligemma2_vit_hip_recommended.sh
REBUILD=0 KEEP_GOING=1 MAX_GENERATED_TOKENS=8 \
  scripts/check_paligemma2_attention_pack_bf16_ab.sh
```

- Recommended 224/448 smoke passed.
- Attention pack A/B passed for 224/448 and the default three prompts.
- Direct on/off generation spot-check passed for 224/448 on
  `Describe the image.`.

Decision:

- Promote direct-QKV to the recommended scripts with `attention_pack_bf16`.
- Keep it scoped to the F32 QK/F32 AV path. It is intentionally incompatible
  with QK-BF16, AV-BF16, AV-dim4, and deferred-softmax experiments because those
  paths expect different Q/K/V or score layouts.
