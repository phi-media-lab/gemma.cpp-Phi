# gemma.cpp-Phi: PaliGemma2 ViT HIP/gfx1150 Branch

This branch is an experimental fork of
[google/gemma.cpp](https://github.com/google/gemma.cpp) focused on
PaliGemma2 ViT inference acceleration on a local AMD ROCm/HIP system.

Branch:

```text
paligemma2-vit-hip-gfx1150
```

Target system used for development:

```text
AMD Ryzen AI 9 HX PRO 370 / Radeon 890M iGPU
ROCm 7.2.1
HIP arch gfx1150
Ubuntu 24.04
```

This is a research branch, not an upstream-ready general GPU backend. The goal
is to find and validate aggressive, shape-specialized acceleration paths for
PaliGemma2's image encoder while keeping the original CPU path available as the
correctness baseline.

## What This Branch Adds

- Experimental PaliGemma2 ViT HIP backend hook integrated at the image-token
  generation boundary.
- Resident-GPU/probe path for validating PaliGemma2 ViT stages.
- Custom HIP/RDNA WMMA kernels for:
  - QKV projection;
  - attention output projection;
  - MLP-up projection;
  - MLP-down projection with fused residual.
- F32 attention path with:
  - direct-QKV rocBLAS QK/AV input;
  - float4 F32 softmax;
  - BF16 pack back into decoder-ready image tokens.
- ROCm/HIP benchmark and profiling harnesses for PaliGemma2 ViT shapes.
- `rocprofv3` hardware-counter profiling workflow for gfx1150.
- Scripts for model conversion, build, smoke tests, A/B checks, and profiling.
- Design/roadmap notes under `docs/`.

## Current Recommended Path

The current 448px recommended PaliGemma2 ViT HIP path is:

```text
attention=f32_direct_qkv_pack_bf16
qkv=wmma2d8x4
attn_out=wmma2d8x4
mlp_up=wmma2d8x4
mlp_down=wmma12+residual
```

For 224px, MLP-down remains on the previous default `wmma8+residual`. For
448px, generation uses `--paligemma_vit_hip_mlp_down_wmma_waves 12`; the
standalone probe/benchmark tools use the shorter
`--hip_mlp_down_wmma_waves 12`.

The stable generation flags are wired through:

```text
--paligemma_vit_backend hip_probe
--paligemma_vit_hip_return_image_tokens 1
--paligemma_vit_hip_attention_pack_bf16 1
--paligemma_vit_hip_attention_direct_qkv 1
--paligemma_vit_hip_qkv_wmma2d 1
--paligemma_vit_hip_attn_out_wmma2d 1
--paligemma_vit_hip_mlp_up_wmma2d 1
--paligemma_vit_hip_mlp_down_wmma8 1
--paligemma_vit_hip_mlp_down_wmma_waves 12
--paligemma_vit_hip_mlp_down_fused_residual 1
```

## Progress Snapshot

Implemented and validated:

- PaliGemma2 HF-to-SBS conversion helpers for 224px and 448px SFP models.
- Experimental ViT HIP backend and probe executable.
- Device-resident projection/MLP kernels for the dominant ViT matrix shapes.
- Direct-QKV F32 attention path, avoiding the previous split Q/K/V scratch path.
- Float4 F32 softmax path.
- 448px `wmma12+residual` MLP-down default in the recommended scripts.
- Extended stress images for QK-BF16 and attention A/B checks.
- HIP event timing, roofline estimates, and hardware-counter profiling docs.

Recent clean 448px backend profile after the wave-count adjustment:

| Phase | Time |
| --- | ---: |
| QKV | 84.164 ms |
| Attention | 169.367 ms |
| Attention output | 30.884 ms |
| MLP | 191.717 ms |
| Device sum | 487.346 ms |

Counter-level finding from the real 448px backend:

| Kernel group | Fetch KiB | Write KiB | L2 hit | MemUnitBusy | Occupancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| QKV WMMA 2D 8x4 | 68,861 | 13,402 | 31.2% | 75.8% | 98.1% |
| Attention QK rocBLAS | 15,129 | 64,218 | 57.9% | 82.2% | 50.1% |
| Softmax float4 | 65,536 | 64,000 | 83.3% | 92.9% | 98.1% |
| Attention AV rocBLAS | 73,318 | 4,261 | 67.1% | 69.3% | 67.9% |
| MLP up WMMA 2D 8x4 + GELU | 87,733 | 8,412 | 46.5% | 78.2% | 98.9% |
| MLP down WMMA 12 + residual | 112,415 | 4,320 | 66.3% | 98.6% | 90.9% |

The key conclusion is that the largest remaining opportunity is attention
score-matrix traffic: QK writes the score matrix, softmax reads and writes it,
and AV rereads it.

## Quick Start On This Branch

Build the HIP probe:

```bash
scripts/build_paligemma2_vit_hip_backend_probe.sh
```

Build the generation binary with the HIP image-token backend:

```bash
scripts/build_gemma_paligemma2_vit_hip.sh
```

Build the HIP microbenchmark:

```bash
scripts/build_paligemma2_vit_hip_bench.sh
```

Run the current recommended smoke test:

```bash
REBUILD=0 MAX_GENERATED_TOKENS=8 \
  scripts/smoke_paligemma2_vit_hip_recommended.sh
```

Run the current recommended profile:

```bash
REBUILD=0 SAMPLES=8 WARMUP=3 ITERS=1 \
  scripts/profile_paligemma2_vit_hip_recommended.sh
```

Run the attention phase microbenchmark:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_recommended_phase_profile_only \
  --samples 120 --warmup 30 --iters 1
```

## Model Artifacts

Model weights are not committed.

The scripts expect converted `.sbs` files under `build/models/`, for example:

```text
build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs
build/models/paligemma2-3b-mix-448-hf/paligemma2-3b-mix-448-sfp-from-hf.sbs
```

Use the conversion helper as a starting point:

```bash
scripts/convert_paligemma2_hf_to_sbs.sh
```

The branch also includes:

```bash
scripts/download_paligemma2_sbs.sh
```

for environments that already use compatible SBS artifacts.

## Profiling

This branch uses two profiling layers:

- HIP event timing for clean phase timing and microbench comparisons.
- `rocprofv3 --pmc` raw hardware counters for gfx1150.

The local `rocprof-compute` package installs successfully, but its built-in
analysis configs do not support `gfx1150`; therefore raw `rocprofv3` CSV is the
working hardware-counter path.

Useful commands:

```bash
./build/paligemma2_vit_hip_bench --kernel_occupancy_report_only

rocprofv3 --list-avail --output-directory build/profiler_check \
  --output-format csv -- \
  ./build/paligemma2_vit_hip_bench --kernel_occupancy_report_only
```

Current profiling outputs are documented in:

- `docs/paligemma2_vit_hip_roofline_profile.md`
- `docs/paligemma2_vit_hip_hardware_counter_profile.md`

## Roadmap

### 1. Make Counter Profiling A Gate

Future kernel changes should be promoted only after:

```text
clean HIP-event timing
rocprofv3 --pmc FETCH_SIZE
rocprofv3 --pmc WRITE_SIZE
rocprofv3 --pmc L2CacheHit
rocprofv3 --pmc MemUnitBusy
rocprofv3 --pmc OccupancyPercent
```

### 2. Reduce F32 Attention Score Traffic

Highest-upside next step:

```text
QK writes score matrix
softmax reads/writes score matrix
AV rereads score matrix
```

The next attention kernel should reduce this score-matrix traffic while
preserving F32 behavior. A simple replacement for rocBLAS QK/AV is not enough
unless it also wins timing and counter behavior.

Likely directions:

- fused softmax/AV;
- tiled online attention;
- shape-specialized F32 attention for PaliGemma2 448;
- direct-QKV layout-aware attention.

### 3. Keep rocBLAS QK/AV As Baseline

Current custom scalar/tiled QK/AV prototypes do not beat rocBLAS. Keep cached
rocBLAS QK/AV solutions as the correctness and performance baseline until a
custom attention path changes the memory traffic profile.

### 4. Continue Incremental WMMA Work

Projection and MLP counters show high occupancy but weak or moderate L2 hit.
Future work should focus on:

- global-load layout;
- LDS staging;
- tile reuse;
- fused epilogues;
- double-buffering only if it improves counters and clean timing.

### 5. Keep 448 `wmma12+residual`

448px MLP-down `wmma12+residual` is now the recommended branch default. 224px
stays on the previous default until a clean total-profile win is shown.

## Documentation

Branch-specific docs:

- `docs/paligemma2_acceleration_plan.md`
- `docs/paligemma2_vit_hip_next_stage_design.md`
- `docs/paligemma2_vit_hip_roofline_profile.md`
- `docs/paligemma2_vit_hip_hardware_counter_profile.md`

Upstream gemma.cpp docs still apply for the original CPU runtime, file format,
and general model support:

- `DEVELOPERS.md`
- `API_SERVER_README.md`
- `docs/CONTRIBUTING.md`

## Upstream Notice

The upstream project is `google/gemma.cpp`. This branch is a fork-specific
research branch for PaliGemma2 ViT HIP acceleration on gfx1150. It should be
treated as experimental until the backend is generalized, build integration is
cleaned up, and the profiling gate is automated.
