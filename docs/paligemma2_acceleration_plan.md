# PaliGemma2 Acceleration Plan

Snapshot date: 2026-05-10

This document records the local investigation for aggressively accelerating
PaliGemma2 inference on this machine. It is intentionally pragmatic: start from
the paths that can be measured and changed in `gemma.cpp`, then expand to larger
GPU/NPU work only when the numbers justify it.

## Goal

Use the local machine as aggressively as possible for PaliGemma2 inference,
without assuming that a generic GPU backend will automatically win.

Primary target:

- Faster PaliGemma2 image-conditioned inference, especially 448px models where
  the image prefix is large.

Secondary targets:

- Preserve the existing CPU decoder path unless a GPU path proves faster.
- Keep optimization boundaries measurable and reversible.
- Prefer specialized kernels for known PaliGemma2 shapes before broad rewrites.

## Local Machine

CPU:

- AMD Ryzen AI 9 HX PRO 370 w/ Radeon 890M.
- 12 physical cores / 24 logical CPUs.
- Heterogeneous layout observed locally:
  - CPUs 0-3 and 12-15: 4 high-frequency cores, max about 5.16 GHz, 16 MiB L3.
  - CPUs 4-11 and 16-23: 8 lower-frequency cores, max about 3.29 GHz, 8 MiB L3.
- Single NUMA node.
- AVX-512 and BF16 are available.
- Current build emits native BF16 instructions such as `vdpbf16ps`.

GPU:

- Integrated Radeon 890M, ROCm-visible as `gfx1150`.
- 16 hardware/marketing compute units. HIP currently reports
  `multiProcessorCount=8` for this `gfx1150` runtime, which is the value used by
  the experimental cache key.
- Wavefront size 32, max clock reported as 2900 MHz.
- `rocm-smi` reports 4 GiB VRAM.
- Kernel log reports:
  - `VRAM: 4096M`
  - `GTT: 29952M`
- `rocminfo` reports APU memory properties and global allocatable pools around
  30 GiB.

NPU:

- XDNA device exists at `/sys/class/accel/accel0`.
- Kernel driver `amdxdna` is loaded.
- No usable user-space NPU toolchain was found in the current shell.

Observed default CPU power policy before benchmarking setup:

- `amd-pstate-epp` driver.
- Governor is `powersave`.
- Energy performance preference is `balance_power`.
- Benchmarking should switch to performance mode before collecting numbers.

## Current Codebase State

`gemma.cpp` is currently a CPU inference runtime. There is no existing
CUDA/ROCm/Vulkan/NPU backend for inference.

The local build is working:

- `cmake --build --preset make -j $(nproc)` succeeds.
- Main targets `gemma` and `libgemma` build.
- A local PaliGemma2 3B 224 SBS produced from HF safetensors is available at
  `build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs`.
- A local PaliGemma2 3B 448 SBS produced from HF safetensors is available at
  `build/models/paligemma2-3b-mix-448-hf/paligemma2-3b-mix-448-sfp-from-hf.sbs`.
- Real image-conditioned PaliGemma2 224 and 448 smoke tests and CPU baselines
  now run.

## PaliGemma2 Architecture in This Repo

PaliGemma2 combines:

- SigLIP-So400m/14 style ViT image encoder.
- Gemma 2 decoder.

Local config support:

- PaliGemma2 3B 224
- PaliGemma2 3B 448
- PaliGemma2 10B 224
- PaliGemma2 10B 448

ViT config highlights from `gemma/configs.cc`:

- Model dim: 1152
- FF hidden dim: 4304
- Heads: 16
- KV heads: 16
- QKV dim: 72
- Layers: 27
- Patch width: 14
- Image tokens:
  - 224px: 256 tokens
  - 448px: 1024 tokens

Decoder mapping:

- PaliGemma2 3B uses the Gemma2 2B decoder config.
- PaliGemma2 10B uses the Gemma2 9B decoder config.

## Important Local Hotspots

### ViT Attention

File: `gemma/vit.cc`

PaliGemma currently uses the direct path:

- `VitAttention::DotSoftmaxWeightedSum()`
- Work unit is roughly head x token.
- It computes Q.K, softmax, and weighted V directly.

There is also a matrix-based path:

- `VitAttention::DotSoftmaxWeightedSumMatrix()`
- It is currently selected only for `PromptWrapping::GEMMA_VLM`.
- PaliGemma does not use it today.

This is the first high-value PaliGemma-specific experiment. The 448px path has
1024 image tokens, so ViT attention is much more important than for 224px.

### Decoder Attention

Files:

- `gemma/gemma.cc`
- `gemma/attention.cc`
- `gemma/flash_attention.cc`

Current behavior:

- `GemmaAttention()` can choose old attention or FlashAttention.
- `gemma/gemma.cc` passes `kAttentionUseOld` when `HWY_NATIVE_DOT_BF16` is true.
- On this machine, native BF16 is available, so the decoder currently takes the
  old attention path.
- A local experimental CLI flag, `--force_flash_attention`, can force the
  FlashAttention path for decoder attention without changing the default.

This is the second high-value experiment: compare old attention and
FlashAttention on this Zen 5 / AVX-512 BF16 CPU, especially for PaliGemma prefix
prefill.

### PaliGemma Prefix Prefill

File: `gemma/run.cc`

When an image is provided and wrapping is PaliGemma:

- `prefix_end = prompt_size`
- `runtime_config.prefill_tbatch_size = prompt_size`

The comment says online softmax is still on the roadmap. This means the full
image/text prefix is processed together. For 448px models, the 1024 image tokens
make prefill a large and important workload.

### Image Preprocessing

Files:

- `paligemma/image.cc`
- `paligemma/image.h`

The image path is simple PPM loading, resizing, normalization, and patch
extraction. This is not the first target unless profiling shows it matters.

## Meaning of 4 GiB VRAM vs 30 GiB GTT

The integrated GPU uses system memory. It does not have discrete-board
GDDR/HBM. The 4 GiB reported by `rocm-smi` is the dedicated UMA/VRAM window
exposed by firmware/driver.

The same system also exposes about 30 GiB of GTT/shared GPU-accessible memory.
This matters for custom ROCm/HIP work:

- A generic framework may refuse or degrade because it sees only 4 GiB VRAM.
- A custom kernel can be designed around ROCm/HSA shared memory pools.
- This helps capacity, but it does not create discrete-GPU memory bandwidth.

Practical implication:

- Do not start by changing BIOS UMA size.
- First test selective iGPU offload using the existing shared/GTT capacity.
- Treat GTT as a capacity enabler, not a guaranteed speed win.

## Optimization Strategy

### Track A: Measurement Baseline

Purpose: get stable numbers before changing kernels.

Required setup:

- Obtain PaliGemma2 `.sbs` weights and tokenizer if needed.
- Switch CPU policy to performance mode.
- Run with `--verbosity 3` to expose matmul autotune choices.
- Keep prompts and images fixed.
- Measure both 224 and 448 variants if weights are available.

Suggested CPU policy commands:

```bash
sudo /usr/lib/linux-oem-6.17-tools-6.17.0-1020/cpupower frequency-set -g performance
for f in /sys/devices/system/cpu/cpufreq/policy*/energy_performance_preference; do
  echo performance | sudo tee "$f" >/dev/null
done
```

Candidate runs:

```bash
# all physical cores, default topology
./build/gemma --weights <model.sbs> --image_file <image.ppm> --prompt "<prompt>" --verbosity 3

# high-frequency 4-core cluster only
taskset -c 0-3 ./build/gemma --weights <model.sbs> --image_file <image.ppm> --prompt "<prompt>" --verbosity 3

# first hardware thread of all physical cores
taskset -c 0-11 ./build/gemma --weights <model.sbs> --image_file <image.ppm> --prompt "<prompt>" --verbosity 3

# test pin/spin explicitly
taskset -c 0-11 ./build/gemma --weights <model.sbs> --image_file <image.ppm> --prompt "<prompt>" --pin 1 --spin 1 --verbosity 3
```

Metrics to capture:

- Startup/load time.
- Image encoder time.
- Prefix prefill time.
- First token latency.
- Decode tokens/sec.
- MatMul best configs printed at verbosity 3.
- CPU frequency and temperature if available.

Notes:

- The local `/usr/bin/perf` wrapper currently does not provide a working kernel
  `perf` binary for this OEM kernel.
- `/proc/sys/kernel/perf_event_paranoid` is `4`, so hardware perf requires sudo
  sysctl changes even after a working perf binary is available.
- Use the built-in Highway profiler first.

### Track B: CPU Specialization

This is the highest-confidence route because the current runtime is CPU-native
and already uses Highway AVX-512 BF16.

Experiments:

1. PaliGemma ViT matrix attention
   - Change PaliGemma to try `DotSoftmaxWeightedSumMatrix()`.
   - Compare 224 and 448.
   - Watch memory use for the 1024 x 1024 score matrix.

2. Decoder old attention vs FlashAttention
   - Add a runtime or compile-time switch for the `HWY_NATIVE_DOT_BF16` old-path
     override.
   - Compare PaliGemma prefix prefill and decode.

3. Heterogeneous-core scheduling
   - Benchmark `taskset -c 0-3`, `0-11`, `4-11`, and default.
   - Avoid assuming `--max_clusters=1` means fast cores, because topology code
     sorts clusters by worker count.
   - Consider a local topology ranking change only after taskset proves it.

4. Fixed-shape matmul benchmark
   - Build a small benchmark target independent of the full test suite.
   - Use PaliGemma2 ViT shapes:
     - Patch embedding: M=256/1024, K=588, N=1152
     - QKV: M=256/1024, K=1152, N=3456
     - Attention output: M=256/1024, K=1152, N=1152
     - MLP up: M=256/1024, K=1152, N=4304
     - MLP down: M=256/1024, K=4304, N=1152

### Track C: Selective iGPU Offload

This is the most promising GPU path, but it should start small.

Do not begin with full decoder GPU inference. The decoder has per-token latency,
kernel launch, synchronization, and shared-memory bandwidth issues. The CPU path
is already strong for batch=1 decode.

Start with ViT/image encoder offload:

- Keep CPU decoder unchanged.
- Move only image encoder computation to HIP/ROCm.
- Return final `image_tokens` to CPU.
- This keeps CPU/GPU synchronization to one coarse boundary per image.

Why this fits PaliGemma2:

- ViT shapes are fixed.
- 448px has 1024 image tokens, enough parallel work to amortize GPU overhead.
- Dense matmul and attention are more GPU-friendly than token-by-token decode.
- GTT/shared memory capacity means the experiment is not blocked by the 4 GiB
  `rocm-smi` number.

Prototype stages:

1. HIP environment check
   - Install or locate `hipcc`, rocBLAS, and ROCm profiling tools.
   - Compile a tiny `gfx1150` HIP kernel.
   - Verify allocations above 4 GiB if needed, but keep the first prototype
     small.

2. rocBLAS matmul microbenchmark
   - Measure the ViT matmul shapes listed in Track B.
   - Compare against current CPU `CallMatMul`.

3. ViT attention prototype
   - Start with QK + softmax + AV for 1024 tokens, 16 heads, head dim 72.
   - Compare direct HIP implementation vs rocBLAS-backed QK/AV.

4. Integrate as optional path
   - Add an explicit flag, for example `--paligemma_vit_backend=cpu|hip`.
   - Keep CPU as default until the GPU path wins reliably.

Acceptance criteria for continuing GPU work:

- 448px ViT-only path is materially faster than CPU after including transfer and
  synchronization.
- CPU decoder speed does not regress when the iGPU was used for image encoding.
- Memory pressure does not cause swapping or visible system instability.

### Track D: NPU/XDNA

This is a high-risk research branch.

Reasons:

- gemma.cpp has no XDNA backend.
- The dynamic decoder is not a natural first target for NPU.
- Tooling likely expects graph-level flows such as ONNX/Ryzen AI rather than
  direct C++ kernel integration.

Possible target:

- Static ViT image encoder graph only.

Do not prioritize this until CPU and iGPU data are available.

## First Implementation Plan

Step 1: Baseline and profiling

- Acquire PaliGemma2 weights.
- Switch CPU policy to performance.
- Run fixed 224/448 image prompts.
- Record built-in profiler output and matmul configs.

Step 2: CPU ViT attention experiment

- Add a local switch for PaliGemma ViT direct vs matrix attention.
- Benchmark 224 and 448.
- Keep the patch small and easy to revert.

Step 3: CPU decoder attention experiment

- Add a switch for old attention vs FlashAttention under native BF16.
- Benchmark PaliGemma prefix prefill and decode.

Step 4: Core-layout experiment

- Use `taskset` first.
- Decide whether topology code needs a machine-specific ranking or a runtime
  affinity recommendation.

Step 5: HIP feasibility prototype

- Install missing ROCm development tools if needed.
- Build a standalone ViT-shape matmul benchmark.
- Only integrate HIP into gemma.cpp after standalone numbers justify it.

## Decision Table

| Finding | Next action |
| --- | --- |
| CPU matrix ViT attention wins | Make it optional or default for PaliGemma after correctness checks. |
| FlashAttention wins under BF16 | Remove or gate the native-BF16 old-attention override only after 448px/long-context data shows a clear win. |
| 4 fast cores beat all cores for decode | Add recommended launch profile or topology handling. |
| 12 physical cores beat 4 fast cores for prefill | Only pursue phase-specific policy with separate phase-aware thread pools/contexts; naive in-process affinity switching regresses. |
| iGPU ViT 448 wins clearly | Build optional HIP ViT backend. |
| iGPU only wins microbench but not end-to-end | Keep GPU work out of default path. |
| NPU toolchain needs graph export | Park NPU until after CPU/GPU paths. |

## Risks

- The iGPU shares memory bandwidth with the CPU; concurrent CPU decode and iGPU
  work may interfere.
- GTT capacity does not imply high bandwidth.
- BIOS UMA changes may help framework capacity checks but will not change the
  underlying memory bandwidth.
- Test targets currently have CMake/GTest/Highway naming issues when enabling
  the full test suite, so small standalone benchmarks are safer.
- Without real PaliGemma2 weights, only microbenchmarks can be measured.

## Open Requirements

- Representative test image(s), preferably PPM or a repeatable conversion path.
- Fixed prompt set for 224 and 448 runs.
- ROCm development tools for HIP prototype:
  - `hipcc`
  - rocBLAS
  - ROCm profiler tools

## Execution Log

### 2026-05-09 CPU Microbenchmark Setup

Completed:

- Switched CPU governor to `performance`.
- Switched CPU energy performance preference to `performance`.
- Added standalone benchmark target:
  - `paligemma2_vit_matmul_bench`
  - Source: `experimental/paligemma2_vit_matmul_bench.cc`
- The benchmark covers fixed PaliGemma2 ViT matmul shapes for 224px and 448px,
  including:
  - patch embedding
  - QKV projection
  - attention output projection
  - MLP up/down projections
  - image-token head projection for 3B/10B decoders
  - F32 per-head QK and AV attention matmul shapes
- Logs are in `build/paligemma2_bench_logs/`.

Commands used:

```bash
cmake --preset make
cmake --build build --target paligemma2_vit_matmul_bench -j $(nproc)

./build/paligemma2_vit_matmul_bench --samples 1 --print_best 0
./build/paligemma2_vit_matmul_bench --samples 3 --print_best 0
taskset -c 0-3 ./build/paligemma2_vit_matmul_bench --samples 3 --print_best 0
taskset -c 0-11 ./build/paligemma2_vit_matmul_bench --samples 3 --print_best 0
taskset -c 4-11 ./build/paligemma2_vit_matmul_bench --samples 3 --print_best 0
```

Key 448px median results:

| Core set | QKV BF16 | MLP up BF16 | MLP down BF16 | QK/head F32 | AV/head F32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all default | 8.218 ms | 10.253 ms | 11.638 ms | 0.810 ms | 0.452 ms |
| fast cores 0-3 | 13.463 ms | 17.147 ms | 17.390 ms | 1.377 ms | 0.313 ms |
| physical 0-11 | 9.390 ms | 11.386 ms | 11.709 ms | 0.917 ms | 0.487 ms |
| slow cores 4-11 | 10.241 ms | 12.624 ms | 14.541 ms | 1.158 ms | 0.261 ms |

Initial observations:

- Large BF16 ViT projection shapes already reach roughly 0.9-1.2 TFLOPS on CPU.
- Default all-core topology is generally strongest for large BF16 shapes.
- Four high-frequency cores alone are not enough for ViT projection throughput.
- The F32 per-head attention shapes behave differently from large BF16 GEMM:
  `vit_av_head_448_f32` was fastest on the 8 lower-frequency cores in this run.
- This supports treating ViT attention as a separate optimization target rather
  than assuming projection-matmul behavior predicts attention behavior.

Model status:

- Local PaliGemma2 3B 224 `.sbs` is now available at
  `build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs`.
- Real end-to-end PaliGemma2 inference can now be benchmarked.

### 2026-05-09 HF `.sbs` Conversion

Completed:

- Added `scripts/download_paligemma2_sbs.sh` for the Kaggle direct `.sbs`
  path. This is the shortest route when Kaggle auth and license acceptance are
  available.
- Added `scripts/convert_paligemma2_hf_to_sbs.sh` for the Hugging Face route:
  download safetensors, build conversion extensions, and produce a local SBS.
- Built Bazel Python extensions:
  - `//compression/python:compression`
  - `//python:configs`
- Downloaded HF weights from `google/paligemma2-3b-mix-224`.
- Downloaded `tokenizer.model` from `google/paligemma-3b-mix-224`, because the
  PaliGemma2 HF repo provides `tokenizer.json` but gemma.cpp conversion expects
  a SentencePiece model.
- Generated:
  - `build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs`
  - `build/models/paligemma2-3b-mix-224-hf/metadata.csv`

Conversion fixes required:

- `python/configs.cc`: removed a stale `PALIGEMMA_448` binding that no longer
  exists in the C++ `Model` enum.
- `python/convert_from_safetensors.py`: set `cols_take_extra_dims` for
  `img_emb_kernel` so the ViT patch embedding kernel is stored as `1152 x 588`,
  matching the runtime reader.
- `gemma/gemma.cc`: fixed generated-token position accounting for full-prefix
  PaliGemma prompts. Without this, `run.cc` aborts after prefill with
  `pos == abs_pos`.

Smoke-test command:

```bash
./build/gemma \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 16 \
  --verbosity 1
```

Smoke-test result:

- Image token generation: 715 ms.
- Prefill: 1982 ms for 261 prompt tokens, 131.65 tokens/s.
- Generate: 829 ms for 16 tokens, 19.29 tokens/s.
- Output began: `A long-shot view of a large white church.`

### 2026-05-09 End-to-End CPU Baselines

Environment requirements:

- `powerprofilesctl` / platform profile must be `performance`.
- CPU governor must be `performance`.
- CPU energy performance preference must be `performance`.
- If any of these fall back to `power-saver`, `low-power`, `powersave`, or
  `power`, PaliGemma2 timings can regress by roughly 40-80%.
- Use `scripts/set_performance_mode.sh` before benchmark runs.

Reusable benchmark script:

```bash
SUDO_PASSWORD=yes scripts/set_performance_mode.sh
scripts/benchmark_paligemma2_cpu.sh
TOKENS=128 scripts/benchmark_paligemma2_cpu.sh
EXTRA_GEMMA_ARGS="--force_flash_attention 1" scripts/benchmark_paligemma2_cpu.sh
```

Median results, 3B 224, prompt `Describe the image.`, top-1 deterministic:

| Case | Samples | Image ms | Prefill ms | TTFT ms | Generate ms | Gen tok/s | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 default | 3 | 617 | 1877 | 1939 | 1612 | 19.84 | 4.67 |
| 32 physical 0-11 | 3 | 626 | 1876 | 1938 | 1629 | 19.63 | 4.70 |
| 32 slow 4-11 | 3 | 326 | 1597 | 1672 | 1978 | 16.17 | 4.49 |
| 32 fast 0-3 | 3 | 430 | 1986 | 2072 | 2271 | 14.09 | 5.16 |
| 128 default | 2 | 610 | 1934 | 1996 | 6505 | 19.68 | 9.65 |
| 128 physical 0-11 | 2 | 620 | 1962 | 2027 | 6512 | 19.66 | 9.73 |
| 128 slow 4-11 | 2 | 328 | 1645 | 1720 | 8312 | 15.43 | 10.93 |

Operational conclusions:

- For low-latency short outputs, `taskset -c 4-11` wins because ViT and prefix
  prefill are substantially faster.
- For longer generation, default scheduling or `taskset -c 0-11` wins because
  decode throughput is about 19.6-19.8 tokens/s versus 15-16 tokens/s on cores
  4-11.
- Cores 0-3 alone are not enough; they lose both wall time and decode.

### 2026-05-09 Decoder FlashAttention Experiment

Completed:

- Added `--force_flash_attention` as an opt-in experiment flag.
- Default behavior remains unchanged: AVX-512 BF16 targets still use the old
  decoder attention path unless this flag is set.

Median results, 3B 224, prompt `Describe the image.`, top-1 deterministic:

| Case | Samples | Image ms | Prefill ms | TTFT ms | Generate ms | Gen tok/s | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 default old attention | 3 | 620 | 1910 | 1976 | 1644 | 19.46 | 4.79 |
| 32 force FlashAttention | 3 | 613 | 1873 | 1939 | 1638 | 19.53 | 4.73 |
| 128 default old attention | 3 | 620 | 1956 | 2022 | 6488 | 19.73 | 9.72 |
| 128 force FlashAttention | 3 | 607 | 1929 | 1996 | 6563 | 19.50 | 9.74 |

Conclusion:

- FlashAttention is correct for this PaliGemma2 3B 224 smoke path.
- It slightly improves 32-token wall time, mostly through prefill/TTFT.
- It does not improve 128-token wall time because decode throughput is flat or
  slightly worse.
- This is not enough evidence to change the default for 224px. Keep the flag for
  448px and longer-prefix experiments, where attention cost should matter more.

Commands:

```bash
./build/gemma \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 32 \
  --top_k 1 \
  --deterministic 1 \
  --force_flash_attention 1 \
  --verbosity 1
```

### 2026-05-09 PaliGemma2 3B 448 Baseline

Completed:

- Fixed `scripts/convert_paligemma2_hf_to_sbs.sh` so it can infer output paths
  and model specifiers for 224/448 repositories instead of hardcoding 224.
- Downloaded and converted `google/paligemma2-3b-mix-448` from HF safetensors.
- Generated:
  - `build/models/paligemma2-3b-mix-448-hf/paligemma2-3b-mix-448-sfp-from-hf.sbs`
  - `build/models/paligemma2-3b-mix-448-hf/metadata.csv`
- Added `EXTRA_GEMMA_ARGS` to `scripts/benchmark_paligemma2_cpu.sh` so
  experimental flags such as `--force_flash_attention 1` can be reused.

Smoke-test result, 3B 448, 16-token cap:

- Image token generation: 11437 ms.
- Prefill: 5445 ms for 1029 prompt tokens, 188.97 tokens/s.
- Generate: 952 ms for 16 tokens, 16.80 tokens/s.
- Output began: `A large building with two towers stands tall in the distance.`

32-token results:

| Case | Samples | Image ms | Prefill ms | TTFT ms | Generate ms | Gen tok/s | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default old attention | 2 | 8865 | 5688 | 5753 | 1814 | 17.66 | 16.94 |
| default force FlashAttention | 2 | 8531 | 4537 | 4601 | 1847 | 17.33 | 15.47 |
| `taskset -c 4-11` old attention | 2 | 5235 | 7922 | 8004 | 2214 | 14.48 | 15.95 |
| `taskset -c 4-11` force FlashAttention | 2 | 5188 | 5825 | 5909 | 2487 | 12.87 | 14.07 |

128-token cap results, both using `--force_flash_attention 1`:

| Case | Samples | Image ms | Prefill ms | TTFT ms | Generate ms | Gen tok/s | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 2 | 10044 | 4977 | 5041 | 6751 | 17.92 | 22.34 |
| `taskset -c 4-11` | 2 | 5256 | 5942 | 6028 | 8697 | 13.91 | 20.47 |

Conclusions:

- PaliGemma2 448 is qualitatively different from 224: image encoding alone is
  about 5-11 s on this machine, and prefix prefill is about 4.5-8.0 s.
- `--force_flash_attention 1` is useful for 448 prefix prefill. In the default
  topology it reduced 32-token prefill from about 5.7 s to about 4.5 s.
- `taskset -c 4-11` is very strong for the 448 image encoder, cutting image
  token generation from about 8.5-10.5 s to about 5.2 s.
- The same `taskset -c 4-11` hurts decoder throughput. Even so, for this image
  prompt it still wins wall time because image encoding dominates.
- The theoretical best near-term CPU policy is not one static CPU set. It is:
  image encoder on `4-11`, decoder prefix with FlashAttention on the default
  topology, and decode on the default topology. Based on measured components,
  this would target roughly 11-12 s for 32 tokens and about 17 s for the
  121-token EOS run, before accounting for any phase-switch overhead.
- The previous naive in-process affinity experiment already proved that simply
  changing affinity after full-machine thread pools are created is invalid.
  A correct implementation needs phase-aware `ThreadingContext`/pool creation or
  another mechanism that aligns worker topology with the intended CPU set.

### 2026-05-09 Phase-Specific Image Threading

Completed:

- Added opt-in image-only threading controls:
  - `--image_skip_lps`
  - `--image_max_lps`
- These create a separate `ThreadingContext` and `MatMulEnv` only for
  `GenerateImageTokens`.
- Decoder prefix and decode continue to use the main context.
- This avoids the failed pattern of changing affinity after full-machine worker
  pools already exist.

Best measured launch profile for PaliGemma2 3B 448 on this machine:

```bash
./build/gemma \
  --weights build/models/paligemma2-3b-mix-448-hf/paligemma2-3b-mix-448-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 32 \
  --top_k 1 \
  --deterministic 1 \
  --force_flash_attention 1 \
  --image_skip_lps 4 \
  --image_max_lps 8 \
  --verbosity 1
```

Results:

| Case | Samples | Image ms | Prefill ms | TTFT ms | Generate ms | Gen tok/s | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 448, 32 tokens, image LPs 4-11 + FlashAttention | 2 | 4815 | 4913 | 4979 | 1814 | 17.64 | 12.10 |
| 448, 128-token cap, image LPs 4-11 + FlashAttention | 2 | 4811 | 5004 | 5071 | 7106 | 17.04 | 17.52 |

Comparison against previous 448 bests:

- 32-token default old attention: about 16.94 s.
- 32-token default FlashAttention: about 15.47 s.
- 32-token static `taskset -c 4-11` + FlashAttention: about 14.07 s.
- 32-token phase-specific image LPs 4-11 + default decoder topology +
  FlashAttention: about 12.10 s.
- 128-token cap default FlashAttention: about 22.34 s.
- 128-token cap static `taskset -c 4-11` + FlashAttention: about 20.47 s.
- 128-token cap phase-specific image LPs 4-11 + default decoder topology +
  FlashAttention: about 17.52 s.

Conclusion:

- This is the first clear end-to-end acceleration win in the codebase.
- The winning policy is phase-specific, not globally pinned:
  - image encoder: LPs 4-11
  - decoder prefix/decode: default topology
  - decoder attention: `--force_flash_attention 1`
- The remaining large target is the 4.8 s image encoder itself. This validates
  the earlier plan to consider a specialized HIP/ROCm ViT path for 448, because
  the CPU image path is still the largest single component after this change.

### 2026-05-09 HIP/rocBLAS ViT GEMM Prototype

Completed:

- Located ROCm development tools under `/opt/rocm-7.2.1`.
- `hipcc` and `rocblas-dev` were installed but `/opt/rocm/bin` was not in the
  shell `PATH`.
- Added standalone HIP/rocBLAS benchmark:
  - Source: `experimental/paligemma2_vit_hip_bench.cc`
  - Build script: `scripts/build_paligemma2_vit_hip_bench.sh`
  - Binary: `build/paligemma2_vit_hip_bench`
- The benchmark uses `rocblas_gemm_ex` for BF16 projection GEMMs and F32
  attention GEMMs, plus `rocblas_gemm_strided_batched_ex` for 16-head attention
  GEMM shapes.
- It also includes a deliberately simple fused HIP attention kernel for measuring
  the lower bound of a direct QK + softmax + AV implementation.

Build/run:

```bash
scripts/build_paligemma2_vit_hip_bench.sh
./build/paligemma2_vit_hip_bench --samples 10 --warmup 5
./build/paligemma2_vit_hip_bench --samples 5 --warmup 2 --pipeline_only
./build/paligemma2_vit_hip_bench --samples 5 --warmup 2 --filter fused_448
```

Observed HIP device:

- `AMD Radeon Graphics`
- `gfx1150`
- HIP-reported CUs: 8
- HIP global memory: 29.3 GiB

Selected 448px median results:

| Shape | CPU median ms | rocBLAS median ms | Note |
| --- | ---: | ---: | --- |
| patch embed BF16 | 1.078 | 0.265 | GPU about 4.1x faster |
| QKV BF16 | 3.958 | 1.157 | GPU about 3.4x faster |
| attention out BF16 | 1.296 | 0.495 | GPU about 2.6x faster |
| MLP up BF16 | 5.084 | 1.465 | GPU about 3.5x faster |
| MLP down BF16 | 5.668 | 1.666 | GPU about 3.4x faster |
| image head 3B BF16 | 2.548 | 0.727 | GPU about 3.5x faster |
| QK/head F32 | 0.483 | 0.271 | GPU faster for this isolated GEMM |
| AV/head F32 | 0.198 | 0.345 | CPU faster for this isolated GEMM |
| QK 16-head batched F32 | roughly 16 x 0.483 | 3.951 | GPU avoids some per-call overhead |
| AV 16-head batched F32 | roughly 16 x 0.198 | 4.734 | GPU slower than CPU-scaled AV GEMM |

448px schedule estimates:

| Schedule | Median ms | Scope |
| --- | ---: | --- |
| `vit448_projection_gemms` | 84.622 | patch embed, 27 x QKV/attn-out/MLP-up/MLP-down, image head |
| `vit448_projection_plus_qk_av` | 294.338 | projection schedule plus 27 x 16-head QK and AV GEMMs |

These exclude layernorm, GELU, softmax, residual adds, layout transforms, and
CPU/GPU transfer. Even with those omissions, the projection-only number is useful:
the current optimized CPU image encoder is about 4.8 s, so projection-heavy iGPU
offload still has large headroom if weights remain resident and only final image
tokens return to CPU.

Model-driven HIP backend probe:

- Added `experimental/paligemma2_vit_hip_backend_probe.cc`.
- Added `scripts/build_paligemma2_vit_hip_backend_probe.sh`.
- The probe loads the real `.sbs` through `Gemma`, reads the real PaliGemma2 ViT
  config and tensor metadata, estimates dense-BF16 GPU residency, uploads the
  real projection weights to the HIP device in rocBLAS-friendly `B[K,N]` layout,
  and runs both dummy-buffer and real-resident-weight projection schedules.

Commands:

```bash
scripts/build_paligemma2_vit_hip_backend_probe.sh

./build/paligemma2_vit_hip_backend_probe \
  --weights build/models/paligemma2-3b-mix-448-hf/paligemma2-3b-mix-448-sfp-from-hf.sbs \
  --hip_upload 1 \
  --hip_resident_schedule 1 \
  --hip_samples 3 \
  --hip_warmup 1
```

Probe results:

| Model | Seq | ViT dim | LLM dim | Projection weights dense BF16 | B[K,N] upload | Dummy schedule median | Real resident schedule median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PaliGemma2 3B 224 | 256 | 1152 | 2304 | 790.41 MiB | 0.546 s | 25.502 ms | 26.124 ms |
| PaliGemma2 3B 448 | 1024 | 1152 | 2304 | 790.41 MiB | 0.533 s | 79.950 ms | 88.389 ms |

Important implications:

- The real local `.sbs` path loads these ViT projection tensors as BF16 in
  `WeightsPtrs::Mode::kRead`, so the first HIP resident-weight prototype does
  not need to solve SFP decompression on GPU.
- 790 MiB for all projection weights is comfortably below HIP's 29.25 GiB
  visible shared/global pool and also below the 4 GiB firmware VRAM window.
- Using real resident weights is close to the dummy GEMM schedule for 224px and
  about 10% slower for 448px in this run. That is still viable, but the eventual
  backend should avoid unnecessary per-matrix overhead and should consider
  grouped/streamed scheduling after correctness is established.
- The next integration step can focus on a resident-weight projection backend:
  pack/copy BF16 weights once, keep CPU decoder unchanged, and initially return
  to CPU around layernorm/GELU/attention until those pieces are ported.

Backend integration step:

- Added a generic `RuntimeConfig::image_tokens_backend_func` hook to the real
  `GenerateImageTokens` flow. The default is null, so CPU-only builds and normal
  inference behavior remain unchanged.
- Moved the resident HIP projection-weight upload and rocBLAS schedule logic
  into `experimental/paligemma2_vit_hip_backend.{h,cc}`.
- Reworked `experimental/paligemma2_vit_hip_backend_probe.cc` to use that
  backend instead of carrying its own copy of the HIP logic.
- Added `scripts/build_gemma_paligemma2_vit_hip.sh`, which builds an
  experimental CLI:

```bash
scripts/build_gemma_paligemma2_vit_hip.sh

./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 2 \
  --top_k 1 \
  --deterministic 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_samples 1 \
  --paligemma_vit_hip_warmup 1 \
  --verbosity 1
```

Current `hip_probe` behavior:

- Upload projection weights once in resident `B[K,N]` layout.
- If `--paligemma_vit_hip_validate_patch 1` or `--hip_patch_validate 1` is set,
  upload a small F32 shadow copy of the patch-embedding weight and compute the
  real image patch embedding on the iGPU.
- Optionally run the resident projection schedule inside the real image-token
  generation entry point.
- Return `false` from the backend hook, so the existing CPU ViT still fills
  `image_tokens` and remains the correctness path.

Smoke results after integration:

| Command | Result |
| --- | --- |
| default `build/gemma`, 224, 2 tokens | Image 614 ms, prefill 1876 ms, generate 115 ms |
| `build/gemma_paligemma2_vit_hip --paligemma_vit_backend hip_probe`, 224, 2 tokens | Upload 0.537 s, resident projection 35.165 ms, image-token phase 1302 ms, then normal CPU output |
| refactored backend probe, 448, 1 sample | resident projection 90.781 ms |

Meaning:

- The optional HIP backend is now wired into the same control-flow boundary that
  a real image-token accelerator must use.
- This is still not a full accelerated ViT implementation: it validates resident
  projection upload and scheduling in-place, then falls back to CPU for
  correctness.
- The next implementation step is to make the backend fill one real tensor at a
  time, starting with patch embedding output, then compare against CPU before
  moving layernorm, QKV, attention, MLP, and final head.

Patch embedding validation:

- rocBLAS `gemm_ex` does not support the exact mixed type used by the current
  CPU path, namely F32 activations multiplied by BF16 weights. The supported
  BF16 fast path requires both A and B to be BF16.
- To validate the first real tensor without changing semantics, the HIP backend
  now uploads an F32 shadow copy of `img_emb_kernel` in `B[K,N]` layout.
- The GPU computes:
  - image patch extraction on CPU into F32 patches;
  - F32 rocBLAS patch GEMM;
  - BF16 activation + BF16 resident-weight rocBLAS patch GEMM for the fast-path
    precision experiment;
  - HIP kernel adding `img_emb_bias` and `img_pos_emb`;
  - device-to-host copy for comparison against a CPU scalar reference.

Patch validation results:

| Model | Patch path | Time after warmup | Max abs error vs F32 CPU ref | RMS error |
| --- | --- | ---: | ---: | ---: |
| PaliGemma2 3B 224 | F32 shadow weight | 0.905 ms | 0.000007 | ~0 |
| PaliGemma2 3B 224 | BF16 patches + BF16 weight | 0.601 ms | 0.008914 | 0.001140 |
| PaliGemma2 3B 448 | F32 shadow weight | 3.402 ms | 0.000006 | ~0 |
| PaliGemma2 3B 448 | BF16 patches + BF16 weight | 1.972 ms | 0.011225 | 0.001159 |

Real-entry smoke with validation:

```bash
./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 2 \
  --top_k 1 \
  --deterministic 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_validate_patch 1 \
  --paligemma_vit_hip_samples 3 \
  --paligemma_vit_hip_warmup 2 \
  --verbosity 1
```

Observed:

- F32 shadow `img_emb_kernel` upload: 2.58 MiB.
- Total resident upload with F32 shadow: 792.99 MiB in 0.518 s.
- Patch embedding validation in real entry:
  - F32 shadow: 0.911 ms, max abs error 0.000007.
  - BF16 fast path: 0.679 ms, max abs error 0.008914, RMS 0.001140.
- Resident projection schedule in real entry: median 29.985 ms for 224.
- CPU fallback still produced the expected beginning: `A long`.

Implication:

- The first real tensor boundary is now correct.
- The exact-F32 path is a correctness bridge, not necessarily the final fastest
  path.
- BF16 A/B is materially faster for patch embedding, but it introduces about
  0.001 RMS error at the first image tensor. That looks small, but it is not yet
  a model-level correctness claim.
- Next step: carry both F32-shadow and BF16 activation variants through QKV or
  MLP projection validation, then run an end-to-end output comparison before
  allowing BF16 activations to replace CPU semantics.

Layer0 QKV validation:

- Added `--hip_qkv_validate 1` to the standalone probe and
  `--paligemma_vit_hip_validate_qkv 1` to the experimental CLI.
- The validation path computes:
  - patch embedding on host via the scalar reference path;
  - layer0 LayerNorm on host with the same formula as `ops::LayerNorm`;
  - layer0 QKV on HIP via both F32-shadow and BF16 activation routes;
  - a row-0 scalar spot check for the F32-shadow QKV layout;
  - full-output BF16-vs-F32 error.

QKV validation results:

| Model | QKV path | Time after warmup | Error |
| --- | --- | ---: | --- |
| PaliGemma2 3B 224 | F32 shadow | 4.519 ms | row0 scalar spot max abs 0.000014 |
| PaliGemma2 3B 224 | BF16 A/B | 2.814 ms | max abs vs F32 0.010305, RMS 0.000538 |
| PaliGemma2 3B 448 | F32 shadow | 14.508 ms | row0 scalar spot max abs 0.000011 |
| PaliGemma2 3B 448 | BF16 A/B | 7.293 ms | max abs vs F32 0.011736, RMS 0.000571 |

Real-entry smoke with patch + QKV validation:

```bash
./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 2 \
  --top_k 1 \
  --deterministic 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_validate_patch 1 \
  --paligemma_vit_hip_validate_qkv 1 \
  --paligemma_vit_hip_samples 3 \
  --paligemma_vit_hip_warmup 2 \
  --verbosity 1
```

Observed:

- Patch BF16 fast path: 0.623 ms, RMS 0.001140.
- Layer0 QKV BF16 fast path: 3.203 ms, RMS 0.000538.
- Resident projection schedule in real entry: median 27.223 ms for 224.
- CPU fallback still produced `A long`.

Implication:

- BF16 activation rounding is still small after the first LayerNorm + QKV
  projection boundary.
- BF16 is now consistently faster than the exact F32-shadow route for both
  patch embedding and QKV.
- This strengthens the case for a BF16-resident ViT projection backend, but it
  still does not prove end-to-end model-output equivalence. The next validation
  boundary should be MLP up/down or a whole first ViT block.

Layer0 MLP up/down validation:

- Added `--hip_mlp_validate 1` to the standalone probe and
  `--paligemma_vit_hip_validate_mlp 1` to the experimental CLI.
- The validation path computes:
  - patch embedding on host via the scalar reference path;
  - `layer_norm_1` on host over the patch-embedding stream;
  - layer0 MLP on HIP as `linear_0 -> GELU -> linear_1`;
  - a row-0 scalar spot check for the F32-shadow MLP layout;
  - full-output BF16-vs-F32 error.
- This is a projection precision boundary, not a full first-block validation:
  the true model applies attention plus residual before `layer_norm_1`. Full
  block validation should wait until attention/residual movement is resident or
  cheap enough to keep in the validation loop.
- The BF16 path intentionally quantizes after `linear_0 + bias`, applies GELU
  from BF16, and feeds BF16 into `linear_1`, matching the current CPU ViT
  intermediate tensor type more closely than a pure-F32 hidden route.

MLP validation results:

| Model | MLP path | Time after warmup | Error |
| --- | --- | ---: | --- |
| PaliGemma2 3B 224 | F32 shadow | 15.383 ms | row0 scalar spot max abs 0.000031 |
| PaliGemma2 3B 224 | BF16 A/B + BF16 GELU input | 8.524 ms | max abs vs F32 0.017377, RMS 0.001012 |
| PaliGemma2 3B 448 | F32 shadow | 38.247 ms | row0 scalar spot max abs 0.000023 |
| PaliGemma2 3B 448 | BF16 A/B + BF16 GELU input | 24.495 ms | max abs vs F32 0.027471, RMS 0.000959 |

Real-entry smoke with patch + QKV + MLP validation:

```bash
./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 2 \
  --top_k 1 \
  --deterministic 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_validate_patch 1 \
  --paligemma_vit_hip_validate_qkv 1 \
  --paligemma_vit_hip_validate_mlp 1 \
  --paligemma_vit_hip_samples 2 \
  --paligemma_vit_hip_warmup 1 \
  --verbosity 1
```

Observed:

- Patch BF16 fast path: 0.867 ms, RMS 0.001140.
- Layer0 QKV BF16 fast path: 4.713 ms, RMS 0.000538.
- Layer0 MLP BF16 fast path: 9.037 ms, RMS 0.001012.
- Resident projection schedule in real entry: median 39.235 ms for 224.
- CPU fallback still produced `A long`.

Implication:

- Patch, QKV, and MLP projection boundaries now all validate with small BF16
  error against F32-shadow references.
- MLP is a meaningful chunk of layer work: at 224 it costs about 9 ms in this
  standalone BF16 route, and at 448 about 24.5 ms.
- The next correctness boundary should be attention output plus residual, then a
  true whole-layer comparison.
- The next performance boundary should avoid re-materializing host references
  and should keep patch/QKV/MLP tensors resident across the image stage.

Layer0 attention output + residual validation:

- Added `--hip_attn_out_validate 1` to the standalone probe and
  `--paligemma_vit_hip_validate_attn_out 1` to the experimental CLI.
- The validation path computes:
  - patch embedding and `layer_norm_0` on host;
  - layer0 QKV through the already-validated F32-shadow HIP route;
  - attention softmax/weighted-sum on host as a deterministic reference input;
  - layer0 `attn_out_w` on HIP through F32-shadow and BF16 routes;
  - attention residual add on HIP;
  - a row-0 scalar spot check for F32-shadow layout/residual placement;
  - full-output BF16-vs-F32 residual error.
- This isolates the projection/residual boundary. It does not claim that
  attention itself is accelerated yet; the host softmax reference is deliberately
  outside the timed HIP projection path.

Attention output validation results:

| Model | Host softmax reference | Attention-output path | Time after warmup | Error |
| --- | ---: | --- | ---: | --- |
| PaliGemma2 3B 224 | 0.054 s | F32 shadow + residual | 1.140 ms | row0 scalar spot max abs 0.000007 |
| PaliGemma2 3B 224 | 0.054 s | BF16 A/B + residual | 0.904 ms | max abs vs F32 0.005169, RMS 0.000124 |
| PaliGemma2 3B 448 | 1.398 s | F32 shadow + residual | 4.068 ms | row0 scalar spot max abs 0.000006 |
| PaliGemma2 3B 448 | 1.398 s | BF16 A/B + residual | 3.362 ms | max abs vs F32 0.006011, RMS 0.000111 |

Real-entry smoke with attention-output validation:

```bash
./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 2 \
  --top_k 1 \
  --deterministic 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_validate_attn_out 1 \
  --paligemma_vit_hip_samples 2 \
  --paligemma_vit_hip_warmup 1 \
  --verbosity 1
```

Observed:

- Host attention softmax/weighted-sum reference: 0.052 s.
- Layer0 attention-output BF16 path: 0.899 ms, RMS 0.000124.
- Resident projection schedule in real entry: median 27.106 ms for 224.
- CPU fallback still produced `A long`.

Implication:

- The attention-output projection plus residual is now validated and has the
  smallest BF16-vs-F32 RMS among the tested layer0 projection boundaries.
- The projection itself is cheap relative to QKV and MLP. The expensive missing
  piece is the attention softmax/weighted-sum, especially at 448 where the host
  scalar reference is already about 1.4 s for one layer.
- Next correctness step: wire this residual into `layer_norm_1` and MLP, then
  compare the whole layer0 output.
- Next performance step: replace the host attention reference with a viable
  tiled HIP attention kernel or a rocBLAS-assisted QK/AV plus fused softmax
  route.

Layer0 whole-block validation:

- Added `--hip_layer0_block_validate 1` to the standalone probe and
  `--paligemma_vit_hip_validate_layer0_block 1` to the experimental CLI.
- The validation path now chains the full layer0 graph:
  - patch embedding reference as `x0`;
  - `layer_norm_0`;
  - QKV projection on HIP;
  - attention softmax/weighted-sum on host reference;
  - `attn_out_w + attention residual` on HIP;
  - `layer_norm_1`;
  - MLP `linear_0 -> GELU -> linear_1` on HIP;
  - final MLP residual on HIP;
  - full final layer0 output BF16-vs-F32 comparison.
- This is the first whole-layer numerical boundary. It is still not a complete
  GPU layer because attention softmax/weighted-sum is a host reference.

Whole layer0 validation results:

| Model | Route | Host attention reference | Device projection sum | Final output error |
| --- | --- | ---: | ---: | --- |
| PaliGemma2 3B 224 | F32 shadow | 0.052 s | 19.819 ms | reference |
| PaliGemma2 3B 224 | BF16 route | 0.052 s | 6.949 ms | max abs 0.031874, RMS 0.000806 |
| PaliGemma2 3B 448 | F32 shadow | 1.350 s | 39.548 ms | reference |
| PaliGemma2 3B 448 | BF16 route | 1.295 s | 26.342 ms | max abs 0.024858, RMS 0.000597 |

Real-entry smoke with whole layer0 validation:

```bash
./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 2 \
  --top_k 1 \
  --deterministic 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_validate_layer0_block 1 \
  --paligemma_vit_hip_samples 2 \
  --paligemma_vit_hip_warmup 1 \
  --verbosity 1
```

Observed:

- Host attention references: F32 0.055 s, BF16 0.056 s.
- Layer0 BF16 device projection sum: 13.267 ms for the real-entry 224 smoke.
- Final layer0 BF16-vs-F32 error: max abs 0.031874, RMS 0.000806.
- Resident projection schedule in real entry: median 25.490 ms for 224.
- CPU fallback still produced `A long`.

Implication:

- The BF16 resident projection route remains numerically stable after one full
  ViT layer boundary.
- The whole-layer BF16 final RMS is lower than the standalone patch/MLP RMS
  numbers, so the first residual+normalization chain is not amplifying error in
  an obvious way.
- Device projection time for one 448 layer is already plausible at about
  26.3 ms, but the host attention reference is about 1.3 s per layer and is the
  dominant blocker.
- Next step: replace the host attention reference with an actual device
  attention path. The prior naive fused kernel was too slow, so the better path
  is likely rocBLAS QK/AV plus a custom tiled softmax, or a more serious fused
  HIP attention kernel with tiled Q/K/V reuse.

Layer0 device attention core validation:

- Added `--hip_device_attention_validate 1` to the standalone probe and
  `--paligemma_vit_hip_validate_device_attention 1` to the experimental CLI.
- The validation path computes QKV with the existing HIP projection routes, then
  replaces host attention with:
  - `SplitQKVForAttentionKernel`: splits Q/K/V into head-major buffers;
  - rocBLAS strided-batched QK;
  - `SoftmaxRowsKernel`: one row-wise softmax block per head/token;
  - rocBLAS strided-batched AV;
  - `PackAttentionHeadsKernel`: packs output back to `[token, head, dim]`.
- Important correctness detail: query scaling is applied as the rocBLAS QK
  alpha after dot accumulation. An earlier local version scaled Q during split;
  that was close for F32 but caused BF16-QKV softmax outliers because the CPU
  reference multiplies scores after the dot.

Device attention validation results:

| Model | Route | Host attention reference | HIP attention core | Error vs same-QKV host reference |
| --- | --- | ---: | ---: | --- |
| PaliGemma2 3B 224 | F32 QKV | 0.061 s | 1.341 ms | max abs 0.000001, RMS ~0 |
| PaliGemma2 3B 224 | BF16 QKV route | 0.056 s | 1.284 ms | max abs 0.000002, RMS ~0 |
| PaliGemma2 3B 448 | F32 QKV | 1.405 s | 14.580 ms | max abs 0.000002, RMS ~0 |
| PaliGemma2 3B 448 | BF16 QKV route | 1.463 s | 10.088 ms | max abs 0.000003, RMS ~0 |

Real-entry smoke with device attention validation:

```bash
./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 2 \
  --top_k 1 \
  --deterministic 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_validate_device_attention 1 \
  --paligemma_vit_hip_samples 2 \
  --paligemma_vit_hip_warmup 1 \
  --verbosity 1
```

Observed:

- Host attention references: F32 0.053 s, BF16 0.055 s.
- Device attention core: F32 1.811 ms, BF16 2.015 ms in the real-entry 224
  smoke.
- Device attention error: max abs 0.000001-0.000002, RMS ~0.
- CPU fallback still produced `A long`.

Implication:

- The previous largest correctness/performance gap now has a working device
  prototype. For 448, the replacement is roughly two orders of magnitude faster
  than the scalar host reference for one layer's attention core.
- This is not yet final inference acceleration because QKV, attention,
  attention-output projection, layernorm, and MLP are still validation stages
  with host round-trips between some boundaries.
- Next step: wire `TimeDeviceAttention`'s kernels into the whole-layer
  validation path so layer0 no longer uses host attention, then remove the host
  round-trip before `attn_out_w`.

Layer0 whole-block validation with device attention:

- Added `--hip_layer0_block_device_attention_validate 1` to the standalone
  probe and `--paligemma_vit_hip_validate_layer0_block_device_attention 1` to
  the experimental CLI.
- This path chains the full layer0 graph with device attention:
  - QKV projection on HIP;
  - device QK/softmax/AV using the validated attention core;
  - direct device handoff into `attn_out_w`;
  - attention residual on HIP;
  - host `layer_norm_1` reference;
  - MLP on HIP;
  - final MLP residual on HIP.
- Compared with the previous whole-layer validation, this removes the host
  attention weighted-sum and the host round-trip before `attn_out_w`. The
  remaining major host boundary is `layer_norm_1`.

Whole layer0 device-attention validation results:

| Model | Route | QKV | Attention | Attn out | MLP | Device sum | Final output error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| PaliGemma2 3B 224 | F32 shadow | 6.650 ms | 1.704 ms | 2.650 ms | 15.978 ms | 26.981 ms | reference |
| PaliGemma2 3B 224 | BF16 route | 4.175 ms | 2.251 ms | 1.255 ms | 10.947 ms | 18.627 ms | max abs 0.031878, RMS 0.000806 |
| PaliGemma2 3B 448 | F32 shadow | 19.760 ms | 19.217 ms | 5.657 ms | 51.450 ms | 96.085 ms | reference |
| PaliGemma2 3B 448 | BF16 route | 13.547 ms | 19.453 ms | 4.310 ms | 37.979 ms | 75.288 ms | max abs 0.024858, RMS 0.000597 |

Real-entry smoke with whole layer0 device-attention validation:

```bash
./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 2 \
  --top_k 1 \
  --deterministic 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_validate_layer0_block_device_attention 1 \
  --paligemma_vit_hip_samples 2 \
  --paligemma_vit_hip_warmup 1 \
  --verbosity 1
```

Observed:

- Layer0 BF16 device-attention block device sum: 16.376 ms in the real-entry
  224 smoke.
- Final layer0 BF16-vs-F32 error: max abs 0.031878, RMS 0.000806.
- CPU fallback still produced `A long`.

Implication:

- The device-attention whole-layer route preserves the same final error as the
  host-attention whole-layer route, so the new device attention handoff is not
  adding measurable layer0 error.
- For 448, one BF16 layer0 validation stage is now about 75 ms of device work
  instead of paying about 1.3 s of host attention reference plus projection
  work.
- This is the first practical shape of a resident GPU ViT block. The next
  acceleration boundary is moving `layer_norm_1` to HIP so the attention
  residual can feed MLP without a host copy.

Layer0 whole-block validation with device input preparation:

- Added `--hip_layer0_block_device_norm_validate 1` to the standalone probe and
  `--paligemma_vit_hip_validate_layer0_block_device_norm 1` to the experimental
  CLI.
- The older `device_attention` validation flag is kept as a compatibility alias
  for the same implementation; use the `device_norm` flag for new runs because
  it describes the current boundary more accurately.
- Added a HIP row-wise layernorm kernel and now use it for both
  `layer_norm_0` and `layer_norm_1` in this validation path.
- The BF16 route now keeps this flow on device:
  - patch embedding GEMM plus image embedding bias and position embedding;
  - device `layer_norm_0`;
  - QKV projection;
  - device QK/softmax/AV;
  - `attn_out_w` plus attention residual;
  - device `layer_norm_1`;
  - BF16 quantization of the normalized activation;
  - MLP up/GELU/down plus final residual.
- Host patch embedding and layernorm results are now only validation oracles.
  They no longer provide the tensors that feed QKV or MLP in this path.

Whole layer0 device-norm validation results:

| Model | Route | Patch | LN0 | QKV | Attention | Attn out | LN1 | MLP | Device sum | Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PaliGemma2 3B 224 | F32 shadow | 0.916 ms | 0.106 ms | 4.492 ms | 1.782 ms | 1.232 ms | 0.053 ms | 10.437 ms | 19.017 ms | patch max abs 0.000007, LN0/LN1 RMS ~0 |
| PaliGemma2 3B 224 | BF16 route | 0.605 ms | 0.143 ms | 2.823 ms | 1.162 ms | 0.734 ms | 0.073 ms | 6.830 ms | 12.370 ms | final max abs 0.029657, RMS 0.001654 |
| PaliGemma2 3B 448 | F32 shadow | 4.267 ms | 0.166 ms | 17.992 ms | 16.016 ms | 5.936 ms | 0.157 ms | 44.902 ms | 89.436 ms | patch max abs 0.000006, LN0/LN1 RMS ~0 |
| PaliGemma2 3B 448 | BF16 route | 2.187 ms | 0.279 ms | 11.889 ms | 15.821 ms | 4.882 ms | 0.268 ms | 32.432 ms | 67.757 ms | final max abs 0.024877, RMS 0.001557 |

Real-entry smoke with device attention plus device `layer_norm_1`:

```bash
./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 2 \
  --top_k 1 \
  --deterministic 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_validate_layer0_block_device_norm 1 \
  --paligemma_vit_hip_samples 2 \
  --paligemma_vit_hip_warmup 1 \
  --verbosity 1
```

Observed:

- Layer0 BF16 device-input block device sum: 14.706 ms in the real-entry 224
  smoke.
- BF16 patch embedding error vs F32 reference: max abs 0.008914, RMS 0.001140.
- Device LN0/LN1 error vs host references: max abs about 0.000003, RMS ~0.
- Final layer0 BF16-vs-F32 error: max abs 0.029657, RMS 0.001654.
- CPU fallback still produced `A long`.

Implication:

- The layer0 validation path no longer requires host-generated `x0`, `pre_att`,
  attention weighted sums, or `pre_ffw` to feed the device graph. Those host
  tensors are now only correctness references.
- Device LN1 is not a bottleneck on this iGPU: about 0.07 ms for 224 and
  0.30 ms for 448 on the BF16 route, including the post-LN BF16 quantization.
- The new main precision cost is BF16 patch embedding. Its patch-level RMS is
  about 0.00115 and it raises one-layer final RMS to about 0.0016.
- The remaining layer0 host activity is validation-only copying. The next
  backend milestone is to extract this layer body into a reusable device-layer
  runner and validate a 2-layer prefix before scaling to the full ViT stack.

Two-layer prefix validation with reusable device-layer runner:

- Added `RunDeviceVitLayer`, a reusable HIP implementation of one ViT block:
  `x -> LN0 -> QKV -> device attention -> attn_out + residual -> LN1 -> MLP +
  residual`.
- Added `--hip_layer_prefix2_device_norm_validate 1` to the standalone probe
  and `--paligemma_vit_hip_validate_layer_prefix2_device_norm 1` to the
  experimental CLI.
- The prefix path uses ping-pong device buffers:
  - patch embedding writes the initial `x`;
  - layer0 reads buffer A and writes buffer B;
  - layer1 reads buffer B and writes buffer A;
  - final readback compares BF16 route vs F32-shadow route.
- Host activity in this path is limited to patch extraction, scalar reference
  construction for patch error, and final validation readback. It does not feed
  intermediate tensors back into the device graph.

Two-layer prefix results:

| Model | Route | Patch | Layer0 | Layer1 | Device sum | Final error |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PaliGemma2 3B 224 | F32 shadow | 0.864 ms | 18.230 ms | 13.387 ms | 32.482 ms | reference |
| PaliGemma2 3B 224 | BF16 route | 0.592 ms | 12.271 ms | 7.334 ms | 20.197 ms | max abs 0.048050, RMS 0.002084 |
| PaliGemma2 3B 448 | F32 shadow | 2.454 ms | 57.494 ms | 50.714 ms | 110.663 ms | reference |
| PaliGemma2 3B 448 | BF16 route | 2.114 ms | 37.875 ms | 39.982 ms | 79.970 ms | max abs 0.057442, RMS 0.002000 |

Real-entry smoke with 2-layer prefix validation:

```bash
./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 2 \
  --top_k 1 \
  --deterministic 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_validate_layer_prefix2_device_norm 1 \
  --paligemma_vit_hip_samples 1 \
  --paligemma_vit_hip_warmup 1 \
  --verbosity 1
```

Observed:

- 224 BF16 two-layer prefix device sum: 21.742 ms in the real-entry smoke.
- Final two-layer BF16-vs-F32 error: max abs 0.048050, RMS 0.002084.
- CPU fallback still produced `A long`.

Implication:

- The reusable layer runner composes across at least two consecutive ViT layers
  without a host intermediate tensor.
- BF16 error grows from about 0.0016 RMS after one layer to about 0.0021 RMS
  after two layers on 224, and about 0.0020 RMS on 448. That is a small and
  expected increase rather than an immediate instability.
- The next boundary is extending the same runner across all ViT layers and then
  adding final encoder norm / image head handling, while keeping the final
  decoder-visible image tokens as the only required copy back to CPU.

Full transformer-stack validation before final norm/head:

- Added `--hip_layer_stack_device_norm_validate 1` to the standalone probe and
  `--paligemma_vit_hip_validate_layer_stack_device_norm 1` to the experimental
  CLI.
- The path starts from HIP patch embedding, then runs all 27 ViT transformer
  layers through `RunDeviceVitLayer` with ping-pong device buffers. It does not
  copy intermediate layer tensors to host.
- This boundary intentionally stops before final encoder norm, pooling, and
  image projection head. The current CLI still returns false from the HIP hook,
  so CPU ViT remains the decoder-visible correctness path.
- Validation still uploads per-layer F32 shadow weights inside
  `RunDeviceVitLayer`; that upload is not counted in the printed device timing
  and is a validation artifact, not the intended steady-state backend design.

Probe command:

```bash
./build/paligemma2_vit_hip_backend_probe \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --hip_schedule 0 \
  --hip_resident_schedule 0 \
  --hip_upload 1 \
  --hip_layer_stack_device_norm_validate 1 \
  --hip_samples 1 \
  --hip_warmup 1 \
  --hip_iters 1 \
  --hip_verbose 1
```

Full-stack probe results:

| Model | Route | Patch | LN0 | QKV | Attention | Attn out | LN1 | MLP | Device sum | Final error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PaliGemma2 3B 224 | F32 shadow | 1.123 ms | 1.953 ms | 124.995 ms | 49.458 ms | 43.319 ms | 1.690 ms | 359.426 ms | 581.964 ms | reference |
| PaliGemma2 3B 224 | BF16 route | 0.673 ms | 2.724 ms | 80.901 ms | 41.460 ms | 28.218 ms | 2.790 ms | 217.793 ms | 374.560 ms | max abs 4.463629, RMS 0.080579 |
| PaliGemma2 3B 448 | F32 shadow | 3.847 ms | 7.153 ms | 603.701 ms | 434.299 ms | 122.640 ms | 4.493 ms | 1019.388 ms | 2195.522 ms | reference |
| PaliGemma2 3B 448 | BF16 route | 3.464 ms | 11.402 ms | 317.702 ms | 396.476 ms | 93.838 ms | 7.764 ms | 727.340 ms | 1557.985 ms | max abs 7.433367, RMS 0.087285 |

Real-entry smoke with full-stack validation:

```bash
./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 2 \
  --top_k 1 \
  --deterministic 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_validate_layer_stack_device_norm 1 \
  --paligemma_vit_hip_samples 1 \
  --paligemma_vit_hip_warmup 1 \
  --paligemma_vit_hip_iters 1 \
  --verbosity 1
```

Observed:

- CLI full-stack 224 BF16 device sum: 363.897 ms.
- CLI final full-stack BF16-vs-F32 error: max abs 5.249403, RMS 0.082663.
- CPU fallback still produced `A long`.

Implication:

- The reusable device-layer runner now composes across the complete 27-layer ViT
  transformer stack without host intermediate activations.
- BF16 is materially faster than the F32-shadow validation path: about 1.55x on
  224 and 1.41x on 448 for the measured device work.
- Full-stack BF16 error is no longer in the small two-layer range. RMS grows from
  about 0.002 after two layers to about 0.08-0.09 after 27 layers. This does not
  prove the generated answer will fail, because the final encoder norm/image head
  and decoder tolerance still need testing, but it does mean the precision policy
  has become a first-order integration question.
- The next boundary should add final encoder norm, pooling, and image projection
  head, then compare produced image tokens against CPU output. If that token
  error is too high, likely mitigations are keeping selected GEMMs or residual
  boundaries in F32, adding better accumulation/quantization boundaries, or
  validating mixed precision per sub-block rather than forcing BF16 everywhere.

Image-token validation with final norm and img_head:

- Added `--hip_image_tokens_device_norm_validate 1` to the standalone probe and
  `--paligemma_vit_hip_validate_image_tokens_device_norm 1` to the experimental
  CLI.
- Added `ValidateImageTokensDeviceNorm`, which extends the full-stack HIP path
  through final encoder LayerNorm and `img_head_kernel + img_head_bias`.
- The standalone probe now computes a CPU `Gemma::GenerateImageTokens` reference
  and compares HIP-produced image tokens against that decoder-visible output.
- Scope: this supports PaliGemma2's unpooled path (`pool_dim == 1`). GEMMA_VLM's
  4x4 pooling and soft-embedding RMSNorm are still a separate boundary.
- The CLI validation path fills image-token storage with the HIP BF16 candidate
  for measurement, prints BF16-vs-F32-shadow error, then still returns false so
  the mature CPU image-token path remains the actual inference source.

Probe command:

```bash
./build/paligemma2_vit_hip_backend_probe \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --hip_schedule 0 \
  --hip_resident_schedule 0 \
  --hip_upload 1 \
  --hip_image_tokens_device_norm_validate 1 \
  --hip_samples 1 \
  --hip_warmup 1 \
  --hip_iters 1 \
  --hip_verbose 1
```

Image-token probe results:

| Model | Route | Stack | Final norm | Head | Device sum | Error vs CPU image tokens |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PaliGemma2 3B 224 | F32 shadow | 278.426 ms | 0.040 ms | 1.296 ms | 279.762 ms | max abs 0.449724, RMS 0.009448 |
| PaliGemma2 3B 224 | BF16 route | 257.838 ms | 0.050 ms | 1.395 ms | 259.282 ms | max abs 0.390941, RMS 0.007700 |
| PaliGemma2 3B 448 | F32 shadow | 1333.934 ms | 0.332 ms | 5.794 ms | 1340.060 ms | max abs 1.137996, RMS 0.011375 |
| PaliGemma2 3B 448 | BF16 route | 993.672 ms | 0.157 ms | 4.911 ms | 998.740 ms | max abs 0.762721, RMS 0.007182 |

Real-entry smoke with image-token validation:

```bash
./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 2 \
  --top_k 1 \
  --deterministic 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_validate_image_tokens_device_norm 1 \
  --paligemma_vit_hip_samples 1 \
  --paligemma_vit_hip_warmup 1 \
  --paligemma_vit_hip_iters 1 \
  --verbosity 1
```

Observed:

- CLI 224 BF16 image-token device sum: 241.225 ms.
- CLI BF16-vs-F32 image-token error: max abs 0.582785, RMS 0.007957.
- CPU fallback still produced `A long`.

Implication:

- The earlier full-stack hidden-state RMS of about 0.08-0.09 is not the right
  final risk metric. After final encoder norm and image head, decoder-visible
  image-token RMS vs CPU is about 0.007-0.011 in these probes.
- The final norm and head are small compared with the transformer stack. The
  optimization target remains the 27-layer body, especially MLP/QKV/attention.
- This is now close to the first real replacement boundary: the backend can fill
  image-token storage. Before returning true by default, the next step should run
  generation A/B with HIP image tokens enabled, compare output stability on more
  prompts/images, and add an explicit opt-in flag for returning HIP tokens.

Explicit HIP image-token return A/B:

- Added `--paligemma_vit_hip_return_image_tokens 1` to the experimental CLI.
- This flag makes the real `GenerateImageTokens` hook return true after filling
  `image_tokens` with the HIP BF16 candidate route. The decoder then consumes
  HIP-produced image tokens instead of CPU-produced image tokens.
- The return path now uses `GenerateImageTokensDeviceNorm`, a BF16-only producer.
  It no longer runs the F32-shadow route from `ValidateImageTokensDeviceNorm`.
- The validation-only `--paligemma_vit_hip_validate_image_tokens_device_norm 1`
  still returns false and falls back to CPU image tokens.

Return-path command:

```bash
./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --max_generated_tokens 16 \
  --top_k 1 \
  --deterministic 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_return_image_tokens 1 \
  --paligemma_vit_hip_samples 1 \
  --paligemma_vit_hip_warmup 1 \
  --paligemma_vit_hip_iters 1 \
  --verbosity 0
```

A/B generation results:

| Model | CPU image tokens | HIP-return image tokens | Result |
| --- | --- | --- | --- |
| PaliGemma2 3B 224 | `A long-shot view of a large white church. The church has two towers` | `A long-shot view of a large white church. The church has two towers` | identical for 16 deterministic tokens |
| PaliGemma2 3B 448 | `A large building with two towers stands tall in the distance. The building has a` | `A large building with two towers stands tall in the distance. The building has a` | identical for 16 deterministic tokens |

Verbose return-path smoke:

- The production return path no longer inserts per-stage HIP event timing.
- Model-level patch/pos/final/head scalars and per-layer LayerNorm/bias vectors
  are now uploaded once as resident F32 vectors.
- BF16 layer scratch is now a backend-persistent workspace. It is allocated on
  the first return call for a given shape and reused across later image-token
  calls in the same backend object.
- 224 cold CLI run:
  - `image_tokens_return_workspace rows=256 vit_dim=1152 scratch=24.37 MiB lifetime=backend`
  - `image_tokens_return_bf16 layers=27 timing=disabled`
  - `[ Timing info ] Image token generation took: 1397 ms`
  - `Resident upload total: 793.07 MiB in 1.000 s`
- 448 cold CLI run:
  - `image_tokens_return_workspace rows=1024 vit_dim=1152 scratch=145.47 MiB lifetime=backend`
  - `image_tokens_return_bf16 layers=27 timing=disabled`
  - `[ Timing info ] Image token generation took: 2909 ms`
  - `Resident upload total: 796.44 MiB in 0.993 s`
- Warm same-process return bench via `paligemma2_vit_hip_backend_probe
  --hip_image_tokens_return_bench 3 --hip_warmup 1 --hip_schedule 0
  --hip_upload 1 --hip_verbose 0`:
  - 224: best 350.240 ms, median 404.875 ms.
  - 448: best 1825.278 ms, median 1834.128 ms.
- Warm return stage profile via `--hip_image_tokens_return_profile 1
  --hip_samples 1 --hip_warmup 1 --hip_iters 1`:
  - 224 host: patch_prepare 0.188 ms, alloc_once 1.900 ms, patch_h2d
    0.066 ms, d2h 0.112 ms, host_copy 0.145 ms, host_sum 2.410 ms.
  - 224 device: patch_embed 0.866 ms, LN0 3.496 ms, QKV 78.185 ms,
    attention 47.129 ms, attn_out 25.982 ms, LN1 2.649 ms, MLP
    219.563 ms, final_norm 0.106 ms, head 1.997 ms, device_sum
    379.973 ms.
  - 448 host: patch_prepare 2.216 ms, alloc_once 6.667 ms, patch_h2d
    0.116 ms, d2h 0.890 ms, host_copy 2.938 ms, host_sum 12.828 ms.
  - 448 device: patch_embed 2.537 ms, LN0 9.919 ms, QKV 339.423 ms,
    attention 474.434 ms, attn_out 110.385 ms, LN1 11.623 ms, MLP
    877.827 ms, final_norm 0.173 ms, head 8.099 ms, device_sum
    1834.418 ms.
- Resident upload now includes BF16 projection weights plus the F32 scalar
  vectors needed by the no-timing return path. 448 has a larger positional
  embedding, so its resident total is slightly larger than 224.
- The output still began with `A long` / `A large`, and the CPU image-token
  fallback was skipped because the HIP hook returned true.

Implication:

- HIP BF16 image tokens are now sufficient for a first decoder-consumed A/B test
  on both 224 and 448 PaliGemma2 sample runs.
- The return path is now separated from the diagnostic F32-shadow validation path.
  The production route now emits only a `timing=disabled` marker; the CLI wall
  time above is a cold end-to-end image-token measurement and includes the first
  resident weight upload.
- The old BF16 device-sum numbers were useful for validating kernel work but are
  no longer the right production latency proxy because they inserted event
  synchronization between stages.
- Per-layer temporary allocation churn, per-image scalar uploads, and per-image
  layer-workspace allocation have been removed from the return loop.
- The warm stage profile changes the priority order. Host patch construction,
  patch upload, final D2H, final norm, and img_head are not first-order
  bottlenecks anymore. The remaining time is overwhelmingly inside the 27-layer
  transformer body.
- 224 optimization priority: MLP first, QKV second, attention third. 448
  priority: MLP first, attention second, QKV third. The attention boundary grows
  sharply at 1024 image tokens because QK/softmax/AV carries sequence-length
  pressure.
- First MLP optimization implemented: the production BF16 return path now fuses
  `linear_0 + bias -> BF16 -> GELU -> BF16` into one
  `AddBiasGeluToBF16Kernel`. The fused kernel deliberately preserves the old
  rounding boundary before GELU, so the second MLP GEMM sees the same BF16
  precision as before.
- Post-fusion warm return bench via `--hip_image_tokens_return_bench 5`:
  - 224: best 348.560 ms, median 383.928 ms.
  - 448: best 1685.199 ms, median 1744.120 ms.
- Post-fusion warm return stage profile:
  - 224 device: MLP 187.634 ms, stack 324.417 ms, device_sum 327.782 ms.
  - 448 device: MLP 821.333 ms, stack 1731.630 ms, device_sum 1743.595 ms.
- Interpretation: fusing the elementwise MLP activation boundary is worthwhile
  but not enough by itself. It removes one launch and one intermediate BF16
  global read/write per layer, but the two MLP GEMMs still dominate.
- The MLP return profile now reports substage timing for the two GEMMs, fused
  activation, and residual add:
  - 224 no-solution split: up GEMM 75.786 ms, activation 4.489 ms, down GEMM
    79.827 ms, residual 3.001 ms, total MLP 163.104 ms.
  - 448 no-solution split: up GEMM 407.208 ms, activation 25.294 ms, down
    GEMM 411.785 ms, residual 15.592 ms, total MLP 859.878 ms.
- Added a probe-only rocBLAS solution-index bench for the layer0 MLP up/down
  shapes. Standalone samples show alternate solutions can beat the default for
  an individual GEMM:
  - 224 standalone: up solution -68 measured 2.245 ms vs default 2.463 ms;
    down solution -69 measured 2.326 ms vs default 2.761 ms.
  - 448 standalone: up solution -73 measured 12.695 ms vs default 13.568 ms;
    down solution -69 measured 10.031 ms vs default 12.953 ms.
- Whole-path results were not stable enough to promote solution-index tuning to
  the production return path. In one 224 combined profile, selected solutions
  `up=-73 down=-69` worsened total MLP to 179.156 ms. In one 448 combined
  profile, selected `up=0 down=-69` improved the down GEMM locally but left
  total MLP around 861.093 ms. A 448 combined warm bench selected
  `up=-68 down=-71` and measured best 1713.938 ms / median 1803.728 ms, which
  was not a stable win over the default path.
- Conclusion: solution-index selection is useful as a measurement probe but is
  too noisy as a hard-coded production default. It should stay behind
  `--hip_mlp_solution_bench` until there is a persistent per-shape autotuner
  that uses repeated medians and stores decisions by exact rocBLAS/ROCm/device
  configuration.
- The solution bench now uses round-robin candidate measurement instead of a
  single sequential pass. For each candidate it reports median/min/max/spread
  across rounds, so we can distinguish a real candidate from a noisy single
  event sample.
  - 224, same-process comparison with `--hip_samples 3 --hip_mlp_solution_bench
    4 --hip_image_tokens_return_profile 1`: selected `up=0 down=-69`. MLP
    improved slightly from 174.338 ms to 169.945 ms, while device_sum moved
    from 305.179 ms to 303.842 ms.
  - 448, same-process comparison with the same probe settings: selected
    `up=-73 down=-71`. MLP improved from 862.576 ms to 795.136 ms, and
    device_sum moved from 1806.513 ms to 1693.481 ms.
  - Interpretation: the more stable probe shows that rocBLAS solution choice
    can matter for the 448 MLP GEMMs, but the effect is shape- and
    run-dependent. The right next step is still a persistent per-shape autotune
    cache, not a hard-coded global default.
- Persistent cache implemented for the probe path:
  - `--hip_mlp_solution_cache PATH` loads the last matching cache entry before
    return bench/profile work.
  - `--hip_mlp_solution_cache_write 1` appends the selected solutions after
    `--hip_mlp_solution_bench N`.
  - The cache key includes HIP runtime version, rocBLAS major/minor/patch,
    GPU arch, CU count, model id, rows, model_dim, and MLP hidden dimension.
    This deliberately avoids applying a solution across different ROCm/device
    or shape configurations.
  - 224 smoke: writing a cache after `--hip_mlp_solution_bench 2` produced
    `up=0 down=-69`; a later run with only `--hip_mlp_solution_cache ...`
    reported a cache hit and used those values in the image-token return
    profile.
- The experimental HIP CLI now has a load-only opt-in parameter:
  `--paligemma_vit_hip_mlp_solution_cache PATH`. This lets
  `gemma_paligemma2_vit_hip` consume a probe-generated cache for real
  image-token return runs without exposing cache writes in the generation CLI.
  A 224 16-token smoke with `--verbosity 0` kept the expected output and did
  not print cache diagnostics before the generated text.
- 448 cache-vs-default stability check:
  - Cache generated with `--hip_samples 5 --hip_warmup 2
    --hip_mlp_solution_bench 4`, selecting `up=-73 down=-71`.
  - Three interleaved profile pairs with `--hip_samples 3 --hip_warmup 1`:
    default MLP 454.090 / 470.798 / 458.872 ms, cache MLP 435.392 / 436.517 /
    433.505 ms.
  - Average MLP improved from 461.253 ms to 435.138 ms, about 5.7%. Average
    device_sum improved from 1039.342 ms to 1013.181 ms, about 2.5%.
  - 448 deterministic 16-token CLI smoke produced the same text with and
    without cache: `A large building with two towers stands tall in the
    distance. The building has a`.
- Next: keep the CLI cache path opt-in. The dedicated scalar HIP MLP GEMM
  prototype below shows that a plain shared-memory tiled kernel is not the right
  path; any custom MLP replacement needs matrix instructions or a specialized
  GEMM generator.

Scalar MLP GEMM prototype:

Implementation:

- Added `--mlp_kernel` and `--mlp_kernel_only` to
  `experimental/paligemma2_vit_hip_bench.cc`.
- The prototype runs BF16 input / F32 output MLP shapes with the same row-major
  layout as gemma.cpp tensors.
- The custom kernel uses a simple 16 x 16 output tile and 32-wide K tile in HIP
  shared memory. Each thread computes one output element with scalar BF16 to F32
  conversion and F32 accumulation.
- This is intentionally a feasibility baseline. It does not use AMD MFMA matrix
  instructions, rocWMMA, CK, or rocBLAS internals.

10-sample run:

```bash
./build/paligemma2_vit_hip_bench --mlp_kernel_only --samples 10 --warmup 3 --iters 1
```

| Shape | rocBLAS median ms | Custom median ms | Custom / rocBLAS | Custom GFLOPS |
| --- | ---: | ---: | ---: | ---: |
| `mlp_up_224` | 2.636 | 5.184 | 1.97x slower | 489.7 |
| `mlp_down_224` | 1.645 | 4.636 | 2.82x slower | 547.6 |
| `mlp_up_448` | 6.815 | 20.939 | 3.07x slower | 484.9 |
| `mlp_down_448` | 7.402 | 21.368 | 2.89x slower | 475.2 |

The custom kernel matched rocBLAS output for this deterministic input
(`max_abs=0`, `rms=0`) but was consistently slower. The meaning is important:

- We should not spend time polishing a scalar shared-memory GEMM kernel for
  PaliGemma2 MLP. The missing capability is not just launch overhead or tiling
  hygiene; it is the lack of matrix-instruction throughput.
- rocBLAS plus per-shape solution-index cache remains the practical near-term
  path for MLP up/down.
- If we continue custom MLP work, the next prototype should be MFMA/rocWMMA or
  a generated kernel path, with fixed PaliGemma2 shapes baked into the tile
  schedule.

RDNA WMMA MLP prototype:

Implementation:

- The local ROCm install does not include rocWMMA headers, so the prototype uses
  clang's AMDGCN builtin directly:
  `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`.
- Added `--mlp_wmma` and `--mlp_wmma_only` to
  `experimental/paligemma2_vit_hip_bench.cc`.
- One wave32 block computes one 16 x 16 F32 output tile from BF16 inputs.
- The fragment loader keeps gemma.cpp tensors in row-major layout:
  - A: `lane % 16` maps to the M row, fragment element maps to K.
  - B: `lane % 16` maps to the N column, fragment element maps to K.
  - C: each lane stores eight F32 elements, with lanes 0-15 writing even rows
    and lanes 16-31 writing odd rows.
- A 16 x 16 row-dependent debug matrix was used to fix the A-fragment mapping.
  The final MLP-shape comparisons report `max_abs=0`, `rms=0` against rocBLAS
  for deterministic BF16 inputs.

10-sample run:

```bash
./build/paligemma2_vit_hip_bench --mlp_wmma_only --samples 10 --warmup 3 --iters 1
```

| Shape | rocBLAS median ms | WMMA median ms | WMMA / rocBLAS | WMMA GFLOPS |
| --- | ---: | ---: | ---: | ---: |
| `mlp_up_224` | 2.268 | 2.202 | 0.97x | 1152.7 |
| `mlp_down_224` | 2.174 | 1.411 | 0.65x | 1798.5 |
| `mlp_up_448` | 7.022 | 12.374 | 1.76x slower | 820.6 |
| `mlp_down_448` | 7.295 | 12.729 | 1.74x slower | 797.8 |

Additional negative variant:

- A half-wave shuffle variant loaded A/B only in lanes 0-15 and replicated the
  fragments into lanes 16-31 with `__shfl`. It was slower on every shape
  despite halving the global loads, so the direct-load version was kept.

Interpretation:

- This is a real matrix-instruction path, unlike the scalar tiled prototype.
- WMMA can beat or match rocBLAS on the smaller 224 MLP shapes, which confirms
  the RDNA matrix path is usable from this codebase.
- The first direct-load WMMA kernel is not good enough for 448px MLP, which is
  the main target. It lacks the inter-wave/blocking strategy rocBLAS uses to
  reuse A/B tiles across many output tiles.
- This motivated the grouped-N WMMA prototype below. The one-wave-per-16x16-tile
  kernel remains useful as a correctness and instruction-path baseline, but not
  as the 448px integration candidate.

Grouped-N RDNA WMMA MLP prototype:

Implementation:

- Added `--mlp_wmma_group`, `--mlp_wmma_group_only`,
  `--mlp_wmma_group_sweep`, and `--mlp_wmma_group_sweep_only`.
- The grouped kernel uses one workgroup with multiple waves. Every wave computes
  a different 16-column N tile for the same 16 M rows.
- For each 16-wide K chunk, the workgroup stages one shared 16 x 16 A tile in
  LDS and reuses it across all N waves. B tiles remain direct global loads
  because each wave owns different columns.
- Sweep variants instantiate 2, 4, and 8 waves per workgroup. The default
  grouped path currently uses 8 waves because it was the best measured candidate
  on this iGPU.

Default 8-wave grouped run:

```bash
./build/paligemma2_vit_hip_bench --mlp_wmma_group_only --samples 10 --warmup 3 --iters 1
```

| Shape | rocBLAS median ms | WMMA8 median ms | WMMA8 / rocBLAS | WMMA8 GFLOPS |
| --- | ---: | ---: | ---: | ---: |
| `mlp_up_224` | 2.725 | 1.452 | 0.53x | 1748.2 |
| `mlp_down_224` | 2.296 | 0.869 | 0.38x | 2919.7 |
| `mlp_up_448` | 7.864 | 7.690 | 0.98x | 1320.5 |
| `mlp_down_448` | 6.425 | 3.819 | 0.59x | 2658.9 |

The grouped kernel matched rocBLAS output for all four deterministic MLP shapes
(`max_abs=0`, `rms=0`).

448-specific 2/4/8 sweep highlights:

- `mlp_up_448`, 10 samples:
  - WMMA2: 10.883 ms
  - WMMA4: 9.358 ms
  - WMMA8: 7.795 ms
- `mlp_down_448`, 10 samples:
  - WMMA2: 7.741 ms
  - WMMA4: 7.686 ms
  - WMMA8: 3.723 ms

Interpretation:

- LDS reuse across neighboring N tiles is the first custom MLP optimization that
  changes the 448 result materially in the right direction.
- 448 down-projection is now a strong custom-kernel integration candidate in
  isolation.
- 448 up-projection is roughly at rocBLAS parity, not a clear win yet. It likely
  needs additional B-side reuse, vectorized/coalesced loads, or a wider fixed
  workgroup schedule before replacing rocBLAS in the end-to-end backend.
- The next engineering step is not more scalar tuning. It is either:
  - integrate the grouped WMMA path behind a probe-only MLP option and measure
    full image-token return profiles; or
  - build a second grouped kernel that also improves B/load scheduling for the
    `mlp_up_*` shape.

Probe-only WMMA8 MLP-down integration:

Implementation:

- Added `PaliGemma2VitHipOptions::use_mlp_down_wmma8`.
- Added probe flag `--hip_mlp_down_wmma8 0|1`.
- The BF16-only ViT layer runner now keeps MLP up on rocBLAS and can replace
  only MLP down with the grouped-N RDNA WMMA8 kernel.
- The profile header prints `mlp_down=rocblas` or `mlp_down=wmma8` so profile
  logs are self-describing.
- The option is disabled by default and is not exposed through the production
  generation CLI yet.

448 return-profile comparison with the existing rocBLAS solution cache loaded:

```bash
./build/paligemma2_vit_hip_backend_probe \
  --weights build/models/paligemma2-3b-mix-448-hf/paligemma2-3b-mix-448-sfp-from-hf.sbs \
  --hip_schedule 0 \
  --hip_resident_schedule 0 \
  --hip_upload 1 \
  --hip_image_tokens_return_profile 1 \
  --hip_mlp_solution_cache build/paligemma2_vit_mlp_solution_cache_448.txt \
  --hip_samples 3 \
  --hip_warmup 1 \
  --hip_iters 1 \
  --hip_verbose 0
```

| 448 profile | MLP up | MLP act | MLP down | MLP total | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| rocBLAS/cache | 192.014 | 15.111 | 197.807 | 413.392 | 965.252 |
| WMMA8 down + cache up | 194.606 | 15.070 | 102.903 | 321.130 | 866.306 |

448 interpretation:

- WMMA8 down cuts the full-stack accumulated MLP-down time by about 48%.
- Total MLP time improves by about 22%.
- Device-sum profile time improves by about 10%.
- This confirms the standalone `mlp_down_448` microbench win transfers into the
  real image-token return graph.

224 return-profile comparison without a cache:

| 224 profile | MLP up | MLP act | MLP down | MLP total | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| rocBLAS | 49.013 | 2.543 | 47.112 | 100.833 | 187.312 |
| WMMA8 down | 49.758 | 2.631 | 28.521 | 82.946 | 167.483 |

224 interpretation:

- WMMA8 down cuts accumulated MLP-down time by about 39%.
- Device-sum profile time improves by about 11%.
- The benefit is smaller in absolute milliseconds than 448, but the shape is
  healthy and the same integration path works for both image sizes.

Current integration decision:

- `--hip_mlp_down_wmma8` is now backed by an internal rocBLAS-vs-WMMA output
  check and exposed to the experimental `gemma_paligemma2_vit_hip` CLI as
  `--paligemma_vit_hip_mlp_down_wmma8`.
- `mlp_up_448` was the next major MLP target after down integration. The
  grouped-M experiment below rejects one simple B-reuse schedule; the grouped-2D
  experiment after that is the first clear custom-kernel win for MLP up.

Grouped-M WMMA experiment for MLP-up:

Implementation:

- Added a benchmark-only grouped-M RDNA WMMA prototype to
  `experimental/paligemma2_vit_hip_bench.cc`.
- The grouped-N kernel above reuses the A activation tile across neighboring N
  tiles. The grouped-M variant instead reuses the B/weight tile across
  neighboring M tiles. This directly tests whether `mlp_up_448` is bottlenecked
  by repeated reads of the wide `linear_0` weight matrix.
- Added `--mlp_wmma_m_group_sweep` and `--mlp_wmma_m_group_sweep_only`, with 2,
  4, and 8 waves per workgroup.

Targeted `mlp_up_448` run:

```bash
./build/paligemma2_vit_hip_bench \
  --mlp_wmma_m_group_sweep_only \
  --filter mlp_up_448 \
  --samples 10 \
  --warmup 3 \
  --iters 1
```

| Shape | Variant | rocBLAS median ms | grouped-M median ms | grouped-M / rocBLAS | GFLOPS |
| --- | --- | ---: | ---: | ---: | ---: |
| `mlp_up_448` | M2 | 10.814 | 10.140 | 0.94x | 1001.4 |
| `mlp_up_448` | M4 | 9.182 | 9.989 | 1.09x | 1016.5 |
| `mlp_up_448` | M8 | 8.463 | 10.111 | 1.19x | 1004.3 |

Same-kernel `mlp_down_448` sanity run:

| Shape | Variant | rocBLAS median ms | grouped-M median ms | grouped-M / rocBLAS | GFLOPS |
| --- | --- | ---: | ---: | ---: | ---: |
| `mlp_down_448` | M2 | 10.031 | 7.777 | 0.78x | 1305.8 |
| `mlp_down_448` | M4 | 7.619 | 8.229 | 1.08x | 1233.9 |
| `mlp_down_448` | M8 | 7.275 | 10.019 | 1.38x | 1013.5 |

Interpretation:

- Grouped-M is numerically correct for these deterministic benchmark shapes, but
  it is not a useful integration candidate.
- For `mlp_up_448`, grouped-M M2 is only a marginal same-run win over rocBLAS and
  is slower than the existing grouped-N 8-wave result under comparable settings.
- For `mlp_down_448`, grouped-M is far slower than the already integrated
  grouped-N WMMA8 path.
- The result rejects the simple "stage only B across M waves" hypothesis. The
  next `mlp_up_448` path should focus on a different schedule: wider/coalesced B
  vector loads, deeper K blocking, or a two-dimensional macro-tile that reuses
  both A and B without adding enough synchronization to erase the benefit.

Grouped-2D WMMA experiment for MLP-up:

Implementation:

- Added a benchmark-only grouped-2D RDNA WMMA prototype to
  `experimental/paligemma2_vit_hip_bench.cc`.
- Each workgroup owns a small M x N macro-tile. Each wave still computes one
  16 x 16 output tile, but the workgroup stages both A and B in LDS for every
  16-wide K chunk.
- The key tested schedules are 2x2, 2x4, 4x2, and 4x4 waves per workgroup. The
  4x4 case uses 16 waves and computes a 64 x 64 macro-tile.

Targeted 448 runs:

```bash
./build/paligemma2_vit_hip_bench \
  --mlp_wmma_2d_group_sweep_only \
  --filter mlp_up_448 \
  --samples 10 \
  --warmup 3 \
  --iters 1
```

| Shape | Variant | rocBLAS median ms | grouped-2D median ms | grouped-2D / rocBLAS | GFLOPS |
| --- | --- | ---: | ---: | ---: | ---: |
| `mlp_up_448` | 2x2 | 8.824 | 8.656 | 0.98x | 1173.2 |
| `mlp_up_448` | 2x4 | 7.975 | 7.125 | 0.89x | 1425.1 |
| `mlp_up_448` | 4x2 | 7.814 | 6.284 | 0.80x | 1615.9 |
| `mlp_up_448` | 4x4 | 8.402 | 4.921 | 0.59x | 2063.4 |

| Shape | Variant | rocBLAS median ms | grouped-2D median ms | grouped-2D / rocBLAS | GFLOPS |
| --- | --- | ---: | ---: | ---: | ---: |
| `mlp_down_448` | 2x2 | 8.051 | 6.994 | 0.87x | 1451.8 |
| `mlp_down_448` | 2x4 | 7.662 | 5.453 | 0.71x | 1862.0 |
| `mlp_down_448` | 4x2 | 7.319 | 5.632 | 0.77x | 1803.0 |
| `mlp_down_448` | 4x4 | 6.940 | 4.653 | 0.67x | 2182.2 |

Targeted 224 runs:

| Shape | Variant | rocBLAS median ms | grouped-2D median ms | grouped-2D / rocBLAS | GFLOPS |
| --- | --- | ---: | ---: | ---: | ---: |
| `mlp_up_224` | 2x2 | 2.956 | 1.927 | 0.65x | 1317.4 |
| `mlp_up_224` | 2x4 | 2.424 | 1.689 | 0.70x | 1503.5 |
| `mlp_up_224` | 4x2 | 2.046 | 1.476 | 0.72x | 1719.8 |
| `mlp_up_224` | 4x4 | 1.859 | 1.225 | 0.66x | 2073.1 |

| Shape | Variant | rocBLAS median ms | grouped-2D median ms | grouped-2D / rocBLAS | GFLOPS |
| --- | --- | ---: | ---: | ---: | ---: |
| `mlp_down_224` | 2x2 | 2.792 | 1.938 | 0.69x | 1310.1 |
| `mlp_down_224` | 2x4 | 2.234 | 1.643 | 0.74x | 1545.3 |
| `mlp_down_224` | 4x2 | 2.077 | 1.690 | 0.81x | 1502.1 |
| `mlp_down_224` | 4x4 | 1.708 | 1.333 | 0.78x | 1904.0 |

Interpretation:

- Grouped-2D 4x4 is the first custom kernel that is clearly better than rocBLAS
  for `mlp_up_448`, and it is also faster than the previous grouped-N MLP-up
  result.
- It is also good for `mlp_up_224`, reducing the standalone up-projection median
  from the previous grouped-N result of about 1.45 ms to about 1.23 ms.
- It should not replace the existing grouped-N WMMA8 down path: grouped-2D down
  beats rocBLAS, but grouped-N down remains faster on both 224 and 448.
- Next integration target: add an experimental MLP-up 4x4 grouped-2D path in the
  HIP backend, keep MLP-down on grouped-N WMMA8, and measure the combined
  full-stack return profile.

Probe-only MLP-up grouped-2D integration:

Implementation:

- Added `PaliGemma2VitHipOptions::use_mlp_up_wmma2d`.
- Added probe flag `--hip_mlp_up_wmma2d 0|1`.
- Added probe check flag `--hip_mlp_up_wmma2d_check 0|1`.
- Added generation CLI flag `--paligemma_vit_hip_mlp_up_wmma2d`.
- The BF16-only ViT layer runner now has two independent opt-in MLP projection
  replacements:
  - MLP up: grouped-2D RDNA WMMA, initially 4x4 and later retuned to 8x4.
  - MLP down: grouped-N RDNA WMMA8.
- The return-profile header now prints both implementation choices, for example
  `mlp_up=wmma2d8x4 mlp_down=wmma8`.

MLP-up WMMA2D real-tensor check:

- The check builds the real layer0 BF16 pre-MLP tensor from the test image and
  compares `linear_0` rocBLAS output against grouped-2D WMMA before bias and
  GELU.

| Model | rocBLAS ms | WMMA2D 4x4 ms | max abs | RMS |
| --- | ---: | ---: | ---: | ---: |
| 224 | 3.899 | 2.144 | 0.000042 | 0.000004 |
| 448 | 12.694 | 5.117 | 0.000043 | 0.000004 |

448 return-profile with MLP-up 2D 4x4 plus MLP-down WMMA8, using the existing
rocBLAS MLP solution cache for remaining rocBLAS MLP paths:

```bash
./build/paligemma2_vit_hip_backend_probe \
  --weights build/models/paligemma2-3b-mix-448-hf/paligemma2-3b-mix-448-sfp-from-hf.sbs \
  --hip_schedule 0 \
  --hip_resident_schedule 0 \
  --hip_upload 1 \
  --hip_image_tokens_return_profile 1 \
  --hip_mlp_solution_cache build/paligemma2_vit_mlp_solution_cache_448.txt \
  --hip_samples 3 \
  --hip_warmup 1 \
  --hip_iters 1 \
  --hip_verbose 0 \
  --hip_mlp_up_wmma2d 1 \
  --hip_mlp_down_wmma8 1
```

| 448 profile | MLP up | MLP act | MLP down | MLP total | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| rocBLAS/cache | 192.014 | 15.111 | 197.807 | 413.392 | 965.252 |
| WMMA8 down + cache up | 194.606 | 15.070 | 102.903 | 321.130 | 866.306 |
| WMMA2D up + WMMA8 down | 138.665 | 11.813 | 103.577 | 263.000 | 770.684 |

448 interpretation:

- Adding MLP-up grouped-2D cuts accumulated MLP-up time by about 29% versus the
  previous down-only profile.
- Combined MLP up/down custom kernels cut total MLP time by about 36% versus
  rocBLAS/cache and about 18% versus down-only WMMA8.
- Device-sum profile time improves by about 20% versus rocBLAS/cache and about
  11% versus down-only WMMA8.

224 return-profile with the same MLP-up 2D 4x4 plus MLP-down WMMA8 combination:

| 224 profile | MLP up | MLP act | MLP down | MLP total | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| rocBLAS | 49.013 | 2.543 | 47.112 | 100.833 | 187.312 |
| WMMA8 down | 49.758 | 2.631 | 28.521 | 82.946 | 167.483 |
| WMMA2D up + WMMA8 down | 33.368 | 2.490 | 24.086 | 61.718 | 145.320 |

224 interpretation:

- MLP-up grouped-2D also transfers to the smaller 224 image path.
- Combined custom MLP kernels cut total MLP time by about 39% versus rocBLAS and
  about 26% versus down-only WMMA8.
- Device-sum profile time improves by about 22% versus rocBLAS and about 13%
  versus down-only WMMA8.

Generation CLI smoke with both custom MLP kernels:

```bash
./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-448-hf/paligemma2-3b-mix-448-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --verbosity 0 \
  --max_generated_tokens 16 \
  --temperature 0 \
  --top_k 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_return_image_tokens 1 \
  --paligemma_vit_hip_mlp_up_wmma2d 1 \
  --paligemma_vit_hip_mlp_down_wmma8 1 \
  --paligemma_vit_hip_mlp_solution_cache build/paligemma2_vit_mlp_solution_cache_448.txt
```

The 448 deterministic 16-token output matched the earlier rocBLAS/cache and
down-only WMMA8 runs:
`A large building with two towers stands tall in the distance. The building has
a`.

The 224 deterministic 16-token output also matched the earlier run:
`A long-shot view of a large white church. The church has two towers`.

MLP-up grouped-2D 8x4 retune:

- The latest profile still showed MLP as the largest 448px stage after attention
  QK/pack work:
  - recommended F32 attention + pack: MLP `235.185 ms`, with MLP-up
    `128.129 ms`;
  - aggressive QK WMMA + pack: MLP `235.949 ms`, with MLP-up `126.667 ms`.
- Expanded the bench-only grouped-2D sweep beyond 2x2/2x4/4x2/4x4 to include
  2x8, 4x8, 8x2, and 8x4 macro-tiles.
- `8x4` was the best MLP-up schedule in the targeted sweep:

| Shape | 4x4 median | 8x4 median | 8x4 GFLOPS | max abs | RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlp_up_224` | 1.160 ms | 0.941 ms | 2697.0 | 0.00000 | 0.00000 |
| `mlp_up_448` | 4.757 ms | 3.500 ms | 2900.9 | 0.00000 | 0.00000 |

- The same wider 2D schedules did not displace the existing MLP-down WMMA8
  route. For example, `mlp_down_448` 2D 4x4 measured `4.221 ms`, while the
  integrated grouped-N WMMA8 down path remains the better runtime choice.
- Updated `PaliGemma2VitHipOptions::use_mlp_up_wmma2d` to use the 8x4 schedule
  and updated the profile label to `mlp_up=wmma2d8x4`.

Real-tensor check after the 8x4 retune:

| Model | rocBLAS ms | WMMA2D 8x4 ms | max abs | RMS |
| --- | ---: | ---: | ---: | ---: |
| 224 | 2.448 | 0.963 | 0.000042 | 0.000004 |
| 448 | 8.119 | 4.123 | 0.000043 | 0.000004 |

Return-profile after the MLP-up 8x4 retune, before the later QKV/attn_out
8x4 retune, keeping QKV 4x4, attn_out 4x4, MLP-down WMMA8, and F32
attention + pack:

| Shape | MLP up | MLP total | Attention | Device sum |
| --- | ---: | ---: | ---: | ---: |
| 224 before 8x4 | 34.244 ms | 63.748 ms | 18.259 ms | 128.262 ms |
| 224 after 8x4 | 27.282 ms | 55.568 ms | 17.749 ms | 117.363 ms |
| 448 before 8x4 | 128.129 ms | 235.185 ms | 186.303 ms | 592.267 ms |
| 448 after 8x4 | 101.953 ms | 210.634 ms | 181.037 ms | 565.866 ms |

Guardrails after the 8x4 retune:

- Recommended 224/448 smoke outputs remain unchanged.
- F32 attention + `attention_pack_bf16` extended 448 stress set: pass,
  including `dark_radial.ppm`.

Probe-only QKV grouped-2D integration:

Implementation:

- Extended the standalone grouped-2D benchmark to non-MLP ViT projection shapes
  with `--projection_wmma_2d_sweep` and
  `--projection_wmma_2d_sweep_only`.
- Generalized the backend grouped-2D kernel into a BF16 GEMM kernel used by both
  MLP-up and QKV.
- Added `PaliGemma2VitHipOptions::use_qkv_wmma2d`.
- Added probe flag `--hip_qkv_wmma2d 0|1`.
- Added probe check flag `--hip_qkv_wmma2d_check 0|1`.
- Added generation CLI flag `--paligemma_vit_hip_qkv_wmma2d`.
- The return-profile header now prints QKV as well, for example
  `qkv=wmma2d8x4 mlp_up=wmma2d8x4 mlp_down=wmma8`.

Projection grouped-2D 4x4 standalone highlights:

| Shape | rocBLAS median ms | WMMA2D 4x4 median ms | WMMA2D / rocBLAS | GFLOPS |
| --- | ---: | ---: | ---: | ---: |
| `qkv_224` | 1.402 | 0.939 | 0.67x | 2170.0 |
| `qkv_448` | 6.434 | 4.064 | 0.63x | 2006.4 |
| `attn_out_224` | 0.500 | 0.357 | 0.71x | 1901.4 |
| `attn_out_448` | 1.813 | 1.313 | 0.72x | 2070.1 |

QKV WMMA2D real-tensor check:

| Model | rocBLAS ms | WMMA2D 4x4 ms | max abs | RMS |
| --- | ---: | ---: | ---: | ---: |
| 224 | 2.340 | 1.383 | 0.000023 | 0.000002 |
| 448 | 6.120 | 3.971 | 0.000027 | 0.000002 |

Return-profile with QKV WMMA2D plus both custom MLP kernels:

| Profile | QKV | MLP total | Device sum |
| --- | ---: | ---: | ---: |
| 224, MLP custom only | 41.925 | 61.718 | 145.320 |
| 224, QKV + MLP custom | 28.170 | 60.522 | 129.079 |
| 448, MLP custom only | 165.281 | 263.000 | 770.684 |
| 448, QKV + MLP custom | 118.475 | 265.830 | 738.131 |

Interpretation:

- QKV grouped-2D transfers cleanly from standalone to the full return graph.
- On 448, QKV time drops by about 28% versus the MLP-custom profile, and
  device-sum profile time improves by another 4%.
- On 224, QKV time drops by about 33%, and device-sum profile time improves by
  another 11%.
- The deterministic 16-token generation smoke still matches for both 224 and
  448 when QKV WMMA2D, MLP-up WMMA2D, and MLP-down WMMA8 are all enabled.

QKV grouped-2D 8x4 retune:

- Extended `--projection_wmma_2d_sweep_only` with the same wider macro-tiles
  used for the MLP-up retune: 2x8, 4x8, 8x2, and 8x4.
- `8x4` is the best single runtime schedule for QKV across the two target image
  sizes. On 224, 4x8 was slightly faster in one microbench run, but 8x4 was
  close and won clearly on 448, so the backend uses one schedule for both.

| Shape | 4x4 median | 8x4 median | 8x4 GFLOPS | max abs | RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qkv_224` | 0.905 ms | 0.785 ms | 2598.1 | 0.00000 | 0.00000 |
| `qkv_448` | 3.873 ms | 3.316 ms | 2459.1 | 0.00000 | 0.00000 |

Real-tensor QKV check after the 8x4 retune:

| Model | rocBLAS ms | WMMA2D 8x4 ms | max abs | RMS |
| --- | ---: | ---: | ---: | ---: |
| 224 | 2.113 | 0.756 | 0.000023 | 0.000002 |
| 448 | 6.129 | 2.974 | 0.000027 | 0.000002 |

Probe-only attention-output grouped-2D integration:

Implementation:

- Added `PaliGemma2VitHipOptions::use_attn_out_wmma2d`.
- Added probe flag `--hip_attn_out_wmma2d 0|1`.
- Added probe check flag `--hip_attn_out_wmma2d_check 0|1`.
- Added generation CLI flag `--paligemma_vit_hip_attn_out_wmma2d`.
- The BF16-only ViT layer runner can now independently replace QKV, attention
  output, MLP up, and MLP down while keeping each replacement behind an explicit
  opt-in flag.

Attention-output WMMA2D real-tensor check:

- The check prepares a deterministic BF16 attention output tensor, then compares
  only the `attn_out` GEMM boundary between rocBLAS BF16 and grouped-2D WMMA
  4x4 before bias/residual.

| Model | rocBLAS ms | WMMA2D 4x4 ms | max abs | RMS |
| --- | ---: | ---: | ---: | ---: |
| 224 | 0.552 | 0.447 | 0.000019 | 0.000000 |
| 448 | 3.538 | 1.728 | 0.000019 | 0.000000 |

Return-profile with QKV, attention-output, and MLP custom kernels:

| Profile | QKV | Attention | Attn out | MLP total | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224, QKV + MLP custom | 28.170 | 20.492 | 15.542 | 60.522 | 129.079 |
| 224, QKV + attn-out + MLP custom | 28.379 | 21.062 | 11.769 | 61.172 | 126.837 |
| 448, QKV + MLP custom | 118.475 | 276.885 | 58.807 | 265.830 | 738.131 |
| 448, QKV + attn-out + MLP custom | 121.357 | 292.138 | 44.748 | 263.767 | 740.830 |

Interpretation:

- The attn_out WMMA2D kernel is correct and consistently improves the isolated
  projection boundary.
- On 224, the full return profile also improves in device_sum.
- On 448, attn_out itself improves by about 24% in the repeated all-custom
  profile, but the full device_sum is roughly flat because the attention core
  dominates and varies more from run to run than the attn_out saving.
- The deterministic 16-token generation smoke still matches for both 224 and
  448 when QKV WMMA2D, attn_out WMMA2D, MLP-up WMMA2D, and MLP-down WMMA8 are
  all enabled.

Attention-output grouped-2D 8x4 retune:

- The wider projection sweep also favored 8x4 for attention output. It improves
  448 most clearly and is best or tied on 224.

| Shape | 4x4 median | 8x4 median | 8x4 GFLOPS | max abs | RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attn_out_224` | 0.367 ms | 0.342 ms | 1987.5 | 0.00000 | 0.00000 |
| `attn_out_448` | 1.155 ms | 1.054 ms | 2578.7 | 0.00000 | 0.00000 |

Real-tensor attention-output check after the 8x4 retune:

| Model | rocBLAS ms | WMMA2D 8x4 ms | max abs | RMS |
| --- | ---: | ---: | ---: | ---: |
| 224 | 0.838 | 0.314 | 0.000019 | 0.000000 |
| 448 | 2.600 | 0.983 | 0.000019 | 0.000000 |

Return-profile after the QKV/attn_out 8x4 retune, keeping MLP-up 8x4,
MLP-down WMMA8, and F32 attention + pack:

| Shape | QKV | Attn out | MLP total | Attention | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 before projection 8x4 | 27.615 ms | 11.664 ms | 55.568 ms | 17.749 ms | 117.363 ms |
| 224 after projection 8x4 | 22.970 ms | 10.446 ms | 54.451 ms | 17.279 ms | 109.216 ms |
| 448 before projection 8x4 | 114.983 ms | 41.216 ms | 210.634 ms | 181.037 ms | 565.866 ms |
| 448 after projection 8x4 | 94.347 ms | 37.720 ms | 217.561 ms | 175.391 ms | 543.476 ms |

Guardrails after the projection 8x4 retune:

- Recommended 224/448 smoke outputs remain unchanged.
- F32 attention + `attention_pack_bf16` extended 448 stress set: pass,
  including `dark_radial.ppm`.

Updated integration decision:

- Keep QKV grouped-2D 8x4, attn_out grouped-2D 8x4, MLP-up grouped-2D 8x4, and
  MLP-down grouped-N WMMA8 as independent, explicit experimental flags for now.
- The next performance target is no longer MLP down, and MLP up is no longer
  rocBLAS-bound in isolation. QKV and attention-output projection are also
  improved. The remaining dominant cost is the attention core itself, which
  still requires a better fused/tiled design before replacing the current
  implementation.

Attention core phase profile:

- Added benchmark-only attention phase flags to
  `experimental/paligemma2_vit_hip_bench.cc`:
  - `--attention_phase_profile`
  - `--attention_bf16_phase_profile`
  - `--attention_bf16_qk_phase_profile`
- The first flag measures the current split + F32 QK + F32 softmax + F32 AV +
  pack path.
- The second flag measures a full BF16 QK/AV experiment. It converts Q/K/V to
  BF16, runs QK into F32 scores, performs F32 softmax, quantizes the full score
  matrix to BF16, then runs BF16 AV.
- The third flag measures a narrower BF16-QK-only experiment. It converts only
  Q/K to BF16, keeps V and softmax scores in F32, then runs F32 AV. This avoids
  quantizing the large `heads x rows x rows` score matrix.

Initial all-shape run:

| Shape | Path | Split | QK | Softmax | Quant | AV | Pack | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | F32 QK + F32 AV | 0.102 | 0.234 | 0.211 | - | 0.457 | 0.039 | 0.864 |
| 224 | BF16 QK + BF16 AV | 0.076 | 0.181 | 0.158 | 0.088 | 0.225 | 0.035 | 0.642 |
| 224 | BF16 QK + F32 AV | 0.085 | 0.140 | 0.159 | - | 0.259 | 0.037 | 0.676 |
| 448 | F32 QK + F32 AV | 0.446 | 4.321 | 2.203 | - | 3.921 | 0.133 | 10.284 |
| 448 | BF16 QK + BF16 AV | 0.367 | 3.247 | 2.722 | 1.305 | 3.818 | 0.151 | 10.237 |
| 448 | BF16 QK + F32 AV | 0.388 | 2.892 | 2.200 | - | 3.953 | 0.134 | 9.622 |

448-only rerun with 30 samples:

| Path | Split | QK | Softmax | Quant | AV | Pack | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F32 QK + F32 AV | 0.423 | 3.837 | 2.176 | - | 4.040 | 0.134 | 10.667 |
| BF16 QK + BF16 AV | 0.330 | 2.606 | 2.277 | 1.345 | 4.934 | 0.152 | 10.189 |
| BF16 QK + F32 AV | 0.364 | 2.880 | 2.271 | - | 3.932 | 0.137 | 9.407 |

Interpretation:

- The attention core is now the largest remaining 448px cost after projection
  custom kernels. In the current F32 path, QK, softmax, and AV are all
  meaningful; no single tiny elementwise kernel explains the whole time.
- Full BF16 QK/AV is not a clean integration target. BF16 QK is faster, but
  quantizing the softmax scores costs about 1.3 ms/layer at 448, and BF16 AV did
  not consistently improve the 448 total.
- BF16-QK-only is the best measured attention-core candidate so far. On the
  448-only rerun it reduced one attention layer from 10.667 ms to 9.407 ms,
  about 12% for that phase. Across 27 ViT layers, the isolated phase-level
  saving is roughly 34 ms before accounting for full-graph variance.
- This is not yet a backend correctness result. It introduces an extra BF16
  rounding boundary on Q/K before softmax, so the next step must validate the
  final image tokens and deterministic generation output before enabling it in
  the HIP return path.

Next attention-core action:

- Add an explicit backend option such as `use_attention_qk_bf16`.
- Keep V, softmax scores, and AV in F32.
- Allocate true BF16 Q/K attention scratch buffers in the resident workspace.
- Compare attention output against the current F32 attention path, then run
  image-token and deterministic generation smoke tests before measuring the
  full return profile.

Probe-only attention QK BF16 backend integration:

- Added `PaliGemma2VitHipOptions::use_attention_qk_bf16`.
- Added probe flag `--hip_attention_qk_bf16 0|1`.
- Added generation CLI flag `--paligemma_vit_hip_attention_qk_bf16`.
- The BF16-only ViT layer runner now has two attention-core choices:
  - default: split Q/K/V to F32, F32 QK, F32 softmax, F32 AV;
  - opt-in: split Q/K to true BF16 scratch, keep V in F32, BF16-input QK with
    F32 scores, F32 softmax, F32 AV.
- The resident workspace now includes true BF16 Q/K attention scratch buffers.
  The older `q_bf16`/`k_bf16` names remain F32 scratch for the default path;
  they were not renamed in this pass to keep the diff focused.

Return-profile with QK BF16 attention plus QKV, attention-output, and MLP custom
kernels:

| Profile | Attention | QKV | Attn out | MLP total | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224, F32 attention | 21.759 | 31.569 | 12.374 | 66.604 | 137.450 |
| 224, QK BF16 attention | 21.042 | 30.870 | 12.601 | 66.209 | 135.445 |
| 448, F32 attention | 276.793 | 117.537 | 41.509 | 247.292 | 702.199 |
| 448, QK BF16 attention | 251.721 | 119.207 | 43.651 | 250.713 | 684.840 |

Interpretation:

- QK BF16 transfers from the standalone attention phase benchmark into the full
  27-layer return/profile graph.
- The 448 accumulated attention phase improves by 25.072 ms, about 9.1%, and
  full device_sum improves by 17.359 ms, about 2.5%, in this run.
- The 224 path also improves slightly, so this opt-in does not appear to be a
  448-only tradeoff.
- Some non-attention stages vary between runs, so the stage-level attention
  number is the more reliable signal than total device_sum for this specific
  change.

Generation CLI smoke with QK BF16 attention plus all currently integrated custom
projection/MLP kernels:

```bash
./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-448-hf/paligemma2-3b-mix-448-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --verbosity 0 \
  --max_generated_tokens 16 \
  --temperature 0 \
  --top_k 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_return_image_tokens 1 \
  --paligemma_vit_hip_attention_qk_bf16 1 \
  --paligemma_vit_hip_qkv_wmma2d 1 \
  --paligemma_vit_hip_attn_out_wmma2d 1 \
  --paligemma_vit_hip_mlp_up_wmma2d 1 \
  --paligemma_vit_hip_mlp_down_wmma8 1 \
  --paligemma_vit_hip_mlp_solution_cache build/paligemma2_vit_mlp_solution_cache_448.txt
```

The 224 deterministic 16-token output still matched:
`A long-shot view of a large white church. The church has two towers`.

The 448 deterministic 16-token output still matched:
`A large building with two towers stands tall in the distance. The building has
a`.

QK BF16 image-token A/B check:

- Added probe-only `--hip_attention_qk_bf16_check 0|1`.
- The check runs the same HIP BF16 image-token return path twice:
  - baseline: F32 attention core;
  - candidate: QK BF16 attention core.
- It keeps the selected QKV, attn_out, MLP-up, and MLP-down custom flags the
  same for both runs, then compares the final decoder-visible image tokens.

Results with QKV WMMA2D, attn_out WMMA2D, MLP-up WMMA2D, and MLP-down WMMA8:

| Model | Compared Boundary | observed max abs | observed RMS |
| --- | --- | ---: | ---: |
| 224 | QK BF16 tokens vs F32-attention tokens | 0.266071-0.455442 | 0.006032-0.009436 |
| 448 | QK BF16 tokens vs F32-attention tokens | 0.467115-0.469411 | 0.004248-0.004259 |

The 224 token-delta metric showed some run-to-run variation, while the 448
metric was effectively stable across the repeated checks. The deterministic
generation A/B below is therefore the more practical guardrail for now.

448 deterministic generation A/B, 16-token cap:

| Prompt | F32 attention | QK BF16 attention |
| --- | --- | --- |
| `Describe the image.` | `A large building with two towers stands tall in the distance. The building has a` | same |
| `What is in the image?` | `A large building with two towers stands tall in the distance. The building has a` | same |
| `Describe the building.` | `A large, old building with two towers stands tall beside a serene body of water` | same |

Repeatable generation A/B harness:

- Added `scripts/check_paligemma2_attention_qk_bf16_ab.sh`.
- The script compares deterministic stdout for the F32-attention baseline and
  the QK-BF16 candidate while keeping QKV WMMA2D, attn_out WMMA2D, MLP-up
  WMMA2D, and MLP-down WMMA8 enabled for both sides.
- It runs both local 224 and 448 SBS models. By default it uses the repo's
  single PPM test image and three prompts:
  - `Describe the image.`
  - `What is in the image?`
  - `Describe the building.`
- Result on the current local image set: all 6 model/prompt pairs matched.
- The script accepts `IMAGE=...`, `MODEL_224=...`, `MODEL_448=...`,
  `MLP_CACHE_448=...`, and `MAX_GENERATED_TOKENS=...` environment overrides,
  so additional PPM images can be swept without changing the C++ code.
- The script now also accepts list files:
  - `IMAGES_FILE=scripts/paligemma2_attention_qk_bf16_images.txt`
  - `PROMPTS_FILE=scripts/paligemma2_attention_qk_bf16_prompts.txt`
- List files use one item per line and ignore blank lines plus `#` comments.
- Both default mode and explicit list-file mode passed on the current local
  image/prompt set.
- Added `scripts/generate_paligemma2_qk_bf16_test_images.py`, a dependency-free
  generator for deterministic synthetic PPM fixtures:
  - `gradient_blocks.ppm`
  - `checker_edges.ppm`
  - `color_quadrants.ppm`
  - `dark_radial.ppm`
- These synthetic images were added to
  `scripts/paligemma2_attention_qk_bf16_images.txt` so the extended sweep now
  covers 5 images x 3 prompts x 2 local models.

Extended synthetic sweep result:

- Command:

```bash
KEEP_GOING=1 MAX_GENERATED_TOKENS=8 \
  IMAGES_FILE=scripts/paligemma2_attention_qk_bf16_images.txt \
  PROMPTS_FILE=scripts/paligemma2_attention_qk_bf16_prompts.txt \
  scripts/check_paligemma2_attention_qk_bf16_ab.sh
```

- 224: all 15 image/prompt cases matched.
- 448: 14 of 15 image/prompt cases matched.
- The one mismatch is a stable negative case:
  - image: `paligemma/testdata/qk_bf16_sweep/dark_radial.ppm`
  - prompt: `Describe the image.`
  - F32 attention, 16-token cap: `a close up of a circle pattern`
  - QK BF16 attention, 16-token cap: `a blue swirl`
- The same negative case has final image-token delta
  max_abs=`0.207040`, RMS=`0.004867` vs F32-attention HIP tokens.

Probe image input cleanup:

- The backend probe no longer hardcodes `paligemma/testdata/image.ppm` in every
  validation branch.
- It now uses `--image_file` when provided and falls back to the repo test image
  when omitted. This applies to patch/QKV/MLP/attention checks, image-token
  validation, return profiling, and `--hip_attention_qk_bf16_check`.

Current attention-core decision:

- Keep QK BF16 behind an explicit opt-in flag for now.
- It is a valid next-stage optimization candidate because it improves the
  current rocBLAS-assisted attention path without introducing the expensive
  full-score BF16 quantization step.
- Do not make QK BF16 the default. The expanded synthetic sweep found a real
  deterministic generation mismatch at 448, so this remains an opt-in speed
  experiment rather than a safe precision boundary.
- The next performance direction should shift away from BF16-ing more of
  attention. Use the negative case as a guardrail while prototyping fused/tiled
  attention that keeps the numerically sensitive softmax boundary in F32.

F32 softmax+AV fused prototype:

- Added benchmark-only `--attention_softmax_av_fused_profile`.
- This keeps QK as F32 rocBLAS and keeps softmax in F32, but replaces the
  separate softmax kernel plus rocBLAS AV with one scalar HIP kernel that:
  - reads one score row;
  - computes F32 softmax max/sum;
  - directly accumulates the weighted V output for that query/head.
- The goal was to test whether avoiding normalized-score writeback and the AV
  rocBLAS launch could offset the loss of rocBLAS AV.

Measured result:

| Shape | Current total | Fused softmax+AV total | Interpretation |
| --- | ---: | ---: | --- |
| 224 | 1.322 ms | 2.382 ms | slower |
| 448 | 10.226 ms | 34.123 ms | much slower |

448 phase detail:

| Path | Split | QK | Softmax | AV | Fused softmax+AV | Pack | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current F32 | 0.489 | 5.380 | 1.998 | 3.557 | - | 0.146 | 10.226 |
| fused softmax+AV | 0.458 | 3.978 | - | - | 29.452 | 0.352 | 34.123 |

Interpretation:

- A scalar fused softmax+AV kernel is not viable on this iGPU. It saves a
  global score write/read boundary, but loses the matrix-instruction throughput
  of rocBLAS AV and rereads V too inefficiently.
- The next attention prototype should not be a simple per-row scalar fusion.
  It needs either:
  - keep rocBLAS AV and optimize around QK/softmax/packing; or
  - implement a real tiled/MFMA AV path with V reuse and multiple query rows per
    workgroup.
- This negative result also argues against integrating any fused attention path
  into the backend until it beats the current rocBLAS-assisted baseline in the
  standalone bench.

Grouped scalar softmax+AV prototype:

- Added benchmark-only `--attention_softmax_av_grouped_profile_only`.
- This keeps cached F32 rocBLAS QK and compares:
  - baseline: cached QK, fixed F32 softmax, cached rocBLAS AV;
  - grouped prototype: cached QK, one scalar HIP kernel that computes softmax
    weights once per row and splits each output-dimension AV reduction across
    multiple lane groups.
- The intent was to repair the old per-row fused softmax+AV prototype's worst
  issue: it recomputed `exp(score - max)` separately for each output dimension
  and did not parallelize a single output dimension across keys.

Command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_softmax_av_grouped_profile_only \
  --samples 50 \
  --warmup 15 \
  --iters 1
```

Measured result:

| Shape | Baseline QK+softmax+AV | Grouped QK+softmax/AV | Relative | max abs | RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.542 ms | 1.838 ms | 3.392 | 0.000001 | 0.000000 |
| 448 | 6.080 ms | 18.073 ms | 2.973 | 0.000001 | 0.000000 |

Interpretation:

- The grouped scalar schedule fixes the reduction-order drift when the 224 path
  uses the same 64-thread softmax shape as the fixed softmax kernel.
- It still cannot compete with rocBLAS AV. The saved normalized-score
  write/read is smaller than the throughput lost by replacing a
  matrix-instruction GEMM with scalar weighted-sum loops.
- Do not integrate this path into `hip_probe`.
- The next attention work should stop spending time on scalar AV/softmax
  kernels. The useful remaining directions are matrix-instruction attention
  kernels or fusing around rocBLAS without replacing the AV GEMM.

F32 softmax block-size sweep:

- Added benchmark-only `--attention_softmax_thread_sweep`.
- This keeps the current attention decomposition unchanged, precomputes the F32
  QK scores, and times only `SoftmaxRowsKernel` with different thread-block
  sizes.
- The goal was to find a low-risk tuning point that preserves the F32 attention
  numerics while reducing one of the visible non-GEMM costs.

Command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_softmax_thread_sweep_only \
  --samples 20 \
  --warmup 5 \
  --iters 1
```

Measured result:

| Shape | 64 threads | 128 threads | 256 threads | 512 threads | 1024 threads | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 224 | 0.166 ms | 0.206 ms | 0.316 ms | 0.624 ms | 1.197 ms | use 64 |
| 448 | 2.129 ms | 1.840 ms | 1.826 ms | 2.109 ms | 3.050 ms | keep 256 |

Backend integration:

- `RunDeviceAttention` and `RunDeviceAttentionQkBf16` now use 64 softmax
  threads when `rows <= 256`.
- The 448 path remains on the original 256-thread setting because the sweep
  shows it is still the best measured choice, and 448 is the main correctness
  guardrail for attention experiments.

Return-profile after the backend softmax tuning, with F32 attention plus QKV,
attention-output, MLP-up, and MLP-down custom kernels enabled:

| Shape | Attention | QKV | Attn out | MLP | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 | 20.954 ms | 30.348 ms | 12.726 ms | 64.785 ms | 131.059 ms |

Generation smoke after the backend softmax tuning, with the same custom kernels
enabled and QK BF16 still disabled:

| Shape | Output |
| --- | --- |
| 224 | `A long-shot view of a large white church. The church has two towers` |
| 448 | `A large building with two towers stands tall in the distance. The building has a` |

Interpretation:

- This is a small but safe backend improvement for 224. It does not change the
  precision boundary or introduce a new fused attention path.
- The 448 attention bottleneck is still mostly QK/AV GEMM work and remains the
  right target for a real tiled/MFMA prototype. Thread-count tuning alone will
  not materially change the 448 profile.

Attention rocBLAS solution-index sweep:

- Added benchmark-only `--attention_rocblas_solution_sweep N`.
- The sweep uses rocBLAS' beta solution enumeration API for the current F32
  strided-batched attention GEMMs:
  - QK: `K^T * Q -> scores`, batched over 16 heads;
  - AV: `V * softmax(scores) -> head-major attention`, batched over 16 heads.
- This keeps the current precision boundary and tensor layout unchanged. It
  only asks whether rocBLAS' default solution index `0` is leaving performance
  on the table for these fixed ViT shapes on gfx1150.

Command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_rocblas_solution_sweep_only 8 \
  --samples 20 \
  --warmup 5 \
  --iters 1
```

Measured result:

| Shape | GEMM | Default | Best explicit solution | Relative |
| --- | --- | ---: | ---: | ---: |
| 224 | QK | 0.388 ms | `-110`: 0.203 ms | 0.523 |
| 224 | AV | 0.446 ms | `-1161823143`: 0.354 ms | 0.795 |
| 448 | QK | 4.233 ms | `-111`: 1.805 ms | 0.426 |
| 448 | AV | 3.140 ms | `-451`: 2.176 ms | 0.693 |

Backend/probe/CLI integration:

- Added explicit HIP backend options:
  - `attention_qk_solution_index`;
  - `attention_av_solution_index`.
- Added probe flags:
  - `--hip_attention_qk_solution INDEX`;
  - `--hip_attention_av_solution INDEX`.
- Added generation CLI flags:
  - `--paligemma_vit_hip_attention_qk_solution INDEX`;
  - `--paligemma_vit_hip_attention_av_solution INDEX`.
- Zero keeps the rocBLAS default. Nonzero uses
  `rocblas_gemm_algo_solution_index` for the corresponding attention GEMM.
- The override is deliberately manual for now. Solution IDs are
  architecture/runtime/shape specific, and the 224/448 best choices are not the
  same. A cache format should come before enabling automatic selection.

Return-profile after manual attention solution overrides, with F32 attention
plus QKV, attention-output, MLP-up, and MLP-down custom kernels enabled:

| Shape | QK solution | AV solution | Attention | QKV | Attn out | MLP | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | `-110` | `-1161823143` | 18.306 ms | 29.168 ms | 11.926 ms | 62.036 ms | 125.785 ms |
| 448 | `-111` | `-451` | 182.779 ms | 124.883 ms | 44.268 ms | 259.879 ms | 630.614 ms |

Generation smoke after manual attention solution overrides:

| Shape | Output |
| --- | --- |
| 224 | `A long-shot view of a large white church. The church has two towers` |
| 448 | `A large building with two towers stands tall in the distance. The building has a` |

Interpretation:

- This is the largest attention-core win so far without changing numerics. It
  keeps QK, softmax, and AV in F32 and only changes rocBLAS' internal kernel
  choice.
- The 448 path is the main beneficiary: attention drops from the previous
  roughly 276 ms class to roughly 183 ms in the full return-profile path.
- The next practical step was to add a shape/ROCm/device keyed attention
  solution cache, similar in spirit to the MLP cache, before considering these
  indices as a reusable default. This is now implemented below.
- A custom tiled/MFMA attention kernel is still relevant, but the bar moved:
  it now has to beat the explicit rocBLAS solutions, not rocBLAS default `0`.

Attention rocBLAS solution cache:

- Added backend methods:
  - `LoadAttentionGemmSolutionCache`;
  - `BenchmarkAttentionGemmSolutions`;
  - `SaveAttentionGemmSolutionCache`.
- Added probe flags:
  - `--hip_attention_solution_bench N`;
  - `--hip_attention_solution_cache PATH`;
  - `--hip_attention_solution_cache_write 0|1`.
- Added generation CLI flag:
  - `--paligemma_vit_hip_attention_solution_cache PATH`.
- The cache key includes HIP runtime, rocBLAS version, device arch, compute
  units, model id, sequence length, model dim, attention heads, and qkv dim.
- The cache value stores separate QK and AV solution indices.
- Manual `--paligemma_vit_hip_attention_qk_solution` and
  `--paligemma_vit_hip_attention_av_solution` remain available for one-off
  overrides.

Generated cache:

```bash
build/paligemma2_vit_attention_solution_cache.txt
```

Cache entries generated on this machine:

| Shape | QK solution | AV solution |
| --- | ---: | ---: |
| 224 | `-111` | `-450` |
| 448 | `-111` | `-451` |

Commands:

```bash
scripts/build_paligemma2_vit_attention_solution_cache.sh
```

Return-profile after loading the attention cache, with F32 attention plus QKV,
attention-output, MLP-up, and MLP-down custom kernels enabled:

| Shape | QK solution | AV solution | Attention | QKV | Attn out | MLP | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | `-111` | `-450` | 20.263 ms | 28.655 ms | 12.190 ms | 62.865 ms | 128.898 ms |
| 448 | `-111` | `-451` | 177.695 ms | 110.349 ms | 39.868 ms | 222.875 ms | 568.867 ms |

Generation CLI smoke after loading the attention cache:

| Shape | Output |
| --- | --- |
| 224 | `A long-shot view of a large white church. The church has two towers` |
| 448 | `A large building with two towers stands tall in the distance. The building has a` |

Notes:

- Do not run multiple GPU profiles in parallel on this iGPU. They contend for
  the same device and produce inflated timings.
- The 224 AV candidates are close and noisy. The cache-generated `-450` entry
  is conservative from the automated solution bench; manual one-off sweeps can
  still test other indices when chasing small 224-only wins.
- The 448 cache entry is stable and material: it preserves F32 attention
  numerics while avoiding rocBLAS default `0` for the two expensive attention
  GEMMs.

Recommended-script baseline:

- Added `scripts/build_paligemma2_vit_attention_solution_cache.sh`.
  - Rebuilds the HIP backend probe by default.
  - Benchmarks 224 and 448 attention QK/AV rocBLAS solutions sequentially.
  - Writes `build/paligemma2_vit_attention_solution_cache.txt`.
- Added `scripts/profile_paligemma2_vit_hip_recommended.sh`.
  - Runs the current recommended HIP profile configuration sequentially.
  - Uses custom QKV, attention-output, MLP-up, MLP-down kernels.
  - Loads the attention solution cache, plus the 448 MLP cache when present.
- Added `scripts/smoke_paligemma2_vit_hip_recommended.sh`.
  - Runs deterministic generation smoke tests for 224 and 448 with the same
    recommended options.

Scripted profile command:

```bash
REBUILD=0 scripts/profile_paligemma2_vit_hip_recommended.sh
```

Scripted profile result:

| Shape | QK solution | AV solution | Attention | QKV | Attn out | MLP | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | `-111` | `-450` | 20.940 ms | 28.978 ms | 12.354 ms | 63.647 ms | 130.674 ms |
| 448 | `-111` | `-451` | 180.431 ms | 116.756 ms | 41.245 ms | 232.361 ms | 589.171 ms |

Scripted smoke command:

```bash
REBUILD=0 scripts/smoke_paligemma2_vit_hip_recommended.sh
```

Scripted smoke output:

| Shape | Output |
| --- | --- |
| 224 | `A long-shot view of a large white church. The church has two towers` |
| 448 | `A large building with two towers stands tall in the distance. The building has a` |

This scripted baseline is the target to beat for the next custom attention
kernel work. Future MFMA/tiled attention prototypes should compare against
these cached rocBLAS solution results, not the earlier default-solution
attention path.

Benchmark-only scalar tiled AV prototype:

- Added `--attention_av_tiled_profile`.
- This isolates the AV half of attention after QK and F32 softmax have already
  produced normalized score rows.
- The custom kernel is `AttentionAvTiledKernel<8, 16, 32>`:
  - one workgroup handles an 8-query-row x 16-output-dim tile for one head;
  - the K dimension is streamed in 32-key chunks;
  - score tiles and V tiles are staged in LDS;
  - accumulation stays F32;
  - output layout matches the current head-major AV buffer.
- This is intentionally a scalar tiled prototype. It tests whether simple LDS
  reuse is enough before investing in MFMA-specific attention code.

Command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_av_tiled_profile_only \
  --samples 20 \
  --warmup 5 \
  --iters 1
```

Measured result:

| Shape | Best rocBLAS AV | Tiled AV | Relative | max abs | RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.369 ms | 0.802 ms | 2.173 | 0.000000 | 0.000000 |
| 448 | 2.456 ms | 8.533 ms | 3.475 | 0.000000 | 0.000000 |

Interpretation:

- The scalar tiled AV kernel is numerically correct but not competitive.
- LDS reuse alone does not offset the loss of rocBLAS' matrix-instruction
  throughput and scheduling for this shape.
- Do not integrate this AV kernel into the backend.
- The next custom attention step should skip scalar tiling and go directly to a
  matrix-instruction path, or focus on fusing around rocBLAS while keeping the
  cached AV GEMM.

Benchmark-only BF16 WMMA AV prototype:

- Added `--attention_av_bf16_wmma_profile`.
- This keeps QK and the softmax reduction in F32, then compares three AV
  options:
  - selected-solution F32 rocBLAS AV, the current numeric baseline;
  - BF16 rocBLAS AV with BF16 scores/V and F32 accumulation;
  - local grouped-2D RDNA WMMA AV with BF16 scores/V and F32 accumulation.
- Added `SoftmaxRowsToBF16Kernel` so the benchmark can also test the more
  realistic direct path: raw F32 scores -> F32 softmax reduction -> BF16
  normalized scores in one kernel. This removes the expensive separate
  F32-score-to-BF16 quantization pass from the candidate path.
- The local WMMA AV kernel is a benchmark-only batched variant of the grouped
  2D WMMA GEMM macro-tile used for projections. It stages BF16 score and V
  tiles in LDS and writes F32 attention output.

Command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_av_bf16_wmma_profile_only \
  --samples 50 \
  --warmup 10 \
  --iters 1
```

Measured result:

| Shape | F32 softmax + F32 AV | Direct softmax-BF16 + WMMA AV | Separate quant + WMMA AV | Direct / F32 | BF16 vs F32 RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.410 ms | 0.305 ms | 0.372 ms | 0.744 | 0.000121 |
| 448 | 4.359 ms | 4.319 ms | 5.792 ms | 0.991 | 0.000119 |

Phase details from the same run:

| Shape | F32 softmax | F32 AV | Softmax to BF16 | Quant scores | Quant V | BF16 WMMA AV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.129 ms | 0.281 ms | 0.143 ms | 0.081 ms | 0.026 ms | 0.136 ms |
| 448 | 1.888 ms | 2.471 ms | 1.697 ms | 1.282 ms | 0.090 ms | 2.532 ms |

Interpretation:

- The matrix-instruction AV direction is much better than the scalar tiled AV
  prototype. The WMMA AV kernel itself is competitive with selected F32 rocBLAS
  AV, especially for 224.
- The separate F32-softmax-then-BF16-quantize path is not viable for 448. The
  score quantization pass alone costs about 1.28 ms/layer at 1024 tokens.
- Direct softmax-to-BF16 removes that quantization cost and makes 224 clearly
  faster for the isolated softmax+AV boundary. For 448 it only reaches parity:
  4.319 ms vs 4.359 ms in this run, which is inside normal run-to-run noise.
- This changes the precision boundary because normalized attention scores are
  stored as BF16 before AV. It is therefore not equivalent to the current F32
  attention path and must not be enabled by default without generation A/B
  guardrails.
- Current decision: keep this as a useful prototype and measurement tool. A
  backend candidate needs either a stronger 448 margin or a broader
  fused/matrix-instruction attention design that improves QK, softmax, and AV
  together while passing deterministic generation checks.

Probe-only AV BF16 WMMA backend guardrail:

- Added `PaliGemma2VitHipOptions::use_attention_av_bf16_wmma`.
- Added generation CLI flag:
  - `--paligemma_vit_hip_attention_av_bf16_wmma 1`
- Added standalone probe flag:
  - `--hip_attention_av_bf16_wmma 1`
- Added `scripts/check_paligemma2_attention_av_bf16_wmma_ab.sh`.
- The path is explicitly opt-in and disabled by default. It keeps F32 QK, then
  uses direct softmax-to-BF16 scores plus BF16 V and the local grouped-2D WMMA
  AV kernel.
- The QK BF16 and AV BF16 WMMA flags are treated as independent experiments;
  enabling both at once is rejected.

Return-profile comparison using the recommended custom QKV, attention-output,
MLP-up, MLP-down kernels and the attention solution cache:

| Shape | Attention mode | Attention | Device sum |
| --- | --- | ---: | ---: |
| 224 | F32 cached rocBLAS AV | 21.034 ms | 128.463 ms |
| 224 | AV BF16 WMMA | 14.767 ms | 121.129 ms |
| 448 | F32 cached rocBLAS AV | 177.663 ms | 591.779 ms |
| 448 | AV BF16 WMMA | 172.938 ms | 579.238 ms |

Default-image deterministic A/B:

```bash
KEEP_GOING=1 scripts/check_paligemma2_attention_av_bf16_wmma_ab.sh
```

Result:

- 224 default image, 3 prompts: pass.
- 448 default image, 3 prompts: pass.

Extended 448 guardrail using the existing QK-BF16 stress images:

```bash
KEEP_GOING=1 \
MODEL_224=/tmp/missing-paligemma2-224.sbs \
IMAGES_FILE=scripts/paligemma2_attention_qk_bf16_images.txt \
PROMPTS_FILE=scripts/paligemma2_attention_qk_bf16_prompts.txt \
scripts/check_paligemma2_attention_av_bf16_wmma_ab.sh
```

Result:

| Image | Prompt | F32 attention | AV BF16 WMMA |
| --- | --- | --- | --- |
| `paligemma/testdata/qk_bf16_sweep/dark_radial.ppm` | `Describe the image.` | `a close up of a circle pattern` | `a blue swirl` |

Interpretation:

- This is the same known-sensitive synthetic case that caught the QK BF16
  experiment, so the guardrail is doing useful work.
- AV BF16 WMMA is faster in the integrated return-profile, especially for 224,
  but it changes generation for a 448 stress case.
- Current decision: keep the code and script as an opt-in experiment and
  regression test, but do not include `--paligemma_vit_hip_attention_av_bf16_wmma`
  in recommended scripts or default backend options.
- The next attention direction should focus on preserving F32 score storage or
  making a larger fused QK/softmax/AV design that passes the same dark-radial
  generation guardrail.

F32 attention pack-to-BF16 optimization:

- Added `PaliGemma2VitHipOptions::use_attention_pack_bf16`.
- Added generation CLI flag:
  - `--paligemma_vit_hip_attention_pack_bf16 1`
- Added standalone probe flag:
  - `--hip_attention_pack_bf16 1`
- Added `scripts/check_paligemma2_attention_pack_bf16_ab.sh`.
- This keeps QK, softmax, and AV in F32. It does not store normalized attention
  scores as BF16. The only change is after AV:
  - old path: pack head-major F32 attention to row-major F32, then convert that
    row-major tensor to BF16 before `attn_out`;
  - new path: leave attention head-major after AV, then pack directly to
    row-major BF16 immediately before `attn_out`.
- During testing, the first implementation packed directly in the attention
  stage. That exposed nondeterministic `dark_radial.ppm` generation results
  unless `HIP_LAUNCH_BLOCKING=1` was set.
- The root issue was that the return path did not explicitly synchronize before
  the final device-to-host image-token copy. Added a single
  `hipDeviceSynchronize()` before the final `hipMemcpy` in
  `GenerateImageTokensDeviceNorm`. This is a once-per-image-token-return
  synchronization, not a per-layer synchronization.
- The same final synchronization was also added to the return-profile D2H path
  so the reported host copy timing is separated from outstanding device work.
- The final implementation keeps the fused pack-to-BF16 kernel in the
  `attn_out` stage, immediately before the following rocBLAS GEMM. This keeps
  the producer/consumer boundary close to the original F32-to-BF16 conversion.

Return-profile comparison from one paired run, using the same recommended
custom QKV, attention-output, MLP-up, MLP-down kernels and attention solution
cache:

| Shape | Attention mode | Attention | Attn out | Device sum |
| --- | --- | ---: | ---: | ---: |
| 224 | F32 cached rocBLAS attention | 20.989 ms | 12.723 ms | 132.378 ms |
| 224 | F32 attention + pack-to-BF16 | 19.612 ms | 11.858 ms | 125.501 ms |
| 448 | F32 cached rocBLAS attention | 181.382 ms | 43.296 ms | 608.536 ms |
| 448 | F32 attention + pack-to-BF16 | 176.879 ms | 43.126 ms | 606.819 ms |

Follow-up recommended-script profile also passed and reported
`attn=f32_pack_bf16`. The 448 full `device_sum` remains noisy because QKV/MLP
timings move between runs; treat this optimization as a small memory-bound
boundary cleanup, not a large attention breakthrough.

Generation guardrails:

```bash
KEEP_GOING=1 scripts/check_paligemma2_attention_pack_bf16_ab.sh

KEEP_GOING=1 \
MODEL_224=/tmp/missing-paligemma2-224.sbs \
IMAGES_FILE=scripts/paligemma2_attention_qk_bf16_images.txt \
PROMPTS_FILE=scripts/paligemma2_attention_qk_bf16_prompts.txt \
scripts/check_paligemma2_attention_pack_bf16_ab.sh
```

Result:

- 224 default image, 3 prompts: pass.
- 448 default image, 3 prompts: pass.
- 448 extended stress images, including `dark_radial.ppm`, 12 prompt/image
  cases: pass.
- After the final D2H synchronization fix, repeated `dark_radial.ppm` runs were
  stable for both baseline and pack-to-BF16 candidate.

Current decision:

- Add `--paligemma_vit_hip_attention_pack_bf16 1` to the recommended generation
  smoke script.
- Add `--hip_attention_pack_bf16 1` to the recommended profile script.
- This is now the preferred F32-semantics attention boundary optimization. Its
  benefit is modest, especially at 448, but it reduces an unnecessary global
  memory boundary without changing F32 QK/softmax/AV semantics.

Recommended attention phase profile:

- Added benchmark flag:
  - `--attention_recommended_phase_profile_only`
- This uses the current gfx1150 attention cache entries:
  - QK solution `-111`;
  - AV solution `-450` for 224 and `-451` for 448.
- It also uses the backend's F32-semantics pack-to-BF16 boundary, so the phases
  match the recommended path more closely than the older default-solution
  attention phase profile.

Command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_recommended_phase_profile_only \
  --samples 50 \
  --warmup 10 \
  --iters 1
```

Measured result:

| Shape | Split | QK | Softmax | AV | Pack BF16 | Sum | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.103 ms | 0.153 ms | 0.135 ms | 0.297 ms | 0.037 ms | 0.726 ms | 0.756 ms |
| 448 | 0.425 ms | 1.533 ms | 1.876 ms | 2.384 ms | 0.096 ms | 6.313 ms | 6.711 ms |

Interpretation:

- Pack-to-BF16 is no longer an important standalone attention cost. At 448 it is
  about 0.10 ms/layer.
- The next F32-safe attention targets are QK, softmax, and AV. For 448, the
  measured per-layer ordering is AV first, then softmax, then QK, with all three
  in the same rough range.
- A future custom attention kernel must beat the cached-solution rocBLAS
  baseline above, not the older default-solution profile.

Fixed-shape F32 softmax specialization:

- Added benchmark flag:
  - `--attention_softmax_fixed_profile_only`
- Added fixed-shape backend softmax kernels for PaliGemma2 attention rows:
  - 224px: `rows=256`, `threads=64`;
  - 448px: `rows=1024`, initially `threads=256`, later retuned to
    `threads=128`.
- The backend now routes normal F32 score softmax through `LaunchSoftmaxRows`.
  Unknown row lengths still fall back to the dynamic `SoftmaxRowsKernel`.
- This does not move the softmax/AV boundary. Scores are still fully normalized
  before AV. The specialization only exposes row length and block size as
  compile-time constants.
- For 448px, a later sequential retune showed `fixed128` was slightly faster
  than `fixed256` on this machine and passed deterministic generation guardrails,
  so the integrated path now uses `fixed128`.

Command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_softmax_fixed_profile_only \
  --samples 120 \
  --warmup 30 \
  --iters 1
```

Original measured result before the later 448px fixed128 retune:

| Shape | Dynamic | Original integrated fixed | Other fixed candidates | Error vs dynamic |
| --- | ---: | ---: | --- | ---: |
| 224 | `t64=0.138 ms` | `fixed64=0.129 ms` | `fixed128=0.136`, `fixed256=0.141` | `max_abs=0`, `rms=0` |
| 448 | `t256=1.886 ms` | `fixed256=1.857 ms` | `fixed64=1.923`, `fixed128=1.860`, `fixed512=1.911` | `max_abs=0`, `rms=0` |

Recommended phase profile after the fixed-shape softmax integration:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_recommended_phase_profile_only \
  --samples 120 \
  --warmup 30 \
  --iters 1
```

| Shape | Split | QK | Softmax | AV | Pack BF16 | Sum | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.100 ms | 0.147 ms | 0.125 ms | 0.292 ms | 0.038 ms | 0.702 ms | 0.740 ms |
| 448 | 0.500 ms | 1.895 ms | 1.861 ms | 2.294 ms | 0.097 ms | 6.647 ms | 6.734 ms |

Generation guardrails:

- Default 224/448 `scripts/check_paligemma2_attention_pack_bf16_ab.sh`: pass.
- Extended 448 stress set, including `dark_radial.ppm`: pass.

Decision:

- Keep fixed-shape softmax in the default F32 attention path. It is a small but
  low-risk specialization because it preserves the original softmax-before-AV
  contract and the integrated thread counts.
- This note was later superseded by the 448px fixed128 retune below, which ran
  the required generation A/B and extended stress guardrails.

Vectorized F32 Q/K/V split:

- Added benchmark flag:
  - `--attention_split_vectorized_profile_only`
- Added a bench-only `SplitQKVForAttentionFloat4Kernel` candidate.
- This is a pure layout/data-movement change:
  - input stays `[row, head, qkv, dim]`;
  - output stays head-major `[head, row, dim]`;
  - QK, softmax, and AV math are unchanged.

Command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_split_vectorized_profile_only \
  --samples 120 \
  --warmup 30 \
  --iters 1
```

Measured result:

| Shape | Scalar split | `float4` split | Relative | Error |
| --- | ---: | ---: | ---: | ---: |
| 224 | 0.100 ms | 0.097 ms | 0.970 | `max_abs=0`, `rms=0` |
| 448 | 0.413 ms | 0.397 ms | 0.961 | `max_abs=0`, `rms=0` |

Follow-up longer retests were mixed:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_split_vectorized_profile_only \
  --samples 160 \
  --warmup 40 \
  --iters 1
```

Examples:

- 448-only run: scalar `0.408 ms`, `float4` `0.420 ms`, relative `1.030`.
- Full 224/448 run:
  - 224: scalar `0.097 ms`, `float4` `0.098 ms`, relative `1.013`;
  - 448: scalar `0.411 ms`, `float4` `0.399 ms`, relative `0.971`.

Original decision:

- Do not integrate vectorized split into the default backend yet. It is
  bitwise-equivalent, but the measured delta is small and not stable enough to
  justify another default kernel boundary.
- Keep the bench-only candidate for future retesting under a more controlled
  benchmark harness.

Later update:

- The runtime backend now promotes vector4 split only for the 224px shape and
  keeps 448px on scalar split. See the later "F32-Safe Attention Split
  Vectorization" section for the guarded promotion.

Attention rocBLAS solution cache resweep:

- Rebuilt `build/paligemma2_vit_attention_solution_cache.txt` with a larger
  candidate cap and more samples:

```bash
REBUILD=0 \
MAX_SOLUTIONS=128 \
SAMPLES=160 \
WARMUP=40 \
ITERS=1 \
scripts/build_paligemma2_vit_attention_solution_cache.sh
```

Current generated cache entries:

| Shape | QK solution | AV solution |
| --- | ---: | ---: |
| 224 | `-110` | `-1161823143` |
| 448 | `-110` | `-451` |

Sweep details:

| Shape | Phase | Best observed | Previous cache | Note |
| --- | --- | ---: | ---: | --- |
| 224 | QK | `-110` at `0.145 ms` | `-111` | `-110` and `-111` were effectively tied, but `-110` had the lower median. |
| 224 | AV | `-1161823143` at `0.263 ms` | `-450` | 224 AV candidates are noisy; this run selected the default-like solution. |
| 448 | QK | `-110` at `1.713 ms` | `-111` | Repeated sweeps favored `-110` over `-111`. |
| 448 | AV | `-451` at `2.307 ms` | `-451` | `-450` and `-451` remain close; generated cache kept `-451`. |

Recommended phase profile after cache resweep:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_recommended_phase_profile_only \
  --samples 120 \
  --warmup 30 \
  --iters 1
```

| Shape | Split | QK | Softmax | AV | Pack BF16 | Sum | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.096 ms | 0.146 ms | 0.127 ms | 0.254 ms | 0.032 ms | 0.656 ms | 0.653 ms |
| 448 | 0.406 ms | 1.750 ms | 1.823 ms | 2.417 ms | 0.099 ms | 6.495 ms | 6.820 ms |

Short return-profile after loading the new cache:

```bash
REBUILD=0 SAMPLES=5 WARMUP=2 ITERS=1 \
scripts/profile_paligemma2_vit_hip_recommended.sh
```

| Shape | QK | AV | Attention | Device sum |
| --- | ---: | ---: | ---: | ---: |
| 224 | `-110` | `-1161823143` | 17.787 ms | 123.841 ms |
| 448 | `-110` | `-451` | 175.351 ms | 574.886 ms |

Generation guardrails:

- Recommended smoke: pass.
- Default `scripts/check_paligemma2_attention_pack_bf16_ab.sh`: pass.
- Extended 448 stress set, including `dark_radial.ppm`: pass.

Decision:

- Keep the regenerated cache. This is the lowest-risk attention improvement in
  this round because it changes only rocBLAS solution selection and preserves
  the F32 QK/softmax/AV contract.
- Treat `-450` vs `-451` for 448 AV as near-tied; do not overfit that choice
  without a more isolated benchmark harness.

Scalar tiled QK experiment:

- Added bench-only flag:
  - `--attention_qk_tiled_profile_only`
- Added `AttentionQkTiledKernel<8,16,32>`, which writes the same row-major
  `[heads, query, key]` score matrix consumed by the existing softmax kernels.
- The kernel stages a small Q tile and K tile in LDS and keeps accumulation in
  scalar F32. It does not use RDNA WMMA/MFMA instructions and is intentionally
  not connected to the runtime backend.

Bench command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_qk_tiled_profile_only \
  --samples 60 \
  --warmup 20 \
  --iters 1
```

| Shape | rocBLAS QK | Tiled QK | Relative | rocBLAS GFLOPS | Tiled GFLOPS | max abs | RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.162 ms | 1.329 ms | 8.190 | 930.3 | 113.6 | 0.000000 | 0.000000 |
| 448 | 1.746 ms | 20.975 ms | 12.010 | 1383.3 | 115.2 | 0.000000 | 0.000000 |

Decision:

- Do not integrate scalar tiled QK into `hip_probe`.
- The result is numerically correct but far slower than cached rocBLAS
  solution `-110`; the gap grows at 448px.
- A custom QK path only makes sense if it uses matrix instructions and/or fuses
  with subsequent softmax work. A standalone scalar HIP QK kernel is not a
  viable direction on this iGPU.

BF16 WMMA QK experiment:

- Added bench-only flag:
  - `--attention_qk_bf16_wmma_profile_only`
- Added `Bf16WmmaGemmF32Grouped2DBatchedTransBKernel`, a grouped-2D RDNA
  WMMA kernel for `scores = Q * K^T`.
- This reuses the existing grouped-2D WMMA tile schedule but changes the B-side
  load so K can stay in its natural head-major `[row, dim]` layout. The kernel
  reads K as a transposed operand and writes the standard row-major score
  matrix expected by the current F32 softmax.
- The benchmark uses deterministic non-BF16-exact inputs so the precision delta
  against F32 QK is visible; the earlier default fill pattern was too close to
  BF16-exact and hid that error.

Bench command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_qk_bf16_wmma_profile_only \
  --samples 80 \
  --warmup 20 \
  --iters 1
```

| Shape | F32 rocBLAS QK | BF16 rocBLAS QK | WMMA 2x4 | WMMA 4x4 | WMMA 4x4 vs F32 | BF16-vs-F32 max abs | BF16-vs-F32 RMS | WMMA-vs-BF16 max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.152 ms | 0.212 ms | 0.127 ms | 0.112 ms | 0.738 | 0.003304 | 0.000873 | 0.000001 |
| 448 | 1.680 ms | 2.374 ms | 1.899 ms | 1.574 ms | 0.937 | 0.003318 | 0.000873 | 0.000001 |

Decision:

- This is the first custom attention sub-kernel in this branch that beats the
  cached F32 rocBLAS QK baseline in the standalone benchmark.
- WMMA 4x4 is the better local schedule among the two measured variants:
  - 224: about 26% faster than cached F32 rocBLAS QK;
  - 448: about 6% faster than cached F32 rocBLAS QK.
- The local WMMA output matches BF16 rocBLAS QK to around `1e-6`, so the kernel
  implementation is consistent with the BF16 math path.
- Do not make this the default attention path yet. It has the same precision
  boundary as the earlier BF16 QK experiment, which already found a deterministic
  448 stress-case generation mismatch (`dark_radial.ppm`).
- Treat this as a strong opt-in integration candidate: useful for aggressive
  local acceleration, but guarded by the existing BF16-QK correctness caveat.

Runtime opt-in integration:

- Added HIP backend option:
  - `use_attention_qk_bf16_wmma`
- Added probe flag:
  - `--hip_attention_qk_bf16_wmma 0|1`
- Added generation CLI flag:
  - `--paligemma_vit_hip_attention_qk_bf16_wmma 0|1`
- Added `RunDeviceAttentionQkBf16Wmma`, which uses the local WMMA 4x4 QK
  kernel, then keeps the existing F32 softmax and cached rocBLAS F32 AV path.
- Added no-pack variants for both BF16-QK paths:
  - `RunDeviceAttentionQkBf16NoPack`
  - `RunDeviceAttentionQkBf16WmmaNoPack`
- These no-pack variants leave attention output in head-major F32 form, allowing
  the existing `attention_pack_bf16` path to pack heads and convert to BF16 once
  at the attn_out boundary. This removes the previous unfair extra pack step
  when comparing BF16/WMMA QK against the recommended F32-attention profile.
- The existing `--hip_attention_qk_bf16_check 1` now validates whichever QK
  BF16 implementation is selected:
  - no WMMA flag: rocBLAS BF16 QK;
  - `--hip_attention_qk_bf16_wmma 1`: local WMMA BF16 QK.
- Extended `scripts/check_paligemma2_attention_qk_bf16_ab.sh` with
  `ATTENTION_MODE=qk_bf16_wmma` so the same deterministic generation A/B
  guardrail can cover the local WMMA implementation.
- Extended the same script with `ATTENTION_PACK_BF16=1`, which enables
  `--paligemma_vit_hip_attention_pack_bf16 1` on both baseline and candidate
  runs for pack-path A/B checks.

Return-profile with the opt-in WMMA QK path:

```bash
./build/paligemma2_vit_hip_backend_probe \
  --hip_image_tokens_return_profile 1 \
  --hip_attention_qk_bf16_wmma 1 \
  --hip_qkv_wmma2d 1 \
  --hip_attn_out_wmma2d 1 \
  --hip_mlp_up_wmma2d 1 \
  --hip_mlp_down_wmma8 1
```

| Shape | Attention mode | Attention | Device sum | Note |
| --- | --- | ---: | ---: | --- |
| 224 | `qk_bf16_wmma` | 17.301 ms | 120.948 ms | Slightly faster than the current recommended F32-attention profile. |
| 448 | `qk_bf16_wmma` | 178.841 ms | 594.015 ms | Not better overall without pack composition. |

Fair return-profile after composing no-pack WMMA QK with `attention_pack_bf16`:

```bash
./build/paligemma2_vit_hip_backend_probe \
  --hip_image_tokens_return_profile 1 \
  --hip_attention_qk_bf16_wmma 1 \
  --hip_attention_pack_bf16 1 \
  --hip_qkv_wmma2d 1 \
  --hip_attn_out_wmma2d 1 \
  --hip_mlp_up_wmma2d 1 \
  --hip_mlp_down_wmma8 1
```

| Shape | Baseline attention | WMMA QK + pack attention | Attention delta | Baseline device sum | WMMA QK + pack device sum | Device delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 17.909 ms | 17.106 ms | 4.5% faster | 124.990 ms | 123.197 ms | 1.4% faster |
| 448 | 180.465 ms | 172.452 ms | 4.4% faster | 594.037 ms | 590.145 ms | 0.7% faster |

Notes:

- The baseline is the current recommended profile with F32 attention plus
  `attention_pack_bf16`.
- The 448 row includes the existing 448 MLP solution cache:
  `build/paligemma2_vit_mlp_solution_cache_448.txt`.
- This is now a real full-path win, but still a small one. Most remaining time
  is in QKV and MLP, so optimizing QK alone cannot move device sum by much.

Image-token A/B check vs F32 attention on `paligemma/testdata/image.ppm`:

| Shape | max abs | RMS |
| --- | ---: | ---: |
| 224 | 0.248995 | 0.005451 |
| 448 | 0.438875 | 0.004610 |

Generation guardrails:

```bash
ATTENTION_MODE=qk_bf16_wmma KEEP_GOING=1 \
scripts/check_paligemma2_attention_qk_bf16_ab.sh

ATTENTION_MODE=qk_bf16_wmma ATTENTION_PACK_BF16=1 KEEP_GOING=1 \
scripts/check_paligemma2_attention_qk_bf16_ab.sh

ATTENTION_MODE=qk_bf16_wmma KEEP_GOING=1 \
MODEL_224=/tmp/missing-paligemma2-224.sbs \
IMAGES_FILE=scripts/paligemma2_attention_qk_bf16_images.txt \
PROMPTS_FILE=scripts/paligemma2_attention_qk_bf16_prompts.txt \
scripts/check_paligemma2_attention_qk_bf16_ab.sh

ATTENTION_MODE=qk_bf16_wmma ATTENTION_PACK_BF16=1 KEEP_GOING=1 \
MODEL_224=/tmp/missing-paligemma2-224.sbs \
IMAGES_FILE=scripts/paligemma2_attention_qk_bf16_images.txt \
PROMPTS_FILE=scripts/paligemma2_attention_qk_bf16_prompts.txt \
scripts/check_paligemma2_attention_qk_bf16_ab.sh
```

- Default 224/448 image prompts: pass.
- Extended 448 stress set: pass in this run.
- Default 224/448 image prompts with `attention_pack_bf16`: pass.
- After the MLP-up 8x4 retune, the extended 448 stress set with
  `attention_pack_bf16` fails one known-sensitive case:
  - image: `paligemma/testdata/qk_bf16_sweep/dark_radial.ppm`
  - prompt: `Describe the image.`
  - F32 attention baseline: `a close up of a circle pattern`
  - `qk_bf16_wmma`: `a blue swirl`

Current decision:

- Keep `qk_bf16_wmma` as an aggressive opt-in, not a recommended default.
- The standalone QK kernel is positive, and the no-pack runtime integration can
  reduce the attention stage when composed with `attention_pack_bf16`.
- The precision boundary remains BF16 Q/K. The 8x4 MLP-up retune also showed
  that this path remains sensitive to unrelated-but-real accumulation-order
  changes elsewhere in the ViT stack. Keep it as an aggressive opt-in only.
- The next attention step should only proceed if it attacks a larger boundary:
  either fuse QK with softmax using matrix instructions, reduce QKV/MLP time, or
  find a way to move AV without paying extra score-storage precision cost.

Scalar fused QK+softmax experiment:

- Added bench-only flag:
  - `--attention_qk_softmax_fused_profile_only`
- Added `AttentionQkSoftmaxRowsKernel`, where one workgroup owns one
  `[head, query]` row. It computes scalar F32 Q.K for all keys, performs stable
  softmax in shared memory, and writes the normalized score row once.
- This directly tests whether removing the raw-score global write/read boundary
  between rocBLAS QK and softmax is enough to win before attempting a more
  complex matrix-instruction fused kernel.

Bench command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_qk_softmax_fused_profile_only \
  --samples 60 \
  --warmup 20 \
  --iters 1
```

| Shape | rocBLAS QK + fixed softmax | Fused scalar QK+softmax | Relative | max abs | RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.273 ms | 0.684 ms | 2.504 | 0.000000 | 0.000000 |
| 448 | 3.734 ms | 28.041 ms | 7.510 | 0.017960 | 0.000067 |

Decision:

- Do not integrate scalar fused QK+softmax into `hip_probe`.
- The 224 case proves the fused row-softmax schedule can match the current
  normalized score output when the reduction shape matches fixed softmax, but
  it is still much slower than rocBLAS QK plus the optimized softmax kernel.
- The 448 case is both much slower and shows small softmax-output drift from
  QK accumulation/order differences. It would require generation guardrails even
  if it were faster, which it is not.
- The next serious attention path should skip scalar QK entirely and either:
  - use RDNA matrix instructions for QK while keeping the F32 softmax contract;
  - or fuse softmax with AV, where avoiding the large normalized score matrix
    has a larger memory-traffic upside.

Deferred softmax-scale experiment:

- Added benchmark flag:
  - `--attention_deferred_softmax_scale_profile_only`
- Added runtime opt-in flags:
  - generation: `--paligemma_vit_hip_attention_defer_softmax_scale 1`
  - standalone probe: `--hip_attention_defer_softmax_scale 1`
- Added guardrail script:
  - `scripts/check_paligemma2_attention_defer_softmax_scale_ab.sh`
- This candidate keeps QK and AV in F32. The scheduling change is:
  - current recommended path: softmax writes normalized scores into the large
    `[heads, rows, rows]` score matrix, then AV consumes normalized scores;
  - deferred-scale path: softmax writes `exp(score - max)` and one `1/sum` per
    row, AV consumes unnormalized exp scores, then pack-to-BF16 multiplies the
    smaller `[heads, rows, qkv_dim]` attention output by `1/sum`.
- The goal was to remove one normalization pass over the large score matrix and
  replace it with one multiply over the much smaller attention-output matrix.

Bench result:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_deferred_softmax_scale_profile_only \
  --samples 80 \
  --warmup 20 \
  --iters 1
```

| Shape | Current total | Deferred total | Relative | Pack-output max abs | Pack-output RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.774 ms | 0.742 ms | 0.959 | 0.001953 | 0.000011 |
| 448 | 6.498 ms | 6.352 ms | 0.977 | 0.003906 | 0.000022 |

Guardrail result:

```bash
KEEP_GOING=1 scripts/check_paligemma2_attention_defer_softmax_scale_ab.sh

KEEP_GOING=1 \
MODEL_224=/tmp/missing-paligemma2-224.sbs \
IMAGES_FILE=scripts/paligemma2_attention_qk_bf16_images.txt \
PROMPTS_FILE=scripts/paligemma2_attention_qk_bf16_prompts.txt \
scripts/check_paligemma2_attention_defer_softmax_scale_ab.sh
```

- Default 224/448 image prompts: pass.
- Extended 448 stress set: fail on `dark_radial.ppm` with prompt
  `Describe the image.`
  - baseline `pack_bf16`: `a blue swirl`
  - deferred-scale: `a close up of a circle pattern`

Decision:

- Do not add deferred softmax-scale to the recommended scripts.
- Keep it as an explicit opt-in experiment because it is useful evidence: even
  mathematically equivalent F32 rescheduling can move decoder-visible tokens on
  stress inputs when the softmax normalization is associated after AV.
- For the next attention step, avoid moving softmax normalization across AV
  unless the new design has stronger numerical guardrails or intentionally
  accepts output drift.

WMMA8 down correctness check:

- Added probe flag `--hip_mlp_down_wmma8_check 0|1`.
- The check builds a real layer0 BF16 MLP hidden tensor from the test image,
  then runs the same resident `linear_1` weight through both rocBLAS and WMMA8.
  It compares the full F32 down-projection output before bias/residual.

| Model | rocBLAS ms | WMMA8 ms | max abs | RMS |
| --- | ---: | ---: | ---: | ---: |
| 224 | 3.073 | 1.704 | 0.000165 | 0.000009 |
| 448 | 8.601 | 2.934 | 0.000187 | 0.000009 |

Generation CLI smoke:

```bash
./build/gemma_paligemma2_vit_hip \
  --weights build/models/paligemma2-3b-mix-448-hf/paligemma2-3b-mix-448-sfp-from-hf.sbs \
  --image_file paligemma/testdata/image.ppm \
  --prompt "Describe the image." \
  --verbosity 0 \
  --max_generated_tokens 16 \
  --temperature 0 \
  --top_k 1 \
  --paligemma_vit_backend hip_probe \
  --paligemma_vit_hip_return_image_tokens 1 \
  --paligemma_vit_hip_mlp_down_wmma8 1 \
  --paligemma_vit_hip_mlp_solution_cache build/paligemma2_vit_mlp_solution_cache_448.txt
```

The 448 deterministic 16-token output matched the rocBLAS/cache run:
`A large building with two towers stands tall in the distance. The building has
a`.

The 224 deterministic 16-token output also matched the rocBLAS run:
`A long-shot view of a large white church. The church has two towers`.

Fused attention prototype:

| Kernel | Median ms | Interpretation |
| --- | ---: | --- |
| `vit_attn_fused_224_f32` | 9.085 | Too slow for 224px. |
| `vit_attn_fused_448_f32` | 101.583 | Far slower than rocBLAS QK+AV for one 448px ViT layer. |

Interpretation:

- rocBLAS strongly validates iGPU offload for the BF16 projection-heavy parts
  of PaliGemma2 ViT.
- Raw 448px BF16 projection GEMMs are in the 5-7 TFLOPS range on the iGPU.
- Attention is more nuanced:
  - QK is plausible on rocBLAS.
  - AV is not an obvious win as a plain rocBLAS GEMM.
  - Softmax is not included in the GEMM numbers.
- The first direct fused HIP attention kernel is intentionally naive and is not a
  viable integration path. A real fused kernel would need tiled Q/K/V reuse,
  wave-level reductions, and better memory layout before it can compete.
- A full ViT GPU backend should not just call rocBLAS for every operation and
  expect an automatic win. The likely path is:
  - keep image weights resident on GPU;
  - offload patch embedding, QKV, attention output, MLP up/down, and image head
    with rocBLAS first;
  - use a custom fused HIP kernel for per-head softmax/weighted-sum attention,
    or keep attention on CPU until a fused kernel proves faster;
  - copy only final image tokens back to CPU before decoder prefill.
- The 4 GiB `rocm-smi` VRAM number did not block this prototype. HIP reported
  the shared/global pool around 29.3 GiB, matching the earlier GTT/shared-memory
  analysis.

Negative results:

- PaliGemma ViT matrix-attention path was tested by enabling the existing
  `DotSoftmaxWeightedSumMatrix()` path. It first required `kOdd` padding for
  temporary Q/K/C matrices, then ran correctly but was slower:
  - default 32-token wall time regressed to about 7.5-7.8 s.
  - slow-core 32-token wall time regressed to about 6.5-6.6 s.
  The direct ViT attention path remains better for 224px on this CPU.
- Naive in-process phase affinity was tested by adding temporary
  `--prefill_cpus` and `--decode_cpus` flags that called `sched_setaffinity`
  for all existing threads before image/prefix prefill and before decode. The
  experiment was reverted because it regressed badly:
  - 32 default baseline: about 4.7 s wall, prefill about 1.86-1.87 s.
  - 32 with `prefill=4-11, decode=0-11`: about 10.9-11.3 s wall, prefill about
    8.05-8.34 s.
  - 128 default baseline: about 9.6-9.8 s wall, prefill about 1.91-1.93 s.
  - 128 with `prefill=4-11, decode=0-11`: about 16.7-18.6 s wall, prefill about
    8.55-9.62 s.
  Meaning: external `taskset -c 4-11` works because `ThreadingContext` and
  Highway worker pools are created after the process topology is already
  constrained. Switching affinity after startup leaves full-machine worker
  counts/topology in place, then squeezes those workers onto fewer CPUs, causing
  oversubscription. Any real phase-specific scheduler must create separate
  phase-aware contexts/pools or otherwise make topology construction match the
  intended CPU set.
- `--to_bf16 1` is slower than the SFP path. The extra memory traffic costs more
  than SFP decompression.
- `--map 0` is slower than the default mapped path.
- `--pin 1 --spin 1`, especially together with `taskset`, is slower on this
  machine and should not be used as the default benchmark mode.

## References

- PaliGemma2 model card: https://ai.google.dev/gemma/docs/paligemma/model-card-2
- PaliGemma2 paper: https://arxiv.org/abs/2412.03555
- AMD Ryzen AI 9 HX 370 product page: https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-9-hx-370.html
- ROCm Radeon/Ryzen compatibility: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html

## Next-Stage MLP-Down Grouped-N Retune

Detailed design document:

- `docs/paligemma2_vit_hip_next_stage_design.md`

Implemented the first next-stage experiment by extending the standalone
`--mlp_wmma_group_sweep_only` candidate set from `2/4/8` waves to
`2/4/6/8/12/16` waves per workgroup.

Result:

- Short 20-sample sweep kept `8` as the best clear candidate.
- Longer 80-sample sweep showed `12` about 1.3% faster than `8` on synthetic
  `mlp_down_224` and `mlp_down_448` buffers.
- A temporary backend promotion of `12` did not produce a robust real-tensor
  win on 448px: `12` measured 3.614 ms and the restored `8` measured 3.668 ms
  in the same session. That isolated ~1.5% difference is too small to justify
  changing the recommended runtime schedule.

Decision:

- Keep the runtime backend on grouped-N WMMA8.
- Keep the expanded sweep in the benchmark harness for future retuning.
- Move the next acceleration step away from simple MLP-down wave-count retuning;
  further MLP-down work needs a different kernel idea, not only wider N
  grouping.

## F32-Safe Attention Split Vectorization

Detailed design/update:

- `docs/paligemma2_vit_hip_next_stage_design.md`

Implemented a vectorized `float4` Q/K/V split kernel in the HIP backend and a
shape-gated launcher:

- 224px (`rows=256`) uses vector4 split when `qkv_dim` is divisible by 4.
- 448px (`rows=1024`) stays on the scalar split because longer samples showed
  the vectorized kernel was not stable there.

Standalone split microbench:

| Shape | Scalar | Vector4 | Relative | Error |
| --- | ---: | ---: | ---: | --- |
| 224, 80 samples | 0.103 ms | 0.097 ms | 0.940 | exact |
| 448, 80 samples | 0.436 ms | 0.382 ms | 0.875 | exact |
| 224, 120 samples | 0.099 ms | 0.094 ms | 0.944 | exact |
| 448, 120 samples | 0.429 ms | 0.469 ms | 1.094 | exact |

Validation:

- Build passed for the backend probe and generation binary.
- Recommended smoke passed for 224 and 448.
- `KEEP_GOING=1 scripts/check_paligemma2_attention_pack_bf16_ab.sh` passed.

Decision:

- Promote vector4 split only for 224px.
- Keep 448px unchanged.
- Treat this as a small F32-safe cleanup, not a major 448px attention win.

## 448px Fixed Softmax Thread Retune

Implemented a F32-safe softmax retune in the HIP backend:

- 224px remains `SoftmaxRowsFixedKernel<256, 64>`.
- 448px changes from `SoftmaxRowsFixedKernel<1024, 256>` to
  `SoftmaxRowsFixedKernel<1024, 128>`.

Sequential softmax sweep:

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

- Backend probe and generation binary rebuilt successfully.
- Recommended smoke passed for 224 and 448.
- Default `check_paligemma2_attention_pack_bf16_ab.sh` passed.
- Extended 448 stress A/B passed, including `dark_radial.ppm`.

Decision:

- Promote fixed128 for the 448px F32 softmax path.
- This is a small, shape-specialized cleanup and does not change the F32
  attention contract.

## Attention Solution Cache Resweep After Softmax Retune

Rebuilt the rocBLAS attention solution cache after the 448px softmax thread
retune:

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

Validation:

- Recommended smoke passed.
- Default `scripts/check_paligemma2_attention_pack_bf16_ab.sh` passed.
- Recommended return profile loaded the regenerated cache:
  - 224: `qk_solution=-111 av_solution=-1161823143`
  - 448: `qk_solution=-110 av_solution=-451`

Decision:

- Keep the regenerated cache.
- Treat the solution-index differences as near-ties. This route is now
  exhausted for meaningful speedups; further 448px attention gains need a data
  movement or kernel-boundary change, not more rocBLAS solution-index churn.

## Corrected Recommended Attention Phase Benchmark

Updated `--attention_recommended_phase_profile_only` so it matches the current
runtime backend instead of the older hardcoded attention path:

- 224px uses the promoted vector4 Q/K/V split.
- 448px keeps scalar split.
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

Interpretation:

- The corrected 448px per-layer attention baseline remains about 6-7 ms.
- Split and pack are now small enough that the next meaningful work must target
  QK, softmax, and AV together.

## Online Head-Major Attention Prototype

Added a bench-only structural prototype:

- Flag: `--attention_online_head_major_profile_only`
- Kernel: `AttentionOnlineHeadMajorKernel`

Design:

- Consumes the same head-major Q/K/V layout as the recommended path.
- One workgroup owns one `[head, query]` row.
- Computes QK, stable F32 softmax, and AV directly.
- Does not write the large `[heads, rows, rows]` score matrix to global memory.

Command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_online_head_major_profile_only \
  --samples 40 \
  --warmup 10 \
  --iters 1
```

Result:

| Shape | Baseline QK+softmax+AV | Online core | Relative | Max abs | RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.797 ms | 1.607 ms | 2.015x slower | 0.000001 | 0.000000 |
| 448 | 6.080 ms | 53.582 ms | 8.812x slower | 0.224916 | 0.000706 |

Decision:

- Do not promote the online head-major kernel.
- Avoid more one-query-row scalar fused kernels. Removing score writes is not
  enough if the kernel gives up rocBLAS matrix throughput and rereads V for each
  query row.
- The next scalar-only question is whether a multi-query tile can share enough
  K/V work across query rows to close the gap.

## Online Multi-Query Attention Prototype

Added a bench-only scalar online attention variant:

- Flag: `--attention_online_multiquery_profile_only`
- Kernel: `AttentionOnlineMultiQueryKernel<kQueryRows>`
- Tested query tiles: `mq2`, `mq4`, `mq8`

Design:

- Consumes the same head-major Q/K/V layout as the recommended path.
- One workgroup owns one head and multiple neighboring query rows.
- Computes QK, stable F32 softmax, and AV without writing the full score matrix
  to global memory.
- Reuses each loaded `K[key, :]` across the query rows during QK and each loaded
  `V[key, dim]` across the query rows during AV.

Command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_online_multiquery_profile_only \
  --samples 40 \
  --warmup 10 \
  --iters 1
```

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

- Do not promote the scalar online multi-query kernel.
- `mq4` proves the idea is directionally better for 448px than one-query online
  attention, but it is still far behind the current rocBLAS path.
- 224px does not benefit because the extra query rows increase register/LDS
  pressure more than they save memory traffic.
- The mq4/mq8 variants also produce max-abs outliers on the deterministic stress
  input, so this prototype is useful as evidence but not as a runtime candidate.

Next direction:

- Stop spending time on scalar online attention variants.
- For attention, the next viable prototype needs matrix-instruction throughput:
  a small MFMA/WMMA QK tile, an AV tile, or a hybrid path that keeps rocBLAS for
  QK/AV and only fuses data-movement-heavy softmax/packing pieces.

## QK BF16 WMMA Tile Retune

Extended the existing BF16 WMMA QK benchmark from two hardcoded candidates
(`2x4`, `4x4`) into a grouped-2D macro-tile sweep:

- `1x4`
- `1x8`
- `2x2`
- `2x4`
- `2x8`
- `4x2`
- `4x4`
- `4x8`
- `8x1`
- `8x2`
- `8x4`

Also changed the benchmark's F32 reference to use `RecommendedQkSolution(rows)`,
so 224px compares against cached solution `-111` and 448px compares against
cached solution `-110`.

Latest serial retune command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_qk_bf16_wmma_profile_only \
  --samples 60 \
  --warmup 20 \
  --iters 1
```

Key sweep result:

| Shape | F32 QK | BF16 rocBLAS QK | Best WMMA tile | Best WMMA QK | Relative to F32 | Existing 4x4 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 224 | 0.159 ms | 0.213 ms | 2x8 | 0.107 ms | 0.672x | 0.109 ms |
| 448 | 1.684 ms | 2.727 ms | 2x8 | 1.369 ms | 0.813x | 1.487 ms |

Implementation:

- Added a shared `LaunchAttentionQkBf16Wmma` helper in the HIP backend.
- Runtime tile choice is now unified:
  - rows `256`: `wmma2d2x8`
  - rows `1024`: `wmma2d2x8`
- Both packed and no-pack `qk_bf16_wmma` paths use that helper.
- Profile/validation output now reports the actual WMMA QK tile name.

Current return-profile comparison with the recommended non-attention kernels:

| Shape | Attention mode | Attention | Device sum |
| --- | --- | ---: | ---: |
| 224 | F32 attention + pack | 17.135 ms | 104.582 ms |
| 224 | QK BF16 WMMA 2x8 + pack | 15.771 ms | 104.686 ms |
| 448 | F32 attention + pack | 164.348 ms | 479.260 ms |
| 448 | QK BF16 WMMA 2x8 + pack | 163.828 ms | 496.382 ms |

Validation:

```bash
ATTENTION_MODE=qk_bf16_wmma ATTENTION_PACK_BF16=1 KEEP_GOING=1 \
scripts/check_paligemma2_attention_qk_bf16_ab.sh
```

- Default 224/448 generation smoke passed with QK BF16 WMMA still disabled.
- 224 image-token check with the current tile reported
  `qk_impl=wmma2d2x8`, `max_abs_vs_f32_attn=0.289477`,
  `rms_vs_f32_attn=0.005729`.
- 448 image-token check with the current tile reported
  `qk_impl=wmma2d2x8`, `max_abs_vs_f32_attn=0.476363`,
  `rms_vs_f32_attn=0.004517`.

Decision:

- Keep the `2x8` QK BF16 WMMA tile for the opt-in aggressive path.
- This improves the isolated QK GEMM and slightly reduces the attention slice,
  but the full image-token path does not show a reliable device-sum win in the
  current profile. AV, softmax, MLP, and run-to-run iGPU contention dominate the
  total.
- Do not make BF16 QK default. The tile retune does not add a new mismatch, but
  the BF16 QK precision boundary remains visible versus F32 attention.

## AV BF16 WMMA Tile Retune

Extended the AV BF16 WMMA benchmark into the same grouped-2D macro-tile sweep:

- `1x4`
- `1x8`
- `2x2`
- `2x4`
- `2x8`
- `4x2`
- `4x4`
- `4x8`
- `8x1`
- `8x2`
- `8x4`

This path keeps QK in F32 and changes only the softmax/AV boundary:

- raw F32 scores
- F32 softmax reduction
- normalized BF16 score storage
- BF16 V
- BF16 WMMA AV with F32 output

Command:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_av_bf16_wmma_profile_only \
  --samples 120 \
  --warmup 30 \
  --iters 1
```

Key sweep result:

| Shape | F32 softmax + F32 AV | BF16 rocBLAS direct | Existing 4x4 direct | Best tile | Best direct | Relative to F32 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 224 | 0.421 ms | 0.400 ms | 0.273 ms | 8x2 | 0.264 ms | 0.628x |
| 448 | 4.368 ms | 5.211 ms | 4.862 ms | 8x2 | 3.685 ms | 0.844x |

Implementation:

- Added benchmark helpers for AV BF16 WMMA tile sweeps.
- Changed the opt-in runtime AV BF16 WMMA launch from 4x4 to 8x2.
- Profile output now reports `attn=av_bf16_wmma2d8x2`.

Return-profile comparison without `attention_pack_bf16`:

| Shape | Attention mode | Attention | Device sum |
| --- | --- | ---: | ---: |
| 224 | F32 attention | 19.335 ms | 118.558 ms |
| 224 | AV BF16 WMMA 8x2 | 14.874 ms | 112.195 ms |
| 448 | F32 attention | 187.462 ms | 563.975 ms |
| 448 | AV BF16 WMMA 8x2 | 178.199 ms | 562.179 ms |

Validation:

```bash
KEEP_GOING=1 scripts/check_paligemma2_attention_av_bf16_wmma_ab.sh
```

- Default 224/448 generation A/B prompts passed.

Extended 448 stress guardrail:

```bash
KEEP_GOING=1 \
MODEL_224=/tmp/missing-paligemma2-224.sbs \
IMAGES_FILE=scripts/paligemma2_attention_qk_bf16_images.txt \
PROMPTS_FILE=scripts/paligemma2_attention_qk_bf16_prompts.txt \
scripts/check_paligemma2_attention_av_bf16_wmma_ab.sh
```

- Still fails the known-sensitive case:
  - image: `paligemma/testdata/qk_bf16_sweep/dark_radial.ppm`
  - prompt: `Describe the image.`
  - F32 attention: `a close up of a circle pattern`
  - AV BF16 WMMA: `a blue swirl`

Decision:

- Keep the 8x2 runtime tile for the opt-in `av_bf16_wmma` experiment.
- Do not add AV BF16 WMMA to recommended scripts or default options. The tile is
  better, but the precision boundary is still not safe for deterministic 448
  stress inputs.
- This result narrows the next safe attention direction: preserve F32
  normalized score/AV semantics, or fuse around the F32 path without storing
  probabilities as BF16.

## F32 Grouped Softmax/AV Retune

Retuned the safe F32-preserving grouped softmax/AV prototype. This path keeps
the numeric contract identical to the recommended F32 attention core:

- F32 QK
- F32 softmax probabilities
- F32 AV accumulation
- F32 V

The benchmark now sweeps `kGroups` for
`SoftmaxAvRowsGroupedKernel<rows, 72, kGroups>` and compares against the current
recommended QK solution, fixed softmax, and AV solution.

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

Interpretation:

- The grouped scalar kernel is safe but not fast enough.
- Increasing lane groups improves the 448 scalar path, but even the best tested
  case is still far behind rocBLAS AV.
- This confirms that simple scalar fusion around softmax/AV is exhausted. The
  next safe attention direction must preserve F32 semantics while keeping
  matrix-level throughput, or should avoid replacing rocBLAS AV.

Decision:

- Keep the sweep as a benchmark diagnostic.
- Do not integrate F32 grouped softmax/AV into the runtime backend.

## BF16 Attention Pack Vectorization

The next safe optimization targeted the recommended `attention_pack_bf16` path
instead of changing attention precision. The pack boundary remains:

- F32 QK
- F32 softmax
- F32 AV
- one F32-to-BF16 conversion while packing head-major attention output for
  `attn_out`

Implementation:

- Added `PackAttentionHeadsToBF16Float4Kernel`.
- Added runtime helper `LaunchPackAttentionHeadsToBF16`.
- Added bench flag `--attention_pack_vectorized_profile_only`.
- Updated recommended phase profiling to use the same helper as runtime.

Microbench:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_pack_vectorized_profile_only \
  --samples 120 \
  --warmup 30 \
  --iters 1
```

| Shape | Scalar pack | Float4 pack | Relative | Max abs | RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.068 ms | 0.037 ms | 0.541x | 0.000000 | 0.000000 |
| 448 | 0.304 ms | 0.196 ms | 0.644x | 0.000000 | 0.000000 |

Runtime/profile check:

```bash
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

- Promote the vectorized pack helper to the recommended runtime path.
- This is a default-safe optimization because it is output-identical to scalar
  pack and does not introduce a new BF16 attention boundary.
- The direct end-to-end gain is modest because pack is a small per-layer phase;
  QKV, AV, and MLP still dominate.

## F32 AV Dim4 Matrix-Tile Prototype

The next high-risk direction tested whether a hand-written F32 matrix-level AV
kernel could beat rocBLAS without changing attention precision. This is safer
than BF16 AV WMMA because the boundary remains:

- F32 QK
- F32 softmax scores
- F32 V
- F32 AV accumulation/output

Implementation:

- Added `AttentionAvTiledDim4Kernel`.
- Extended `--attention_av_tiled_profile_only` with a dim4 tile sweep.
- Added runtime/probe flag `--hip_attention_av_f32_dim4 1`.
- Added generation flag `--paligemma_vit_hip_attention_av_f32_dim4 1`.
- Runtime is shape-gated to 224px (`rows=256`, `qkv_dim=72`); 448px falls back
  to rocBLAS.

Microbench:

```bash
./build/paligemma2_vit_hip_bench \
  --attention_av_tiled_profile_only \
  --samples 80 \
  --warmup 20 \
  --iters 1
```

| Shape | rocBLAS AV | Old scalar tile | Best dim4 tile | Relative |
| --- | ---: | ---: | ---: | ---: |
| 224 | 0.287 ms | 0.488 ms | 0.272 ms (`bm32/bk16`) | 0.950x |
| 448 | 2.499 ms | 8.389 ms | 3.816 ms (`bm32/bk16`) | 1.527x |

Runtime check on 224:

| Mode | Attention | Device sum |
| --- | ---: | ---: |
| Recommended F32 pack | 17.978 ms | 113.987 ms |
| AV F32 dim4 + pack | 18.530 ms | 115.064 ms |

448 fallback profile with the flag enabled:

| Shape | Reported attention mode | Attention | Device sum |
| --- | --- | ---: | ---: |
| 448 | `f32_pack_bf16` | 186.586 ms | 574.069 ms |

Smoke:

- `build/gemma_paligemma2_vit_hip` with
  `--paligemma_vit_hip_attention_av_f32_dim4 1` generated the expected 224
  church description for `paligemma/testdata/image.ppm`.

Decision:

- Keep the F32 dim4 AV path as an explicit experiment.
- Do not make it recommended. The local AV microbench win at 224 disappears in
  the complete layer profile, and the 448 shape remains slower than rocBLAS.
- This narrows the next high-risk direction: pure hand-written scalar F32 AV is
  not enough. A useful attention breakthrough has to fuse QK/softmax/AV while
  preserving throughput, or the work should move back to BF16-matrix paths
  where RDNA WMMA is actually available.

## BF16 WMMA Output Fusion

Moved back to BF16 matrix paths after the F32 AV dim4 experiment failed to
improve full-layer runtime. This pass fuses elementwise work into local WMMA
store sites:

- QKV WMMA writes `acc + qkv_bias[col]` directly.
- MLP-up WMMA writes the post-bias/post-GELU BF16 hidden tensor directly.

The MLP-up fusion keeps the same BF16 precision contract as the previous
`AddBiasGeluToBF16Kernel`:

```text
BF16(F32_accum + bias) -> GELU -> BF16
```

Recommended profile:

```bash
REBUILD=0 SAMPLES=12 WARMUP=4 \
  scripts/profile_paligemma2_vit_hip_recommended.sh
```

| Shape | QKV | Attention | MLP total | Device sum |
| --- | ---: | ---: | ---: | ---: |
| 224 | 20.907 ms | 16.802 ms | 51.766 ms | 104.251 ms |
| 448 | 79.641 ms | 171.861 ms | 195.867 ms | 500.750 ms |

Previous nearby profile:

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

- Keep QKV bias fusion and MLP-up bias/GELU/BF16 fusion in the recommended WMMA
  path.
- This is materially better than the F32 AV dim4 direction because it removes
  large tensor passes while staying on hardware BF16 matrix instructions.
  Further work should prefer this pattern before attempting another scalar
  attention replacement.

## Attention-Output Residual Fusion

Extended WMMA-store fusion to the attention-output projection. The recommended
attn-out WMMA path now writes:

```text
(WMMA_acc + attn_out_bias[col]) + residual[row, col]
```

This removes the separate bias and residual kernels for `attn_out` while keeping
the old F32 addition order. The rocBLAS fallback still uses the existing
elementwise kernels.

MLP-down bias/residual fusion was tested and rejected. It removed the residual
kernel, but the added bias/residual reads inside the MLP-down WMMA store made
the down kernel slower enough to lose the benefit.

Recommended profile:

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

- Keep attention-output bias/residual fusion.
- Do not keep MLP-down bias/residual fusion.
- Future store fusion should be stage-specific. It is profitable for QKV,
  MLP-up, and attn-out, but not automatically profitable for narrower stores
  whose WMMA kernel is already close to its resource limit.

## LayerNorm BF16 Store Fusion

Fused the BF16 LayerNorm boundary used by the return/inference path. The old
path wrote a full F32 normalized activation and then launched `F32ToBF16Kernel`;
the new `LayerNormToBF16Kernel` keeps the same F32 statistics/math and rounds at
the final store:

```text
old: LayerNormKernel -> F32 staging -> F32ToBF16Kernel -> BF16 GEMM input
new: LayerNormToBF16Kernel -> BF16 GEMM input
```

Applied to:

- Per-layer LN0 before QKV.
- Per-layer LN1 before MLP-up.
- Final encoder norm before the image-head GEMM.
- BF16 return workspace allocation; the LN0/LN1 F32 staging buffers are no
  longer part of the hot workspace.

Recommended profile after the change, run twice:

| Shape | LN0 | LN1 | Final norm | Head | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 run 1 | 0.895 ms | 0.890 ms | 0.034 ms | 1.535 ms | 106.241 ms |
| 224 run 2 | 0.918 ms | 0.920 ms | 0.034 ms | 1.528 ms | 108.362 ms |
| 448 run 1 | 2.766 ms | 2.721 ms | 0.105 ms | 3.721 ms | 502.582 ms |
| 448 run 2 | 2.791 ms | 2.737 ms | 0.102 ms | 3.665 ms | 506.656 ms |

Previous nearby profile:

| Shape | LN0 | LN1 | Device sum |
| --- | ---: | ---: | ---: |
| 224 | 1.346 ms | 1.335 ms | 100.843 ms |
| 448 | 6.291 ms | 6.262 ms | 495.650 ms |

Validation:

```bash
bash scripts/build_paligemma2_vit_hip_backend_probe.sh
bash scripts/build_gemma_paligemma2_vit_hip.sh
KEEP_GOING=1 scripts/check_paligemma2_attention_pack_bf16_ab.sh
```

- 224/448 generation A/B prompts passed.

Conclusion:

- Local LayerNorm work improved and scratch memory decreased.
- End-to-end device time did not improve in the measured runs because large
  QKV/MLP timings moved upward and dominated the total.
- Treat this as a safe cleanup and launch/memory reduction, not as the next
  headline speedup. The roadmap should continue toward larger matrix-level
  kernels where milliseconds are concentrated.

## MLP-Up Shape-Specific WMMA Tile

Used the existing matrix-level benchmark to retune the recommended MLP-up WMMA
kernel instead of changing precision. The sweep found:

- `mlp_down_448`: current `wmma8` is still best.
- `qkv_224` and `attn_out_224`: current `wmma2d8x4` is still best.
- `mlp_up_224`: `wmma2d4x8` is slightly faster than `wmma2d8x4`.
- `mlp_up_448`: `wmma2d8x4` remains best.

Runtime selection now uses:

```text
rows=256  -> MLP-up WMMA 4x8
rows=1024 -> MLP-up WMMA 8x4
```

This keeps the same fused MLP-up boundary:

```text
F32 accumulate -> BF16(acc + bias) -> GELU -> BF16
```

Recommended profile:

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

- 224/448 generation A/B prompts passed.

Conclusion:

- Keep the shape-specific MLP-up tile.
- The measured win is small and mostly affects 224px, but it is a clean
  matrix-level improvement with no semantic change.
- The next larger opportunity is still attention or MLP matrix scheduling, not
  more scalar cleanup.

## Attention Softmax Float4 Pass

Added a fixed-shape vectorized softmax variant for the stable F32 attention
path. This keeps the current order:

```text
QK rocBLAS -> F32 softmax in score matrix -> AV rocBLAS -> pack to BF16
```

Only the in-place softmax row pass changes from scalar loads/stores to `float4`
loads/stores for the known PaliGemma2 row sizes:

```text
rows=256  -> SoftmaxRowsFixedFloat4Kernel<256, 64>
rows=1024 -> SoftmaxRowsFixedFloat4Kernel<1024, 128>
```

Standalone fixed-softmax benchmark:

| Shape | Scalar fixed | Float4 fixed |
| --- | ---: | ---: |
| 224 | 0.133 ms | 0.111 ms |
| 448 | 1.810 ms | 1.779 ms |

Recommended profile after runtime integration:

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

- 224/448 generation A/B prompts passed.

Conclusion:

- Keep the float4 softmax implementation.
- This is a local attention-stage improvement, not a full-profile breakthrough;
  the measured device sum was dominated by MLP timing noise in the same run.
- The remaining high-value attention work must attack QK/AV or a conservative
  softmax+AV fusion. Split/pack/softmax scalar cleanup is now mostly depleted.

## MLP-Down Wave-Count Runtime Probe

Added a runtime selector for the existing grouped-N MLP-down WMMA kernel:

```text
--hip_mlp_down_wmma_waves N
--paligemma_vit_hip_mlp_down_wmma_waves N
```

Supported values are `2`, `4`, `6`, `8`, `12`, and `16`; default remains `8`.
The selector is only active when the existing MLP-down WMMA flag is enabled.

Before promoting any deeper attention fusion, rechecked the current F32
softmax+AV grouped prototype:

| Shape | Recommended QK+softmax+AV | Best grouped softmax+AV |
| --- | ---: | ---: |
| 224 | 0.563 ms | 0.962 ms |
| 448 | 6.188 ms | 15.162 ms |

Conclusion: do not promote softmax+AV fusion. It is still much slower than
rocBLAS AV plus the optimized softmax kernel.

MLP-down full profile with wave-count variants:

| Shape | Waves | MLP-down | MLP total | Device sum |
| --- | ---: | ---: | ---: | ---: |
| 448 | 8 | 86.186 ms | 195.913 ms | 491.544 ms |
| 448 | 12 | 88.338 ms | 201.012 ms | 495.845 ms |
| 448 | 16 | 89.019 ms | 201.300 ms | 498.778 ms |
| 224 | 8 | 25.058 ms | 54.739 ms | 109.375 ms |
| 224 | 12 | 24.275 ms | 54.010 ms | 105.366 ms |

Default recommended profile after adding the selector:

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

- 224/448 generation A/B prompts passed.

Conclusion:

- Keep the wave-count selector as experimental tooling.
- Do not change the recommended default: `wmma8` remains best for 448, and 448
  is the target shape where image encoding dominates.
- Further MLP-down gains likely require a different memory schedule, not just a
  wider grouped-N macro-tile.

## MLP-Down Fused Epilogue Retry

Reimplemented MLP-down bias/residual fusion as a separate grouped-N WMMA
epilogue instead of changing the plain down-projection path. The fused store
preserves the old F32 addition order:

```text
value = down_acc + linear_1_bias[col]
value = value + residual[row, col]
```

New flags:

```text
--hip_mlp_down_fused_residual 0|1
--paligemma_vit_hip_mlp_down_fused_residual 0|1
```

Recommended scripts now enable the flag with the existing `wmma8` MLP-down path.

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

Conclusion:

- Keep the fused epilogue in the recommended scripts.
- This is not a large MLP breakthrough, but it is a defaultable memory-pass
  cleanup: two activation-wide elementwise passes disappear, while the fused
  WMMA store absorbs the bias/residual loads.
- The next larger gains still require attention QK/AV or a different MLP-down
  memory schedule rather than another simple epilogue fusion.

## MLP-Down Schedule Recheck And F32 Attention Triage

Rechecked MLP-down schedules after the fused epilogue. The parallel sweep was
used only to find candidates; the decisions below use serial reruns.

448 grouped-N serial:

| Waves | Time |
| ---: | ---: |
| 2 | 6.358 ms |
| 4 | 6.393 ms |
| 6 | 6.506 ms |
| 8 | 3.194 ms |
| 12 | 3.208 ms |
| 16 | 3.363 ms |

448 grouped-2D serial:

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

Conclusion: keep grouped-N `wmma8`. The best grouped-2D option is still slower
than the current grouped-N path, and grouped-M remains out of range.

Current F32 attention phase profile:

| Shape | Split | QK | Softmax | AV | Pack BF16 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.096 ms | 0.150 ms | 0.128 ms | 0.266 ms | 0.026 ms | 0.664 ms |
| 448 | 0.418 ms | 1.758 ms | 1.811 ms | 2.530 ms | 0.095 ms | 6.856 ms |

Existing F32 tiled prototypes remain negative:

| Prototype | 224 | 448 |
| --- | ---: | ---: |
| QK tiled `8x16x32` | 1.299 ms vs 0.149 ms rocBLAS | 22.356 ms vs 1.879 ms rocBLAS |
| AV dim4 best | 0.280 ms vs 0.252 ms rocBLAS | 3.818 ms vs 2.413 ms rocBLAS |

Implementation cleanup:

- Fixed `--attention_qk_tiled_profile_only` to use
  `RecommendedQkSolution(rows)`. It now reports 224px `-111` and 448px `-110`
  instead of hardcoding `-110` for both shapes.

Conclusion:

- Do not promote any current F32 tiled QK/AV kernel.
- Future attention work should focus on reducing softmax/AV memory traffic while
  preserving F32 semantics, or on a genuinely matrix-throughput-class kernel;
  scalar per-element tiling is not competitive with rocBLAS on this iGPU.

## Softmax Recompute And Direct-QKV Attention

Tested a softmax variant that avoids storing intermediate unnormalized exp
scores. It recomputes `exp(score - max)` for the final normalized store:

| Shape | Current float4 | Best recompute | Error |
| --- | ---: | ---: | --- |
| 224 | 0.110 ms | 0.112 ms | exact |
| 448 | 1.787 ms | 1.787 ms | exact |

Conclusion: do not promote recompute softmax. It is exact but not faster than
the current float4 fixed-shape kernel.

The better F32-preserving attention win is direct-QKV rocBLAS input. QK and AV
read directly from the interleaved QKV projection with `lda=heads*3*qkv_dim` and
`batch_stride=3*qkv_dim`, so the split kernel and Q/K/V scratch writes are
skipped. QK, softmax, AV, and pack-to-BF16 semantics stay unchanged.

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

New flags:

```text
--hip_attention_direct_qkv 0|1
--paligemma_vit_hip_attention_direct_qkv 0|1
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

Conclusion:

- Enable direct-QKV in the recommended scripts together with
  `attention_pack_bf16`.
- Keep it limited to the F32 attention path; it is not wired into QK-BF16,
  AV-BF16, AV-dim4, or deferred-softmax experiments.

## Profiling And Roofline Pass

The dedicated roofline note is
`docs/paligemma2_vit_hip_roofline_profile.md`.
The follow-up hardware-counter note is
`docs/paligemma2_vit_hip_hardware_counter_profile.md`.

Key result:

- Projection/MLP kernels have high arithmetic intensity and low implied
  bandwidth requirements; they are WMMA schedule/shape limited, not DRAM
  bandwidth limited.
- Custom QKV, attention-output, and MLP kernels are already faster than rocBLAS
  for these shapes.
- HIP occupancy estimates show the custom kernels are not occupancy-collapsed:
  WMMA 2D 8x4 reaches 100% thread occupancy, and MLP-down WMMA 12 reaches 93.8%
  while outperforming WMMA 8 on 448px.
- 448px MLP-down now uses `wmma12+residual` in the recommended scripts; 224px
  stays on the previous default because the total-profile win was not clear.

Clean 448px backend profile after the wave-count adjustment:

| Phase | Time |
| --- | ---: |
| QKV | 84.164 ms |
| Attention | 169.367 ms |
| Attention output | 30.884 ms |
| MLP | 191.717 ms |
| Device sum | 487.346 ms |

Direction:

- Continue matrix-kernel work only where it improves WMMA tiling, LDS/global
  staging, or fused epilogues.
- For larger attention gains, focus on a F32-fidelity attention path that
  reduces score-matrix traffic or fuses softmax/AV. The simple recompute
  softmax path remains rejected.

Hardware-counter refinement:

- `rocprofv3 --pmc` works on the local gfx1150 iGPU after installing
  `rocprofiler-sdk`, `hsa-amd-aqlprofile`, and related packages.
- `rocprof-compute` is installed but does not provide built-in analysis configs
  for gfx1150, so raw `rocprofv3` counter CSV is the working path.
- The real 448 backend shows softmax reading about 64 MiB and writing about
  64 MiB per dispatch with high memory-unit busy. This makes softmax/AV fusion
  or score-traffic reduction the highest-leverage attention direction.
- QKV and MLP-up show high occupancy but weak/moderate L2 hit, pointing toward
  load-layout/LDS-staging work rather than VRAM budget tuning.

Roadmap impact:

1. Make `rocprofv3` counter collection a promotion gate for future kernels:
   clean timing plus `FETCH_SIZE`, `WRITE_SIZE`, `L2CacheHit`, `MemUnitBusy`,
   and `OccupancyPercent`.
2. Move F32-fidelity attention traffic reduction ahead of more projection/MLP
   micro-optimization. The target is the score-matrix pipeline: QK writes the
   score matrix, softmax reads/writes it, and AV rereads it.
3. Keep rocBLAS QK/AV as the baseline until a custom attention path wins both
   timing and counter behavior. Reimplementing QK/AV without reducing score
   traffic is not a sufficient roadmap item.
4. Treat QKV, attention-output, and MLP-up as incremental WMMA schedule/LDS
   reuse work. Their counters point to load-layout and cache/reuse issues, not
   capacity or launch overhead as the primary blocker.
5. Promote 448 `wmma12+residual` from candidate to default-path decision while
   keeping 224 on the previous default.
