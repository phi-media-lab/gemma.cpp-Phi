# PaliGemma2 ViT HIP Hardware Counter Profile

Date: 2026-05-10

This note records the first hardware-counter profiling pass after installing
the local ROCm profiling tools.

## Tool Setup

Installed packages:

```text
rocprofiler-sdk
rocprofiler-compute
rocprofiler
amd-smi-lib
hsa-amd-aqlprofile
```

Command availability:

```text
rocprofv3           ~/.local/bin/rocprofv3 -> /opt/rocm/bin/rocprofv3
rocprof             ~/.local/bin/rocprof   -> /opt/rocm/bin/rocprof
amd-smi             ~/.local/bin/amd-smi   -> /opt/rocm/bin/amd-smi
rocprof-compute     ~/.local/bin/rocprof-compute
rocprofiler-compute ~/.local/bin/rocprofiler-compute
```

`rocprof-compute` required Python dependencies that were not provided by the
system package, so they are isolated in:

```text
~/.local/share/rocprofiler-compute-venv
```

Important limitation: `rocprof-compute` 3.4.0 starts, but its built-in analysis
configuration set only includes `gfx908`, `gfx90a`, `gfx940`, `gfx941`,
`gfx942`, and `gfx950`. It reports that no supported architecture is present for
this local `gfx1150` iGPU. Therefore this pass uses `rocprofv3 --pmc` raw
hardware counter CSVs, not `rocprof-compute analyze`.

## Commands

Available counters were verified with:

```bash
rocprofv3 --list-avail --output-directory build/profiler_check \
  --output-format csv -- \
  ./build/paligemma2_vit_hip_bench --kernel_occupancy_report_only
```

The first attempt to collect many derived counters in one pass failed with
`Request exceeds the capabilities of the hardware to collect`. The working
method is one counter per run.

Main outputs:

```text
build/rocprofv3_paligemma2_hwc/
  backend448_summary.csv
  summary.csv
  trace_backend_448/
  trace_attention/
  trace_mlp/
  trace_mlp_up/
  backend448_<counter>/
  attention_<counter>/
  mlp_down_<counter>/
  mlp_up_<counter>/
```

Counters collected:

```text
FETCH_SIZE
WRITE_SIZE
L2CacheHit
MemUnitBusy
OccupancyPercent
Wavefronts
VALUInsts
LDSBankConflict
```

`FETCH_SIZE` and `WRITE_SIZE` are reported by ROCm in KiB.

## Real Backend 448 Profile

This table is from the real 448px backend probe with the current recommended
path:

```text
attention=f32_direct_qkv_pack_bf16
qkv=wmma2d8x4
attn_out=wmma2d8x4
mlp_up=wmma2d8x4
mlp_down=wmma12+residual
```

| Kernel group | Fetch KiB | Write KiB | L2 hit | MemUnitBusy | Occupancy | Median duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QKV WMMA 2D 8x4 | 68,861 | 13,402 | 31.2% | 75.8% | 98.1% | 3.59 ms |
| Attention QK rocBLAS | 15,129 | 64,218 | 57.9% | 82.2% | 50.1% | 1.76 ms |
| Softmax float4 | 65,536 | 64,000 | 83.3% | 92.9% | 98.1% | 1.75 ms |
| Attention AV rocBLAS | 73,318 | 4,261 | 67.1% | 69.3% | 67.9% | 2.43 ms |
| Attention pack BF16 | 4,608 | 1,673 | 78.6% | 85.3% | 87.8% | 0.09 ms |
| Attention out WMMA 2D 8x4 | 29,945 | 4,121 | 22.3% | 53.0% | 93.9% | 1.19 ms |
| MLP up WMMA 2D 8x4 + GELU | 87,733 | 8,412 | 46.5% | 78.2% | 98.9% | 4.33 ms |
| MLP down WMMA 12 + residual | 112,415 | 4,320 | 66.3% | 98.6% | 90.9% | 3.52 ms |

Notes:

- These are median per-dispatch counters from the profiled run. Counter runs
  perturb timing, so the duration column should be used for relative context,
  not as the clean timing baseline.
- Occupancy can exceed or differ from the HIP API estimate because this is a
  derived hardware-counter metric on gfx1150; use it as directional evidence.

## Interpretation

The previous event-timing roofline conclusion mostly holds, but the counters
make it sharper:

- **Softmax is score-matrix traffic dominated.** The kernel reads about 64 MiB
  and writes about 64 MiB per dispatch, with `MemUnitBusy` around 93% and high
  occupancy. This is the clearest attention target for score-traffic reduction
  or softmax/AV fusion.
- **QK is dominated by score writes and rocBLAS shape quality.** It writes about
  64 MiB per dispatch and has only about 50% counter-derived occupancy. This
  supports the earlier conclusion that low-K attention GEMM shape matters.
- **AV is mostly score/V read traffic.** It fetches about 73 MiB and writes very
  little. Its occupancy is higher than QK but still well below the custom
  softmax/WMMA kernels.
- **QKV and MLP-up are not external-bandwidth-only kernels, but they do have
  cache/memory-pipeline pressure.** L2 hit is low for QKV and moderate for
  MLP-up while occupancy is high. The next gains should come from WMMA schedule,
  tile shape, global-load layout, and LDS reuse rather than larger VRAM/GTT
  budgets.
- **MLP-down `wmma12+residual` is high memory-unit pressure but still wins.**
  The fused real backend kernel has about 91% occupancy, high L2 hit, and high
  `MemUnitBusy`. This validates the 448-specific switch to `wmma12+residual`;
  the remaining issue is repeated operand traffic, not launch overhead.

## Next Direction

For attention:

- Prioritize a F32-fidelity softmax/AV fusion or tiled attention path that
  reduces the 128 MiB score read/write softmax traffic and the AV score reread.
- Keep cached rocBLAS QK/AV as the correctness/performance baseline until a
  custom kernel beats both timings and counters.

For MLP/projection:

- Keep `wmma12+residual` for 448 MLP-down.
- Investigate better global-load/LDS staging for QKV and MLP-up, where the
  hardware counters show low or moderate L2 hit despite high occupancy.
- Treat memory budget changes as low priority for these kernels; the bottleneck
  is traffic shape and reuse, not capacity.

## Roadmap Impact

The hardware counters do not invalidate the previous roadmap, but they change
the priority order.

### Priority 1: Make Counter Profiling A Gate

Timing alone is no longer enough for promotion decisions. Every meaningful
attention/MLP/projection kernel change should be checked with:

```text
clean HIP-event timing
rocprofv3 --pmc FETCH_SIZE
rocprofv3 --pmc WRITE_SIZE
rocprofv3 --pmc L2CacheHit
rocprofv3 --pmc MemUnitBusy
rocprofv3 --pmc OccupancyPercent
```

Because `rocprof-compute` does not support local gfx1150 analysis configs, the
working profiling path is `rocprofv3` raw counter CSV plus a local summary
parser.

### Priority 2: F32-Fidelity Attention Traffic Reduction

The main structural bottleneck is the score-matrix pipeline:

```text
QK writes score matrix
softmax reads and writes score matrix
AV rereads score matrix
```

Counter evidence:

- QK writes about 64 MiB per dispatch.
- Softmax reads about 64 MiB and writes about 64 MiB per dispatch.
- AV fetches about 73 MiB per dispatch.
- Softmax has high occupancy and high memory-unit busy, so simple thread-count
  tuning is unlikely to unlock a large win.

Roadmap implication: the next high-upside work should be a F32-preserving
tiled/fused attention path that reduces score traffic, especially softmax/AV
fusion or an online/tiled attention flow. A custom kernel that merely reproduces
rocBLAS QK and AV while still materializing the full score matrix is not enough.

### Priority 3: Keep rocBLAS QK/AV As The Baseline

The counters show QK has weak occupancy and large score writes, but previous
custom scalar/tiled QK and AV prototypes were slower than rocBLAS. Therefore the
baseline remains:

```text
direct-QKV rocBLAS QK
float4 F32 softmax
rocBLAS AV
BF16 pack
```

Any replacement must beat both:

- clean timing; and
- score traffic / memory-unit counter behavior.

### Priority 4: Matrix Kernel Work Becomes Incremental

QKV, attention-output, and MLP-up have high occupancy but low/moderate L2 hit.
That means further work should focus on:

- global-load layout;
- LDS staging and reuse;
- WMMA macro-tile shape;
- fused epilogues.

This is still useful, but the expected gain is incremental compared with
removing attention score traffic.

### Priority 5: Promote 448 `wmma12+residual`

The counter pass supports the earlier timing decision:

```text
448 MLP-down: wmma12+residual
224 MLP-down: existing default path
```

`wmma12+residual` has slightly lower occupancy than the theoretical maximum but
wins in the real 448 path. This is now a default-path decision, not just a
candidate experiment.
