# PaliGemma2 ViT HIP Roofline Profile

Date: 2026-05-10

This is the HIP-event based profiling and roofline pass for the PaliGemma2 ViT
HIP backend on the local gfx1150 iGPU. It has since been followed by a
hardware-counter pass in
`docs/paligemma2_vit_hip_hardware_counter_profile.md`.

## Measurement Setup

Original tool state before installing profiler packages:

- `rocprofv3`, `rocprof`, `rocprofiler-compute`, and `amd-smi` are not installed.
- `rocm-smi` is available and reports the local device as STRIXEMU/gfx1150 with
  8 CUs.
- The benchmark therefore uses HIP event timings, `rocm-smi` device context,
  and `hipOccupancyMaxActiveBlocksPerMultiprocessor` for custom-kernel
  occupancy estimates.

Commands used for the clean pass:

```bash
REBUILD=0 SAMPLES=8 WARMUP=3 ITERS=1 \
  scripts/profile_paligemma2_vit_hip_recommended.sh

./build/paligemma2_vit_hip_bench \
  --attention_recommended_phase_profile_only \
  --samples 120 --warmup 30 --iters 1

./build/paligemma2_vit_hip_bench \
  --attention_softmax_fixed_profile_only \
  --samples 120 --warmup 30 --iters 1

./build/paligemma2_vit_hip_bench --kernel_occupancy_report_only
```

## Clean Backend Profile

Current recommended path before the MLP-down wave-count adjustment:

| Shape | QKV | Attention | Attn out | MLP | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 | 21.547 ms | 14.575 ms | 9.715 ms | 54.093 ms | 103.593 ms |
| 448 | 86.102 ms | 165.486 ms | 31.189 ms | 201.988 ms | 495.974 ms |

448 with `--hip_mlp_down_wmma_waves 12`:

| Shape | QKV | Attention | Attn out | MLP | Device sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 448 | 84.164 ms | 169.367 ms | 30.884 ms | 191.717 ms | 487.346 ms |

The 448 MLP improves by about 10.3 ms and the total device sum improves by about
8.6 ms despite unrelated attention timing noise. The 224 path did not show a
clear total win with `wmma12`, so the recommended scripts keep 224 on the
default `wmma8` and pass `wmma12` only for 448.

## Attention Breakdown

After aligning the benchmark helper with the backend's promoted float4 softmax:

| Shape | Split | Direct QK | Softmax | Direct AV | Pack BF16 | Direct total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 0.096 ms | 0.148 ms | 0.110 ms | 0.248 ms | 0.026 ms | 0.548 ms |
| 448 | 0.421 ms | 1.855 ms | 1.802 ms | 2.401 ms | 0.095 ms | 6.309 ms |

The direct-QKV path removes the split phase. QK is slightly less ideal because
it reads Q/K from interleaved QKV with a strided batch layout, but the removed
split traffic still wins overall.

Softmax variants:

| Shape | Best current | Recompute best | Conclusion |
| --- | ---: | ---: | --- |
| 224 | float4 0.110 ms | recompute 0.114 ms | Keep current float4 |
| 448 | float4 1.764 ms | recompute 1.773 ms | Keep current float4 |

Recompute saves theoretical score-matrix traffic, but the extra `expf` work and
row reduction structure erase the gain on this device.

## Matrix Kernel Throughput

Relevant 448 microbench medians:

| Stage | Current/candidate kernel | Time/layer | Throughput |
| --- | --- | ---: | ---: |
| QKV | WMMA 2D 8x4 | 2.965 ms | 2.75 TFLOP/s |
| Attn out | WMMA 2D 8x4 | 0.987 ms | 2.75 TFLOP/s |
| MLP up | WMMA 2D 8x4 | 3.942 ms | 2.58 TFLOP/s |
| MLP down | WMMA grouped-N 8 | 3.686 ms | 2.75 TFLOP/s |
| MLP down | WMMA grouped-N 12 | 3.391 ms | 2.99 TFLOP/s |

The custom WMMA kernels are already materially faster than rocBLAS for these
projection/MLP shapes. The next matrix-kernel gains are therefore not likely to
come from switching back to rocBLAS.

## Roofline Interpretation

448 per-layer arithmetic intensity estimates use minimum algorithmic bytes and
therefore are lower bounds for real traffic.

| Stage | FLOPs/layer | Min bytes | AI | Observed throughput | Min bandwidth implied |
| --- | ---: | ---: | ---: | ---: | ---: |
| QKV | 8.154 GF | 24.5 MB | 333 FLOP/B | 2.62 TFLOP/s | 7.9 GB/s |
| Attn out | 2.718 GF | 9.7 MB | 279 FLOP/B | 2.38 TFLOP/s | 8.5 GB/s |
| MLP up | 10.154 GF | 21.1 MB | 481 FLOP/B | 2.67 TFLOP/s | 5.5 GB/s |
| MLP down | 10.154 GF | 28.2 MB | 360 FLOP/B | 3.09 TFLOP/s | 8.6 GB/s |
| QK | 2.416 GF | 76.5 MB | 31.6 FLOP/B | 1.30 TFLOP/s | 41.3 GB/s |
| AV | 2.416 GF | 76.5 MB | 31.6 FLOP/B | 1.01 TFLOP/s | 31.9 GB/s |

Conclusions:

- QKV, attn-out, and MLP are high-arithmetic-intensity kernels. Their implied
  minimum bandwidth is tiny relative to what the iGPU memory system should
  sustain, so they are compute/schedule/WMMA-pipeline limited rather than DRAM
  bandwidth limited.
- QK and AV are lower-intensity because they materialize/read the score matrix.
  They are still not pure bandwidth copies: rocBLAS shape quality, low K=72,
  score-matrix traffic, and launch/solution choice all matter.
- Softmax has heavy score-matrix traffic. The 448 current float4 path touches on
  the order of 335 MB algorithmic traffic and takes about 1.8 ms, roughly
  186 GB/s effective algorithmic bandwidth plus `expf`/reduction overhead. Since
  recompute is not faster, the bottleneck is not only global memory bandwidth.

## Occupancy

HIP occupancy estimates for custom kernels:

| Kernel | Block threads | Active blocks/CU | Active waves/CU | Thread occupancy |
| --- | ---: | ---: | ---: | ---: |
| WMMA 2D 8x4 | 1024 | 2 | 64 | 100.0% |
| WMMA 2D 4x8 | 1024 | 2 | 64 | 100.0% |
| WMMA 2D 4x4 | 512 | 4 | 64 | 100.0% |
| MLP-down WMMA 8 | 256 | 8 | 64 | 100.0% |
| MLP-down WMMA 12 | 384 | 5 | 60 | 93.8% |
| MLP-down WMMA 16 | 512 | 4 | 64 | 100.0% |
| Softmax 448 float4 128 | 128 | 16 | 64 | 100.0% |

The important point is that occupancy is not collapsing. `wmma12` is slightly
below full thread occupancy but still wins for 448, so wave grouping and memory
reuse dominate the small occupancy difference.

## Direction

Immediate recommendation:

- Keep direct-QKV F32 attention, float4 softmax, BF16 pack, QKV/attn-out/MLP-up
  WMMA 2D 8x4.
- Use `wmma12+residual` for 448 MLP-down and keep 224 on the existing default.

Next development direction:

- Matrix kernels: focus on WMMA schedule quality, global-load shape, LDS reuse,
  and possibly double-buffering/vectorized staging. Do not prioritize VRAM/GTT
  budget changes for these high-AI kernels.
- Attention: the larger but riskier opportunity is a shape-specialized F32
  attention path that reduces score-matrix traffic or fuses softmax/AV without
  losing numerical fidelity. The previous simple recompute and online attempts
  do not justify promotion yet.
- rocBLAS: keep using cached solution indices for QK/AV while custom attention
  work continues. For projection/MLP shapes, current custom WMMA kernels are
  already faster than rocBLAS.
