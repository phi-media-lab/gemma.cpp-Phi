# gemma.cpp-Phi：PaliGemma2 ViT HIP/gfx1150 分支说明

[English README](README.md)

这个分支是基于 [google/gemma.cpp](https://github.com/google/gemma.cpp) 的实验性 fork，重点是为本机 AMD ROCm/HIP 环境调研并实现 PaliGemma2 ViT 图像编码器的专用加速。

当前分支：

```text
paligemma2-vit-hip-gfx1150
```

开发和验证目标机器：

```text
AMD Ryzen AI 9 HX PRO 370 / Radeon 890M iGPU
ROCm 7.2.1
HIP arch gfx1150
Ubuntu 24.04
```

这个分支不是上游可直接合入的通用 GPU backend。它的目标是围绕 PaliGemma2 ViT 的固定形状，验证更激进的 HIP/RDNA 专用 kernel、profiling 方法和后续优化路线，同时保留原始 CPU 路径作为正确性基线。

## 快速入口

如果只是想确认当前分支是否能跑通，建议从这个入口开始：

```bash
REBUILD=0 MAX_GENERATED_TOKENS=8 \
  scripts/smoke_paligemma2_vit_hip_recommended.sh
```

如果想看当前推荐路径的阶段耗时：

```bash
REBUILD=0 SAMPLES=8 WARMUP=3 ITERS=1 \
  scripts/profile_paligemma2_vit_hip_recommended.sh
```

如果只做 HIP microbenchmark，不跑完整生成链路：

```bash
scripts/build_paligemma2_vit_hip_bench.sh

./build/paligemma2_vit_hip_bench \
  --attention_recommended_phase_profile_only \
  --samples 120 --warmup 30 --iters 1
```

## 这个分支新增了什么

- 在 PaliGemma2 image-token 生成边界接入了实验性 HIP backend。
- 增加了 resident-GPU/probe 路径，用来验证 ViT 各阶段。
- 针对 RDNA/gfx1150 写了多组 WMMA kernel：
  - QKV projection；
  - attention output projection；
  - MLP-up projection；
  - MLP-down projection，并融合 residual。
- 增加了 F32 attention 路径：
  - direct-QKV rocBLAS QK/AV 输入；
  - float4 F32 softmax；
  - BF16 pack 回 decoder 所需 image tokens。
- 增加了 PaliGemma2 ViT 形状专用的 HIP benchmark 和 profiling harness。
- 增加了 `rocprofv3` 硬件 counter profiling 工作流。
- 增加了模型转换、构建、smoke test、A/B check 和 profiling 脚本。
- 在 `docs/` 下沉淀了设计、profiling 和 roadmap 文档。

## 当前推荐路径

当前 448px PaliGemma2 ViT HIP 推荐路径是：

```text
attention=f32_direct_qkv_pack_bf16
qkv=wmma2d8x4
attn_out=wmma2d8x4
mlp_up=wmma2d8x4
mlp_down=wmma12+residual
```

224px 的 MLP-down 仍使用之前的默认 `wmma8+residual`。448px 推荐使用 `wmma12+residual`，完整生成路径的参数是：

```text
--paligemma_vit_hip_mlp_down_wmma_waves 12
```

probe/benchmark 工具中对应的短参数是：

```text
--hip_mlp_down_wmma_waves 12
```

完整生成链路当前稳定使用的核心参数：

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

## 当前进度

已经完成并验证的内容：

- 支持从 Hugging Face PaliGemma2 224px/448px 模型转换出本地 `.sbs`。
- 实现了实验性 ViT HIP backend 和 probe 可执行文件。
- 将主要 projection/MLP 阶段迁移到 device-resident HIP kernel。
- 实现 direct-QKV F32 attention，避免旧路径里拆 Q/K/V 的额外 scratch。
- 实现 float4 F32 softmax。
- 将 448px `wmma12+residual` MLP-down 纳入推荐脚本。
- 增加 QK-BF16 和 attention A/B 的压力图测试。
- 增加 HIP event timing、roofline 估算和硬件 counter profiling 文档。

最近一轮干净的 448px backend profile：

| 阶段 | 耗时 |
| --- | ---: |
| QKV | 84.164 ms |
| Attention | 169.367 ms |
| Attention output | 30.884 ms |
| MLP | 191.717 ms |
| Device sum | 487.346 ms |

真实 448px backend 的 counter 结论：

| Kernel group | Fetch KiB | Write KiB | L2 hit | MemUnitBusy | Occupancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| QKV WMMA 2D 8x4 | 68,861 | 13,402 | 31.2% | 75.8% | 98.1% |
| Attention QK rocBLAS | 15,129 | 64,218 | 57.9% | 82.2% | 50.1% |
| Softmax float4 | 65,536 | 64,000 | 83.3% | 92.9% | 98.1% |
| Attention AV rocBLAS | 73,318 | 4,261 | 67.1% | 69.3% | 67.9% |
| MLP up WMMA 2D 8x4 + GELU | 87,733 | 8,412 | 46.5% | 78.2% | 98.9% |
| MLP down WMMA 12 + residual | 112,415 | 4,320 | 66.3% | 98.6% | 90.9% |

阶段性结论：当前最大的剩余收益点不是继续简单替换 rocBLAS GEMM，而是减少 F32 attention score matrix 的读写流量。现在的 attention 路径仍然是 QK 写 score matrix，softmax 读写 score matrix，AV 再读 score matrix。

## 构建

构建 HIP probe：

```bash
scripts/build_paligemma2_vit_hip_backend_probe.sh
```

构建带 HIP image-token backend 的生成二进制：

```bash
scripts/build_gemma_paligemma2_vit_hip.sh
```

构建 HIP microbenchmark：

```bash
scripts/build_paligemma2_vit_hip_bench.sh
```

## 模型文件

模型权重不提交到仓库。当前脚本默认从 `build/models/` 下读取转换后的 `.sbs`，例如：

```text
build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs
build/models/paligemma2-3b-mix-448-hf/paligemma2-3b-mix-448-sfp-from-hf.sbs
```

HF 到 SBS 的转换入口：

```bash
scripts/convert_paligemma2_hf_to_sbs.sh
```

如果环境里已经有兼容的 SBS artifacts，也可以参考：

```bash
scripts/download_paligemma2_sbs.sh
```

## Profiling

这个分支目前使用两层 profiling：

- HIP event timing：用于干净的阶段耗时和 microbenchmark 对比。
- `rocprofv3 --pmc`：用于 gfx1150 raw hardware counters。

本机可以安装 `rocprof-compute`，但它内置的 analysis config 当前不支持 `gfx1150`，所以真正可用的硬件 counter 路径是 raw `rocprofv3` CSV。

常用命令：

```bash
./build/paligemma2_vit_hip_bench --kernel_occupancy_report_only

rocprofv3 --list-avail --output-directory build/profiler_check \
  --output-format csv -- \
  ./build/paligemma2_vit_hip_bench --kernel_occupancy_report_only
```

相关文档：

- [docs/paligemma2_vit_hip_roofline_profile.md](docs/paligemma2_vit_hip_roofline_profile.md)
- [docs/paligemma2_vit_hip_hardware_counter_profile.md](docs/paligemma2_vit_hip_hardware_counter_profile.md)

## Roadmap

### 1. 把硬件 counter profiling 变成晋级门槛

后续 kernel 只有同时满足下面条件，才应该进入推荐路径：

```text
干净的 HIP-event timing
rocprofv3 --pmc FETCH_SIZE
rocprofv3 --pmc WRITE_SIZE
rocprofv3 --pmc L2CacheHit
rocprofv3 --pmc MemUnitBusy
rocprofv3 --pmc OccupancyPercent
```

### 2. 优先减少 F32 attention score traffic

当前最大瓶颈链路是：

```text
QK 写 score matrix
softmax 读写 score matrix
AV 再读 score matrix
```

下一阶段 attention kernel 应该在保持 F32 保真的前提下降低这部分内存流量。单纯写一个自定义 QK/AV kernel 意义不大，除非它同时在耗时和 counter 上超过 rocBLAS。

可能方向：

- fused softmax/AV；
- tiled online attention；
- PaliGemma2 448 shape-specialized F32 attention；
- direct-QKV layout-aware attention。

### 3. 保留 rocBLAS QK/AV 作为基线

目前已有的 custom scalar/tiled QK/AV prototype 没有超过 rocBLAS。后续应继续把 cached rocBLAS QK/AV 作为正确性和性能基线，直到 custom attention 路径能真正改变 memory traffic profile。

### 4. 继续做 WMMA 增量优化

Projection 和 MLP 的 counter 显示 occupancy 已经较高，但 L2 hit 仍然偏弱或中等。后续可以继续做：

- global-load layout；
- LDS staging；
- tile reuse；
- fused epilogue；
- 只有在 counter 和 timing 同时改善时才引入 double-buffering。

### 5. 保留 448px `wmma12+residual`

448px MLP-down 的 `wmma12+residual` 已经是当前推荐默认路径。224px 继续保持旧默认，直到 total-profile 显示明确收益。

## 文档索引

分支相关文档：

- [docs/paligemma2_acceleration_plan.md](docs/paligemma2_acceleration_plan.md)
- [docs/paligemma2_vit_hip_next_stage_design.md](docs/paligemma2_vit_hip_next_stage_design.md)
- [docs/paligemma2_vit_hip_roofline_profile.md](docs/paligemma2_vit_hip_roofline_profile.md)
- [docs/paligemma2_vit_hip_hardware_counter_profile.md](docs/paligemma2_vit_hip_hardware_counter_profile.md)

上游 gemma.cpp 的 CPU runtime、文件格式和通用模型支持文档仍然适用：

- [DEVELOPERS.md](DEVELOPERS.md)
- [API_SERVER_README.md](API_SERVER_README.md)
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)

## 上游关系

上游项目是 `google/gemma.cpp`。本分支是面向 PaliGemma2 ViT HIP/gfx1150 加速的 fork-specific research branch。在 backend 泛化、构建系统清理、profiling gate 自动化之前，应把它视为实验性实现。
