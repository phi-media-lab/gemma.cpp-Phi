// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef THIRD_PARTY_GEMMA_CPP_EXPERIMENTAL_PALIGEMMA2_VIT_HIP_BACKEND_H_
#define THIRD_PARTY_GEMMA_CPP_EXPERIMENTAL_PALIGEMMA2_VIT_HIP_BACKEND_H_

#include <stddef.h>

#include <memory>
#include <string>

#include "gemma/gemma_args.h"

namespace gcpp {
struct ModelConfig;
struct WeightsPtrs;

namespace experimental {

// Standalone HIP/rocBLAS backend for the PaliGemma2 ViT image encoder.
//
// This is intentionally not a general GPU backend for gemma.cpp. It is scoped to
// the fixed PaliGemma2 ViT shapes so we can measure and replace one image-stage
// tensor at a time while preserving the CPU decoder path.
struct PaliGemma2VitHipTiming {
  float best_ms = 0.0f;
  float median_ms = 0.0f;
  double gflops = 0.0;
};

struct PaliGemma2VitHipOptions {
  // Timing controls used for HIP event measurements. The same controls are used
  // by the projection schedule and patch-embedding validation so numbers remain
  // comparable across experiments.
  int samples = 1;
  int warmup = 1;
  int iters = 1;
  bool verbose = true;

  // Run the resident-weight projection schedule after upload. This benchmarks
  // GEMM throughput with real weights, but it does not yet produce inference
  // results.
  bool benchmark_projection_schedule = true;

  // Compute the real patch embedding on HIP and compare it to a CPU scalar
  // reference. This allocates a small F32 shadow copy of the patch embedding
  // weights in addition to the resident BF16 copy.
  bool validate_patch_embedding = false;

  // Validate the first ViT layer's QKV projection using a host-computed
  // layernorm input. This compares F32-shadow and BF16 activation routes, but
  // still falls back to CPU for inference results.
  bool validate_layer0_qkv = false;

  // Validate the first ViT layer's MLP up/GELU/down projection route. This is
  // currently a projection precision boundary using a layernormed patch stream;
  // full block validation waits until attention/residual movement is device
  // resident.
  bool validate_layer0_mlp = false;

  // Validate the first ViT layer's attention-output projection and residual.
  // QKV/softmax/weighted-sum are prepared deterministically for this boundary;
  // the measured HIP work is attn_out_w plus residual add.
  bool validate_layer0_attn_out = false;

  // Validate the first complete ViT layer output by chaining QKV, attention
  // output residual, layer_norm_1, MLP, and final residual. Attention softmax is
  // still a host reference in this stage.
  bool validate_layer0_block = false;

  // Validate a device-side attention core for layer0. QKV still comes from the
  // existing HIP projection path; QK, softmax, and AV run on HIP.
  bool validate_layer0_device_attention = false;

  // Validate a complete layer0 path that uses device-side attention. This is
  // kept as the original flag name and currently aliases the device-norm/input
  // path below, because that path is now the most complete layer0 device graph.
  bool validate_layer0_block_device_attention = false;

  // Validate a complete layer0 path that keeps patch embedding, layer_norm_0,
  // device attention, and layer_norm_1 on HIP. Host tensors are only
  // correctness oracles in this path.
  bool validate_layer0_block_device_norm = false;

  // Validate a two-layer ViT prefix using the reusable HIP layer runner. This
  // keeps patch embedding and two complete transformer layers on device, then
  // compares the BF16 route against the F32-shadow route.
  bool validate_layer_prefix2_device_norm = false;

  // Validate the full ViT transformer stack using the same reusable HIP layer
  // runner. This stops after the last transformer layer; final encoder norm,
  // pooling, and image projection head are intentionally left for the next
  // integration boundary.
  bool validate_layer_stack_device_norm = false;

  // Validate decoder-visible image tokens by extending the full-stack path with
  // final encoder norm and img_head projection. The current implementation is
  // scoped to PaliGemma2's unpooled ViT path; GEMMA_VLM pooling/RMSNorm remains
  // a separate boundary.
  bool validate_image_tokens_device_norm = false;

  // Explicit opt-in to return HIP-produced BF16-route image tokens from the
  // real GenerateImageTokens hook. Unlike the validation path above, this runs
  // only the candidate BF16 route so it can be used for latency-oriented A/B
  // generation tests.
  bool return_image_tokens_device_norm = false;

  // Probe-only experimental attention-core replacement. When enabled, the BF16
  // ViT layer runner converts only Q/K to BF16 for the attention QK GEMM while
  // keeping V, softmax scores, and AV in F32. Disabled by default because it
  // changes the attention-score precision boundary and must remain opt-in.
  bool use_attention_qk_bf16 = false;

  // Probe-only local-kernel variant of the QK BF16 experiment. It uses the
  // grouped-2D RDNA WMMA 2x8 transposed-B kernel for the BF16 QK score GEMM,
  // then keeps F32 softmax and F32 AV. This inherits the same precision caveat
  // as use_attention_qk_bf16 and is kept separate for A/B testing against
  // rocBLAS BF16 QK.
  bool use_attention_qk_bf16_wmma = false;

  // Probe-only experimental AV replacement. QK stays in F32, but the softmax
  // kernel stores normalized scores as BF16 and AV runs through the local RDNA
  // WMMA BF16->F32 kernel. Disabled by default because it changes normalized
  // attention-score storage precision and needs generation A/B guardrails.
  bool use_attention_av_bf16_wmma = false;

  // Probe-only F32-preserving AV replacement. QK and softmax stay on the
  // existing F32 path; only the AV GEMM is replaced by a shape-specialized
  // float4 tiled HIP kernel for the 224px rows=256, qkv_dim=72 case. Disabled
  // by default because the 448px shape is still faster through rocBLAS.
  bool use_attention_av_f32_dim4 = false;

  // Probe-only F32-semantics attention-output packing optimization. QK,
  // softmax, and AV stay F32; only the existing post-attention pack and
  // F32-to-BF16 conversion for the following attn_out GEMM are fused.
  bool use_attention_pack_bf16 = false;

  // Probe-only F32 attention layout optimization. QK and AV stay rocBLAS F32,
  // but they read Q/K/V directly from the row-major interleaved QKV projection
  // using a larger leading dimension instead of first splitting to contiguous
  // head-major buffers.
  bool use_attention_direct_qkv = false;

  // Probe-only scheduling variant for the pack-to-BF16 route. Softmax stores
  // exp(score - max) and one F32 reciprocal denominator per row; AV runs before
  // row normalization, then pack-to-BF16 applies the saved scale on the smaller
  // attention output. This keeps QK/AV in F32 but changes floating-point
  // association, so it remains opt-in behind generation A/B tests.
  bool use_attention_defer_softmax_scale = false;

  // Probe-only rocBLAS solution overrides for the F32 attention core. Zero
  // keeps rocBLAS' default selection. Nonzero values are architecture/runtime
  // specific solution indices discovered by the experimental bench; they remain
  // explicit knobs until a cache format is added.
  int attention_qk_solution_index = 0;
  int attention_av_solution_index = 0;

  // Probe-only experimental QKV replacement. When enabled, the BF16 ViT layer
  // runner computes QKV with the local grouped-2D RDNA WMMA 8x4 macro-tile
  // kernel instead of rocBLAS. Disabled by default; this is a shape-specialized
  // performance experiment.
  bool use_qkv_wmma2d = false;

  // Probe-only experimental attention-output projection replacement. When
  // enabled, the BF16 ViT layer runner computes attn_out with the local
  // grouped-2D RDNA WMMA 8x4 macro-tile kernel instead of rocBLAS.
  bool use_attn_out_wmma2d = false;

  // Probe-only experimental MLP up replacement. When enabled, the BF16 ViT
  // layer runner computes MLP up with the local grouped-2D RDNA WMMA 8x4
  // macro-tile kernel. This is disabled by default because the kernel is
  // shape-specialized and remains an opt-in performance experiment.
  bool use_mlp_up_wmma2d = false;

  // Probe-only experimental MLP down replacement. When enabled, the BF16 ViT
  // layer runner computes MLP down with the local grouped-N RDNA WMMA kernel.
  // This is disabled by default because the kernel is shape-specialized and
  // remains an opt-in performance experiment.
  bool use_mlp_down_wmma8 = false;

  // Optional epilogue fusion for the local MLP-down WMMA path. This preserves
  // the existing F32 order `down + linear_1_bias + residual`, but writes the
  // final layer output directly from the WMMA kernel instead of launching
  // separate bias and residual kernels.
  bool use_mlp_down_fused_residual = false;
  int mlp_down_wmma_waves = 8;
};

struct PaliGemma2VitHipStats {
  std::string device_name;
  std::string device_arch;
  int compute_units = 0;
  size_t total_bytes = 0;
  size_t free_bytes = 0;
  size_t resident_projection_bytes = 0;
  double upload_seconds = 0.0;
  double upload_gib_per_second = 0.0;
  bool patch_embedding_validated = false;
  // F32 path: exact semantic bridge for the current CPU input type
  // (F32 patches x BF16 weights widened to F32 shadow weights).
  float patch_embedding_ms = 0.0f;
  float patch_embedding_max_abs_error = 0.0f;
  float patch_embedding_rms_error = 0.0f;
  // BF16 path: faster rocBLAS A/B BF16 route. This is a precision experiment,
  // not yet the default correctness path.
  float patch_embedding_bf16_ms = 0.0f;
  float patch_embedding_bf16_max_abs_error = 0.0f;
  float patch_embedding_bf16_rms_error = 0.0f;
  bool layer0_qkv_validated = false;
  float layer0_qkv_f32_ms = 0.0f;
  float layer0_qkv_f32_spot_max_abs_error = 0.0f;
  float layer0_qkv_bf16_ms = 0.0f;
  float layer0_qkv_bf16_max_abs_error = 0.0f;
  float layer0_qkv_bf16_rms_error = 0.0f;
  bool layer0_mlp_validated = false;
  float layer0_mlp_f32_ms = 0.0f;
  float layer0_mlp_f32_spot_max_abs_error = 0.0f;
  float layer0_mlp_bf16_ms = 0.0f;
  float layer0_mlp_bf16_max_abs_error = 0.0f;
  float layer0_mlp_bf16_rms_error = 0.0f;
  bool layer0_attn_out_validated = false;
  float layer0_attn_out_f32_ms = 0.0f;
  float layer0_attn_out_f32_spot_max_abs_error = 0.0f;
  float layer0_attn_out_bf16_ms = 0.0f;
  float layer0_attn_out_bf16_max_abs_error = 0.0f;
  float layer0_attn_out_bf16_rms_error = 0.0f;
  bool layer0_block_validated = false;
  double layer0_block_attention_f32_seconds = 0.0;
  double layer0_block_attention_bf16_seconds = 0.0;
  float layer0_block_qkv_f32_ms = 0.0f;
  float layer0_block_qkv_bf16_ms = 0.0f;
  float layer0_block_attn_out_f32_ms = 0.0f;
  float layer0_block_attn_out_bf16_ms = 0.0f;
  float layer0_block_mlp_f32_ms = 0.0f;
  float layer0_block_mlp_bf16_ms = 0.0f;
  float layer0_block_bf16_max_abs_error = 0.0f;
  float layer0_block_bf16_rms_error = 0.0f;
  bool layer0_device_attention_validated = false;
  double layer0_device_attention_host_f32_seconds = 0.0;
  double layer0_device_attention_host_bf16_seconds = 0.0;
  float layer0_device_attention_f32_ms = 0.0f;
  float layer0_device_attention_f32_max_abs_error = 0.0f;
  float layer0_device_attention_f32_rms_error = 0.0f;
  float layer0_device_attention_bf16_ms = 0.0f;
  float layer0_device_attention_bf16_max_abs_error = 0.0f;
  float layer0_device_attention_bf16_rms_error = 0.0f;
  bool layer0_block_device_attention_validated = false;
  // Timings and validation errors for the device-input layer0 graph. Patch and
  // LN0 are separated from the attention/MLP fields so we can see whether the
  // new input boundary changes the cost or numerical drift before scaling this
  // into a reusable multi-layer runner.
  float layer0_block_device_patch_f32_ms = 0.0f;
  float layer0_block_device_patch_bf16_ms = 0.0f;
  float layer0_block_device_patch_f32_max_abs_error = 0.0f;
  float layer0_block_device_patch_f32_rms_error = 0.0f;
  float layer0_block_device_patch_bf16_max_abs_error = 0.0f;
  float layer0_block_device_patch_bf16_rms_error = 0.0f;
  float layer0_block_device_ln0_f32_ms = 0.0f;
  float layer0_block_device_ln0_bf16_ms = 0.0f;
  float layer0_block_device_ln0_f32_max_abs_error = 0.0f;
  float layer0_block_device_ln0_f32_rms_error = 0.0f;
  float layer0_block_device_ln0_bf16_max_abs_error = 0.0f;
  float layer0_block_device_ln0_bf16_rms_error = 0.0f;
  float layer0_block_device_attention_qkv_f32_ms = 0.0f;
  float layer0_block_device_attention_qkv_bf16_ms = 0.0f;
  float layer0_block_device_attention_attn_f32_ms = 0.0f;
  float layer0_block_device_attention_attn_bf16_ms = 0.0f;
  float layer0_block_device_attention_attn_out_f32_ms = 0.0f;
  float layer0_block_device_attention_attn_out_bf16_ms = 0.0f;
  float layer0_block_device_attention_ln1_f32_ms = 0.0f;
  float layer0_block_device_attention_ln1_bf16_ms = 0.0f;
  float layer0_block_device_attention_ln1_f32_max_abs_error = 0.0f;
  float layer0_block_device_attention_ln1_f32_rms_error = 0.0f;
  float layer0_block_device_attention_ln1_bf16_max_abs_error = 0.0f;
  float layer0_block_device_attention_ln1_bf16_rms_error = 0.0f;
  float layer0_block_device_attention_mlp_f32_ms = 0.0f;
  float layer0_block_device_attention_mlp_bf16_ms = 0.0f;
  float layer0_block_device_attention_bf16_max_abs_error = 0.0f;
  float layer0_block_device_attention_bf16_rms_error = 0.0f;
  bool layer0_block_device_norm_validated = false;
  bool layer_prefix2_device_norm_validated = false;
  float layer_prefix2_patch_f32_ms = 0.0f;
  float layer_prefix2_patch_bf16_ms = 0.0f;
  float layer_prefix2_ln0_f32_ms = 0.0f;
  float layer_prefix2_ln0_bf16_ms = 0.0f;
  float layer_prefix2_qkv_f32_ms = 0.0f;
  float layer_prefix2_qkv_bf16_ms = 0.0f;
  float layer_prefix2_attn_f32_ms = 0.0f;
  float layer_prefix2_attn_bf16_ms = 0.0f;
  float layer_prefix2_attn_out_f32_ms = 0.0f;
  float layer_prefix2_attn_out_bf16_ms = 0.0f;
  float layer_prefix2_ln1_f32_ms = 0.0f;
  float layer_prefix2_ln1_bf16_ms = 0.0f;
  float layer_prefix2_mlp_f32_ms = 0.0f;
  float layer_prefix2_mlp_bf16_ms = 0.0f;
  float layer_prefix2_patch_f32_max_abs_error = 0.0f;
  float layer_prefix2_patch_f32_rms_error = 0.0f;
  float layer_prefix2_patch_bf16_max_abs_error = 0.0f;
  float layer_prefix2_patch_bf16_rms_error = 0.0f;
  float layer_prefix2_bf16_max_abs_error = 0.0f;
  float layer_prefix2_bf16_rms_error = 0.0f;
  bool layer_stack_device_norm_validated = false;
  float layer_stack_patch_f32_ms = 0.0f;
  float layer_stack_patch_bf16_ms = 0.0f;
  float layer_stack_ln0_f32_ms = 0.0f;
  float layer_stack_ln0_bf16_ms = 0.0f;
  float layer_stack_qkv_f32_ms = 0.0f;
  float layer_stack_qkv_bf16_ms = 0.0f;
  float layer_stack_attn_f32_ms = 0.0f;
  float layer_stack_attn_bf16_ms = 0.0f;
  float layer_stack_attn_out_f32_ms = 0.0f;
  float layer_stack_attn_out_bf16_ms = 0.0f;
  float layer_stack_ln1_f32_ms = 0.0f;
  float layer_stack_ln1_bf16_ms = 0.0f;
  float layer_stack_mlp_f32_ms = 0.0f;
  float layer_stack_mlp_bf16_ms = 0.0f;
  float layer_stack_patch_f32_max_abs_error = 0.0f;
  float layer_stack_patch_f32_rms_error = 0.0f;
  float layer_stack_patch_bf16_max_abs_error = 0.0f;
  float layer_stack_patch_bf16_rms_error = 0.0f;
  float layer_stack_bf16_max_abs_error = 0.0f;
  float layer_stack_bf16_rms_error = 0.0f;
  bool image_tokens_device_norm_validated = false;
  float image_tokens_stack_f32_ms = 0.0f;
  float image_tokens_stack_bf16_ms = 0.0f;
  float image_tokens_final_norm_f32_ms = 0.0f;
  float image_tokens_final_norm_bf16_ms = 0.0f;
  float image_tokens_head_f32_ms = 0.0f;
  float image_tokens_head_bf16_ms = 0.0f;
  float image_tokens_bf16_vs_f32_max_abs_error = 0.0f;
  float image_tokens_bf16_vs_f32_rms_error = 0.0f;
  float image_tokens_f32_vs_cpu_max_abs_error = 0.0f;
  float image_tokens_f32_vs_cpu_rms_error = 0.0f;
  float image_tokens_bf16_vs_cpu_max_abs_error = 0.0f;
  float image_tokens_bf16_vs_cpu_rms_error = 0.0f;
  bool image_tokens_attention_qk_bf16_validated = false;
  float image_tokens_attention_qk_bf16_max_abs_error = 0.0f;
  float image_tokens_attention_qk_bf16_rms_error = 0.0f;
  bool image_tokens_device_norm_returned = false;
  float image_tokens_return_stack_bf16_ms = 0.0f;
  float image_tokens_return_final_norm_bf16_ms = 0.0f;
  float image_tokens_return_head_bf16_ms = 0.0f;
  PaliGemma2VitHipTiming model_projection_schedule;
  PaliGemma2VitHipTiming resident_projection_schedule;
};

size_t PaliGemma2VitProjectionBytes(const WeightsPtrs& weights);
void PrintPaliGemma2VitHipModelSummary(const ModelConfig& config,
                                       const WeightsPtrs& weights);

class PaliGemma2VitHipBackend {
 public:
  explicit PaliGemma2VitHipBackend(PaliGemma2VitHipOptions options);
  ~PaliGemma2VitHipBackend();

  PaliGemma2VitHipBackend(const PaliGemma2VitHipBackend&) = delete;
  PaliGemma2VitHipBackend& operator=(const PaliGemma2VitHipBackend&) = delete;

  bool Initialize();

  // Uploads all ViT projection weights once into GPU-resident layouts. The main
  // copy is BF16 B[K,N] for rocBLAS; when patch validation is enabled, a F32
  // shadow of img_emb_kernel is also uploaded to preserve CPU patch semantics.
  bool UploadProjectionWeights(const WeightsPtrs& weights);

  // Synthetic schedule using model-derived dimensions and dummy buffers. Useful
  // for isolating rocBLAS shape performance without weight upload/layout cost.
  bool RunModelProjectionSchedule(const ModelConfig& config);

  // Same projection schedule, but using the real resident BF16 weights uploaded
  // by UploadProjectionWeights.
  bool RunResidentProjectionSchedule(const ModelConfig& config);

  // First real tensor boundary. Computes patch embedding on HIP, adds bias and
  // position embedding, copies results back, and compares F32 and BF16 variants
  // against a CPU scalar reference.
  bool ValidatePatchEmbedding(const ModelConfig& model_config,
                              const WeightsPtrs& weights, const Image& image);
  bool ValidateLayer0QKV(const ModelConfig& model_config,
                         const WeightsPtrs& weights, const Image& image);
  bool ValidateLayer0MLP(const ModelConfig& model_config,
                         const WeightsPtrs& weights, const Image& image);
  bool ValidateLayer0AttentionOut(const ModelConfig& model_config,
                                  const WeightsPtrs& weights,
                                  const Image& image);
  bool ValidateLayer0Block(const ModelConfig& model_config,
                           const WeightsPtrs& weights, const Image& image);
  bool ValidateLayer0DeviceAttention(const ModelConfig& model_config,
                                     const WeightsPtrs& weights,
                                     const Image& image);
  bool ValidateLayer0BlockDeviceAttention(const ModelConfig& model_config,
                                          const WeightsPtrs& weights,
                                          const Image& image);
  bool ValidateLayer0BlockDeviceNorm(const ModelConfig& model_config,
                                     const WeightsPtrs& weights,
                                     const Image& image);
  bool ValidateLayerPrefix2DeviceNorm(const ModelConfig& model_config,
                                      const WeightsPtrs& weights,
                                      const Image& image);
  bool ValidateLayerStackDeviceNorm(const ModelConfig& model_config,
                                    const WeightsPtrs& weights,
                                    const Image& image);
  bool ValidateImageTokensDeviceNorm(const ModelConfig& model_config,
                                     const WeightsPtrs& weights,
                                     const Image& image,
                                     ImageTokens* image_tokens,
                                     const ImageTokens* cpu_reference);
  bool ValidateMlpDownWmma8(const ModelConfig& model_config,
                            const WeightsPtrs& weights, const Image& image);
  bool ValidateMlpUpWmma2D(const ModelConfig& model_config,
                           const WeightsPtrs& weights, const Image& image);
  bool ValidateQkvWmma2D(const ModelConfig& model_config,
                         const WeightsPtrs& weights, const Image& image);
  bool ValidateAttnOutWmma2D(const ModelConfig& model_config,
                             const WeightsPtrs& weights, const Image& image);
  bool ValidateAttentionQkBf16ImageTokens(const ModelConfig& model_config,
                                          const WeightsPtrs& weights,
                                          const Image& image,
                                          ImageTokens& f32_attention_tokens,
                                          ImageTokens& qk_bf16_tokens);
  bool GenerateImageTokensDeviceNorm(const ModelConfig& model_config,
                                     const WeightsPtrs& weights,
                                     const Image& image,
                                     ImageTokens& image_tokens);
  bool ProfileImageTokensReturnDeviceNorm(const ModelConfig& model_config,
                                          const WeightsPtrs& weights,
                                          const Image& image,
                                          ImageTokens& image_tokens);
  bool LoadMlpGemmSolutionCache(const ModelConfig& model_config,
                                const WeightsPtrs& weights,
                                const std::string& path);
  bool BenchmarkMlpGemmSolutions(const ModelConfig& model_config,
                                 const WeightsPtrs& weights,
                                 int max_solutions);
  bool SaveMlpGemmSolutionCache(const ModelConfig& model_config,
                                const WeightsPtrs& weights,
                                const std::string& path) const;
  bool LoadAttentionGemmSolutionCache(const ModelConfig& model_config,
                                      const WeightsPtrs& weights,
                                      const std::string& path);
  bool BenchmarkAttentionGemmSolutions(const ModelConfig& model_config,
                                       const WeightsPtrs& weights,
                                       int max_solutions);
  bool SaveAttentionGemmSolutionCache(const ModelConfig& model_config,
                                      const WeightsPtrs& weights,
                                      const std::string& path) const;

  // Current integration stage: upload/benchmark resident projection weights,
  // then return false so the existing CPU ViT path remains the source of
  // inference results. A later stage will fill image_tokens and return true.
  bool TryGenerateImageTokens(const ModelConfig& model_config,
                              const WeightsPtrs& weights, size_t seq_len,
                              const Image& image, ImageTokens& image_tokens,
                              MatMulEnv& env);

  const PaliGemma2VitHipStats& stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

bool PaliGemma2VitHipImageTokensBackend(void* backend,
                                        const ModelConfig& model_config,
                                        const WeightsPtrs& weights,
                                        size_t seq_len, const Image& image,
                                        ImageTokens& image_tokens,
                                        MatMulEnv& env);

}  // namespace experimental
}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_EXPERIMENTAL_PALIGEMMA2_VIT_HIP_BACKEND_H_
