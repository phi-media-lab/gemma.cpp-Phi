// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Shared between various frontends.

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_ARGS_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_ARGS_H_

#include <stddef.h>
#include <stdio.h>

#include <functional>
#include <string>

#include "io/io.h"  // Path
#include "util/args.h"
#include "util/basics.h"  // Tristate
#include "util/mat.h"
#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"               // HWY_ABORT
#include "hwy/profiler.h"

namespace gcpp {

struct MatMulEnv;
struct ModelConfig;
struct WeightsPtrs;
class Image;

struct LoaderArgs : public ArgsBase<LoaderArgs> {
  LoaderArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }
  LoaderArgs(const std::string& tokenizer_path,
             const std::string& weights_path) {
    Init();  // Init sets to defaults, so assignments must come after Init().
    tokenizer.path = tokenizer_path;
    weights.path = weights_path;
  };

  Path tokenizer;
  Path weights;  // weights file location
  Tristate map;
  Tristate to_bf16;
  Tristate wrapping;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(tokenizer, "tokenizer", Path(),
            "Path name of tokenizer model; only required for pre-2025 format.");
    visitor(weights, "weights", Path(),
            "Path name of model weights (.sbs) file.\n  Required argument.\n");
    visitor(map, "map", Tristate::kDefault,
            "Enable memory-mapping? -1 = auto, 0 = no, 1 = yes.");
    visitor(to_bf16, "to_bf16", Tristate::kDefault,
            "Convert weights to bf16? -1 = auto, 0 = no, 1 = yes.");
    visitor(wrapping, "wrapping", Tristate::kDefault,
            "Enable prompt wrapping? Specify 0 for pre-2025 format PT models.");
  }
};

using PromptTokens = hwy::Span<const int>;

// Batches of independent queries have their own prompt, previous token,
// position in the sequence, and KVCache.
using QueriesPromptTokens = hwy::Span<const PromptTokens>;
using QueriesToken = hwy::Span<const int>;
using QueriesPos = hwy::Span<const size_t>;

// ImageTokens are represented as a matrix, where each row corresponds
// to a token for an image patch as computed by the image encoder.
using ImageTokens = MatStorageT<float>;

// Experimental hook for replacing the image-token generator.
//
// Contract:
// - `backend` is opaque state owned by the caller that builds RuntimeConfig.
// - Implementations may inspect model weights and run device-side work.
// - Return true only after fully populating `image_tokens` with the same layout
//   expected by the decoder prefix path.
// - Return false to fall back to the normal CPU ViT implementation. The current
//   HIP probe intentionally does this after validation so correctness remains
//   anchored to the existing CPU code.
using ImageTokensBackendFunc =
    bool (*)(void* backend, const ModelConfig& model_config,
             const WeightsPtrs& weights, size_t seq_len, const Image& image,
             ImageTokens& image_tokens, MatMulEnv& env);

// StreamFunc is called with (token, probability). For prompt tokens,
// probability is 0.0f. StreamFunc should return false to stop generation and
// true to continue generation.
using StreamFunc = std::function<bool(int, float)>;
// BatchStreamFunc is called with (query_idx, pos, token, probability).
// For prompt tokens, probability is 0.0f. Generation continues if this returns
// true and stops if it returns false. Note that query_idx is absolute, not
// relative to the batch.
using BatchStreamFunc = std::function<bool(size_t, size_t, int, float)>;
// If not empty, AcceptFunc is called with token. It should return false for
// tokens you don't want to generate and true for tokens you want to generate.
using AcceptFunc = std::function<bool(int, float)>;
// If not empty, SampleFunc is called concurrently from worker thread(s) with
// query_idx, pos, logits for the next token (which it may modify/overwrite),
// and worker. It returns the next generated token and its probability.
using SampleFunc = std::function<TokenAndProb(size_t, size_t, Logits, size_t)>;
// If not empty, LayersOutputFunc is called for layer outputs, specified with:
// - index of query within containing batch (if any); zero otherwise.
// - position in the tokens sequence
// - name of the data, e.g. "tokens" for token IDs
// - layer index (or -1 for global outputs)
// - pointer to the data array
// - size of the data array
using LayersOutputFunc = std::function<void(size_t, size_t, const std::string&,
                                            int, const float*, size_t)>;
// If not empty, ActivationsObserverFunc is invoked after each layer with:
// - per-query position within the tokens sequence
// - layer index (or -1 for post-norm output)
// - activations
struct Activations;
using ActivationsObserverFunc =
    std::function<void(const QueriesPos& queries_pos, int, const Activations&)>;

// RuntimeConfig holds configuration for a single generation run.
// TODO: move into InferenceArgs, use that directly.
struct RuntimeConfig {
  // If non-null, `batch_stream_token` is called for each token in the batch,
  // otherwise `stream_token`. `query_idx` is absolute, not batch-relative.
  // This is called sequentially from the main thread.
  bool StreamToken(size_t query_idx, size_t pos, int token, float prob) const {
    PROFILER_ZONE("Gen.StreamToken");
    if (batch_stream_token) {
      return batch_stream_token(query_idx, pos, token, prob);
    }
    return stream_token(token, prob);
  }

  // Limit on the number of tokens generated.
  size_t max_generated_tokens;

  // These defaults are overridden by InferenceArgs::CopyTo(*this):
  // Max tokens per batch during prefill.
  size_t prefill_tbatch_size = 256;
  // Max queries per batch (one token from each) during decode.
  size_t decode_qbatch_size = 16;

  // Sampling-related parameters.
  float temperature;  // Temperature for sampling.

  size_t top_k = 1;  // Top-k for sampling.

  int verbosity;  // Controls verbosity of printed messages.

  // Functions operating on the generated tokens.
  StreamFunc stream_token;
  BatchStreamFunc batch_stream_token;
  AcceptFunc accept_token;  // if empty, accepts all tokens.
  SampleFunc sample_func;   // if empty, uses SampleTopK.

  // Observer callbacks for intermediate data.
  LayersOutputFunc layers_output;  // if not empty, called after each layer.
  ActivationsObserverFunc activations_observer;  // if set, called per-layer.

  // Experimental decoder-attention override. On AVX-512 BF16 targets the
  // existing default intentionally routes Gemma decoder attention through the
  // older BF16 dot-product path. PaliGemma2 448 has a large image/text prefix,
  // so this flag lets benchmarks force FlashAttention without changing the
  // production default for other models.
  bool force_flash_attention = false;

  // If not empty, these point to the image tokens and are used in the
  // PaliGemma prefix-LM style attention.
  const ImageTokens* image_tokens = nullptr;

  // Optional backend hook for the image encoder. The pointer/function pair is
  // intentionally opaque so experimental device backends can live outside
  // libgemma and still plug into the real GenerateImageTokens flow. libgemma
  // does not own or delete `image_tokens_backend`; the creator of RuntimeConfig
  // must keep it alive for the duration of GenerateImageTokens.
  void* image_tokens_backend = nullptr;
  ImageTokensBackendFunc image_tokens_backend_func = nullptr;

  // Whether to use thread spinning to reduce barrier synchronization latency.
  // Mutable so we can change kDefault to kTrue/kFalse during Generate, because
  // RuntimeConfig is const there and is not passed to the Gemma ctor. This
  // default decision is likely sufficient because it is based on whether
  // threads are successfully pinned.
  mutable Tristate use_spinning = Tristate::kDefault;
};

struct InferenceArgs : public ArgsBase<InferenceArgs> {
  InferenceArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }
  InferenceArgs() { Init(); };

  bool IsInteractive() const { return prompt.empty() && prompt_file.Empty(); }

  int verbosity;

  size_t seq_len;
  size_t max_generated_tokens;

  size_t prefill_tbatch_size;
  size_t decode_qbatch_size;

  float temperature;
  size_t top_k;
  bool deterministic;
  bool multiturn;

  // Experimental PaliGemma image-phase controls. These are CLI-level knobs, not
  // RuntimeConfig fields, because they affect ThreadingContext construction
  // before GenerateImageTokens runs. A zero image_max_lps keeps the original
  // single-context behavior.
  bool force_flash_attention;
  Path image_file;
  size_t image_skip_lps;
  size_t image_max_lps;
  std::string paligemma_vit_backend;
  int paligemma_vit_hip_samples;
  int paligemma_vit_hip_warmup;
  int paligemma_vit_hip_iters;
  bool paligemma_vit_hip_validate_patch;
  bool paligemma_vit_hip_validate_qkv;
  bool paligemma_vit_hip_validate_mlp;
  bool paligemma_vit_hip_validate_attn_out;
  bool paligemma_vit_hip_validate_layer0_block;
  bool paligemma_vit_hip_validate_device_attention;
  bool paligemma_vit_hip_validate_layer0_block_device_attention;
  bool paligemma_vit_hip_validate_layer0_block_device_norm;
  bool paligemma_vit_hip_validate_layer_prefix2_device_norm;
  bool paligemma_vit_hip_validate_layer_stack_device_norm;
  bool paligemma_vit_hip_validate_image_tokens_device_norm;
  bool paligemma_vit_hip_return_image_tokens;
  bool paligemma_vit_hip_attention_qk_bf16;
  bool paligemma_vit_hip_attention_qk_bf16_wmma;
  bool paligemma_vit_hip_attention_av_bf16_wmma;
  bool paligemma_vit_hip_attention_av_f32_dim4;
  bool paligemma_vit_hip_attention_pack_bf16;
  bool paligemma_vit_hip_attention_direct_qkv;
  bool paligemma_vit_hip_attention_defer_softmax_scale;
  int paligemma_vit_hip_attention_qk_solution;
  int paligemma_vit_hip_attention_av_solution;
  bool paligemma_vit_hip_qkv_wmma2d;
  bool paligemma_vit_hip_attn_out_wmma2d;
  bool paligemma_vit_hip_mlp_up_wmma2d;
  bool paligemma_vit_hip_mlp_down_wmma8;
  bool paligemma_vit_hip_mlp_down_fused_residual;
  int paligemma_vit_hip_mlp_down_wmma_waves;
  std::string paligemma_vit_hip_mlp_solution_cache;
  std::string paligemma_vit_hip_attention_solution_cache;

  int port;            // Server port
  std::string model;   // Model name for API endpoints
  std::string prompt;  // Bypasses std::getline
  // For prompts longer than the Linux terminal's 4K line edit buffer.
  Path prompt_file;
  std::string eot_line;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(verbosity, "verbosity", 1,
            "Show verbose developer information\n    0 = only print generation "
            "output\n    1 = standard user-facing terminal ui\n    2 = show "
            "developer/debug info).\n    Default = 1.",
            1);

    visitor(seq_len, "seq_len", size_t{8192},
            "Sequence length, capped by ModelConfig.max_seq_len.");
    visitor(max_generated_tokens, "max_generated_tokens", size_t{4096},
            "Maximum number of tokens to generate.");

    visitor(prefill_tbatch_size, "prefill_tbatch", size_t{256},
            "Prefill: max tokens per batch.");
    visitor(decode_qbatch_size, "decode_qbatch", size_t{16},
            "Decode: max queries per batch.");

    visitor(temperature, "temperature", 1.0f, "Temperature for top-K", 2);
    visitor(top_k, "top_k", size_t{1}, "Number of top-K tokens to sample from",
            2);
    visitor(deterministic, "deterministic", false,
            "Make top-k sampling deterministic", 2);
    visitor(multiturn, "multiturn", false,
            "Multiturn mode\n    0 = clear KV cache after every "
            "interaction\n    1 = continue KV cache after every interaction\n  "
            "  Default : 0 (conversation "
            "resets every turn)");

    // Benchmark-only switch for comparing decoder attention implementations on
    // the same binary and weights. It only influences Gemma decoder layers; the
    // ViT image encoder has its own attention implementation.
    visitor(force_flash_attention, "force_flash_attention", false,
            "Experimental: force decoder FlashAttention instead of the default "
            "attention path.",
            2);
    visitor(image_file, "image_file", Path(), "Image file to load.");

    // Benchmark-only image-threading switch. `image_skip_lps` and
    // `image_max_lps` are forwarded into a separate ThreadingContext used only
    // for GenerateImageTokens, so the decoder can keep the main topology.
    visitor(image_skip_lps, "image_skip_lps", size_t{0},
            "Experimental: first logical processor to use for image-token "
            "generation when image_max_lps is nonzero.",
            2);
    visitor(image_max_lps, "image_max_lps", size_t{0},
            "Experimental: max logical processors to use for image-token "
            "generation. Default 0 uses the main threading context.",
            2);
    visitor(paligemma_vit_backend, "paligemma_vit_backend",
            std::string("cpu"),
            "Experimental: PaliGemma image encoder backend. Values: cpu, "
            "hip_probe. hip_probe uploads resident HIP projection weights and "
            "falls back to CPU for correctness.",
            2);
    visitor(paligemma_vit_hip_samples, "paligemma_vit_hip_samples", 1,
            "Experimental: samples for hip_probe resident projection timing.",
            3);
    visitor(paligemma_vit_hip_warmup, "paligemma_vit_hip_warmup", 1,
            "Experimental: warmup iterations for hip_probe resident projection "
            "timing.",
            3);
    visitor(paligemma_vit_hip_iters, "paligemma_vit_hip_iters", 1,
            "Experimental: timed inner iterations for hip_probe resident "
            "projection timing.",
            3);
    visitor(paligemma_vit_hip_validate_patch,
            "paligemma_vit_hip_validate_patch", false,
            "Experimental: validate HIP patch embedding against a CPU scalar "
            "reference inside hip_probe.",
            3);
    visitor(paligemma_vit_hip_validate_qkv, "paligemma_vit_hip_validate_qkv",
            false,
            "Experimental: validate HIP layer0 QKV projection F32/BF16 paths "
            "inside hip_probe.",
            3);
    visitor(paligemma_vit_hip_validate_mlp, "paligemma_vit_hip_validate_mlp",
            false,
            "Experimental: validate HIP layer0 MLP up/GELU/down F32/BF16 paths "
            "inside hip_probe.",
            3);
    visitor(paligemma_vit_hip_validate_attn_out,
            "paligemma_vit_hip_validate_attn_out", false,
            "Experimental: validate HIP layer0 attention-output projection and "
            "residual F32/BF16 paths inside hip_probe.",
            3);
    visitor(paligemma_vit_hip_validate_layer0_block,
            "paligemma_vit_hip_validate_layer0_block", false,
            "Experimental: validate full HIP layer0 output F32/BF16 paths "
            "inside hip_probe, with attention softmax as a host reference.",
            3);
    visitor(paligemma_vit_hip_validate_device_attention,
            "paligemma_vit_hip_validate_device_attention", false,
            "Experimental: validate layer0 HIP device attention core "
            "(QK + softmax + AV) inside hip_probe.",
            3);
    visitor(paligemma_vit_hip_validate_layer0_block_device_attention,
            "paligemma_vit_hip_validate_layer0_block_device_attention", false,
            "Experimental: validate full HIP layer0 output using device "
            "attention inside hip_probe. This legacy flag currently aliases "
            "the device input/layernorm validation path.",
            3);
    visitor(paligemma_vit_hip_validate_layer0_block_device_norm,
            "paligemma_vit_hip_validate_layer0_block_device_norm", false,
            "Experimental: validate full HIP layer0 output using device "
            "patch embedding, device attention, and device layernorms inside "
            "hip_probe.",
            3);
    visitor(paligemma_vit_hip_validate_layer_prefix2_device_norm,
            "paligemma_vit_hip_validate_layer_prefix2_device_norm", false,
            "Experimental: validate a two-layer HIP ViT prefix using device "
            "patch embedding, the reusable device layer runner, and BF16 vs "
            "F32-shadow comparison inside hip_probe.",
            3);
    visitor(paligemma_vit_hip_validate_layer_stack_device_norm,
            "paligemma_vit_hip_validate_layer_stack_device_norm", false,
            "Experimental: validate the full HIP ViT transformer stack through "
            "the reusable device layer runner, stopping before final encoder "
            "norm and image head.",
            3);
    visitor(paligemma_vit_hip_validate_image_tokens_device_norm,
            "paligemma_vit_hip_validate_image_tokens_device_norm", false,
            "Experimental: validate HIP PaliGemma image tokens through final "
            "encoder norm and image head. The CLI compares BF16 vs F32-shadow "
            "inside hip_probe and then falls back to CPU image tokens.",
            3);
    visitor(paligemma_vit_hip_return_image_tokens,
            "paligemma_vit_hip_return_image_tokens", false,
            "Experimental: return HIP-produced BF16 PaliGemma image tokens from "
            "hip_probe so the decoder consumes them. This is an explicit A/B "
            "test knob and still runs the F32-shadow diagnostics.",
            3);
    visitor(paligemma_vit_hip_attention_qk_bf16,
            "paligemma_vit_hip_attention_qk_bf16", false,
            "Experimental: use BF16 Q/K for the HIP PaliGemma ViT attention "
            "QK GEMM while keeping V, softmax scores, and AV in F32. This only "
            "affects hip_probe return/profile paths and remains opt-in.",
            3);
    visitor(paligemma_vit_hip_attention_qk_bf16_wmma,
            "paligemma_vit_hip_attention_qk_bf16_wmma", false,
            "Experimental: use the local grouped-2D RDNA WMMA BF16 QK kernel "
            "for HIP PaliGemma ViT attention while keeping V, softmax scores, "
            "and AV in F32. This has the same precision caveat as "
            "paligemma_vit_hip_attention_qk_bf16 and remains opt-in.",
            3);
    visitor(paligemma_vit_hip_attention_av_bf16_wmma,
            "paligemma_vit_hip_attention_av_bf16_wmma", false,
            "Experimental: keep HIP PaliGemma ViT attention QK in F32, but "
            "store normalized softmax scores as BF16 and run AV with the local "
            "RDNA WMMA BF16->F32 kernel. This changes attention-score storage "
            "precision and remains opt-in.",
            3);
    visitor(paligemma_vit_hip_attention_av_f32_dim4,
            "paligemma_vit_hip_attention_av_f32_dim4", false,
            "Experimental: keep HIP PaliGemma ViT attention QK, softmax, and "
            "AV in F32, but replace the 224px AV GEMM with a local float4 "
            "tiled HIP kernel. 448px falls back to rocBLAS.",
            3);
    visitor(paligemma_vit_hip_attention_pack_bf16,
            "paligemma_vit_hip_attention_pack_bf16", false,
            "Experimental: keep HIP PaliGemma ViT attention QK, softmax, and "
            "AV in F32, but fuse the post-attention head pack with the existing "
            "BF16 conversion for attn_out. This is scoped to hip_probe return "
            "and profile paths.",
            3);
    visitor(paligemma_vit_hip_attention_direct_qkv,
            "paligemma_vit_hip_attention_direct_qkv", false,
            "Experimental: keep HIP PaliGemma ViT attention QK, softmax, and "
            "AV in F32, but have rocBLAS read Q/K/V directly from the "
            "interleaved QKV projection layout instead of splitting to "
            "head-major scratch buffers.",
            3);
    visitor(paligemma_vit_hip_attention_defer_softmax_scale,
            "paligemma_vit_hip_attention_defer_softmax_scale", false,
            "Experimental: with attention_pack_bf16, store softmax exp scores "
            "and apply the per-row reciprocal scale while packing attention "
            "heads to BF16. QK and AV stay F32, but floating-point association "
            "changes, so this remains opt-in.",
            3);
    visitor(paligemma_vit_hip_attention_qk_solution,
            "paligemma_vit_hip_attention_qk_solution", 0,
            "Experimental: rocBLAS solution index override for HIP PaliGemma "
            "ViT F32 attention QK. Zero keeps the rocBLAS default. Nonzero "
            "values are architecture/runtime-specific bench results.",
            3);
    visitor(paligemma_vit_hip_attention_av_solution,
            "paligemma_vit_hip_attention_av_solution", 0,
            "Experimental: rocBLAS solution index override for HIP PaliGemma "
            "ViT F32 attention AV. Zero keeps the rocBLAS default. Nonzero "
            "values are architecture/runtime-specific bench results.",
            3);
    visitor(paligemma_vit_hip_qkv_wmma2d,
            "paligemma_vit_hip_qkv_wmma2d", false,
            "Experimental: use the grouped-2D RDNA WMMA 8x4 kernel for HIP "
            "PaliGemma ViT QKV projection. This only affects hip_probe "
            "return/profile paths and remains opt-in.",
            3);
    visitor(paligemma_vit_hip_attn_out_wmma2d,
            "paligemma_vit_hip_attn_out_wmma2d", false,
            "Experimental: use the grouped-2D RDNA WMMA 8x4 kernel for HIP "
            "PaliGemma ViT attention-output projection. This only affects "
            "hip_probe return/profile paths and remains opt-in.",
            3);
    visitor(paligemma_vit_hip_mlp_up_wmma2d,
            "paligemma_vit_hip_mlp_up_wmma2d", false,
            "Experimental: use the grouped-2D RDNA WMMA 8x4 kernel for HIP "
            "PaliGemma ViT MLP up projection. This only affects hip_probe "
            "return/profile paths and remains opt-in.",
            3);
    visitor(paligemma_vit_hip_mlp_down_wmma8,
            "paligemma_vit_hip_mlp_down_wmma8", false,
            "Experimental: use the grouped-N RDNA WMMA kernel for HIP "
            "PaliGemma ViT MLP down projection. This only affects hip_probe "
            "return/profile paths and remains opt-in.",
            3);
    visitor(paligemma_vit_hip_mlp_down_fused_residual,
            "paligemma_vit_hip_mlp_down_fused_residual", false,
            "Experimental: fuse MLP-down bias and residual into the grouped-N "
            "RDNA WMMA store. Only used with "
            "paligemma_vit_hip_mlp_down_wmma8.",
            3);
    visitor(paligemma_vit_hip_mlp_down_wmma_waves,
            "paligemma_vit_hip_mlp_down_wmma_waves", 8,
            "Experimental: grouped-N RDNA WMMA wave count for HIP PaliGemma "
            "ViT MLP down projection. Supported values are 2, 4, 6, 8, 12, "
            "and 16. Only used when paligemma_vit_hip_mlp_down_wmma8 is set.",
            3);
    visitor(paligemma_vit_hip_mlp_solution_cache,
            "paligemma_vit_hip_mlp_solution_cache", std::string(""),
            "Experimental: load a probe-generated rocBLAS MLP solution cache "
            "for HIP PaliGemma ViT return/profile runs. Empty keeps the "
            "rocBLAS default solution selection.",
            3);
    visitor(paligemma_vit_hip_attention_solution_cache,
            "paligemma_vit_hip_attention_solution_cache", std::string(""),
            "Experimental: load a probe-generated rocBLAS attention QK/AV "
            "solution cache for HIP PaliGemma ViT return/profile runs. Empty "
            "keeps the explicit attention solution flags or rocBLAS default.",
            3);

    // Since it is not used in the CLI version, the print_verbosity is set
    // higher than others.
    visitor(port, "port", 8080, "Server port (default: 8080)", 3);
    visitor(model, "model", std::string("gemma3-4b"),
            "Model name for API endpoints (default: gemma3-4b)", 3);

    visitor(prompt, "prompt", std::string(""),
            "Initial prompt for non-interactive mode. When specified, "
            "generates a response and exits.",
            1);
    visitor(prompt_file, "prompt_file", Path(),
            "Path to file containing the prompt for non-interactive mode. When "
            " specified, generates a response and exits.",
            1);

    visitor(
        eot_line, "eot_line", std::string(""),
        "End of turn line. "
        "When you specify this, the prompt will be all lines "
        "before the line where only the given string appears.\n    Default = "
        "When a newline is encountered, that signals the end of the turn.",
        2);
  }

  void CopyTo(RuntimeConfig& runtime_config) const {
    runtime_config.max_generated_tokens = max_generated_tokens;
    runtime_config.prefill_tbatch_size = prefill_tbatch_size;
    runtime_config.decode_qbatch_size = decode_qbatch_size;
    if (prefill_tbatch_size > kMaxBatchSize) {
      HWY_ABORT(
          "prefill_tbatch_size %zu > kMaxBatchSize %zu: specify a "
          "smaller value, or increase kMaxBatchSize.\n",
          prefill_tbatch_size, kMaxBatchSize);
    }
    if (decode_qbatch_size > kMaxBatchSize) {
      HWY_ABORT(
          "decode_qbatch_size %zu > kMaxBatchSize %zu: specify a "
          "smaller value, or increase kMaxBatchSize.\n",
          decode_qbatch_size, kMaxBatchSize);
    }

    runtime_config.temperature = temperature;
    runtime_config.top_k = top_k;

    // This is copied into RuntimeConfig because decoder layer execution is
    // several stack frames below argument parsing and only sees RuntimeConfig.
    runtime_config.force_flash_attention = force_flash_attention;
  }
};

struct ClientArgs : public ArgsBase<ClientArgs> {
  ClientArgs(int argc, char* argv[]) { InitAndParse(argc, argv); }
  ClientArgs() { Init(); };

  std::string host;
  int port;
  std::string api_key;
  std::string model;
  std::string prompt;
  bool interactive;

  template <class Visitor>
  void ForEach(const Visitor& visitor) {
    visitor(host, "host", std::string("localhost"),
            "Server host (default: localhost)");
    visitor(port, "port", 8080,
            "Server port (default: 8080)");
    visitor(api_key, "api_key", std::string(""),
            "Use public API with key (changes host to "
            "generativelanguage.googleapis.com:443)");
    visitor(model, "model", std::string("gemma3-4b"),
            "Model name to use (default: gemma3-4b)");
    visitor(prompt, "prompt", std::string("Hello! How are you?"),
            "Prompt for generation (default: 'Hello! How are you?')");
    visitor(interactive, "interactive", false,
            "Start interactive chat mode (0 = no, 1 = yes)");
  }
};

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_ARGS_H_
