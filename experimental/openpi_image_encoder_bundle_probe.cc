// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr size_t kBatch = 1;
constexpr size_t kImageH = 224;
constexpr size_t kImageW = 224;
constexpr size_t kChannels = 3;
constexpr size_t kPatch = 14;
constexpr size_t kGrid = 16;
constexpr size_t kTokens = 256;
constexpr size_t kWidth = 1152;
constexpr size_t kHeads = 16;
constexpr size_t kHeadDim = 72;
constexpr size_t kMlpDim = 4304;
constexpr size_t kDecoderWidth = 2048;

struct Args {
  std::string bundle;
  bool strict = true;
  bool block00_forward = false;
  bool all_layers_forward = false;
  double stem_tol = 1e-4;
  double posemb_tol = 1e-4;
  double head_tol = 0.25;
  double residual_tol = 0.75;
  double block00_forward_tol = 0.75;
  double all_layers_forward_tol = 3.0;
};

struct Metrics {
  double max_abs = 0.0;
  double rms = 0.0;
  double mean_abs = 0.0;
};

struct BlockForwardResult {
  std::vector<float> sa;
  std::vector<float> plus_sa;
  std::vector<float> mlp;
  std::vector<float> plus_mlp;
};

bool ParseBool(const char* value, bool& out) {
  if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0) {
    out = true;
    return true;
  }
  if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0) {
    out = false;
    return true;
  }
  return false;
}

bool ParseDouble(const char* value, double& out) {
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || *end != '\0') return false;
  out = parsed;
  return true;
}

bool ParseArgs(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--bundle" && i + 1 < argc) {
      args.bundle = argv[++i];
      continue;
    }
    if (arg == "--strict" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.strict)) return false;
      continue;
    }
    if (arg == "--block00_forward" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.block00_forward)) return false;
      continue;
    }
    if (arg == "--all_layers_forward" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.all_layers_forward)) return false;
      continue;
    }
    if (arg == "--stem_tol" && i + 1 < argc) {
      if (!ParseDouble(argv[++i], args.stem_tol)) return false;
      continue;
    }
    if (arg == "--posemb_tol" && i + 1 < argc) {
      if (!ParseDouble(argv[++i], args.posemb_tol)) return false;
      continue;
    }
    if (arg == "--head_tol" && i + 1 < argc) {
      if (!ParseDouble(argv[++i], args.head_tol)) return false;
      continue;
    }
    if (arg == "--residual_tol" && i + 1 < argc) {
      if (!ParseDouble(argv[++i], args.residual_tol)) return false;
      continue;
    }
    if (arg == "--block00_forward_tol" && i + 1 < argc) {
      if (!ParseDouble(argv[++i], args.block00_forward_tol)) return false;
      continue;
    }
    if (arg == "--all_layers_forward_tol" && i + 1 < argc) {
      if (!ParseDouble(argv[++i], args.all_layers_forward_tol)) return false;
      continue;
    }
    std::fprintf(stderr, "unknown or incomplete argument: %s\n", arg.c_str());
    return false;
  }
  return !args.bundle.empty();
}

std::string JoinPath(const std::string& dir, const std::string& file) {
  if (dir.empty() || dir.back() == '/') return dir + file;
  return dir + "/" + file;
}

bool LoadF32(const std::string& path, size_t count, std::vector<float>& out) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    std::fprintf(stderr, "failed to open %s\n", path.c_str());
    return false;
  }
  const std::streamsize bytes = stream.tellg();
  const std::streamsize expected =
      static_cast<std::streamsize>(count * sizeof(float));
  if (bytes != expected) {
    std::fprintf(stderr, "unexpected size for %s: got %lld expected %lld\n",
                 path.c_str(), static_cast<long long>(bytes),
                 static_cast<long long>(expected));
    return false;
  }
  stream.seekg(0, std::ios::beg);
  out.resize(count);
  if (!stream.read(reinterpret_cast<char*>(out.data()), bytes)) {
    std::fprintf(stderr, "failed to read %s\n", path.c_str());
    return false;
  }
  return true;
}

Metrics Compare(const std::vector<float>& actual,
                const std::vector<float>& expected) {
  Metrics metrics;
  double sum_sq = 0.0;
  double sum_abs = 0.0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const double diff = static_cast<double>(actual[i]) - expected[i];
    const double abs_diff = std::abs(diff);
    metrics.max_abs = std::max(metrics.max_abs, abs_diff);
    sum_sq += diff * diff;
    sum_abs += abs_diff;
  }
  metrics.rms = std::sqrt(sum_sq / actual.size());
  metrics.mean_abs = sum_abs / actual.size();
  return metrics;
}

float RoundBfloat16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7FFFu + lsb;
  bits &= 0xFFFF0000u;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void RoundVectorBfloat16(std::vector<float>& values) {
  for (float& value : values) {
    value = RoundBfloat16(value);
  }
}

void PrintMetrics(const char* name, const Metrics& metrics) {
  std::printf("%s max_abs=%.9g rms=%.9g mean_abs=%.9g\n", name,
              metrics.max_abs, metrics.rms, metrics.mean_abs);
}

bool CheckTolerance(const char* name, const Metrics& metrics, double tol,
                    bool strict) {
  if (strict && metrics.max_abs > tol) {
    std::fprintf(stderr, "%s max_abs %.9g exceeds tolerance %.9g\n", name,
                 metrics.max_abs, tol);
    return false;
  }
  return true;
}

size_t ImageIndex(size_t y, size_t x, size_t c) {
  return ((y * kImageW + x) * kChannels) + c;
}

size_t KernelIndex(size_t py, size_t px, size_t c, size_t out) {
  return (((py * kPatch + px) * kChannels + c) * kWidth) + out;
}

size_t StemIndex(size_t gy, size_t gx, size_t out) {
  return ((gy * kGrid + gx) * kWidth) + out;
}

std::vector<float> ComputePatchEmbed(const std::vector<float>& image,
                                     const std::vector<float>& kernel,
                                     const std::vector<float>& bias) {
  std::vector<float> stem(kTokens * kWidth);
  for (size_t gy = 0; gy < kGrid; ++gy) {
    for (size_t gx = 0; gx < kGrid; ++gx) {
      for (size_t out = 0; out < kWidth; ++out) {
        double acc = bias[out];
        for (size_t py = 0; py < kPatch; ++py) {
          for (size_t px = 0; px < kPatch; ++px) {
            for (size_t c = 0; c < kChannels; ++c) {
              acc += static_cast<double>(
                         image[ImageIndex(gy * kPatch + py, gx * kPatch + px,
                                          c)]) *
                     kernel[KernelIndex(py, px, c, out)];
            }
          }
        }
        stem[StemIndex(gy, gx, out)] = static_cast<float>(acc);
      }
    }
  }
  return stem;
}

std::vector<float> AddPosemb(const std::vector<float>& stem,
                             const std::vector<float>& pos_embedding) {
  std::vector<float> out(stem.size());
  for (size_t i = 0; i < stem.size(); ++i) {
    out[i] = stem[i] + pos_embedding[i];
  }
  return out;
}

std::vector<float> CastVectorBfloat16(const std::vector<float>& values) {
  std::vector<float> out(values);
  RoundVectorBfloat16(out);
  return out;
}

std::vector<float> AddVectors(const std::vector<float>& a,
                              const std::vector<float>& b) {
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = a[i] + b[i];
  }
  return out;
}

std::vector<float> LayerNormBfloat16(const std::vector<float>& x,
                                     const std::vector<float>& scale,
                                     const std::vector<float>& bias) {
  std::vector<float> y(x.size());
  for (size_t token = 0; token < kTokens; ++token) {
    const size_t base = token * kWidth;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (size_t d = 0; d < kWidth; ++d) {
      const float value = x[base + d];
      sum += value;
      sum_sq += value * value;
    }
    const float mean = sum / static_cast<float>(kWidth);
    float variance = sum_sq / static_cast<float>(kWidth) - mean * mean;
    variance = std::max(variance, 0.0f);
    const float inv = 1.0f / std::sqrt(variance + 1e-6f);
    for (size_t d = 0; d < kWidth; ++d) {
      y[base + d] = RoundBfloat16((x[base + d] - mean) * inv * scale[d] +
                                  bias[d]);
    }
  }
  return y;
}

std::vector<float> DenseBfloat16(const std::vector<float>& x, size_t rows,
                                 size_t in_dim, const std::vector<float>& kernel,
                                 const std::vector<float>& bias,
                                 size_t out_dim) {
  std::vector<float> y(rows * out_dim);
  for (size_t row = 0; row < rows; ++row) {
    float* out = y.data() + row * out_dim;
    std::copy(bias.begin(), bias.begin() + out_dim, out);
    for (size_t in = 0; in < in_dim; ++in) {
      const float value = x[row * in_dim + in];
      const float* weights = kernel.data() + in * out_dim;
      for (size_t out_col = 0; out_col < out_dim; ++out_col) {
        out[out_col] += value * weights[out_col];
      }
    }
    for (size_t out_col = 0; out_col < out_dim; ++out_col) {
      out[out_col] = RoundBfloat16(out[out_col]);
    }
  }
  return y;
}

std::vector<float> AttentionBfloat16(const std::vector<float>& query,
                                     const std::vector<float>& key,
                                     const std::vector<float>& value) {
  std::vector<float> context(kTokens * kHeads * kHeadDim);
  std::vector<float> logits(kTokens);
  std::vector<float> probs(kTokens);
  const float scale = RoundBfloat16(1.0f / std::sqrt(static_cast<float>(kHeadDim)));
  for (size_t head = 0; head < kHeads; ++head) {
    for (size_t q_token = 0; q_token < kTokens; ++q_token) {
      float max_logit = -std::numeric_limits<float>::infinity();
      for (size_t k_token = 0; k_token < kTokens; ++k_token) {
        float acc = 0.0f;
        for (size_t d = 0; d < kHeadDim; ++d) {
          const size_t q_index = (q_token * kHeads + head) * kHeadDim + d;
          const size_t k_index = (k_token * kHeads + head) * kHeadDim + d;
          acc += RoundBfloat16(query[q_index] * scale) * key[k_index];
        }
        logits[k_token] = acc;
        max_logit = std::max(max_logit, acc);
      }
      float denom = 0.0f;
      for (size_t k_token = 0; k_token < kTokens; ++k_token) {
        const float prob = std::exp(logits[k_token] - max_logit);
        probs[k_token] = prob;
        denom += prob;
      }
      for (size_t k_token = 0; k_token < kTokens; ++k_token) {
        probs[k_token] = RoundBfloat16(probs[k_token] / denom);
      }
      for (size_t d = 0; d < kHeadDim; ++d) {
        float acc = 0.0f;
        for (size_t k_token = 0; k_token < kTokens; ++k_token) {
          const size_t v_index = (k_token * kHeads + head) * kHeadDim + d;
          acc += probs[k_token] * value[v_index];
        }
        const size_t out_index = (q_token * kHeads + head) * kHeadDim + d;
        context[out_index] = RoundBfloat16(acc);
      }
    }
  }
  return context;
}

float GeluApprox(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  return 0.5f * x *
         (1.0f + std::tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
}

void ApplyGeluBfloat16(std::vector<float>& values) {
  for (float& value : values) {
    value = RoundBfloat16(GeluApprox(value));
  }
}

Metrics CheckHeadSamples(const std::vector<float>& encoded,
                         const std::vector<float>& head_kernel,
                         const std::vector<float>& head_bias,
                         const std::vector<float>& golden_tokens) {
  const size_t columns[] = {0, 1, 127, 1024, 2047};
  double sum_sq = 0.0;
  double sum_abs = 0.0;
  size_t count = 0;
  Metrics metrics;
  for (size_t token = 0; token < kTokens; ++token) {
    for (const size_t col : columns) {
      double acc = head_bias[col];
      for (size_t d = 0; d < kWidth; ++d) {
        acc += static_cast<double>(encoded[token * kWidth + d]) *
               head_kernel[d * kDecoderWidth + col];
      }
      const double expected = golden_tokens[token * kDecoderWidth + col];
      const double diff = acc - expected;
      const double abs_diff = std::abs(diff);
      metrics.max_abs = std::max(metrics.max_abs, abs_diff);
      sum_sq += diff * diff;
      sum_abs += abs_diff;
      ++count;
    }
  }
  metrics.rms = std::sqrt(sum_sq / count);
  metrics.mean_abs = sum_abs / count;
  return metrics;
}

std::vector<float> SliceLayer(const std::vector<float>& stacked,
                              size_t layer, size_t layer_size) {
  const size_t offset = layer * layer_size;
  return std::vector<float>(stacked.begin() + offset,
                            stacked.begin() + offset + layer_size);
}

BlockForwardResult RunBlockBfloat16(
    const std::vector<float>& input, size_t layer,
    const std::vector<float>& block_ln0_scale,
    const std::vector<float>& block_ln0_bias,
    const std::vector<float>& block_q_kernel,
    const std::vector<float>& block_q_bias,
    const std::vector<float>& block_k_kernel,
    const std::vector<float>& block_k_bias,
    const std::vector<float>& block_v_kernel,
    const std::vector<float>& block_v_bias,
    const std::vector<float>& block_attn_out_kernel,
    const std::vector<float>& block_attn_out_bias,
    const std::vector<float>& block_ln1_scale,
    const std::vector<float>& block_ln1_bias,
    const std::vector<float>& block_mlp_up_kernel,
    const std::vector<float>& block_mlp_up_bias,
    const std::vector<float>& block_mlp_down_kernel,
    const std::vector<float>& block_mlp_down_bias) {
  BlockForwardResult result;
  const std::vector<float> ln0 = LayerNormBfloat16(
      input, SliceLayer(block_ln0_scale, layer, kWidth),
      SliceLayer(block_ln0_bias, layer, kWidth));
  const std::vector<float> q = DenseBfloat16(
      ln0, kTokens, kWidth,
      SliceLayer(block_q_kernel, layer, kWidth * kHeads * kHeadDim),
      SliceLayer(block_q_bias, layer, kHeads * kHeadDim), kHeads * kHeadDim);
  const std::vector<float> k = DenseBfloat16(
      ln0, kTokens, kWidth,
      SliceLayer(block_k_kernel, layer, kWidth * kHeads * kHeadDim),
      SliceLayer(block_k_bias, layer, kHeads * kHeadDim), kHeads * kHeadDim);
  const std::vector<float> v = DenseBfloat16(
      ln0, kTokens, kWidth,
      SliceLayer(block_v_kernel, layer, kWidth * kHeads * kHeadDim),
      SliceLayer(block_v_bias, layer, kHeads * kHeadDim), kHeads * kHeadDim);
  const std::vector<float> attn = AttentionBfloat16(q, k, v);
  result.sa = DenseBfloat16(
      attn, kTokens, kHeads * kHeadDim,
      SliceLayer(block_attn_out_kernel, layer, kHeads * kHeadDim * kWidth),
      SliceLayer(block_attn_out_bias, layer, kWidth), kWidth);

  result.plus_sa = AddVectors(input, result.sa);
  RoundVectorBfloat16(result.plus_sa);

  const std::vector<float> ln1 = LayerNormBfloat16(
      result.plus_sa, SliceLayer(block_ln1_scale, layer, kWidth),
      SliceLayer(block_ln1_bias, layer, kWidth));
  std::vector<float> mlp_up = DenseBfloat16(
      ln1, kTokens, kWidth,
      SliceLayer(block_mlp_up_kernel, layer, kWidth * kMlpDim),
      SliceLayer(block_mlp_up_bias, layer, kMlpDim), kMlpDim);
  ApplyGeluBfloat16(mlp_up);
  result.mlp = DenseBfloat16(
      mlp_up, kTokens, kMlpDim,
      SliceLayer(block_mlp_down_kernel, layer, kMlpDim * kWidth),
      SliceLayer(block_mlp_down_bias, layer, kWidth), kWidth);

  result.plus_mlp = AddVectors(result.plus_sa, result.mlp);
  RoundVectorBfloat16(result.plus_mlp);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, args)) {
    std::fprintf(stderr,
                 "Usage: %s --bundle DIR [--strict 0|1] [--stem_tol F] "
                 "[--posemb_tol F] [--head_tol F] [--residual_tol F] "
                 "[--block00_forward 0|1] [--block00_forward_tol F] "
                 "[--all_layers_forward 0|1] [--all_layers_forward_tol F]\n",
                 argv[0]);
    return 2;
  }

  std::vector<float> image;
  std::vector<float> patch_kernel;
  std::vector<float> patch_bias;
  std::vector<float> pos_embedding;
  std::vector<float> golden_stem;
  std::vector<float> golden_posemb;
  std::vector<float> encoded;
  std::vector<float> head_kernel;
  std::vector<float> head_bias;
  std::vector<float> golden_tokens;
  std::vector<float> block00_sa;
  std::vector<float> block00_plus_sa;
  std::vector<float> block00_mlp;
  std::vector<float> block00_plus_mlp;
  std::vector<float> block_ln0_scale;
  std::vector<float> block_ln0_bias;
  std::vector<float> block_q_kernel;
  std::vector<float> block_q_bias;
  std::vector<float> block_k_kernel;
  std::vector<float> block_k_bias;
  std::vector<float> block_v_kernel;
  std::vector<float> block_v_bias;
  std::vector<float> block_attn_out_kernel;
  std::vector<float> block_attn_out_bias;
  std::vector<float> block_ln1_scale;
  std::vector<float> block_ln1_bias;
  std::vector<float> block_mlp_up_kernel;
  std::vector<float> block_mlp_up_bias;
  std::vector<float> block_mlp_down_kernel;
  std::vector<float> block_mlp_down_bias;
  std::vector<float> encoder_norm_scale;
  std::vector<float> encoder_norm_bias;

  bool ok = true;
  ok &= LoadF32(JoinPath(args.bundle, "golden_image.f32"),
                kBatch * kImageH * kImageW * kChannels, image);
  ok &= LoadF32(JoinPath(args.bundle, "weight_patch_embed_kernel.f32"),
                kPatch * kPatch * kChannels * kWidth, patch_kernel);
  ok &= LoadF32(JoinPath(args.bundle, "weight_patch_embed_bias.f32"), kWidth,
                patch_bias);
  ok &= LoadF32(JoinPath(args.bundle, "weight_pos_embedding.f32"),
                kTokens * kWidth, pos_embedding);
  ok &= LoadF32(JoinPath(args.bundle, "golden_intermediate_stem.f32"),
                kTokens * kWidth, golden_stem);
  ok &= LoadF32(JoinPath(args.bundle, "golden_intermediate_with_posemb.f32"),
                kTokens * kWidth, golden_posemb);
  ok &= LoadF32(JoinPath(args.bundle, "golden_intermediate_encoded.f32"),
                kTokens * kWidth, encoded);
  ok &= LoadF32(JoinPath(args.bundle, "weight_head_kernel.f32"),
                kWidth * kDecoderWidth, head_kernel);
  ok &= LoadF32(JoinPath(args.bundle, "weight_head_bias.f32"), kDecoderWidth,
                head_bias);
  ok &= LoadF32(JoinPath(args.bundle, "golden_tokens.f32"),
                kTokens * kDecoderWidth, golden_tokens);
  ok &= LoadF32(JoinPath(args.bundle, "golden_intermediate_encoder_block00_sa.f32"),
                kTokens * kWidth, block00_sa);
  ok &= LoadF32(
      JoinPath(args.bundle, "golden_intermediate_encoder_block00_plus_sa.f32"),
      kTokens * kWidth, block00_plus_sa);
  ok &= LoadF32(JoinPath(args.bundle, "golden_intermediate_encoder_block00_mlp.f32"),
                kTokens * kWidth, block00_mlp);
  ok &= LoadF32(
      JoinPath(args.bundle, "golden_intermediate_encoder_block00_plus_mlp.f32"),
      kTokens * kWidth, block00_plus_mlp);
  if (args.block00_forward || args.all_layers_forward) {
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_ln0_scale.f32"),
                  27 * kWidth, block_ln0_scale);
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_ln0_bias.f32"),
                  27 * kWidth, block_ln0_bias);
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_q_kernel.f32"),
                  27 * kWidth * kHeads * kHeadDim, block_q_kernel);
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_q_bias.f32"),
                  27 * kHeads * kHeadDim, block_q_bias);
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_k_kernel.f32"),
                  27 * kWidth * kHeads * kHeadDim, block_k_kernel);
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_k_bias.f32"),
                  27 * kHeads * kHeadDim, block_k_bias);
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_v_kernel.f32"),
                  27 * kWidth * kHeads * kHeadDim, block_v_kernel);
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_v_bias.f32"),
                  27 * kHeads * kHeadDim, block_v_bias);
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_attn_out_kernel.f32"),
                  27 * kHeads * kHeadDim * kWidth, block_attn_out_kernel);
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_attn_out_bias.f32"),
                  27 * kWidth, block_attn_out_bias);
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_ln1_scale.f32"),
                  27 * kWidth, block_ln1_scale);
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_ln1_bias.f32"),
                  27 * kWidth, block_ln1_bias);
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_mlp_up_kernel.f32"),
                  27 * kWidth * kMlpDim, block_mlp_up_kernel);
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_mlp_up_bias.f32"),
                  27 * kMlpDim, block_mlp_up_bias);
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_mlp_down_kernel.f32"),
                  27 * kMlpDim * kWidth, block_mlp_down_kernel);
    ok &= LoadF32(JoinPath(args.bundle, "weight_block_mlp_down_bias.f32"),
                  27 * kWidth, block_mlp_down_bias);
  }
  if (args.all_layers_forward) {
    ok &= LoadF32(JoinPath(args.bundle, "weight_encoder_norm_scale.f32"),
                  kWidth, encoder_norm_scale);
    ok &= LoadF32(JoinPath(args.bundle, "weight_encoder_norm_bias.f32"),
                  kWidth, encoder_norm_bias);
  }
  if (!ok) return 1;

  const std::vector<float> stem =
      ComputePatchEmbed(image, patch_kernel, patch_bias);
  const Metrics stem_metrics = Compare(stem, golden_stem);
  PrintMetrics("patch_embed_vs_intermediate_stem", stem_metrics);
  ok &= CheckTolerance("patch_embed_vs_intermediate_stem", stem_metrics,
                       args.stem_tol, args.strict);

  const std::vector<float> with_posemb = AddPosemb(stem, pos_embedding);
  const Metrics posemb_metrics = Compare(with_posemb, golden_posemb);
  PrintMetrics("posemb_vs_intermediate_with_posemb", posemb_metrics);
  ok &= CheckTolerance("posemb_vs_intermediate_with_posemb", posemb_metrics,
                       args.posemb_tol, args.strict);

  const Metrics block00_plus_sa_metrics =
      Compare(AddVectors(with_posemb, block00_sa), block00_plus_sa);
  PrintMetrics("block00_with_posemb_plus_sa_residual",
               block00_plus_sa_metrics);
  ok &= CheckTolerance("block00_with_posemb_plus_sa_residual",
                       block00_plus_sa_metrics, args.residual_tol,
                       args.strict);

  const Metrics block00_plus_mlp_metrics =
      Compare(AddVectors(block00_plus_sa, block00_mlp), block00_plus_mlp);
  PrintMetrics("block00_plus_sa_plus_mlp_residual",
               block00_plus_mlp_metrics);
  ok &= CheckTolerance("block00_plus_sa_plus_mlp_residual",
                       block00_plus_mlp_metrics, args.residual_tol,
                       args.strict);

  if (args.block00_forward) {
    std::printf("running block00 CPU reference forward\n");
    const std::vector<float> x0 = CastVectorBfloat16(with_posemb);
    const BlockForwardResult block00 = RunBlockBfloat16(
        x0, 0, block_ln0_scale, block_ln0_bias, block_q_kernel, block_q_bias,
        block_k_kernel, block_k_bias, block_v_kernel, block_v_bias,
        block_attn_out_kernel, block_attn_out_bias, block_ln1_scale,
        block_ln1_bias, block_mlp_up_kernel, block_mlp_up_bias,
        block_mlp_down_kernel, block_mlp_down_bias);

    const Metrics sa_forward_metrics = Compare(block00.sa, block00_sa);
    PrintMetrics("block00_forward_sa_vs_golden", sa_forward_metrics);
    ok &= CheckTolerance("block00_forward_sa_vs_golden", sa_forward_metrics,
                         args.block00_forward_tol, args.strict);

    const Metrics plus_sa_forward_metrics =
        Compare(block00.plus_sa, block00_plus_sa);
    PrintMetrics("block00_forward_plus_sa_vs_golden",
                 plus_sa_forward_metrics);
    ok &= CheckTolerance("block00_forward_plus_sa_vs_golden",
                         plus_sa_forward_metrics, args.block00_forward_tol,
                         args.strict);

    const Metrics mlp_forward_metrics = Compare(block00.mlp, block00_mlp);
    PrintMetrics("block00_forward_mlp_vs_golden", mlp_forward_metrics);
    ok &= CheckTolerance("block00_forward_mlp_vs_golden", mlp_forward_metrics,
                         args.block00_forward_tol, args.strict);

    const Metrics plus_mlp_forward_metrics =
        Compare(block00.plus_mlp, block00_plus_mlp);
    PrintMetrics("block00_forward_plus_mlp_vs_golden",
                 plus_mlp_forward_metrics);
    ok &= CheckTolerance("block00_forward_plus_mlp_vs_golden",
                         plus_mlp_forward_metrics, args.block00_forward_tol,
                         args.strict);
  }

  if (args.all_layers_forward) {
    std::printf("running 27-layer CPU reference forward\n");
    std::vector<float> x = CastVectorBfloat16(with_posemb);
    for (size_t layer = 0; layer < 27; ++layer) {
      const BlockForwardResult block = RunBlockBfloat16(
          x, layer, block_ln0_scale, block_ln0_bias, block_q_kernel,
          block_q_bias, block_k_kernel, block_k_bias, block_v_kernel,
          block_v_bias, block_attn_out_kernel, block_attn_out_bias,
          block_ln1_scale, block_ln1_bias, block_mlp_up_kernel,
          block_mlp_up_bias, block_mlp_down_kernel, block_mlp_down_bias);
      x = block.plus_mlp;
    }
    const std::vector<float> encoded_forward =
        LayerNormBfloat16(x, encoder_norm_scale, encoder_norm_bias);
    const Metrics encoded_forward_metrics =
        Compare(encoded_forward, encoded);
    PrintMetrics("all_layers_forward_encoded_vs_golden",
                 encoded_forward_metrics);
    ok &= CheckTolerance("all_layers_forward_encoded_vs_golden",
                         encoded_forward_metrics,
                         args.all_layers_forward_tol, args.strict);

    const std::vector<float> tokens_forward = DenseBfloat16(
        encoded_forward, kTokens, kWidth, head_kernel, head_bias,
        kDecoderWidth);
    const Metrics tokens_forward_metrics =
        Compare(tokens_forward, golden_tokens);
    PrintMetrics("all_layers_forward_tokens_vs_golden",
                 tokens_forward_metrics);
    ok &= CheckTolerance("all_layers_forward_tokens_vs_golden",
                         tokens_forward_metrics,
                         args.all_layers_forward_tol, args.strict);
  }

  const Metrics head_metrics =
      CheckHeadSamples(encoded, head_kernel, head_bias, golden_tokens);
  PrintMetrics("head_sample_f32_vs_tokens", head_metrics);
  ok &=
      CheckTolerance("head_sample_f32_vs_tokens", head_metrics, args.head_tol,
                     args.strict);

  std::printf("bundle %s\n", args.bundle.c_str());
  std::printf("strict %d\n", args.strict ? 1 : 0);
  return ok ? 0 : 1;
}
