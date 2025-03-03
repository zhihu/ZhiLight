#include "nn/position/rope_preparer.h"
#include "nn/position/rope_common.cuh"
#include "nn/position/rotary_embedding.h"
#include "bmengine/functions/index_select.h"
#include "bmengine/functions/utils.cuh"
#include "bmengine/functions/transpose.h"
#include "bmengine/logger/std_log_op.hpp"
#include <numeric>
#include <assert.h>

namespace nn {

using namespace bmengine;
using bmengine::core::DataType;
using bmengine::core::Tensor;

class RopePreparer::impl {
public:
    class NormalImpl;
    class Llama3Impl;
    class YarnImpl;

    int dim_head;
    float rope_theta;
    std::string type;
    float scaling_factor;
    int max_position_embeddings;
    bool neox_style = true;

    impl(const core::Context& ctx, model::ModelConfig cfg)
        : dim_head(cfg.dim_head),
          rope_theta(cfg.rope_theta),
          type(cfg.rope_cfg.type),
          scaling_factor(cfg.rope_cfg.factor),
          max_position_embeddings(cfg.max_position_embeddings),
          neox_style(cfg.rope_cfg.neox_style) {
        if (cfg.qk_rope_head_dim > 0)
            dim_head = cfg.qk_rope_head_dim;
    }
    virtual ~impl() {}

    virtual std::tuple<core::Tensor, core::Tensor> compute_cos_sin(
        const core::Context& ctx,
        const core::Tensor& pos // (batch, seq_len)
        ) = 0;
};

// gridDim (seq_len),   blockDim (dim_head)
static __global__ void KERNEL_rope_cos_sin(
    const int32_t* __restrict__ pos, // (seq_len)
    float* __restrict__ g_cos,       // (seq_len, dim_head)
    float* __restrict__ g_sin,       // (seq_len, dim_head)
    float base,
    float scaling_factor = 1,
    bool neox_style=true) {
    int m = pos[blockIdx.x];
    int dim_head = blockDim.x;
    int half_dim = dim_head / 2;
    int col = threadIdx.x;
    // 0 ~ half_dim
    int i = get_half_dim_index(col, half_dim, neox_style);

    float inv_freq = powf(base, -float(i * 2) / dim_head);

    float freq = m * inv_freq;
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    g_cos[offset] = cos(freq);
    g_sin[offset] = sin(freq);
}

class RopePreparer::impl::NormalImpl : public RopePreparer::impl {
public:
    NormalImpl(const core::Context& ctx, model::ModelConfig cfg) : impl(ctx, cfg) {}

    std::tuple<core::Tensor, core::Tensor> compute_cos_sin(
        const core::Context& ctx,
        const core::Tensor& pos // (batch, seq_len)
    ) override {
        auto shape = pos.shape();
        shape.push_back(dim_head);
        Tensor cos = ctx.tensor(shape, DataType::kFloat);
        Tensor sin = ctx.tensor(shape, DataType::kFloat);

        auto stream = ctx.current_stream()->ptr;
        KERNEL_rope_cos_sin<<<pos.numel(), dim_head, 0, stream>>>(
            pos.data<int>(),
            cos.mutable_data<float>(),
            sin.mutable_data<float>(),
            rope_theta,
            scaling_factor,
            neox_style
        );
        BM_CUDART_ASSERT(cudaGetLastError());
        return {cos, sin};
    }
};
// Original llama3 implementation:
//   https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py
//   def _compute_llama3_parameters(...):
//    # Gets the default RoPE parameters
//    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len, **rope_kwargs)
//
//    factor = config.rope_scaling["factor"]  # `8` in the original implementation
//    low_freq_factor = config.rope_scaling["low_freq_factor"]  # `1` in the original implementation
//    high_freq_factor = config.rope_scaling["high_freq_factor"]  # `4` in the original implementation
//    old_context_len = config.rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation
//
//    low_freq_wavelen = old_context_len / low_freq_factor
//    high_freq_wavelen = old_context_len / high_freq_factor
//
//    wavelen = 2 * math.pi / inv_freq
//    # wavelen < high_freq_wavelen: do nothing
//    # wavelen > low_freq_wavelen: divide by factor
//    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
//    # otherwise: interpolate between the two, using a smooth factor
//    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
//    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
//    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
//    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
//
//    return inv_freq_llama, attention_factor

// gridDim (seq_len),   blockDim (dim_head)
static __global__ void KERNEL_rope_cos_sin_llama3(
    const int32_t* __restrict__ pos, // (seq_len)
    float* __restrict__ g_cos,       // (seq_len, dim_head)
    float* __restrict__ g_sin,       // (seq_len, dim_head)
    float base,
    float factor,
    float low_freq_factor,
    float high_freq_factor,
    float old_context_len,
    bool neox_style=true) {
    int m = pos[blockIdx.x];
    int dim_head = blockDim.x;
    int half_dim = dim_head / 2;
    int col = threadIdx.x;
    // 0 ~ half_dim
    int i = get_half_dim_index(col, half_dim, neox_style);
    float inv_freq = powf(base, -float(i * 2) / dim_head);

    float low_freq_wavelen = old_context_len / low_freq_factor;
    float high_freq_wavelen = old_context_len / high_freq_factor;

    float pi = 3.141592653589793;
    float wavelen = 2.f * pi / inv_freq;
    if (wavelen < high_freq_wavelen) {
        // wavelen < high_freq_wavelen: do nothing
    } else if (wavelen > low_freq_wavelen) {
        inv_freq = inv_freq / factor;  // divide by factor
    } else {
        // otherwise: interpolate between the two, using a smooth factor
        float smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
        inv_freq = (1.f - smooth_factor) * inv_freq / factor + smooth_factor * inv_freq;
    }

    float freq = m * inv_freq;
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    g_cos[offset] = cos(freq);
    g_sin[offset] = sin(freq);
}

class RopePreparer::impl::Llama3Impl : public RopePreparer::impl {
    float low_freq_factor;
    float high_freq_factor;
    float old_context_len;
public:
    Llama3Impl(const core::Context& ctx, model::ModelConfig cfg) : impl(ctx, cfg) {
        low_freq_factor = cfg.rope_cfg.low_freq_factor;
        high_freq_factor = cfg.rope_cfg.high_freq_factor;
        old_context_len = cfg.rope_cfg.original_max_position;
    }

    std::tuple<core::Tensor, core::Tensor> compute_cos_sin(
        const core::Context& ctx,
        const core::Tensor& pos // (batch, seq_len)
    ) override {
        auto shape = pos.shape();
        shape.push_back(dim_head);
        Tensor cos = ctx.tensor(shape, DataType::kFloat);
        Tensor sin = ctx.tensor(shape, DataType::kFloat);

        auto stream = ctx.current_stream()->ptr;
        KERNEL_rope_cos_sin_llama3<<<pos.numel(), dim_head, 0, stream>>>(
            pos.data<int>(),
            cos.mutable_data<float>(),
            sin.mutable_data<float>(),
            rope_theta,
            scaling_factor,
            low_freq_factor,
            high_freq_factor,
            old_context_len
        );
        BM_CUDART_ASSERT(cudaGetLastError());
        return {cos, sin};
    }
};

RopePreparer::RopePreparer(const core::Context& ctx, model::ModelConfig cfg) {
    // std::cout << "RopePreparer " << cfg.rope_cfg.type << "\n";
    if (cfg.rope_cfg.type == "") {
        pimpl = std::make_unique<impl::NormalImpl>(ctx, cfg);
    } else if (cfg.rope_cfg.type == "llama3") {
        pimpl = std::make_unique<impl::Llama3Impl>(ctx, cfg);
    } else {
        throw std::runtime_error("RopePreparer: Not implemented rope type: " + cfg.rope_cfg.type);
    }
};

RopePreparer::~RopePreparer() { }

std::tuple<core::Tensor, core::Tensor> RopePreparer::forward(
    const core::Context& ctx,
    const core::Tensor& tokens,
    const core::Tensor& pos // (seq_len)
) {
    core::EventScope ev(ctx, "RopePreparer", 3);
    auto ret = pimpl->compute_cos_sin(ctx, pos);
    // Compare llama3 with python implementation:
    // from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    // from transformers import AutoConfig
    // cfg = AutoConfig.from_pretrained('Llama-3.2-1B-Instruct/')
    // inv_freq = ROPE_INIT_FUNCTIONS["llama3"](cfg, None)[0]
    // cos, sin = inv_freq.cos(), inv_freq.sin()
//    if (tokens.size(0) > 1 && ctx.rank() == 0) {
//        auto [cos, sin] = ret;
//        size_t dim = cos.size(-1);
//        std::cout << "cos: " << cos.slice_dim0_len(1, 1).view({dim}).slice_dim0_len(0, dim / 2) << endl;
//        std::cout << "sin: " << sin.slice_dim0_len(1, 1).view({dim}).slice_dim0_len(0, dim / 2) << endl;
//    }
    return ret;
}

}