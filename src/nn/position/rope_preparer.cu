#include "nn/position/rope_preparer.h"
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
    int i;
    if (neox_style) {
        i = (col < half_dim) ? col : (col - half_dim); // = col % half_dim
    } else {
        i = col / 2;
    }

    float inv_freq = powf(base, -float(i * 2) / dim_head);

    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;

    float freq = m * inv_freq;
    g_cos[offset] = cos(freq);
    g_sin[offset] = sin(freq);
}

class RopePreparer::impl {
public:
    class NormalImpl;
    class DynamicNTKImpl;
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

RopePreparer:: RopePreparer(const core::Context& ctx, model::ModelConfig cfg) {
    if (cfg.rope_cfg.type == "") {
        pimpl = std::make_unique<impl::NormalImpl>(ctx, cfg);
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
    return pimpl->compute_cos_sin(ctx, pos);
}

}