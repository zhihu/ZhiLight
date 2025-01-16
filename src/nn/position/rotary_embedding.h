#pragma once
#include <bmengine/core/core.h>
#include "model/model_config.hpp"

namespace nn {
using namespace bmengine;

class RotaryEmbedding : public core::Layer {
    BM_LAYER_DEF(RotaryEmbedding)

    RotaryEmbedding(const core::Context& ctx, model::ModelConfig block_config);

    std::tuple<core::Tensor, core::Tensor> forward(
        const core::Context& ctx,
        const core::Tensor& pos, // (batch, seq_len)
        const core::Tensor& q,   // (batch, seq_len, dim_model)
        const core::Tensor& k    // (batch, seq_len, dim_model)
    );

    core::Tensor rotate(
        const core::Context& ctx,
        const core::Tensor& pos, // (batch, seq_len)
        const core::Tensor& q,   // (batch, seq_len, dim_model)
        core::Tensor* output = nullptr
    );

    void rotate_inplace(
        const core::Context& ctx,
        const core::Tensor& pos, // (batch, seq_len)
        core::Tensor& q          // (batch, seq_len, dim_model)
    );

    bool is_normal() const;
    bool is_neox_style() const;
};

// fuse kernel for attention
void rotary_embedding_qk(
    const core::Context& ctx,
    const core::Tensor& pos,     // (seq_len)
    const core::Tensor& in,      // (seq_len, all_num_heads * dim_head)
    core::Tensor& out_q,         // (seq_len, num_heads * dim_head)
    core::Tensor& out_k,         // (seq_len, num_kv_heads * dim_head)
    core::Tensor& out_v,         // (seq_len, num_kv_heads * dim_head)
    size_t num_heads,
    size_t num_kv_heads,
    size_t dim_head,
    float rope_theta,
    core::DataType dtype
);

void rope_qk_cache(
    const core::Context& ctx,
    const core::Tensor& cos, // (seq_len, dim_head)
    const core::Tensor& sin, // (seq_len, dim_head)
    const core::Tensor& in,  // (seq_len, all_num_heads * dim_head)
    core::Tensor& out_q,     // (seq_len, num_heads * dim_head)
    core::Tensor& out_k,     // (seq_len, num_kv_heads * dim_head)
    core::Tensor& out_v,     // (seq_len, num_kv_heads * dim_head)
    size_t num_heads,
    size_t num_kv_heads,
    size_t dim_head,
    core::DataType dtype,
    bool neox_style
);
}
