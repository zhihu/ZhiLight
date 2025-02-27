#pragma once
#include "model/model_config.hpp"
#include <bmengine/core/core.h>
#include <string>
#include <tuple>

namespace nn {
using namespace bmengine;

class Linear;

class FeedForward : public core::Layer {
BM_LAYER_DEF(FeedForward);

    FeedForward(
        const core::Context& ctx,
        model::ModelConfig block_config,
        model::QuantConfig quant_config,
        bool parallel);

    core::Tensor forward(const core::Context& ctx, const core::Tensor& inp);

    const Linear& w_out() const;

    void load_state_dict(
        const core::Context& ctx,
        const std::map<std::string, const core::Tensor>& state_dict,
        const std::string& prefix,
        bool allow_missing);

    void dequant_cache_weight(core::Context& ctx, const core::Tensor& fake_input);
};

core::Tensor gate_fuse(
    const core::Context& ctx,
    const core::Tensor& input,
    const std::string& act_fn_type
);

std::tuple<core::Tensor, core::Tensor> top_k_softmax(
    const core::Context& ctx,
    const core::Tensor& input,
    const core::Tensor& worker_load,
    int k,
    int k_ext,
    bool norm_topk_prob,
    float weight_scale,
    const std::string& scoring_func
);

std::tuple<core::Tensor, core::Tensor> group_topk_softmax(
    const core::Context& ctx,
    const core::Tensor& input,
    const core::Tensor& score_correction_bias,
    const core::Tensor& worker_load,
    int num_group,
    int topk_group,
    int top_k,
    int top_k_ext,
    bool norm_topk_prob,
    float weight_scale,
    const std::string& scoring_func
);

core::Tensor sum_experts(
    const core::Context& ctx,
    const core::Tensor& input, // (seq_len * k, dim_model)
    const core::Tensor& index, // (seq_len * k)
    const core::Tensor& weights // (seq_len, k)
); // return (seq_len, dim_model)

core::Tensor sum_experts(
    const core::Context& ctx,
    std::vector<core::Tensor> inputs, // m => (0~seq_len, dim_model)
    const core::Tensor& concat_inputs,
    const core::Tensor& experts, // (seq_len * k)
    const core::Tensor& index, // (seq_len * k)
    const core::Tensor& weights, // (seq_len, k)
    bool exp_parallel,
    int world_size = 0,
    int local_rank = 0
); // return (seq_len, dim_model)

void route_shared_lb(
    const core::Context& ctx,
    core::Tensor& exp_ids,
    core::Tensor& exp_weights,
    core::Tensor& worker_load,
    int top_k,
    int num_local_experts);
}
