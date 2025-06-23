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

}
