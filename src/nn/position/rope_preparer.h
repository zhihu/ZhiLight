#pragma once
#include <bmengine/core/core.h>
#include "model/model_config.hpp"

namespace model {
class ModelContext;
}

namespace nn {
using namespace bmengine;

class RopePreparer : public core::Layer {
    BM_LAYER_DEF( RopePreparer)

    RopePreparer(const core::Context& ctx, model::ModelConfig block_config);

    // return cos, sin
    std::tuple<core::Tensor, core::Tensor> forward(
        const core::Context& ctx,
        const core::Tensor& tokens,      // (seq_len)
        const core::Tensor& position_ids // (seq_len)
    );
};

}
