#pragma once
#include <tuple>
#include "bmengine/core/core.h"

namespace bmengine {
namespace functions {

std::pair<core::Tensor, core::Tensor> sort_pair_1d(
    const core::Context& ctx,
    const core::Tensor& keys,
    const core::Tensor& values,
    int max_key=0);

std::pair<core::Tensor, core::Tensor> sort_with_indices_1d(
    const core::Context& ctx,
    const core::Tensor& keys,
    int max_key=0);
}
}
