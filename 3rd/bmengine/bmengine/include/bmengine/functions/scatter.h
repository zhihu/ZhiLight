#pragma once
#include "bmengine/core/core.h"

namespace bmengine {
namespace functions {

void scatter_update_dim0(
    const core::Context& ctx,
    core::Tensor& dst,
    const core::Tensor& dst_index,
    const core::Tensor& src,
    const core::Tensor& src_index
);
}  // namespace functions
}  // namespace bmengine
