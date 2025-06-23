#pragma once
#include "bmengine/core/core.h"

namespace bmengine {

namespace functions {

void zeros_(const core::Context& ctx, const core::Tensor& x);
void ones_(const core::Context& ctx, const core::Tensor& x);
core::Tensor arange(const core::Context& ctx, int start, int end, int step=1);

void fill(const core::Context& ctx, const core::Tensor& x, float value);
void normal_(const core::Context& ctx, curandGenerator_t& gen, const core::Tensor& x);
}

}