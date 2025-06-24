#pragma once

#include <bmengine/core/core.h>

namespace nn::fp8 {

using namespace bmengine;

core::Tensor cvt_half_to_fp8(const core::Context& ctx, const core::Tensor& input, float scale, int round_up=32);

core::Tensor cvt_fp8_to_half(const core::Context& ctx, const core::Tensor& input, float scale);

core::Tensor calc_scale(
    const core::Context& ctx,
    const core::Tensor& input,
    float MAX_E4M3=448
);

core::Tensor dynamic_scaled_quant(
    const core::Context& ctx,
    const core::Tensor& input,
    float MAX_E4M3=448
);

core::Tensor per_token_cast_to_fp8(
    const core::Context& ctx,
    const core::Tensor& input,
    bool scale_col_major=true,
    float MAX_E4M3=448
);

core::Tensor dequant_fp8_block_weight(
    const core::Context& ctx,
    const core::Tensor& input,
    const core::Tensor& scale,
    core::DataType out_type
);

}
