#pragma once
#include "bmengine/core/core.h"

namespace bmengine {

namespace functions {

class BinaryElementwiseOp : public core::Layer {
    BM_LAYER_DEF(BinaryElementwiseOp)

    enum Op { Add, Sub, Mul, Div, Max };

    BinaryElementwiseOp(const core::Context& ctx, Op op);

    core::Tensor forward(const core::Context& ctx, const core::Tensor& x, const core::Tensor& y, core::Tensor* out = nullptr);

    void inplace(const core::Context& ctx, const core::Tensor& x, const core::Tensor& y);

    core::Tensor broadcast_y(const core::Context& ctx, const core::Tensor& x, const core::Tensor& y);
};

void check_numeric(const core::Context& ctx, const core::Tensor& tensor);
void check_equal(const core::Context& ctx, const core::Tensor& a, const core::Tensor& b);

core::Tensor divide(const core::Context& ctx, const core::Tensor& a, float divisor);

core::Tensor pow(const core::Context& ctx, const core::Tensor& a, float exp);

core::Tensor clamp(const core::Context& ctx, const core::Tensor& a, float min, float max);

core::Tensor sigmoid(const core::Context& ctx, const core::Tensor& tensor);

} // namespace functions

} // namespace bmengine
