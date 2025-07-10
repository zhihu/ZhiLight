#include "nn/block/block_kernel.h"
#include "nn/quant/int8/quant_kernel.h"
#include <bmengine/logger/std_log_op.hpp>

namespace nn {

template<typename T>
static __global__ void BM_KERNEL(element_add_scale)(
    size_t n, const T* a, const T* b, float scale, T* c, bool scale_residual) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
	if (scale_residual)
	    c[idx] = (a[idx] + b[idx]) * T(scale);
	else
	    c[idx] = a[idx] + b[idx] * T(scale);
    }
}

void element_add_scale_out(
    const core::Context& ctx,
    const core::Tensor& a,
    const core::Tensor& b,
    core::Tensor& c,
    float scale,
    bool scale_residual) {
    BM_ASSERT_EQ(a.shape(), b.shape(), "a,b shape mismatch");
    size_t n = a.numel();
    if (b.dtype() == core::DataType::kInt8) {
        int event_level = ctx.current_layer() == 300 && ctx.rank() == 0  ? 0 : 2;
        core::EventScope event_scope(ctx, "element_add_scale_out", event_level);
        BM_ASSERT(b.quant_scale, "No scale for quantized tensor");
        BM_ASSERT_EQ(scale, 1.0f, "support scale");
        core::Tensor out = int8_op::dequant_group_fuse_add(ctx, b, *b.quant_scale, a);
        c = out;
        return;
    }

    BM_ASSERT_EQ(a.dtype(), b.dtype(), "type mismatch");
    int threads = round_up(min((size_t) 1024, n), 32);
    int blocks = round_up(n, threads) / threads;
    dim3 gridDim(blocks, 1, 1);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(a.dtype(), {
        BM_KERNEL(element_add_scale)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
        n, a.data<scalar_t>(), b.data<scalar_t>(), scale, c.data<scalar_t>(), scale_residual);
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

core::Tensor element_add_scale(
    const core::Context& ctx, const core::Tensor& a, const core::Tensor& b, float scale, bool scale_residual) {
    core::Tensor ret = ctx.tensor(a.size(), a.dtype());
    element_add_scale_out(ctx, a, b, ret, scale, scale_residual);
    return ret;
}
}
