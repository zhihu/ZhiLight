#include "bmengine/functions/index_select.h"
#include "bmengine/logger/std_log_op.hpp"
#include "private/tensor_ops.h"
#include <assert.h>

namespace bmengine {
namespace functions {

typedef unsigned int DimT;

// gridDim (N), blockDim (1024)
template<typename T>
static __global__ void KERNEL_scatter_update_dim0(
    T* __restrict__ dst,                // (X, D)
    const int* __restrict__ dst_index,  // (N)
    const T* __restrict__ src,          // (Y, D)
    const int* __restrict__ src_index,  // (N)
    DimT X,
    DimT Y,
    DimT D
) {
    DimT x = dst_index[blockIdx.x];
    DimT y = src_index ? src_index[blockIdx.x] : blockIdx.x;
    assert(x < X);
    assert(y < Y);

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        size_t dst_offset = x * D + d;
        size_t src_offset = y * D + d;
        dst[dst_offset] = src[src_offset];
    }
}

void scatter_update_dim0(
    const core::Context& ctx,
    core::Tensor& dst,
    const core::Tensor& dst_index,
    const core::Tensor& src,
    const core::Tensor& src_index
) {
    BM_ASSERT_EQ(dst.dtype(), src.dtype(), "dtype mismatch");
    BM_ASSERT_EQ(dst.ndim(), 2, "dst is not 2D");
    BM_ASSERT_EQ(src.ndim(), 2, "src is not 2D");
    BM_ASSERT_EQ(dst.size(-1), src.size(-1), "src and dst have different last dim");
    BM_ASSERT(dst.numel() < 2147483647, "src too big");
    BM_ASSERT(src.numel() < 2147483647, "src too big");

    BM_ASSERT_EQ(dst_index.dtype(), core::DataType::kInt32, "dst_index is not int");
    BM_ASSERT_EQ(dst_index.ndim(), 1, "dst_index is not 1D");
    BM_ASSERT_LE(dst_index.numel(), dst.size(0), "dst_index element number is out of range");
    if (!src_index.empty()) {
        BM_ASSERT_EQ(src_index.dtype(), core::DataType::kInt32, "src_index is not int");
        BM_ASSERT_EQ(src_index.ndim(), 1, "src_index is not 1D");
        BM_ASSERT_EQ(src_index.numel(), dst_index.numel(), "src_index and dst_index have different number");
        // src_index is NOT unique. don't need this assert.
        // BM_ASSERT_LE(src_index.numel(), src.size(0), "src_index element number is out of range");
    }

    DimT D = src.size(-1);
    DimT num_threads = round_up_thread(D);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(dst.dtype(), {
        KERNEL_scatter_update_dim0<scalar_t><<<dst_index.numel(), num_threads, 0, stream>>>(
            dst.mutable_data<scalar_t>(),
            dst_index.data<int>(),
            src.data<scalar_t>(),
            src_index.data<int>(),
            dst.size(0),
            src.size(0),
            D
        );
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}


}
}
