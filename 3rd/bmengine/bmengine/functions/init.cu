#include "bmengine/functions/init.h"

namespace bmengine {

namespace functions {

// gridDim(n / 1024, 1, 1), blockDim(1024, 1, 1)
template<typename T>
__global__ void KERNEL_fill(size_t n, T *x, T value) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = value;
    }
}

template<typename T>
__global__ void BM_KERNEL(fill_ones)(size_t n, T* x) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = T(1.);
    }
}

template<typename T>
static __global__ void BM_KERNEL(convert_fp16)(
    size_t n, const float* __restrict__ a, T* __restrict__ b) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < n) {
        b[pos] = T(a[pos]);
    }
}

void zeros_(const core::Context& ctx, const core::Tensor& x) {
    BM_ASSERT(ctx.active_device() == x.device(), "Tensor not on current device");
    size_t n = x.nbytes();
    BM_CUDART_ASSERT(cudaMemsetAsync(x.data(), 0, n, ctx.current_stream()->ptr));
}

void ones_(const core::Context& ctx, const core::Tensor& x) {
    BM_ASSERT(ctx.active_device() == x.device(), "Tensor not on current device");
    size_t n = x.numel();
    int threads = min(round_up(n, 32), (size_t) 1024);
    dim3 gridDim(round_up(n, threads) / threads, 1, 1);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH(x.dtype(), {
        /* CUDA 11.0 doesn't support __half on host
        BM_KERNEL(fill)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            n,
            x.data<scalar_t>(),
            scalar_t(1)
        );
        */
        BM_KERNEL(fill_ones)<scalar_t><<<gridDim, blockDim, 0, stream>>>(n, x.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

void normal_(const core::Context& ctx, curandGenerator_t& gen, const core::Tensor& x) {
    BM_ASSERT(ctx.active_device() == x.device(), "Tensor not on current device");
    // BM_ASSERT(core::DataType::kHalf == x.dtype(), "Only fp16 tensor supported");
    size_t n = x.numel();
    int threads = min((size_t) 1024, round_up(n, 32));
    dim3 gridDim(round_up(n, threads) / threads, 1, 1);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    auto temp = ctx.tensor(x.size(), core::DataType::kFloat);
    CURAND_CHECK(curandGenerateNormal(gen, temp.data<float>(), temp.numel(), 0, 1.0 / sqrtf(n)));
    BM_DTYPE_DISPATCH_FLOAT(x.dtype(), {
        BM_KERNEL(convert_fp16)<<<gridDim, blockDim, 0, stream>>>(
            n, temp.data<float>(), x.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

void fill(const core::Context& ctx, const core::Tensor& x, float value) {
    const size_t n = x.numel();
    size_t threads = 1024;
    size_t blocks = round_up(n, threads) / threads;
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(x.dtype(), {
        KERNEL_fill<<<blocks, threads, 0, stream>>>(
        n, x.data<scalar_t>(), scalar_t(value));
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}
}

}