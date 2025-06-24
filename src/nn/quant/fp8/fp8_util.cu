// Author: Gaojunmin@zhihu.com

#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>
#include "bmengine/functions/reduce.cuh"
#include <bmengine/logger/std_log_op.hpp>
#include <assert.h>

#include <cstdint>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <assert.h>
#include "fp8.h"


__device__ __forceinline__ uint16_t half2_to_e4m3(const uint32_t a)
{
    uint16_t val;
#if __CUDA_ARCH__ >= 890
    asm volatile("{ cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;}\n" : "=h"(val) : "r"(a));
#else
    assert(false);
#endif
    return val;
}

__device__ __forceinline__ uint32_t e4m3_to_half2(const uint16_t a)
{
    uint32_t val;
#if __CUDA_ARCH__ >= 890
    asm volatile("{ cvt.rn.f16x2.e4m3x2 %0, %1;}\n" : "=r"(val) : "h"(a));
#else
    assert(false);
#endif
    return val;
}

// (N/1024/2), 1024
__global__ void KERNEL_cvt_half_fp8(const half2 *__restrict__ in, // (N/2)
                                    uint16_t *__restrict__ out,   // (N/2)
                                    uint32_t N,
                                    float scale) {
    uint32_t n_2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n_2 < N / 2) {
        half2 scale_h2 = __float2half2_rn(scale);
        half2 h2 = in[n_2];
        half2 h2s = __hmul2(h2, scale_h2);
        out[n_2] = half2_to_e4m3(*(uint32_t*) &h2s);
    }
}

// (N/1024/2), 1024
template<bool IS_BF16=false>
__global__ void T_KERNEL_cvt_half_fp8(const half2 *__restrict__ in, // (N/2)
                                      uint16_t *__restrict__ out,   // (N/2)
                                      uint32_t N,
                                      float* scale_ptr) {
    uint32_t n_2 = blockIdx.x * blockDim.x + threadIdx.x;
    nv_bfloat16 bf2[2];
    if (n_2 < N / 2) {
        float scale = 1.f / *scale_ptr;
        half2 h2s;
        if constexpr(IS_BF16) {
            *(half2*)&bf2 = in[n_2];
            h2s = __floats2half2_rn(scale * (float)bf2[0], scale * (float)bf2[1]);
        } else {
            half2 scale_h2 = __float2half2_rn(scale);
            half2 h2 = in[n_2];
            h2s = __hmul2(h2, scale_h2);
        }
        out[n_2] = half2_to_e4m3(*(uint32_t*) &h2s);
    }
}

// (N/1024/2), 1024
__global__ void KERNEL_cvt_fp8_half(const uint16_t *__restrict__ in, // (N/2)
                                    half2 *__restrict__ out,         // (N/2)
                                    uint32_t N,
                                    float scale) {
    uint32_t n_2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n_2 < N / 2) {
        half2 scale_h2 = __float2half2_rn(scale);
        uint16_t fp8x2 = in[n_2];
        uint32_t h2 = e4m3_to_half2(fp8x2);
        out[n_2] = __hmul2(*(half2*) &h2, scale_h2);
    }
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = (value >= 0)
          ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
          : __uint_as_float(
            atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

// Compute the absolute maximum m of the input tensor and store
// m / float8_e4m3::max() in *scale. Each thread block performs a
// reduction tree and the memory in scale is atomically updated.
// So to get the right answer, *scale needs to be initialized to
// a value <= 0.0 and we need to wait for all thread blocks to
// finish before consuming *scale.
template <typename scalar_t>
__global__ void segmented_max_reduction(float* __restrict__ scale,
                                        const scalar_t* __restrict__ input,
                                        int64_t num_elems,
                                        float MAX_E4M3=448) {
    __shared__ float cache[1024];
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // First store maximum for all values processes by
    // the current thread in cache[threadIdx.x]
    scalar_t tmp = 0.0;
    while (i < num_elems) {
        float x = static_cast<float>(input[i]);
        tmp = max(tmp, fabs(x));
        i += blockDim.x * gridDim.x;
    }
    cache[threadIdx.x] = tmp;

    __syncthreads();

    // Now perform parallel reduction within the thread block
    int ib = blockDim.x / 2;
    while (ib != 0) {
        if (threadIdx.x < ib && cache[threadIdx.x + ib] > cache[threadIdx.x]) {
            cache[threadIdx.x] = cache[threadIdx.x + ib];
        }
        __syncthreads();
        ib /= 2;
    }
    // Finally, since cache[0] contains the maximum for this thread block,
    // atomically write the max to the target location
    if (threadIdx.x == 0) {
        atomicMaxFloat(scale, cache[0] / MAX_E4M3);
    }
}

namespace nn::fp8 {

using namespace bmengine;

core::Tensor cvt_half_to_fp8(const core::Context& ctx, const core::Tensor& input, float scale, int round_up_m) {
    BM_ASSERT_EQ(input.dtype(), core::DataType::kHalf, "");
    core::Tensor fp8_out = ctx.tensor(input.shape(), core::DataType::kInt8, "", round_up_m * input.size(-1));
    auto stream = ctx.current_stream()->ptr;
    KERNEL_cvt_half_fp8<<<round_up(input.numel() / 2, 1024) / 1024, 1024, 0, stream>>>(
        input.data<half2>(),
        fp8_out.mutable_data<uint16_t>(),
        input.numel(),
        scale
    );
    BM_CUDART_ASSERT(cudaGetLastError());
    return fp8_out;
}

core::Tensor cvt_fp8_to_half(const core::Context& ctx, const core::Tensor& input, float scale) {
    core::Tensor half_out = ctx.tensor(input.shape(), core::DataType::kHalf);
    auto stream = ctx.current_stream()->ptr;
    KERNEL_cvt_fp8_half<<<round_up(input.numel() / 2, 1024) / 1024, 1024, 0, stream>>>(
        input.data<uint16_t>(),
        half_out.mutable_data<half2>(),
        input.numel(),
        scale
    );
    BM_CUDART_ASSERT(cudaGetLastError());
    return half_out;
}

core::Tensor calc_scale(
    const core::Context& ctx,
    const core::Tensor& input,
    float MAX_E4M3
) {
    const cudaStream_t stream = ctx.current_stream()->ptr;
    core::Tensor scale = ctx.tensor({1}, core::DataType::kFloat);  // [1]
    BM_CUDART_ASSERT(cudaMemsetAsync(scale.data(), 0, sizeof(float), stream));

    int64_t num_tokens = input.numel() / input.size(-1);
    int64_t num_elems = input.numel();
    dim3 grid(num_tokens);
    dim3 block(1024);
    BM_DTYPE_DISPATCH_HALF(input.dtype(), {
        segmented_max_reduction<scalar_t><<<grid, block, 0, stream>>>(
            scale.data<float>(), input.data<scalar_t>(), num_elems, MAX_E4M3);
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return scale;
}

core::Tensor dynamic_scaled_quant(
    const core::Context& ctx,
    const core::Tensor& input,
    float MAX_E4M3
) {
    const cudaStream_t stream = ctx.current_stream()->ptr;
    size_t round_up_size = std::max(32 * input.size(-1), 1024UL);
    core::Tensor out = ctx.tensor(input.shape(), core::DataType::kInt8, "", round_up_size);
    core::Tensor scale = calc_scale(ctx, input, MAX_E4M3);

    dim3 block(1024);
    dim3 grid(round_up(input.numel() / 2, 1024) / 1024);
    if (input.dtype() == core::DataType::kHalf) {
        T_KERNEL_cvt_half_fp8<false><<<grid, block, 0, stream>>>(
            input.data<half2>(),
            out.mutable_data<uint16_t>(),
            input.numel(),
            scale.data<float>()
        );
    } else {
        T_KERNEL_cvt_half_fp8<true><<<grid, block, 0, stream>>>(
            input.data<half2>(),
            out.mutable_data<uint16_t>(),
            input.numel(),
            scale.data<float>()
        );
    }
    BM_CUDART_ASSERT(cudaGetLastError());

    out.quant_scale = std::make_shared<core::Tensor>();
    *out.quant_scale = scale;
    return out;
}

// (m, n/128), 32
template<class HALF_T>
__global__ void KERNEL_per_token_cast_to_fp8(
    const HALF_T *__restrict__ g_in, // (m, n)
    uint8_t *__restrict__ g_out,     // (m, n)
    uint32_t aligned_m,
    float* scale_ptr,                // (n/128, aligned_m)
    bool scale_col_major
) {
    uint32_t N = gridDim.y * 128;
    // each thread process 4 x half
    HALF_T in_h[4];
    float in_f[4];
    // read 4 x half
    size_t offset = blockIdx.x * N + blockIdx.y * 128 + threadIdx.x * 4;
    *(double*)&in_h = *(const double*)(g_in + offset);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        in_f[i] = in_h[i];
    }
    // amax
    float amax = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        amax = fmaxf(amax, fabs(in_f[i]));
    }
    amax = bmengine::functions::warpReduceMaxB(amax);
    if (amax < 1e-4)
        amax = 1e-4; // clamp

    // convert
    __nv_fp8_e4m3 out_fp8[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        out_fp8[i] = __nv_fp8_e4m3(in_f[i] * (448.0f / amax));
    }
    // write 4 X fp8
    *(int*)(g_out + offset) = *(int*)out_fp8;

    size_t offset_scale = scale_col_major
        ? (blockIdx.y * aligned_m + blockIdx.x)
        : (blockIdx.x * gridDim.y + blockIdx.y);
    if (threadIdx.x == 0)
        scale_ptr[offset_scale] = amax / 448.0f;
}

// Python code:
//def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
//    assert x.dim() == 2 and x.size(1) % 128 == 0
//    m, n = x.shape
//    x_view = x.view(m, -1, 128)
//    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
//    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)
core::Tensor per_token_cast_to_fp8(
    const core::Context& ctx,
    const core::Tensor& input,
    bool scale_col_major,
    float MAX_E4M3
) {
    const cudaStream_t stream = ctx.current_stream()->ptr;
    BM_ASSERT_EQ(input.ndim(), 2, "FP8 block_scale: input is not 2D");
    BM_ASSERT_EQ(input.size(1) % 128, 0, "FP8 block_scale: input.size(1) can't divide 128");

    size_t m = input.size(0);
    size_t n = input.size(1);
    size_t round_up_size = std::max(32 * input.size(-1), 1024UL);
    core::Tensor out = ctx.tensor(input.shape(), core::DataType::kFP8_E4M3, "", round_up_size);

    size_t aligned_m = round_up(m, 16 / sizeof(float));
    core::Tensor scale = scale_col_major
        ? ctx.tensor({n / 128UL, aligned_m}, core::DataType::kFloat)
        : ctx.tensor({aligned_m, n / 128UL}, core::DataType::kFloat);

    dim3 block(1024);
    dim3 grid(m, n / 128);
    BM_DTYPE_DISPATCH_HALF(input.dtype(), {
        KERNEL_per_token_cast_to_fp8<scalar_t><<<grid, 32, 0, stream>>>(
            input.data<scalar_t>(),
            out.data<uint8_t>(),
            aligned_m,
            scale.mutable_data<float>(),
            scale_col_major);
    });
    BM_CUDART_ASSERT(cudaGetLastError());

    out.set_quant_scale(scale);
    return out;
}

// (m, n/128), 32
template<class HALF_T>
__global__ void KERNEL_dequant_fp8_block(
    const uint8_t *__restrict__ g_in, // (m, n)
    HALF_T *__restrict__ g_out,       // (m, n)
    const float* scale_ptr, // round_up(m, 128) / 128, round_up(n, 128) / 128
    const uint32_t N,
    uint32_t stride_scale
) {
    uint32_t n = blockIdx.y * 128 + threadIdx.x * 4;
    if (n >= N) return;

    // each thread process 4 x fp8
    __nv_fp8_e4m3 in[4];
    // read 4 x fp8
    size_t offset = blockIdx.x * N + n;
    *(uint32_t*)&in = *(const uint32_t*)(g_in + offset);

    size_t offset_scale = (blockIdx.x / 128) * stride_scale + blockIdx.y;
    float scale = scale_ptr[offset_scale];

    // convert
    HALF_T out[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        out[i] = float(in[i]) * scale;
//        if (blockIdx.x == 9082 && blockIdx.y == 0 && threadIdx.x == 0) {
//            printf("in=%.5f, scale=%.5f, out=%.5f", float(in[i]), scale, float(out[i]));
//            assert(!isinf(float(out[i])));
//        }
    }
    // write 4 X fp8
    *(double *)(g_out + offset) = *(double*)out;
}

core::Tensor dequant_fp8_block_weight(
    const core::Context& ctx,
    const core::Tensor& weight,
    const core::Tensor& scale, // inv
    core::DataType out_type
) {
    BM_ASSERT_EQ(scale.dtype(), core::DataType::kFloat, "");
    BM_ASSERT_EQ(weight.ndim(), 2, "weight is not 2d");
    BM_ASSERT_EQ(scale.ndim(), 2, "weight is not 2d");
    BM_ASSERT_LE(weight.size(0), scale.size(0) * 128, "weight and scale dim0 mismatch");
    BM_ASSERT_LE(weight.size(1), scale.size(1) * 128, "weight and scale dim0 mismatch");

    core::Tensor out = ctx.tensor(weight.shape(), out_type);
    cudaStream_t stream = ctx.current_stream()->ptr;
    dim3 gridDim(weight.size(0), scale.size(1));
    BM_DTYPE_DISPATCH_HALF(out_type, {
        KERNEL_dequant_fp8_block<scalar_t><<<gridDim, 32, 0, stream>>>(
            weight.data<uint8_t>(),
            out.mutable_data<scalar_t>(),
            scale.data<float>(),
            weight.size(1),
            scale.size(1));
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return out;
}

} // namespace nn::fp8