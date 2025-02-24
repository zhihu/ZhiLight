// Author: Gaojunmin@zhihu.com

#include "nn/quant/gptq/gptq.h"
#include "model/model_context.h"
#include "nn/quant/fp8/fp8.h"
#include "nn/quant/int8/quant_kernel.h"
#include "utils/env.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>
#include <bmengine/logger/std_log_op.hpp>
#include <bmengine/functions/reduce.cuh>
#include <assert.h>

#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "qdq_4.cuh"
#include "qdq_util.cuh"
#include "gptq.h"

namespace nn {
namespace gptq {

#define WARP_SIZE 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

using namespace bmengine;

template<int BYTES>
static __forceinline__ __device__ void cp_async(uint32_t smem_dst, const void* src) {
    asm volatile ("cp.async.ca.shared.global [%0], [%1], %2;\n"
        :: "r"(smem_dst), "l"(src), "n"(BYTES));
}

__forceinline__ __device__ void copy_a(void* smem_dst, const void* src_a, const uint32_t K_8) {
#if __CUDA_ARCH__ >= 800
    uint32_t smem_addr = __cvta_generic_to_shared(smem_dst);
    for (uint32_t k_8 = threadIdx.y * blockDim.x + threadIdx.x; k_8 < K_8;) {
        cp_async<16>(smem_addr + k_8 * 16, (char*)src_a + k_8 * 16);
        k_8 += blockDim.x * blockDim.y;
    }
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();
#else
    assert(false);
#endif
}

__forceinline__ __device__ void preprocess_zero(
    const uint32_t zero,
    half2& z1,
    half2& z16
) {
    __half_raw z1r = {uint16_t(0xe400 | zero)}; // half(-1024.0f - zero);
    z1 = __half2half2(z1r);
    __half_raw NEG_64 = {0xd400}; // -64 = -1024 / 16
    z16 = __half2half2(NEG_64 - __int2half_rn(zero));
}

template<uint32_t A, uint32_t B>
static __forceinline__ __device__ uint32_t and_or(uint32_t in) {
    static constexpr uint32_t LUT = (0xf0 & 0xcc) | 0xaa;
    uint32_t res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res)
    : "r"(in), "n"(A), "n"(B), "n"(LUT));
    return res;
//    return in & A | B;
}

__forceinline__ __device__ void dequant_8x4bit (
    const uint32_t q_load,
    half2 (&dq)[4],
    half2 z1,
    half2 z16
) {
    static constexpr uint32_t H2_1024 = 0x64006400; // half2 of 1024

    uint32_t qa = q_load;
    half2_uint32 q0(and_or<0x000f000f, H2_1024>(qa)); // half2( q[0]      + 1024, q[1]      + 1024 )
    half2_uint32 q1(and_or<0x00f000f0, H2_1024>(qa)); // half2( q[2] * 16 + 1024, q[3] * 16 + 1024 )
    qa >>= 8;
    half2_uint32 q2(and_or<0x000f000f, H2_1024>(qa)); // half2( q[4]      + 1024, q[5]      + 1024 )
    half2_uint32 q3(and_or<0x00f000f0, H2_1024>(qa)); // half2( q[6] * 16 + 1024, q[7] * 16 + 1024 )

    static constexpr uint32_t ONE_16TH = 0x2c002c00; // 1/16
    half2 one_16th = half2_uint32(ONE_16TH).as_half2;
    {
        // z1  = z1z16[0] = -(1024 + zero)
        // z16 = z1z16[1] = -(1024/16 + zero) = -(64 + zero)
        dq[0] = __hadd2(q0.as_half2,           z1);  // half2( q[0] - z, q[1] - z )
        dq[1] = __hfma2(q1.as_half2, one_16th, z16);  // half2( q[2] - z, q[3] - z )
        dq[2] = __hadd2(q2.as_half2,           z1);  // half2( q[4] - z, q[5] - z )
        dq[3] = __hfma2(q3.as_half2, one_16th, z16);  // half2( q[6] - z, q[7] - z )
    }
}

__forceinline__ __device__ float dot_8_half(half2(&dq)[4], half2(&a2_ptr)[4]) {
    half2 result = {};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        result = __hfma2(dq[i], a2_ptr[i], result);
    }
    return __half2float(__low2half(result)) + __half2float(__high2half(result));
}

__forceinline__ __device__ float dot_8_float(half2(&dq)[4], half2(&a)[4]) {
    float result = 0.;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        result += __low2float(dq[i]) * __low2float(a[i]);
        result += __high2float(dq[i]) * __high2float(a[i]);
    }
    return result;
}

template <int WRAP_M>
__forceinline__ __device__ void zero_acc(float (&acc)[WRAP_M]) {
    for (int m = 0; m < WRAP_M; m++) {
        acc[m] = 0;
    }
}

template <int WRAP_M, bool SYM>
__forceinline__ __device__ void DEV_gemm_warp_reduce(
    const half* __restrict__ A,                 // (WRAP_M, K)
    const uint32_t* __restrict__ b_q_weight,    // (N, K / 8)
    const uint8_t* __restrict__ b_gptq_qzeros,  // (N, K / group_size)
    const half* __restrict__ b_gptq_scales,     // (N , K / group_size)
    float (&acc_th)[WRAP_M],
    const int M,
    const int N,
    const int K_8,
    const int GROUP_SIZE_8
) {
    const int n = blockIdx.x * blockDim.y + threadIdx.y;

    // slice matrix
    const uint4* __restrict__ A8 =reinterpret_cast<const uint4 *>(A); // uint4 read 8 x half
    const uint32_t* __restrict__ q_weight = b_q_weight + n * K_8;
    const uint8_t* __restrict__ q_zeros = b_gptq_qzeros + n * (K_8 / GROUP_SIZE_8);
    const half* __restrict__ scales = b_gptq_scales + n *  (K_8 / GROUP_SIZE_8);

    half2 z1, z16;
    if constexpr (SYM) {
        preprocess_zero(8, z1, z16);
    }
    const int G_STEP = WARP_SIZE / GROUP_SIZE_8;
    for (int k_8 = threadIdx.x, g_i = k_8 / GROUP_SIZE_8
        ; k_8 < K_8
        ; k_8 += WARP_SIZE, g_i += G_STEP) {
        // load and decode weight
        half scale = scales[g_i];
        if constexpr (!SYM) {
            uint32_t zero = q_zeros[g_i];
            preprocess_zero(zero, z1, z16);
        }

        half2 dq[4];
         uint32_t q_load = __ldcs(&q_weight[k_8]);
        dequant_8x4bit(q_load, dq, z1, z16);

        __align__(16) half2 a8[WRAP_M][4];
#pragma unroll
        for (int m = 0; m < WRAP_M; m++) {
            *reinterpret_cast<uint4 *>(a8[m]) = A8[m * K_8 + k_8];
            acc_th[m] = fma(dot_8_half(dq, a8[m]), scale, acc_th[m]);
        }
    }
}

// (N / BLOCK_N, M / WRAP_M), (32, BLOCK_N)
template <int WRAP_M, bool SYM, class OUT_T=half>
__global__ void KERNEL_gemm_warp_reduce(
    const half* __restrict__ GLOBAL_A,          // (M, K)
    const uint32_t* __restrict__ b_q_weight,    // (N, K / 8)
    const uint8_t* __restrict__ b_gptq_qzeros,  // (N, K / group_size)
    const half* __restrict__ b_gptq_scales,     // (N, K / group_size)
    const half* __restrict__ bias_ptr,          // (N)
    OUT_T* GLOBAL_C,
    const int M,
    const int N,
    const int K_8,
    const int GROUP_SIZE_8,
    const int* __restrict__ expert_ids,         // (1)
    const float* __restrict__ expert_weights,   // (1)
    bool ADD_C=false
) {
    const int offset_m = blockIdx.y * WRAP_M;
    const int n = blockIdx.x * blockDim.y + threadIdx.y;
    if (n >= N) return;
    // assert(K % WARP_K == 0);
    // assert(WARP_K % GROUP_SIZE == 0);

    const half* __restrict__ A = GLOBAL_A + offset_m * K_8 * 8; // offset_m * K
    OUT_T* C = GLOBAL_C + offset_m * N;

    // MOE
    if (expert_ids) {
        int expert_id = expert_ids[0];
        b_q_weight += expert_id * N * K_8;
        b_gptq_qzeros += expert_id * N * K_8 / GROUP_SIZE_8;
        b_gptq_scales += expert_id * N * K_8 / GROUP_SIZE_8;
    }

//    extern __shared__ half smem_fuse_gate_in_a[][8];
//    copy_a(&smem_fuse_gate_in_a[0][0], A, K_8 * WRAP_M);
//    A = &smem_fuse_gate_in_a[0][0];

    float acc_th[WRAP_M];
    zero_acc(acc_th);
    DEV_gemm_warp_reduce<WRAP_M, SYM>(A, b_q_weight, b_gptq_qzeros, b_gptq_scales, acc_th, M, N, K_8, GROUP_SIZE_8);

    __shared__ float acc_s[WRAP_M][32];
    for (int m = 0; m < WRAP_M; m++) {
        float acc = functions::warpReduceSum(acc_th[m]);
        if (threadIdx.x == 0) {
            acc_s[m][threadIdx.y] = acc;
        }
    }
    __syncthreads();
    float alpha = expert_weights ? *expert_weights : 1.;
    // write in wrap 0
    const int n1 = blockIdx.x * blockDim.y + threadIdx.x;
    if (threadIdx.y == 0 && threadIdx.x < blockDim.y && n1 < N) {
        float bias = bias_ptr ? float(bias_ptr[n1]) : 0.;
        for (int m = 0; m < WRAP_M; m++) {
            if (ADD_C)
                C[m * N + n1] = float(C[m * N + n1]) + alpha * acc_s[m][threadIdx.x] + bias;
            else
                C[m * N + n1] = alpha * acc_s[m][threadIdx.x] + bias;
        }
    }
}

static inline __device__ float silu(float x) {
    return x / (1.0 + expf(-x));
}

// (N / BLOCK_N, M, TOP_K), (32, BLOCK_N)
template <bool SYM=false>
__global__ void KERNEL_gemm_moe_up(
    const int REAL_TOP_K,
    const int SHARED_EXP_ID,
    const half* __restrict__ GLOBAL_A,     // (M, K)
    const uint32_t* __restrict__ qweight1, // (N, K / 8)
    const uint8_t* __restrict__ qzeros1,   // (N, K / group_size)
    const half* __restrict__ scales1,      // (N, K / group_size)
    const uint16_t* __restrict__ rev_prm1, // (K)
    const uint32_t* __restrict__ qweight2, // (N, K / 8)
    const uint8_t* __restrict__ qzeros2,   // (N, K / group_size)
    const half* __restrict__ scales2,      // (N, K / group_size)
    const uint16_t* __restrict__ rev_prm2, // (K)
    const uint32_t* expert_ids,            // (M, REAL_TOP_K)
    half* GLOBAL_C,                        // (M, TOP_K, N)
    const uint32_t M,
    const uint32_t N,
    const uint32_t K_8,
    const uint32_t GROUP_SIZE_8,
    bool exp_parallel,
    int world_size,
    int local_rank
) {
    const uint32_t m = blockIdx.y;
    const uint32_t K = K_8 * 8;
    const uint32_t TOP_K = gridDim.z;

    // A
    const half* __restrict__ A = GLOBAL_A + m * K;
    extern __shared__ half smem_fuse_gate_in_a[][8];
    copy_a(&smem_fuse_gate_in_a[0][0], A, K_8);
    A = &smem_fuse_gate_in_a[0][0];
    // B1, B2
    int expert_id;
    if (blockIdx.z < REAL_TOP_K) {
        expert_id = expert_ids[m * REAL_TOP_K + blockIdx.z];
        if (exp_parallel) {
            if ((expert_id % world_size) != local_rank) {
                return; // Skip NOT local experts
            }
            expert_id /= world_size;
        }
    } else {
        expert_id = (SHARED_EXP_ID + blockIdx.z - REAL_TOP_K);
    }
    size_t expert_offset = size_t(expert_id * N) * K_8;
    qweight1 += expert_offset;
    qweight2 += expert_offset;
    expert_offset = size_t(expert_id * N) * (K_8 / GROUP_SIZE_8);
    qzeros1 += expert_offset;
    qzeros2 += expert_offset;
    scales1 += expert_offset;
    scales2 += expert_offset;
    // C
    half* C = GLOBAL_C + m * TOP_K * N + blockIdx.z * N;

    float acc_th1[1];
    float acc_th2[1];
    acc_th1[0] = 0;
    acc_th2[0] = 0;
    DEV_gemm_warp_reduce<1, SYM>(A, qweight1, qzeros1, scales1, acc_th1, M, N, K_8, GROUP_SIZE_8);
    DEV_gemm_warp_reduce<1, SYM>(A, qweight2, qzeros2, scales2, acc_th2, M, N, K_8, GROUP_SIZE_8);

    __shared__ float acc_s[32];
    float acc1 = functions::warpReduceSum(acc_th1[0]);
    float acc2 = functions::warpReduceSum(acc_th2[0]);
    if (threadIdx.x == 0) {
        acc_s[threadIdx.y] = silu(acc1) * acc2;
    }
    __syncthreads();
    // write in wrap 0
    const int n1 = blockIdx.x * blockDim.y + threadIdx.x;
    if (threadIdx.y == 0 && threadIdx.x < blockDim.y && n1 < N) {
        C[n1] = acc_s[threadIdx.x];
    }
}

// (N / BLOCK_N, M), (32, BLOCK_N)
template <bool SYM, class OUT_T=half>
__global__ void KERNEL_gemm_moe_down(
    const int TOP_K,
    const int REAL_TOP_K,
    const int SHARED_EXP_ID,
    const int* __restrict__ expert_ids,         // (M, REAL_TOP_K)
    const float* __restrict__ expert_weights,   // (M, REAL_TOP_K)
    const half* __restrict__ GLOBAL_A,          // (M, TOP_K, K)
    const uint32_t* __restrict__ b_q_weight,    // (EXP, N, K / 8)
    const uint8_t* __restrict__ b_gptq_qzeros,  // (EXP, N, K / group_size)
    const half* __restrict__ b_gptq_scales,     // (EXP, N, K / group_size)
    OUT_T* GLOBAL_C,                            // (M, N)
    const int M,
    const int N,
    const int K_8,
    const int GROUP_SIZE_8,
    bool exp_parallel,
    int world_size,
    int local_rank,
    bool ADD_C=false
) {
    const int m = blockIdx.y;

    const int K = K_8 * 8;
    const half *__restrict__ A = GLOBAL_A + m * TOP_K * K;
    float acc_all = 0;  // sum of top k
    for (int t = 0; t < TOP_K; ++t, A += K) {
        int expert_id;
        if (t < REAL_TOP_K) {
            expert_id = expert_ids[m * REAL_TOP_K + t];
            if (exp_parallel) {
                if ((expert_id % world_size) != local_rank) {
                    continue; // Skip NOT local experts
                }
                expert_id /= world_size;
            }
        } else {
            expert_id = (SHARED_EXP_ID + t - REAL_TOP_K);
        }
        const uint32_t* exp_w = b_q_weight + expert_id * N * K_8;
        const uint8_t* exp_z = b_gptq_qzeros + expert_id * N * K_8 / GROUP_SIZE_8;
        const half* exp_s = b_gptq_scales + expert_id * N * K_8 / GROUP_SIZE_8;

        float acc_th[1];
        acc_th[0] = 0;
        DEV_gemm_warp_reduce<1, SYM>(A, exp_w, exp_z, exp_s, acc_th, M, N, K_8, GROUP_SIZE_8);

        float weight = t < REAL_TOP_K ? expert_weights[m * REAL_TOP_K + t] : 1;
        acc_all += acc_th[0] * weight;
    }

    __shared__ float acc_s[32];
    {
        float acc = functions::warpReduceSum(acc_all);
        if (threadIdx.x == 0) {
            acc_s[threadIdx.y] = acc;
        }
    }
    __syncthreads();
    // write in wrap 0
    const int n1 = blockIdx.x * blockDim.y + threadIdx.x;
    if (threadIdx.y == 0 && threadIdx.x < blockDim.y && n1 < N) {
        OUT_T* C = GLOBAL_C + m * N;
        if (ADD_C)
            C[n1] = float(C[n1]) + acc_s[threadIdx.x];
        else
            C[n1] = acc_s[threadIdx.x];
    }
}

core::Tensor gemm_moe_up(
    const core::Context& ctx,
    const core::Tensor& a,          // (M,  K)
    const core::Tensor& q_weight1,  // (EXP, N, K / 8)
    const core::Tensor& qzeros1,    // (EXP, N, K / group_size)
    const core::Tensor& scales1,    // (EXP, N, K / group_size)
    const core::Tensor& rev_perm1,  // (EXP, K)
    const core::Tensor& q_weight2,  // (EXP, N, K / 8)
    const core::Tensor& qzeros2,    // (EXP, N, K / group_size)
    const core::Tensor& scales2,    // (EXP, N, K / group_size)
    const core::Tensor& rev_perm2,  // (EXP, K)
    bool sym,
    const core::Tensor& expert_ids, // (M, top_k)
    int n_shared_expert,
    bool exp_parallel
) {
    BM_ASSERT_LE(a.ndim(), 2, "Wrong ndim");
    BM_ASSERT_LE(q_weight1.ndim(), 3, "Wrong ndim");
    BM_ASSERT_EQ(q_weight1.shape(), q_weight2.shape(), "in and gate should have same shape.");
    BM_ASSERT(qzeros1.dtype() == core::DataType::kInt8, "qzeros must be int8");
    BM_ASSERT(qzeros2.dtype() == core::DataType::kInt8, "qzeros must be int8");

    const size_t M = a.size(0);
    const size_t K = a.size(1);
    const size_t N = q_weight1.size(-2);
    const size_t GROUP_SIZE = K / scales1.size(-1);
    const size_t TOP_K = expert_ids.size(-1) + n_shared_expert;
    auto stream = ctx.current_stream()->ptr;

    core::Tensor c = ctx.tensor({M, TOP_K, N}, a.dtype());
    functions::zeros_(ctx, c);

    {
        size_t PARALLEL = 8;
        BM_ASSERT(N % PARALLEL == 0, "");
        dim3 gridDim(DIVIDE(N, PARALLEL), M, TOP_K);
        dim3 blockDim(32, PARALLEL);
        int dyn_mem_size = K * sizeof(half);

        auto kernel = sym ? KERNEL_gemm_moe_up<true> : KERNEL_gemm_moe_up<false>;
        BM_CUDART_ASSERT(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_mem_size));

        kernel<<<gridDim, blockDim, dyn_mem_size, stream>>>(
            expert_ids.size(-1),
            q_weight1.size(0) - n_shared_expert,
            a.data<half>(),
            q_weight1.data<uint32_t>(),
            qzeros1.data<uint8_t>(),
            scales1.data<half>(),
            nullptr,
            q_weight2.data<uint32_t>(),
            qzeros2.data<uint8_t>(),
            scales2.data<half>(),
            nullptr,
            expert_ids.data<uint32_t>(),
            c.mutable_data<half>(),
            M,
            N,
            K / 8,
            GROUP_SIZE / 8,
            exp_parallel,
            ctx.world_size(),
            ctx.rank()
        );
        BM_CUDART_ASSERT(cudaGetLastError());
        // functions::check_numeric(ctx, c);
    }
    return c;
}

core::Tensor gemm_moe_down(
    const core::Context& ctx,
    const core::Tensor& a,              // (M, TOP_K, K)
    const core::Tensor& q_weight,       // (NUM_EXP, N, K / 8)
    const core::Tensor& qzeros,         // (NUM_EXP, N, K / group_size)
    const core::Tensor& scales,         // (NUM_EXP, N, K / group_size)
    const core::Tensor& expert_ids,     // (M, TOP_K)
    const core::Tensor& expert_weights, // (M, TOP_K)
    bool sym,
    int n_shared_expert,
    bool exp_parallel,
    core::Tensor* output
) {
    BM_ASSERT_EQ(a.ndim(), 3, "not 3-D");
    BM_ASSERT_EQ(q_weight.ndim(), 3, "not 3-D");
    const size_t M = a.size(0);
    const size_t TOP_K = a.size(1);
    const size_t K = a.size(2);
    const size_t N = q_weight.size(-2);
    const size_t GROUP_SIZE = K / scales.size(-1);
    BM_ASSERT_EQ(TOP_K - n_shared_expert, expert_weights.size(-1), "");
    BM_ASSERT_EQ(expert_ids.shape(), expert_weights.shape(), "");
    BM_ASSERT(expert_ids.dtype() == core::DataType::kInt32, "");
    BM_ASSERT(expert_weights.dtype() == core::DataType::kFloat, "");
    if (output) {
        BM_ASSERT_EQ(output->size(0), M, "");
        BM_ASSERT_EQ(output->size(1), N, "");
    }

    core::Tensor c = output ? *output : ctx.tensor({M, N}, a.dtype());
    if (exp_parallel)
        functions::zeros_(ctx, c);

    auto stream = ctx.current_stream()->ptr;
    size_t PARALLEL = 8;
    BM_ASSERT_EQ(N % PARALLEL, 0, "");
    dim3 gridDim(DIVIDE(N, PARALLEL), M);
    dim3 blockDim(32, PARALLEL);
    auto kernel = sym ? KERNEL_gemm_moe_down<true> : KERNEL_gemm_moe_down<false>;

    kernel<<<gridDim, blockDim, 0, stream>>>(
        TOP_K,
        TOP_K - n_shared_expert,
        q_weight.size(0) - n_shared_expert,
        expert_ids.data<int>(),
        expert_weights.data<float>(),
        a.data<half>(),
        q_weight.data<uint32_t>(),
        qzeros.data<uint8_t>(),
        scales.data<half>(),
        c.mutable_data<half>(),
        M,
        N,
        K / 8,
        GROUP_SIZE / 8,
        exp_parallel,
        ctx.world_size(),
        ctx.rank(),
        output != nullptr
    );
    BM_CUDART_ASSERT(cudaGetLastError());
    // functions::check_numeric(ctx, c);
    return c;
}

// (N / BLOCK_N, M / WRAP_M), (32, BLOCK_N)
template <uint32_t WRAP_M, bool SYM=false>
__global__ void KERNEL_gemm_fuse_gate_in(
    const half* __restrict__ GLOBAL_A,     // (M, K)
    const uint32_t* __restrict__ qweight1, // (N, K / 8)
    const uint8_t* __restrict__ qzeros1,   // (N, K / group_size)
    const half* __restrict__ scales1,      // (N, K / group_size)
    const uint16_t* __restrict__ rev_prm1, // (K)
    const uint32_t* __restrict__ qweight2, // (N, K / 8)
    const uint8_t* __restrict__ qzeros2,   // (N, K / group_size)
    const half* __restrict__ scales2,      // (N, K / group_size)
    const uint16_t* __restrict__ rev_prm2, // (K)
    half* GLOBAL_C,
    const uint32_t M,
    const uint32_t N,
    const uint32_t K_8,
    const uint32_t GROUP_SIZE_8
) {
    const uint32_t offset_m = blockIdx.y * WRAP_M;

    const half* __restrict__ A = GLOBAL_A + offset_m * K_8 * 8; // offset_m * K
    half* C = GLOBAL_C + offset_m * N;

    extern __shared__ half smem_fuse_gate_in_a[][8];
    copy_a(&smem_fuse_gate_in_a[0][0], A, K_8 * WRAP_M);
    A = &smem_fuse_gate_in_a[0][0];

    float acc_th1[WRAP_M];
    float acc_th2[WRAP_M];
    zero_acc(acc_th1);
    zero_acc(acc_th2);
    DEV_gemm_warp_reduce<WRAP_M, SYM>(A, qweight1, qzeros1, scales1, acc_th1, M, N, K_8, GROUP_SIZE_8);
    DEV_gemm_warp_reduce<WRAP_M, SYM>(A, qweight2, qzeros2, scales2, acc_th2, M, N, K_8, GROUP_SIZE_8);

    __shared__ float acc_s[WRAP_M][32];
    for (int m = 0; m < WRAP_M; m++) {
        float acc1 = functions::warpReduceSum(acc_th1[m]);
        float acc2 = functions::warpReduceSum(acc_th2[m]);
        if (threadIdx.x == 0) {
            acc_s[m][threadIdx.y] = silu(acc1) * acc2;
        }
    }
    __syncthreads();
    // write in wrap 0
    const int n1 = blockIdx.x * blockDim.y + threadIdx.x;
    if (threadIdx.y == 0 && threadIdx.x < blockDim.y && n1 < N) {
        for (int m = 0; m < WRAP_M; m++) {
            C[m * N + n1] = acc_s[m][threadIdx.x];
        }
    }
}

template<int MAX_BLOCK_M=8, bool SYM=false>
static void gemm_warp_reduce(
    const core::Context& ctx,
    const core::Tensor& a,        // (M, K)
    const core::Tensor& q_weight, // (N, K / 8)
    const core::Tensor& qzeros,   // (N, K / group_size)
    const core::Tensor& scales,   // (N, K / group_size)
    const core::Tensor* bias,
    core::Tensor* c
) {
    const size_t M = a.size(0);
    const size_t K = a.size(1);
    const size_t N = q_weight.size(0);
    const size_t GROUP_SIZE = K / scales.size(1);

    c->set_name(q_weight.name() + "_out");

    int num_chunk = M / MAX_BLOCK_M;
    int last_chunk_m = M % MAX_BLOCK_M;

    auto stream = ctx.current_stream()->ptr;
    size_t PARALLEL = std::min(8UL, DIVIDE(N, size_t(ctx.get_mp_count())));
    PARALLEL = 8;
    const half* bias_ptr = bias ? bias->data<half>() : nullptr;
    if (num_chunk) {
        // std::cout << "num_chunk " << num_chunk << "\n";
        dim3 gridDim(DIVIDE(N, PARALLEL), num_chunk);
        dim3 blockDim(32, PARALLEL);
        KERNEL_gemm_warp_reduce<MAX_BLOCK_M, SYM><<<gridDim, blockDim, 0, stream>>>(
            a.data<half>(),
            q_weight.data<uint32_t>(),
            qzeros.data<uint8_t>(),
            scales.data<half>(),
            bias_ptr,
            c->mutable_data<half>(),
            num_chunk * MAX_BLOCK_M,
            N,
            K / 8,
            GROUP_SIZE / 8,
            nullptr,
            nullptr,
            false
        );
        BM_CUDART_ASSERT(cudaGetLastError());
    }
    if (last_chunk_m) {
        dim3 gridDim(DIVIDE(N, PARALLEL), 1);
        dim3 blockDim(32, PARALLEL);
        auto kernel = KERNEL_gemm_warp_reduce<1, SYM>;
        if (last_chunk_m == 1) {
            kernel = KERNEL_gemm_warp_reduce<1, SYM>;
        } else if (last_chunk_m == 2) {
            kernel = KERNEL_gemm_warp_reduce<2, SYM>;
        } else if (last_chunk_m == 3) {
            kernel = KERNEL_gemm_warp_reduce<3, SYM>;
        } else if (last_chunk_m == 4) {
            kernel = KERNEL_gemm_warp_reduce<4, SYM>;
        } else if (last_chunk_m == 5) {
            kernel = KERNEL_gemm_warp_reduce<5, SYM>;
        } else if (last_chunk_m == 6) {
            kernel = KERNEL_gemm_warp_reduce<6, SYM>;
        } else if (last_chunk_m == 7) {
            kernel = KERNEL_gemm_warp_reduce<7, SYM>;
        } else if (last_chunk_m == 8) {
            kernel = KERNEL_gemm_warp_reduce<8, SYM>;
        } else if (last_chunk_m == 9) {
            kernel = KERNEL_gemm_warp_reduce<9, SYM>;
        } else if (last_chunk_m == 10) {
            kernel = KERNEL_gemm_warp_reduce<10, SYM>;
        } else if (last_chunk_m == 11) {
            kernel = KERNEL_gemm_warp_reduce<11, SYM>;
        } else if (last_chunk_m == 12) {
            kernel = KERNEL_gemm_warp_reduce<12, SYM>;
        } else if (last_chunk_m == 13) {
            kernel = KERNEL_gemm_warp_reduce<13, SYM>;
        } else if (last_chunk_m == 14) {
            kernel = KERNEL_gemm_warp_reduce<14, SYM>;
        } else if (last_chunk_m == 15) {
            kernel = KERNEL_gemm_warp_reduce<15, SYM>;
        } else {
            BM_ASSERT(false, "last_chunk_m out of range");
        }

        int dyn_mem_size = 0;
//        dyn_mem_size = last_chunk_m * K * sizeof(half);
//        BM_CUDART_ASSERT(cudaFuncSetAttribute(
//            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_mem_size));

        half* last_c = c->mutable_data<half>() + num_chunk * MAX_BLOCK_M * N;
        kernel<<<gridDim, blockDim, dyn_mem_size, stream>>>(
            a.data<half>() + num_chunk * MAX_BLOCK_M * K,
            q_weight.data<uint32_t>(),
            qzeros.data<uint8_t>(),
            scales.data<half>(),
            bias_ptr,
            last_c,
            last_chunk_m,
            N,
            K / 8,
            GROUP_SIZE / 8,
            nullptr,
            nullptr,
            false
        );
        BM_CUDART_ASSERT(cudaGetLastError());
    }
}

template<bool SYM>
static core::Tensor gemm_moe1(
    const core::Context& ctx,
    const core::Tensor& a,        // (TOP_K, M, K)
    const core::Tensor& q_weight, // (NUM_EXP, N, K / 8)
    const core::Tensor& qzeros,   // (NUM_EXP, N, K / group_size)
    const core::Tensor& scales,   // (NUM_EXP, N, K / group_size)
    const core::Tensor* expert_ids, // (M, TOP_K)
    const core::Tensor* expert_weights // (M, TOP_K)
) {
    const size_t M = 1;
    const size_t TOP_K = a.size(0);
    const size_t K = a.size(1);
    const size_t N = q_weight.size(-2);
    const size_t GROUP_SIZE = K / scales.size(-1);
    BM_ASSERT_EQ(TOP_K, expert_weights->size(-1), "");

    core::Tensor c = ctx.tensor({M, N}, a.dtype());
    c.set_name(q_weight.name() + "_out");

    auto stream = ctx.current_stream()->ptr;
    size_t PARALLEL = 8;
    bool inplace = true;
    functions::BinaryElementwiseOp add_op(ctx, functions::BinaryElementwiseOp::Add);
    for (size_t k = 0; k < TOP_K; ++k){
        dim3 gridDim(DIVIDE(N, PARALLEL), 1);
        dim3 blockDim(32, PARALLEL);
        auto kernel = KERNEL_gemm_warp_reduce<1, SYM, half>;

        core::Tensor tmp = ctx.tensor({M, N}, a.dtype());
        core::Tensor *cur = k == 0 || inplace ? &c : & tmp;
        kernel<<<gridDim, blockDim, 0, stream>>>(
            a.data<half>() + k * K,
            q_weight.data<uint32_t>(),
            qzeros.data<uint8_t>(),
            scales.data<half>(),
            nullptr,
            cur->mutable_data<half>(),
            1,
            N,
            K / 8,
            GROUP_SIZE / 8,
            expert_ids->data<int>() + k,
            expert_weights->data<float>() + k,
            inplace && k > 0);
        BM_CUDART_ASSERT(cudaGetLastError());
        if (!inplace && k)
            add_op.inplace(ctx, c, tmp);
    }
    return functions::typecast(ctx, c, a.dtype());
}

template<bool SYM>
auto select_fuse_in_kernel(int last_chunk_m) {
    auto kernel = KERNEL_gemm_fuse_gate_in<1, SYM>;
    if (last_chunk_m == 1) {
        kernel = KERNEL_gemm_fuse_gate_in<1, SYM>;
    } else if (last_chunk_m == 2) {
        kernel = KERNEL_gemm_fuse_gate_in<2, SYM>;
    } else if (last_chunk_m == 3) {
        kernel = KERNEL_gemm_fuse_gate_in<3, SYM>;
    } else if (last_chunk_m == 4) {
        kernel = KERNEL_gemm_fuse_gate_in<4, SYM>;
    } else if (last_chunk_m == 5) {
        kernel = KERNEL_gemm_fuse_gate_in<5, SYM>;
    } else if (last_chunk_m == 6) {
        kernel = KERNEL_gemm_fuse_gate_in<6, SYM>;
    } else if (last_chunk_m == 7) {
        kernel = KERNEL_gemm_fuse_gate_in<7, SYM>;
    } else if (last_chunk_m == 8) {
        kernel = KERNEL_gemm_fuse_gate_in<8, SYM>;
    } else {
        BM_ASSERT(false, "last_chunk_m out of range");
    }
    return kernel;
}

core::Tensor gemm_fuse_gate_in(
    const core::Context& ctx,
    const core::Tensor& a,         // (M, K)
    const core::Tensor& q_weight1, // (N, K / 8)
    const core::Tensor& qzeros1,   // (N, K / group_size)
    const core::Tensor& scales1,   // (N, K / group_size)
    const core::Tensor& rev_perm1, // (K)
    const core::Tensor& q_weight2, // (N, K / 8)
    const core::Tensor& qzeros2,   // (N, K / group_size)
    const core::Tensor& scales2,   // (N, K / group_size)
    const core::Tensor& rev_perm2, // (K)
    bool sym
) {
    core::EventScope event_scope(ctx, "GPTQ_GEMM_fuse_gate_in:" + q_weight1.name(), 3);
    BM_ASSERT_LE(a.ndim(), 2, "Wrong ndim");
    BM_ASSERT_LE(a.size(0), 8, "Too big input dim0");
    BM_ASSERT_EQ(q_weight1.shape(), q_weight2.shape(), "in and gate should have same shape.");
    BM_ASSERT(qzeros1.dtype() == core::DataType::kInt8, "qzeros must be int8");
    BM_ASSERT(qzeros2.dtype() == core::DataType::kInt8, "qzeros must be int8");

    const size_t M = a.size(0);
    const size_t K = a.size(1);
    const size_t N = q_weight1.size(-2);
    const size_t GROUP_SIZE = K / scales1.size(-1);
    auto stream = ctx.current_stream()->ptr;

    core::Tensor c = ctx.tensor({M, N}, a.dtype());
    functions::zeros_(ctx, c);
    c.set_name(q_weight1.name() + "_out");

    int last_chunk_m = M;
    BM_ASSERT_LE(last_chunk_m, 8, "gemm_fuse_gate_in: M is too big");

    size_t PARALLEL = std::min(8UL, DIVIDE(N, size_t(ctx.get_mp_count())));
    PARALLEL = 8;
    BM_ASSERT(N % PARALLEL == 0, "");
    {
        dim3 gridDim(DIVIDE(N, PARALLEL), 1);
        dim3 blockDim(32, PARALLEL);
        auto kernel = sym ? select_fuse_in_kernel<true>(last_chunk_m) : select_fuse_in_kernel<false>(last_chunk_m);

        int dyn_mem_size = last_chunk_m * K * sizeof(half);
        BM_CUDART_ASSERT(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_mem_size));

        kernel<<<gridDim, blockDim, dyn_mem_size, stream>>>(
            a.data<half>(),
            q_weight1.data<uint32_t>(),
            qzeros1.data<uint8_t>(),
            scales1.data<half>(),
            nullptr,
            q_weight2.data<uint32_t>(),
            qzeros2.data<uint8_t>(),
            scales2.data<half>(),
            nullptr,
            c.mutable_data<half>(),
            last_chunk_m,
            N,
            K / 8,
            GROUP_SIZE / 8
        );
        BM_CUDART_ASSERT(cudaGetLastError());
    }
    return c;
}

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

// (N), (32)
template<class T=half, int OUT_TYPE=0>
__global__ void KERNEL_dequant(
    const uint32_t* __restrict__ b_q_weight,   // (N, K / 8)
    const uint8_t* __restrict__ b_gptq_qzeros, // (N, K / group_size)
    const half* __restrict__ b_gptq_scales,    // (N, K / group_size)
    void* __restrict__ b_out,                  // (N, K)
    const float* __restrict__ out_scale,      // (N)
    const int N,
    const int K_8,
    const int GROUP_SIZE_8
) {
    const int n = blockIdx.x * blockDim.y + threadIdx.y;
    if (n >= N) return;
    // assert(K % WARP_K == 0);
    // assert(WARP_K % GROUP_SIZE == 0);

    // slice matrix
    T* __restrict__ out = (T*)b_out + n * K_8 * 8;
    const uint32_t* __restrict__ q_weight = b_q_weight + n * K_8;
    const uint8_t* __restrict__ q_zeros = b_gptq_qzeros + n * (K_8 / GROUP_SIZE_8);
    const half* __restrict__ scales = b_gptq_scales + n *  (K_8 / GROUP_SIZE_8);

    float reciprocal_scale = 1.f;
    if (OUT_TYPE == 1) reciprocal_scale = 1.f / float(out_scale[n]);
    if (OUT_TYPE == 2) reciprocal_scale = 1.f / *(float*)out_scale;
    half2 scale_h2 = __float2half2_rn(reciprocal_scale);

    for (int k_8 = threadIdx.x; k_8 < K_8; k_8 += WARP_SIZE) {
        // load and decode weight
        half2 scale = __half2half2(scales[k_8 / GROUP_SIZE_8]);
        uint32_t zero = q_zeros[k_8 / GROUP_SIZE_8];
        half2 z1, z16;
        preprocess_zero(zero, z1, z16);

        __align__(16) half2 dq[4];
        uint32_t q_load = q_weight[k_8];
        dequant_8x4bit(q_load, dq, z1, z16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            dq[i] = __hmul2(dq[i], scale);
        }

        if constexpr (OUT_TYPE == 0) {
            *(reinterpret_cast<int4 *>(out) + k_8) = *reinterpret_cast<int4 *>(&dq);
        } else if constexpr (OUT_TYPE == 1) {
            __align__(8) uint8_t i8x8[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                i8x8[i * 2] = int8_t(nearbyintf(__low2float(dq[i]) * reciprocal_scale));
                i8x8[i * 2 + 1] = int8_t(nearbyintf(__high2float(dq[i]) * reciprocal_scale));
            }
            *(reinterpret_cast<int2 *>(out) + k_8) = *reinterpret_cast<int2 *>(&i8x8);
        } else if constexpr (OUT_TYPE == 2) {
            __align__(8) uint16_t i16x4[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                half2 h2s = __hmul2(dq[i], scale_h2);
                i16x4[i] = half2_to_e4m3(*(uint32_t * ) & h2s);
            }
            *(reinterpret_cast<int2 *>(out) + k_8) = *reinterpret_cast<int2 *>(&i16x4);
        }
    }
 }

core::Tensor dequant_k_major(
    const core::Context& ctx,
    const core::Tensor& q_weight, // (N, K / 8)
    const core::Tensor& qzeros,   // (N, K / group_size)
    const core::Tensor& scales,   // (N, K / group_size)
    int out_type  // 1: quant to int8; 2: quant to fp8
) {
    const size_t K = q_weight.size(-1) * 8;
    const size_t N = q_weight.size(-2);
    const size_t GROUP_SIZE = K / scales.size(-1);
    core::Tensor out_weight = out_type == 0
        ? ctx.tensor({N, K}, scales.dtype())
        : ctx.tensor({N, K}, core::DataType::kInt8);

    std::string ev_name = std::string("GPTQ_dequant_");
    if (out_type)
        ev_name += out_type == 1 ? "to_int8" : "to_fp8";
    else
        ev_name += "to_fp16";
    core::EventScope ev(ctx, ev_name + ":" + q_weight.name(), 3);
    if (out_type) {
        BM_ASSERT(q_weight.quant_scale, "");
    }

    auto stream = ctx.current_stream()->ptr;
    auto kernel = KERNEL_dequant<half, 0>;
    if (out_type == 1) {
        kernel = KERNEL_dequant<int8_t, 1>;
    } else if (out_type == 2) {
        kernel = KERNEL_dequant<int8_t, 2>;
    } else {
        BM_ASSERT_EQ(out_type, 0, "");
    }
    kernel<<<N, 32, 0, stream>>>(
        q_weight.data<uint32_t>(),
        qzeros.data<uint8_t>(),
        scales.data<half>(),
        out_weight.mutable_data(),
        out_type ? q_weight.quant_scale->data<float>() : nullptr,
        N,
        K / 8,
        GROUP_SIZE / 8);
    BM_CUDART_ASSERT(cudaGetLastError());

    return out_weight;
}

using bmengine::core::Tensor;
using model::ModelContext;

core::Tensor gptq_gemm_k_major(
    const core::Context& ctx,
    const core::Tensor& a,        // (M, K)
    const core::Tensor& q_weight, // (N, K / 8)
    const core::Tensor& qzeros,   // (N, K / group_size)
    const core::Tensor& scales,   // (N, K / group_size)
    const core::Tensor& q_perm,   // (K)
    const core::Tensor& rev_perm, // (K)
    const core::Tensor* bias,
    bool sym,
    bool cache_only,
    core::Tensor* output
) {
    const size_t K = a.size(-1);
    const size_t M = a.numel() / K;
    const size_t N = q_weight.size(-2);
    const size_t GROUP_SIZE = K / scales.size(-1);

    // BM_ASSERT_EQ(2, a.ndim(), "Wrong input dim");
    if (!cache_only) {
        BM_ASSERT_EQ(K, q_weight.size(-1) * 8, "size K mismatch");
        if (q_perm.numel()) {
            BM_ASSERT_EQ(K, q_perm.size(0) * 2, "q_perm is not int16");
            BM_ASSERT_EQ(K, rev_perm.size(0) * 2, "q_perm is not int16");
        }
    }
    if (output) {
        std::vector<size_t> out_shape{M, N};
        BM_ASSERT_EQ(output->size(), out_shape, "Wrong output size()");
        BM_ASSERT_EQ(output->dtype(), a.dtype(), "Wrong output dtype");
    }
    BM_ASSERT_EQ(a.dtype(), core::DataType::kHalf, "A must be half");
    BM_ASSERT_EQ(qzeros.dtype(), core::DataType::kInt8, "qzeros must be int8");

    int ev_level = 3;

    /********************************** W4A8 ********************************/
    ModelContext* m_ctx = dynamic_cast<ModelContext*>(const_cast<core::Context*>(&ctx));
    static int w4_fp8 = utils::get_int_env("W4_FP8_ALGO", 0);
    static int w4_int8 = utils::get_int_env("W4_INT8_ALGO", 0);
    static int a8_thres = utils::get_int_env("W4_A8_M_THRES", 40);
    static int skip_last_layers = utils::get_int_env("W4_A8_SKIP_LAST_LAYERS", 0);
    bool skip = m_ctx && ctx.current_layer() >= m_ctx->num_layers() - skip_last_layers;

    BM_ASSERT(!q_weight.name().empty(), "");
    /********************************* W4A8 FP8 ********************************/
    auto name_fp8 = logger::str_cat("GPTQ_GEMM_W4_FP8:", q_weight.name());
    if (w4_fp8 && (M > a8_thres || m_ctx->dual_stream() && m_ctx->layer_cache().count(name_fp8)) && !bias && N > 1024 && !skip) {
        static int MAX_ACT_E4M3 = utils::get_int_env("MAX_ACT_E4M3", 448);
        core::EventScope event_scope(ctx, name_fp8, ev_level);
        BM_ASSERT(q_weight.quant_scale, "");
        if (m_ctx->layer_cache().count(name_fp8) == 0) {
            m_ctx->layer_cache()[name_fp8] = dequant_k_major(ctx, q_weight, qzeros, scales, 2);
            if (cache_only) return core::Tensor();
        }
        core::Tensor w8 = m_ctx->layer_cache()[name_fp8];

        // quant a
        core::Tensor a_quant; // (M, K)
        if (!a.quant_scale) {
            core::EventScope event_scope(ctx, "quantA_calc_scale");
            a_quant = nn::fp8::dynamic_scaled_quant(ctx, a, MAX_ACT_E4M3);
            int8_op::set_quant_scale(const_cast<Tensor&>(a), a_quant);
        } else {
            a_quant = *a.quant_scale;
        }

        functions::Gemm gemm(ctx, core::DataType::kFP8_E4M3, false, true);
        gemm.set_A_scale(*a_quant.quant_scale);
        gemm.set_B_scale(*q_weight.quant_scale);
        Tensor ret = gemm.forward(ctx, a_quant, w8);
        if (ctx.rank() == 100 && q_weight.name() == "llama.layers.0.attn.project_k")
            std::cout << "w4a8_out: " << ret << endl;
        return ret;
    }

    /********************************* W4A8 INT8 ********************************/
    static int w4_int8_attn = utils::get_int_env("W4_INT8_ATTN", 0);
    if (w4_int8_attn == 0 && q_weight.name().find(".attn.") != std::string::npos)
        skip = true;
    auto name_int8 = logger::str_cat("GPTQ_GEMM_W4_INT8:", q_weight.name());
    if (w4_int8 && (M > a8_thres || m_ctx->dual_stream() && m_ctx->layer_cache().count(name_int8)) && !bias && N > 1024 && !skip) {
        // core::EventScope event_scope(ctx, name_int8, cache_only ? 100 : ev_level);
        BM_ASSERT(q_weight.quant_scale, "");
        if (m_ctx->layer_cache().count(name_int8) == 0) {
            m_ctx->layer_cache()[name_int8] = dequant_k_major(ctx, q_weight, qzeros, scales, 1);
            if (cache_only) return core::Tensor();
        }
        core::Tensor w8 = m_ctx->layer_cache()[name_int8];
        auto s_scales = q_weight.quant_scale;

        // quant a
        core::Tensor a_quant; // (M, K)
        if (!a.quant_scale) {
            core::EventScope event_scope(ctx, "GPTQ_quant_a_to_int8", ev_level);
            a_quant = int8_op::quant_calc_scale(ctx, a);
            int8_op::set_quant_scale(const_cast<Tensor&>(a), a_quant);
        } else {
            a_quant = *a.quant_scale;
        }
        BM_ASSERT(a_quant.quant_scale, "quant tensor has no scale");
        auto a_scale = a_quant.quant_scale;

        ctx.recordEvent(name_int8, ev_level);
        functions::Gemm gemm(ctx, core::DataType::kInt8, false, true);
        Tensor ret8 = gemm.forward(ctx, a_quant, w8);
        ctx.recordEvent("quant_back", ev_level);
        Tensor ret = int8_op::quant_scale_back(ctx, ret8, a_scale.get(), s_scales.get(), core::DataType::kHalf, output);
        if (ctx.rank() == 100 && q_weight.name() == "llama.layers.0.attn.project_k")
            std::cout << "w4a8_out: " << ret << endl;
        return ret;
        // return gemm_groupwise_int8(ctx, a_quant, w8, *q_scales, *s_scales, *a_scale, bias);
    }

    /********************************* W4A16 ********************************/
    if (M > 40) {
        auto name1 = "GPTQ_GEMM_W4_A16:" + q_weight.name();
        if (m_ctx->layer_cache().count(name1) == 0) {
            m_ctx->layer_cache()[name1] = dequant_k_major(ctx, q_weight, qzeros, scales);
            if (cache_only) return core::Tensor();
        }
        core::Tensor half_weight = m_ctx->layer_cache()[name1];
        // call gemm
        functions::Gemm gemm(ctx, a.dtype(), false, true);
        if (ctx.high_precision() >= 1 || N <= 1024)
            gemm.set_compute_type(CUBLAS_COMPUTE_32F);
        core::Tensor permuted_a = q_perm.numel() ? permute_input(ctx, a, q_perm) : a;
        ctx.recordEvent(name1, ev_level);
        auto c = gemm.forward(ctx, permuted_a, half_weight, output, bias);
        if (ctx.rank() == 100 && q_weight.name() == "llama.layers.0.attn.project_k") {
            std::cout << "f16_out: " << c << endl;
        }
        return c;
    } else {
        // core::Tensor c = ctx.tensor({M, N}, a.dtype());
        core::Tensor c = output ? *output : ctx.tensor({M, N}, a.dtype());
        if (q_perm.numel()) {
            core::Tensor permuted_a = permute_input(ctx, a, q_perm);
            gemm_warp_reduce(ctx, permuted_a, q_weight, qzeros, scales, bias, &c);
        } else if (sym) {
            core::EventScope event_scope(ctx, "GPTQ_GEMM_SYM_KERNEL:" + q_weight.name(), ev_level);
            gemm_warp_reduce<16, true>(ctx, a, q_weight, qzeros, scales, bias, &c);
        } else {
            core::EventScope event_scope(ctx, "GPTQ_GEMM_KERNEL:" + q_weight.name(), ev_level);
            gemm_warp_reduce<16>(ctx, a, q_weight, qzeros, scales, bias, &c);
        }
        return c;
    }
}

}  // namespace gptq
}  // namespace nn

