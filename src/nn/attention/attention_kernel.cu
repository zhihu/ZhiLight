#include "nn/attention/attention_kernel.h"
#include "nn/attention/quant_attention.cuh"

#include <bmengine/core/core.h>
#include <bmengine/functions/utils.cuh>
#include <bmengine/functions/reduce.cuh>
#include <bmengine/functions/gemm.h>
#include <bmengine/functions/transpose.h>
#include <bmengine/functions/element.h>
#include <bmengine/logger/std_log_op.hpp>
#include "nn/quant/int8/quant_kernel.h"
#include "utils/env.h"

#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <vector_types.h>

namespace nn {

using bmengine::core::DataType;
using bmengine::core::Tensor;

#define WARP_SIZE 32
template<typename T, typename T2 = T, int DIM_HEAD = 128>
static __inline__ __device__ void multiply_q_k_block(
    const T* __restrict__ g_q, // (dim_head)
    const T* __restrict__ g_k, // (len_buf, dim_head)
    T2* __restrict__ logit,    // (len_buf)
    int len_buf,
    int stride_k = DIM_HEAD) {
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpNum = blockDim.x / WARP_SIZE;
    assert(DIM_HEAD == WARP_SIZE * 4);

    short4 a4 = *reinterpret_cast<const short4*>(g_q + laneId * 4);
    for (int col = warpId; col < len_buf; col += warpNum) {
        float res = 0;
        short4 b4 = *reinterpret_cast<const short4*>(g_k + col * stride_k + laneId * 4);
        res += float(*reinterpret_cast<T*>(&a4.x)) * float(*reinterpret_cast<T*>(&b4.x));
        res += float(*reinterpret_cast<T*>(&a4.y)) * float(*reinterpret_cast<T*>(&b4.y));
        res += float(*reinterpret_cast<T*>(&a4.z)) * float(*reinterpret_cast<T*>(&b4.z));
        res += float(*reinterpret_cast<T*>(&a4.w)) * float(*reinterpret_cast<T*>(&b4.w));
        res = functions::warpReduceSum<float>(res);
        if (laneId == 0)
            logit[col] = T2(res);
    }
}
template<typename T, typename T2 = T, int DIM_HEAD = 64, int HEAD1 = 8, typename T_LOAD = double2>
static __forceinline__ __device__ void multiply_q_k_block1(
    const T* __restrict__ s_q, // (dim_head)
    const T* __restrict__ g_k, // (len_buf, dim_head)
    T2* __restrict__ logit,    // (len_buf)
    int len_buf,
    int stride_k = DIM_HEAD) {
    static_assert(DIM_HEAD % 16 == 0);

    // __align__(16) T q_chunk[HEAD1];
    __align__(16) T k_chunk[HEAD1];
    for (int col = threadIdx.x; col < len_buf; col += blockDim.x) {
        float res = 0;
        const T* key = g_k + col * stride_k;
        for (int d1 = 0; d1 < DIM_HEAD; d1 += HEAD1) {
            // *reinterpret_cast<T_LOAD*>(&q_chunk[0]) = *reinterpret_cast<const T_LOAD*>(s_q + d1);
            *reinterpret_cast<T_LOAD*>(&k_chunk[0]) = *reinterpret_cast<const T_LOAD*>(key + d1);
            const T* query = s_q + d1;
#pragma unroll
            for (int d = 0; d < HEAD1; d++) {
                res += float(query[d]) * float(k_chunk[d]);
                // res+= float(q_chunk[d]) * float(k_chunk[d]);
            }
        }
        logit[col] = T2(res);
    }
}
template<typename T, typename T2 = T, int DIM_HEAD = 64>
static __inline__ __device__ void multiply_q_k_block_d64(
    const T* __restrict__ g_q, // (dim_head)
    const T* __restrict__ g_k, // (len_buf, dim_head)
    T2* __restrict__ logit,    // (len_buf)
    int len_buf,
    int stride_k = DIM_HEAD) {
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpNum = blockDim.x / WARP_SIZE;
    static_assert(DIM_HEAD == WARP_SIZE * 2);

    short2 a2 = *reinterpret_cast<const short2*>(g_q + laneId * 2);
    for (int col = warpId; col < len_buf; col += warpNum) {
        float res = 0;
        short2 b2 = *reinterpret_cast<const short2*>(g_k + col * stride_k + laneId * 2);
        res += float(*reinterpret_cast<T*>(&a2.x)) * float(*reinterpret_cast<T*>(&b2.x));
        res += float(*reinterpret_cast<T*>(&a2.y)) * float(*reinterpret_cast<T*>(&b2.y));
        res = functions::warpReduceSum<float>(res);
        if (laneId == 0)
            logit[col] = T2(res);
    }
}

template<int THREAD_NUM, typename T, typename T2>
__inline__ __device__ void copy_8x128(T* dst, const T2* src) {
    dst[threadIdx.x] = src[threadIdx.x];
}
template<>
__inline__ __device__ void copy_8x128<1024, __half, float>(__half* dst, const float* src) {
    dst[threadIdx.x] = __half(src[threadIdx.x]);
}
template<>
__inline__ __device__ void copy_8x128<1024, __half, __half>(__half* dst, const __half* src) {
    dst[threadIdx.x] = src[threadIdx.x];
}
template<>
__inline__ __device__ void copy_8x128<1024, float, __half>(float* dst, const __half* src) {
    dst[threadIdx.x] = src[threadIdx.x];
}

template<typename T, typename T2 = T, int DIM_HEAD = 128, int M_QUERY = 8>
static __inline__ __device__ void multiply_q_k_block_mq1(
    const T* __restrict__ g_q, // (m_query, dim_head), in shared memory
    const T* __restrict__ g_k, // (len_buf, dim_head)
    T2* __restrict__ logit,    // (m_query, len_buf)
    int len_buf) {
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpNum = blockDim.x / WARP_SIZE;
    assert(DIM_HEAD == WARP_SIZE * 4);

    for (int col = warpId; col < len_buf; col += warpNum) {
        short4 b4 = *reinterpret_cast<const short4*>(g_k + col * DIM_HEAD + laneId * 4);
#pragma unroll
        for (int q = 0; q < M_QUERY; ++q) {
            float res = 0;
            short4 a4 = *reinterpret_cast<const short4*>(g_q + q * DIM_HEAD + laneId * 4);
            res += float(*reinterpret_cast<T*>(&a4.x)) * float(*reinterpret_cast<T*>(&b4.x));
            res += float(*reinterpret_cast<T*>(&a4.y)) * float(*reinterpret_cast<T*>(&b4.y));
            res += float(*reinterpret_cast<T*>(&a4.z)) * float(*reinterpret_cast<T*>(&b4.z));
            res += float(*reinterpret_cast<T*>(&a4.w)) * float(*reinterpret_cast<T*>(&b4.w));
            res = functions::warpReduceSum<float>(res);
            if (laneId == 0)
                logit[q * len_buf + col] = T2(res);
        }
    }
}

template<typename T, int DIM_HEAD = 128, typename T2 = T>
static __device__ void multiply_score_v_block(
    const float* score,        // (len_buf)
    const T* __restrict__ g_v, // (len_buf, dim_head)
    T* __restrict__ output,    // (dim_head)
    int len_buf,
    int stride_v = DIM_HEAD) {
    assert(DIM_HEAD == WARP_SIZE * 4);

    if (threadIdx.x >= DIM_HEAD)
        return;
    const int x = threadIdx.x;

    float res = 0;
    for (int i = 0; i < len_buf; ++i) {
        res += float(score[i]) * float(g_v[i * stride_v + x]);
    }
    output[x] = T(res);
}
template<typename T, int DIM_HEAD = 128, typename T2 = T, int NUM_SPLIT = 1024 / DIM_HEAD>
static __device__ void multiply_score_v_block2(
    const float* score,        // (len_buf)
    const T* __restrict__ g_v, // (len_buf, DIM_HEAD)
    T2* __restrict__ output,   // (DIM_HEAD)
    int len_buf,
    int stride_v = DIM_HEAD) {
    static_assert((DIM_HEAD % WARP_SIZE) == 0);
    static_assert((1024 % DIM_HEAD) == 0);
    assert(blockDim.x == 1024);
    const int len_buf_s = (len_buf + NUM_SPLIT - 1) / NUM_SPLIT;

    const int shard = threadIdx.x / DIM_HEAD;
    const int d = threadIdx.x % DIM_HEAD;

    float res = 0;
    int start = shard * len_buf_s;
    int end = min(start + len_buf_s, len_buf);
    for (int i = start; i < end; ++i) {
        res += float(score[i]) * float(g_v[i * stride_v + d]);
    }
    // reduce sum of all shards
    static __shared__ float tmp[1024];
    tmp[d * NUM_SPLIT + shard] = res;
    __syncthreads();
    float x = tmp[threadIdx.x];
    if (NUM_SPLIT == 16)
        x += __shfl_down_sync(0xFFFFFFFF, x, 8);
    if (NUM_SPLIT >= 8)
        x += __shfl_down_sync(0xFFFFFFFF, x, 4);
    if (NUM_SPLIT >= 4)
        x += __shfl_down_sync(0xFFFFFFFF, x, 2);
    x += __shfl_down_sync(0xFFFFFFFF, x, 1);
    if ((threadIdx.x % NUM_SPLIT) == 0) {
        output[threadIdx.x / NUM_SPLIT] = T2(x);
    }
}
template<typename T, typename T2 = T, int DIM_HEAD = 128, int M_QUERY = 8>
static __device__ void multiply_score_v_block_mq1(
    const float* score,        // (m_query, len_buf)
    const T* __restrict__ g_v, // (len_buf, dim_head)
    T* __restrict__ output,    // (m_query, dim_head)
    int len_buf) {
    assert(DIM_HEAD == WARP_SIZE * 4);

    if (threadIdx.x >= DIM_HEAD)
        return;
    const int x = threadIdx.x;

    float res[M_QUERY];
#pragma unroll
    for (int q = 0; q < M_QUERY; ++q) {
        res[q] = 0;
    }
    for (int i = 0; i < len_buf; ++i) {
        float v = float(g_v[i * DIM_HEAD + x]); // global memory
#pragma unroll
        for (int q = 0; q < M_QUERY; ++q) {
            res[q] += float(score[q * len_buf + i]) * v; // shared memory
        }
    }
#pragma unroll
    for (int q = 0; q < M_QUERY; ++q) {
        output[q * DIM_HEAD + x] = T(res[q]);
    }
}

using namespace nvcuda;

template<typename T, int M_Q = 8, int N_BUF = 32, int K_DIM = 16, int DIM_HEAD = 128>
static __forceinline__ __device__ void multiply_q_k_block_wmma(
    const T* __restrict__ g_q, // (m_query, dim_head), in shared memory
    const T* __restrict__ g_k, // (len_buf, dim_head)
    float* logit,              // (m_query, len_buf)
    int len_buf) {
#if (__CUDA_ARCH__ >= 800) || !defined(__linux__)
    assert(len_buf % N_BUF == 0);

    const int warpId = threadIdx.x / WARP_SIZE;
    const int warpNum = blockDim.x / WARP_SIZE;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, M_Q, N_BUF, K_DIM, T, wmma::row_major> fragment_a;
    wmma::fragment<wmma::matrix_b, M_Q, N_BUF, K_DIM, T, wmma::col_major> fragment_b;
    wmma::fragment<wmma::accumulator, M_Q, N_BUF, K_DIM, float> fragment_acc;

    // divide len_buf by warpNum
    for (int n = warpId * N_BUF; n < len_buf; n += warpNum * N_BUF) {
        wmma::fill_fragment(fragment_acc, 0.0f);
        for (int k = 0; k < DIM_HEAD; k += K_DIM) {
            // Load the inputs
            wmma::load_matrix_sync(fragment_a, g_q + k, DIM_HEAD);
            wmma::load_matrix_sync(fragment_b, g_k + n * DIM_HEAD + k, DIM_HEAD);

            // Perform the matrix multiplication
            wmma::mma_sync(fragment_acc, fragment_a, fragment_b, fragment_acc);
        }
        // printf("B: %p %d\n", logit + n, n);
        // Store acc
        wmma::store_matrix_sync(logit + n, fragment_acc, len_buf, wmma::mem_row_major);
    }
#endif
}

// version1: use DIM_HEAD(128) threads
template<
    typename T,
    int M_Q = 8,
    int N_DIM = 32,
    int K_BUF = 16,
    int DIM_HEAD = 128,
    int THREAD_NUM = 1024>
static __forceinline__ __device__ void multiply_score_v_block_wmma1(
    const T* score,            // (m_query, len_buf)
    const T* __restrict__ g_v, // (len_buf, dim_head)
    T* output,                 // (m_query, dim_head)
    int len_buf,
    float* float_out,          // (m_query, dim_head)
    int stride_v = DIM_HEAD,
    int q_head_offset = 0) {
    const int warpId = threadIdx.x / WARP_SIZE;

    // each warp do a tile multiplication: (M_Q, len_buf) * (len_buf, N_DIM) => (M_Q, N_DIM)
    if (warpId < DIM_HEAD / N_DIM) {
        assert(len_buf % K_BUF == 0);

        g_v += warpId * N_DIM;

        // __frag_base<__half, 16>
        wmma::fragment<wmma::matrix_a, M_Q, N_DIM, K_BUF, T, wmma::row_major> fragment_a;
        wmma::fragment<wmma::matrix_b, M_Q, N_DIM, K_BUF, T, wmma::row_major> fragment_b;
        wmma::fragment<wmma::accumulator, M_Q, N_DIM, K_BUF, float> fragment_acc;

        wmma::fill_fragment(fragment_acc, 0.0f);
        for (int k = 0; k < len_buf; k += K_BUF) {
            // Load the inputs
            wmma::load_matrix_sync(fragment_a, score + k, len_buf);
            wmma::load_matrix_sync(fragment_b, g_v + k * stride_v, DIM_HEAD);

            // Perform the matrix multiplication
            wmma::mma_sync(fragment_acc, fragment_a, fragment_b, fragment_acc);
        }
        // Store acc
        wmma::store_matrix_sync(
            float_out + warpId * N_DIM, fragment_acc, DIM_HEAD, wmma::mem_row_major);
    }

    // copy to output
    __syncthreads();
    if constexpr (DIM_HEAD == 128) {
        copy_8x128<THREAD_NUM>(output, float_out);
        if (M_Q >= 16) {
            copy_8x128<THREAD_NUM>(output + q_head_offset, float_out + 1024);
        }
        if (M_Q == 32) {
            copy_8x128<THREAD_NUM>(output + 2 * q_head_offset, float_out + 2 * 1024);
            copy_8x128<THREAD_NUM>(output + 3 * q_head_offset, float_out + 3 * 1024);
        }
    } else {
        assert(DIM_HEAD <= blockDim.x);
        if (threadIdx.x < DIM_HEAD) {
            for (int i = 0; i < M_Q; ++i) {
                output[i * DIM_HEAD + threadIdx.x] = float_out[i * DIM_HEAD + threadIdx.x];
            }
        }
    }
}

// version2: use 1024 threads
template<
    typename T,
    int M_Q = 8,
    int N_DIM = 32,
    int K_BUF = 16,
    int DIM_HEAD = 128,
    int WRAP_NUM = 32,
    int WARP_H = (DIM_HEAD / N_DIM),
    int SPLIT = (WRAP_NUM / WARP_H)>
static __forceinline__ __device__ void multiply_score_v_block_wmma2(
    const T* score,            // (m_query, len_buf)
    const T* __restrict__ g_v, // (len_buf, dim_head)
    T* __restrict__ output,    // (m_query, dim_head)
    const int len_buf,
    float* float_out,
    int q_head_offset = 0) {
    static_assert(WARP_H <= WRAP_NUM);
    assert(
        blockDim.x == WRAP_NUM * WARP_SIZE); // = DIM_HEAD * SPLIT, DIM_HEAD = N_DIM(WARP_SIZE) * 4
    assert(len_buf % K_BUF == 0);
    const int len_buf_chunk = ((len_buf + 127) / 128) * 128 / SPLIT; // multiple of K_BUF

    const int warpId = threadIdx.x / WARP_SIZE; // 0~32
    const int splitId = warpId / WARP_H;        // 0~8

    wmma::fragment<wmma::matrix_a, M_Q, N_DIM, K_BUF, T, wmma::row_major> fragment_a;
    wmma::fragment<wmma::matrix_b, M_Q, N_DIM, K_BUF, T, wmma::row_major> fragment_b;
    wmma::fragment<wmma::accumulator, M_Q, N_DIM, K_BUF, float> fragment_acc;

    wmma::fill_fragment(fragment_acc, 0.0f);

    g_v += (warpId % WARP_H) * N_DIM;
    for (int k = splitId * len_buf_chunk; k < len_buf && k < (splitId + 1) * len_buf_chunk;
         k += K_BUF) {
        // Load the inputs
        wmma::load_matrix_sync(fragment_a, score + k, len_buf);
        wmma::load_matrix_sync(fragment_b, g_v + k * DIM_HEAD, DIM_HEAD);

        // Perform the matrix multiplication
        wmma::mma_sync(fragment_acc, fragment_a, fragment_b, fragment_acc);
    }
    // accumulate of all splits
    float v = 0;
    float v1 = 0;
    for (int i = 0; i < SPLIT; ++i) {
        if (splitId == i) {
            wmma::store_matrix_sync(
                float_out + (warpId % WARP_H) * N_DIM, fragment_acc, DIM_HEAD, wmma::mem_row_major);
        }
        __syncthreads();
        v += float_out[threadIdx.x];
        if (M_Q == 16)
            v1 += reinterpret_cast<volatile float*>(float_out)[1024 + threadIdx.x];
    }
    output[threadIdx.x] = T(v);
    if (M_Q == 16)
        output[q_head_offset + threadIdx.x] = T(v1);
}

// use outer shared memory
template<typename T>
__inline__ __device__ T blockReduceMax(T x, T* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    x = functions::warpReduceMax<T>(x);
    if (lane == 0)
        shared[wid] = x;
    __syncthreads();

    if (wid == 0) {
        x = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -functions::Inf<T>();
        x = functions::warpReduceMax<T>(x);
        if (lane == 0)
            shared[32] = x;
    }
    __syncthreads();
    return shared[32]; // avoid RAW hazard
}

// use outer shared memory
template<typename T>
__inline__ __device__ T blockReduceSum(T x, T* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    x = functions::warpReduceSum<T>(x);
    if (lane == 0)
        shared[wid] = x;
    __syncthreads();

    if (wid == 0) {
        x = (threadIdx.x < blockDim.x / 32) ? shared[lane] : T(0.);
        x = functions::warpReduceSum<T>(x);
        if (lane == 0)
            shared[32] = x;
    }
    __syncthreads();
    return shared[32]; // avoid RAW hazard
}

template<typename T>
static inline __device__ void softmax_mask_block(
    float* reduce_buffer,
    float* smem,                         // (len_buf)
    const int8_t* __restrict__ mask,     // (len_buf)
    const T* __restrict__ position_bias, // (len_buf)
    float scale,
    int len_buf,
    T* data = nullptr,
    int len_sys = 0,
    float* local_max_out = nullptr,
    float* local_sum_out = nullptr) {
    if (data) {
        for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
            smem[i] = __ldg(mask + i) ? float(data[i]) * scale : -functions::Inf<float>();
        }
    } else if (!position_bias) {
        for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
            smem[i] = __ldg(mask + i) ? smem[i] * scale : -functions::Inf<float>();
        }
    } else {
        for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
            smem[i] = __ldg(mask + i) ? smem[i] * scale + float(position_bias[i])
                                      : -functions::Inf<float>();
        }
    }
    float local_max = -1e20;
    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        local_max = fmaxf(local_max, smem[i]);
    }
    local_max = blockReduceMax<float>(local_max, reduce_buffer);

    float local_sum = 1e-20;
    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        float v = expf(float(smem[i]) - local_max);
        smem[i] = v;
        local_sum += v;
    }
    local_sum = blockReduceSum<float>(local_sum, reduce_buffer);

    if (data) {
        for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
            data[i] = T(float(smem[i]) / local_sum);
        }
    } else {
        for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
            smem[i] = float(smem[i]) / local_sum;
        }
    }
    if (local_max_out && threadIdx.x == 0) {
        *local_max_out = local_max;
    }
    if (local_sum_out && threadIdx.x == 0) {
        *local_sum_out = local_sum;
    }
}
template<typename T>
static inline __device__ void softmax_mask_block_simple(
    float* reduce_buffer,
    float* smem,                     // (len_buf)
    const int8_t* __restrict__ mask, // (len_buf)
    float scale,
    int len_buf) {
    assert(len_buf <= blockDim.x);

    auto i = threadIdx.x;
    float v = (i < len_buf && __ldg(mask + i)) ? smem[i] * scale : -functions::Inf<float>();

    float max_v = blockReduceMax<float>(fmaxf(-1e20, v), reduce_buffer);

    float e = expf(v - max_v); // range: (0, 1)

    float sum_e = blockReduceSum<float>(e, reduce_buffer);

    if (i < len_buf) {
        smem[i] = sum_e > 1e-20 ? e / sum_e : 0;
    }
}

template<typename T>
static inline __device__ void cast_inline_block(float* buf, int len_buf) {
    T* buf_t = reinterpret_cast<T*>(buf);
    int round = (len_buf + blockDim.x - 1) / blockDim.x;
    for (int r = 0, i = threadIdx.x; r < round; ++r, i += blockDim.x) {
        T t = i < len_buf ? T(buf[i]) : T(0.);
        //        assert(t == t);
        __syncthreads();
        if (i < len_buf) {
            buf_t[i] = t;
        }
    }
    __syncthreads();
}

template<typename T>
__device__ T* get_align32_shared_memory() {
    extern __shared__ double4 _mem[];
    float* align_mem = reinterpret_cast<float*>(_mem);
    while (reinterpret_cast<uintptr_t>(align_mem) % 32 != 0)
        align_mem++; // align to 32
    return reinterpret_cast<T*>(align_mem);
}

// gridDim (batch, num_kv_heads, len_q),  blockDim (dim_head)
template<typename T, int DIM_HEAD = 128>
static __global__ void BM_KERNEL(mul_qk_rag_buffer)(
    const T* __restrict__ g_q,        // (batch, num_kv_heads, len_q, dim_head）
    const int* __restrict__ buf_lens, // (batch)
    T** __restrict__ key_buf_addrs,   // (batch) => (num_kv_heads, len_buf, dim_head)
    T* __restrict__ total_score       // (batch) => (num_kv_heads, len_q, len_buf)
) {
    const int len_buf = buf_lens[blockIdx.x];
    const int head = blockIdx.y;
    g_q += ((blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * DIM_HEAD;
    T* g_k = key_buf_addrs[blockIdx.x]; // (num_heads, len_buf, dim_head)
    g_k += head * len_buf * DIM_HEAD;

    int len_offset = 0;
    for (int i = 0; i < blockIdx.x; ++i) {
        len_offset += buf_lens[i];
    }
    T* logit = total_score + len_offset * gridDim.y * gridDim.z; // * num_heads * len_q
    // logit += (head * len_q + qi) * len_buf;
    logit += (head * gridDim.z + blockIdx.z) * len_buf;

    // Not optimize for len_q > 1. i.e. call len_q times
    multiply_q_k_block<T>(g_q, g_k, logit, len_buf);
}

// gridDim (batch, num_kv_heads, len_q),  blockDim (dim_head)
template<typename T, int DIM_HEAD = 128>
static __global__ void BM_KERNEL(mul_qk_softmax_rag_buffer)(
    const T* __restrict__ g_q,            // (batch, num_kv_heads, len_q, dim_head）
    const int* __restrict__ buf_lens,     // (batch)
    T** __restrict__ key_buf_addrs,       // (batch) => (num_kv_heads, len_buf, dim_head)
    const int8_t* __restrict__ mask,      // (batch) => (len_q, len_buf)
    T** __restrict__ position_bias_addrs, // (batch) => (num_kv_heads, len_q, len_buf)
    T* __restrict__ total_score,          // (batch) => (num_kv_heads, len_q, len_buf)
    float scale) {
    functions::SharedMemory<float> shared;
    float* smem = shared.getPointer();

    const int len_buf = buf_lens[blockIdx.x];
    const unsigned int head = blockIdx.y;
    //    const unsigned int num_heads = gridDim.y;
    const unsigned int len_q = gridDim.z;

    g_q += ((blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * DIM_HEAD;
    T* g_k = key_buf_addrs[blockIdx.x]; // (num_heads, len_buf, dim_head)
    g_k += head * len_buf * DIM_HEAD;

    int len_offset = 0;
    for (int i = 0; i < blockIdx.x; ++i) {
        len_offset += buf_lens[i];
    }
    T* logit = total_score + len_offset * gridDim.y * gridDim.z; // * num_heads * len_q
    // logit += (head * len_q + qi) * len_buf;
    logit += (head * gridDim.z + blockIdx.z) * len_buf;

    // Not optimize for len_q > 1. i.e. call len_q times
    multiply_q_k_block<T, float>(g_q, g_k, smem, len_buf);

    mask += len_q * len_offset + blockIdx.z * len_buf;
    // position_bias += num_heads * len_q * len_offset;  // (num_heads, len_q, len_buf)
    T* position_bias = position_bias_addrs[blockIdx.x];
    position_bias += (head * gridDim.z + blockIdx.z) * len_buf;

    __syncthreads();
    static __shared__ float reduce_buffer[33];
    softmax_mask_block(reduce_buffer, smem, mask, position_bias, scale, len_buf);
    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        logit[i] = T(smem[i]);
    }
}

// gridDim (batch, num_heads, len_q),  blockDim (dim_head)
template<typename T, int DIM_HEAD = 128, int O_DIM_HEAD = DIM_HEAD >
static __global__ void KERNEL_attn_qpk_rag_buffer(
    const T* __restrict__ g_q,            // (batch, len_q, num_heads, dim_head）
    const int* __restrict__ buf_lens,     // (batch)
    T** __restrict__ key_buf_addrs,       // (batch) => (num_heads, len_buf, dim_head)
    T** __restrict__ val_buf_addrs,       // (batch) => (num_heads, len_buf, dim_head)
    const int8_t* __restrict__ mask,      // (batch) => (len_q, len_buf)
    T** __restrict__ position_bias_addrs, // (batch) => (num_heads, len_q, len_buf)
    T* __restrict__ output,               // (batch, len_q, num_heads, O_DIM_HEAD)
    float scale,
    bool BSHD) {
    const int len_buf = buf_lens[blockIdx.x];
    const unsigned int head = blockIdx.y;
    const unsigned int num_heads = gridDim.y;
    const unsigned int len_q = gridDim.z;

    g_q += ((blockIdx.x * gridDim.z + blockIdx.z) * gridDim.y + blockIdx.y) * DIM_HEAD;
    output += ((blockIdx.x * gridDim.z + blockIdx.z) * gridDim.y + blockIdx.y) * O_DIM_HEAD;
    T* g_k = key_buf_addrs[blockIdx.x]; // (num_heads, len_buf, dim_head)
    T* g_v = val_buf_addrs[blockIdx.x]; // (num_heads, len_buf, dim_head)
    int stride_kv = BSHD ? num_heads * DIM_HEAD : DIM_HEAD;
    if (BSHD) {
        g_k += head * DIM_HEAD; // (len_buf, num_heads, dim_head)
        g_v += head * DIM_HEAD;
    } else {
        g_k += head * len_buf * DIM_HEAD;
        g_v += head * len_buf * DIM_HEAD;
    }

    int len_offset = 0;
    for (int i = 0; i < blockIdx.x; ++i) {
        len_offset += buf_lens[i];
    }

    // Not optimize for len_q > 1. i.e. call len_q times
    T* s_q = get_align32_shared_memory<T>();
    float* smem = reinterpret_cast<float*>(s_q + DIM_HEAD);
    if (DIM_HEAD != 128) {
        if (threadIdx.x < DIM_HEAD)
            s_q[threadIdx.x] = g_q[threadIdx.x];
        __syncthreads();
        multiply_q_k_block1<T, float, DIM_HEAD>(&s_q[0], g_k, smem, len_buf, stride_kv);
        // multiply_q_k_block_d64<T, float>(g_q, g_k, smem, len_buf, stride_kv);
    } else {
        multiply_q_k_block<T, float, DIM_HEAD>(g_q, g_k, smem, len_buf, stride_kv);
    }

    mask += len_q * len_offset + blockIdx.z * len_buf;
    // position_bias += num_heads * len_q * len_offset;  // (num_heads, len_q, len_buf)
    T* position_bias = nullptr;
    if (position_bias_addrs) {
        position_bias = position_bias_addrs[blockIdx.x];
        position_bias += (head * gridDim.z + blockIdx.z) * len_buf;
    }

    __syncthreads();
    static __shared__ float reduce_buffer[33];
    softmax_mask_block(reduce_buffer, smem, mask, position_bias, scale, len_buf);
    __syncthreads();

    multiply_score_v_block2<T, O_DIM_HEAD>(smem, g_v, output, len_buf, stride_kv);
}

// gridDim (len_q, num_kv_heads * m_query, batch),  blockDim (dim_head)
template<typename T, int DIM_HEAD = 128, int O_DIM_HEAD = DIM_HEAD>
static __global__ void KERNEL_mqa_rag_buffer1(
    const T* __restrict__ g_q,        // (batch, len_q, num_kv_heads * m_query, dim_head）
    const int* __restrict__ buf_lens, // (batch)
    T** __restrict__ key_buf_addrs,   // (batch) => (num_kv_heads, len_buf, dim_head)
    T** __restrict__ val_buf_addrs,   // (batch) => (num_kv_heads, len_buf, dim_head)
    const int8_t* __restrict__ mask,  // (batch) => (len_q, len_buf)
    T* __restrict__ output,           // (batch, len_q, num_kv_heads, m_query, dim_head)
    float scale,
    int m_query,
    bool BSHD) {
    const unsigned int b = blockIdx.z;
    const unsigned int head = blockIdx.y;
    const unsigned int head_kv = blockIdx.y / m_query;
    const unsigned int num_heads = gridDim.y;
    // const unsigned int num_kv_heads = num_heads / 8;
    const unsigned int q = blockIdx.x;
    const unsigned int len_q = gridDim.x;
    const unsigned int len_buf = buf_lens[b];

    g_q += ((b * len_q + q) * num_heads + head) * DIM_HEAD;    // => (dim_head)
    output += ((b * len_q + q) * num_heads + head) * O_DIM_HEAD; // => (dim_head)

    int stride_kv = BSHD ? num_heads / m_query * DIM_HEAD : DIM_HEAD;
    int offset_kv = BSHD ? head_kv * DIM_HEAD : head_kv * len_buf * DIM_HEAD;
    T* key_buf = key_buf_addrs[b] + offset_kv; // (num_kv_heads, len_buf, dim_head) or
    T* val_buf = val_buf_addrs[b] + offset_kv; // BSHD: (len_buf, num_kv_heads, dim_head)

    functions::SharedMemory<float> shared;
    float* smem = shared.getPointer();

    // Q * K (dim_head) * (len_buf, dim_head) => (len_buf);
    if constexpr(DIM_HEAD == 64) {
        multiply_q_k_block_d64<T, float, DIM_HEAD>(g_q, key_buf, smem, len_buf, stride_kv);
    } else if constexpr(DIM_HEAD != 128) {
        multiply_q_k_block1<T, float, DIM_HEAD>(g_q, key_buf, smem, len_buf, stride_kv);
    } else {
        multiply_q_k_block<T, float, DIM_HEAD>(g_q, key_buf, smem, len_buf, stride_kv);
    }
    __syncthreads();

    int len_offset = 0;
    for (int i = 0; i < b; ++i) {
        len_offset += buf_lens[i];
    }
    mask += len_q * len_offset + q * len_buf;
    static __shared__ float reduce_buffer[33];
    softmax_mask_block<T>(reduce_buffer, smem, mask, nullptr, scale, len_buf);
    __syncthreads();

    multiply_score_v_block2<T, O_DIM_HEAD>(smem, val_buf, output, len_buf, stride_kv);
}

#define NO_SPLIT 512
#define MIN_LEN_SPLIT 128
// gridDim (len_q * split, num_kv_heads * m_query, batch),  blockDim (dim_head)
template<typename T, int DIM_HEAD = 128, int O_DIM_HEAD = DIM_HEAD>
static __global__ void KERNEL_mqa_rag_buffer_split_kv(
    const T* __restrict__ g_q,         // (batch, len_q, num_kv_heads * m_query, dim_head）
    const int* __restrict__ buf_lens,  // (batch)
    T** __restrict__ key_buf_addrs,    // (batch) => (num_kv_heads, len_buf, dim_head)
    T** __restrict__ val_buf_addrs,    // (batch) => (num_kv_heads, len_buf, dim_head)
    const int8_t* __restrict__ mask,   // (batch) => (len_q, len_buf)
    T* __restrict__ output,            // (batch, len_q, num_kv_heads, m_query, dim_head)
    float* __restrict__ cache,         // (batch, len_q, num_kv_heads, m_query, num_split, dim_head)
    float* __restrict__ local_max,     // (batch, len_q, num_kv_heads, m_query, num_split)
    float* __restrict__ local_sum_exp, // (batch, len_q, num_kv_heads, m_query, num_split)
    float scale,
    int m_query,
    int num_split,
    bool BSHD) {
    const unsigned int b = blockIdx.z;
    const unsigned int head = blockIdx.y;
    const unsigned int head_kv = blockIdx.y / m_query;
    const unsigned int num_heads = gridDim.y;
    // const unsigned int num_kv_heads = num_heads / 8;
    const unsigned int q = blockIdx.x / num_split;
    const unsigned int split = blockIdx.x % num_split;
    const unsigned int len_q = gridDim.x / num_split;
    const int len_buf = buf_lens[b];

    if (len_buf <= NO_SPLIT && split > 0)
        return; // don't split small len_buf
    const int len_split = (len_buf + num_split - 1) / num_split;
    const int len_buf2 =
        len_buf <= NO_SPLIT ? len_buf : min(len_split, len_buf - split * len_split);

    const int virtual_head = (b * len_q + q) * num_heads + head;
    g_q += virtual_head * DIM_HEAD;    // => (dim_head)
    output += virtual_head * O_DIM_HEAD; // => (dim_head)

    int stride_kv = BSHD ? num_heads / m_query * DIM_HEAD : DIM_HEAD;
    int offset_kv = BSHD ? head_kv * DIM_HEAD : head_kv * len_buf * DIM_HEAD;
    offset_kv += split * len_split * stride_kv;
    T* key_buf = key_buf_addrs[b] + offset_kv; // (num_kv_heads, len_buf, dim_head) or
    T* val_buf = val_buf_addrs[b] + offset_kv; // BSHD: (len_buf, num_kv_heads, dim_head)

    functions::SharedMemory<float> shared;
    float* smem = shared.getPointer();

    // Q * K (dim_head) * (len_buf, dim_head) => (len_buf);
    if (DIM_HEAD != 128) {
        multiply_q_k_block1<T, float, DIM_HEAD>(g_q, key_buf, smem, len_buf2, stride_kv);
    } else {
        multiply_q_k_block<T, float, DIM_HEAD>(g_q, key_buf, smem, len_buf2, stride_kv);
    }
    __syncthreads();

    int len_offset = 0;
    for (int i = 0; i < b; ++i) {
        len_offset += buf_lens[i];
    }
    mask += len_q * len_offset + q * len_buf + split * len_split;
    local_max += virtual_head * num_split + split;
    local_sum_exp += virtual_head * num_split + split;
    static __shared__ float reduce_buffer[33];
    softmax_mask_block<T>(
        reduce_buffer, smem, mask, nullptr, scale, len_buf2, nullptr, 0, local_max, local_sum_exp);
    __syncthreads();

    if (len_buf <= NO_SPLIT) {
        multiply_score_v_block2<T, O_DIM_HEAD>(smem, val_buf, output, len_buf2, stride_kv);
    } else {
        cache += (virtual_head * num_split + split) * O_DIM_HEAD;
        multiply_score_v_block2<T, O_DIM_HEAD, float>(smem, val_buf, cache, len_buf2, stride_kv);
    }
}

// gridDim (len_q * split, num_kv_heads * m_query, batch),  blockDim (dim_head)
template<typename T, int DIM_HEAD = 128, int O_DIM_HEAD = DIM_HEAD>
static __global__ void KERNEL_mqa_rag_buffer_split_kv_quant(
    const half* __restrict__ g_q,      // (batch, len_q, num_kv_heads * m_query, dim_head）
    const int* __restrict__ buf_lens,  // (batch)
    uint8_t** __restrict__ key_buf_addrs,    // (batch) => (len_buf, num_kv_heads, dim_head)
    uint8_t** __restrict__ val_buf_addrs,    // (batch) => (len_buf, num_kv_heads, dim_head)
    float** __restrict__ scale_key_addrs,    // (batch) => (len_buf, num_kv_heads)
    float** __restrict__ scale_val_addrs,    // (batch) => (len_buf, num_kv_heads)
    const int8_t* __restrict__ mask,   // (batch) => (len_q, len_buf)
    T* __restrict__ output,            // (batch, len_q, num_kv_heads, m_query, dim_head)
    float* __restrict__ cache,         // (batch, len_q, num_kv_heads, m_query, num_split, dim_head)
    float* __restrict__ local_max,     // (batch, len_q, num_kv_heads, m_query, num_split)
    float* __restrict__ local_sum_exp, // (batch, len_q, num_kv_heads, m_query, num_split)
    float scale,
    int m_query,
    int num_split,
    bool BSHD) {
    const unsigned int b = blockIdx.z;
    const unsigned int head = blockIdx.y;
    const unsigned int head_kv = blockIdx.y / m_query;
    const unsigned int num_heads = gridDim.y;
    const unsigned int num_kv_heads = gridDim.y / m_query;
    // const unsigned int num_kv_heads = num_heads / 8;
    const unsigned int q = blockIdx.x / num_split;
    const unsigned int split = blockIdx.x % num_split;
    const unsigned int len_q = gridDim.x / num_split;
    const int len_buf = buf_lens[b];

    if (len_buf <= NO_SPLIT && split > 0)
        return; // don't split small len_buf
    const int len_split = (len_buf + num_split - 1) / num_split;
    const int len_buf2 =
        len_buf <= NO_SPLIT ? len_buf : min(len_split, len_buf - split * len_split);

    const int virtual_head = (b * len_q + q) * num_heads + head;
    g_q += virtual_head * DIM_HEAD;    // => (dim_head)
    output += virtual_head * O_DIM_HEAD; // => (dim_head)

    int stride_kv = num_kv_heads * DIM_HEAD;
    int offset_kv = head_kv * DIM_HEAD;
    offset_kv += split * len_split * stride_kv;
    uint8_t* key_buf = key_buf_addrs[b] + offset_kv; // (len_buf, num_kv_heads, dim_head)
    uint8_t* val_buf = val_buf_addrs[b] + offset_kv; // (len_buf, num_kv_heads, dim_head)
    float* scale_key = scale_key_addrs[b] + head_kv; // (len_buf, num_kv_heads)
    float* scale_val = scale_val_addrs[b] + head_kv; // (len_buf, num_kv_heads)
    functions::SharedMemory<float> shared;
    float* smem = shared.getPointer();

    // Q * K:  (dim_head) * (len_buf, dim_head) => (len_buf);
    assert(DIM_HEAD == 128);
    DEV_mul_qk_quant_h128<T, float>(g_q, key_buf, smem, len_buf2, stride_kv);
    __syncthreads();
    DEV_mul_logit_scale(smem, scale_key, len_buf2, num_kv_heads);
    __syncthreads();

    int len_offset = 0;
    for (int i = 0; i < b; ++i) {
        len_offset += buf_lens[i];
    }
    mask += len_q * len_offset + q * len_buf + split * len_split;
    local_max += virtual_head * num_split + split;
    local_sum_exp += virtual_head * num_split + split;
    static __shared__ float reduce_buffer[33];
    softmax_mask_block<T>(
        reduce_buffer, smem, mask, nullptr, scale, len_buf2, nullptr, 0, local_max, local_sum_exp);
    __syncthreads();
    DEV_mul_logit_scale(smem, scale_val, len_buf2, num_kv_heads);
    __syncthreads();

    if (len_buf <= NO_SPLIT) {
        DEV_mul_score_v_v1<T, O_DIM_HEAD>(smem, val_buf, output, len_buf2, stride_kv);
    } else {
        cache += (virtual_head * num_split + split) * O_DIM_HEAD;
        DEV_mul_score_v_v1<T, O_DIM_HEAD, float>(smem, val_buf, cache, len_buf2, stride_kv);
    }
}

// gridDim (num_virtual_heads / batch, batch),  blockDim (dim_head)
template<typename T>
static __global__ void KERNEL_mqa_combine(
    const int* __restrict__ buf_lens, // (batch)
    T* __restrict__ output,           // (batch, len_q, num_kv_heads, m_query, dim_head)
    float* __restrict__ cache,        // (batch, len_q, num_kv_heads, m_query, num_split, dim_head)
    float* __restrict__ g_local_max,  // (batch, len_q, num_kv_heads, m_query, num_split)
    float* __restrict__ g_local_sum_exp, // (batch, len_q, num_kv_heads, m_query, num_split)
    int num_split) {
    const int DIM_HEAD = blockDim.x;
    if (buf_lens[blockIdx.y] <= NO_SPLIT)
        return;
    int h = blockIdx.y * gridDim.x + blockIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    float local_max = -1e20;
    float local_sum_exp = 0.;
    if (laneId < num_split) {
        local_max = g_local_max[h * num_split + laneId];
        local_sum_exp = g_local_sum_exp[h * num_split + laneId];
    }
    float global_max = functions::warpReduceMax<float>(local_max);
    global_max = __shfl_sync(0xFFFFFFFF, global_max, 0);
    float scale1 = laneId < num_split ? expf(local_max - global_max) : 0.;
    float local_sum_exp2 = local_sum_exp * scale1;
    float global_sum_exp = functions::warpReduceSum<float>(local_sum_exp2);
    global_sum_exp = __shfl_sync(0xFFFFFFFF, global_sum_exp, 0);
    float scale2 = local_sum_exp / global_sum_exp * scale1;
    //    if (threadIdx.x == 1 && h == 1)
    //        printf("h=%d local_max=%f, global_max=%f, local_sum_exp2=%f, global_sum_exp=%f\n", h,
    //        local_max, global_max, local_sum_exp2, global_sum_exp); printf("h=%d local_max=%f,
    //        global_max=%f\n", h, local_max, global_max);

    float res = 0;
    cache += (h * num_split) * DIM_HEAD + threadIdx.x;
    for (int i = 0; i < num_split; ++i) {
        // float v = cache[(h * num_split + i) * DIM_HEAD + threadIdx.x];
        float v = cache[i * DIM_HEAD];
        float scale = __shfl_sync(0xFFFFFFFF, scale2, i);
        //        if (threadIdx.x == 0 && h == 1)
        //            printf("h=%d v=%f, scale=%f\n", h, v, scale);
        res += v * scale;
    }
    output[h * DIM_HEAD + threadIdx.x] = res;
}

// gridDim (len_q, num_kv_heads, batch),  blockDim (dim_head)
template<typename T, int DIM_HEAD = 128, int O_DIM_HEAD = DIM_HEAD, int M_QUERY = 8>
static __global__ void KERNEL_mqa_rag_buffer(
    const T* __restrict__ g_q,        // (batch, len_q, num_kv_heads * m_query, dim_head）
    const int* __restrict__ buf_lens, // (batch)
    T** __restrict__ key_buf_addrs,   // (batch) => (num_kv_heads, len_buf, dim_head)
    T** __restrict__ val_buf_addrs,   // (batch) => (num_kv_heads, len_buf, dim_head)
    const int8_t* __restrict__ mask,  // (batch) => (len_q, len_buf)
    T* __restrict__ output,           // (batch, len_q, num_kv_heads * m_query, dim_head)
    float scale) {
#if (__CUDA_ARCH__ >= 800) || !defined(__linux__)
    const unsigned int b = blockIdx.z;
    const unsigned int head = blockIdx.y;
    const unsigned int num_kv_heads = gridDim.y;
    const unsigned int q = blockIdx.x;
    const unsigned int len_q = gridDim.x;
    const unsigned int len_buf = buf_lens[b];

    g_q += ((b * len_q + q) * num_kv_heads + head) * M_QUERY * DIM_HEAD;
    output += ((b * len_q + q) * num_kv_heads + head) * M_QUERY * O_DIM_HEAD;

    T* g_k = key_buf_addrs[b]; // (num_heads, len_buf, dim_head)
    T* g_v = val_buf_addrs[b]; // (num_heads, len_buf, dim_head)
    g_k += head * len_buf * DIM_HEAD;
    g_v += head * len_buf * DIM_HEAD;

    int len_offset = 0;
    for (int i = 0; i < b; ++i) {
        len_offset += buf_lens[i]; // accumulate to current batch
    }
    mask += len_q * len_offset + q * len_buf;

    extern __shared__ __align__(32) double4 smem[];
    float* align_mem = reinterpret_cast<float*>(smem);
    while (reinterpret_cast<uintptr_t>(align_mem) % 32 != 0)
        align_mem++;                               // align to 32
    T* s_q = reinterpret_cast<T*>(align_mem);      // (M_QUERY, DIM_HEAD)
    float* score = align_mem + M_QUERY * DIM_HEAD; // (M_QUERY, len_buf)
    T* score_t = reinterpret_cast<T*>(score);

    // copy q from global memory to shared memory
    for (int i = threadIdx.x; i < M_QUERY * DIM_HEAD; i += blockDim.x) {
        s_q[i] = g_q[i];
    }
    __syncthreads();

    // Q * K: (M_QUERY, DIM_HEAD) * (len_buf, DIM_HEAD) => (M_QUERY, len_buf)
    // multiply_q_k_block_mq1(s_q, g_k, score, len_buf);
    if constexpr(M_QUERY == 8) {
        multiply_q_k_block_wmma<T, 8, 32, 16, DIM_HEAD>(s_q, g_k, score, len_buf);
    } else if constexpr(M_QUERY == 16) {
        multiply_q_k_block_wmma<T, 16, 16, 16, DIM_HEAD>(s_q, g_k, score, len_buf);
    } else if constexpr(M_QUERY == 32) {
        multiply_q_k_block_wmma<T, 32, 8, 16, DIM_HEAD>(s_q, g_k, score, len_buf);
    }
    __syncthreads();

    // softmax
    static __shared__ float reduce_buffer[33];
#pragma unroll
    for (int q1 = 0; q1 < M_QUERY; ++q1) {
        softmax_mask_block<T>(reduce_buffer, score + q1 * len_buf, mask, nullptr, scale, len_buf);
    }
    __syncthreads();

    // multiply_score_v_block_mq1(score, g_v, output, len_buf);

    // cast score from float to half inline
    cast_inline_block<T>(score, M_QUERY * len_buf); // (M_QUERY, len_buf)

    // Score * V
    float* float_out = reinterpret_cast<float*>(s_q);
    if constexpr (O_DIM_HEAD == 512) {
        if constexpr(M_QUERY == 8) {
            multiply_score_v_block_wmma1<T, 8, 32, 16, 512, 512>(score_t, g_v, output, len_buf, float_out, DIM_HEAD);
        } else if constexpr(M_QUERY == 16) {
            multiply_score_v_block_wmma1<T, 16, 16, 16, 512, 512>(score_t, g_v, output, len_buf, float_out, DIM_HEAD);
        } else if constexpr(M_QUERY == 32) {
            multiply_score_v_block_wmma1<T, 32, 8, 16, 512, 512>(score_t, g_v, output, len_buf, float_out, DIM_HEAD);
        }
    } else if (len_buf < 512) {
        multiply_score_v_block_wmma1<T, 8, 32>(score_t, g_v, output, len_buf, float_out);
    } else {
        multiply_score_v_block_wmma2<T, 8, 32>(score_t, g_v, output, len_buf, float_out);
//        if constexpr(M_QUERY == 8) {
//        } else if constexpr(M_QUERY == 16) {
//            multiply_score_v_block_wmma2<T, 16, 16>(score_t, g_v, output, len_buf, float_out);
//        } else if constexpr(M_QUERY == 32) {
//            multiply_score_v_block_wmma2<T, 32, 8>(score_t, g_v, output, len_buf, float_out);
//        }
    }
#endif
}

// gridDim (len_q, num_kv_heads, 1),  blockDim (dim_head)
template<typename T, int DIM_HEAD = 128, int M_QUERY = 8, int THREAD_NUM = 1024>
static __global__ void KERNEL_mqa_self(
    const T* __restrict__ g_q,       // (len_q, num_kv_heads * m_query, dim_head）
    const unsigned int len_buf,      // (batch)
    const T* __restrict__ g_k,       // (num_kv_heads, len_buf, dim_head)
    const T* __restrict__ g_v,       // (num_kv_heads, len_buf, dim_head)
    const int8_t* __restrict__ mask, // (len_q, len_buf)
    T* __restrict__ output,          // (len_q, num_kv_heads * m_query, dim_head)
    float scale,
    int high_precision) {
    const unsigned int head = blockIdx.y;
    const unsigned int num_kv_heads = gridDim.y;
    const unsigned int q = blockIdx.x;

    g_q += (q * num_kv_heads + head) * M_QUERY * DIM_HEAD;
    output += (q * num_kv_heads + head) * M_QUERY * DIM_HEAD;

    g_k += head * len_buf * DIM_HEAD; // (len_buf, dim_head)
    g_v += head * len_buf * DIM_HEAD; // (len_buf, dim_head)

    mask += q * len_buf;

    extern __shared__ __align__(32) double4 smem[];
    float* align_mem = reinterpret_cast<float*>(smem);
    while (reinterpret_cast<uintptr_t>(align_mem) % 32 != 0)
        align_mem++;                               // align to 32
    T* s_q = reinterpret_cast<T*>(align_mem);      // (M_QUERY, DIM_HEAD)
    float* score = align_mem + M_QUERY * DIM_HEAD; // (M_QUERY, len_buf)
    T* score_t = reinterpret_cast<T*>(score);

    // copy Q from global memory to shared memory
    copy_8x128<THREAD_NUM>(s_q, g_q);
    __syncthreads();

    const unsigned int len_buf_cut = (q + 32) / 32 * 32; // little speed up after cut
    // const unsigned int len_buf_cut = len_buf;
    // Q * K  (m_query, dim_head) * (len_buf, dim_head) => (m_query, len_buf)
    if (high_precision)
        multiply_q_k_block_mq1(s_q, g_k, score, len_buf_cut);
    else
        multiply_q_k_block_wmma<T>(s_q, g_k, score, len_buf_cut);
    __syncthreads();

    // Softmax
    static __shared__ float reduce_buffer[33];
#pragma unroll
    for (int q1 = 0; q1 < M_QUERY; ++q1) {
        softmax_mask_block<T>(
            reduce_buffer, score + q1 * len_buf_cut, mask, nullptr, scale, len_buf_cut);
    }
    __syncthreads();

    // Score * V (m_query, len_buf) * (len_buf, dim_head) => (m_query, dim_head)
    if (high_precision > 0) {
        multiply_score_v_block_mq1(score, g_v, output, len_buf_cut);
    } else {
        // cast score from float to half inline
        cast_inline_block<T>(score, M_QUERY * len_buf_cut);

        if (len_buf < 1024) {
            multiply_score_v_block_wmma1<T>(
                score_t, g_v, output, len_buf_cut, reinterpret_cast<float*>(s_q));
        } else {
            multiply_score_v_block_wmma2<T>(
                score_t, g_v, output, len_buf_cut, reinterpret_cast<float*>(s_q));
        }
    }
}

void mul_qk_rag_buffer(
    const core::Context& ctx,
    const core::Tensor& batch_q,       // (batch, num_kv_heads, n_rep * len_q, dim_head）
    const core::Tensor& buf_lens,      // (batch)
    const core::Tensor& key_buf_addrs, // (batch) => (num_kv_heads, len_buf, dim_head)
    core::Tensor& total_score) {
    BM_ASSERT_EQ(batch_q.size(0), buf_lens.size(0), "batch mismatch");
    BM_ASSERT_EQ(batch_q.ndim(), 4, "batch_q is not 4d");
    BM_ASSERT_EQ(batch_q.size(0), key_buf_addrs.numel(), "batch mismatch");
    BM_ASSERT_EQ(batch_q.size(-1), 128, "dim_head mismatch");

    BM_ASSERT(batch_q.dtype() == bmengine::core::DataType::kHalf, "not half");

    dim3 gridDim(batch_q.size(0), batch_q.size(1), batch_q.size(2));
    auto stream = ctx.current_stream()->ptr;

    BM_KERNEL(mul_qk_rag_buffer)<<<gridDim, 512, 0, stream>>>(
        batch_q.data<__half>(),
        buf_lens.data<int>(),
        key_buf_addrs.data<__half*>(),
        total_score.mutable_data<__half>());

    BM_CUDART_ASSERT(cudaGetLastError());
}

void mul_qk_softmax_rag_buffer(
    const core::Context& ctx,
    const core::Tensor& batch_q,       // (batch, num_kv_heads, n_rep * len_q, dim_head）
    const core::Tensor& buf_lens,      // (batch)
    const core::Tensor& key_buf_addrs, // (batch) => (num_kv_heads, len_buf, dim_head)
    const core::Tensor& mask,          // (batch) => (len_q, len_buf)
    const core::Tensor& position_bias, // (batch) => (num_kv_heads, len_q, len_buf)
    float scale,
    int max_len_buf,
    core::Tensor& total_score // (batch) => (num_kv_heads, len_q, len_buf)
) {
    BM_ASSERT_EQ(batch_q.size(0), buf_lens.size(0), "batch mismatch");
    BM_ASSERT_EQ(batch_q.ndim(), 4, "batch_q is not 4d");
    BM_ASSERT_EQ(batch_q.size(0), key_buf_addrs.numel(), "batch mismatch");
    BM_ASSERT_EQ(batch_q.size(-1), 128, "dim_head mismatch");
    //    BM_ASSERT_EQ(position_bias.numel(), total_score.numel(), "dim_head mismatch");

    BM_ASSERT(batch_q.dtype() == bmengine::core::DataType::kHalf, "not half");

    dim3 gridDim(batch_q.size(0), batch_q.size(1), batch_q.size(2));
    auto stream = ctx.current_stream()->ptr;

    size_t dynamic_size = max_len_buf * sizeof(float);

    BM_KERNEL(mul_qk_softmax_rag_buffer)<<<gridDim, 256, dynamic_size, stream>>>(
        batch_q.data<__half>(),
        buf_lens.data<int>(),
        key_buf_addrs.data<__half*>(),
        mask.data<int8_t>(),
        position_bias.numel() ? position_bias.data<__half*>() : nullptr,
        total_score.mutable_data<__half>(),
        scale);

    BM_CUDART_ASSERT(cudaGetLastError());
}

void attention_qkv_rag_buffer(
    const core::Context& ctx,
    const core::Tensor& batch_q,       // (batch, len_q, num_heads, dim_head）
    const core::Tensor& buf_lens,      // (batch)
    const core::Tensor& key_buf_addrs, // (batch) => (num_heads, len_buf, dim_head)
    const core::Tensor& val_buf_addrs, // (batch) => (num_heads, len_buf, dim_head)
    const core::Tensor& mask,          // (batch) => (len_q, len_buf)
    const core::Tensor& position_bias, // (batch) => (num_heads, len_q, len_buf)
    float scale,
    int max_len_buf,
    core::Tensor& output // (batch, len_q, num_heads, dim_head
) {
    BM_ASSERT_EQ(batch_q.ndim(), 4, "batch_q is not 4d");
    BM_ASSERT_EQ(batch_q.size(0), buf_lens.size(0), "batch mismatch");
    BM_ASSERT_EQ(batch_q.size(0), key_buf_addrs.numel(), "batch mismatch");
    BM_ASSERT_EQ(batch_q.size(0), val_buf_addrs.numel(), "batch mismatch");
    if (position_bias.numel()) {
        BM_ASSERT_EQ(batch_q.size(0), position_bias.numel(), "batch mismatch");
    }
    size_t dim_head = batch_q.size(-1);
    BM_ASSERT(dim_head == 128 || dim_head == 64 || dim_head == 192, "dim_head mismatch");
    if (dim_head == 192) {
        // MLA: v_dim_head = 128
        BM_ASSERT_EQ(128UL, output.size(-1), "v_dim_head mismatch");
        BM_ASSERT_EQ(batch_q.numel() / dim_head, output.numel() / 128UL, "shape mismatch");
    } else {
        BM_ASSERT_EQ(batch_q.shape(), output.shape(), "shape mismatch");
    }

    BM_ASSERT(
        batch_q.dtype() == DataType::kHalf || batch_q.dtype() == DataType::kBFloat16, "not half");

    dim3 gridDim(batch_q.size(0), batch_q.size(2), batch_q.size(1));
    auto stream = ctx.current_stream()->ptr;

    size_t dynamic_size = max_len_buf * sizeof(float) + 1024 + dim_head * sizeof(half);

    auto attr = cudaFuncAttributeMaxDynamicSharedMemorySize;
    BM_DTYPE_DISPATCH_HALF(batch_q.dtype(), {
        auto kernel = KERNEL_attn_qpk_rag_buffer<scalar_t, 128>;
        if (dim_head == 64) {
            kernel = KERNEL_attn_qpk_rag_buffer<scalar_t, 64>;
        } else if (dim_head == 128) {
            kernel = KERNEL_attn_qpk_rag_buffer<scalar_t, 128>;
        } else if (dim_head == 192) {
            // MLA: v_dim_head = 128
            kernel = KERNEL_attn_qpk_rag_buffer<scalar_t, 192, 128>;
        }

        BM_CUDART_ASSERT(cudaFuncSetAttribute(kernel, attr, dynamic_size));
        kernel<<<gridDim, 1024, dynamic_size, stream>>>(
            batch_q.data<scalar_t>(),
            buf_lens.data<int>(),
            key_buf_addrs.data<scalar_t*>(),
            val_buf_addrs.data<scalar_t*>(),
            mask.data<int8_t>(),
            position_bias.numel() ? position_bias.data<scalar_t*>() : nullptr,
            output.mutable_data<scalar_t>(),
            scale,
            ctx.is_BSHD()
        );
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

static int get_max_num_split(
    const core::Context& ctx,
    const core::Tensor& batch_q,  // (batch, len_q, num_kv_heads * m_query, dim_head)
    int max_len_buf,
    bool is_quantized) {
    size_t num_virtual_heads = batch_q.numel() / batch_q.size(-1);
    int num_split = 1;
    if (num_virtual_heads < ctx.get_mp_count()) {
        std::min(ctx.get_mp_count() / int(num_virtual_heads), 32);
    }
    // multiply_score_v_block2 use 1024 * 4
    int max_len_mem = (ctx.get_max_shared_memory() - 4096 - 1024 - 32 - 256) / sizeof(float);
    if (max_len_buf > max_len_mem) {
        num_split = std::max(num_split, max_len_buf / max_len_mem + 1);
    }
    if (is_quantized) {
        num_split = std::max(num_split, (max_len_buf + 1023) / 1024);
    }
    BM_ASSERT(num_split > 0, "");
    return num_split;
}

AttentionWorkspace get_mqa_workspace(
    const core::Context& ctx,
    const core::Tensor& batch_q,  // (batch, len_q, num_kv_heads * m_query, dim_head)
    int max_len_buf,
    bool is_quantized
) {
    size_t dim_head = batch_q.size(-1);
    size_t num_virtual_heads = batch_q.numel() / batch_q.size(-1);
    size_t num_split = get_max_num_split(ctx, batch_q, max_len_buf, is_quantized);
    Tensor cache = ctx.tensor({ num_virtual_heads, num_split, dim_head }, DataType::kFloat);
    Tensor local_max = ctx.tensor({ num_virtual_heads, num_split }, DataType::kFloat);
    Tensor local_sum_exp = ctx.tensor({ num_virtual_heads, num_split }, DataType::kFloat);
    return {cache, local_max, local_sum_exp};
}

void multi_query_attention_rag_buffer(
    const core::Context& ctx,
    const core::Tensor& batch_q,       // (batch, len_q, num_kv_heads * m_query, dim_head)
    const core::Tensor& buf_lens,      // (batch)
    const core::Tensor& key_buf_addrs, // (batch) => (num_kv_heads, len_buf, dim_head)
    const core::Tensor& val_buf_addrs, // (batch) => (num_kv_heads, len_buf, dim_head)
    const core::Tensor& mask,          // (batch) => (len_q, len_buf)
    const float scale,
    const int max_len_buf,
    core::Tensor& output, // (batch, len_q, num_kv_heads * m_query, dim_head)
    const int m_query,
    int algo_id,
    const AttentionWorkspace& ws,
    const core::Tensor& scale_key_addrs,
    const core::Tensor& scale_val_addrs,
    core::DataType dequant_dtype) {
    BM_ASSERT_EQ(ctx.active_device(), batch_q.device(), "device mismatch");
    BM_ASSERT_EQ(ctx.active_device(), buf_lens.device(), "device mismatch");
    BM_ASSERT_EQ(ctx.active_device(), key_buf_addrs.device(), "device mismatch");
    BM_ASSERT_EQ(ctx.active_device(), val_buf_addrs.device(), "device mismatch");
    BM_ASSERT_EQ(ctx.active_device(), mask.device(), "device mismatch");
    BM_ASSERT_EQ(ctx.active_device(), output.device(), "device mismatch");

    size_t batch_size = batch_q.size(0);
    BM_ASSERT_EQ(batch_q.ndim(), 4, "batch_q is not 4d");
    BM_ASSERT_EQ(batch_size, buf_lens.size(0), "batch mismatch");
    BM_ASSERT_EQ(batch_size, key_buf_addrs.numel(), "batch mismatch");
    BM_ASSERT_EQ(batch_size, val_buf_addrs.numel(), "batch mismatch");
    const size_t dim_head = batch_q.size(-1);
    BM_ASSERT(dim_head == 64 || dim_head == 128 || dim_head == 576, "dim_head mismatch");
    if (dim_head == 576) {
        BM_ASSERT_EQ(output.size(-1), 512, "dim_head mismatch");
    } else {
        BM_ASSERT_EQ(batch_q.shape(), output.shape(), "shape mismatch");
    }

    BM_ASSERT(
        batch_q.dtype() == DataType::kHalf || batch_q.dtype() == DataType::kBFloat16, "not half");

    auto stream = ctx.current_stream()->ptr;
    auto attr = cudaFuncAttributeMaxDynamicSharedMemorySize;
    const int len_q = batch_q.size(1);
    const int num_kv_heads = batch_q.size(2) / m_query;
    const size_t num_virtual_heads = batch_q.numel() / batch_q.size(-1);

    static int split_kv_thres = utils::get_int_env("ATTN_SPLIT_KV_THRES", 1024);
    size_t num_split = 1;
    if (max_len_buf > split_kv_thres) {
        num_split = get_max_num_split(ctx, batch_q, max_len_buf, !scale_key_addrs.empty());
    }
    // std::cout << "max_len_buf=" << max_len_buf << ", num_split=" << num_split << ", split_kv_thres=" << split_kv_thres << endl;

    static bool use_mma = utils::get_int_env("CPM_MQ_ATTN_MMA", 0) && m_query == 8;
    if (algo_id == -1 || batch_q.dtype() == DataType::kBFloat16)
        algo_id = 1; // default to 1
    if (num_split > 1 && algo_id == 1) {
        dim3 gridDim(len_q * num_split, num_kv_heads * m_query, batch_size);
        int max_len_split = (max_len_buf + num_split - 1) / num_split;
        size_t dynamic_size = max_len_split * sizeof(float) + dim_head * sizeof(half) + 32;
        BM_ASSERT_LE(dynamic_size, ctx.get_max_shared_memory(), " dynamic_size too big");

        BM_ASSERT(ws.cache.numel(), "workspace not created");

        auto out_dtype = batch_q.dtype();
        if (scale_key_addrs.numel()) {
            BM_ASSERT(dim_head == 128, "Unsupported dim_head");
            BM_ASSERT_EQ(batch_q.dtype(), core::DataType::kHalf, "input must be half");
            out_dtype = dequant_dtype;
            BM_DTYPE_DISPATCH_HALF(dequant_dtype, {
                auto kernel = KERNEL_mqa_rag_buffer_split_kv_quant<scalar_t, 128>;
                BM_CUDART_ASSERT(cudaFuncSetAttribute(kernel, attr, dynamic_size));
                kernel<<<gridDim, 1024, dynamic_size, stream>>>(
                    batch_q.data<half>(), // must be half
                    buf_lens.data<int>(),
                    key_buf_addrs.data<uint8_t*>(),
                    val_buf_addrs.data<uint8_t*>(),
                    scale_key_addrs.data<float*>(),
                    scale_val_addrs.data<float*>(),
                    mask.data<int8_t>(),
                    output.mutable_data<scalar_t>(),
                    ws.cache.data<float>(),
                    ws.local_max.data<float>(),
                    ws.local_sum_exp.data<float>(),
                    scale,
                    m_query,
                    num_split,
                    ctx.is_BSHD()
                );
            });
            BM_CUDART_ASSERT(cudaGetLastError());
        } else {
            BM_DTYPE_DISPATCH_HALF(out_dtype, {
                auto kernel = KERNEL_mqa_rag_buffer_split_kv<scalar_t, 128>;
                if (dim_head == 64) {
                    kernel = KERNEL_mqa_rag_buffer_split_kv<scalar_t, 64>;
                } else if (dim_head == 576) {
                    kernel = KERNEL_mqa_rag_buffer_split_kv<scalar_t, 576, 512>;
                }
                BM_CUDART_ASSERT(cudaFuncSetAttribute(kernel, attr, dynamic_size));
                kernel<<<gridDim, 1024, dynamic_size, stream>>>(
                batch_q.data<scalar_t>(),
                buf_lens.data<int>(),
                key_buf_addrs.data<scalar_t*>(),
                val_buf_addrs.data<scalar_t*>(),
                mask.data<int8_t>(),
                output.mutable_data<scalar_t>(),
                ws.cache.data<float>(),
                ws.local_max.data<float>(),
                ws.local_sum_exp.data<float>(),
                scale,
                m_query,
                num_split,
                ctx.is_BSHD());
            });
            //        std::cout << "cache: " << cache << endl;
            //        std::cout << "local_max: " << local_max << endl;
            //        std::cout << "local_sum_exp: " << local_sum_exp << endl;
            BM_CUDART_ASSERT(cudaGetLastError());
        }
        dim3 gridDim2(num_virtual_heads / batch_size, batch_size);
        BM_DTYPE_DISPATCH_HALF(out_dtype, {
            KERNEL_mqa_combine<<<gridDim2, output.size(-1), 0, stream>>>(
                buf_lens.data<int>(),
                output.mutable_data<scalar_t>(),
                ws.cache.data<float>(),
                ws.local_max.data<float>(),
                ws.local_sum_exp.data<float>(),
                num_split);
        });
        BM_CUDART_ASSERT(cudaGetLastError());
    } else if (!use_mma && algo_id == 1) {
        dim3 gridDim(len_q, num_kv_heads * m_query, batch_size);
        size_t dynamic_size = max_len_buf * sizeof(float) + dim_head * sizeof(half) + 32;

        BM_DTYPE_DISPATCH_HALF(batch_q.dtype(), {
            auto kernel = KERNEL_mqa_rag_buffer1<scalar_t, 128>;
            if (dim_head == 64) {
                kernel = KERNEL_mqa_rag_buffer1<scalar_t, 64>;
            } else if (dim_head == 576) {
                kernel = KERNEL_mqa_rag_buffer1<scalar_t, 576, 512>;
            }
            cudaFuncAttributes attrs;
            BM_CUDART_ASSERT(cudaFuncGetAttributes(&attrs, kernel));
            // std::cerr << "attrs.sharedSizeBytes:" << attrs.sharedSizeBytes << endl;
            BM_ASSERT_LE(dynamic_size, ctx.get_max_shared_memory() - attrs.sharedSizeBytes, " dynamic_size too big");

            BM_CUDART_ASSERT(cudaFuncSetAttribute(kernel, attr, dynamic_size));
            kernel<<<gridDim, 1024, dynamic_size, stream>>>(
                batch_q.data<scalar_t>(),
                buf_lens.data<int>(),
                key_buf_addrs.data<scalar_t*>(),
                val_buf_addrs.data<scalar_t*>(),
                mask.data<int8_t>(),
                output.mutable_data<scalar_t>(),
                scale,
                m_query,
                ctx.is_BSHD());
        });
        BM_CUDART_ASSERT(cudaGetLastError());
    } else {
        BM_ASSERT(m_query == 8 || m_query == 16 || m_query == 32, "");
        BM_ASSERT(batch_q.dtype() == bmengine::core::DataType::kHalf, "not half");
        // wmma implementation
        int max_shared_memory = ctx.get_max_shared_memory() - 1024;

        dim3 gridDim(len_q, num_kv_heads, batch_q.size(0));
        int threads = dim_head == 576 ? 512 : 1024;
        size_t mem_q = m_query * dim_head * sizeof(float);  // float for float_out
        size_t mem_score = m_query * (std::max(dim_head, size_t(max_len_buf))) * sizeof(float);
        size_t dynamic_size = mem_q + mem_score;
        dynamic_size += 32; // align
        BM_ASSERT_LE(dynamic_size, max_shared_memory, " dynamic_size too big");

        {
            auto kernel = KERNEL_mqa_rag_buffer<__half, 128, 128, 8>;
            if (dim_head == 128) {
                // default
            } else if (dim_head == 576) {
                kernel = KERNEL_mqa_rag_buffer<__half, 576, 512, 8>;
                if (m_query == 16) {
                    kernel = KERNEL_mqa_rag_buffer<__half, 576, 512, 16>;
                } else if (m_query == 32) {
                    kernel = KERNEL_mqa_rag_buffer<__half, 576, 512, 32>;
                }
            } else {
                throw std::runtime_error("Unsupported dim_head");
            }
//            if (m_query == 16) {
//                kernel = KERNEL_mqa_rag_buffer<__half, 128, 16>;
//            } else if (m_query == 32) {
//                kernel = KERNEL_mqa_rag_buffer<__half, 128, 32>;
//            }
            BM_CUDART_ASSERT(cudaFuncSetAttribute(kernel, attr, dynamic_size));
            kernel<<<gridDim, threads, dynamic_size, stream>>>(
                batch_q.data<__half>(),
                buf_lens.data<int>(),
                key_buf_addrs.data<__half*>(),
                val_buf_addrs.data<__half*>(),
                mask.data<int8_t>(),
                output.mutable_data<__half>(),
                scale);
        }
    }

    BM_CUDART_ASSERT(cudaGetLastError());
}

void multi_query_self_attention(
    const core::Context& ctx,
    const core::Tensor& query,   // (len_q, num_kv_heads * m_query, dim_head)
    const core::Tensor& key_buf, // (num_kv_heads, len_buf, dim_head)
    const core::Tensor& val_buf, // (num_kv_heads, len_buf, dim_head)
    const core::Tensor& mask,    // (len_q, len_buf)
    float scale,
    core::Tensor& output, // (len_q, num_kv_heads * m_query, dim_head)
    int high_precision) {
    BM_ASSERT_EQ(query.ndim(), 3, "q is not 3d");
    BM_ASSERT_EQ(query.size(0), mask.size(0), "len_q mismatch");
    BM_ASSERT_EQ(key_buf.size(1), mask.size(1), "len_buf mismatch");
    const int dim_head = 128;
    BM_ASSERT_EQ(query.size(-1), dim_head, "dim_head mismatch");
    BM_ASSERT_EQ(query.shape(), output.shape(), "shape mismatch");
    BM_ASSERT_EQ(key_buf.shape(), val_buf.shape(), "shape mismatch");

    BM_ASSERT(query.dtype() == bmengine::core::DataType::kHalf, "not half");

    auto stream = ctx.current_stream()->ptr;
    const int len_q = query.size(0);
    const int len_buf = key_buf.size(1);
    const int m_query = 8;
    const int num_kv_heads = query.size(1) / m_query;

    {
        dim3 gridDim(len_q, num_kv_heads);
        size_t dynamic_size = 8 * (dim_head + std::max(2048, len_buf)) * sizeof(float) + 32;

        BM_CUDART_ASSERT(cudaFuncSetAttribute(
            KERNEL_mqa_self<__half, 128, 8>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_size));
        KERNEL_mqa_self<__half, 128, 8><<<gridDim, 1024, dynamic_size, stream>>>(
            query.data<__half>(),
            len_buf,
            key_buf.data<__half>(),
            val_buf.data<__half>(),
            mask.data<int8_t>(),
            output.mutable_data<__half>(),
            scale,
            high_precision);
    }

    BM_CUDART_ASSERT(cudaGetLastError());
}
// clang-format on
}
