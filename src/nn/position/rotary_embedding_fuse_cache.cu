// Fuse rope cuda kernel with cos, sin cache.
// Author: spetrel@gmail.com

#include "rotary_embedding.h"
#include "rope_common.cuh"
#include <bmengine/core/core.h>
#include <bmengine/functions/utils.cuh>
#include <bmengine/functions/reduce.cuh>
#include <bmengine/logger/std_log_op.hpp>
#include "utils/env.h"
#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <vector_types.h>

namespace nn {

using bmengine::core::DataType;
using bmengine::core::Tensor;

// gridDim (seq_len, all_num_heads),  blockDim (dim_head)
template<typename T_IN, typename T>
static __global__ void KERNEL_rope_qk_with_cache(
    const float *__restrict__ g_cos, // (seq_len, dim_head)
    const float *__restrict__ g_sin, // (seq_len, dim_head)
    const T_IN *__restrict__ in,     // (seq_len, all_num_heads * dim_head)
    T *__restrict__ q,               // (seq_len, num_heads * dim_head)
    T *__restrict__ k,               // (seq_len, num_kv_heads * dim_head)
    T *__restrict__ v,               // (seq_len, num_kv_heads * dim_head)
    int all_num_heads,
    int num_heads,
    int num_kv_heads,
    int dim_head,
    float rope_theta,
    bool neox_style) {
    const int head = blockIdx.y;
    int col = threadIdx.x;
    int half_dim = dim_head / 2;
    int offset = ((blockIdx.x * all_num_heads) + head) * dim_head + col;

    // v
    if (head >= num_heads + num_kv_heads) {
        int row = head - num_heads - num_kv_heads;
        int offset_v = ((blockIdx.x * num_kv_heads) + row) * dim_head + col;
        v[offset_v] = in[offset];
        return;
    }

    float cos_freq = g_cos[blockIdx.x * dim_head + col];
    float sin_freq = g_sin[blockIdx.x * dim_head + col];
    float t = rope_one_value(in, offset, cos_freq, sin_freq, col, half_dim, neox_style);

    if (head >= num_heads) {
        // k
        int row = head - num_heads;
        int offset_k = ((blockIdx.x * num_kv_heads) + row) * dim_head + col;
        k[offset_k] = t;
    } else {
        // q
        int offset_q = ((blockIdx.x * num_heads) + head) * dim_head + col;
        q[offset_q] = t;
    }
}

void rope_qk_cache(
    const core::Context& ctx,
    const core::Tensor& cos, // (seq_len, dim_head)
    const core::Tensor& sin, // (seq_len, dim_head)
    const core::Tensor& in,  // (seq_len, all_num_heads * dim_head)
    core::Tensor& out_q,     // (seq_len, num_heads * dim_head)
    core::Tensor& out_k,     // (seq_len, num_kv_heads * dim_head)
    core::Tensor& out_v,     // (seq_len, num_kv_heads * dim_head)
    size_t num_heads,
    size_t num_kv_heads,
    size_t dim_head,
    core::DataType dtype,
    bool neox_style) {
    size_t seq_len = in.size(0);
    BM_ASSERT_EQ(cos.size(0), in.size(0), "batch mismatch");
    BM_ASSERT_LT(in.numel(), INT_MAX, "number bigger than INT_MAX");

    out_q = ctx.tensor({seq_len, num_heads * dim_head}, dtype);
    out_k = ctx.tensor({seq_len, num_kv_heads * dim_head}, dtype);
    out_v = ctx.tensor({seq_len, num_kv_heads * dim_head}, dtype);

    dim3 gridDim(seq_len, num_heads + 2 * num_kv_heads);
    auto stream = ctx.current_stream()->ptr;

    float rope_theta = 1.; // ignore
    if (in.dtype() == core::DataType::kFloat) {
        BM_DTYPE_DISPATCH_FLOAT(dtype, {
            KERNEL_rope_qk_with_cache<<<gridDim, dim_head, 0, stream>>>(
                cos.data<float>(),
                sin.data<float>(),
                in.data<float>(),
                out_q.mutable_data<scalar_t>(),
                out_k.mutable_data<scalar_t>(),
                out_v.mutable_data<scalar_t>(),
                num_heads + 2 * num_kv_heads,
                num_heads,
                num_kv_heads,
                dim_head,
                rope_theta,
                neox_style);
        });
    } else {
        BM_ASSERT_EQ(in.dtype(), dtype, "");
        BM_DTYPE_DISPATCH_FLOAT(dtype, {
            KERNEL_rope_qk_with_cache<scalar_t><<<gridDim, dim_head, 0, stream>>>(
                cos.data<float>(),
                sin.data<float>(),
                in.data<scalar_t>(),
                out_q.mutable_data<scalar_t>(),
                out_k.mutable_data<scalar_t>(),
                out_v.mutable_data<scalar_t>(),
                num_heads + 2 * num_kv_heads,
                num_heads,
                num_kv_heads,
                dim_head,
                rope_theta,
                neox_style);
        });
    }
    BM_CUDART_ASSERT(cudaGetLastError());
}

} // namespace nn
