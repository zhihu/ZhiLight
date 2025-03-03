#include "kvcache/ragged_buffer_kernel.h"

#include <bmengine/core/core.h>
#include <bmengine/functions/utils.cuh>
#include <bmengine/functions/reduce.cuh>
#include <bmengine/functions/element.h>
#include <bmengine/logger/std_log_op.hpp>
#include "utils/env.h"

#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <vector_types.h>

namespace nn {

using bmengine::core::DataType;
using bmengine::core::Tensor;

// gridDim (batch, len_kv, num_heads),  blockDim (dim_head)
template<typename T>
static __global__ void KERNEL_copy_to_buffer_batch_place(
    int max_batch,
    int len_buf,
    const int* __restrict__ batch_place, // (batch, len_kv)
    const T* __restrict__ src,           // (batch, len_kv, num_heads, dim_head)
    T* __restrict__ dst                  // (max_batch, num_heads, len_buf, dim_head)
) {
    int batch_idx = blockIdx.x;
    int num_heads = gridDim.z;
    int dim_head = blockDim.x;

    int batch_dst = batch_place[batch_idx];
    assert(batch_dst < max_batch);

    int offset_src = ((batch_idx * gridDim.y + blockIdx.y) * num_heads + blockIdx.z) * dim_head;
    int pos_buf = blockIdx.y;
    int offset_dst = ((batch_dst * num_heads + blockIdx.z) * len_buf + pos_buf) * dim_head;

    dst[offset_dst + threadIdx.x] = src[offset_src + threadIdx.x];
}

void copy_to_buffer_batch_place(
    cudaStream_t stream,
    const core::Tensor& batch_place, // batch
    const core::Tensor& src,         // (batch, len_q, num_heads, dim_head)
    core::Tensor* dst                // (max_batch, num_heads, len_buf, dim_head)
) {
    BM_ASSERT_EQ(src.ndim(), 4, "src is not 4d");
    BM_ASSERT_EQ(dst->ndim(), 4, "dst is not 4d");

    BM_ASSERT_EQ(batch_place.numel(), src.size(0), "batch mismatch");
    uint32_t batch = src.size(0);
    uint32_t max_batch = dst->size(0);
    uint32_t len_kv = src.size(1);
    uint32_t num_heads = src.size(2);
    uint32_t dim_head = src.size(3);
    BM_ASSERT(batch <= max_batch, "batch is too big");

    BM_ASSERT_EQ(dst->size(1), num_heads, "dim mismatch");
    uint32_t len_buf = dst->size(2);
    BM_ASSERT(len_kv <= len_buf, "len_kv too big");
    BM_ASSERT(dim_head <= 1024, "dim_head too big");
    BM_ASSERT_EQ(dst->size(3), dim_head, "dim mismatch");

    dim3 gridDim(batch, len_kv, num_heads);
    BM_DTYPE_DISPATCH(src.dtype(), {
        KERNEL_copy_to_buffer_batch_place<<<gridDim, dim_head, 0, stream>>>(
        max_batch,
        len_buf,
        batch_place.data<int>(),
        src.data<scalar_t>(),
        dst->mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

// gridDim (len_kv, 1, num_heads),  blockDim (dim_head)
template<typename T>
static __global__ void KERNEL_copy_to_buffer_batch_and_place(
    int max_batch,
    int len_buf,
    const int* __restrict__ batch_place, // (total_len_kv)
    const int* __restrict__ placement,   // (total_len_kv)
    const T* __restrict__ src,           // (total_len_kv, num_heads, dim_head)
    T* __restrict__ dst                  // (max_batch, num_heads, len_buf, dim_head)
) {
    int t = blockIdx.x;
    int num_heads = gridDim.z;
    int dim_head = blockDim.x;

    int batch_dst = batch_place[t];
    assert(batch_dst < max_batch);
    int pos_buf = placement[t];
    if (pos_buf < 0)
        return;

    int offset_src = (t * num_heads + blockIdx.z) * dim_head;
    int offset_dst = ((batch_dst * num_heads + blockIdx.z) * len_buf + pos_buf) * dim_head;

    dst[offset_dst + threadIdx.x] = src[offset_src + threadIdx.x];
}

void copy_to_buffer_batch_place(
    cudaStream_t stream,
    const core::Tensor& batch_place, // (total_len_kv,)
    const core::Tensor& placement,   // (total_len_kv,)
    const core::Tensor& src,         // (total_len_kv, num_heads, dim_head)
    core::Tensor* dst                // (max_batch, num_heads, len_buf, dim_head)
) {
    BM_ASSERT_EQ(src.ndim(), 3, "src is not 3d");
    BM_ASSERT_EQ(dst->ndim(), 4, "dst is not 4d");

    BM_ASSERT_EQ(batch_place.numel(), src.size(0), "batch mismatch");
    uint32_t max_batch = dst->size(0);
    uint32_t total_len_kv = src.size(0);
    uint32_t num_heads = src.size(1);
    uint32_t dim_head = src.size(2);

    BM_ASSERT_EQ(dst->size(1), num_heads, "dim mismatch");
    uint32_t len_buf = dst->size(2);
    BM_ASSERT(dim_head <= 1024, "dim_head too big");
    BM_ASSERT_EQ(dst->size(3), dim_head, "dim mismatch");
    BM_ASSERT_LE(dim_head, 1024, "dim mismatch");

    dim3 gridDim(total_len_kv, 1, num_heads);
    BM_DTYPE_DISPATCH(src.dtype(), {
        KERNEL_copy_to_buffer_batch_and_place<<<gridDim, dim_head, 0, stream>>>(
        max_batch,
        len_buf,
        batch_place.data<int>(),
        placement.data<int>(),
        src.data<scalar_t>(),
        dst->mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}



// gridDim (batch, len_q, num_kv_heads),  blockDim (dim_head)
template<typename T>
static __global__ void KERNEL_copy_to_rag_buffer(
    const T* __restrict__ src,         // (batch, len_q, num_kv_heads, dim_head)
    const int* __restrict__ placement, // (batch, len_q)
    const int* __restrict__ buf_lens,  // (batch)
    T** __restrict__ buf_addrs         // (batch) => (num_kv_heads, len_buf, dim_head)
) {
    const int len_buf = buf_lens[blockIdx.x];
    T* dst = buf_addrs[blockIdx.x]; // (num_heads, len_buf, dim_head)
    const int num_heads = gridDim.z;
    const int dim_head = blockDim.x;

    int x = blockIdx.x * gridDim.y + blockIdx.y;
    int pos_buf = placement[x];
    if (pos_buf < 0)
        return;
    assert(pos_buf < len_buf);

    int head = blockIdx.z;
    size_t offset_src = size_t(x * num_heads + head) * dim_head;
    size_t offset_dst = size_t(head * len_buf + pos_buf) * dim_head;

    dst[offset_dst + threadIdx.x] = src[offset_src + threadIdx.x];
}

void copy_to_rag_buffer(
    const core::Context& ctx,
    const core::Tensor& src,       // (batch, len_q, num_kv_heads, dim_head)
    const core::Tensor& placement, // (batch, len_q)
    const core::Tensor& buf_lens,
    const core::Tensor& d_addrs) {
    BM_ASSERT_EQ(placement.ndim(), 2, "placement is not 2d");
    BM_ASSERT_EQ(src.ndim(), 4, "src is not 4d");
    BM_ASSERT_EQ(src.size(0), placement.size(0), "batch mismatch");
    BM_ASSERT_EQ(src.size(1), placement.size(1), "len_q mismatch");
    BM_ASSERT_EQ(src.size(0), d_addrs.numel(), "batch mismatch");
    BM_ASSERT_EQ(src.size(1), placement.size(1), "len_q mismatch");

    dim3 gridDim(src.size(0), src.size(1), src.size(2));
    auto stream = ctx.current_stream()->ptr;
    BM_DTYPE_DISPATCH(src.dtype(), {
        KERNEL_copy_to_rag_buffer<<<gridDim, src.size(3), 0, stream>>>(
            src.data<scalar_t>(),
            placement.data<int>(),
            buf_lens.data<int>(),
            d_addrs.data<scalar_t*>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

// gridDim (batch, len_q, num_kv_heads),  blockDim (dim_head)
template<typename T>
static __global__ void KERNEL_copy_to_rag_buffer2(
    const int* __restrict__ placement, // (batch, len_q)
    const int* __restrict__ buf_lens,  // (batch)
    const T* __restrict__ k_src,       // (batch, len_q, num_kv_heads, dim_head)
    const T* __restrict__ v_src,       // (batch, len_q, num_kv_heads, dim_head)
    T** __restrict__ buf_k_addrs,      // (batch) => (num_kv_heads, len_buf, dim_head)
    T** __restrict__ buf_v_addrs,      // (batch) => (num_kv_heads, len_buf, dim_head)
    bool BSHD) {
    const int len_buf = buf_lens[blockIdx.x];
    T* k_dst = buf_k_addrs[blockIdx.x]; // (num_heads, len_buf, dim_head)
    T* v_dst = buf_v_addrs[blockIdx.x]; // (num_heads, len_buf, dim_head)
    const int num_heads = gridDim.z;
    const int dim_head = blockDim.x;

    int x = blockIdx.x * gridDim.y + blockIdx.y;
    int pos_buf = placement[x];
    if (pos_buf < 0)
        return;
    assert(pos_buf < len_buf);

    int head = blockIdx.z;
    size_t offset_src = size_t(x * num_heads + head) * dim_head;
    size_t offset_dst =
        BSHD ? size_t(pos_buf * num_heads + head) * dim_head : // (len_buf, num_heads, dim_head)
        size_t(head * len_buf + pos_buf) * dim_head;       // (num_heads, len_buf, dim_head)

    k_dst[offset_dst + threadIdx.x] = k_src[offset_src + threadIdx.x];
    v_dst[offset_dst + threadIdx.x] = v_src[offset_src + threadIdx.x];
}

// gridDim (batch, len_q),  blockDim (num_kv_heads)
template<typename T=int8_t>
static __global__ void KERNEL_copy_to_rag_buffer2_scale(
    const int* __restrict__ placement, // (batch, len_q)
    const int* __restrict__ buf_lens,  // (batch)
    const T* __restrict__ k_src,       // (batch, len_q, num_kv_heads)
    const T* __restrict__ v_src,       // (batch, len_q, num_kv_heads)
    T** __restrict__ buf_k_addrs,      // (batch) => (len_buf, num_kv_heads)
    T** __restrict__ buf_v_addrs,      // (batch) => (len_buf, num_kv_heads)
    const int num_heads) {
    const int len_buf = buf_lens[blockIdx.x];
    T* k_dst = buf_k_addrs[blockIdx.x]; // (len_buf, num_kv_heads)
    T* v_dst = buf_v_addrs[blockIdx.x]; // (len_buf, num_kv_heads)

    int x = blockIdx.x * gridDim.y + blockIdx.y;
    int pos_buf = placement[x];
    if (pos_buf < 0)
        return;
    assert(pos_buf < len_buf);

    int head = threadIdx.x;
    size_t offset_src = size_t(x * num_heads + head);
    size_t offset_dst = size_t(pos_buf * num_heads + head); // (len_buf, num_heads)

    if (head < num_heads) {
        k_dst[offset_dst] = k_src[offset_src];
        v_dst[offset_dst] = v_src[offset_src];
    }
}

void copy_to_rag_buffer2(
    const core::Context& ctx,
    const core::Tensor& placement, // (batch, len_q)
    const core::Tensor& buf_lens,
    const core::Tensor& k_src,  // (batch, len_q, num_kv_heads, dim_head)
    const core::Tensor& v_src,  // (batch, len_q, num_kv_heads, dim_head)
    core::Tensor* buf_k_addr,
    core::Tensor* buf_v_addr,
    bool is_scale) {
    BM_ASSERT_EQ(placement.ndim(), 2, "placement is not 2d");
    BM_ASSERT_EQ(k_src.ndim(), (is_scale ? 3 : 4), "src is not 4d");
    BM_ASSERT_EQ(k_src.size(0), placement.size(0), "batch mismatch");
    BM_ASSERT_EQ(k_src.size(1), placement.size(1), "len_q mismatch");
    BM_ASSERT_EQ(k_src.size(0), buf_k_addr->numel(), "batch mismatch");
    BM_ASSERT_EQ(k_src.size(0), buf_v_addr->numel(), "batch mismatch");
    BM_ASSERT_EQ(k_src.size(-1), v_src.size(-1), "dim_head mismatch");

    dim3 gridDim(k_src.size(0), k_src.size(1), is_scale ? 1 : k_src.size(2));
    auto stream = ctx.current_stream()->ptr;
    if (is_scale) {
        // k_src: (batch, len_q, num_kv_heads)
        BM_ASSERT_EQ(k_src.dtype(), core::DataType::kFloat, "scale is not float");
        using scalar_t = float;
        KERNEL_copy_to_rag_buffer2_scale<<<gridDim, k_src.size(2), 0, stream>>>(
            placement.data<int>(),
            buf_lens.data<int>(),
            k_src.data<scalar_t>(),
            v_src.data<scalar_t>(),
            buf_k_addr->data<scalar_t*>(),
            buf_v_addr->data<scalar_t*>(),
            k_src.size(2));
        BM_CUDART_ASSERT(cudaGetLastError());
        return;
    }

    BM_DTYPE_DISPATCH(k_src.dtype(), {
        KERNEL_copy_to_rag_buffer2<<<gridDim, k_src.size(3), 0, stream>>>(
            placement.data<int>(),
            buf_lens.data<int>(),
            k_src.data<scalar_t>(),
            v_src.data<scalar_t>(),
            buf_k_addr->data<scalar_t*>(),
            buf_v_addr->data<scalar_t*>(),
            ctx.is_BSHD());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

} // namespace nn