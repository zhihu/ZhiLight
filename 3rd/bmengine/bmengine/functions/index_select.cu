#include "bmengine/functions/index_select.h"
#include "bmengine/logger/std_log_op.hpp"
#include "private/tensor_ops.h"
#include <assert.h>

namespace bmengine {
namespace functions {

typedef unsigned int DimT;

// gridDim (M_new, R / 1024, L),
// blockDim (1024)
template<typename T>
static __global__ void BM_KERNEL(index_select)(
    DimT M_old,
    DimT M_new,
    DimT R,
    const T* __restrict__ input,   // (L, M_old, R)
    const int* __restrict__ index, // (M_new)
    T* __restrict__ out            // (L, M_new, R)
) {
    DimT l = blockIdx.z;
    DimT m_idx = blockIdx.x;
    DimT r = blockIdx.y * blockDim.x + threadIdx.x;
    if (r < R) {
        int m = index[m_idx];
        assert(m < M_old);
        T v = input[l * M_old * R + m * R + r];
        out[l * M_new * R + m_idx * R + r] = v;
    }
}

core::Tensor index_select(
    const core::Context& ctx,
    const core::Tensor& input,
    int dim,
    const core::Tensor& index,  // the 1-D tensor containing the indices to index
    core::Tensor* out_ptr
) {
    auto shape = input.shape();
    int rank = int(shape.size());
    if (dim < 0)
        dim += rank;
    BM_ASSERT(dim >= 0 && dim < rank, "Dimension out of range");

    const DimT M_new = index.numel();

    std::vector<size_t> out_shape = shape;
    out_shape[dim] = M_new;
    if (out_ptr) {
        BM_ASSERT_EQ(out_ptr->shape(), out_shape, "shape mismatch");
    }
    core::Tensor out = out_ptr ? *out_ptr : ctx.tensor(out_shape, input.dtype());

    auto stream = ctx.current_stream()->ptr;

    index_select(stream, input, dim, index, out);
    return out;
}

void index_select(
    cudaStream_t stream,
    const core::Tensor& input,
    int dim,
    const core::Tensor& index, // the 1-D tensor containing the indices to index,
    core::Tensor& out) {
    BM_ASSERT(index.shape().size() == 1, "index must be 1-D tensor");
    BM_ASSERT(index.dtype() == core::DataType::kInt32, "index must be int32 tensor");
    BM_ASSERT(index.numel() < 2147483647, "index too long");

    auto shape = input.shape();
    int rank = int(shape.size());
    if (dim < 0)
        dim += rank;
    BM_ASSERT(dim >= 0 && dim < rank, "Dimension out of range");
    // -1 dim indexing not efficent, but works.
    // BM_ASSERT(dim < rank - 1 , "Unsupported Dimension");  // dim == rank-1 not implemented yet.

    DimT L = 1, R = 1;
    for (int d = 0; d < dim; d++) {
        L *= shape[d];
    }
    for (int d = dim + 1; d < rank; d++) {
        R *= shape[d];
    }
    const DimT M_old = shape[dim];
    const DimT M_new = index.numel();

    DimT num_threads = round_up_thread(R);
    DimT grid1 = round_up(R, num_threads) / num_threads;
    dim3 gridDim(index.numel(), grid1, L);
    dim3 blockDim(num_threads);

    BM_DTYPE_DISPATCH(out.dtype(), {
        BM_KERNEL(index_select)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            M_old,
            M_new,
            R,
            input.data<scalar_t>(),
            index.data<int>(),
            out.mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

// gridDim (R / 1024, M_new, L),
// blockDim (1024)
template<typename T>
static __global__ void BM_KERNEL(index_along_dim)(
    DimT M_old,
    DimT M_new,
    DimT L,
    DimT R,
    const T* __restrict__ input,   // (L, M_old, R)
    const int* __restrict__ index, // (M_new)
    T* __restrict__ out            // (L, M_new, R)
) {
    DimT l = blockIdx.z;
    DimT m_idx = blockIdx.y;
    DimT r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < R) {
        int m = index[l * M_new + m_idx];
        assert(m < M_old);
        T v = input[l * M_old * R + m * R + r];
        out[l * M_new * R + m_idx * R + r] = v;
    }
}

core::Tensor index_along_dim(
    const core::Context& ctx,
    const core::Tensor& input,
    int dim,
    const core::Tensor& index // the (dim + 1)-D tensor containing the indices to index
) {
    BM_ASSERT(index.ndim() == dim + 1, "index must be `dim + 1` dimentional tensor");
    auto shape = input.shape();
    int rank = int(shape.size());
    if (dim < 0)
        dim += rank;
    BM_ASSERT(dim >= 0 && dim < rank, "Dimension out of range");

    std::vector<size_t> out_shape = shape;
    out_shape[dim] = index.size(dim);
    core::Tensor out = ctx.tensor(out_shape, input.dtype());

    auto stream = ctx.current_stream()->ptr;

    index_along_dim(stream, input, dim, index, out);
    return out;
}

void index_along_dim(
    cudaStream_t stream,
    const core::Tensor& input,
    int dim,
    const core::Tensor& index, // the (dim + 1)-D tensor containing the indices to index,
    core::Tensor& out) {
    BM_ASSERT(index.ndim() == dim + 1, "index must be `dim + 1` dimentional tensor");
    BM_ASSERT(index.dtype() == core::DataType::kInt32, "index must be int32 tensor");
    BM_ASSERT(index.size(dim) < 65536, "index too long");

    auto shape = input.shape();
    int rank = int(shape.size());
    if (dim < 0)
        dim += rank;
    BM_ASSERT(dim >= 0 && dim < rank, "Dimension out of range");

    DimT L = 1, R = 1;
    for (int d = 0; d < dim; d++) {
        L *= shape[d];
    }
    for (int d = dim + 1; d < rank; d++) {
        R *= shape[d];
    }
    const DimT M_old = shape[dim];
    const DimT M_new = index.size(dim);

    DimT num_threads = round_up_thread(R);
    DimT grid0 = round_up(R, num_threads) / num_threads;
    dim3 gridDim(grid0, index.size(dim), L);
    dim3 blockDim(num_threads);

    BM_DTYPE_DISPATCH(out.dtype(), {
        BM_KERNEL(index_along_dim)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            M_old,
            M_new,
            L,
            R,
            input.data<scalar_t>(),
            index.data<int>(),
            out.mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}
// gridDim (n / stride_dst), blockDim (1024)
template<typename T>
static __global__ void BM_KERNEL(copy_last_dim)(
    size_t stride_src,
    size_t stride_dst,
    size_t from,
    const T* __restrict__ input,   // (..., stride_src)
    T* __restrict__ out            // (..., stride_dst)
) {
    size_t offset_src = blockIdx.x * stride_src + from;
    size_t offset_dst = blockIdx.x * stride_dst;
    for (int i = threadIdx.x; i < stride_dst; i += blockDim.x) {
        out[offset_dst + i] = (from + i < stride_src) ? input[offset_src + i] : T(0.);
    }
}

void copy_last_dim(
    cudaStream_t stream,
    const core::Tensor& input,
    core::Tensor& output,
    int from,
    int to,
    bool padding_zero) {
    BM_ASSERT_EQ(input.dtype(), output.dtype(), "type mismatch");
    BM_ASSERT_EQ(input.ndim(), output.ndim(), "rank mismatch");
    DimT stride = output.size(-1);
    if (to == -1) {
        to = from + stride;
    }
    if (!padding_zero) {
        BM_ASSERT_LE(to, input.size(-1), "to out of range");
    }

    DimT num_threads = round_up_thread(stride);
    dim3 gridDim(output.numel() / stride);
    dim3 blockDim(num_threads);

    BM_DTYPE_DISPATCH(output.dtype(), {
        BM_KERNEL(copy_last_dim)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
        input.size(-1),
        stride,
        from,
        input.data<scalar_t>(),
        output.mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

core::Tensor slice_last_dim(
    const core::Context& ctx,
    const core::Tensor& input,
    int from,
    int len
) {
    auto shape = input.shape();
    shape[shape.size() - 1] = len;
    auto output = ctx.tensor(shape, input.dtype());
    copy_last_dim(ctx.current_stream()->ptr, input, output, from, from + len);
    return output;
}

}
}
