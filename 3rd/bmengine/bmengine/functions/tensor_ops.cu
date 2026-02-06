#include "bmengine/core/core.h"
#include "bmengine/core/stream.h"
#include "bmengine/functions/reduce.cuh"
#include "bmengine/functions/tensor_ops.h"
#include "bmengine/functions/transpose.h"
#include "private/tensor_ops.h"
#include "private/allocator.h"

namespace bmengine {
namespace functions {

// gridDim (n/1024, 1, 1),   blockDim (1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(concat_tensor)(
    size_t n,
    size_t part_a,
    size_t part_b,
    T* __restrict__ a, // (n / (part_a + part_b), part_a)
    T* __restrict__ b, // (n / (part_a + part_b), part_b)
    T* __restrict__ c  // (n / (part_a + part_b), part_a + part_b)
) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < n) {
        size_t row = pos / (part_a + part_b);
        size_t col = pos % (part_a + part_b);
        if (col < part_a) {
            c[pos] = a[row * part_a + col];

        } else {
            c[pos] = b[row * part_b + col - part_a];
        }
    }
}

core::Tensor concat_tensor(
    const core::Context& ctx, const core::Tensor& A, const core::Tensor& B, int dim) {
    if (A.numel() == 0)
        return B;
    if (B.numel() == 0)
        return A;

    BM_ASSERT_EQ(A.ndim(), B.ndim(), "logits ndim not the same");

    dim = dim >= 0 ? dim : A.ndim() + dim;

    size_t dim_out = A.size(dim) + B.size(dim);
    auto out_shape = A.size();
    out_shape[dim] = dim_out;

    core::Tensor out = ctx.tensor(out_shape, A.dtype());

    concat_tensor(ctx.current_stream()->ptr, A, B, out, dim);

    return out;
}

void concat_tensor(
    cudaStream_t stream, const core::Tensor& A, const core::Tensor& B, core::Tensor& out, int dim) {
    BM_ASSERT_EQ(A.ndim(), B.ndim(), "logits ndim not the same");

    dim = dim >= 0 ? dim : A.ndim() + dim;

    for (int d = 0; d < A.ndim(); ++d) {
        if (d == dim)
            continue;
        BM_ASSERT_EQ(A.size(d), B.size(d), "logits length not the same");
    }
    size_t dim_out = A.size(dim) + B.size(dim);
    auto out_shape = A.size();
    BM_ASSERT_EQ(dim_out, out.size(dim), "logits ndim not the same");

    size_t n = out.numel();

    int threads = min((size_t) 1024, round_up(n, 32));
    dim3 gridDim(round_up(n, threads) / threads, 1, 1);
    dim3 blockDim(threads, 1, 1);

    BM_DTYPE_DISPATCH(A.dtype(), {
        BM_KERNEL(concat_tensor)<<<gridDim, blockDim, 0, stream>>>(
            n,
            dim > 0 ? A.stride(dim - 1) : A.numel(),
            dim > 0 ? B.stride(dim - 1) : B.numel(),
            A.data<scalar_t>(),
            B.data<scalar_t>(),
            out.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

core::Tensor concat_tensor(
    const core::Context& ctx, const std::vector<core::Tensor>& org_tensors, int dim) {
    std::vector<core::Tensor> tensors;
    for (auto& t: org_tensors) {
        if (t.numel()) tensors.push_back(t);
    }
    BM_ASSERT(!tensors.empty(), "no tensors");
    BM_ASSERT_EQ(dim, 0, "dim>0 not implemented");
    for (auto& t : tensors) {
        BM_ASSERT_EQ(t.ndim(), tensors[0].ndim(), "dim mismatch");
    }
    auto dtype = tensors[0].dtype();
    int ndim = tensors[0].ndim();
    auto shape = tensors[0].shape();
    shape[0] = 0;
    for (auto& t : tensors) {
        for (int d = 1; d < ndim; ++d) {
            BM_ASSERT_EQ(shape[d], t.size(d), "shape mismatch dim " + std::to_string(d));
        }
        shape[0] += t.size(0);
    }
    auto ret = ctx.tensor(shape, dtype);
    auto stream = ctx.current_stream()->ptr;
    auto d2d = cudaMemcpyDeviceToDevice;
    auto dst = ret.data<char>();
    for (auto& a : tensors) {
        BM_CUDART_ASSERT(cudaMemcpyAsync(dst, a.data(), a.nbytes(), d2d, stream));
        dst += a.nbytes();
    }

    return ret;
}

// gridDim(L, M), blockDim(part_c)
template<typename T>
static __global__ void KERNEL_concat_broadcast(
    size_t part_a,
    size_t part_b,
    size_t part_c, // = part_a + part_b
    T* __restrict__ a, // (L, M, part_a)
    T* __restrict__ b, // (L, part_b)
    T* __restrict__ c  // (L, M, part_c)
) {
    for (unsigned int i = threadIdx.x; i < part_c; i += blockDim.x) {
        size_t pos_c = (blockIdx.x * gridDim.y + blockIdx.y) * part_c + i;
        if (i < part_a) {
            size_t pos_a = (blockIdx.x * gridDim.y + blockIdx.y) * part_a + i;
            c[pos_c] = a[pos_a];
        } else {
            size_t pos_b = (blockIdx.x) * part_b + (i - part_a);
            c[pos_c] = b[pos_b];
        }
    }
}

core::Tensor concat_broadcast_b(
    const core::Context& ctx, const core::Tensor& A, const core::Tensor& B) {
    BM_ASSERT_EQ(A.ndim(), 3, "");
    BM_ASSERT_EQ(B.ndim(), 2, "");
    BM_ASSERT_EQ(A.size(0), B.size(0), "");

    int dim = -1;
    size_t dim_out = A.size(dim) + B.size(dim);
    auto out_shape = A.size();
    out_shape[A.ndim() - 1] = dim_out;
    auto out = ctx.tensor(out_shape, A.dtype());

    int threads = round_up_thread(dim_out);
    dim3 gridDim(A.size(0), A.size(1));
    dim3 blockDim(threads, 1, 1);

    auto stream = ctx.current_stream()->ptr;
    BM_DTYPE_DISPATCH(A.dtype(), {
        KERNEL_concat_broadcast<<<gridDim, blockDim, 0, stream>>>(
            A.size(-1),
            B.size(-1),
            A.size(-1) + B.size(-1),
            A.data<scalar_t>(),
            B.data<scalar_t>(),
            out.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return out;
}

// gridDim (n/1024, 1, batch),   blockDim (1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(stack_tensor)(
    size_t n, int out_stride, T** __restrict__ ptrs, T* __restrict__ c) {
    int batch_id = blockIdx.z;
    T* a = ptrs[batch_id];
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < n) {
        c[batch_id * out_stride + pos] = a[pos];
    }
}

__host__ core::Tensor stack_tensor(
    const core::Context& ctx, const std::vector<core::Tensor>& tensors) {
    if (tensors.size() == 0) {
        return core::Tensor();
    }
    int ndim = tensors[0].ndim();
    auto shape = tensors[0].size();
    auto dtype = tensors[0].dtype();
    size_t n = tensors[0].numel();
    for (auto it = tensors.begin(); it != tensors.end(); ++it) {
        BM_ASSERT(it->numel() > 0, "tensor empty");
        BM_ASSERT(ndim == it->ndim(), "tensor ndim not the same");

        for (int d = 0; d < ndim; ++d) {
            BM_ASSERT(shape[d] == it->size(d), "tensor length not the same");
        }
    }
    int tensor_stride = tensors[0].stride(0);

    auto out_shape = shape;
    out_shape.insert(out_shape.begin(), tensors.size());

    core::Tensor out = ctx.tensor(out_shape, dtype);

    int batch = out.size(0);
    int out_stride = out.stride(0);

    int threads = min((size_t) 1024, round_up(n, 32));
    dim3 gridDim(round_up(n, threads) / threads, 1, batch);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH(out.dtype(), {
        std::vector<scalar_t*> pointers;
        scalar_t** device_pointers;
        BM_CUDART_ASSERT(cudaMalloc((void**) &device_pointers, tensors.size() * sizeof(scalar_t*)));
        for (int i = 0; i < tensors.size(); ++i) {
            pointers.push_back(tensors[i].data<scalar_t>());
        }
        BM_CUDART_ASSERT(cudaMemcpyAsync(
            device_pointers,
            pointers.data(),
            tensors.size() * sizeof(scalar_t*),
            cudaMemcpyHostToDevice,
            stream));
        BM_KERNEL(stack_tensor)<<<gridDim, blockDim, 0, stream>>>(
            n, out_stride, device_pointers, out.data<scalar_t>());
        // cuda stream synchronized by cudaFree, so that pointers can be freed safely.
        BM_CUDART_ASSERT(cudaFree(device_pointers));
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return out;
}

} // namespace functions
} // namespace bmengine
