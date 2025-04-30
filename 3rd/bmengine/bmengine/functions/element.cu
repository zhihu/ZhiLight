#include "bmengine/functions/element.h"
#include "bmengine/logger/std_log_op.hpp"
#include <cuda_runtime.h>
#include <assert.h>
#include <limits>

namespace bmengine {
namespace functions {

using bmengine::core::Tensor;

template<typename T>
class TensorOP_add {
public:
    __device__ TensorOP_add() { }
    inline static __device__ T op(const T& a, const T& b) { return a + b; }
};

template<typename T>
class TensorOP_sub {
public:
    inline static __device__ T op(const T& a, const T& b) { return a - b; }
};

template<typename T>
class TensorOP_mul {
public:
    inline static __device__ T op(const T& a, const T& b) { return a * b; }
};

template<typename T>
class TensorOP_div {
public:
    inline static __device__ T op(const T& a, const T& b) { return a / b; }
};

template<typename T>
class TensorOP_max {
public:
    inline static __device__ T op(const T& a, const T& b) { return a > b ? a : b; }
};

// block <n / 1024>,    thread <1024>
template<typename T, template<typename> class Op>
static __global__ void BM_KERNEL(elementwise_op)(int n, const T* a, const T* b, T* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = Op<T>::op(a[idx], b[idx]);
    }
}

// block <batch>,    thread <n>
template<typename T, template<typename> class Op>
static __global__ void BM_KERNEL(elementwise_op_broadcast_b1)(int n, const T* a, const T* b, T* c) {
    size_t offset = blockIdx.x * size_t(n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        c[offset + i] = Op<T>::op(a[offset + i], b[blockIdx.x]);
    }
}

// block <batch>,    thread <n>
template<typename T, template<typename> class Op>
static __global__ void BM_KERNEL(elementwise_op_broadcast_b)(int n, const T* a, const T* b, T* c) {
    size_t offset = blockIdx.x * size_t(n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        c[offset + i] = Op<T>::op(a[offset + i], b[i]);
    }
}

template<template<typename> class Op>
static void elementwise_op(
    const core::Tensor& a, const core::Tensor& b, const core::Tensor& c, cudaStream_t stream) {
    BM_ASSERT(vector_equal(a.size(), b.size()), "Tensor size mismatch");
    BM_ASSERT_EQ(a.dtype(), b.dtype(), "Type mismatch");
    size_t total_size = a.numel();
    int threads = round_up(min((size_t) 1024, total_size), 32);
    int blocks = (total_size + threads - 1) / threads;
    dim3 gridDim(blocks, 1, 1);
    dim3 blockDim(threads, 1, 1);

    auto dtype = a.dtype();

    BM_DTYPE_DISPATCH(a.dtype(), {
        BM_KERNEL(elementwise_op)<scalar_t, Op><<<gridDim, blockDim, 0, stream>>>(
            total_size, a.data<scalar_t>(), b.data<scalar_t>(), c.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

template<template<typename> class Op>
static void elementwise_op_b_y(
    const Tensor& a, const Tensor& b, const Tensor& c, cudaStream_t stream, int stride, bool y1) {
    BM_ASSERT(vector_equal(a.size(), c.size()), "Tensor size mismatch");
    size_t total_size = a.numel();
    int threads = round_up_thread(stride);
    int blocks = total_size / size_t(stride);
    // std::cout << "blocks=" << blocks << ", stride=" << stride << std::endl;

    if (y1) {
        BM_DTYPE_DISPATCH(a.dtype(), {
            BM_KERNEL(elementwise_op_broadcast_b1) < scalar_t, Op><<<blocks, threads, 0, stream>>>(
            stride, a.data<scalar_t>(), b.data<scalar_t>(), c.data<scalar_t>());
        });
    } else {
        BM_DTYPE_DISPATCH(a.dtype(), {
            BM_KERNEL(elementwise_op_broadcast_b) < scalar_t, Op><<<blocks, threads, 0, stream>>>(
            stride, a.data<scalar_t>(), b.data<scalar_t>(), c.data<scalar_t>());
        });
    }
    BM_CUDART_ASSERT(cudaGetLastError());
}

class BinaryElementwiseOp::impl {
public:
    Op op;
    impl(const core::Context& ctx, Op op) : op(op) { }
    impl(const impl&) = delete;
    impl(impl&&) = delete;
    ~impl() = default;

    void forward(
        const core::Context& ctx,
        const core::Tensor& a,
        const core::Tensor& b,
        const core::Tensor& c) {
        switch (op) {
            case Op::Add: elementwise_op<TensorOP_add>(a, b, c, ctx.current_stream()->ptr); break;
            case Op::Sub: elementwise_op<TensorOP_sub>(a, b, c, ctx.current_stream()->ptr); break;
            case Op::Mul: elementwise_op<TensorOP_mul>(a, b, c, ctx.current_stream()->ptr); break;
            case Op::Div: elementwise_op<TensorOP_div>(a, b, c, ctx.current_stream()->ptr); break;
            case Op::Max: elementwise_op<TensorOP_max>(a, b, c, ctx.current_stream()->ptr); break;
            default: BM_EXCEPTION("Unsupported op");
        }
    }

    void broadcast_y(
        const core::Context& ctx,
        const core::Tensor& a,
        const core::Tensor& b,
        const core::Tensor& c,
        int stride,
        bool y1) {
        auto stream = ctx.current_stream()->ptr;
        switch (op) {
            case Op::Add: elementwise_op_b_y<TensorOP_add>(a, b, c, stream, stride, y1); break;
            case Op::Sub: elementwise_op_b_y<TensorOP_sub>(a, b, c, stream, stride, y1); break;
            case Op::Mul: elementwise_op_b_y<TensorOP_mul>(a, b, c, stream, stride, y1); break;
            case Op::Div: elementwise_op_b_y<TensorOP_div>(a, b, c, stream, stride, y1); break;
            case Op::Max: elementwise_op_b_y<TensorOP_max>(a, b, c, stream, stride, y1); break;
            default: BM_EXCEPTION("Unsupported op");
        }
    }
};

BinaryElementwiseOp::BinaryElementwiseOp(const core::Context& ctx, Op op)
    : pimpl(new impl(ctx, op)) { }
BinaryElementwiseOp::~BinaryElementwiseOp() = default;

core::Tensor BinaryElementwiseOp::forward(
    const core::Context& ctx, const core::Tensor& x, const core::Tensor& y, core::Tensor* out) {
    core::Tensor ret = out ? *out : ctx.tensor(x.size(), x.dtype());
    pimpl->forward(ctx, x, y, ret);
    return ret;
}

void BinaryElementwiseOp::inplace(
    const core::Context& ctx, const core::Tensor& x, const core::Tensor& y) {
    pimpl->forward(ctx, x, y, x);
}

core::Tensor BinaryElementwiseOp::broadcast_y(
    const core::Context& ctx, const core::Tensor& x, const core::Tensor& y) {
    BM_ASSERT_EQ(x.dtype(), y.dtype(), "dtype mismatch");
    BM_ASSERT(x.ndim() > 1, "wrong dim");
    if (y.size(-1) == 1) {
        // broadcast last dim
        BM_ASSERT_EQ(x.ndim(), y.ndim(), "wrong dim");
        for (int i = 0; i < x.ndim() - 1; ++i) {
            BM_ASSERT_EQ(x.size(i), y.size(i), "shape mismatch of dim:" + std::to_string(i));
        }
        core::Tensor ret = ctx.tensor(x.size(), x.dtype());
        pimpl->broadcast_y(ctx, x, y, ret, x.size(-1), true);
        return ret;
    }
    // last n dim of x is same as y
    BM_ASSERT_LE(y.ndim() + 1, x.ndim(), "wrong dim");
    for (int i = 1; i <= y.ndim(); ++i) {
        BM_ASSERT_EQ(x.size(-i), y.size(-i), "shape mismatch of dim:-" + std::to_string(i));
    }
    core::Tensor ret = ctx.tensor(x.size(), x.dtype());
    pimpl->broadcast_y(ctx, x, y, ret, y.numel(), false);
    return ret;
}

template<typename T>
static __global__ void BM_KERNEL_check_numeric(size_t last_dim, const T* a) {
    size_t idx = size_t(blockIdx.y) * blockDim.x + threadIdx.x;
    if (idx < last_dim) {
        size_t offset = size_t(blockIdx.x) * last_dim + idx;
        assert(!isinf(float(a[offset])));
        assert(!isnan(float(a[offset])));
    }
}

void check_numeric(const core::Context& ctx, const core::Tensor& tensor) {
    size_t last_dim = tensor.size(-1);
    size_t threads = round_up_thread(last_dim);
    dim3 gridDim(tensor.numel() / last_dim, round_up(last_dim, threads) / threads);

    auto stream = ctx.current_stream()->ptr;
    BM_DTYPE_DISPATCH_FLOAT(tensor.dtype(), {
        BM_KERNEL_check_numeric<scalar_t><<<gridDim, threads, 0, stream>>>(
            last_dim, tensor.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaStreamSynchronize(stream));
    BM_CUDART_ASSERT(cudaGetLastError());
}

template<typename T>
static __global__ void KERNEL_check_equal(size_t last_dim, const T* a, const T* b) {
    size_t idx = size_t(blockIdx.y) * blockDim.x + threadIdx.x;
    if (idx < last_dim) {
        size_t offset = size_t(blockIdx.x) * last_dim + idx;
        assert(a[offset] == b[offset]);
    }
}

void check_equal(const core::Context& ctx, const core::Tensor& a, const core::Tensor& b) {
    BM_ASSERT_EQ(a.dtype(), b.dtype(), "a, b have different dtype.");
    BM_ASSERT_EQ(a.shape(), b.shape(), "a, b have different shapes.");
    size_t last_dim = a.size(-1);
    size_t threads = round_up_thread(last_dim);
    dim3 gridDim(a.numel() / last_dim, round_up(last_dim, threads) / threads);

    auto stream = ctx.current_stream()->ptr;
    BM_DTYPE_DISPATCH_FLOAT(a.dtype(), {
        KERNEL_check_equal<scalar_t><<<gridDim, threads, 0, stream>>>(
            last_dim, a.data<scalar_t>(), b.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaStreamSynchronize(stream));
    BM_CUDART_ASSERT(cudaGetLastError());
}

template<typename T>
struct UnaryOpPow {
    float exp;
    explicit __host__ __device__ UnaryOpPow(float exp) : exp(exp) { }
    inline __device__ T apply(T t) const { return T(powf(float(t), exp)); }
};

template<typename T>
struct UnaryOpClamp {
    float min_v;
    float max_v;
    explicit __host__ __device__ UnaryOpClamp(float min, float max) : min_v(min), max_v(max) { }
    inline __device__ T apply(T v) const {
        if (float(v) < min_v)
            return T(min_v);
        else if (float(v) > max_v)
            return T(max_v);
        return v;
    }
};

template<typename T>
struct UnaryOpSigmoid {
    inline __device__ T apply(T v) const {
        return 1.f / (1.f + expf(-(float)v));
    }
};

// block <n / 1024>,    thread <1024>
template<typename T, template<typename> class Op>
__global__ void BM_KERNEL(unary_op)(int n, const T* in, T* out, const Op<T> op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = op.apply(in[idx]);
    }
}

// block <n / 1024>,    thread <1024>
template<typename T, template<typename> class Op>
core::Tensor unary_op(const core::Context& ctx, const core::Tensor& a, Op<T>& op) {
    auto ret = ctx.tensor(a.size(), a.dtype());
    size_t n = a.numel();
    BM_ASSERT_LE(n, std::numeric_limits<int>::max(), "Out of range");
    size_t threads = 1024;
    dim3 gridDim(round_up(n, threads) / threads);

    auto stream = ctx.current_stream()->ptr;
    BM_KERNEL(unary_op)<T><<<gridDim, threads, 0, stream>>>(
        n, a.data<T>(), ret.mutable_data<T>(), op);
    BM_CUDART_ASSERT(cudaGetLastError());
    return ret;
}

core::Tensor pow(const core::Context& ctx, const core::Tensor& a, float exp) {
    BM_DTYPE_DISPATCH_FLOAT(a.dtype(), {
        UnaryOpPow<scalar_t> op(exp);
        return unary_op<scalar_t, UnaryOpPow>(ctx, a, op);
    });
}

core::Tensor clamp(const core::Context& ctx, const core::Tensor& a, float min, float max) {
    BM_DTYPE_DISPATCH_FLOAT(a.dtype(), {
        UnaryOpClamp<scalar_t> op(min, max);
        return unary_op<scalar_t, UnaryOpClamp>(ctx, a, op);
    });
}

core::Tensor sigmoid(const core::Context& ctx, const core::Tensor& tensor) {
    BM_DTYPE_DISPATCH_FLOAT(tensor.dtype(), {
        UnaryOpSigmoid<scalar_t> op;
        return unary_op<scalar_t, UnaryOpSigmoid>(ctx, tensor, op);
    });
}

} // namespace functions
} // namespace bmengine
