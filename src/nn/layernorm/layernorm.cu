#include "nn/layernorm/layernorm.h"
#include "nn/quant/int8/quant_kernel.h"
#include <bmengine/functions/reduce.cuh>
#include <bmengine/functions/element.h>

namespace nn {

// gridDim (seq_len, 1, batch),     blockDim(1024, 1, 1)
template<typename T, typename T_W=T>
static __global__ void KERNEL_layernorm_rms(
    int dim_model,
    float eps,
    int input_stride,                // seq_len * dim_model
    const void* __restrict__ weight, // (dim_model)
    const T* __restrict__ input,     // (batch, seq_len, dim_model)
    T* __restrict__ output,          // (batch, seq_len, dim_model)
    float scale,
    T* input2 = nullptr,
    T* out_sum = nullptr
) {
    extern __shared__ float smem[];
    const T_W* weight_t = reinterpret_cast<const T_W*>(weight);

    int batch_id = blockIdx.z;
    int offset = batch_id * input_stride + blockIdx.x * dim_model;
    float local_sqr_sum = 0;
    bool need_add = out_sum != nullptr;
    for (int i = threadIdx.x; i < dim_model; i += blockDim.x) {
        float v = float(input[offset + i]);
        if (need_add) {
            v += float(input2[offset + i]);
            out_sum[offset + i] = T(v);
        }
        smem[i] = v;
        local_sqr_sum += v * v;
    }
    local_sqr_sum = functions::blockReduceSum<float>(local_sqr_sum) / (float) dim_model;
    local_sqr_sum = rsqrtf(local_sqr_sum + eps);
    for (int i = threadIdx.x; i < dim_model; i += blockDim.x) {
        output[offset + i] = T(smem[i] * local_sqr_sum * float(weight_t[i]) / scale);
    }
}

// gridDim (seq_len, 1, batch),     blockDim(1024, 1, 1)
template<typename T, typename T_W=T>
static __global__ void KERNEL_layer_norm_std(
    int dim_model,
    float eps,
    int input_stride,             // seq_len * dim_model
    const void* __restrict__ weight, // (dim_model)
    const T* __restrict__ input,  // (batch, seq_len, dim_model)
    T* __restrict__ output,       // (batch, seq_len, dim_model)
    float scale,
    T* b = nullptr,
    T* c = nullptr
) {
     functions::SharedMemory<T> shared;
     T* smem = shared.getPointer();
    const T_W* weight_t = reinterpret_cast<const T_W*>(weight);

    size_t batch_id = blockIdx.z;
    size_t offset = batch_id * input_stride + blockIdx.x * size_t(dim_model);
    float local_sum = 0;
    for (int i = threadIdx.x; i < dim_model; i += blockDim.x) {
        float v = float(input[offset + i]);
        smem[i] = v;
        local_sum += v;
    }
    float mean = functions::blockReduceSum<float>(local_sum) / float(dim_model);

    float local_sqr_sum = 0;
    for (int i = threadIdx.x; i < dim_model; i += blockDim.x) {
        float v = float(smem[i]) - mean;
        local_sqr_sum += v * v;
    }
    local_sqr_sum = functions::blockReduceSum<float>(local_sqr_sum) / float(dim_model);
    local_sqr_sum = rsqrtf(local_sqr_sum + eps);
    for (int i = threadIdx.x; i < dim_model; i += blockDim.x) {
        output[offset + i] = T((float(smem[i]) - mean) * local_sqr_sum * float(weight_t[i]));
    }
}

__host__ void layernorm(
    const core::Tensor& input,  // (batch, seq_len, dim_model)
    const core::Tensor& weight, // (dim_model)
    core::Tensor* output,       // (batch, seq_len, dim_model)
    float eps,
    float scale,
    bool rms,
    cudaStream_t stream,
    const core::Tensor* input2 = nullptr,
    core::Tensor* out_sum = nullptr) {
    int batch = (input.ndim() == 2) ? 1 : input.size(0);
    int input_stride = (input.ndim() == 2) ? 0 : input.stride(0);
    int seq_len = input.size(-2);
    int dim_model = input.size(-1);

    dim3 gridDim(seq_len, 1, batch);
    dim3 blockDim(min(round_up(dim_model, 32), 1024), 1, 1);

    BM_DTYPE_DISPATCH_FLOAT(input.dtype(), {
        auto kernel = KERNEL_layernorm_rms<scalar_t, scalar_t>;
        if (weight.dtype() == core::DataType::kFloat) {
            kernel = KERNEL_layernorm_rms<scalar_t, float>;
        }
        int smem_size = dim_model * sizeof(float);
        if (!rms) {
            kernel = KERNEL_layer_norm_std<scalar_t>;
            smem_size = dim_model * sizeof(scalar_t) + 4096;
        }
        if (smem_size > 48000)
            BM_CUDART_ASSERT(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        scalar_t* input2_ptr = input2 == nullptr ? nullptr : input2->data<scalar_t>();
        scalar_t* out_sum_ptr = out_sum == nullptr ? nullptr : out_sum->data<scalar_t>();
        kernel<<<gridDim, blockDim, smem_size, stream>>>(
            dim_model,
            eps,
            input_stride,
            weight.data(),
            input.data<scalar_t>(),
            output->mutable_data<scalar_t>(),
            scale,
            input2_ptr,
            out_sum_ptr
		);
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

// gridDim (seq_len, batch?),     blockDim(1024)
template<typename T, typename T_W=T>
static __device__ __inline__ void DEV_layernorm(
    const T* __restrict__ input,     // (batch?, seq_len, dim_model)
    T* __restrict__ output,          // (batch?, seq_len, dim_model)
    const void* __restrict__ weight, // (dim_model)
    int stride_in,
    int stride_out,
    int dim_model,
    float eps,
    float scale,
    size_t batch_stride_in = 0,
    size_t batch_stride_out = 0
) {
    extern __shared__ float smem[];
    const T_W* weight_t = reinterpret_cast<const T_W*>(weight);

    size_t offset = blockIdx.y * batch_stride_in + size_t(blockIdx.x) * stride_in;
    float local_sqr_sum = 0;
    for (int i = threadIdx.x; i < dim_model; i += blockDim.x) {
        float v = float(input[offset + i]);
        smem[i] = v;
        local_sqr_sum += v * v;
    }
    local_sqr_sum = functions::blockReduceSum<float>(local_sqr_sum) / (float) dim_model;
    local_sqr_sum = rsqrtf(local_sqr_sum + eps);

    offset = blockIdx.y * batch_stride_out + size_t(blockIdx.x) * stride_out;
    for (int i = threadIdx.x; i < dim_model; i += blockDim.x) {
        output[offset + i] = T(smem[i] * local_sqr_sum * float(weight_t[i]) / scale);
    }
}

// gridDim (seq_len),     blockDim(1024)
template<typename T, typename T_W=T>
static __global__ void KERNEL_layernorm_new(
    const T* __restrict__ input,  // (seq_len, dim_model)
    T* __restrict__ output,  // (seq_len, dim_model)
    const void* __restrict__ weight, // (dim_model)
    int stride_in,
    int stride_out,
    int dim_model,
    float eps,
    float scale,
    size_t batch_stride_in,
    size_t batch_stride_out
) {
    DEV_layernorm<T, T_W>(
        input, output, weight, stride_in, stride_out, dim_model, eps, scale,
        batch_stride_in, batch_stride_out);
}

// gridDim (seq_len, 2),     blockDim(1024)
template<typename T>
static __global__ void KERNEL_layernorm_new2(
    T* __restrict__ a,  // (seq_len, dim_a)
    T* __restrict__ b,  // (seq_len, dim_b)
    T* __restrict__ a_out,  // (seq_len, dim_a)
    T* __restrict__ b_out,  // (seq_len, dim_b)
    const T* __restrict__ weight_a, // (dim_a)
    const T* __restrict__ weight_b, // (dim_b)
    int stride_a,
    int stride_b,
    int stride_a_out,
    int stride_b_out,
    int dim_a,
    int dim_b,
    float eps,
    float scale
) {
    if (blockIdx.y == 0) {
        DEV_layernorm(a, a_out, weight_a, stride_a, stride_a_out, dim_a, eps, scale);
    } else {
        DEV_layernorm(b, b_out, weight_b, stride_b, stride_b_out, dim_b, eps, scale);
    }
}

class LayerNorm::impl {
public:
    class QuantImpl;
    class MultiHeadImpl;
    core::Tensor weight;
    core::Tensor bias;
    float eps;
    float scale;
    bool rms { true };
    impl(const core::Context& ctx, unsigned int dim_model, float eps, float scale, core::DataType dtype)
        : weight(ctx.parameter({ dim_model }, dtype)), eps(eps), scale(scale) { }
    virtual ~impl() = default;
    impl(const impl&) = delete;
    impl(impl&&) = default;

    void set_rms(bool b) {
        rms = b;
    }

    virtual core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& input // (batch, seq_len, dim_model)
    ) {
        if (weight.dtype() != core::DataType::kFloat) {
            BM_ASSERT(input.dtype() == weight.dtype(), "dtype mismatch");
        }
        BM_ASSERT_EQ(input.size(-1), weight.size(0), "dim mismatch");
        BM_ASSERT(input.ndim() >= 2, "input.ndim() must be >= 2");
        BM_ASSERT(input.device() == weight.device(), "Input and weight must be on the same device");
        core::Tensor output = ctx.tensor(input.shape(), input.dtype(), "ln_out", 16 * input.size(-1));
        BM_ASSERT(
            output.device() == weight.device(), "Output and weight must be on the same device");
        layernorm(input, weight, &output, eps, scale, rms, ctx.current_stream()->ptr);
        return output;
    }

    virtual bool support_fuse_add() const { return true; }
    virtual core::Tensor fuse_add(
        const core::Context& ctx,
        const core::Tensor& input, // (batch, seq_len, dim_model)
        const core::Tensor& input2, // (batch, seq_len, dim_model)
        core::Tensor& sum // sum = input + input2
    ) {
        BM_ASSERT(input.dtype() == weight.dtype(), "dtype mismatch");
        BM_ASSERT_EQ(input.size(-1), weight.size(0), "dim mismatch");
        BM_ASSERT(input.ndim() >= 2, "input.ndim() must be >= 2");
        BM_ASSERT(input.device() == weight.device(), "Input and weight must be on the same device");
        core::Tensor output = ctx.tensor(input.shape(), input.dtype());
        BM_ASSERT(
            output.device() == weight.device(), "Output and weight must be on the same device");
        layernorm(input, weight, &output, eps, scale, rms, ctx.current_stream()->ptr, &input2, &sum);
        return output;
    }

    virtual void inplace(
        const core::Context& ctx,
        core::Tensor& input // (seq_len, dim_model)
    ) {
        if (weight.dtype() != core::DataType::kFloat) {
            BM_ASSERT(input.dtype() == weight.dtype(), "dtype mismatch");
        }
        BM_ASSERT_EQ(input.size(-1), weight.size(0), "dim mismatch");
        BM_ASSERT(input.ndim() == 2 || input.ndim() == 3, "input.ndim() must be >= 2");

        int batch = input.ndim() == 3 ? input.size(0) : 1;
        int batch_stride = input.ndim() == 3 ? input.stride(0) : 0;
        int seq_len = input.size(-2);
        int stride = input.stride(-2);
        int dim_model = input.size(-1);

        int threads = round_up_thread(dim_model);
        dim3 gridDim(seq_len, batch);
        int smem_size = dim_model * sizeof(float);
        auto stream = ctx.current_stream()->ptr;

        BM_DTYPE_DISPATCH_HALF(input.dtype(), {
            auto kernel = KERNEL_layernorm_new<scalar_t>;
            if (weight.dtype() == core::DataType::kFloat) {
                kernel = KERNEL_layernorm_new<scalar_t, float >;
            }
            kernel<<<gridDim, threads, smem_size, stream>>>(
                input.data<scalar_t>(),
                input.data<scalar_t>(),
                weight.data(),
                stride,
                stride,
                dim_model,
                eps,
                scale,
                batch_stride,
                batch_stride
            );
        });
        BM_CUDART_ASSERT(cudaGetLastError());
    }
};

class LayerNorm::impl::QuantImpl : public LayerNorm::impl {
public:
    QuantImpl(const core::Context& ctx, unsigned int dim_model, float eps, float scale, core::DataType dtype)
        : LayerNorm::impl(ctx, dim_model, eps, scale, dtype) { }

    virtual ~QuantImpl() = default;
    QuantImpl(const QuantImpl&) = delete;
    QuantImpl(QuantImpl&&) = default;

    core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& input // (batch, seq_len, dim_model)
    ) {
        BM_ASSERT(input.dtype() == weight.dtype(), "dtype mismatch");
        BM_ASSERT_EQ(input.size(-1), weight.size(0), "dim mismatch");
        BM_ASSERT(input.ndim() >= 2, "input.ndim() must be >= 2");
        BM_ASSERT(input.device() == weight.device(), "Input and weight must be on the same device");

        core::Tensor output, output_int8, output_scale;
        int8_op::layernorm_quant(ctx, input, weight, &output, &output_int8, &output_scale, eps, scale);
        int8_op::set_quant_scale(output_int8, output_scale);
        int8_op::set_quant_scale(output, output_int8);
        return output;
    }

    virtual bool support_fuse_add() const { return false; }
};

// gridDim (seq_len, num_head),     blockDim(head_dim)
 template<typename T>
 static __global__ void KERNEL_layernorm_multi_head(
     const T* __restrict__ input,  // (seq_len, num_head, head_dim)
     T* __restrict__ output,       // (seq_len, num_head, head_dim)
     const T* __restrict__ weight, // (num_head, head_dim)
     float eps
 ) {
    size_t offset = size_t(blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
    size_t offset_w = size_t(blockIdx.y) * blockDim.x + threadIdx.x;

    float v = float(input[offset]);
    float mean = functions::blockReduceSum<float>(v) / float(blockDim.x);
    v = v - mean;

    float local_sqr_sum = v * v;
    local_sqr_sum = functions::blockReduceSum<float>(local_sqr_sum) / float(blockDim.x);
    local_sqr_sum = rsqrtf(local_sqr_sum + eps);

    output[offset] = T(v * local_sqr_sum * float(weight[offset_w]));
}

class LayerNorm::impl::MultiHeadImpl : public LayerNorm::impl {
public:
    MultiHeadImpl(
        const core::Context& ctx, unsigned int dim_model, float eps, float scale, core::DataType dtype, size_t num_head)
        : LayerNorm::impl(ctx, dim_model, eps, scale, dtype) {
        weight = ctx.parameter({ num_head, size_t(dim_model) / num_head }, dtype);
    }
    virtual ~MultiHeadImpl() = default;

    core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& input // (batch, seq_len, dim_model)
    ) override {
        BM_ASSERT(input.dtype() == weight.dtype(), "dtype mismatch");
        BM_ASSERT_EQ(input.size(-1), weight.numel(), "dim_model mismatch");
        BM_ASSERT(input.ndim() >= 2, "input.ndim() must be >= 2");

        core::Tensor output = ctx.tensor(input.shape(), input.dtype());

        dim3 gridDim(input.size(0), weight.size(0));
        int threads = weight.size(1);
        auto stream = ctx.current_stream()->ptr;

        BM_DTYPE_DISPATCH_HALF(input.dtype(), {
            KERNEL_layernorm_multi_head<scalar_t><<<gridDim, threads, 0, stream>>>(
                input.data<scalar_t>(),
                output.mutable_data<scalar_t>(),
                weight.data<scalar_t>(),
                eps
            );
        });
        BM_CUDART_ASSERT(cudaGetLastError());

        return output;
    }

    virtual bool support_fuse_add() const { return false; }
};

LayerNorm::LayerNorm(
    const core::Context& ctx, int dim_model, bool quant, float eps, float scale, core::DataType dtype, int num_head)
    : core::Layer() {
    if (num_head > 1) {
        pimpl.reset(new impl::MultiHeadImpl(ctx, dim_model, eps, scale, dtype, num_head));
    } else {
        pimpl.reset(quant ?
                    new impl::QuantImpl(ctx, dim_model, eps, scale, dtype) :
                    new impl(ctx, dim_model, eps, scale, dtype));
    }
    add_parameter("weight", pimpl->weight);
}
LayerNorm::~LayerNorm() = default;

core::Tensor LayerNorm::forward(const core::Context& ctx, const core::Tensor& x) {
    size_t M = x.numel() / x.size(-1);
    core::EventScope event_scope(ctx, "LayerNorm[M=" + std::to_string(M) + "]", 1, 2 * x.nbytes());
    return pimpl->forward(ctx, x);
}

core::Tensor LayerNorm::fuse_add(const core::Context& ctx, const core::Tensor& x, const core::Tensor& b, core::Tensor& c) {
    size_t M = x.numel() / x.size(-1);
    core::EventScope event_scope(ctx, "AddLayerNorm[M=" + std::to_string(M) + "]", 1, 4 * x.nbytes());
    if (pimpl->support_fuse_add()) {
        return pimpl->fuse_add(ctx, x, b, c);
    } else {
        using BinaryOp = bmengine::functions::BinaryElementwiseOp;
        BinaryOp add_op(ctx, BinaryOp::Add);
        add_op.forward(ctx, x, b, &c);
        return pimpl->forward(ctx, c);
    }
}

void LayerNorm::inplace(const core::Context& ctx, core::Tensor& x) {
    // BM_ASSERT_EQ(x.ndim(), 2, "");
    size_t M = x.numel() / x.size(-1);
    core::EventScope event_scope(ctx, "LayerNorm[M=" + std::to_string(M) + "]");
    return pimpl->inplace(ctx, x);
}

void LayerNorm::forward_2(
    const core::Context& ctx,
    core::Tensor& x, core::Tensor& y,
    core::Tensor& x_out, core::Tensor& y_out,
    LayerNorm* la, LayerNorm* lb) {
    size_t M = x.numel() / x.size(-1);
    core::EventScope event_scope(ctx, "LayerNorm2[M=" + std::to_string(M) + "]");

    auto& weight_a = la->pimpl->weight;
    auto& weight_b = lb->pimpl->weight;
    auto eps = lb->pimpl->eps;
    auto scale = lb->pimpl->scale;
    BM_ASSERT_EQ(x.ndim(), 2, "");
    BM_ASSERT_EQ(y.ndim(), 2, "");
    BM_ASSERT(x.dtype() == weight_a.dtype(), "dtype mismatch");
    BM_ASSERT_EQ(x.size(0), y.size(0), "seq_len mismatch");
    BM_ASSERT_EQ(x.size(-1), weight_a.size(0), "dim mismatch");
    BM_ASSERT_EQ(y.size(-1), weight_b.size(0), "dim mismatch");

    uint32_t seq_len = x.size(-2);
    int stride_a = x.stride(-2);
    int stride_b = y.stride(-2);
    int stride_a_out = x_out.stride(-2);
    int stride_b_out = y_out.stride(-2);
    int dim_a = x.size(-1);
    int dim_b = y.size(-1);

    dim3 gridDim(seq_len, 2U);
    int threads = round_up_thread(std::max(dim_a, dim_b));
    int smem_size = std::max(dim_a, dim_b) * sizeof(float);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_HALF(x.dtype(), {
        KERNEL_layernorm_new2<scalar_t><<<gridDim, threads, smem_size, stream>>>(
            x.data<scalar_t>(),
            y.data<scalar_t>(),
            x_out.data<scalar_t>(),
            y_out.data<scalar_t>(),
            weight_a.data<scalar_t>(),
            weight_b.data<scalar_t>(),
            stride_a,
            stride_b,
            stride_a_out,
            stride_b_out,
            dim_a,
            dim_b,
            eps,
            scale
        );
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

void LayerNorm::set_rms(bool b) {
    pimpl->set_rms(b);
}
void LayerNorm::load_state_dict(
    const core::Context& ctx,
    const std::map<std::string, const core::Tensor>& state_dict,
    const std::string& prefix,
    bool allow_missing) {
    impl::MultiHeadImpl* p = dynamic_cast<impl::MultiHeadImpl*>(pimpl.get());
    if (p) {
        auto name = prefix + ".weight";
        ctx.load_parameter(&p->weight, name, state_dict, true, core::DistLayout::ROW);
    } else {
        auto name = prefix + ".weight";
        auto it = state_dict.find(name);
        BM_ASSERT(it != state_dict.end(), "param " + name + " not found in state_dict");
        auto& param = it->second;
        if (param.dtype() == core::DataType::kFloat) {
            pimpl->weight = ctx.tensor(pimpl->weight.shape(), core::DataType::kFloat);
        }

        core::Layer::load_state_dict(ctx, state_dict, prefix, allow_missing);
    }
}
}
