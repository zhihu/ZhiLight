#include "nn/embedding/embedding.h"
#include "model/model_context.h"
#include "utils/env.h"
#include <bmengine/c10d/c10d.h>
#include <bmengine/core/core.h>
#include <bmengine/functions/element.h>
#include <bmengine/functions/gemm.h>
#include <bmengine/functions/index_select.h>
#include <bmengine/functions/init.h>
#include <bmengine/functions/transpose.h>
#include <bmengine/functions/typecast.h>
#include <bmengine/functions/reduce.cuh>
#include <bmengine/logger/std_log_op.hpp>
#include <iostream>
#include <assert.h>

namespace nn {

using model::ModelContext;

// gridDim (seq_len, dim_model / 1024, 1),   blockDim (1024, 1, 1)
template<typename T>
__global__ void BM_KERNEL(embedding)(
    int begin,
    int end,
    size_t dim_model,
    float scale,
    const int32_t* __restrict__ idx, // (batch, seq_len)
    const T* __restrict__ weight,    // (vocab_size, dim_model)
    T* __restrict__ out              // (batch, seq_len, dim_model)
) {
    int target_id = idx[blockIdx.x];
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    size_t offset = blockIdx.x * dim_model;
    bool in_range = target_id >= begin && target_id < end;
    target_id -= begin;
    if (col < dim_model) {
        out[offset + col] = in_range
            ? T(float(weight[size_t(target_id) * dim_model + col]) * scale)
            : T(0.);
    }
}

// gridDim (seq_len, dim_model / 1024, batch),   blockDim (1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(rotary_embedding)(
    int dim_model,
    int batch_stride,
    int emb_stride,
    const int32_t* __restrict__ pos, // (batch, seq_len)
    const T* __restrict__ emb,       // (batch, seq_len, dim_model)
    T* __restrict__ out              // (batch, seq_len, dim_model)
) {
    int batch_id = blockIdx.z;
    int target_pos = pos[batch_id * batch_stride + blockIdx.x] * 16;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    int offset = batch_id * emb_stride + blockIdx.x * dim_model;

    int half_dim_model = dim_model / 2;
    if (col < half_dim_model) {
        float freq = target_pos * powf(10000, -float(col * 2) / dim_model);
        float cos_freq = cos(freq);
        float sin_freq = sin(freq);
        T rorate_v = -emb[offset + col + half_dim_model];
        out[offset + col] = emb[offset + col] * T(cos_freq) + rorate_v * T(sin_freq);
    } else if (col < dim_model) {
        float freq = target_pos * powf(10000, -float((col - half_dim_model) * 2) / dim_model);
        float cos_freq = cos(freq);
        float sin_freq = sin(freq);
        T rotate_v = emb[offset + col - half_dim_model];
        out[offset + col] = emb[offset + col] * T(cos_freq) + rotate_v * T(sin_freq);
    }
}

static __host__ void embedding(
    const core::Tensor& idx,
    const core::Tensor& weight,
    const core::Tensor& out,
    int begin,
    int end,
    float scale,
    cudaStream_t stream) {
    int seq_len = idx.numel();
    int dim_model = weight.size(1);
    int threads = round_up_thread(dim_model);
    dim3 gridDim(seq_len, round_up(dim_model, threads) / threads);
    dim3 blockDim(threads);

    BM_DTYPE_DISPATCH_FLOAT(weight.dtype(), {
        BM_KERNEL(embedding)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            begin,
            end,
            dim_model,
            scale,
            idx.data<int32_t>(),
            weight.data<scalar_t>(),
            out.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

static __host__ core::Tensor rotary_embedding(
    const core::Context& ctx,
    const core::Tensor& pos, // (batch, seq_len)
    const core::Tensor& emb  // (batch, seq_len, dim_model)
) {
    int batch = pos.ndim() == 1 ? 1 : pos.size(0);
    int batch_stride = pos.ndim() == 1 ? 0 : pos.stride(0);
    int emb_stride = emb.ndim() == 2 ? 0 : emb.stride(0);
    int seq_len = emb.size(-2);
    int dim_model = emb.size(-1);
    int threads = min(1024, round_up(dim_model, 32));
    dim3 gridDim(seq_len, round_up(dim_model, threads) / threads, batch);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;
    auto out = ctx.tensor(emb.size(), emb.dtype());
    BM_DTYPE_DISPATCH_FLOAT(emb.dtype(), {
        BM_KERNEL(rotary_embedding)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            dim_model,
            batch_stride,
            emb_stride,
            pos.data<int32_t>(),
            emb.data<scalar_t>(),
            out.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return out;
}

class Embedding::impl {
public:
    core::Tensor weight;
    unsigned int dim_model;
    unsigned int vocab_size;
    core::DataType dtype;

    float scale_factor;
    functions::Gemm local_gemm;
    functions::Gemm local_gemm_alpha;
    impl(
        const core::Context& ctx,
        unsigned int vocab_size,
        unsigned int dim_model,
        bool scale_weights,
        core::DataType dtype)
        : weight(ctx.parameter({ vocab_size, dim_model }, dtype)),
          dim_model(dim_model),
          vocab_size(vocab_size),
          dtype(dtype),
          scale_factor(scale_weights ? 1.0 / sqrtf(dim_model) : 1.0),
          local_gemm(ctx, dtype, false, true),
          local_gemm_alpha(ctx, dtype, false, true, scale_factor) {
        if (ctx.high_precision() >= 1) {
            local_gemm.set_compute_type(CUBLAS_COMPUTE_32F);
            local_gemm_alpha.set_compute_type(CUBLAS_COMPUTE_32F);
        }
    }

    void set_scale_weights(bool b) {
        scale_factor = (b ? 1.0 / sqrtf(dim_model) : 1.0);
    }

    core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& ids,    // (batch, seq_len)
        const core::Tensor& ids_sub // (batch, seq_len)
    ) {
        BM_ASSERT(ids.dtype() == core::DataType::kInt32, "ids dtype mismatch");
        BM_ASSERT(ids.ndim() == 1 || ids.ndim() == 2, "ids must be 1d or 2d");

        auto shape = ids.size();
        shape.emplace_back(dim_model);
        core::Tensor ret = ctx.tensor(shape, dtype);

        embedding(ids, weight, ret, 0, weight.size(0), scale_factor, ctx.current_stream()->ptr);
        return rotary_embedding(ctx, ids_sub, ret);
    }

    std::tuple<core::Tensor, core::Tensor> projection(
        const core::Context& ctx,
        const core::Tensor& input,    // (seq_len, dim_model)
        const core::Tensor& ext_table // (ext_len, dim_model)
    ) {
        // merge dim to accelerate gemm with tensor core, otherwise cublas would uses gemv which is
        // much slower.
        auto logits = local_gemm_alpha.forward(
            ctx,
            input, // (batch?, seq_len, dim_model)
            weight // (vocab_size, dim_model)T
        );  // (seq_len, vocab_size)

        auto logits_ext = core::Tensor();
        if (ext_table.numel() > 0) {
            logits_ext = local_gemm.forward(
                ctx,
                input,    // (seq_len, dim_model)
                ext_table // (ext_len, dim_model)T
            );            // (seq_len, ext_len)
        }
        return std::make_tuple(logits, logits_ext);
    }
    };

class RawEmbedding::impl {
public:
    class NormalImpl;
    class ParallelImpl;
    class RowParallelImpl;
    float logit_scale { 1. }; // For Cohere model
    virtual ~impl()= default;

    virtual core::Tensor& get_weight() = 0;

    virtual void set_scale_weights(bool b) = 0;
    virtual void set_scale_factor(float b) = 0;
    virtual void set_logit_scale(float s) { logit_scale = s; };

    virtual core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& ids // (seq_len)
    ) = 0;

    virtual core::Tensor projection(
        const core::Context& ctx,
        const core::Tensor& input // (seq_len, dim_model)
    ) = 0;

};

class RawEmbedding::impl::NormalImpl : public RawEmbedding::impl {
public:
    core::Tensor weight;
    unsigned int dim_model;
    core::DataType dtype;
    unsigned int begin;
    unsigned int end;
    float scale_factor;
    NormalImpl(
        const core::Context& ctx,
        unsigned int vocab_size,
        unsigned int dim_model,
        bool scale_weights,
        core::DataType dtype)
        : weight(ctx.parameter({ vocab_size, dim_model }, dtype)),
          dim_model(dim_model),
          dtype(dtype),
          begin(0),
          end(vocab_size),
          scale_factor(scale_weights ? 1.0 / sqrtf(dim_model) : 1.0) { }

    core::Tensor& get_weight() { return weight; }

    void set_scale_weights(bool b) {
        scale_factor = (b ? 1.0 / sqrtf(dim_model) : 1.0);
    }

    void set_scale_factor(float b) {
	    scale_factor = b;
    }
    core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& ids // (seq_len)
    ) {
        BM_ASSERT(ids.dtype() == core::DataType::kInt32, "ids dtype mismatch");
        BM_ASSERT(ids.ndim() == 1 || ids.ndim() == 2, "ids must be 1d or 2d");

        auto out_shape = ids.shape();
        out_shape.push_back(dim_model);
        core::Tensor ret = ctx.tensor(out_shape, dtype);
        embedding(ids, weight, ret, begin, end, scale_factor, ctx.current_stream()->ptr);
        return ret;
    }

    core::Tensor projection(
        const core::Context& ctx,
        const core::Tensor& input // (seq_len, dim_model)
    ) {
        functions::Gemm local_gemm(ctx, dtype, false, true, scale_factor * logit_scale);
        if (ctx.high_precision() >= 1) {
            local_gemm.set_compute_type(CUBLAS_COMPUTE_32F);
        }
        auto logits = local_gemm.forward(
            ctx,
            input, // (seq_len, dim_model)
            weight // (vocab_size, dim_model)T
        );         // (seq_len, vocab_size)
        return logits;
    }
};

class RawEmbedding::impl::RowParallelImpl : public RawEmbedding::impl::NormalImpl {
public:
    unsigned int vocab_size;
    RowParallelImpl(
        const core::Context& ctx,
        unsigned int vocab_size,
        unsigned int dim_model,
        bool scale_weights,
        core::DataType dtype)
        : NormalImpl(ctx, vocab_size, dim_model, scale_weights, dtype), vocab_size(vocab_size) {}

    void load_state_dict(
        const core::Context& ctx,
        const std::map<std::string, const core::Tensor>& state_dict,
        const std::string& prefix,
        bool allow_missing) {
        unsigned int round_size = round_up(vocab_size, 128);
        unsigned int part_size = round_size / ctx.world_size();
        begin = ctx.rank() * part_size;
        end = begin + part_size;

        auto it = state_dict.find(prefix + ".weight");
        BM_ASSERT(it != state_dict.end(), "Weight not found: " + prefix + ".weight");
        auto part_src =it->second.slice_dim0(begin, std::min(end, vocab_size));
        weight = ctx.tensor({part_size, dim_model}, dtype);
        functions::zeros_(ctx, weight);
        auto weight_t = weight.slice_dim0(0, part_src.size(0));
        ctx.assign_or_copy(&weight_t, &part_src);

//        auto weight_t = ctx.tensor({round_size, dim_model}, dtype);
//        functions::zeros_(ctx, weight_t);
//        if (ctx.rank() == 0) {
//            auto weight_t1 = weight_t.slice_dim0(0, vocab_size);
//            std::cout << "Load weight_t1: " << weight_t1.shape() << "\n";
//            core::Layer::load_param_from_state_dict(
//                ctx, state_dict, prefix + ".weight", &weight_t1, allow_missing);
//        }
//        weight = ctx.distribute_parameter(weight_t, bmengine::core::DistLayout::ROW);
//        std::cout << "Loaded weight: " << weight.shape() << ", begin:" << begin << ", end:" << end << "\n";
    }

    core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& ids // (seq_len)
    ) {
        BM_ASSERT(ids.dtype() == core::DataType::kInt32, "ids dtype mismatch");
        BM_ASSERT(ids.ndim() == 1 || ids.ndim() == 2, "ids must be 1d or 2d");

        auto out_shape = ids.shape();
        out_shape.push_back(dim_model);
        core::Tensor ret = ctx.tensor(out_shape, dtype);
        BM_CUDART_ASSERT(cudaMemsetAsync(ret.data(), 0, ret.nbytes(), ctx.current_stream()->ptr));
        embedding(ids, weight, ret, begin, end, scale_factor, ctx.current_stream()->ptr);

        return ctx.reduce_sum(ret, dtype);
    }

    core::Tensor projection(
        const core::Context& ctx,
        const core::Tensor& input // (seq_len, dim_model)
    ) {
        BM_ASSERT_EQ(input.size(-1), dim_model, "size mismatch");
        size_t seq_len = input.numel() / input.size(-1);
        functions::Gemm local_gemm(ctx, dtype, false, true, scale_factor * logit_scale);
        functions::Transpose transpose(ctx);
        local_gemm.set_compute_type(CUBLAS_COMPUTE_32F);
        auto input_2d = input.view({seq_len, dim_model});
        auto part_logits = local_gemm.forward(
            ctx,
            weight,    // (vocab_size, dim_model)
            input_2d   // (seq_len, dim_model)
        );         // (part_vocab_size, seq_len)
        size_t world_size = ctx.world_size();
        auto all = ctx.tensor({world_size, part_logits.size(0), seq_len}, input.dtype());
        ModelContext* m_ctx = dynamic_cast<ModelContext*>(const_cast<core::Context*>(&ctx));
        int gather_to_master = utils::get_int_env("GATHER_LOGIT_MASTER_ONLY", 0);
        if (m_ctx && m_ctx->dyn_batch() && gather_to_master) {
            auto chunk = all.chunk();
            ncclGroupStart();
            c10d::NCCLSend(ctx, part_logits, 0);
            if (ctx.rank() == 0) {
                for (int r=0; r < ctx.world_size(); r++)
                    c10d::NCCLRecv(ctx, chunk[r], r);
            }
            ncclGroupEnd();
        } else {
            c10d::NCCLAllGather(ctx, part_logits, all);
        }
        all = all.view({world_size * part_logits.size(0), seq_len});
        auto res = transpose(ctx, all.slice_dim0(0, vocab_size)); // (seq_len, vocab_size)
        if (input.ndim() == 3)
            return res.view({ input.size(0), input.size(1), vocab_size });
        else
            return res;
    }
};

Embedding::Embedding(const core::Context& ctx, int dim_model, int vocab_size, bool scale_weights, core::DataType dtype)
    : pimpl(new impl(ctx, vocab_size, dim_model, scale_weights, dtype)), core::Layer() {
    add_parameter("weight", pimpl->weight);
    // gemm has no weight; add only for set prefix
    add_submodule("local_gemm", pimpl->local_gemm);
    add_submodule("local_gemm_alpha", pimpl->local_gemm_alpha);
}
Embedding::~Embedding() = default;

void Embedding::set_scale_weights(bool b) {
    pimpl->set_scale_weights(b);
}

core::Tensor Embedding::forward(
    const core::Context& ctx,
    const core::Tensor& ids,    // (seq_len)
    const core::Tensor& ids_sub // (seq_len)

) {
    return pimpl->forward(ctx, ids, ids_sub);
}

std::tuple<core::Tensor, core::Tensor> Embedding::projection(
    const core::Context& ctx,
    const core::Tensor& input,    // (seq_len, dim_model)
    const core::Tensor& ext_table // (ext_len, dim_model)
) {
    return pimpl->projection(ctx, input, ext_table);
}

RawEmbedding::RawEmbedding(
    const core::Context& ctx,
    int dim_model,
    int vocab_size,
    bool scale_weights,
    core::DataType dtype,
    bool parallel)
    : core::Layer() {
    int row_parallel = utils::get_int_env("CPM_EMB_ROW_PAR", 1);
    if (parallel) {
        pimpl.reset(new impl::RowParallelImpl(ctx, vocab_size, dim_model, scale_weights, dtype));
    } else {
        pimpl.reset(new impl::NormalImpl(ctx, vocab_size, dim_model, scale_weights, dtype));
    }
    add_parameter("weight", pimpl->get_weight());
}
RawEmbedding::~RawEmbedding() = default;

void RawEmbedding::set_scale_weights(bool b) {
    pimpl->set_scale_weights(b);
}

void RawEmbedding::set_scale_factor(float b) {
    pimpl->set_scale_factor(b);
}
void RawEmbedding::set_logit_scale(float b) {
    pimpl->set_logit_scale(b);
}
core::Tensor RawEmbedding::forward(
    const core::Context& ctx,
    const core::Tensor& ids // (seq_len)

) {
    return pimpl->forward(ctx, ids);
}

core::Tensor RawEmbedding::projection(
    const core::Context& ctx,
    const core::Tensor& input // (seq_len, dim_model)
) {
    return pimpl->projection(ctx, input);
}

void RawEmbedding::load_state_dict(
    const core::Context& ctx,
    const std::map<std::string, const core::Tensor>& state_dict,
    const std::string& prefix,
    bool allow_missing) {
    auto row_ptr = dynamic_cast<impl::RowParallelImpl*>(pimpl.get());
    if (row_ptr) {
        row_ptr->load_state_dict(ctx, state_dict, prefix, allow_missing);
    } else {
        core::Layer::load_state_dict(ctx, state_dict, prefix, allow_missing);
    }
}
}
