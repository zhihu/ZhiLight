#include "nn/attention/attention.h"
#include "nn/attention/attention_base.hpp"
#include "nn/layernorm/layernorm.h"
#include "nn/linear/linear.h"
#include "nn/attention/attention_kernel.h"
#include "nn/attention/flash_decoding.h"
#include "nn/position/rotary_embedding.h"
#include "nn/quant/int8/quant_kernel.h"
#include "model/model_context.h"
#include "model/dyn_batch_context.h"
#include "model/rag_buffer_context.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>
#include "bmengine/logger/kernel_time_trace.hpp"
#include <bmengine/logger/std_log_op.hpp>
#include "private/allocator.h"
#include "utils/env.h"
#include <iostream>

#include <cuda_runtime.h>

namespace nn {

using bmengine::core::DataType;
using bmengine::core::DistLayout;
using bmengine::core::Tensor;
using bmengine::functions::BinaryElementwiseOp;
using bmengine::functions::concat_tensor;
using bmengine::logger::str_cat;
using model::ModelContext;
using model::RagBufferContext;
using std::vector;
typedef std::vector<size_t> ShapeT;

class Attention::impl::NormalImpl : public Attention::impl {
public:
    unsigned int dim_model;
    unsigned int dim_head;
    unsigned int num_heads;
    unsigned int num_kv_heads;
    unsigned int num_head_groups;
    core::DataType dtype { DataType::kHalf };
    bool parallel;

    std::string pos_bias_type;
    float attn_scale;
    model::QuantConfig quant_kv;
    bool scale_weights;
    bool weight_transposed;
    float rope_theta;

    Linear project_q, project_k, project_v;
    Linear attn_out;

    RotaryEmbedding rotary_embedding;

    // fuse project_q, project_k and project_v
    std::unique_ptr<Linear> linear_qkv;
    std::unique_ptr<LayerNorm> q_norm;
    std::unique_ptr<LayerNorm> k_norm;

    functions::Gemm gemm_attn;
    functions::Gemm gemm_transB;
    functions::Gemm gemm_score_v;
    functions::Transpose transpose;

    FlashDecoding flash_decoding;

    int max_shared_memory;

    static model::QuantConfig as_quant_kv(model::QuantConfig quant) {
        if (quant.quant_weight_kv == 0) {
            quant.quant_type = model::QuantType::NoQuant;
        }
        return quant;
    }

    NormalImpl(const core::Context& ctx, model::ModelConfig cfg, model::QuantConfig quant, bool parallel)
        : dim_model(cfg.dim_model),
          dim_head(cfg.dim_head),
          num_heads(cfg.num_heads),
          num_kv_heads(cfg.num_kv_heads),
          num_head_groups(num_heads / num_kv_heads),
          dtype(cfg.dtype),
          parallel(parallel),
          pos_bias_type(cfg.pos_bias_type),
          attn_scale(1. / sqrtf(dim_head)),
          quant_kv(as_quant_kv(quant)),
          scale_weights(cfg.scale_weights),
          weight_transposed(cfg.weight_transposed),
          rotary_embedding(ctx, cfg),
          rope_theta(cfg.rope_theta),
          // clang-format off
          project_q(ctx, dim_model, dim_head * num_heads, "", quant, scale_weights, weight_transposed, parallel, DistLayout::COLUMNAR, dtype),
          project_k(ctx, dim_model, dim_head * num_kv_heads, "", quant_kv, scale_weights, weight_transposed, parallel, num_kv_heads > 1 ? DistLayout::COLUMNAR : DistLayout::REPLICATED, dtype),
          project_v(ctx, dim_model, dim_head * num_kv_heads, "", quant_kv, scale_weights, weight_transposed, parallel, num_kv_heads > 1 ? DistLayout::COLUMNAR : DistLayout::REPLICATED, dtype),
          attn_out(ctx, dim_head * num_heads, dim_model, "", quant, scale_weights, weight_transposed, parallel, DistLayout::ROW, dtype),
          gemm_attn(ctx, dtype, true, true),
          gemm_transB(ctx, dtype, false, true),
          gemm_score_v(ctx, dtype, false, false),
          transpose(ctx),
          flash_decoding(ctx) {
        if (cfg.model_type == "qwen2" || cfg.model_type == "qwen2_moe" || cfg.model_type == "qwen2_vl") {
            project_q.set_has_bias(true);
            project_k.set_has_bias(true);
            project_v.set_has_bias(true);
        }
        if (cfg.use_qk_norm) {
            q_norm = std::make_unique<LayerNorm>(ctx, dim_head * num_heads, false, cfg.eps, 1, dtype, num_heads);
            k_norm = std::make_unique<LayerNorm>(ctx, dim_head * num_kv_heads, false, cfg.eps, 1, dtype, num_kv_heads);
        }
        if (parallel) {
            if (ctx.high_precision() >= 2) {
                // use float to reduce sum
                attn_out.set_output_type(DataType::kFloat);
            }
            int ws = ctx.world_size();
            BM_ASSERT(num_heads % ws == 0, "num_heads must be dividable by world_size");
            BM_ASSERT(num_kv_heads % ws == 0,"num_kv_heads must be dividable by world_size");
            this->num_heads = num_heads / ctx.world_size();
            this->num_kv_heads = num_kv_heads / ctx.world_size();
        }
        max_shared_memory = ctx.get_max_shared_memory();
    }

    NormalImpl(const NormalImpl&) = delete;
    NormalImpl(NormalImpl&&) = default;
    virtual ~NormalImpl() = default;

    core::Tensor dynamic_batch_forward(
        model::ModelContext& ctx,
        const core::Tensor& hidden_q,
        const core::Tensor& position_or_bias,
      core::Tensor *output);

    Tensor attn_encode_group(
        model::ModelContext& ctx,
        Tensor h_q_enc,
        Tensor h_k_enc,
        Tensor h_v_enc,
        Tensor attn_value_enc);

    int get_split(model::ModelContext& ctx, size_t mem) {
        auto allocator = ctx.get_allocator();
        size_t free_memory = allocator->get_memory_limit() - allocator->used_memory();
        // const size_t mem_limit = 128 * 1024 * 1024 * sizeof(half);
        const size_t mem_limit = free_memory / 2;
        int n_split = 1;
        while (mem > mem_limit && (n_split * 2) <= num_kv_heads
               && (num_kv_heads % (n_split * 2) == 0)) {
            n_split *= 2;
            mem /= 2;
        }
        return n_split;
    }

    void attn_search_rag(
        model::ModelContext& ctx,
        const Tensor& h_q,
        const Tensor& h_k_s,
        const Tensor& h_v_s,
        const Tensor& placement_s,
        Tensor& attn_value_s);

    void attn_search_rag_naive(
        model::ModelContext& ctx,
        const Tensor& h_q,
        const Tensor& h_k_s,
        const Tensor& h_v_s,
        const Tensor& placement_s,
        Tensor& attn_value_s);

    Tensor mul_q_k_no_batch(
        model::ModelContext& ctx,
        const Tensor& q_trans, // (batch, num_kv_heads, num_head_groups * len_q, dim_head)
        const Tensor& buf_k    // (batch, num_kv_heads, len_buf, dim_head)
    );

    void mul_score_v(
        const core::Context& ctx,
        const Tensor& attn_score, // (batch, num_kv_heads, num_head_groups * len_q, len_buf)
        const Tensor& buf_v,      // (batch, num_kv_heads, len_buf, dim_head)
        Tensor* output);

    int get_event_level(const core::Context& ctx) {
        if (ctx.current_layer() == 1000 && ctx.active_device() == 0 && ctx.rank() == 0) {
            return 0;
        }
        return 2;
    }
    virtual core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& hidden_q, // (batch?, len_q, dim_model)
        const core::Tensor& mask,     // (batch?, len_q, len_buf) int8
        const core::Tensor&
            position_bias, // if relative (batch, num_head, len_q, len_buf) else if rotary (len_q)
        const core::Tensor& seqlens_q,   // (batch?, 1,)    int32
        const core::Tensor& seqlens_kv,  // (batch?, 1,)    int32
        core::Tensor* past_k,            // (batch, num_heads, len_buf, dim_head)
        core::Tensor* past_v,            // (batch, num_heads, len_buf, dim_head)
        const core::Tensor* block_table, // (batch, blocks_per_seq)
        const core::Tensor* placement,   // (batch?, len_q,)    int32
        core::Tensor* output) {
        if (seqlens_kv.numel() == 0) {
            core::EventScope event_scope(ctx, "Attention", 1);
            return forward_BHSD(
                ctx, hidden_q, mask, position_bias, past_k, past_v, placement);
        } else {
            core::EventScope event_scope(ctx, "Attention(Flash)", 1);
            return forward_BSHD(
                ctx, hidden_q, position_bias, seqlens_q, seqlens_kv, past_k, past_v, block_table);
        }
    }

    core::Tensor forward_BHSD(
        const core::Context& ctx,
        const core::Tensor& hidden_q, // (batch?, len_q, dim_model)
        const core::Tensor& mask,     // (batch?, len_q, len_buf) int8
        const core::Tensor&
            position_bias, // if relative (batch, num_head, len_q, len_buf) else if rotary (len_q)
        core::Tensor* past_k,          // (batch, num_heads, len_buf, dim_head)
        core::Tensor* past_v,          // (batch, num_heads, len_buf, dim_head)
        const core::Tensor* placement  // (batch?, len_q,)    int32
    ) {
        int event_level = get_event_level(ctx);

        size_t batch = (mask.ndim() == 2) ? 1 : mask.size(0);
        uint32_t len_q = mask.size(-2);
        uint32_t len_buf = mask.size(-1);

        const core::Tensor& key_buf =
            past_k == nullptr ? ctx.tensor({ batch, num_kv_heads, len_buf, dim_head }, dtype)
                              : past_k->view({ batch, num_kv_heads, len_buf, dim_head });
        const core::Tensor& val_buf =
            past_v == nullptr ? ctx.tensor({ batch, num_kv_heads, len_buf, dim_head }, dtype)
                              : past_v->view({ batch, num_kv_heads, len_buf, dim_head });

        int active_dev = ctx.active_device();
        BM_ASSERT(active_dev == key_buf.device(), "Invalid past_k device");
        BM_ASSERT(active_dev == val_buf.device(), "Invalid past_v device");
        if (placement != nullptr) {
            BM_ASSERT(active_dev == placement->device(), "Invalid placement device");
        }

        Tensor h_q = project_q(ctx, hidden_q); // (batch?, len_q, num_heads * dim_head)
        Tensor h_k = project_k(ctx, hidden_q); // (batch?, len_q, num_kv_heads * dim_head)
        Tensor h_v = project_v(ctx, hidden_q); // (batch?, len_q, num_kv_heads * dim_head)

        if (pos_bias_type == "rotary") {
            ctx.recordEvent("rotary", event_level);
            auto h_qk = rotary_embedding(ctx, position_bias, h_q, h_k);
            h_q = std::get<0>(h_qk);
            h_k = std::get<1>(h_qk);
        }

        cudaStream_t stream = ctx.current_stream()->ptr;
        ctx.recordEvent("copy_to_buffer,K&V", event_level);
        h_k = h_k.view({ batch, len_q, num_kv_heads, dim_head });
        h_v = h_v.view({ batch, len_q, num_kv_heads, dim_head });
        copy_to_buffer(num_kv_heads, len_q, len_buf, dim_head, placement, h_k, key_buf, stream);
        copy_to_buffer(num_kv_heads, len_q, len_buf, dim_head, placement, h_v, val_buf, stream);

        // (batch, len_q, num_heads, dim_head) => (batch, num_heads, len_q, dim_head)
        ctx.recordEvent("transposeQ", event_level);
        h_q = transpose_2_1(ctx, h_q.view({ batch, len_q, num_heads, dim_head }));
        h_q = h_q.view({ batch, num_kv_heads, num_head_groups * len_q, dim_head });

        // Q * K
        ctx.recordEvent("Q*K", event_level);
        Tensor attn_score = gemm_transB.forward(
            ctx,
            h_q,    // ColMajor: (batch, num_kv_heads, dim_head, num_head_groups * len_q)
            key_buf // ColMajor: (batch, num_kv_heads, len_buf, dim_head)T
        );          // (batch, num_kv_heads, num_head_groups * len_q, len_buf)

        // attn_softmax in-place update attn_score
        ctx.recordEvent("attn_softmax", event_level);
        const Tensor& pos_bias = pos_bias_type == "relative" ? position_bias : core::Tensor();
        Tensor attn_score_q = attn_score.view({ batch, num_heads, len_q, len_buf });
        attn_softmax(ctx, attn_scale, attn_score_q, mask, pos_bias);

        // Score * V
        ctx.recordEvent("Score*V", event_level);
        Tensor attn_res = gemm_score_v(
            ctx,
            attn_score, // ColMajor: (batch, num_kv_heads, len_buf, num_head_groups * len_q)
            val_buf     // ColMajor: (batch, num_kv_heads, dim_head, len_buf)
        );              // (batch, num_kv_heads, num_head_groups * len_q, dim_head)

        // transpose: (batch, num_heads, len_q, dim_head) => (batch, len_q, num_heads, dim_head)
        ctx.recordEvent("transposeAV", event_level);
        Tensor attn_value_t =
            transpose_2_1(ctx, attn_res.view({ batch, num_heads, len_q, dim_head }));
        ctx.recordEvent("End>transposeAV", event_level);

        ShapeT attn_value_shape = (mask.ndim() == 2)
                                    ? ShapeT({ len_q, num_heads * dim_head })
                                    : ShapeT({ batch, len_q, num_heads * dim_head });
        return attn_out(
            ctx, attn_value_t.view(attn_value_shape)); // return (batch?, len_q, dim_model)
    }

    core::Tensor forward_BSHD(
        const core::Context& ctx,
        const core::Tensor& hidden_q,   // (batch, len_q, dim_model)
        const core::Tensor& position,   // (len_q)  int32
        const core::Tensor& seqlens_q,  // (batch + 1)
        const core::Tensor& seqlens_kv, // (batch)
        const core::Tensor* past_k,     // (batch, len_buf, num_heads, dim_head)
        const core::Tensor* past_v,
        const core::Tensor* block_table) // (batch, blocks_per_seq)
    {                                    // (batch, len_buf, num_heads, dim_head)

        uint32_t batch_size = seqlens_kv.size(0);
        uint32_t len_q = hidden_q.size(-2);

        // BSHD for flash decoding, assuming even q/kv lengths when cache is not present.
        const core::Tensor key_cache =
            past_k == nullptr ? ctx.tensor({ batch_size, len_q, num_kv_heads, dim_head }, dtype)
                              : *past_k;
        const core::Tensor val_cache =
            past_v == nullptr ? ctx.tensor({ batch_size, len_q, num_kv_heads, dim_head }, dtype)
                              : *past_v;

        Tensor h_q = project_q(ctx, hidden_q); // (batch?, len_q, num_heads * dim_head)
        Tensor h_k = project_k(ctx, hidden_q); // (batch?, len_q, num_kv_heads * dim_head)
        Tensor h_v = project_v(ctx, hidden_q); // (batch?, len_q, num_kv_heads * dim_head)

        if (position.numel() == 0) {
            // unexpected
            BM_EXCEPTION("position is empty");
        }

        std::tie(h_q, h_k) = rotary_embedding(ctx, position, h_q, h_k);

        h_q = h_q.view({ batch_size, len_q, num_heads, dim_head });
        h_k = h_k.view({ batch_size, len_q, num_kv_heads, dim_head });
        h_v = h_v.view({ batch_size, len_q, num_kv_heads, dim_head });

        auto attn_res = flash_decoding.compact_kv_fwd(
            ctx, h_q, key_cache, val_cache, seqlens_q, seqlens_kv, h_k, h_v, block_table, Tensor());
        return attn_out(ctx, attn_res);
    }
};

void check_past_kv_dev(
    int active_dev, const Tensor* past_k, const Tensor* past_v, const Tensor* placement = nullptr) {
    BM_ASSERT(past_k != nullptr, "Invalid past_k");
    BM_ASSERT(past_v != nullptr, "Invalid past_v");
    BM_ASSERT(past_k->numel(), "Empty past_k");
    BM_ASSERT(past_v->numel(), "Empty past_v");
    BM_ASSERT_EQ(active_dev, past_k->device(), "Invalid past_k device");
    BM_ASSERT_EQ(active_dev, past_v->device(), "Invalid past_v device");
    if (placement != nullptr) {
        BM_ASSERT_EQ(active_dev, placement->device(), "Invalid placement device");
    }
}

static Tensor slice_dim0(Tensor* t, size_t i) {
    // remove dim0
    return t->slice_dim0(i, i + 1).chunk()[0];
}

Tensor Attention::impl::NormalImpl::mul_q_k_no_batch(
    model::ModelContext& ctx,
    const Tensor& q_trans, // (batch, num_kv_heads, num_head_groups * len_q, dim_head)
    const Tensor& buf_k    // (batch, num_kv_heads, len_buf, dim_head)
) {
    if (q_trans.ndim() == 3) {
        return gemm_transB.forward(ctx, q_trans, buf_k);
    }
    size_t batch = buf_k.size(0);
    size_t len_q = q_trans.size(-2) / num_head_groups;
    size_t len_buf = buf_k.size(-2);

    Tensor attn_score =
        ctx.tensor({ batch, num_kv_heads, num_head_groups * len_q, len_buf }, buf_k.dtype());

    for (size_t i = 0; i < batch; ++i) {
        Tensor tmp = attn_score.slice_dim0(i, i + 1);
        gemm_transB.forward(
            ctx,
            q_trans.slice_dim0(i, i + 1), // (1, num_kv_heads, dim_head, num_head_groups * len_q)
            buf_k.slice_dim0(i, i + 1),   // (1, num_kv_heads, len_buf, dim_head)T
            &tmp                          // (1, num_kv_heads, num_head_groups * len_q, len_buf)
        );
    }
    return attn_score;
}

void Attention::impl::NormalImpl::mul_score_v(
    const core::Context& ctx,
    const Tensor& attn_score, // (batch, num_kv_heads, num_head_groups * len_q, len_buf)
    const Tensor& buf_v,      // (batch, num_kv_heads, len_buf, dim_head)
    Tensor* output            // (batch, num_heads * dim_head, len_q) 3d

) {
    BM_ASSERT_EQ(attn_score.size(0), buf_v.size(0), "shape mismatch");
    BM_ASSERT_EQ(attn_score.size(1), buf_v.size(1), "shape mismatch");

    size_t batch = buf_v.size(0);
    size_t len_q = attn_score.size(-2) / num_head_groups;
    ShapeT shape_4d({ batch, num_kv_heads, dim_head, num_head_groups * len_q });
    ShapeT shape_3d({ batch, num_heads * dim_head, len_q });

    uintptr_t org_ptr = uintptr_t(output->data());
    *output = output->view(shape_4d);

    core::EventScope event_scope(ctx, "mul_score_v", 3);
    // (batch, num_kv_heads, dim_head, num_head_groups * len_q)
    core::Tensor attn_res = gemm_attn(ctx, buf_v, attn_score, output); // Trans A, B
    BM_ASSERT_EQ(attn_res.shape(), shape_4d, "shape mismatch");

    if (num_head_groups == 1) {
        *output = attn_res.view(shape_3d);
    } else {
        attn_res = transpose(ctx, attn_res).view({ batch * num_heads, len_q, dim_head });
        Tensor tmp = output->view({ batch * num_heads, dim_head, len_q });
        *output = transpose(ctx, attn_res, &tmp).view(shape_3d);
    }
    BM_ASSERT_EQ(uintptr_t(output->data()), org_ptr, "data ptr mismatch");
}

// clang-format off
Tensor copy_to_quant_buffer(model::ModelContext& ctx, Tensor& k);
Tensor Attention::impl::NormalImpl::attn_encode_group(
    model::ModelContext& ctx,
    Tensor h_q_enc,
    Tensor h_k_enc,
    Tensor h_v_enc,
    Tensor attn_value_enc  // (num_enc, num_heads * dim_head)
) {
    model::DynBatchContext* dyn_batch = ctx.dyn_batch().get();
    RagBufferContext* rag_buffer = ctx.rag_buffer().get();

    Tensor* past_k = ctx.buf_k(ctx.current_layer()); // (batch, num_heads, len_buf, dim_head)
    Tensor* past_v = ctx.buf_v(ctx.current_layer()); // (batch, num_heads, len_buf, dim_head)

    cudaStream_t stream = ctx.current_stream()->ptr;
    size_t n_rep = num_head_groups;
    int event_level = get_event_level(ctx);
    size_t num_enc = dyn_batch->e_placement.numel();
    attn_value_enc = attn_value_enc.view({num_enc, num_heads * dim_head});

    const Tensor* e_batch = ctx.identity(&dyn_batch->e_batch, "e_batch");
    // const Tensor* e_input_len = ctx.identity(&dyn_batch->e_input_len, "e_input_len");
    const Tensor* e_placement = ctx.identity(&dyn_batch->e_placement, "e_placement");

    h_q_enc = h_q_enc.view({ num_enc, num_heads, dim_head });
    h_k_enc = h_k_enc.view({ num_enc, num_kv_heads, dim_head });
    h_v_enc = h_v_enc.view({ num_enc, num_kv_heads, dim_head });

    BM_ASSERT(rag_buffer, "No rag buffer");

    size_t offset = 0;
    size_t batch_enc = dyn_batch->ev_batch.size();
    for (size_t i = 0; i < batch_enc; ++i) {
        int b = dyn_batch->ev_batch[i];
        size_t input_len = dyn_batch->ev_input_len[i];
        size_t full_input_len = dyn_batch->full_input_len[i]; // = input_len + cache_len
        size_t len_buf_b = dyn_batch->ev_len_buf[i];
        size_t len_buf = !rag_buffer ? past_v->size(ctx.is_BSHD() ? -3 : -2) : 0;
        // split current batch from group
        Tensor h_q = h_q_enc.slice_dim0_len(offset, input_len).view({input_len, num_heads, dim_head});
        Tensor h_k = h_k_enc.slice_dim0_len(offset, input_len).view({input_len, num_kv_heads, dim_head});
        Tensor h_v = h_v_enc.slice_dim0_len(offset, input_len).view({input_len, num_kv_heads, dim_head});
        Tensor placement = e_placement->slice_dim0_len(offset, input_len);

        auto ev_name = str_cat("Encode[", ctx.is_BSHD() ? "flash=True," : "", "heads=", num_heads, "]");
        core::EventScope ev_encode1(ctx, ev_name, event_level);

        Tensor key_buf;
        Tensor val_buf;
        Tensor mask = dyn_batch->encode_mask(ctx, i);
        mask = *ctx.identity(&mask, "mask");
        Tensor v_t = attn_value_enc.slice_dim0_len(offset, input_len)
            .view({input_len, num_heads, dim_head });
        if (rag_buffer->buf_k(b).is_quantized()) {
            BM_ASSERT(ctx.is_BSHD(), "flash attention only");
            size_t old_len = full_input_len - input_len;
            if (dyn_batch->unquant_key_buf.get() && (old_len == 0 || !dyn_batch->unquant_key_buf->empty())) {
                if (old_len == 0) {
                    BM_ASSERT(dyn_batch->unquant_key_buf->empty(), "");
                    size_t tmp_cache_len = round_up(dyn_batch->input_len_no_split + 20, 64);
                    *dyn_batch->unquant_key_buf = ctx.tensor({tmp_cache_len, num_kv_heads, dim_head}, dtype);
                    *dyn_batch->unquant_val_buf = ctx.tensor({tmp_cache_len, num_kv_heads, dim_head}, dtype);
                }
                key_buf = *dyn_batch->unquant_key_buf;
                val_buf = *dyn_batch->unquant_val_buf;
                copy_to_buffer(num_kv_heads, input_len, len_buf_b, dim_head, &placement, h_k, key_buf, stream, ctx.is_BSHD());
                copy_to_buffer(num_kv_heads, input_len, len_buf_b, dim_head, &placement, h_v, val_buf, stream, ctx.is_BSHD());
                rag_buffer->buf_k(b).copy(ctx, ctx.current_layer(), h_k, placement, old_len, false);
                rag_buffer->buf_v(b).copy(ctx, ctx.current_layer(), h_v, placement, old_len, false);
            } else {
                if (ctx.is_layer(0)) {
                    std::cout << "WARNING: de-quantize prompt kv cache!\n";
                }
                key_buf = rag_buffer->buf_k(b).copy(ctx, ctx.current_layer(), h_k, placement, old_len, true);
                val_buf = rag_buffer->buf_v(b).copy(ctx, ctx.current_layer(), h_v, placement, old_len, true);
            }
        } else {
            size_t old_len = full_input_len - input_len;
            key_buf = rag_buffer->buf_k(b).copy(ctx, ctx.current_layer(), h_k, placement, old_len);
            val_buf = rag_buffer->buf_v(b).copy(ctx, ctx.current_layer(), h_v, placement, old_len);
//            key_buf = rag_buffer->buf_k(b, ctx.current_layer());
//            val_buf = rag_buffer->buf_v(b, ctx.current_layer());
//            copy_to_buffer(num_kv_heads, input_len, len_buf_b, dim_head, &placement, h_k, key_buf, stream, ctx.is_BSHD());
//            copy_to_buffer(num_kv_heads, input_len, len_buf_b, dim_head, &placement, h_v, val_buf, stream, ctx.is_BSHD());
            int custom_attn = utils::get_int_env("CPM_CUSTOM_SELF_ATTN", 0);
            if (custom_attn && num_head_groups == 8 && !ctx.is_BSHD()) {
                multi_query_self_attention(ctx, h_q, key_buf, val_buf, mask, attn_scale, v_t, 0);
                offset += input_len;
                continue;
            }
        }

        if (ctx.is_BSHD()) {
            h_q = h_q.view({ 1, input_len, num_heads, dim_head });
            auto key_cache = key_buf.slice_dim0_len(0, full_input_len)
                .view({ 1, full_input_len, num_kv_heads, dim_head });
            auto val_cache = val_buf.slice_dim0_len(0, full_input_len)
                .view({ 1, full_input_len, num_kv_heads, dim_head });
            auto fa_out = v_t.view(h_q.size());
            // TODO: do varlen batch prefill, interface is ready, but need organize input/output/kv_cache
            flash_decoding(ctx, h_q, key_cache, val_cache, &fa_out);
            offset += input_len;
            continue;
        }

        ctx.recordEvent("transposeQ", event_level);
        h_q = transpose_2_1(ctx, h_q).view({ num_kv_heads, n_rep * input_len, dim_head });
        if (len_buf_b < len_buf) {
            ctx.recordEvent("CopyKV", event_level);
            Tensor tmp = ctx.tensor({ 2 * num_kv_heads, len_buf_b, dim_head }, dtype);
            BM_CUDART_ASSERT(cudaMemsetAsync(tmp.data(), 0, tmp.nbytes(), stream));
            key_buf = tmp.slice_dim0(0, num_kv_heads);
            val_buf = tmp.slice_dim0(num_kv_heads, 2 * num_kv_heads);
            copy_to_buffer(num_kv_heads, input_len, len_buf_b, dim_head, &placement, h_k, key_buf, stream);
            copy_to_buffer(num_kv_heads, input_len, len_buf_b, dim_head, &placement, h_v, val_buf, stream);
        }

        const Tensor& pos_bias = pos_bias_type == "relative" ? dyn_batch->e_position_bias(ctx, i) : Tensor();
        Tensor attn_res = ctx.tensor({num_kv_heads, num_head_groups * input_len, dim_head}, dtype);
        // split attn_score(space: O(n^2)) to reduce memory usage
        size_t attn_score_memory = num_heads * input_len * len_buf_b * core::get_elem_size(dtype);
        int n_split = rag_buffer ? get_split(ctx, attn_score_memory) : 1;
        if (n_split == 1) {
            // Q * K
            ctx.recordEvent("Q*K", event_level);
            Tensor attn_score = gemm_transB.forward(
                ctx,
                h_q,     // ColMajor: (num_kv_heads, dim_head, n_rep * input_len)
                key_buf  // ColMajor: (num_kv_heads, len_buf, dim_head)T
            );           // (num_kv_heads, n_rep * input_len, len_buf_b)

            // attn_softmax in-place update attn_score
            ctx.recordEvent("attn_softmax", event_level);
            Tensor attn_score_q = attn_score.view({num_heads, input_len, len_buf_b});
            attn_softmax(ctx, attn_scale, attn_score_q, mask, pos_bias);

            // Score * V
            ctx.recordEvent("Score*V", event_level);
            gemm_score_v(
                ctx,
                attn_score, // ColMajor: (num_kv_heads, len_buf, num_head_groups * len_q)
                val_buf,    // ColMajor: (num_kv_heads, dim_head, len_buf)
                &attn_res   // (num_kv_heads, num_head_groups * len_q, dim_head)
            );
        } else {
            if (ctx.debug() && ctx.current_layer() == 0)
                std::cout << "split=" << n_split << ", num_kv_heads=" << num_kv_heads
                    << ", input=" << input_len << ", len_buf=" << len_buf_b << "\n";
            auto ev_split = core::EventScope(ctx, "SlitAttn", event_level);
            size_t s_heads = num_kv_heads / n_split;
            // s_attn_score is reused
            Tensor s_attn_score = ctx.tensor({s_heads, n_rep * input_len, len_buf_b}, dtype);
            for (int j = 0; j < n_split; ++j) {
                gemm_transB.forward(
                    ctx,
                    h_q.slice_dim0_len(s_heads * j, s_heads),
                    key_buf.slice_dim0_len(s_heads * j, s_heads),
                    &s_attn_score);

                Tensor attn_score_q = s_attn_score.view({num_heads / n_split, input_len, len_buf_b});
                Tensor pos_bias1 = pos_bias_type == "relative"
                                   ? pos_bias.slice_dim0_len(s_heads * j, s_heads)
                                   : Tensor();
                attn_softmax(ctx, attn_scale, attn_score_q, mask, pos_bias1);

                Tensor attn_res1 = attn_res.slice_dim0_len(s_heads * j, s_heads);
                gemm_score_v(
                    ctx,
                    s_attn_score,
                    val_buf.slice_dim0_len(s_heads * j, s_heads),
                    &attn_res1
                );
            }
        }
        // transpose: (num_heads, len_q, dim_head) => (len_q, num_heads * dim_head)
        ctx.recordEvent("transposeAV", event_level);
        transpose_2_1(ctx, attn_res.view({ num_heads, input_len, dim_head }), &v_t);

        offset += input_len;
    }
    return attn_value_enc;
}
// （batch, len_q, num_heads, dim_head）=> （batch, num_heads, len_q, dim_head）
static core::Tensor transpose_q(
    const core::Context& ctx,
    const core::Tensor& h_q,
    int len_q,
    int event_level) {
    if (len_q == 1)
        return h_q; // no need transpose
    ctx.recordEvent("transpose_Q", event_level);
    return transpose_2_1(ctx, h_q);
}
//#pragma GCC push_options
//#pragma GCC optimize ("O0")
void Attention::impl::NormalImpl::attn_search_rag(
    model::ModelContext& ctx,
    const Tensor& h_q_s,  // （batch, len_q, num_heads, dim_head）
    const Tensor& h_k_s,  // （batch, len_q, num_kv_heads, dim_head）
    const Tensor& h_v_s,  // （batch, len_q, num_kv_heads, dim_head）
    const Tensor& placement_s,
    Tensor& attn_value_search  // (batch, len_q, num_heads, dim_head)
) {
    model::DynBatchContext* dyn_batch = ctx.dyn_batch().get();
    RagBufferContext* rag_buffer = ctx.rag_buffer().get();
    Tensor h_q = h_q_s;
    int event_level = get_event_level(ctx) + 10;
    size_t batch = h_q.size(0);
    size_t len_q = h_k_s.size(1);

    // workspace must create before rag_buffer->buf_k_addr() to prevent addresses change.
    auto ws = nn::get_mqa_workspace(ctx, h_q, dyn_batch->get_max_len_buf(), rag_buffer->is_cache_quant());
    ctx.recordEvent("copy_to_rag_buffer", event_level);
    Tensor quant_k;
    Tensor quant_v;
    if (rag_buffer->is_cache_quant()) {
        h_q = functions::typecast(ctx, h_q_s, DataType::kHalf);
        // +128 => unsigned
        quant_k = int8_op::quant_calc_scale(ctx, h_k_s, 127, 128);
        quant_v = int8_op::quant_calc_scale(ctx, h_v_s, 127, 128);
    }
    int layer = ctx.current_layer();
    auto gc_stopper = std::make_unique<core::GCStopper>(ctx); // Stop gc to make sure buffer k/v address's are valid
    Tensor buf_k_addr = rag_buffer->buf_k_addr(ctx, layer); // (batch) => (num_kv_heads, len_buf, dim_head)
    Tensor buf_v_addr = rag_buffer->buf_v_addr(ctx, layer); // (batch) => (num_kv_heads, len_buf, dim_head)
    Tensor scale_k_addr = rag_buffer->scale_k_addr(ctx, layer);
    Tensor scale_v_addr = rag_buffer->scale_v_addr(ctx, layer);
    const Tensor* s_len_buf = ctx.identity(&dyn_batch->s_len_buf, "s_len_buf");
    const Tensor* s_mask = ctx.identity(&dyn_batch->s_mask, "s_mask");
//    copy_to_rag_buffer(ctx, h_k_s, placement_s, *s_len_buf, buf_k_addr);
//    copy_to_rag_buffer(ctx, h_v_s, placement_s, *s_len_buf, buf_v_addr);
    if (rag_buffer->is_cache_quant()) {
        copy_to_rag_buffer2(ctx, placement_s, *s_len_buf, quant_k, quant_v, &buf_k_addr, &buf_v_addr);
        Tensor scale_k = *quant_k.quant_scale;
        Tensor scale_v = *quant_v.quant_scale;
        copy_to_rag_buffer2(ctx, placement_s, *s_len_buf, scale_k, scale_v, &scale_k_addr, &scale_v_addr, true);
    } else {
        copy_to_rag_buffer2(ctx, placement_s, *s_len_buf, h_k_s, h_v_s, &buf_k_addr, &buf_v_addr);
    }
    ctx.recordEvent("End->copy_to_rag_buffer", event_level);

    static const int use_fa_decoding = utils::get_int_env("USE_FA_DECODING", 0);
    // Warning: You need use our modified flash-attn if you want to use this feature.
    if (use_fa_decoding > 0 && dyn_batch->cu_q_seqlens.numel() > 0) {
        BM_ASSERT(len_q == 1, "len_q must be 1");
        auto batch_q = h_q_s.view({batch * len_q, num_heads, dim_head});
        auto batch_k = buf_k_addr.view_unchecked({dyn_batch->total_k, num_kv_heads, dim_head}, h_k_s.dtype());
        auto batch_v = buf_v_addr.view_unchecked({dyn_batch->total_k, num_kv_heads, dim_head}, h_v_s.dtype());
        auto fa_out = attn_value_search.view({batch * len_q, num_heads, dim_head});
        flash_decoding(ctx,
                       batch_q,
                       batch_k,
                       batch_v,
                       &fa_out,
                       &dyn_batch->cu_q_seqlens,
                       &dyn_batch->cu_k_seqlens,
                       dyn_batch->max_q_seqlen,
                       dyn_batch->max_k_seqlen,
                       true);
        return;
    }

    size_t sum_of_len_buf = dyn_batch->sum_len_buf();
    int max_len_buf = dyn_batch->get_max_len_buf();
    static int fuse_thres = utils::get_int_env("FUSE_ATTN_THRES", 0);
    if (num_head_groups == 1 && batch > 0) {
        // case1: attention of ragged kv buffer with fused kernel
        // Tensor attn_score_all = ctx.tensor({sum_of_len_buf, num_heads, len_q}, dtype);
        core::EventScope ev(ctx, "attention_qkv_rag_buffer", event_level);
        attention_qkv_rag_buffer(
            ctx, h_q, *s_len_buf,
            buf_k_addr,
            buf_v_addr,
            *s_mask,
            dyn_batch->get_position_bias_addresses(ctx),
            attn_scale,
            max_len_buf,
            attn_value_search);
        return;
    } else if (num_head_groups > 1 && batch > fuse_thres) {
        // case2: multiple-query-attention of ragged kv buffer with fused kernel
        // if (ctx.current_layer() == 1) event_level = 0;
        core::EventScope ev_tag(ctx,
            str_cat("multi_query_attention_rag_buffer,batch=", batch, ",total_len_buf=", sum_of_len_buf, ",max=", max_len_buf),
            event_level);
        multi_query_attention_rag_buffer(
            ctx, h_q, *s_len_buf,
            buf_k_addr,
            buf_v_addr,
            *s_mask,
            attn_scale,
            max_len_buf,
            attn_value_search,
            int(num_head_groups),
            -1,
            ws,
            scale_k_addr,
            scale_v_addr,
            dtype);
//        if (ctx.is_layer(0)) {
//            std::cout << "attn_value_search: " << attn_value_search << endl;
//        }
        return;
    }
    gc_stopper.reset();

    attn_search_rag_naive(ctx, h_q_s, h_k_s, h_v_s, placement_s, attn_value_search);
}

// OLD implementation without flash-attn and fused kernel.
void Attention::impl::NormalImpl::attn_search_rag_naive(
        model::ModelContext& ctx,
        const Tensor& h_q,
        const Tensor& h_k_s,
        const Tensor& h_v_s,
        const Tensor& placement_s,
        Tensor& attn_value_search) {
    model::DynBatchContext *dyn_batch = ctx.dyn_batch().get();
    RagBufferContext *rag_buffer = ctx.rag_buffer().get();
    int layer = ctx.current_layer();
    int event_level = get_event_level(ctx) + 10;
    size_t batch = h_q.size(0);
    size_t len_q = h_k_s.size(1);
    size_t sum_of_len_buf = dyn_batch->sum_len_buf();
    int max_len_buf = dyn_batch->get_max_len_buf();

    auto h_q_t = transpose_q(ctx, h_q, len_q, event_level)
        .view({ batch, num_kv_heads, num_head_groups * len_q, dim_head });
    auto h_q_chunk = h_q_t.chunk();  // batch => (num_kv_heads, num_head_groups * len_q, dim_head)
    auto attn_value_chunk = attn_value_search.chunk();

    // case3: attention of ragged kv buffer with multiple streams
    core::EventScope ev_tag(ctx, str_cat("attn_search_rag,batch=", batch, ",total_len_buf=", sum_of_len_buf), event_level);

    auto main_stream = ctx.current_stream();
    // create sub-streams; syn with main_stream/main_event
    std::vector<core::Stream> streams(std::min(size_t(8), batch));
    cudaEvent_t main_event;
    BM_CUDART_ASSERT(cudaEventCreate(&main_event));
    BM_CUDART_ASSERT(cudaEventRecord(main_event, main_stream->ptr));
    streams[0] = main_stream;
    for (size_t i = 1; i < streams.size(); ++i) {
        streams[i] = ctx.get_stream();
        // beginning rendezvous: sub-streams wait main_stream
        BM_CUDART_ASSERT(cudaStreamWaitEvent(streams[i]->ptr, main_event));
    }
    BM_CUDART_ASSERT(cudaEventDestroy(main_event));

    vector<Tensor> attn_scores(batch);  // (num_kv_heads, num_head_groups * len_q, len_buf)
    vector<Tensor> attn_results(batch);
    // Do attention of each task in a separate stream
    for (size_t i = 0; i < batch; ++i) {
        auto cur_stream = streams[i % streams.size()];
        ctx.set_current_stream(cur_stream);
        BM_ASSERT_EQ(uint64_t(cur_stream->ptr), uint64_t(ctx.current_stream()->ptr), "Wrong stream");

        Tensor key_buf = rag_buffer->buf_k(i, layer);  // Normal: (num_kv_heads, len_buf, dim_head)
        Tensor val_buf = rag_buffer->buf_v(i, layer);  // BSHD: (len_buf, num_kv_heads, dim_head)
        if (ctx.is_BSHD()) {
            // Transpose to (num_kv_heads, len_buf, dim_head)
            key_buf = key_buf.virtual_transpose(0, 1);
            val_buf = val_buf.virtual_transpose(0, 1);
        }
        size_t len_buf = key_buf.size(-2);
        // Q * K
        ctx.recordEvent("Q * K", event_level);
        // h_q_chunk[i]: (num_kv_heads, num_head_groups * len_q, dim_head)
        attn_scores[i] = gemm_transB.batch_3d(ctx, h_q_chunk[i], key_buf);

        // attn_softmax in-place update attn_score
        Tensor attn_score_q = attn_scores[i].view({ num_heads, len_q, len_buf });
        Tensor mask = dyn_batch->search_mask(ctx, i, len_q);
        const Tensor& pos_bias = pos_bias_type == "relative" ? dyn_batch->s_position_bias(ctx, i) : core::Tensor();
        ctx.recordEvent("attn_softmax", event_level);
        attn_softmax(ctx, attn_scale, attn_score_q, mask, pos_bias);

        // Score * V
        ctx.recordEvent("Score*V", event_level, 2UL * attn_scores[i].numel() * dim_head);
        Tensor tmp = attn_value_chunk[i].view({num_kv_heads, num_head_groups * len_q, dim_head});
        attn_results[i] = gemm_score_v.batch_3d(
            ctx,
            attn_scores[i],  // (num_kv_heads, num_head_groups * len_q, len_buf)
            val_buf,         // (num_kv_heads, len_buf, dim_head)
            len_q > 1 ? nullptr : &tmp);

        if (len_q > 1) {
            // (batch, num_heads, len_q, dim_head) => (batch, len_q, num_heads * dim_head)
            ctx.recordEvent("transposeAV", event_level);
            transpose_2_1(ctx, attn_results[i].view({num_heads, len_q, dim_head}), &attn_value_chunk[i]);
        }
        ctx.enable_debug(0);
    }
    ctx.set_current_stream(main_stream);

    for (size_t i = 1; i < streams.size(); ++i) {
        cudaEvent_t e;
        BM_CUDART_ASSERT(cudaEventCreate(&e));
        BM_CUDART_ASSERT(cudaEventRecord(e, streams[i]->ptr));
        //  ending rendezvous: main_stream wait sub-streams
        BM_CUDART_ASSERT(cudaStreamWaitEvent(main_stream->ptr, e));
        BM_CUDART_ASSERT(cudaEventDestroy(e));
    }
    // BM_CUDART_ASSERT(cudaDeviceSynchronize());
}

Tensor Attention::impl::NormalImpl::dynamic_batch_forward(
    model::ModelContext& ctx,
    const Tensor& hidden_q,  // (group_len_q, dim_model)
    const core::Tensor& position_bias,
    core::Tensor* output
) {
    model::DynBatchContext* dyn_batch = ctx.dyn_batch().get();
    cudaStream_t stream = ctx.current_stream()->ptr;
    size_t n_rep = num_head_groups;

    BM_ASSERT(ctx.rag_buffer(), "");
    int event_level = get_event_level(ctx);
    core::EventScope ev(ctx, "Attention(DynBatch)", 1);

    Tensor g_h_q;
    Tensor g_h_k;
    Tensor g_h_v;
    static int fuse_pkv = utils::get_int_env("CPM_FUSE_QKV", 0);
    static int fuse_v2_thres = utils::get_int_env("FUSE_V2_THRES", 8);
    if (linear_qkv.get() && (fuse_pkv == 1 || fuse_pkv == 2 && hidden_q.size(0) <= fuse_v2_thres)) {
        // fuse Q, K, V
        auto a = linear_qkv->forward(ctx, hidden_q); // (group_len_q, (num_heads + 2 * num_kv_heads) * dim_head)
        BM_ASSERT_EQ(a.size(-1), (num_heads + 2 * num_kv_heads) * dim_head, "");
        if (dyn_batch->rope_cache.cos.numel() > 0) {
            if (ctx.is_layer(1000)) std::cout << "rope_qk_cache\n";
            rope_qk_cache(ctx,
                          dyn_batch->rope_cache.cos,
                          dyn_batch->rope_cache.sin,
                          a,
                          g_h_q, g_h_k, g_h_v,
                          num_heads, num_kv_heads, dim_head, dtype,
                          rotary_embedding.is_neox_style());
        } else if (rotary_embedding.is_normal() && rotary_embedding.is_neox_style()) {
        rotary_embedding_qk(ctx, position_bias, a, g_h_q, g_h_k, g_h_v, num_heads, num_kv_heads, dim_head, rope_theta, dtype);
        } else {
            // if (ctx.is_layer(1)) std::cout << "Split to rotary_embedding\n";
            // g_h_q = functions::slice_last_dim(ctx, a, 0, num_heads * dim_head);
            // g_h_k = functions::slice_last_dim(ctx, a,  num_heads * dim_head, num_kv_heads * dim_head);
            g_h_q = a.virtual_slice(0, num_heads * dim_head);
            g_h_k = a.virtual_slice(num_heads * dim_head, num_kv_heads * dim_head);
            g_h_v = functions::slice_last_dim(ctx, a, (num_heads + num_kv_heads) * dim_head, num_kv_heads * dim_head);
            g_h_q = rotary_embedding.rotate(ctx, position_bias, g_h_q);
            g_h_k = rotary_embedding.rotate(ctx, position_bias, g_h_k);
        }
    } else {
        g_h_q = project_q(ctx, hidden_q);   // (group_len_q, num_heads * dim_head)
        g_h_k = project_k(ctx, hidden_q);   // (group_len_q, num_kv_heads * dim_head)
        g_h_v = project_v(ctx, hidden_q);   // (group_len_q, num_kv_heads * dim_head)

        if (q_norm) {
            g_h_q = q_norm->forward(ctx, g_h_q);
            g_h_k = k_norm->forward(ctx, g_h_k);
        }
        if (pos_bias_type == "rotary") {
            auto h_qk = rotary_embedding(ctx, position_bias, g_h_q, g_h_k);
            g_h_q = std::get<0>(h_qk);
            g_h_k = std::get<1>(h_qk);
        }
    }

    bool has_encode = !dyn_batch->ev_batch.empty();
    size_t num_enc = dyn_batch->e_placement.numel();
    size_t num_s = dyn_batch->s_placement.numel();
//    std::cout << "num_enc=" << num_enc << ", num_s=" << num_s << ", all=" << g_h_q.size(0) << "\n";
    BM_ASSERT_EQ(num_enc + num_s, g_h_q.size(0), "dim mismatch");
    Tensor attn_val_g = ctx.tensor({g_h_q.size(0), num_heads * dim_head}, dtype);

    Tensor attn_value_enc;
    if (has_encode) {
        attn_value_enc = attn_encode_group(
            ctx,
            g_h_q.slice_dim0(0, num_enc),
            g_h_k.slice_dim0(0, num_enc),
            g_h_v.slice_dim0(0, num_enc),
            attn_val_g.slice_dim0(0, num_enc));
    }
    if (num_s == 0) {
        return attn_out.forward(ctx, attn_val_g); // (group_len_q, dim_model)
    }

    // search part
    ctx.recordEvent("Start>Search,len=" + std::to_string(num_s), event_level);
    const Tensor* s_placement = ctx.identity(&dyn_batch->s_placement, "s_placement"); // (batch, len_q)
    BM_ASSERT_EQ(2, s_placement->ndim(), "placement is not 2d");
    size_t batch = s_placement->size(0);
    size_t len_q = s_placement->size(1);

    Tensor h_q = g_h_q.slice_dim0(num_enc, num_enc + num_s).view({batch, len_q, num_heads, dim_head});
    Tensor h_k = g_h_k.slice_dim0(num_enc, num_enc + num_s).view({batch, len_q, num_kv_heads, dim_head});
    Tensor h_v = g_h_v.slice_dim0(num_enc, num_enc + num_s).view({batch, len_q, num_kv_heads, dim_head});

    Tensor attn_value_search = attn_val_g.slice_dim0_len(num_enc, num_s)
        .view({batch, len_q, num_heads, dim_head});

    attn_search_rag(ctx, h_q, h_k, h_v, *s_placement, attn_value_search);
    ctx.recordEvent("End>Search,len=" + std::to_string(num_s), event_level);

    auto ret = attn_out.forward(ctx, attn_val_g); // (group_len_q, dim_model)
//    std::cout << "#### ret: " << ret << endl;
    return ret;
}

Attention::Attention(
    const core::Context& ctx, model::ModelConfig cfg, model::QuantConfig quant_cfg, bool parallel)
    : core::Layer() {
    impl::NormalImpl* normal_impl = nullptr;
    if (cfg.kv_lora_rank > 0) {
        pimpl.reset(impl::create_mla_impl(ctx, cfg, quant_cfg));
        pimpl->add_submodules(this);
    } else {
        normal_impl = new impl::NormalImpl(ctx, cfg, quant_cfg, parallel);
    }

    if (normal_impl) {
        add_submodule("project_q", normal_impl->project_q);
        add_submodule("project_k", normal_impl->project_k);
        add_submodule("project_v", normal_impl->project_v);
        add_submodule("attn_out", normal_impl->attn_out);
        // gemm has no weight; add only for set prefix
        add_submodule("gemm_attn", normal_impl->gemm_attn);
        add_submodule("gemm_transB", normal_impl->gemm_transB);
        if (ctx.high_precision() >= 1) {
            normal_impl->gemm_attn.set_compute_type(CUBLAS_COMPUTE_32F);
            normal_impl->gemm_transB.set_compute_type(CUBLAS_COMPUTE_32F);
        }
        if (normal_impl->q_norm) {
            add_submodule("q_norm", normal_impl->q_norm.get());
            add_submodule("k_norm", normal_impl->k_norm.get());
        }
        pimpl.reset(normal_impl);
    }
}
Attention::~Attention() = default;

core::Tensor Attention::forward(
    const core::Context& ctx,
    const core::Tensor& hidden_q, // (len_q, dim_model)
    const core::Tensor& mask,     // (len_q, len_buf) int8
    const core::Tensor&
        position_bias,              // if relative (num_head, len_q, len_buf) else if rotary (len_q)
    const core::Tensor& seqlens_q,  // (batch? 1,)    int32
    const core::Tensor& seqlens_kv, // (batch? 1,)    int32
    const core::Tensor* c_past_k,   // (num_heads, len_buf, dim_head)
    const core::Tensor* c_past_v,   // (num_heads, len_buf, dim_head)
    const core::Tensor* block_table, // (batch, blocks_per_seq)
    const core::Tensor* placement,   // (batch? len_q,)    int32
    core::Tensor* output) {
    ModelContext* m_ctx = dynamic_cast<ModelContext*>(const_cast<core::Context*>(&ctx));
    if (m_ctx && m_ctx->dyn_batch()) {
        return pimpl->dynamic_batch_forward(*m_ctx, hidden_q, position_bias, output);
    }
    // core::EventScope event_scope(ctx, "Attention", 1);
    Tensor* past_k = const_cast<Tensor*>(c_past_k);
    Tensor* past_v = const_cast<Tensor*>(c_past_v);
    impl::NormalImpl* p = dynamic_cast<impl::NormalImpl*>(pimpl.get());
    return p->forward(
        ctx,
        hidden_q,
        mask,
        position_bias,
        seqlens_q,
        seqlens_kv,
        past_k,
        past_v,
        block_table,
        placement,
        output);
}

core::Tensor Attention::dyn_rag_forward(
    model::ModelContext& ctx,
    const core::Tensor& inp,         // (grouped_len_q, dim_model)
    const core::Tensor& position,    // (grouped_len_q)
    core::Tensor* output) {
    return pimpl->dynamic_batch_forward(ctx, inp, position, output);
}

const Linear& Attention::att_out() const {
    impl::NormalImpl* p = dynamic_cast<impl::NormalImpl*>(pimpl.get());
    return p->attn_out;
}

void Attention::load_state_dict(
    const core::Context& ctx,
    const std::map<std::string, const core::Tensor>& state_dict,
    const std::string& prefix,
    bool allow_missing) {
    core::Layer::load_state_dict(ctx, state_dict, prefix, allow_missing);
    int fuse_pkv = utils::get_int_env("CPM_FUSE_QKV", 0);
    impl::NormalImpl* p = dynamic_cast<impl::NormalImpl*>(pimpl.get());
    if (fuse_pkv && p) {
        auto a = Linear::fuse(ctx, p->project_q, p->project_k, p->project_v);
        p->linear_qkv = std::unique_ptr<Linear>(a);
    }
    pimpl->on_load(ctx);
}
}
