#include "nn/attention/attention.h"
#include "nn/attention/attention_base.hpp"
#include "nn/attention/ds_flash_mla_api.h"
#include "nn/layernorm/layernorm.h"
#include "nn/linear/linear.h"
#include "nn/attention/attention_kernel.h"
#include "nn/attention/flash_decoding.h"
#include "nn/position/rotary_embedding.h"
#include "model/model_context.h"
#include "model/dyn_batch_context.h"
#include "model/rag_buffer_context.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>
#include <bmengine/functions/transpose.h>
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
using bmengine::functions::transpose_2_1;
using bmengine::logger::str_cat;
using model::ModelContext;
using std::vector;
typedef std::vector<size_t> ShapeT;
typedef std::unique_ptr<Linear> LinearPtr;

Tensor all_gather(model::ModelContext& ctx, Tensor& a, size_t real_len) {
    auto a_shape = a.shape();
    size_t part_len = ceil_div(real_len, ctx.world_size());
    Tensor a1 = a;
    if (a_shape[0] < part_len) {
        BM_ASSERT_EQ(ctx.rank() + 1, ctx.world_size(), "Not last rank");
        a_shape[0] = part_len;
        // TODO: need copy?
        a1 = a.view_unchecked(a_shape, a.dtype());
    }
    Tensor x = ctx.all_gather(a1).slice_dim0(0, real_len);
    BM_CUDART_ASSERT(cudaStreamSynchronize(ctx.current_cuda_stream()));
    functions::check_numeric(ctx, x);
    return x;
}

// clang-format off
class Attention::impl::MLAImpl : public Attention::impl {
    DataType dtype;
    size_t dim_model;
    size_t hidden_size;
    size_t num_heads;
    size_t num_kv_heads;

    size_t q_lora_rank;
    size_t kv_lora_rank;
    size_t nope_head_dim;
    size_t pe_head_dim;
    size_t qk_head_dim; // nope_head_dim + pe_head_dim
    size_t v_head_dim;

    float attn_scale;

    RotaryEmbedding rotary_emb;
    FlashDecoding flash_decoding;
    functions::Gemm gemm;
    functions::Gemm gemm_transB;

    LinearPtr q_proj;
    LinearPtr q_a_proj; // (hidden_size, q_lora_rank)T
    std::unique_ptr<LayerNorm> q_a_layernorm;
    LinearPtr q_b_proj; // (q_lora_rank, num_heads * qk_head_dim)T
    LinearPtr q_proj_nope;
    LinearPtr q_proj_pe;

    LinearPtr kv_a_proj_with_mqa; // (hidden_size, kv_lora_rank + pe_head_dim)T
    LinearPtr kv_b_proj;  // (kv_lora_rank, num_heads * (nope_head_dim + v_head_dim)T
    std::unique_ptr<LayerNorm> kv_a_layernorm;

    LinearPtr kv_a_proj_lora; // (hidden_size, kv_lora_rank)T
    LinearPtr k_a_proj_pe; // (hidden_size, pe_head_dim)T
    LinearPtr k_proj; // Split from kv_b_proj. (kv_lora_rank, num_heads * nope_head_dim)T
    LinearPtr v_proj; // Split from kv_b_proj. (kv_lora_rank, num_heads * v_head_dim)T

    LinearPtr qkv_a_proj_with_pe; // (hidden_size, q_lora_rank + kv_lora_rank + pe_head_dim)T

    LinearPtr o_proj; // (num_heads * v_head_dim, hidden_size)T

    // Data parallel:
    // 1=>v1, 2=>v2.
    // v1: attn dp, cache replicated;
    // v2: attn dp, cache dp.
    int data_parallel;
    int data_parallel_out;
    LinearPtr q_proj_full;
    LinearPtr q_b_proj_full;
    LinearPtr kv_b_proj_full; // kv_lora_rank => num_heads * (nope_head_dim + v_head_dim)
    LinearPtr k_proj_full; // (kv_lora_rank, num_heads * nope_head_dim)T
    LinearPtr v_proj_full; // (kv_lora_rank, num_heads * v_head_dim)T
    LinearPtr o_proj_full;

    int event_level { 2 };

    double yarn_get_mscale(double scale, double mscale) {
        if (scale <= 1.)
            return 1.;
        return 0.1 * mscale * log(scale) + 1.0;
    }

public:
    MLAImpl(const core::Context& ctx, const model::ModelConfig& cfg, model::QuantConfig quant)
    : dtype(cfg.dtype),
      dim_model(cfg.dim_model),
      hidden_size(dim_model),
      num_heads(cfg.num_heads),
      num_kv_heads(cfg.num_kv_heads),
      q_lora_rank(cfg.q_lora_rank),
      kv_lora_rank(cfg.kv_lora_rank),
      nope_head_dim(cfg.qk_nope_head_dim),
      pe_head_dim(cfg.qk_rope_head_dim),
      qk_head_dim(nope_head_dim + pe_head_dim),
      attn_scale(1. / sqrt(qk_head_dim)),
      v_head_dim(cfg.v_head_dim),
      rotary_emb(ctx, cfg),
      flash_decoding(ctx),
      gemm(ctx, dtype, false, false),
      gemm_transB(ctx, dtype, false, true)
    {
        if (utils::get_int_env("MLA_FORCE_HALF")) {
            dtype = DataType::kHalf;  // TODO: cast weight for layernorm
        }
        if (cfg.rope_cfg.mscale_all_dim > 0.) {
            double mscale = yarn_get_mscale(cfg.rope_cfg.factor, cfg.rope_cfg.mscale_all_dim);
            attn_scale = float(attn_scale * mscale * mscale);
        }

        data_parallel = utils::get_int_env("ATTN_DATA_PARALLEL", 0);
        data_parallel_out = utils::get_int_env("ATTN_DATA_PARALLEL_OUT", data_parallel);
        DistLayout b_layout = DistLayout::COLUMNAR;
        DistLayout o_layout = DistLayout::ROW;
        if (data_parallel) {
            b_layout = DistLayout::REPLICATED;
        }
        if (data_parallel_out) {
            o_layout = DistLayout::REPLICATED;
        }
        if (q_lora_rank <= 0) {
            q_proj = std::make_unique<Linear>(ctx, hidden_size, num_heads * qk_head_dim, quant, b_layout, dtype);
        } else {
            q_a_proj = std::make_unique<Linear>(ctx, hidden_size, q_lora_rank, quant, DistLayout::REPLICATED, dtype);
            q_a_layernorm = std::make_unique<LayerNorm>(ctx, q_lora_rank, false, cfg.eps, 1.0, dtype);
            q_b_proj = std::make_unique<Linear>(ctx, q_lora_rank, num_heads * qk_head_dim, quant, b_layout, dtype);
        }

        // kv_lora_rank + qk_rope_head_dim = 512 + 64
        kv_a_proj_with_mqa = std::make_unique<Linear>(ctx, hidden_size, kv_lora_rank + pe_head_dim, quant, DistLayout::REPLICATED, dtype);
        kv_a_layernorm = std::make_unique<LayerNorm>(ctx, kv_lora_rank, false, cfg.eps, 1.0, dtype);
        kv_b_proj = std::make_unique<Linear>(ctx, kv_lora_rank, num_heads * (nope_head_dim + v_head_dim), quant, b_layout, dtype);

        o_proj = std::make_unique<Linear>(ctx, num_heads * v_head_dim, hidden_size, quant, o_layout, dtype);

        BM_ASSERT(num_heads % ctx.world_size() == 0, "num_heads must be dividable by world_size");
        BM_ASSERT(num_kv_heads % ctx.world_size() == 0,"num_kv_heads must be dividable by world_size");
        this->num_heads = num_heads / ctx.world_size();
        this->num_kv_heads = num_kv_heads / ctx.world_size();

        // TODO: check this
        gemm.set_compute_type(CUBLAS_COMPUTE_32F);
        gemm_transB.set_compute_type(CUBLAS_COMPUTE_32F);
    }

    void split_out(const core::Context& ctx, const LinearPtr& full, LinearPtr& part) {
        vector<Linear*> splits = full->split(ctx, ctx.world_size(), true);
        if (!splits.empty()) {
            part.reset(splits[ctx.rank()]);
            splits[ctx.rank()] = nullptr;
            for (auto p: splits) delete p;
            return;
        }
        Tensor w = full->get_dequant_weight(ctx);
        size_t dim_out = w.size(0) / ctx.world_size();
        Tensor w_slice = w.slice_dim0_len(ctx.rank() * dim_out, dim_out);
        w_slice = ctx.copy(w_slice);
        part = std::make_unique<Linear>(ctx, full->name, w_slice);
    }

    void split_in(const core::Context& ctx, const LinearPtr& full, LinearPtr& part) {
        vector<Linear*> splits = full->split(ctx, ctx.world_size(), false);
        if (!splits.empty()) {
            part.reset(splits[ctx.rank()]);
            splits[ctx.rank()] = nullptr;
            for (auto p: splits) delete p;
            return;
        }
        Tensor w = full->get_dequant_weight(ctx);
        int dim_in = w.size(1) / ctx.world_size();
        Tensor w_slice = functions::slice_last_dim(ctx, w, ctx.rank() * dim_in, dim_in);
        part = std::make_unique<Linear>(ctx, full->name, w_slice);
    }

    void on_load(const core::Context& ctx) override {
        static int latent_cache = utils::get_int_env("LATENT_CACHE", 0);
        if (latent_cache == 0) return;
        if (data_parallel > 0) {
            q_proj_full.swap(q_proj);
            q_b_proj_full.swap(q_b_proj);
            kv_b_proj_full.swap(kv_b_proj);
            if (q_lora_rank <= 0) {
                split_out(ctx, q_proj_full, q_proj);
            } else {
                q_b_proj_full->name = "q_b_proj:q_lora=>H*192";
                split_out(ctx, q_b_proj_full, q_b_proj);
            }
            split_out(ctx, kv_b_proj_full, kv_b_proj);
            if (data_parallel_out) {
                o_proj_full.swap(o_proj);
                split_in(ctx, o_proj_full, o_proj);
            }
            // kv_lora_rank => num_heads * (nope_head_dim + v_head_dim)
            Tensor w = kv_b_proj_full->get_dequant_weight(ctx);
            Tensor w3d = w.view({ctx.world_size() * num_heads, nope_head_dim + v_head_dim, kv_lora_rank});
            auto [w_a, w_b] = split_dim1(ctx, w3d, nope_head_dim, v_head_dim);
            k_proj_full = std::make_unique<Linear>(ctx, "k_proj", w_a);
            v_proj_full = std::make_unique<Linear>(ctx, "v_proj", w_b);
        }
        if (false) {
            Tensor w = (q_lora_rank <= 0 ? q_proj : q_b_proj)->get_dequant_weight(ctx);
            Tensor w3d = w.view({num_heads, qk_head_dim, w.size(-1)});
            // slit out dim
            auto [w_a, w_b] = split_dim1(ctx, w3d, nope_head_dim, pe_head_dim);
            q_proj_nope = std::make_unique<Linear>(ctx, "q_proj_nope", w_a);
            q_proj_pe = std::make_unique<Linear>(ctx, "q_proj_pe", w_b);
            // (q_lora_rank <= 0 ? q_proj : q_b_proj).reset();
        }
        if (false) {
            Tensor w = kv_a_proj_with_mqa->get_dequant_weight(ctx);
            auto [w_a, w_b] = split_dim0(w, kv_lora_rank, pe_head_dim);
            kv_a_proj_lora = std::make_unique<Linear>(ctx, "kv_a_proj_lora", w_a);
            k_a_proj_pe = std::make_unique<Linear>(ctx, "k_a_proj_pe", w_b);
            // kv_a_proj_with_mqa.reset();
        }
        {
            Tensor w = kv_b_proj->get_dequant_weight(ctx);
            Tensor w3d = w.view({num_heads, nope_head_dim + v_head_dim, kv_lora_rank});
            auto [w_a, w_b] = split_dim1(ctx, w3d, nope_head_dim, v_head_dim);
            k_proj = std::make_unique<Linear>(ctx, "k_proj", w_a);
            v_proj = std::make_unique<Linear>(ctx, "v_proj", w_b);
        }
        if (q_lora_rank > 0) {
            qkv_a_proj_with_pe.reset(Linear::fuse(ctx, *q_a_proj, *kv_a_proj_with_mqa));
            // BM_ASSERT(qkv_a_proj_with_pe.get(), "");
            if (!qkv_a_proj_with_pe) {
                // BM_ASSERT(false, "qkv_a_proj_with_pe is null");
                Tensor w1 = q_a_proj->get_dequant_weight(ctx);
                Tensor w2 = kv_a_proj_with_mqa->get_dequant_weight(ctx);
                Tensor w_all = functions::concat_tensor(ctx, w1, w2, 0);
                qkv_a_proj_with_pe = std::make_unique<Linear>(ctx, "qkv_a_proj_with_pe", w_all);
            }
            if (qkv_a_proj_with_pe)
                qkv_a_proj_with_pe->name = "qkv_a_proj_with_pe";
        }
        if (is_cache_dp()) {
            BM_ASSERT(qkv_a_proj_with_pe, "");
        }
    }

    bool is_cache_dp() {
        return data_parallel == 2;
    }

    void add_submodules(core::Layer* layer) override {
        if (q_lora_rank <= 0) {
            layer->add_submodule("project_q", q_proj.get());
        } else {
            layer->add_submodule("q_a_proj", q_a_proj.get());
            layer->add_submodule("q_a_layernorm", q_a_layernorm.get());
            layer->add_submodule("q_b_proj", q_b_proj.get());
        }
        layer->add_submodule("kv_a_proj_with_mqa", kv_a_proj_with_mqa.get());
        layer->add_submodule("kv_a_layernorm", kv_a_layernorm.get());
        layer->add_submodule("kv_b_proj", kv_b_proj.get());

        layer->add_submodule("attn_out", o_proj.get());
    }

    void attn_encode_group(
        model::ModelContext& ctx,
        Tensor h_q_enc,
        Tensor h_k_enc,
        Tensor h_v_enc,
        Tensor attn_value_enc  // (num_enc, num_heads * dim_head)
    );

    void attn_search_rag(
        model::ModelContext& ctx,
        const Tensor& h_q_s,
        const Tensor& h_k_s,
        const Tensor& h_v_s,
        const Tensor& placement_s,
        Tensor& attn_value_s);

    std::pair<Tensor, Tensor> split_dim0(const Tensor& q, size_t sz_a, size_t sz_b) {
        BM_ASSERT_EQ(q.size(0), sz_a + sz_b, "size mismatch");
        Tensor a = q.slice_dim0_len(0, sz_a);
        Tensor b = q.slice_dim0_len(sz_a, sz_b);
        return std::make_pair(a, b);
    }

    std::pair<Tensor, Tensor> split_dim1(const core::Context& ctx, const Tensor& q, size_t sz_a, size_t sz_b) {
        BM_ASSERT_EQ(q.ndim(), 3, "dim mismatch");
        BM_ASSERT_EQ(q.size(1), sz_a + sz_b, "size mismatch");
        Tensor tmp = transpose_2_1(ctx, q);
        auto [a, b] = split_dim0(tmp, sz_a, sz_b);
        a = transpose_2_1(ctx, a);
        b = transpose_2_1(ctx, b);
        a = a.view({a.size(0) * a.size(1), a.size(2)});
        b = b.view({b.size(0) * b.size(1), b.size(2)});
        return std::make_pair(a, b);
    }

    std::pair<Tensor, Tensor> split(const core::Context& ctx, const Tensor& q, size_t sz_a, size_t sz_b) {
        BM_ASSERT_EQ(q.size(-1), sz_a + sz_b, "size mismatch");
        Tensor a = functions::slice_last_dim(ctx, q, 0, sz_a);
        Tensor b = functions::slice_last_dim(ctx, q, sz_a, sz_b);
        return std::make_pair(a, b);
    }

    core::Tensor dynamic_batch_forward(
        model::ModelContext& ctx,
        const Tensor& hidden_q,  // (group_len_q, dim_model)
        const core::Tensor& position,
        core::Tensor* output
    ) override ;

    // NOTE: if q_lora_rank, h is q_a
    Tensor forward_q(const core::Context& ctx, const Tensor& h, bool norm=true, bool full=false) {
        Tensor q; // (len_q, num_heads * qk_head_dim)
        if (q_lora_rank <= 0) {
            q = (full ? q_proj_full : q_proj)->forward(ctx, h);
        } else {
            // q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
            Tensor q_low_rank = !norm ? h : q_a_layernorm->forward(ctx, q_a_proj->forward(ctx, h));
            q = (full ? q_b_proj_full : q_b_proj)->forward(ctx, q_low_rank);
        }
        return q.view({q.size(0), q.size(1) / qk_head_dim, qk_head_dim});
    }

    std::pair<Tensor, Tensor> forward_q_sep(const core::Context& ctx, const Tensor& h) {
        Tensor q0 = h;
        if (q_lora_rank > 0) {
            q0 = q_a_layernorm->forward(ctx, q_a_proj->forward(ctx, h));
        }
        Tensor q_nope = q_proj_nope->forward(ctx, q0);
        Tensor q_pe = q_proj_pe->forward(ctx, q0);
        return {q_nope, q_pe};
    }

    Tensor forward_q_and_pe(const core::Context& ctx, const Tensor& h, const Tensor& position) {
        Tensor q = forward_q(ctx, h); // (len_q, num_heads, qk_head_dim)
        {
            auto[q_nope, q_pe] = split(ctx, q, nope_head_dim, pe_head_dim);
            q_pe = rotary_emb.rotate(ctx, position, q_pe);
            return functions::concat_tensor(ctx, q_nope, q_pe);
        }
    }

    std::tuple<Tensor, Tensor, Tensor> forward_kv_cache(const core::Context& ctx, const Tensor& compressed_kv) {
        // k_pe was already rotated
        auto [kv_a, k_pe] = split(ctx, compressed_kv, kv_lora_rank, pe_head_dim);
        size_t len_q = kv_a.size(0);
        // up => 3D: {len_q, num_heads, (qk_nope_head_dim + v_head_dim)}
        if (k_proj) {
            Tensor k_nope = k_proj->forward(ctx, kv_a).view({len_q, num_kv_heads, nope_head_dim});
            Tensor v = v_proj->forward(ctx, kv_a).view({len_q, num_kv_heads, v_head_dim});
            return {k_nope, k_pe, v};
        }
        Tensor kv = kv_b_proj->forward(ctx, kv_a);
        kv = kv.view({len_q, num_kv_heads, (nope_head_dim + v_head_dim)});
        auto [k_nope, v] = split(ctx, kv, nope_head_dim, v_head_dim);
        return {k_nope, k_pe, v};
    }

    core::Tensor forward_compressed_cache(
        model::ModelContext& ctx,
        const Tensor& hidden_q,  // (group_len_q, dim_model)
        const core::Tensor& position,
        core::Tensor* output
    );
    core::Tensor forward_compressed_dp_v1(
        model::ModelContext& ctx,
        const Tensor& hidden_q,  // (group_len_q, dim_model)
        const core::Tensor& position,
        core::Tensor* output
    );
    void copy_to_compressed_cache(
        model::ModelContext& ctx,
        model::DynBatchContext* dyn_batch,
        const Tensor& compressed_kv  // (input_len, lora_rank + pe_head_dim)
    );

    void encode_compressed_cache(
        model::ModelContext& ctx,
        model::DynBatchContext* dyn_batch,
        Tensor q,              // (input_len, num_heads, nope_head_dim + pe_head_dim)
        Tensor compressed_kv,  // (input_len, lora_rank + pe_head_dim)
        Tensor attn_value_enc  // (input_len, num_heads * dim_head)
    );
    void search_compressed_cache(
        model::ModelContext& ctx,
        model::DynBatchContext* dyn_batch,
        Tensor q,              // (input_len, num_heads, nope_head_dim + pe_head_dim)
        Tensor compressed_kv,  // (input_len, lora_rank + pe_head_dim)
        Tensor attn_output  // (input_len, num_heads * dim_head)
    );
    void search_compressed_cache_naive(
        model::ModelContext& ctx,
        model::DynBatchContext* dyn_batch,
        Tensor q,              // (input_len, num_heads, nope_head_dim + pe_head_dim)
        Tensor compressed_kv,  // (input_len, lora_rank + pe_head_dim)
        Tensor attn_output  // (input_len, num_heads * dim_head)
    );
    void attn_by_gemm(
        model::ModelContext& ctx,
        model::DynBatchContext* dyn_batch,
        const Tensor& q_adj,          // (len_q, num_heads, cache_dim)
        const Tensor& compressed_kv,  // (len_buf, 1, cache_dim)
        const Tensor& mask,
        Tensor& attn_score,
        Tensor attn_result            // (len_q, num_heads, kv_lora_rank)
    );
    void attn_by_flash_mla(
        model::ModelContext& ctx,
        model::DynBatchContext* dyn_batch,
        int b,
        const Tensor& q_adj,          // (len_q, num_heads, cache_dim)
        const Tensor& compressed_kv,  // (len_buf, 1, cache_dim)
        const Tensor& mask,
        Tensor& attn_score,
        Tensor attn_result            // (len_q, num_heads, kv_lora_rank)
    );
    void attn_by_flash_mla_batch(
        model::ModelContext& ctx,
        model::DynBatchContext* dyn_batch,
        int begin,
        int end,
        const Tensor& q_adj,          // (len_q, num_heads, cache_dim)
        Tensor attn_result            // (len_q, num_heads, kv_lora_rank)
    );

    Tensor get_compressed_kv_v1(const core::Context& ctx, const Tensor& h, const Tensor& position) {
        BM_ASSERT(kv_a_proj_lora.get(), "");
        // down => 2D: (len_q, kv_lora_rank + qk_rope_head_dim)
        Tensor kv_lora = kv_a_proj_lora->forward(ctx, h); // (len_q, kv_lora_rank)
        kv_lora = kv_a_layernorm->forward(ctx, kv_lora);
        Tensor k_pe = k_a_proj_pe->forward(ctx, h); // (len_q, qk_rope_head_dim)
        k_pe = rotary_emb.rotate(ctx, position, k_pe);
        return functions::concat_tensor(ctx, kv_lora, k_pe);
    }
    // proj, ln, rotary
    Tensor get_compressed_kv_v2(const core::Context& ctx, const Tensor& h, const Tensor& position) {
        // down => 2D: (len_q, kv_lora_rank + qk_rope_head_dim)
        Tensor compressed_kv = kv_a_proj_with_mqa->forward(ctx, h);
        Tensor kv_lora = compressed_kv.virtual_slice(0, kv_lora_rank);
        Tensor k_pe = compressed_kv.virtual_slice(kv_lora_rank, pe_head_dim);
        kv_a_layernorm->inplace(ctx, kv_lora);
        rotary_emb.rotate_inplace(ctx, position, k_pe);
        return compressed_kv;
    }

    // Return q, kv
    std::pair<Tensor, Tensor> process_compressed_all_v1(
        const core::Context& ctx, const Tensor& h, const Tensor& position, bool full=false) {
        // Step 0: q
        Tensor q = forward_q(ctx, h, full); // (len_q, num_heads, qk_head_dim)
        {
            // q = q.view({q.size(0), q.size(1) / qk_head_dim, qk_head_dim});
            Tensor q_pe = q.virtual_slice(nope_head_dim, pe_head_dim);
            rotary_emb.rotate_inplace(ctx, position, q_pe);
        }

        // Step 1: k,v
        Tensor kv = get_compressed_kv_v2(ctx, h, position);
        return {q, kv};
    }

    Tensor project_qkv_a(model::ModelContext& ctx, const Tensor& h) {
        Tensor a = qkv_a_proj_with_pe->forward(ctx, h);
        if (ctx.dyn_batch()->input_len_before_dp > 0) {
            ctx.recordEvent("AllGatherQPVa", 2);
            a = all_gather(ctx, a, ctx.dyn_batch()->input_len_before_dp);
            ctx.dyn_batch()->input_len_before_dp = 0;
        }
        return a;
    }

    // Return q (num_token, num_heads, nope_head_dim + pe_head_dim), kv if up=true
    //        q_lora_norm (num_token, q_lora_rank), kv if up=false
    std::pair<Tensor, Tensor> process_compressed_all_v2(
        model::ModelContext& ctx, const Tensor& h, const Tensor& position, bool up=true, bool full=false) {
        if (!qkv_a_proj_with_pe) {
            BM_ASSERT(up, "Up should be true");
            return process_compressed_all_v1(ctx, h, position, full);
        }
        // step 0: project a for all
        Tensor a = project_qkv_a(ctx, h);
        // Note len(a) != len(h) if PROJ_A_DP is effect !!!
        size_t len_q = a.size(0);

        Tensor q_a = a.virtual_slice(0, q_lora_rank);
        Tensor kv_a = a.virtual_slice(q_lora_rank, kv_lora_rank);
        Tensor k_pe1 = a.virtual_slice(q_lora_rank + kv_lora_rank, pe_head_dim);

        // step 1: layer norm for all
        Tensor q_a_norm = ctx.tensor(q_a.shape(), q_a.dtype());
        Tensor compressed_kv = ctx.tensor({len_q, kv_lora_rank + pe_head_dim}, h.dtype());
        Tensor kv_a_norm = compressed_kv.virtual_slice(0, kv_lora_rank);
        Tensor k_pe = compressed_kv.virtual_slice(kv_lora_rank, pe_head_dim);
        LayerNorm::forward_2(ctx, q_a, kv_a, q_a_norm, kv_a_norm, q_a_layernorm.get(), kv_a_layernorm.get());
        rotary_emb.rotate(ctx, position, k_pe1, &k_pe);
        if (!up) {
            return {q_a_norm, compressed_kv};
        }

        // step 2: proj b(UP) for Q.
        Tensor q = (full ? q_b_proj_full : q_b_proj)->forward(ctx, q_a_norm);
        q = q.view({q.size(0), q.size(1) / qk_head_dim, qk_head_dim});

        Tensor q_pe = q.virtual_slice(nope_head_dim, pe_head_dim);
        rotary_emb.rotate_inplace(ctx, position, q_pe);

        return {q, compressed_kv};
    }

    // A bit faster than project_qkv_a(), because layer norm is in DP.
    std::pair<Tensor, Tensor> process_qkv_down_dp(
        model::ModelContext& ctx, const Tensor& h, const Tensor& position) {
        // project in DP.
        Tensor a = qkv_a_proj_with_pe->forward(ctx, h);
        // layer norm in DP.
        Tensor q_a = a.virtual_slice(0, q_lora_rank);
        Tensor kv_a = a.virtual_slice(q_lora_rank, kv_lora_rank);
        q_a_layernorm->inplace(ctx, q_a);
        kv_a_layernorm->inplace(ctx, kv_a);
        {
            ctx.recordEvent("AllGatherQPVa", 2);
            a = all_gather(ctx, a, ctx.dyn_batch()->input_len_before_dp);
        }
        // full
        q_a = a.virtual_slice(0, q_lora_rank);  // Attention! it is virtual!
//        q_a = functions::slice_last_dim(ctx, a, 0, q_lora_rank);
        Tensor compressed_kv = functions::slice_last_dim(ctx, a, q_lora_rank, kv_lora_rank + pe_head_dim);

        Tensor k_pe1 = functions::slice_last_dim(ctx, compressed_kv, kv_lora_rank, pe_head_dim);
        Tensor k_pe = compressed_kv.virtual_slice(kv_lora_rank, pe_head_dim, -1);
        rotary_emb.rotate(ctx, position, k_pe1, &k_pe);

        ctx.dyn_batch()->input_len_before_dp = 0;  // clear flag
        return {q_a, compressed_kv};
    }

    std::pair<Tensor, Tensor> process_qkv_down(
        model::ModelContext& ctx, const Tensor& h, const Tensor& position) {
        if (qkv_a_proj_with_pe) {
            // return q_lora_norm, kv(norm, rotated)
            if (ctx.dyn_batch()->input_len_before_dp > 0) {
                return process_qkv_down_dp(ctx, h, position);
            }
            return process_compressed_all_v2(ctx, h, position, false, false /*no use*/);
        } else {
            BM_ASSERT(q_lora_rank <= 0, "No qkv_a_proj_with_pe");
            auto kv = get_compressed_kv_v2(ctx, h, position);
            return {h, kv};
        }
    }

    std::pair<Tensor, Tensor>  process_q_up(
        const core::Context& ctx, const Tensor& qa_or_h, const Tensor& position, bool full, bool rotate_inp=false) {
        Tensor q;
        if (q_lora_rank <= 0) {
            q = (full ? q_proj_full : q_proj)->forward(ctx, qa_or_h);
        } else {
            q = (full ? q_b_proj_full : q_b_proj)->forward(ctx, qa_or_h);
        }
        q = q.view({q.size(0), q.size(1) / qk_head_dim, qk_head_dim});

        Tensor q_pe = q.virtual_slice(nope_head_dim, pe_head_dim, -1);
        if (rotate_inp) {
            rotary_emb.rotate_inplace(ctx, position, q_pe);
        } else {
            q_pe = rotary_emb.rotate(ctx, position, q_pe);
        }

        return {q, q_pe};
    }
};

core::Tensor Attention::impl::MLAImpl::forward_compressed_cache(
    model::ModelContext& ctx,
    const Tensor& hidden_q,  // (group_len_q, dim_model)
    const core::Tensor& position,
    core::Tensor* output
) {
    // Step 0: q
    auto [q_a, kv] = process_qkv_down(ctx, hidden_q, position);
    auto [q, q_pe] = process_q_up(ctx, q_a, position, false, true);

    model::DynBatchContext* dyn_batch = ctx.dyn_batch().get();
    BM_ASSERT_EQ(hidden_q.ndim(), 2, "hidden_q.ndim() != 2");
    size_t len_q = q_a.size(0);

    // Encode part
    bool has_encode = !dyn_batch->ev_batch.empty();
    size_t num_enc = dyn_batch->e_placement.numel();
    size_t num_s = dyn_batch->s_placement.numel();
    BM_ASSERT_EQ(num_enc + num_s, len_q, "dim mismatch");
    Tensor attn_val_g = ctx.tensor({len_q, num_heads * v_head_dim}, dtype);
    if (has_encode) {
        auto ev_name = str_cat("Encode[", ctx.is_BSHD() ? "flash=True," : "", "heads=", num_heads, "]");
        core::EventScope ev_encode(ctx, ev_name, 2);
        encode_compressed_cache(
            ctx,
            dyn_batch,
            q.slice_dim0(0, num_enc),
            kv.slice_dim0(0, num_enc),
            attn_val_g.slice_dim0(0, num_enc));
        if (ctx.debug() > 4)
            std::cout << "EncodeCompress: " << attn_val_g.slice_dim0(0, num_enc) << endl;
    }

    // Search part
    if (num_s > 0) {
        core::EventScope ev_search(ctx, "Search[num_s=" + std::to_string(num_s) + "]", 2);
        const Tensor q_search = q.slice_dim0_len(num_enc, num_s);
        const Tensor kv_search = kv.slice_dim0_len(num_enc, num_s);
        const Tensor attn_out_search = attn_val_g.slice_dim0_len(num_enc, num_s);
        search_compressed_cache(
            ctx,
            dyn_batch,
            q_search,
            kv_search,
            attn_out_search);
        if (ctx.debug() > 4)
            std::cout << "SearchCompress: " << attn_out_search << endl;
    }

    // (len_q, num_heads * v_head_dim) => (len_q, hidden_size)
    auto ret = o_proj->forward(ctx, attn_val_g); // (group_len_q, dim_model)
    return ret;
}

void Attention::impl::MLAImpl::encode_compressed_cache(
    model::ModelContext& ctx,
    model::DynBatchContext* dyn_batch,
    Tensor q,              // (input_len, num_heads, nope_head_dim + pe_head_dim)
    Tensor compressed_kv,  // (input_len, lora_rank + pe_head_dim)
    Tensor attn_value_enc  // (input_len, num_heads * dim_head)
) {
    std::shared_ptr<model::RagBufferContext> rag_buffer = ctx.rag_buffer();
    cudaStream_t stream = ctx.current_stream()->ptr;

    BM_ASSERT_EQ(dyn_batch->ev_batch.size(), 1, "Expect single batch");
    BM_ASSERT(rag_buffer, "");

    size_t num_enc = dyn_batch->e_placement.numel();
    int b = dyn_batch->ev_batch[0];
    size_t input_len = dyn_batch->ev_input_len[0];
    size_t full_input_len = dyn_batch->full_input_len[0]; // = input_len + cache_len
    size_t len_buf = dyn_batch->ev_len_buf[0];
    BM_ASSERT_EQ(input_len, num_enc, "Expect single batch");

    // Step 0: Copy to buffer
    size_t cache_dim = kv_lora_rank + pe_head_dim;
    Tensor buf = rag_buffer->buf_k(b, ctx.current_layer()); // (len_buf, 1, cache_dim)
    Tensor placement = dyn_batch->e_placement;
    // fake num_head = 1
    Tensor compressed_kv_3d = compressed_kv.view({input_len, 1, cache_dim});
    copy_to_buffer(1, input_len, len_buf, cache_dim, &placement, compressed_kv_3d, buf, stream, ctx.is_BSHD());

    BM_ASSERT(ctx.is_BSHD(), "FlashAttention only.");
    size_t attn_dim = qk_head_dim; // 128 + 64 = 192

    compressed_kv = buf.slice_dim0(0, full_input_len).view({full_input_len, cache_dim});
    auto [k_nope, k_pe, v] = forward_kv_cache(ctx, compressed_kv);

    Tensor k = functions::concat_broadcast_b(ctx, k_nope, k_pe); // (full_input_len, num_kv_heads, attn_dim)

    // Pad v: v_head_dim => attn_dim
    BM_ASSERT_LE(v_head_dim + 1, attn_dim, "");
    v = v.view({{full_input_len, num_kv_heads, v_head_dim}});
    Tensor v_pad = ctx.tensor({full_input_len, num_kv_heads, attn_dim}, v.dtype());
    functions::copy_last_dim(stream, v, v_pad, 0, -1, true);

    // insert 1 as batch for FA
    Tensor q1 = q.view({1, input_len, num_heads, attn_dim});
    Tensor k1 = k.slice_dim0_len(0, full_input_len)
        .view({1, full_input_len, num_kv_heads, attn_dim});
    Tensor v1 = v_pad.slice_dim0_len(0, full_input_len)
        .view({1, full_input_len, num_kv_heads, attn_dim});

    Tensor ret =
        flash_decoding(ctx, q1, k1, v1, nullptr, nullptr, nullptr, 0, 0, true, -1, -1, attn_scale);
    // slice v_head_dim
    ret = ret.view({input_len, num_heads, attn_dim});
    attn_value_enc = attn_value_enc.view({num_enc, num_heads, v_head_dim});
    functions::copy_last_dim(stream, ret, attn_value_enc, 0, v_head_dim);
}

void Attention::impl::MLAImpl::search_compressed_cache_naive(
    model::ModelContext& ctx,
    model::DynBatchContext* dyn_batch,
    Tensor q,              // (g_len, num_heads, nope_head_dim + pe_head_dim)
    Tensor compressed_kv,  // (g_len, lora_rank + pe_head_dim)
    Tensor attn_output     // (g_len, num_heads * dim_head)
) {
    const Tensor* s_placement = &dyn_batch->s_placement; // (batch, len_q)
    BM_ASSERT_EQ(2, s_placement->ndim(), "placement is not 2d");
    size_t batch = s_placement->size(0);
    size_t len_q = s_placement->size(1);
    size_t batch_len_q = batch * len_q;

    // Step 0: Copy to buffer
    std::shared_ptr<model::RagBufferContext> rag_buffer = ctx.rag_buffer();
    size_t cache_dim = kv_lora_rank + pe_head_dim;
    Tensor buf_addr = rag_buffer->buf_k_addr(ctx, ctx.current_layer()); // batch => (len_buf, 1, cache_dim)
    const Tensor* s_len_buf = ctx.identity(&dyn_batch->s_len_buf, "s_len_buf");
    const Tensor* s_mask = ctx.identity(&dyn_batch->s_mask, "s_mask");
    // fake num_head = 1 for copy_to_rag_buffer
    Tensor compressed_kv_4d = compressed_kv.view({batch, len_q, 1, cache_dim});
    ctx.recordEvent("copy_to_rag_buffer", 4);
    copy_to_rag_buffer(ctx, compressed_kv_4d, *s_placement, *s_len_buf, buf_addr);
    ctx.recordEvent("End->copy_to_rag_buffer", 4);

    q = q.view({batch_len_q, num_heads, qk_head_dim});
    auto [q_nope, q_pe] = split(ctx, q, nope_head_dim, pe_head_dim);
    q_nope = q_nope.view({batch_len_q, num_heads, nope_head_dim});
    q_pe = q_pe.view({batch, len_q, num_heads, pe_head_dim});

    static int fuse_search = utils::get_int_env("FUSE_ATTN_SEARCH", 0);
    if (fuse_search) {
    }

    functions::Gemm gemm_transB(ctx, dtype, false, true);
    functions::Gemm gemm_score_v(ctx, dtype, false, false);

    size_t attn_dim = qk_head_dim; // 128 + 64 = 192
    size_t dim_head = qk_head_dim; // 128 + 64 = 192

    auto h_q_t = transpose_2_1(ctx, q.view({ batch, len_q, num_kv_heads, dim_head }))
        .view({ batch, num_kv_heads, len_q, dim_head });
    auto h_q_chunk = h_q_t.chunk();
    vector<Tensor> attn_scores(batch);
    vector<Tensor> attn_results(batch);
    auto attn_value_chunk = attn_output.view({batch, len_q, num_heads, v_head_dim}).chunk();
    for (size_t i = 0; i < batch; ++i) {
        Tensor compressed_kv = ctx.rag_buffer()->buf_k(i, ctx.current_layer()); // (len_buf, 1, cache_dim)
        size_t len_buf = compressed_kv.size(ctx.is_BSHD() ? -3 : -2);

        compressed_kv = compressed_kv.view({len_buf, kv_lora_rank + pe_head_dim});
        auto [k_nope, k_pe, v] = forward_kv_cache(ctx, compressed_kv);
        Tensor k = functions::concat_broadcast_b(ctx, k_nope, k_pe); // (len_q, num_kv_heads, attn_dim)
        k = transpose_2_1(ctx, k);

        // Pad v: v_head_dim => attn_dim
        BM_ASSERT_LE(v_head_dim + 1, attn_dim, "");
        v = v.view({{len_buf, num_kv_heads, v_head_dim}});
        Tensor v_pad = ctx.tensor({len_buf, num_kv_heads, attn_dim}, v.dtype());
        auto stream = ctx.current_stream()->ptr;
        functions::copy_last_dim(stream, v, v_pad, 0, -1, true);
        v = transpose_2_1(ctx, v);

        // Q * K
        if (ctx.debug() > 4) {
            q_nope = transpose_2_1(ctx, q_nope);
            k_nope = transpose_2_1(ctx, k_nope);
            Tensor attn_w1 = gemm_transB.forward(ctx, q_nope, k_nope);
            std::cout << "#### attn_w1: " << attn_w1 << endl;
        }
        ctx.recordEvent("Q * K", event_level);
        attn_scores[i] = gemm_transB.forward(ctx, h_q_chunk[i], k);

        // attn_softmax in-place update attn_score
        Tensor attn_score_q = attn_scores[i].view({ num_heads, len_q, len_buf });
        Tensor mask = dyn_batch->search_mask(ctx, i, len_q);
        ctx.recordEvent("attn_softmax", event_level);
        attn_softmax(ctx, attn_scale, attn_score_q, mask, Tensor());
        if (ctx.debug() > 4) {
            std::cout << "attn_scores[i]: " << attn_scores[i] << endl;
        }
        // Score * V
        ctx.recordEvent("Score*V", event_level);
        Tensor tmp_q1 = attn_value_chunk[i].view({num_kv_heads, len_q, v_head_dim});
        attn_results[i] = gemm_score_v(
            ctx,
            attn_scores[i],
            v,
            len_q > 1 ? nullptr : &tmp_q1);

        if (len_q > 1) {
            // (batch, num_heads, len_q, dim_head) => (batch, len_q, num_heads * dim_head)
            ctx.recordEvent("transposeAV", event_level);
            transpose_2_1(ctx, attn_results[i].view({num_heads, len_q, dim_head}), &attn_value_chunk[i]);
        }
    }
}

void Attention::impl::MLAImpl::copy_to_compressed_cache(
    model::ModelContext& ctx,
    model::DynBatchContext* dyn_batch,
    const Tensor& compressed_kv  // (input_len, lora_rank + pe_head_dim)
) {
    std::shared_ptr<model::RagBufferContext> rag_buffer = ctx.rag_buffer();
    size_t cache_dim = kv_lora_rank + pe_head_dim;
    Tensor buf_addr = rag_buffer->buf_k_addr(ctx, ctx.current_layer()); // batch => (len_buf, 1, cache_dim)
    const Tensor* s_len_buf = ctx.identity(&dyn_batch->s_len_buf, "s_len_buf");
    // const Tensor* s_mask = ctx.identity(&dyn_batch->s_mask, "s_mask");
    const Tensor* s_placement = &dyn_batch->s_placement; // (batch, len_q)

    BM_ASSERT_EQ(2, s_placement->ndim(), "placement is not 2d");
    size_t batch = s_placement->size(0);
    size_t len_q = s_placement->size(1);
    // fake num_head = 1 for copy_to_rag_buffer
    Tensor compressed_kv_4d = compressed_kv.view({batch, len_q, 1, cache_dim});
    ctx.recordEvent("copy_to_rag_buffer", 4);
    copy_to_rag_buffer(ctx, compressed_kv_4d, *s_placement, *s_len_buf, buf_addr);
    ctx.recordEvent("End->copy_to_rag_buffer", 4);
}

void Attention::impl::MLAImpl::attn_by_gemm(
    model::ModelContext& ctx,
    model::DynBatchContext* dyn_batch,
    const Tensor& q_adj,          // (len_q, num_heads, cache_dim)
    const Tensor& compressed_kv,  // (len_buf, 1, cache_dim)
    const Tensor& mask,
    Tensor& attn_score,
    Tensor attn_result            // (len_q, num_heads, kv_lora_rank)
) {
    BM_ASSERT_EQ(q_adj.ndim(), 3, "q_adj is not 3d");
    BM_ASSERT_EQ(compressed_kv.ndim(), 3, "compressed_kv is not 3d");
    BM_ASSERT(compressed_kv.size(0) == 1 || compressed_kv.size(1) == 1, "num_head_k is not 1");
    size_t num_heads = q_adj.size(1);
    size_t cache_dim = kv_lora_rank + pe_head_dim; // 576
    size_t len_q = q_adj.size(0);
    Tensor kv = compressed_kv.squeeze();
    size_t len_buf = kv.size(0);

    ctx.recordEvent("Q_Adj*Cache", event_level + 1);
    // (num_heads, len_q, cache_dim）* (len_buf, cache_dim) => (num_heads, len_q, len_buf)
    attn_score = gemm_transB.forward(ctx, q_adj, kv);

    // attn_softmax in-place update attn_score
    Tensor attn_score_q = attn_score.view({num_heads, len_q, len_buf});
    ctx.recordEvent("attn_softmax", event_level + 1);
    attn_softmax(ctx, attn_scale, attn_score_q, mask, Tensor());
    if (ctx.debug() > 4) {
        std::cout << "#### ADJ attn_scores: " << attn_score << endl;
    }

    // Score * V
    ctx.recordEvent("Score*Cache", event_level + 1);
    Tensor kv_lora = kv.virtual_slice(0, kv_lora_rank);
    Tensor v_ext = gemm( // 2D gemm
        ctx,
        attn_score, // (num_heads, len_q, len_buf)
        kv_lora, // (len_buf, kv_lora_rank)
        &attn_result); // (len_q, num_heads, kv_lora_rank)
    // v_ext = transpose_2_1(ctx, v_ext); // (len_q, H, kv_lora_rank+)
    // functions::copy_last_dim(stream, v_ext, attn_results[i], 0, kv_lora_rank);
}

// Note: FlashMLA doesn't support mask
void Attention::impl::MLAImpl::attn_by_flash_mla(
    model::ModelContext& ctx,
    model::DynBatchContext* dyn_batch,
    int b,
    const Tensor& q_adj,          // (len_q, num_heads, cache_dim)
    const Tensor& compressed_kv,  // (len_buf, 1, cache_dim)
    const Tensor& mask,
    Tensor& attn_score,
    Tensor attn_result            // (len_q, num_heads, kv_lora_rank)
) {
    BM_ASSERT(ctx.get_compute_capability() >= 90, "FlashMLA need cc 90+ GPUs.");
    BM_ASSERT_EQ(q_adj.ndim(), 3, "q_adj is not 3d");
    BM_ASSERT_EQ(compressed_kv.ndim(), 3, "compressed_kv is not 3d");
    BM_ASSERT(compressed_kv.size(0) == 1 || compressed_kv.size(1) == 1, "num_head_k is not 1");
    size_t num_heads = q_adj.size(1);
    size_t cache_dim = kv_lora_rank + pe_head_dim; // 576
    size_t len_q = q_adj.size(0);
    Tensor kv = compressed_kv.squeeze(); // num_blocks x page_block_size x num_heads_k x head_size
    size_t len_buf = kv.size(0);
    BM_ASSERT_EQ(len_buf % 64U, 0, "buf length can't divide page size");
    size_t num_blocks = len_buf / 64U;

    int pos = dyn_batch->sv_position.at(b * len_q);
    Tensor seqlens_k = ctx.tensor_of(std::vector<int>({pos + 1}));
    Tensor tile_scheduler_metadata;
    Tensor num_splits;
    std::tie(tile_scheduler_metadata, num_splits) =
        ds::get_mla_metadata(ctx, seqlens_k, len_q * num_heads);

    Tensor q = q_adj.view({1, len_q, num_heads, cache_dim});
    kv = kv.view({len_buf / 64, 64L, 1L, cache_dim});
    int head_size_v = 512;
    std::vector<int> table(num_blocks);
    std::iota(table.begin(), table.end(), 0);
    Tensor block_table = ctx.tensor_of(table, {1, num_blocks});
    core::Tensor mla_out = attn_result.view({1, len_q, num_heads, head_size_v});
    ds::mha_fwd_kvcache_mla(
        ctx, q, kv, head_size_v, seqlens_k, block_table, attn_scale, false, tile_scheduler_metadata, num_splits, mla_out);
}

static std::tuple<Tensor, Tensor, Tensor> fake_paged_buffer(
    model::ModelContext& ctx,
    model::DynBatchContext* dyn_batch,
    int begin,
    int end,
    int len_q) {
    size_t batch = end - begin;
    auto allocator = ctx.get_cache_allocator();
    char* base_ptr = allocator->get_base_ptr();

    vector<int> len_buffers;
    int max_num_blocks = 0;
    for (int b = begin; b < end; ++b) {
        int len_buf = dyn_batch->sv_position.at(b * len_q) + 1;
        len_buffers.push_back(len_buf);
        max_num_blocks = std::max(max_num_blocks, ceil_div(len_buf, 64));
    }
    // FIXME
    size_t num_layers = ctx.rag_buffer()->buf_k_[0]->get_num_layers();
    vector<int> table(num_layers * batch * max_num_blocks);
    const size_t PAGE_BYTES = 64 * 576 * 2;
    for (int i = 0; i < num_layers; ++i) {
        for (int b = begin; b < end; ++b) {
            Tensor buf = ctx.rag_buffer()->buf_k(b, i);
            size_t offset = buf.data<char>() - base_ptr;
            BM_ASSERT_EQ(offset % PAGE_BYTES, 0, "offset can't mode page size");
            int page0 = offset / PAGE_BYTES;
            int len_buf = dyn_batch->sv_position.at(b * len_q);
            int num_blocks = ceil_div(len_buf, 64);
            for (int n = 0; n < num_blocks; ++n) {
                table[i * batch * max_num_blocks + (b - begin) * max_num_blocks + n] = page0 + n;
            }
        }
    }

    Tensor seqlens_k = ctx.tensor_of(len_buffers); // (batch)
    Tensor block_table = ctx.tensor_of(table, {num_layers, batch, max_num_blocks});
    auto dtype = ctx.rag_buffer()->buf_k_[0]->get_layer(0).dtype();
    Tensor paged_kv = Tensor::from_external(
        {1U, 64L, 1L, 576}, dtype, base_ptr, PAGE_BYTES, ctx.active_device_idx());
    // For ALL layers
    return {seqlens_k, paged_kv, block_table};
}

// Note: FlashMLA doesn't support mask
void Attention::impl::MLAImpl::attn_by_flash_mla_batch(
    model::ModelContext& ctx,
    model::DynBatchContext* dyn_batch,
    int begin,
    int end,
    const Tensor& q_adj,          // (batch, len_q, num_heads, cache_dim)
    Tensor attn_result            // (batch, len_q, num_heads, kv_lora_rank)
) {
    BM_ASSERT(ctx.get_compute_capability() >= 90, "FlashMLA need cc 90+ GPUs.");
    static int pre_alloc = utils::get_int_env("PRE_ALLOC_ALL_TOKEN", 1);
    BM_ASSERT(pre_alloc > 0, "flash_mla need set PRE_ALLOC_ALL_TOKEN");

    BM_ASSERT_EQ(q_adj.ndim(), 4, "q_adj is not 4d");
    BM_ASSERT_EQ((end - begin), q_adj.size(0), "batch mismatch");
    size_t batch = end - begin;
    size_t len_q = q_adj.size(1);
    size_t num_heads = q_adj.size(2);
    size_t cache_dim = kv_lora_rank + pe_head_dim; // 576
    size_t head_size_v = 512;

    if (ctx.current_layer() == 0) {
        BM_ASSERT(dyn_batch->paged_kv.empty(), "dyn_batch is not clean.");
        std::tie(dyn_batch->seqlens_k, dyn_batch->paged_kv, dyn_batch->block_table) =
            fake_paged_buffer(ctx, dyn_batch, begin, end, q_adj.size(1));
        // metadata and num_splits will reused for all layers
        std::tie(dyn_batch->tile_scheduler_metadata, dyn_batch->num_splits) =
            ds::get_mla_metadata(ctx, dyn_batch->seqlens_k, len_q * num_heads);
    }
    Tensor block_table = dyn_batch->block_table.index_dim0(ctx.current_layer());
    core::Tensor mla_out = attn_result.view({batch, len_q, num_heads, head_size_v});
    Tensor q = q_adj;
    ds::mha_fwd_kvcache_mla(
        ctx,
        q,
        dyn_batch->paged_kv,
        head_size_v,
        dyn_batch->seqlens_k,
        block_table,
        attn_scale,
        false,
        dyn_batch->tile_scheduler_metadata,
        dyn_batch->num_splits,
        mla_out);
}
void Attention::impl::MLAImpl::search_compressed_cache(
    model::ModelContext& ctx,
    model::DynBatchContext* dyn_batch,
    Tensor q,              // (g_len, num_heads, nope_head_dim + pe_head_dim)
    Tensor compressed_kv,  // (g_len, lora_rank + pe_head_dim)
    Tensor attn_output     // (g_len, num_heads * dim_head)
) {
    const Tensor* s_placement = &dyn_batch->s_placement; // (batch, len_q)
    BM_ASSERT_EQ(2, s_placement->ndim(), "placement is not 2d");
    size_t batch = s_placement->size(0);
    size_t len_q = s_placement->size(1);
    size_t batch_len_q = batch * len_q;

    // Step 0: Copy to buffer
    copy_to_compressed_cache(ctx, dyn_batch, compressed_kv);
    std::shared_ptr<model::RagBufferContext> rag_buffer = ctx.rag_buffer();
    size_t cache_dim = kv_lora_rank + pe_head_dim;
    Tensor buf_addr = rag_buffer->buf_k_addr(ctx, ctx.current_layer()); // batch => (len_buf, 1, cache_dim)
    const Tensor* s_len_buf = ctx.identity(&dyn_batch->s_len_buf, "s_len_buf");
    const Tensor* s_mask = ctx.identity(&dyn_batch->s_mask, "s_mask");

    q = q.view({batch_len_q, num_heads, qk_head_dim});
    auto [q_nope, q_pe] = split(ctx, q, nope_head_dim, pe_head_dim);

    // q_adj
    // TODO: gemm w/o split by adjust stride
    q_nope = q_nope.view({batch_len_q, num_heads, nope_head_dim});
    Tensor q_nope1 = transpose_2_1(ctx, q_nope);  //（num_heads, batch_len_q, nope_head_dim）
    BM_ASSERT_EQ(k_proj->get_weight().size(-1), kv_lora_rank, "");
    auto w1 = k_proj->get_weight().view({num_heads, nope_head_dim, kv_lora_rank});
    // w1 = functions::Transpose(ctx).forward(ctx, w1);
    Tensor q_adj_nope;
    {
        core::EventScope ev(ctx, "Gemm(q_adj_nope)(H*192=>H*512)", event_level);
        q_adj_nope = gemm.forward(ctx, q_nope1, w1);  //（num_heads, batch_len_q, kv_lora_rank）
    }
    if (ctx.debug() > 4) {
        std::cout << "#### ADJ q_adj_nope: " << q_adj_nope << endl;
    }
    Tensor q_adj_nope1 = q_adj_nope;

    q_adj_nope = transpose_2_1(ctx, q_adj_nope);
    q_adj_nope = q_adj_nope.view({batch, len_q, num_heads, kv_lora_rank});

    q_pe = q_pe.view({batch, len_q, num_heads, pe_head_dim});
    Tensor q_adj = functions::concat_tensor(ctx, q_adj_nope, q_pe);

    static int fuse_search = utils::get_int_env("FUSE_ATTN_SEARCH", 0);
    Tensor v_attn = ctx.tensor({batch, len_q, num_heads, kv_lora_rank}, q_adj.dtype());
    if (fuse_search) {
        core::EventScope ev(ctx, "attention_qkv_rag_buffer", event_level);
        multi_query_attention_rag_buffer(
            ctx,
            q_adj, // need 4-D
            *s_len_buf,
            buf_addr,
            buf_addr,
            *s_mask,
            attn_scale,
            dyn_batch->get_max_len_buf(),
            v_attn,
            num_heads,
            -1);
    } else {
        // q_adj = transpose_2_1(ctx, q_adj); // (batch, num_heads, len_q, kv_lora_rank + pe_dim)
        auto q_adj_chunk = q_adj.chunk();
        vector<Tensor> attn_scores(batch);
        vector<Tensor> attn_results = v_attn.chunk();
        auto stream = ctx.current_stream()->ptr;
        for (size_t i = 0; i < batch; ++i) {
            Tensor kv = ctx.rag_buffer()->buf_k(i, ctx.current_layer()); // (len_buf, 1, cache_dim)
            Tensor mask = dyn_batch->search_mask(ctx, i, len_q);
            attn_by_gemm(ctx, dyn_batch, q_adj_chunk[i], kv, mask, attn_scores[i], attn_results[i]);
        }
    }

    v_attn = v_attn.view({batch_len_q, num_heads, kv_lora_rank}); // back to 3-D
    v_attn = transpose_2_1(ctx, v_attn); //（num_heads, batch_len_q, kv_lora_rank）
    auto w2 = v_proj->get_weight().view({num_heads, v_head_dim, kv_lora_rank});
    core::EventScope ev_v(ctx, "Gemm(v:kv_lora=>H*v_dim)(H*512=>H*128)", event_level);
    if (batch_len_q == 1) {
        Tensor o = attn_output.view({num_heads, 1, v_head_dim});
        gemm_transB.forward(ctx, v_attn, w2, &o);
    } else {
        Tensor tmp = gemm_transB.forward(ctx, v_attn, w2); //（num_heads, batch_len_q, v_head_dim）
        Tensor o = attn_output.view({batch_len_q, num_heads, v_head_dim});
        transpose_2_1(ctx, tmp, &o);
    }
}

// Note: KV cache is still replicated in current implementation.
core::Tensor Attention::impl::MLAImpl::forward_compressed_dp_v1(
    model::ModelContext& ctx,
    const Tensor& hidden_q,  // (batch * len_q, dim_model)
    const core::Tensor& position,
    core::Tensor* output
) {
    Tensor batch_ret = output ? * output : ctx.tensor(hidden_q.shape(), dtype);
    functions::zeros_(ctx, batch_ret);
    Tensor attn_val_batch = ctx.tensor({hidden_q.size(0), num_heads * ctx.world_size(), kv_lora_rank}, dtype);
    Tensor attn_out_batch;
    if (!data_parallel_out) {
        attn_out_batch = ctx.tensor({hidden_q.size(0), num_heads * ctx.world_size() * v_head_dim}, dtype);
        functions::zeros_(ctx, attn_out_batch);
    }

    model::DynBatchContext* dyn_batch = ctx.dyn_batch().get();
    BM_ASSERT_EQ(dyn_batch->e_placement.numel(), 0, "");
    const Tensor* s_placement = &dyn_batch->s_placement; // (batch, len_q)
    BM_ASSERT_EQ(2, s_placement->ndim(), "placement is not 2d");
    int batch = s_placement->size(0);
    size_t len_q = s_placement->size(1);
    BM_ASSERT_EQ(len_q, 1, "Support 1 only");
    size_t cache_dim = kv_lora_rank + pe_head_dim;

    Tensor q_a_part, ret_part;
    int start, end;
    if (is_cache_dp()) {
//        // V2
//        std::tie(start, end) = get_dp_start_end(batch, ctx.rank(), ctx.world_size());
//        // Step 0: q_a, kv.
//        Tensor h_part = hidden_q.slice_dim0(start, end);
//        Tensor position_part = position.slice_dim0(start, end);
//        auto [q_part, kv_part] = process_compressed_all_v2(ctx, h_part, position_part);
//        // Step 1: Copy to buffer
//        copy_to_compressed_cache_dp(ctx, dyn_batch, kv_part, start, end);
    } else {
        // V1: kv cache is replicated. so we need process batch kv
        // Step 0: q_a, kv.
        Tensor batch_q_a, batch_kv;
        if (qkv_a_proj_with_pe) {
            std::tie(batch_q_a, batch_kv) =
                process_compressed_all_v2(ctx, hidden_q, position, false);
        } else {
            BM_ASSERT(q_lora_rank <= 0, "");
            batch_q_a = hidden_q;
            batch_kv = get_compressed_kv_v2(ctx, hidden_q, position);
        }

        // Step 1: Copy to buffer
        copy_to_compressed_cache(ctx, dyn_batch, batch_kv);

        int part_batch = ceil_div(batch, ctx.world_size());
        static int debug_part_size = utils::get_int_env("ATTN_DP_PART_SIZE", 0);
        part_batch = debug_part_size > part_batch ? debug_part_size : part_batch;
        // Tensor ret = ctx.tensor({size_t(part_batch), batch_inputs.size(-1)}, dtype);
        start = ctx.rank() * part_batch;
        end = std::min(start + part_batch, batch);
        if (start < end) {
            q_a_part = batch_q_a.slice_dim0(start, end);
            ret_part = batch_ret.slice_dim0(start, end);
        }
    }
    if (start < end) {
        size_t cur_batch = end - start;
        // Step 2: q up
        Tensor position_part = position.slice_dim0(start, end);
        Tensor q_part = forward_q(ctx, q_a_part, false, true);
        const size_t H = q_part.size(1);
        Tensor q_pe = q_part.virtual_slice(nope_head_dim, pe_head_dim);
        q_pe = rotary_emb.rotate(ctx, position_part, q_pe);

        // Step 3: q_adj
        ctx.recordEvent("Gemm(q_adj_nope)(H*192=>H*512)", event_level);
        Tensor q_nope = q_part.virtual_slice(0, nope_head_dim, -1);
        q_nope = q_nope.view_uncontinuous({cur_batch, H, nope_head_dim});
        q_nope = q_nope.virtual_transpose(0, 1);
        auto w1 = k_proj_full->get_weight().view({H, nope_head_dim, kv_lora_rank});
        Tensor q_adj_nope = gemm.batch_3d(ctx, q_nope, w1);  //（H, cur_batch, kv_lora_rank）
        q_adj_nope = transpose_2_1(ctx, q_adj_nope); // (cur_batch, H, kv_lora_rank)
        Tensor q_adj = functions::concat_tensor(ctx, q_adj_nope, q_pe); // (cur_batch, H, kv_lora_rank+)

        // Step4: cal v_attn
        Tensor v_attn = ctx.tensor({cur_batch, H, kv_lora_rank}, q_adj.dtype());
        static int use_flash_mla = utils::get_int_env("USE_FLASH_MLA", 0);
        if (use_flash_mla == 2) {
            Tensor q = q_adj.view({cur_batch, len_q, H, kv_lora_rank + pe_head_dim});
            attn_by_flash_mla_batch(ctx, dyn_batch, start, end, q, v_attn);
        } else {
            vector<Tensor> q_adj_chunk = q_adj.chunk(); // cur_batch X (H, kv_lora_rank+)
            vector<Tensor> attn_results = v_attn.chunk(); // cur_batch X (H, kv_lora_rank)
            for (int j = 0; j < cur_batch; ++j) {
                int i = start + j;
                Tensor q_adj_i = q_adj_chunk[j].view({len_q, H, kv_lora_rank + pe_head_dim});
                Tensor kv_i = ctx.rag_buffer()->buf_k(i, ctx.current_layer()); // (len_buf, 1, cache_dim)
                Tensor mask = dyn_batch->search_mask(ctx, i, len_q);
                Tensor attn_score;
                Tensor attn_out = attn_results[j].view({len_q, H, kv_lora_rank});
                if (use_flash_mla) {
                    attn_by_flash_mla(ctx, dyn_batch, i, q_adj_i, kv_i, mask, attn_score, attn_out);
                } else {
                    attn_by_gemm(ctx, dyn_batch, q_adj_i, kv_i, mask, attn_score, attn_out);
                }
            }
        }

        Tensor attn_output = ctx.tensor({cur_batch * len_q, H * v_head_dim}, dtype);
        if (!data_parallel_out) {
            attn_output = attn_out_batch.slice_dim0(start * len_q, end * len_q);
        }
        v_attn = v_attn.view({cur_batch * len_q, H, kv_lora_rank}); // back to 3-D
        v_attn = v_attn.virtual_transpose(0, 1); //（H, batch_len_q, kv_lora_rank）
        auto w2 = v_proj_full->get_weight().view({H, v_head_dim, kv_lora_rank});
        {
            // batch=H, (cur_batch * len_q, kv_lora_rank) @ (v_head_dim, kv_lora_rank) => (cur_batch * len_q, v_head_dim)
            core::EventScope ev_v(ctx, "Gemm(v:kv_lora=>H*v_dim)(H*512=>H*128)", event_level);
            Tensor o = attn_output.view({cur_batch * len_q, H, v_head_dim});
            o = o.virtual_transpose(0, 1);  // (H, cur_batch * len_q, v_head_dim)
            gemm_transB.batch_3d(ctx, v_attn, w2, &o);
        }
//        Tensor ret1 = ret.slice_dim0(0, end - start);
        if (data_parallel_out) {
            Tensor ret1 = o_proj_full->forward(ctx, attn_output);
            ctx.copy2(ret1, &ret_part);
        }
    } else {
        // no job
    }
    if (!data_parallel_out) {
        // Re-partition to call o_proj TP
        // TODO: all gather should be faster
        attn_out_batch = ctx.reduce_sum(attn_out_batch, dtype);
        int dim_part = num_heads * v_head_dim;
        Tensor attn_out_repart = functions::slice_last_dim(ctx, attn_out_batch, ctx.rank() * dim_part, dim_part);
        return o_proj->forward(ctx, attn_out_repart);
    }
    // TODO: use all gather
    return batch_ret;
}

core::Tensor Attention::impl::MLAImpl::dynamic_batch_forward(
    model::ModelContext& ctx,
    const Tensor& hidden_states_org,  // (group_len_q, dim_model)
    const core::Tensor& position,
    core::Tensor* output
) {
    core::EventScope ev(ctx, "Attention(DynBatch)", event_level - 1);
    model::DynBatchContext* dyn_batch = ctx.dyn_batch().get();
    BM_ASSERT_EQ(hidden_states_org.ndim(), 2, "");
    Tensor hidden_states = functions::typecast(ctx, hidden_states_org, dtype);
    size_t num_enc = dyn_batch->e_placement.numel();
    size_t num_s = dyn_batch->s_placement.numel();
    size_t len_q = dyn_batch->s_placement.size(1);
    // TODO: check len_q > 1
    static int data_parallel_min_batch = utils::get_int_env("ATTN_DATA_PARALLEL_MIN_BATCH", 1);
    if (data_parallel && ctx.latent_cache() &&
        num_enc == 0 && num_s >= data_parallel_min_batch &&
        len_q == 1) {
        return forward_compressed_dp_v1(ctx, hidden_states, position, output);
    }
    if (ctx.latent_cache()) {
        return forward_compressed_cache(ctx, hidden_states, position, output);
    }

    len_q = hidden_states.numel() / hidden_states.size(-1);  // group_len_q
    cudaStream_t stream = ctx.current_stream()->ptr;
    // Step 1: q
    Tensor q = forward_q(ctx, hidden_states); // (len_q, num_heads, qk_head_dim)
    if (false) {
        auto[q_nope, q_pe] = split(ctx, q, nope_head_dim, pe_head_dim);
        q_pe = rotary_emb.rotate(ctx, position, q_pe);
        q = functions::concat_tensor(ctx, q_nope, q_pe);
    } else {
        q = q.view({len_q, num_heads, qk_head_dim});
        Tensor q_pe = q.virtual_slice(nope_head_dim, pe_head_dim);
        rotary_emb.rotate_inplace(ctx, position, q_pe);
    }
    // Step 1: k,v
    // down => 2D: (len_q, kv_lora_rank + qk_rope_head_dim)
    Tensor compressed_kv = kv_a_proj_with_mqa->forward(ctx, hidden_states);
    auto [kv_a, k_pe] = split(ctx, compressed_kv, kv_lora_rank, pe_head_dim);
    // up => 3D: {len_q, num_heads, (qk_nope_head_dim + v_head_dim)}
    Tensor kv = kv_b_proj->forward(ctx, kv_a_layernorm->forward(ctx, kv_a))
        .view({len_q, num_kv_heads, (nope_head_dim + v_head_dim)});
    auto [k_nope, v] = split(ctx, kv, nope_head_dim, v_head_dim);
    if (ctx.debug() > 4 && len_q > 1)
        std::cout << "k_nope Normal: " << k_nope.slice_dim0(0, len_q-1) << endl;

    bool print = ctx.current_layer() == 0;
    k_pe = rotary_emb.rotate(ctx, position, k_pe);
    Tensor k = functions::concat_broadcast_b(ctx, k_nope, k_pe);

    // Pad v: v_head_dim => qk_head_dim
    BM_ASSERT_LE(v_head_dim + 1, qk_head_dim, "");
    Tensor v1 = v.view({{len_q, num_kv_heads, v_head_dim}});
    Tensor v_pad = ctx.tensor({len_q, num_kv_heads, qk_head_dim}, v1.dtype());
    functions::copy_last_dim(stream, v1, v_pad, 0, -1, true);

    // Encode part
    bool has_encode = !dyn_batch->ev_batch.empty();
    BM_ASSERT_EQ(num_enc + num_s, len_q, "dim mismatch");
    Tensor attn_val_g = ctx.tensor({len_q, num_heads * v_head_dim}, dtype);
    if (has_encode) {
        attn_encode_group(
            ctx,
            q.slice_dim0(0, num_enc),
            k.slice_dim0(0, num_enc),
            v_pad.slice_dim0(0, num_enc),
            attn_val_g.slice_dim0(0, num_enc));
        if (ctx.debug() > 4)
            std::cout << "GroupNormal: " << attn_val_g.slice_dim0(0, num_enc) << endl;
    }
    if (num_s == 0) {
        return o_proj->forward(ctx, attn_val_g); // (group_len_q, dim_model)
    }

    // search part
    ctx.recordEvent("Start>Search,len=" + std::to_string(num_s), event_level);
    const Tensor* s_placement = ctx.identity(&dyn_batch->s_placement, "s_placement"); // (batch, len_q)
    BM_ASSERT_EQ(2, s_placement->ndim(), "placement is not 2d");
    size_t batch = s_placement->size(0);
    len_q = s_placement->size(1);

    Tensor h_q = q.slice_dim0_len(num_enc, num_s).view({batch, len_q, num_heads, qk_head_dim});
    Tensor h_k = k.slice_dim0_len(num_enc, num_s).view({batch, len_q, num_kv_heads, qk_head_dim});
    // Tensor h_v = v.slice_dim0_len(num_enc, num_s).view({batch, len_q, num_kv_heads, v_head_dim});
    Tensor h_v = v_pad.slice_dim0_len(num_enc, num_s).view({batch, len_q, num_kv_heads, qk_head_dim});

    Tensor attn_value_search = attn_val_g.slice_dim0_len(num_enc, num_s)
        .view({batch, len_q, num_heads, v_head_dim});

    attn_search_rag(ctx, h_q, h_k, h_v, *s_placement, attn_value_search);
    ctx.recordEvent("End>Search,len=" + std::to_string(num_s), event_level);
    if (ctx.debug() > 4)
        std::cout << "GroupNormal: " << attn_value_search << endl;

    auto ret = o_proj->forward(ctx, attn_val_g); // (group_len_q, dim_model)
    return ret;
}

void Attention::impl::MLAImpl::attn_encode_group(
    model::ModelContext& ctx,
    Tensor h_q_enc,
    Tensor h_k_enc,
    Tensor h_v_enc,
    Tensor attn_value_enc  // (num_enc, num_heads * dim_head)
) {
    auto ev_name = str_cat("Encode[", ctx.is_BSHD() ? "flash=True," : "", "heads=", num_heads, "]");
    core::EventScope ev_encode1(ctx, ev_name, 3);
    model::DynBatchContext* dyn_batch = ctx.dyn_batch().get();
    cudaStream_t stream = ctx.current_stream()->ptr;

    BM_ASSERT_EQ(dyn_batch->ev_batch.size(), 1, "Expect single batch");
    BM_ASSERT(ctx.rag_buffer(), "");

    size_t num_enc = dyn_batch->e_placement.numel();
    int b = dyn_batch->ev_batch[0];
    size_t input_len = dyn_batch->ev_input_len[0];
    size_t full_input_len = dyn_batch->full_input_len[0]; // = input_len + cache_len
    size_t len_buf = dyn_batch->ev_len_buf[0];
    BM_ASSERT_EQ(input_len, num_enc, "Expect single batch");

    size_t dim_head = qk_head_dim;
    Tensor h_q = h_q_enc.view({ num_enc, num_heads, dim_head });
    Tensor h_k = h_k_enc.view({ num_enc, num_kv_heads, dim_head });
    Tensor h_v = h_v_enc.view({ num_enc, num_kv_heads, dim_head });

    Tensor key_buf = ctx.rag_buffer()->buf_k(b, ctx.current_layer());
    Tensor val_buf = ctx.rag_buffer()->buf_v(b, ctx.current_layer());
    Tensor placement = *ctx.identity(&dyn_batch->e_placement, "e_placement");
    copy_to_buffer(num_kv_heads, input_len, len_buf, dim_head, &placement, h_k, key_buf, stream, ctx.is_BSHD());
    copy_to_buffer(num_kv_heads, input_len, len_buf, dim_head, &placement, h_v, val_buf, stream, ctx.is_BSHD());

    if (ctx.is_BSHD()) {
        // insert 1 as batch for FA
        Tensor q1 = h_q.view({1, input_len, num_heads, dim_head});
        Tensor k1 = key_buf.slice_dim0_len(0, full_input_len)
            .view({1, full_input_len, num_kv_heads, dim_head});
        Tensor v1 = val_buf.slice_dim0_len(0, full_input_len)
            .view({1, full_input_len, num_kv_heads, dim_head});
        Tensor ret =
            flash_decoding(ctx, q1, k1, v1, nullptr, nullptr, nullptr, 0, 0, true, -1, -1, attn_scale);
        // slice v_head_dim
        ret = ret.view({input_len, num_heads, dim_head});
        attn_value_enc = attn_value_enc.view({num_enc, num_heads, v_head_dim});
        functions::copy_last_dim(stream, ret, attn_value_enc, 0, v_head_dim);
    } else {
        functions::Gemm gemm_transB(ctx, dtype, false, true);
        // TODO: compute type
        functions::Gemm gemm_score_v(ctx, dtype, false, false);
        // Q * K
        ctx.recordEvent("Q*K", 3);
        h_q = transpose_2_1(ctx, h_q);
        Tensor attn_score = gemm_transB.forward(
            ctx,
            h_q,     // ColMajor: (num_kv_heads, dim_head, input_len)
            key_buf  // ColMajor: (num_kv_heads, len_buf, dim_head)T
        );           // (num_kv_heads, input_len, len_buf)

        // attn_softmax in-place update attn_score
        ctx.recordEvent("attn_softmax", 3);
        Tensor attn_score_q = attn_score.view({num_heads, input_len, len_buf});
        Tensor mask = dyn_batch->encode_mask(ctx, 0);
        attn_softmax(ctx, attn_scale, attn_score_q, mask, Tensor());
        // Score * V
        ctx.recordEvent("Score*V", 3);
        Tensor attn_res = gemm_score_v.forward(
            ctx,
            attn_score, // ColMajor: (num_kv_heads, len_buf, len_q)
            val_buf,    // ColMajor: (num_kv_heads, dim_head, len_buf)
            nullptr   // (num_kv_heads, len_q, dim_head)
        );
        ctx.recordEvent("transposeAV", 3);
        Tensor ret = transpose_2_1(ctx, attn_res.view({ num_heads, input_len, dim_head }));
        // slice v_head_dim
        attn_value_enc = attn_value_enc.view({num_enc, num_heads, v_head_dim});
        functions::copy_last_dim(stream, ret, attn_value_enc, 0, v_head_dim);
    }
}

void Attention::impl::MLAImpl::attn_search_rag(
    model::ModelContext& ctx,
    const Tensor& h_q_s,  // （batch, len_q, num_heads, dim_head）
    const Tensor& h_k_s,  // （batch, len_q, num_kv_heads, dim_head）
    const Tensor& h_v_s,  // （batch, len_q, num_kv_heads, dim_head）
    const Tensor& placement_s,
    Tensor& attn_value_search  // (batch, len_q, num_heads, dim_head)
) {
    model::DynBatchContext* dyn_batch = ctx.dyn_batch().get();
    auto rag_buffer = ctx.rag_buffer().get();

    Tensor buf_k_addr = rag_buffer->buf_k_addr(ctx, ctx.current_layer()); // (batch) => (num_kv_heads, len_buf, dim_head)
    Tensor buf_v_addr = rag_buffer->buf_v_addr(ctx, ctx.current_layer()); // (batch) => (num_kv_heads, len_buf, dim_head)
    const Tensor* s_len_buf = ctx.identity(&dyn_batch->s_len_buf, "s_len_buf");
    const Tensor* s_mask = ctx.identity(&dyn_batch->s_mask, "s_mask");

    ctx.recordEvent("copy_to_rag_buffer", 4);
    copy_to_rag_buffer2(ctx, placement_s, *s_len_buf, h_k_s, h_v_s, &buf_k_addr, &buf_v_addr);
    ctx.recordEvent("End->copy_to_rag_buffer", 4);

    static int fuse_search = utils::get_int_env("FUSE_ATTN_SEARCH", 0);
    if (ctx.is_BSHD() || fuse_search) {
        core::EventScope ev(ctx, "attention_qkv_rag_buffer", 3);
        attention_qkv_rag_buffer(
            ctx, h_q_s, *s_len_buf,
            buf_k_addr,
            buf_v_addr,
            *s_mask,
            dyn_batch->get_position_bias_addresses(ctx),
            attn_scale,
            dyn_batch->get_max_len_buf(),
            attn_value_search);
    }

    BM_ASSERT(!ctx.is_BSHD(), "Not supported");
    functions::Gemm gemm_transB(ctx, dtype, false, true);
    functions::Gemm gemm_score_v(ctx, dtype, false, false);
    const Tensor* s_placement = &dyn_batch->s_placement; // (batch, len_q)
    BM_ASSERT_EQ(2, s_placement->ndim(), "placement is not 2d");
    size_t batch = s_placement->size(0);
    size_t len_q = s_placement->size(1);

    size_t dim_head = qk_head_dim;
    auto h_q_t = transpose_2_1(ctx, h_q_s.view({ batch, len_q, num_kv_heads, dim_head }))
        .view({ batch, num_kv_heads, len_q, dim_head });
    auto h_q_chunk = h_q_t.chunk();
    Tensor pad_results = ctx.tensor({batch, len_q, num_heads, dim_head}, dtype);
    auto attn_value_chunk = pad_results.chunk();
    vector<Tensor> attn_scores(batch);
    vector<Tensor> attn_results(batch);
    for (size_t i = 0; i < batch; ++i) {
        Tensor key_buf = ctx.rag_buffer()->buf_k(i, ctx.current_layer());
        Tensor val_buf = ctx.rag_buffer()->buf_v(i, ctx.current_layer());
        size_t len_buf = key_buf.size(-2);
        // Q * K
        ctx.recordEvent("Q * K", event_level);
        attn_scores[i] = gemm_transB.forward(ctx, h_q_chunk[i], key_buf);

        // attn_softmax in-place update attn_score
        Tensor attn_score_q = attn_scores[i].view({ num_heads, len_q, len_buf });
        Tensor mask = dyn_batch->search_mask(ctx, i, len_q);
        ctx.recordEvent("attn_softmax", event_level);
        attn_softmax(ctx, attn_scale, attn_score_q, mask, Tensor());

        // Score * V
        ctx.recordEvent("Score*V", event_level);
        Tensor tmp_q1 = attn_value_chunk[i].view({num_kv_heads, len_q, dim_head});
        attn_results[i] = gemm_score_v(
            ctx,
            attn_scores[i],
            val_buf,
            len_q > 1 ? nullptr : &tmp_q1);

        if (len_q > 1) {
            // (batch, num_heads, len_q, dim_head) => (batch, len_q, num_heads * dim_head)
            ctx.recordEvent("transposeAV", event_level);
            transpose_2_1(ctx, attn_results[i].view({num_heads, len_q, dim_head}), &attn_value_chunk[i]);
        }
    }
    auto stream = ctx.current_stream()->ptr;
    functions::copy_last_dim(stream, pad_results, attn_value_search, 0, v_head_dim);
}

Attention::impl* Attention::impl::create_mla_impl(
    const core::Context& ctx, const model::ModelConfig& cfg, model::QuantConfig quant) {
    return new MLAImpl(ctx, cfg, quant);
}
}
