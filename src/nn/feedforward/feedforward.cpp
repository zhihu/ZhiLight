#include "nn/feedforward/feedforward.h"
#include "nn/linear/activation_kernel.h"
#include "nn/linear/gemm_grouped.h"
#include "nn/linear/linear.h"
#include "model/model_context.h"
#include "nn/quant/int8/quant_kernel.h"
#include "nn/quant/gptq/gptq.h"
#include "utils/env.h"
#include <bmengine/c10d/c10d.h>
#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>
#include <bmengine/logger/std_log_op.hpp>
#include <algorithm>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>

namespace nn {

using namespace bmengine;
using bmengine::core::DataType;
using bmengine::core::DistLayout;
using bmengine::core::Tensor;
using model::ModelContext;
using std::tuple;
using std::vector;

std::mutex log_mutex;

class FeedForward::impl {
public:
    class NormalImpl;
    class Int8Impl;
    class MOEImpl;
    class FusedMOE;
    class GPTQMOE;
    class FP8BlockMOE;

    std::string name;

    virtual ~impl() = default;
    virtual Tensor forward(const core::Context& ctx, const Tensor& input, bool quant_back) = 0;

    enum class WeightType {
        kIn,
        kGated,
        kOut,
    };
};

// clang-format off
class FeedForward::impl::NormalImpl : public FeedForward::impl {
public:
    int dim_model;
    int dim_ff;
    std::string act_fn_type;
    bool scale_weights;
    bool weight_transposed;
    core::DataType dtype;
    model::QuantConfig quant;
    bool parallel;
    Linear w_in, w_gated, w_out;
    functions::BinaryElementwiseOp gated_op;
    std::unique_ptr<Linear> w_fuse_up;

    NormalImpl(
        const core::Context& ctx,
        model::ModelConfig cfg,
        model::QuantConfig quant,
        bool parallel = false)
        : dim_model(cfg.dim_model),
          dim_ff(cfg.dim_ff),
          act_fn_type(cfg.activate_fn),
          scale_weights(cfg.scale_weights),
          weight_transposed(cfg.weight_transposed),
          dtype(cfg.dtype),
          quant(quant),
          parallel(parallel),
          // clang-format off
          w_in(ctx, dim_model, dim_ff, "", quant, scale_weights, weight_transposed, parallel, core::DistLayout::COLUMNAR, dtype),
          w_gated(ctx, dim_model, dim_ff, "", quant, scale_weights, weight_transposed, parallel, core::DistLayout::COLUMNAR, dtype),
          w_out(ctx, dim_ff, dim_model, "", quant, scale_weights, weight_transposed, parallel, core::DistLayout::ROW, dtype),
          // clang-format on
          gated_op(ctx, functions::BinaryElementwiseOp::Mul) {
        name = "FeedForward";
    }

    virtual ~NormalImpl() = default;

    int get_dim_ff(const core::Context& ctx) {
        return parallel ? dim_ff / ctx.world_size() : dim_ff;
    }

    Tensor gptq_fused_up(const core::Context& ctx, const Tensor& input) {
        // Fuse 'w_in' and 'w_gated' and activation into a kernel.
        // i.e. activate(input X 'w_in') * (input * 'w_gated')
        auto[qw1, qz1, scales1, sym1] = w_in.get_gptq_weights();
        auto[qw2, qz2, scales2, sym2] = w_gated.get_gptq_weights();
        return nn::gptq::gemm_fuse_gate_in(
            ctx, input, qw1, qz1, scales1, Tensor(), qw2, qz2, scales2, Tensor(), sym1);
    }

    void try_fuse_up_weights(const core::Context& ctx) {
        auto a = Linear::fuse(ctx, w_in, w_gated);
        w_fuse_up = std::unique_ptr<Linear>(a);
    }

    virtual Tensor forward(const core::Context& ctx, const Tensor& input, bool quant_back) {
        Tensor up;
        static int fuse_v2_thres = utils::get_int_env("FUSE_V2_THRES", 8);
        if (w_in.support_fuse_gptq_gate_in(input)
                && input.ndim() == 2 && input.size(0) <= fuse_v2_thres
                && act_fn_type == "silu"
                && input.size(0) <= 2
                && input.nbytes() < ctx.get_max_shared_memory()) {
            up = gptq_fused_up(ctx, input);
        } else if (w_fuse_up) {
            Tensor fuse_ret = w_fuse_up->forward(ctx, input);
            up = gate_fuse(ctx, fuse_ret, act_fn_type);
        } else {
            auto w_0 = w_in.forward(ctx, input);
            {
                auto w_1 = w_gated.forward(ctx, input);
                ctx.recordEvent("gate_activate_multiply", 3);
                // activate(w_0) * w_1
                nn::gate_mul_inplace(ctx, w_0, w_1, act_fn_type);
            }
            up = w_0;
        }
        auto ret = w_out.forward(ctx, up, parallel || !quant.fuse_block() || quant_back);
        return ret;
    }

    const Linear& get_weight(WeightType weight_type) {
        switch (weight_type) {
            case WeightType::kIn: return w_in;
            case WeightType::kGated: return w_gated;
            default: return w_out;
        }
    }
};

class FeedForward::impl::Int8Impl : public FeedForward::impl::NormalImpl {
public:
    Int8Impl(
        const core::Context& ctx,
        model::ModelConfig cfg,
        model::QuantConfig quant_config,
        bool parallel)
        : NormalImpl(ctx, cfg, quant_config, parallel) {
        if (parallel && ctx.high_precision() >= 2)
            w_out.set_output_type(core::DataType::kFloat);
        name = "FeedForward(Int8)";
    }
    virtual ~Int8Impl() = default;

    Tensor forward(const core::Context& ctx, const Tensor& input, bool quant_back) override {
        Tensor w_0 = w_in.forward(ctx, input, false); // w_in is gate, maybe name is wrong
        Tensor w_1 = w_gated.forward(ctx, input, false);
        BM_ASSERT_EQ(w_0.dtype(), w_1.dtype(), "");

        Tensor ret;
        if (w_0.dtype() == dtype) {
            nn::gate_mul_inplace(ctx, w_0, w_1, act_fn_type);
            ret = w_0;
        } else {
            BM_ASSERT_EQ(w_0.dtype(), core::DataType::kInt32, "");
            ret = int8_op::quant_back_act_mul(
                ctx,
                w_0,
                w_0.quant_scale.get(),
                w_in.get_weight_scale(),
                w_1,
                w_1.quant_scale.get(),
                w_gated.get_weight_scale(),
                act_fn_type);
            BM_ASSERT_EQ(ret.dtype(), dtype, "dtype mismatch");
        }

        Tensor output = w_out.forward(ctx, ret, parallel || !quant.fuse_block() || quant_back);
        return output;
    }
};

class FeedForward::impl::MOEImpl : public FeedForward::impl {
public:
    int dim_model, dim_ff;
    int num_experts;
    int num_experts_may_share;
    int num_local_experts;
    int top_k;
    int top_k_may_share;
    bool norm_topk_prob;
    float routed_scaling_factor;
    int topk_group;
    int n_group;
    std::string scoring_func;
    std::string topk_method;
    core::DataType dtype;
    bool parallel;
    bool exp_parallel;
    bool dyn_shared { false };
    int n_shared_experts { 0 };
    const int world_size;
    const int local_rank;

    Linear router;
    Tensor e_score_correction_bias;

    std::vector<NormalImpl*> experts;
    std::shared_ptr<NormalImpl> shared_expert;
    std::shared_ptr<Linear> shared_expert_gate;
    const int MAX_SEQ_LEN = 320000;
    int* pin_buf;

    MOEImpl(
        const core::Context& ctx,
        model::ModelConfig cfg,
        model::QuantConfig quant_config,
        bool parallel,
        bool dyn_shared = false)
        : dim_model(cfg.dim_model),
          dim_ff(cfg.dim_ff),
          num_experts(cfg.moe_num_experts),
          num_experts_may_share(cfg.moe_num_experts),
          num_local_experts(cfg.moe_num_experts),
          top_k(cfg.moe_top_k),
          top_k_may_share(cfg.moe_top_k),
          norm_topk_prob(cfg.norm_topk_prob),
          routed_scaling_factor(cfg.routed_scaling_factor),
          topk_group(cfg.moe_topk_group),
          n_group(cfg.moe_n_group),
          scoring_func(cfg.router_scoring_func),
          topk_method(cfg.moe_topk_method),
          dtype(cfg.dtype),
          parallel(parallel),
          dyn_shared(dyn_shared),
          world_size(ctx.world_size()),
          local_rank(ctx.rank()),
          router(ctx, dim_model, num_experts, "", 0, false, false, false, DistLayout::ROW, dtype) {
        if (cfg.moe_intermediate_size > 0) {
            cfg.dim_ff = cfg.moe_intermediate_size;
        }
        exp_parallel = utils::get_int_env("MOE_EXP_PARALLEL", 0);
        if (exp_parallel) {
            BM_ASSERT_EQ(num_experts % ctx.world_size(), 0, "");
            num_local_experts /= ctx.world_size();
        }
        std::string share_type = dyn_shared ? "dyn" : "TP";
        name = logger::str_cat("MOE(EP=", exp_parallel, ",Shared=", share_type, ")");
        bool tp = parallel && !exp_parallel;
        for (int i = 0; i < num_local_experts; ++i) {
            experts.push_back(new impl::NormalImpl(ctx, cfg, quant_config, tp));
        }

        if (cfg.shared_expert_intermediate_size > 0 && ctx.current_layer() >= cfg.first_k_dense_replace) {
            cfg.dim_ff = cfg.shared_expert_intermediate_size;
            tp = parallel && !dyn_shared;
            shared_expert = std::make_shared<impl::NormalImpl>(ctx, cfg, quant_config, tp);
            shared_expert_gate = std::make_shared<Linear>(ctx, dim_model, 1, "", 0, false, false, false, DistLayout::ROW, dtype);
            if (dyn_shared) {
                BM_ASSERT(exp_parallel, "");
                BM_ASSERT_EQ(cfg.shared_expert_intermediate_size % cfg.moe_intermediate_size, 0, "");
                n_shared_experts = cfg.shared_expert_intermediate_size / cfg.moe_intermediate_size;
                num_experts_may_share = num_experts + n_shared_experts * world_size;
                top_k_may_share = top_k + n_shared_experts;
                for (int i = 0; i < n_shared_experts; ++i) {
                    experts.push_back(new impl::NormalImpl(ctx, cfg, quant_config, false));
                }
            }
        }
        if (topk_method == "noaux_tc") {
            e_score_correction_bias = ctx.parameter({size_t(num_experts)}, dtype);
        }
        router.set_output_type(core::DataType::kFloat);
        BM_CUDART_ASSERT(cudaHostAlloc(&pin_buf, MAX_SEQ_LEN * sizeof(int) * 4, 0));
    }

    ~MOEImpl() override {
        for (auto p: experts) {
            delete p;
        }
        cudaFreeHost(pin_buf);
    };

    bool is_local_expert(int global_id) const {
        return !exp_parallel || (global_id % world_size) == local_rank;
    }
    int local_expert_id(int global_id) const {
        return exp_parallel ? (global_id / world_size) : global_id;
    }
    int global_expert_id(int id) const {
        return exp_parallel ? (id * world_size + local_rank) : id;
    }

    // filter LOCAL experts' tokens; concat to flat buffer
    // I.E. Result is concat of:
    //    [token list] of exp1
    //    [token list] of exp3
    //    ...
    size_t filter_concat_exp_tokens(
        const vector<vector<int>>& exp_tokens, // Ragged: (num_experts, seq_len in expert)
        int* flat_buf_ptr,  // out
        vector<int>& active_exp,
        vector<int>& num_tokens,
        vector<int>& acc_offset
    ) const {
        size_t local_tokens_k = 0;  // total numbers of tokens after filter.
        acc_offset.push_back(0);
        BM_ASSERT_EQ(exp_tokens.size(), num_experts_may_share, "exp_tokens.size() mismatch");
        for (int exp = 0; exp < num_experts_may_share; ++exp) {
            if (is_local_expert(exp) && !exp_tokens[exp].empty()) {  // filter LOCAL experts.
                active_exp.push_back(exp);
                auto& tokens = exp_tokens[exp];
                std::copy(tokens.begin(), tokens.end(), flat_buf_ptr + local_tokens_k);
                num_tokens.push_back(tokens.size());
                acc_offset.push_back(local_tokens_k);
                local_tokens_k += tokens.size();
            }
        }
        return local_tokens_k;
    }

    void fill_reverse_index(
        const vector<int>& token_experts,      // (num_tokens, top_k)
        int* reverse_idx
    ) const {
        vector<int> index_in_exp(num_experts_may_share, 0);
        for (int i = 0; i < token_experts.size(); ++i) {
            int exp = token_experts[i];
            reverse_idx[i] = index_in_exp[exp]++;
        }
    }

    // Return:
    //   group_idx_d: LOCAL
    //   reverse_idx_d: GLOBAL
    tuple<Tensor, Tensor> get_idx_tensor(
        const core::Context& ctx,
        const vector<int>& token_experts,      // (num_tokens, top_k)
        const vector<vector<int>>& exp_tokens, // Ragged: (num_experts, seq_len in expert)
        size_t num_input_tokens
    ) const {
        // Note:
        //   num_experts: is num_experts_may_share if (dyn_shared == true).
        //   exp mean expert id.
        //   token means token index in the input.
        const int total_tokens_k = int(num_input_tokens) * top_k_may_share;

        // concat ragged to flat buffer
        int* flat_buf_ptr = pin_buf + MAX_SEQ_LEN * 2;
        vector<int> active_exp, exp_num_tokens, acc_offset;
        size_t local_tokens_k = filter_concat_exp_tokens(
            exp_tokens, flat_buf_ptr, active_exp, exp_num_tokens, acc_offset);

        if (exp_parallel) {
            if (local_tokens_k == 0) {
                return {Tensor(), Tensor()};
            }
        } else {
            BM_ASSERT_EQ(local_tokens_k, total_tokens_k, "");
        }

        // build reverse index
        int* reverse_idx = flat_buf_ptr + local_tokens_k;
        fill_reverse_index(token_experts, reverse_idx);

        Tensor tensor2 = ctx.tensor({local_tokens_k + total_tokens_k }, core::DataType::kInt32);
        tensor2.from_buffer(flat_buf_ptr, ctx.current_cuda_stream());

        // for reorder/permute input for experts
        Tensor group_idx_d = tensor2.slice_dim0_len(0, local_tokens_k);
        // for reorder experts output
        Tensor reverse_idx_d = tensor2.slice_dim0_len(local_tokens_k, total_tokens_k);

        return {group_idx_d, reverse_idx_d};
    }

    tuple<Tensor, Tensor> permute_input(
        const core::Context& ctx,
        const Tensor& h,
        const vector<int>& routing_experts,         // (num_experts, top_k)
        const vector<vector<int>>& group_token_idx, // Ragged: (num_experts, seq_len in expert)
        size_t total_seq_len) {
        if (total_seq_len > 1) {
            ctx.recordEvent("get_idx_tensor", 3);
            auto [idx_d, rev_idx_d] = // idx_d is Local!
                get_idx_tensor(ctx, routing_experts, group_token_idx, total_seq_len);

            Tensor h_reorder;
            if (idx_d.numel() > 0) {
                ctx.recordEvent("h_reorder", 3);
                h_reorder = functions::index_select(ctx, h, 0, idx_d);
            }
            // h_reorder: (seq_len * top_k) if NOT exp_parallel
            // rev_idx_d: (seq_len * top_k) always
            return {h_reorder, rev_idx_d};
        } else {
            // total_seq_len == 1, all expert's input is original input, no need permutation.
            return {h, Tensor()};
        }
    }

    // Return
    //   exp_ids    (num_token, top_k_may_share)
    //   exp_weight (num_token, top_k_may_share)
    std::tuple<Tensor, Tensor> route(const core::Context& ctx, const Tensor& input, bool with_shared=false) {
        // ctx.recordEvent("router_logits", 3);
        Tensor logit = router.forward(ctx, input); // (total_seq_len, n_experts)
        ctx.recordEvent("top_k_softmax", 2);
        Tensor exp_ids, exp_weights, worker_load;
        size_t top_k_ext = with_shared ? top_k_may_share : top_k;
        if (with_shared) {
            worker_load = ctx.tensor({size_t(ctx.world_size())}, core::DataType::kInt32);
            functions::zeros_(ctx, worker_load);
        }
        if (ctx.rank() == 0 || true) {
            if (topk_group > 1) {
                Tensor score = ctx.tensor(logit.shape(), logit.dtype());
//                ctx.recordEvent("softmax", 2);
//                functions::softmax(ctx, logit, score);
                // std::cout << "rooting score: " << score << endl;
                std::tie(exp_weights, exp_ids) = group_topk_softmax(
                    ctx,
                    logit,
                    e_score_correction_bias,
                    worker_load,
                    n_group, topk_group, top_k, top_k_ext, norm_topk_prob, routed_scaling_factor, scoring_func);
                if (ctx.is_layer(300)) {
//                Tensor ids = exp_ids.slice_dim0(0, 1).view({size_t(top_k_may_share)});
//                Tensor w = functions::index_select(ctx, score.slice_dim0(0, 1), -1, ids);
//                std::cout << "expert_weights(slice) " << w << endl;
                    std::cout << "logit " << logit << endl;
                    std::cout << "expert_weights " << exp_weights << endl;
                    std::cout << "exp_ids " << exp_ids << endl;
                }
            } else {
                std::tie(exp_weights, exp_ids) = top_k_softmax(
                    ctx, logit, worker_load, top_k, top_k_ext, norm_topk_prob, routed_scaling_factor, scoring_func);
            }
            if (with_shared) {
//                std::cout << "expert_ids " << exp_ids << endl;
//                std::cout << "expert_weights " << exp_weights << endl;
//                std::cout << "worker_load " << worker_load << endl;
                ctx.recordEvent("route_shared_lb", 2);
                route_shared_lb(ctx, exp_ids, exp_weights, worker_load, top_k, num_local_experts);
//                std::cout << "# expert_ids " << exp_ids << endl;
//                std::cout << "# expert_weights " << exp_weights << endl;
//                std::cout << "# worker_load " << worker_load << endl;
//                exp_ids = functions::slice_last_dim(ctx, exp_ids, 0, top_k);
//                exp_weights = functions::slice_last_dim(ctx, exp_weights, 0, top_k);
            }
        } else {
            exp_ids = ctx.tensor({input.size(0), top_k_ext}, core::DataType::kInt32);
            exp_weights = ctx.tensor({input.size(0), top_k_ext}, core::DataType::kFloat);
        }
        if (ctx.world_size() > 1) {
            ctx.recordEvent("broadcast_top_k_result", 2);
            // Broadcast to make sure all shards have the ids/weights
            c10d::NCCLBroadcast(ctx, exp_weights, exp_weights, 0);
            c10d::NCCLBroadcast(ctx, exp_ids, exp_ids, 0);
        }
        // (total_seq_len, top_k)
        return {exp_ids, exp_weights};
    }

    Tensor with_share(const core::Context& ctx, const Tensor& input, const Tensor& ret) {
        if (!dyn_shared && shared_expert) {
            core::EventScope ev(ctx, "shared_expert_gate", 2);
            Tensor shared_ret = shared_expert->forward(ctx, input, true);
            if (ret.numel() == 0) return shared_ret;
            functions::BinaryElementwiseOp add_op(ctx, functions::BinaryElementwiseOp::Add);
            return add_op.forward(ctx, ret, shared_ret);
        }
        return ret;
    }

    tuple<vector<int>, vector<size_t>> filter_active_experts(vector<vector<int>>& group_token_idx) {
        vector<int> expert_ids;
        vector<size_t> expert_token_num;
        for (int i = 0; i < num_experts_may_share; ++i) {
            if (group_token_idx[i].empty()) continue;
            if (!is_local_expert(i)) {
                continue; // Skip NOT local experts
            }
            int local_exp = local_expert_id(i);
            expert_ids.push_back(local_exp);
            expert_token_num.push_back(group_token_idx[i].size());
        }
        return {expert_ids, expert_token_num};
    }

    vector<Tensor> slice_dim0(const Tensor& h, vector<size_t>& lengths, bool copy) {
        vector<Tensor> results;
        results.reserve(lengths.size());
        size_t offset = 0;
        for (auto len : lengths) {
            if (!copy) {
                Tensor slice = h.slice_dim0_len(offset, len);
                results.push_back(std::move(slice));
            } else {
                results.push_back(h);
            }
            offset += len;
        }
        if (!copy)
            BM_ASSERT_EQ(offset, h.size(0), "sum of lengths mismatch h.size(0)");
        return std::move(results);
    }

    vector<Tensor> get_active_weights(
        const core::Context& ctx, const vector<int>& active_experts, WeightType weight_type) {
        vector<Tensor> weights;
        weights.reserve(active_experts.size());
        for (int i : active_experts) {
            Tensor w = experts[i]->get_weight(weight_type).get_dequant_weight(ctx);
            weights.push_back(std::move(w));
        }
        return std::move(weights);
    }


    vector<vector<int>> dispatch_token(
        const core::Context& ctx,
        const Tensor& input,
        vector<int>& routing_experts, // (seq_len, top_k)
        int seq_len) {
        vector<vector<int>> group_token_idx(num_experts_may_share);
        const int seq_len_top_k = seq_len * top_k_may_share;
        BM_ASSERT_LE(seq_len_top_k, routing_experts.size(), "seq_len_top_k out of range");
        bool out_of_range = false;
        for (int i = 0; i < seq_len_top_k; ++i) {
            int exp_id = routing_experts[i];
            BM_ASSERT_LT(-1, exp_id, "exp_id is negative");
            if (bm_unlikely(exp_id >= num_experts_may_share)) {
                out_of_range = true;
            } else {
                group_token_idx[exp_id].push_back(i / top_k_may_share);
            }
        }
        if (out_of_range) {
            Tensor logit = router.forward(ctx, input);
            std::cout << "input: " << input << endl;
            std::cout << "logit: " << logit << endl;
            BM_ASSERT(false, "out_of_range");
        }
        return std::move(group_token_idx);
    }

    Tensor forward_shared_or_zero(const core::Context& ctx, const Tensor& input) {
        BM_ASSERT(exp_parallel, "");
        if (shared_expert) {
            return shared_expert->forward(ctx, input, true);
        } else {
            Tensor out = ctx.tensor({input.size(0), size_t(dim_model)}, dtype);
            functions::zeros_(ctx, out);
            return out;
        }
    }

    Tensor forward(const core::Context& ctx, const Tensor& input, bool quant_back) override {
        int event_level = 3;
        size_t total_seq_len = input.size(0);
        const Tensor& h = input;

        // (num_tokens, top_k_may_share)
        auto [routing_experts_t, routing_weights_t] = route(ctx, h, dyn_shared);
        vector<int> routing_experts(total_seq_len * top_k_may_share);
        ctx.recordEvent("exp_ids_to_buffer", event_level);
        BM_ASSERT_EQ(routing_experts.size(), routing_experts_t.numel(), "");
        routing_experts_t.to_buffer(routing_experts.data(), ctx.current_stream()->ptr);

        // num_experts_may_share of [routed tokens to this expert] (could be empty)
        vector<vector<int>> all_expert_tokens = dispatch_token(ctx, h,routing_experts, total_seq_len);
        // permute input for LOCAL experts
        auto [h_reorder, rev_idx_d] = permute_input(ctx, h, routing_experts, all_expert_tokens, total_seq_len);

        auto [local_experts, expert_token_num] = filter_active_experts(all_expert_tokens);
        size_t sum_token_num = std::accumulate(expert_token_num.begin(), expert_token_num.end(), 0UL);
        if (local_experts.empty()) {
            // in expert parallel mode, and not any local experts hits, return immediately
            return forward_shared_or_zero(ctx, input);
        }

        vector<Tensor> expert_inputs = slice_dim0(h_reorder, expert_token_num, total_seq_len == 1);
        // Note: sparse MOE may consume too much memory
        vector<Tensor> expert_ret(num_experts_may_share); // GLOBAL (because routing_weights_t is global)

        {
            int d = ctx.debug();
            ctx.enable_debug(d - 1);
            // loop over all the expert towers
            for (size_t j = 0; j < local_experts.size(); ++j) {
                int local_exp = local_experts[j];
                int global_exp = global_expert_id(local_exp);

                ctx.recordEvent("experts" + std::to_string(global_exp), event_level);
                expert_ret[global_exp] = experts[local_exp]->forward(ctx, expert_inputs[j], true);
                if (ctx.is_layer(300)) {
                    std::cout << "j=" << j << ", global_exp=" << global_exp << endl;
                    std::cout << "ret: " << expert_ret[global_exp] << endl;
                }
            }
            ctx.enable_debug(d);
        }
        h_reorder = Tensor(); // free memory

        ctx.recordEvent("sum_experts", event_level);
        Tensor ret = sum_experts(
            ctx,
            expert_ret,
            Tensor(),
            routing_experts_t,
            rev_idx_d,
            routing_weights_t,
            exp_parallel,
            ctx.world_size(),
            ctx.rank());

//        if (false && total_seq_len > 1) {
//            std::lock_guard<std::mutex> lock (log_mutex);
//            auto weights = routing_weights_t.slice_dim0(0, 1).to_vector<float>();
//            std::cout << "exp_ids: " << routing_experts << endl;
//            std::cout << "exp_weights: " << weights << endl;
//            vector<float> local_w;
//            vector<float> local_out;
//            for (int k = 0; k < top_k_may_share; ++k) {
//                int exp = routing_experts[k];
//                std::cout << "#### k=" << k << ", exp=" << exp << ", is_local=" << is_local_expert(exp) << endl;
//                if (is_local_expert(exp)) {
//                    const Tensor& out = expert_ret[exp].slice_dim0(0, 1);
//                    std::cout << "expert_out: " << out << endl;
//                    local_w.push_back(weights[k]);
//                    local_out.push_back(functions::typecast(ctx, out, core::DataType::kFloat).to_vector<float>()[0]);
//                }
//            }
//            float sum = 0;
//            for (int i = 0; i < local_w.size(); ++i) {
//                if (i) std::cout << " + ";
//                else std::cout << "!!!!!!! ";
//                std::cout << local_w[i] << " * " << local_out[i];
//                sum += local_w[i] * local_out[i];
//            }
//            std::cout << " = " << sum << endl << endl;
//
//            std::cout << "sum_ret: " << ret.slice_dim0(0, 1) << endl;
//            std::cout.flush();
//        }

        return with_share(ctx, input, ret);
    }
};

class FeedForward::impl::FusedMOE : public FeedForward::impl::MOEImpl {
protected:
    std::shared_ptr<nn::Linear> all_in;
    std::shared_ptr<nn::Linear> all_gated;
    std::shared_ptr<nn::Linear> all_out;
    Tensor shared_ids;
    Tensor shared_weights;
    bool fused_shared{false};
public:
    FusedMOE(
        const core::Context& ctx,
        model::ModelConfig cfg,
        model::QuantConfig quant_config,
        bool parallel,
        bool dyn_shared)
        : FeedForward::impl::MOEImpl(ctx, cfg, quant_config, parallel, dyn_shared) {
        name = "FusedMOE";
    }

    virtual ~FusedMOE() = default;

    void post_load(const core::Context& ctx) {
        std::vector<nn::Linear*> all_gateds;
        std::vector<nn::Linear*> all_ins;
        std::vector<nn::Linear*> all_outs;
        for (int i = 0; i < num_local_experts; ++i) {
            all_gateds.push_back(&experts[i]->w_gated);
            all_ins.push_back(&experts[i]->w_in);
            all_outs.push_back(&experts[i]->w_out);
        }
        // Fuse shared_experts
        int gptq_kernel_algo = utils::get_int_env("GPTQ_KERNEL_ALGO", 1);
        std::vector<nn::Linear*> shared_gateds;
        std::vector<nn::Linear*> shared_ins;
        std::vector<nn::Linear*> shared_outs;
        if ((dyn_shared) && shared_expert.get() &&
            gptq_kernel_algo >= 1) {
            BM_ASSERT(n_shared_experts > 0, "");
            shared_gateds = shared_expert->w_gated.split(ctx, n_shared_experts, true);
            shared_ins = shared_expert->w_in.split(ctx, n_shared_experts, true);
            shared_outs = shared_expert->w_out.split(ctx, n_shared_experts, false);
        }
        if (!shared_gateds.empty()) {
            BM_ASSERT_EQ(all_gateds.size(), all_ins.size(), "");
            BM_ASSERT_EQ(all_gateds.size(), all_outs.size(), "");

            all_gateds.insert(all_gateds.end(), shared_gateds.begin(), shared_gateds.end());
            all_ins.insert(all_ins.end(), shared_ins.begin(), shared_ins.end());
            all_outs.insert(all_outs.end(), shared_outs.begin(), shared_outs.end());

            if (ctx.rank() == 0 && ctx.current_layer() == 1000)
                std::cout << "Fuse shared_expert(s) n_shared_experts=" << shared_outs.size() << endl;
            fused_shared = true;
        }

        all_in.reset(nn::Linear::fuse(ctx, all_ins));
        all_gated.reset(nn::Linear::fuse(ctx, all_gateds));
        all_out.reset(nn::Linear::fuse(ctx, all_outs));

        if (dyn_shared) {
            BM_ASSERT_EQ(experts.size(), num_local_experts + n_shared_experts, "");
            BM_ASSERT_EQ(shared_gateds.size(), size_t(n_shared_experts), "");
            // move shared_experts to experts
            for (int i = 0; i < n_shared_experts; ++i) {
                experts[num_local_experts + i]->w_gated.move(*shared_gateds[i]);
                experts[num_local_experts + i]->w_in.move(*shared_ins[i]);
                experts[num_local_experts + i]->w_out.move(*shared_outs[i]);
            }
            shared_expert.reset();
        }
        for (auto p: shared_gateds) delete p;
        for (auto p: shared_ins) delete p;
        for (auto p: shared_outs) delete p;
    }

};

class FeedForward::impl::GPTQMOE : public FeedForward::impl::FusedMOE {
public:
    GPTQMOE(
        const core::Context& ctx,
        model::ModelConfig cfg,
        model::QuantConfig quant_config,
        bool parallel,
        bool dyn_shared)
        : FeedForward::impl::FusedMOE(ctx, cfg, quant_config, parallel, dyn_shared) {
        name = "GPTQMOE";
    }
    virtual ~GPTQMOE() = default;

    Tensor forward(const core::Context& ctx, const Tensor& input, bool quant_back) override {
        static int m_threshold = utils::get_int_env("GPTQ_MOE_M_THRES", 1);
        if (input.size(0) > m_threshold || !all_in) {
            return FeedForward::impl::MOEImpl::forward(ctx, input, quant_back);
        }

        BM_ASSERT_EQ(input.ndim(), 2, "");
        // (num_token, top_k_may_share)
        auto [expert_ids, expert_weights] = route(ctx, input, dyn_shared);

        auto[qw1, qz1, scales1, sym1] = all_in->get_gptq_weights(); // nExp * ...
        auto[qw2, qz2, scales2, sym2] = all_gated->get_gptq_weights();
        auto[qw3, qz3, scales3, sym3] = all_out->get_gptq_weights();
        std::string ev_name = "GPTP_fuse_moe";
        if (ctx.debug() >= 2 && ctx.rank() == 0) {
            auto ids = expert_ids.to_vector<int>();
            // std::cout << "ids: " << ids << endl;
            int local_count = 0;
            for (int id : ids) {
                if (is_local_expert(id)) local_count++;
            }
            int K = input.size(-1);
            int N = qw1.size(1);
            ev_name += logger::str_cat(":E=", local_count, ",K=", K, ",N=", N);
        }
        ctx.recordEvent("Start>" + ev_name, 2);
        ctx.recordEvent("Up", 2);
        Tensor up = nn::gptq::gemm_moe_up(
            ctx, input, qw1, qz1, scales1, Tensor(), qw2, qz2, scales2, Tensor(), sym1, expert_ids, 0, exp_parallel);
        ctx.recordEvent("Down", 2);
        Tensor ret = nn::gptq::gemm_moe_down(
            ctx, up, qw3, qz3, scales3, expert_ids, expert_weights, sym3, 0, exp_parallel);
        ctx.recordEvent("End>" + ev_name, 2);

        return with_share(ctx, input, ret);
    }
};

class FeedForward::impl::FP8BlockMOE : public FeedForward::impl::FusedMOE {
    const int block_m = 64;
public:
    FP8BlockMOE(
        const core::Context& ctx,
        model::ModelConfig cfg,
        model::QuantConfig quant_config,
        bool parallel,
        bool dyn_shared)
        : FeedForward::impl::FusedMOE(ctx, cfg, quant_config, parallel, dyn_shared) {
        name = "FP8BlockMOE";
    }
    ~FP8BlockMOE() override = default;

    void fill_padded_block_idx(
        const vector<int> exp_num_tokens,
        int* reverse_idx
    ) {
        int offset = 0;
        int src_index = 0;
        for (int e = 0; e < exp_num_tokens.size(); ++e) {
            int num = exp_num_tokens[e];
            for (int i = 0; i < num; ++i) {
                reverse_idx[src_index++] = offset + i;
            }
            offset += round_up(num, block_m);
        }
    }

    Tensor fill_m_indices(const core::Context& ctx,
                          size_t sum_m,
                          const vector<int>& active_exp,
                          const vector<int>& exp_num_tokens) {
        vector<int> indices;
        indices.reserve(sum_m);
        for (int e = 0; e < active_exp.size(); ++e) {
            int exp = local_expert_id(active_exp[e]);
            int num = round_up(exp_num_tokens[e], block_m);
            for (int i = 0; i < num; ++i) {
                indices.push_back(exp);
            }
        }
        return ctx.tensor_of(indices);
    }

    // Return grouped input align m to block_m=64
    void get_grouped_input(
        const core::Context& ctx,
        const Tensor& input,                   // (num_tokens, dim_model)
        const vector<int>& token_experts,      // (num_experts, top_k)，map: {token_id -> exp_id}
        const vector<vector<int>>& exp_tokens, // Ragged: (num_experts, seq_len in expert)， map: {exp_id -> token_list}
        Tensor& grouped_input,                 // (num_block * block_m, dim_model)
        Tensor& m_indices,                     // (num_block * block_m)
        Tensor& padded_block_idx,
        Tensor& reverse_idx_d,
        vector<int>& exp_num_tokens
    ) {
        // Note:
        //   num_experts: is num_experts_may_share if (dyn_shared == true).
        //   exp: mean expert id.
        //   token: means token index in the input.
        size_t num_input_tokens = input.size(0);
        const int total_tokens_k = int(num_input_tokens) * top_k_may_share;

        // concat ragged to flat buffer
        int* flat_buf_ptr = pin_buf;
        vector<int> active_exp, acc_offset;
        size_t local_tokens_k = filter_concat_exp_tokens(
            exp_tokens, flat_buf_ptr, active_exp, exp_num_tokens, acc_offset);
#if 0
        size_t num_block = 0;
        vector<int> index_in_grouped_input; // (local_tokens_k)
        vector<int> index_in_input;         // (local_tokens_k)
        BM_ASSERT_EQ(exp_tokens.size(), num_experts_may_share, "exp_tokens.size() mismatch");
        for (int exp = 0; exp < num_experts_may_share; ++exp) {
            if (is_local_expert(exp) && !exp_tokens[exp].empty()) {
                auto& tokens = exp_tokens[exp];
                for (int j = 0; j < tokens.size(); ++j) {
                    index_in_grouped_input.push_back(num_block * block_m + j);
                }
                index_in_input.insert(index_in_input.end(), tokens.begin(), tokens.end());
                local_tokens_k += tokens.size();
                size_t aligned_m = round_up(tokens.size(), block_m);
                num_block += aligned_m / block_m;
            }
        }
#endif

        if (exp_parallel) {
            if (local_tokens_k == 0) {
                return;
            }
        } else {
            BM_ASSERT_EQ(local_tokens_k, num_input_tokens, "");
        }

        int total_tokens_aligned = 0;
        for (auto num: exp_num_tokens) {
            total_tokens_aligned += round_up(num, block_m);
        }
        m_indices = fill_m_indices(ctx, total_tokens_aligned, active_exp, exp_num_tokens);
        grouped_input = ctx.tensor({size_t(total_tokens_aligned), input.size(1)}, input.dtype());
        functions::zeros_(ctx, grouped_input);

        // build reverse index
        int* reverse_idx = flat_buf_ptr + local_tokens_k;
        fill_reverse_index(token_experts, reverse_idx);

        int* padded_block_idx_h = reverse_idx + total_tokens_k;
        fill_padded_block_idx(exp_num_tokens, padded_block_idx_h);

        size_t total_size = local_tokens_k + total_tokens_k + local_tokens_k;
        Tensor all = ctx.tensor({total_size}, core::DataType::kInt32);
        all.from_buffer(flat_buf_ptr, ctx.current_cuda_stream());

        // for reorder/permute input for experts
        Tensor group_idx_d = all.slice_dim0_len(0, local_tokens_k);
        // for reorder experts output
        reverse_idx_d = all.slice_dim0_len(local_tokens_k, total_tokens_k);
        // for un-pad
        padded_block_idx = all.slice_dim0_len(local_tokens_k + total_tokens_k, local_tokens_k);

        functions::scatter_update_dim0(ctx, grouped_input, padded_block_idx, input, group_idx_d);
        if (ctx.debug() > 1 && ctx.is_layer(3)) {
            std::cout << "check_equal\n";
//            std::cout << "exp_tokens: " << exp_tokens << endl;
//            std::cout << "group_idx_d: " << group_idx_d << endl;
//            std::cout << "padded_block_idx: " << padded_block_idx << endl;
            Tensor reorder = functions::index_select(ctx, input, 0, group_idx_d);
//            std::cout << "input: " << input << endl;
//            std::cout << "grouped_input: " << grouped_input << endl;
//            std::cout << "reorder: " << reorder << endl;
            Tensor unpad = ctx.tensor(reorder.shape(), reorder.dtype());
            std::vector<int> seq(unpad.size(0));
            std::iota(seq.begin(), seq.end(), 0);
            Tensor seq_d = ctx.tensor_of(seq);
            functions::scatter_update_dim0(ctx, unpad, seq_d, grouped_input, padded_block_idx);
//            std::cout << "unpad: " << unpad << endl;
            functions::check_equal(ctx, reorder, unpad);
            Tensor unpad1 = functions::index_select(ctx, grouped_input, 0, padded_block_idx);
            functions::check_equal(ctx, reorder, unpad1);
            std::cout << "Done check_equal\n";
        }

//        size_t group_offset = 0;
//        for (int j = 0; j < exp_num_tokens.size(); ++j) {
//            // call index_select INPLACE for each expert
//            int num = exp_num_tokens[j];
//            Tensor exp_input = grouped_input.slice_dim0_len(group_offset, num);
//            Tensor idx = group_idx_d.slice_dim0_len(acc_offset[j], num);
//            functions::index_select(ctx, input, 0, idx, &exp_input);
//            group_offset += round_up(num, block_m);
//        }
    }

    Tensor forward(const core::Context& ctx, const Tensor& input, bool quant_back) override {
        BM_ASSERT(all_in, "No fused weight");
//        static int m_threshold = utils::get_int_env("FP8BLOCK_MOE_M_THRES", 0);
//        if (input.size(0) > m_threshold) {
//            return FeedForward::impl::MOEImpl::forward(ctx, input, quant_back);
//        }

        int event_level = 3;
        size_t total_seq_len = input.size(0);
        const Tensor& h = input;
        // (num_token, top_k_may_share)
        auto [ids_t, weights] = route(ctx, h, dyn_shared); // may include shared
        vector<int> routing_experts(total_seq_len * top_k_may_share);
        ctx.recordEvent("exp_ids_to_buffer", event_level);
        BM_ASSERT_EQ(routing_experts.size(), ids_t.numel(), "");
        ids_t.to_buffer(routing_experts.data(), ctx.current_stream()->ptr);

        // num_experts_may_share of [routed tokens to this expert] (could be empty)
        vector<vector<int>> all_expert_tokens = dispatch_token(ctx, h,routing_experts, total_seq_len);
        // permute input for LOCAL experts
        // h_reorder: exp1 tokens, exp3 tokens...
        auto [h_reorder, rev_idx_d] = permute_input(ctx, h, routing_experts, all_expert_tokens, total_seq_len);

        auto [local_experts, expert_token_num] = filter_active_experts(all_expert_tokens);
        size_t sum_token_num = std::accumulate(expert_token_num.begin(), expert_token_num.end(), 0UL);

        // Special case0: not any local experts hits
        if (local_experts.empty()) {
            // in expert parallel mode, and not any local experts hits, return immediately
            return forward_shared_or_zero(ctx, input);
        }
        // Special case1: Only one local expert hits, we don't need grouped gemm
        if (local_experts.size() == 1) {
            // ... Low priority
        }

        // Padding to block_m
        Tensor grouped_input; // (num_block * block_m, dim_model)
        Tensor m_indices;     // (num_block * block_m)
        Tensor padded_block_idx;
        Tensor reverse_idx_d;
        vector<int> exp_num_tokens;
        get_grouped_input(ctx, input, routing_experts, all_expert_tokens,
                          grouped_input, m_indices, padded_block_idx, reverse_idx_d, exp_num_tokens);

        Tensor w_0 = all_in->grouped_gemm_fp8_block(ctx, grouped_input, m_indices, experts.size());
        Tensor w_1 = all_gated->grouped_gemm_fp8_block(ctx, grouped_input, m_indices, experts.size());
        nn::gate_mul_inplace(ctx, w_0, w_1, experts[0]->act_fn_type);
        Tensor w_2 = all_out->grouped_gemm_fp8_block(ctx, w_0, m_indices, experts.size());
        if (ctx.is_layer(300)) {
            std::cout << "m_indices: " << m_indices << endl;
            for (int i = 0; i < w_2.size(0) / block_m; ++i) {
                std::cout << "Block:" << i << endl;
                std::cout << "IN:" << grouped_input.slice_dim0_len(i * block_m, block_m) << endl;
                if (i < 4) {
                    std::cout << "w_0:" << w_0.slice_dim0_len(i * block_m, block_m) << endl;
                }
                std::cout << "OUT:" << w_2.slice_dim0_len(i * block_m, block_m) << endl;
            }
        }
        Tensor w_2a = functions::index_select(ctx, w_2, 0, padded_block_idx);
//        Tensor w_0  = ctx.tensor({padded_block_idx.size(0), in.size(-1)}, in.dtype());
//        Tensor w_1 = ctx.tensor(w_0.shape(), w_0.dtype());
//        Tensor w_0 = functions::index_select(ctx, in, 0, padded_block_idx);
//        Tensor w_1 = functions::index_select(ctx, gated, 0, padded_block_idx);
//        nn::gate_mul_inplace(ctx, w_0, w_1, experts[0]->act_fn_type);

        vector<Tensor> slice_ret = slice_dim0(w_2a, expert_token_num, false);

        vector<Tensor> expert_ret(num_experts_may_share); // GLOBAL (because routing_weights_t is global)
        for (size_t j = 0; j < local_experts.size(); ++j) {
            int local_exp = local_experts[j];
            int global_exp = global_expert_id(local_exp);
            expert_ret[global_exp] = slice_ret[j];
//            if (ctx.is_layer(3)) {
//                std::cout << "j=" << j << "global_exp=" << global_exp << endl;
//                std::cout << "ret: " << slice_ret[j] << endl;
//            }
        }

        ctx.recordEvent("sum_experts", event_level);
        Tensor ret = sum_experts(
            ctx,
            expert_ret,
            Tensor(),
            ids_t,
            reverse_idx_d,
            weights,
            exp_parallel,
            ctx.world_size(),
            ctx.rank());
        return with_share(ctx, input, ret);
    }
};

FeedForward::FeedForward(
    const core::Context& ctx,
    model::ModelConfig cfg,
    model::QuantConfig quant_config,
    bool parallel) {
    if (cfg.moe_num_experts > 0 && ctx.current_layer() >= cfg.first_k_dense_replace) {
        BM_ASSERT(cfg.moe_top_k > 0, "moe_top_k unset");
        bool fuse_moe = utils::get_int_env("FUSE_GPTQ_MOE", 0) > 0;
        int gptq_kernel_algo = utils::get_int_env("GPTQ_KERNEL_ALGO", 1);
        bool exp_parallel = utils::get_int_env("MOE_EXP_PARALLEL", 0) > 0;
        bool dyn_shared = utils::get_int_env("MOE_DYN_SHARED", 0) > 0;
        bool grouped_fp8_gemm = utils::get_int_env("GROUPED_FP8_GEMM", 0) > 0;
        if (dyn_shared) {
            BM_ASSERT(exp_parallel, "MOE_DYN_SHARED only uses in EP mode");
        }
//        if (ctx.is_layer(cfg.first_k_dense_replace)) {
//            std::cout << "fuse_moe=" << fuse_moe
//                << ", exp_parallel=" << exp_parallel
//                << ", dyn_shared=" << dyn_shared << endl;
//        }
        impl::MOEImpl* ptr;
        if (grouped_fp8_gemm) {
            ptr = new impl::FP8BlockMOE(ctx, cfg, quant_config, parallel, dyn_shared);
        } else if (fuse_moe && gptq_kernel_algo == 1) {
            ptr = new impl::GPTQMOE(ctx, cfg, quant_config, parallel, dyn_shared);
        } else {
            ptr = new impl::MOEImpl(ctx, cfg, quant_config, parallel);
        }
        add_submodule("router", ptr->router);
        for (int i = 0; i < ptr->num_local_experts; ++i) {
            auto p = ptr->experts[i];
            int exp_id = ptr->global_expert_id(i);
            add_submodule("experts." + std::to_string(exp_id) + ".w_in", p->w_in);
            add_submodule("experts." + std::to_string(exp_id) + ".w_gated", p->w_gated);
            add_submodule("experts." + std::to_string(exp_id) + ".w_out", p->w_out);
        }
        if (ptr->shared_expert) {
            add_submodule("shared_expert.w_in", ptr->shared_expert->w_in);
            add_submodule("shared_expert.w_gated", ptr->shared_expert->w_gated);
            add_submodule("shared_expert.w_out", ptr->shared_expert->w_out);
            // add_submodule("shared_expert_gate", ptr->shared_expert_gate.get());
        }
        pimpl.reset(ptr);
    } else {
        impl::NormalImpl* p = quant_config.fuse_ff() ?
            new impl::Int8Impl(ctx, cfg, quant_config, parallel) :
            new impl::NormalImpl(ctx, cfg, quant_config, parallel);
        pimpl.reset(p);
        add_submodule("w_in", p->w_in);
        add_submodule("w_gated", p->w_gated);
        add_submodule("w_out", p->w_out);
    }
}

FeedForward::~FeedForward() = default;

core::Tensor FeedForward::forward(const core::Context& ctx, const core::Tensor& input) {
    core::EventScope event_scope(ctx, pimpl->name, 1);
    auto shape2d = {input.numel() / input.size(-1), input.size(-1)};
    const Tensor& input2d = input.ndim() == 2 ? input : input.view(shape2d);
    Tensor ret = pimpl->forward(ctx, input2d, true);
    if (input.ndim() == 2) {
        return ret;
    } else {
        auto shape_nd = input.shape();
        *shape_nd.rbegin() = ret.size(-1);
        return ret.view(shape_nd);
    }
}

const Linear& FeedForward::w_out() const {
    return dynamic_cast<impl::NormalImpl*>(pimpl.get())->w_out;
}

void FeedForward::load_state_dict(
    const core::Context& ctx,
    const std::map<std::string, const core::Tensor>& state_dict,
    const std::string& prefix,
    bool allow_missing) {
    core::Layer::load_state_dict(ctx, state_dict, prefix, allow_missing);
    int fuse_w_in = utils::get_int_env("CPM_FUSE_FF_IN", 0);
    auto normal_impl = dynamic_cast<impl::NormalImpl*>(pimpl.get());
    auto moe_impl = dynamic_cast<impl::MOEImpl*>(pimpl.get());
    auto fused_moe = dynamic_cast<impl::FusedMOE*>(pimpl.get());
    if (fuse_w_in && normal_impl) {
        normal_impl->try_fuse_up_weights(ctx);
    } else if (fused_moe) {
        fused_moe->post_load(ctx);
        if (moe_impl->shared_expert) {
            moe_impl->shared_expert->try_fuse_up_weights(ctx);
        }
    } else if (fuse_w_in && moe_impl) {
        for (int i = 0; i < moe_impl->num_local_experts; ++i) {
            moe_impl->experts[i]->try_fuse_up_weights(ctx);
        }
        if (moe_impl->shared_expert) {
            moe_impl->shared_expert->try_fuse_up_weights(ctx);
        }
    }
    if (moe_impl && moe_impl->topk_method == "noaux_tc") {
        // Load and auto cast to DataType::kFloat
        auto it = state_dict.find(prefix + ".router.e_score_correction_bias");
        BM_ASSERT(it != state_dict.end(), "No e_score_correction_bias.");
        auto shape = moe_impl->e_score_correction_bias.shape();
        auto dtype = it->second.dtype() == DataType::kFloat ? DataType::kFloat : moe_impl->dtype;
        auto temp = ctx.parameter(shape, dtype);
        ctx.assign_or_copy(&temp, &it->second);
        moe_impl->e_score_correction_bias = functions::typecast(ctx, temp, core::DataType::kFloat);
    }
}

void FeedForward::dequant_cache_weight(core::Context& ctx, const core::Tensor& in) {
    auto normal_impl = dynamic_cast<impl::NormalImpl*>(pimpl.get());
    if (normal_impl) {
        normal_impl->w_out.dequant_cache_weight(ctx, in);
        if (normal_impl->w_fuse_up) {
            normal_impl->w_fuse_up->dequant_cache_weight(ctx, in);
        } else {
            normal_impl->w_gated.dequant_cache_weight(ctx, in);
            normal_impl->w_in.dequant_cache_weight(ctx, in);
        }
    }
}
}
