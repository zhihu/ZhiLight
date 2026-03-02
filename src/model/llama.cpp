#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>
#include "model/llama.h"
#include "model/allocate_util.hpp"
#include "model/dyn_batch_context.h"
#include "utils/env.h"
#include "kvcache/kvcache.h"

namespace model {

LLaMA::LLaMA(core::Context& ctx, ModelConfig model_config, QuantConfig quant_config, bool parallel)
    : LLaMALike(model_config),
      ln_after_enc(
          ctx,
          dim_model,
          false,
          eps,
          model_config.model_type == "cpm_dragonfly" ? dim_model / model_config.dim_model_base
                                                     : 1.0,
          dtype),
      token_embedding(ctx, dim_model, vocab_size, false, dtype, parallel),
      lm_head(ctx, dim_model, vocab_size, false, dtype, parallel),
      parallel(parallel),
      tie_lm_head(model_config.tie_lm_head) {

    std::vector<int> devices = partition_layer_devices(ctx, num_layers);

    for (int i = 0; i < num_layers; i++) {
        ctx.switch_to_device(devices[i]);

        ctx.set_current_layer(i);
        encoder.append(ctx, model_config, quant_config, parallel);
        encoder[i].output_dev = devices[i];
    }
    encoder[num_layers - 1].output_dev = 0;
    ctx.switch_to_device(0);

    if (model_config.model_type == "cpm_dragonfly")
        token_embedding.set_scale_factor(model_config.scale_emb);

    add_submodule("layers", encoder);
    add_submodule("output_layernorm", ln_after_enc);
    add_submodule("token_embedding", token_embedding);
    if (!tie_lm_head)
        add_submodule("lm_head", lm_head);

    if (model_config.model_type == "cohere") {
        tie_lm_head = true;
        token_embedding.set_logit_scale(model_config.logit_scale);
        ln_after_enc.set_rms(false);
    }

    if (utils::get_int_env("ROPE_CACHE") == 1) {
        rope_preparer = std::make_unique<nn::RopePreparer>(ctx, model_config);
    }
}

core::Tensor LLaMA::forward(
    ModelContext& ctx,
    const core::Tensor& ids,        // int32 (batch, len_q)
    const core::Tensor& pos_ids,    // int32 (batch, len_q)
    const core::Tensor& seqlens_q,  // int32 (batch) cumulative seqlens of query.
    const core::Tensor& seqlens_kv, // int32 (batch)
    const core::Tensor& mask,       // int8 (batch, len_q, len_buf)
    const core::Tensor& placement,
    const core::Tensor& hidden_pass, // half (batch, len_q, dim_model)
    core::Tensor* hidden_ptr) {
    auto hidden =
        encode(ctx, ids, pos_ids, seqlens_q, seqlens_kv, mask, placement, hidden_pass, true);
    if (hidden_ptr)
        *hidden_ptr = hidden;
    return get_logits(ctx, hidden, false);
}

core::Tensor LLaMA::encode(
    ModelContext& ctx,
    const core::Tensor& ids,     // int32 (len_q)
    const core::Tensor& pos_ids, // int32 (len_q)
    const core::Tensor& seqlens_q,
    const core::Tensor& seqlens_kv,
    const core::Tensor& mask, // int8 (len_q, len_buf)
    const core::Tensor& placement,
    const core::Tensor& hidden_pass, // half (batch, len_q, dim_model)
    bool ln_output,
    std::vector<core::Tensor>* hidden_states) {
    ctx.set_current_layer(-1);
    Tensor hidden;
    if (hidden_pass.empty()) {
        hidden = token_embedding(ctx, ids);
    } else {
        hidden = functions::typecast(ctx, hidden_pass, dtype);
        // std::cout << "hidden: " << hidden << std::endl;
    }
    if (hidden_states) {
        hidden_states->clear();
        hidden_states->reserve(static_cast<size_t>(num_layers) + 1);
    }
    if (rope_preparer && ctx.dyn_batch()) {
        auto& rope_cache = ctx.dyn_batch()->rope_cache;
        std::tie(rope_cache.cos, rope_cache.sin) = rope_preparer->forward(ctx, ids, pos_ids);
    }
    bool dual_stream = utils::get_int_env("DUAL_STREAM", 0) > 0 && ctx.world_size() > 1;
    int dual_stream_thres = utils::get_int_env("DUAL_STREAM_THRESHOLD", 1024);
    if (hidden_states) {
        // Need per-layer hidden states; fall back to the standard path.
        dual_stream = false;
    }
    if (dual_stream && ctx.get_compute_capability() > 80 && ids.size(0) > dual_stream_thres) {
        // auto hidden1 = token_embedding(ctx, ctx.dyn_aux()->e_token);
        hidden = nn::EncoderLayer::dual_stream_encode(ctx, encoder, hidden, pos_ids);
    } else {
        int debug_layer = utils::get_int_env("CPM_DEBUG_LAYER", -1);
        int debug_layer_level = utils::get_int_env("CPM_DEBUG_LAYER_LEVEL", 2);
        int event_level = utils::get_int_env("CPM_DEBUG_LAYER_EV_LEVEL", debug_layer_level);
        for (int i = 0; i < num_layers; i++) {
            if (hidden_states) {
                hidden_states->push_back(hidden);
            }
            ctx.set_current_layer(i);
            auto org_debug_level = ctx.debug();
            if (i == debug_layer && ctx.rank() == 0) {
                ctx.enable_debug(debug_layer_level);
                ctx.set_event_level(event_level);
            }
            hidden = encoder[i](
                ctx,
                hidden,
                mask,
                pos_ids,
                seqlens_q,
                seqlens_kv,
                ctx.buf_k(i),
                ctx.buf_v(i),
                ctx.block_table(i),
                &placement);
            ctx.enable_debug(org_debug_level);
            ctx.set_event_level(-1);
        }
    }
    ctx.set_current_layer(-1);
    if (ln_output) {
        hidden = ln_after_enc(ctx, hidden);
    }
    if (hidden_states) {
        hidden_states->push_back(hidden);
    }
    ctx.print_events();
    if (ctx.dyn_batch())
        ctx.dyn_batch()->rope_cache.clear();
    return hidden;
}

core::Tensor LLaMA::get_input_embeddings(
    ModelContext& ctx, const core::Tensor& ids) {
    ctx.set_current_layer(-1);
    return token_embedding.forward(ctx, ids);
}

core::Tensor LLaMA::get_logits(ModelContext& ctx, const core::Tensor& hidden, bool ln_input) {
    auto ln_hidden = ln_input ? ln_after_enc(ctx, hidden) : hidden;
    Tensor ret = tie_lm_head
        ? token_embedding.projection(ctx, ln_hidden)
        : lm_head.projection(ctx, ln_hidden);
    return ret;
}

float LLaMA::calc_loss(
    const core::Context& ctx,
    const core::Tensor& ids,     // int32 (len_q)
    const core::Tensor& pos_ids, // int32 (len_q)
    const core::Tensor& mask,    // int8  (len_q, len_buf)
    const core::Tensor& label    // int32 (len_q)
) {
    auto hidden = token_embedding(ctx, ids);
    for (int i = 0; i < num_layers; i++) {
        hidden = encoder[i](
            ctx,
            hidden,
            mask,
            pos_ids,
            core::Tensor(),
            core::Tensor(),
            nullptr,
            nullptr,
            nullptr,
            nullptr);
    }
    auto grads = nn::cross_entropy_raw(
        ctx, lm_head.projection(ctx, ln_after_enc(ctx, hidden)), label, -100, 1.0);
    float loss = std::get<0>(grads);
    return loss;
}

int LLaMA::calc_greedy_match(
    const core::Context& ctx,
    const core::Tensor& ids,     // int32 (len_q)
    const core::Tensor& pos_ids, // int32 (len_q)
    const core::Tensor& mask,    // int8  (len_q, len_buf)
    const core::Tensor& label    // int32 (len_q)
) {
    auto hidden = token_embedding(ctx, ids);
    for (int i = 0; i < num_layers; i++) {
        hidden = encoder[i](
            ctx,
            hidden,
            mask,
            pos_ids,
            core::Tensor(),
            core::Tensor(),
            nullptr,
            nullptr,
            nullptr,
            nullptr);
    }
    int greedy_match =
        nn::greedy_match_raw(ctx, lm_head.projection(ctx, ln_after_enc(ctx, hidden)), label, -100);
    return greedy_match;
}

std::tuple<float, core::Tensor> LLaMA::calc_log_prob(
    const core::Context& ctx,
    const core::Tensor& ids,     // int32 (len_q)
    const core::Tensor& pos_ids, // int32 (len_q)
    const core::Tensor& mask,    // int8  (len_q, len_buf)
    const core::Tensor& label    // int32 (len_q)
) {
    auto hidden = token_embedding(ctx, ids);
    for (int i = 0; i < num_layers; i++) {
        hidden = encoder[i](
            ctx,
            hidden,
            mask,
            pos_ids,
            core::Tensor(),
            core::Tensor(),
            nullptr,
            nullptr,
            nullptr,
            nullptr);
    }
    auto log_prob =
        nn::log_prob_raw(ctx, lm_head.projection(ctx, ln_after_enc(ctx, hidden)), label, -100);
    return log_prob;
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> LLaMA::
    calc_logits(
        const core::Context& ctx,
        const core::Tensor& ids,     // int32 (len_q)
        const core::Tensor& pos_ids, // int32 (len_q)
        const core::Tensor& mask,    // int8  (len_q, len_buf)
        const core::Tensor& label,   // int32 (len_q)
        bool return_hidden_states) {
    std::vector<std::vector<float>> logit_list;
    std::vector<std::vector<std::vector<float>>> hidden_list;
    auto hidden = token_embedding(ctx, ids);
    for (int i = 0; i < num_layers; i++) {
        if (return_hidden_states) {
            auto converted = convert_fp32(ctx, hidden);
            std::vector<float> flatten;
            flatten.resize(converted.numel());
            converted.to_buffer(flatten.data());
            std::vector<std::vector<float>> res;
            for (int i = 0; i < converted.size(0); ++i) {
                std::vector<float> inner;
                for (int j = 0; j < converted.size(1); ++j) {
                    inner.emplace_back(flatten[i * converted.size(1) + j]);
                }
                res.emplace_back(inner);
            }
            hidden_list.emplace_back(res);
        }
        hidden = encoder[i](
            ctx,
            hidden,
            mask,
            pos_ids,
            core::Tensor(),
            core::Tensor(),
            nullptr,
            nullptr,
            nullptr,
            nullptr);
    }
    hidden = ln_after_enc(ctx, hidden);
    if (return_hidden_states) {
        auto converted = convert_fp32(ctx, hidden);
        std::vector<float> flatten;
        flatten.resize(converted.numel());
        converted.to_buffer(flatten.data());
        std::vector<std::vector<float>> res;
        for (int i = 0; i < converted.size(0); ++i) {
            std::vector<float> inner;
            for (int j = 0; j < converted.size(1); ++j) {
                inner.emplace_back(flatten[i * converted.size(1) + j]);
            }
            res.emplace_back(inner);
        }
        hidden_list.emplace_back(res);
    }
    core::Tensor logits = lm_head.projection(ctx, hidden);
    {
        auto converted = convert_fp32(ctx, logits);
        std::vector<float> flatten;
        flatten.resize(converted.numel());
        converted.to_buffer(flatten.data());
        for (int i = 0; i < converted.size(0); ++i) {
            std::vector<float> inner;
            for (int j = 0; j < converted.size(1); ++j) {
                inner.emplace_back(flatten[i * converted.size(1) + j]);
            }
            logit_list.emplace_back(inner);
        }
    }
    return std::make_tuple(logit_list, hidden_list);
}

void LLaMALike::set_mask_modules(const std::vector<std::vector<bool>>& mask_modules) {
    auto& encoder = get_encoder();
    for (size_t i = 0; i < encoder.size(); i++) {
        encoder[i].set_mask_modules(mask_modules[i]);
    }
}
}
