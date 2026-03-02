#pragma once

#include "model/model.h"
#include "model/model_context.h"
#include "nn/nn.h"
#include "nn/position/rope_preparer.h"
#include <memory>

namespace model {

class ModelContext;
class GenerationContext;

class LLaMALike : public ModelBase {
public:
    explicit LLaMALike(ModelConfig model_config) : ModelBase(model_config) { }

    virtual core::Tensor forward(
        ModelContext& ctx,
        const core::Tensor& ids,        // int32 (batch, len_q)
        const core::Tensor& pos_ids,    // int32 (batch, len_ext)
        const core::Tensor& seqlens_q,  // int32 (batch)
        const core::Tensor& seqlens_kv, // int32 (batch)
        const core::Tensor& mask,       // int8 (batch, len_q, len_buf)
        const core::Tensor& placement,
        const core::Tensor& hidden_pass, // half (batch, len_q, dim_model)
        core::Tensor* hidden_ptr = nullptr) = 0;

    virtual core::Tensor encode(
        ModelContext& ctx,
        const core::Tensor& ids,        // int32 (batch, len_q)
        const core::Tensor& pos_ids,    // int32 (batch, len_ext)
        const core::Tensor& seqlens_q,  // int32 (batch)
        const core::Tensor& seqlens_kv, // int32 (batch)
        const core::Tensor& mask,       // int8 (batch, len_q, len_buf)
        const core::Tensor& placement,
        const core::Tensor& hidden_pass, // half (batch, len_q, dim_model)
        bool ln_output = true,
        std::vector<core::Tensor>* hidden_states = nullptr) = 0;

    virtual core::Tensor get_input_embeddings(
        ModelContext& ctx, const core::Tensor& ids) = 0;

    virtual core::Tensor get_logits(
        ModelContext& ctx, const core::Tensor& hidden, bool ln_input) = 0;

    virtual functions::ModuleList<nn::EncoderLayer>& get_encoder() = 0;

    // set to all encoder layers
    void set_mask_modules(const std::vector<std::vector<bool>>& mask_modules);

    // set to all encoder layers
    void set_residual_scale(float residual_scale);
};

class LLaMA : public LLaMALike {
private:
    bool parallel;
    bool tie_lm_head;

    functions::ModuleList<nn::EncoderLayer> encoder;
    nn::LayerNorm ln_after_enc;
    nn::RawEmbedding lm_head;
    nn::RawEmbedding token_embedding;
    std::unique_ptr<nn::RopePreparer> rope_preparer;

    BM_LAYER_DEF_PUBLIC(LLaMA);

    LLaMA(core::Context& ctx, ModelConfig model_config, QuantConfig quant_config, bool parallel = false);

    functions::ModuleList<nn::EncoderLayer>& get_encoder() override { return encoder; }

    core::Tensor forward(
        ModelContext& ctx,
        const core::Tensor& ids,        // int32 (batch, len_q)
        const core::Tensor& pos_ids,    // int32 (batch, len_ext)
        const core::Tensor& seqlens_q,  // int32 (batch)
        const core::Tensor& seqlens_kv, // int32 (batch)
        const core::Tensor& mask,       // int8 (batch, len_q, len_buf)
        const core::Tensor& placement,
        const core::Tensor& hidden_pass, // half (batch, len_q, dim_model)
        core::Tensor* hidden_ptr = nullptr) override;

    core::Tensor encode(
        ModelContext& ctx,
        const core::Tensor& ids,        // int32 (batch, len_q)
        const core::Tensor& pos_ids,    // int32 (batch, len_ext)
        const core::Tensor& seqlens_q,  // int32 (batch)
        const core::Tensor& seqlens_kv, // int32 (batch)
        const core::Tensor& mask,       // int8 (batch, len_q, len_buf)
        const core::Tensor& placement,
        const core::Tensor& hidden_pass, // half (batch, len_q, dim_model)
        bool ln_output = true,
        std::vector<core::Tensor>* hidden_states = nullptr) override;

    core::Tensor get_input_embeddings(ModelContext& ctx, const core::Tensor& ids) override;

    core::Tensor get_logits(ModelContext& ctx, const core::Tensor& hidden, bool ln_input) override;

    float calc_loss(
        const core::Context& ctx,
        const core::Tensor& ids,     // int32 (len_q)
        const core::Tensor& pos_ids, // int32 (len_q)
        const core::Tensor& mask,    // int8  (len_q, len_buf)
        const core::Tensor& label    // int32 (len_q)
    );

    int calc_greedy_match(
        const core::Context& ctx,
        const core::Tensor& ids,     // int32 (len_q)
        const core::Tensor& pos_ids, // int32 (len_q)
        const core::Tensor& mask,    // int8  (len_q, len_buf)
        const core::Tensor& label    // int32 (len_q)
    );

    std::tuple<float, core::Tensor> calc_log_prob(
        const core::Context& ctx,
        const core::Tensor& ids,     // int32 (len_q)
        const core::Tensor& pos_ids, // int32 (len_q)
        const core::Tensor& mask,    // int8  (len_q, len_buf)
        const core::Tensor& label    // int32 (len_q)
    );

    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>>
    calc_logits(
        const core::Context& ctx,
        const core::Tensor& idx,     // int32 (len_q)
        const core::Tensor& pos_ids, // int32 (len_q)
        const core::Tensor& mask,    // int8  (len_q, len_buf)
        const core::Tensor& label,   // int32 (len_q)
        bool return_hidden_states);
};

} // namespace model