#pragma once

#include <bmengine/core/context.h>
#include <bmengine/core/core.h>
#include <curand.h>
#include <unordered_map>
#include <vector>

namespace model {
class ModelContext;
}

namespace beam_utility {
using namespace bmengine;
template <class>
class BeamBufferManager;

core::Tensor log_softmax_bias(
    const core::Context& ctx,
    const core::Tensor& logits, // half (batch, dim_logits)
    const core::Tensor& bias    // float32 (batch)
);

void log_softmax_bias(
    const core::Context& ctx,
    const core::Tensor& logits, // half (batch, dim_logits)
    const core::Tensor& bias,   // float32 (batch)
    float temperature,
    core::Tensor* out);

core::Tensor log_softmax_bias(
    const core::Context& ctx,
    const core::Tensor& logits, // half (batch, dim_logits)
    const core::Tensor& bias,   // float32 (batch)
    float temperature);

core::Tensor gather_logits(
    const core::Context& ctx, const core::Tensor& indexes, const core::Tensor& logits);

core::Tensor apply_gumbel_softmax(
    const core::Context& ctx, curandGenerator_t& gen, const core::Tensor& logits);

void beam_repetition_penalty(
    const core::Context& ctx,
    const std::vector<float>& penalty_factor,
    const std::vector<int32_t>& tokens,
    const std::vector<int32_t>& batch_id,
    core::Tensor& logits,
    const std::vector<float>& presence_penalty = {});

void scatter_update(
    const core::Context& ctx,
    const std::vector<float>& values,
    const std::vector<int32_t>& token_ids,  // indices[1]
    const std::vector<int32_t>& batch_ids,  // indices[0]
    core::Tensor& logits,
    bool add=false);

std::unordered_map<int, float> calc_repetition_ngram(
    const std::vector<int>& token_ids, float ngram_penalty);

void apply_beam_repetition_penalty(
    model::ModelContext& ctx,
    const BeamBufferManager<int>& bm,
    const std::vector<int>& hypotheses_last_pos,
    float ngram_penalty,
    float repetition_penalty,
    core::Tensor* logits_all);

void batch_apply_repetition_penalty(
    model::ModelContext& ctx,
    const std::vector<std::vector<std::vector<int>>>& output_sequences, // [batch, hyp_num, tokens]
    float ngram_penalty,
    float repetition_penalty,
    core::Tensor& logits_all);

void init_curand_gen(const core::Context& ctx, curandGenerator_t& gen, int seed);
}
