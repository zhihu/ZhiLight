#include "generator/beam_util.h"
#include "generator/beam_buffer_manager.hpp"
#include "model/model_context.h"
#include "utils/exception.h"
#include "utils/print.h"

#include <bmengine/core/core.h>
#include <bmengine/functions/reduce.cuh>
#include <bmengine/logger/std_log_op.hpp>
#include <unordered_set>

namespace beam_utility {

using namespace bmengine;
using bmengine::core::Tensor;
using std::vector;

// gridDim (batch, 1, 1),   blockDim (1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(log_softmax_bias)(
    size_t len_vocab,
    float temperature,
    const float* __restrict__ bias, // (batch)
    const T* __restrict__ logits,   // (batch, len_vocab)
    T* __restrict__ out             // (batch, len_vocab)
) {
    size_t offset = blockIdx.x * len_vocab;
    float local_max = -1e20;
    for (int i = threadIdx.x; i < len_vocab; i += blockDim.x) {
        local_max = fmaxf(local_max, logits[offset + i]);
    }
    local_max = functions::blockReduceMax<float>(local_max);
    float local_sum = 1e-20;
    for (int i = threadIdx.x; i < len_vocab; i += blockDim.x) {
        local_sum += expf(((float) logits[offset + i] - local_max) / temperature);
    }
    local_sum = functions::blockReduceSum<float>(local_sum);
    float b = bias[blockIdx.x];
    for (int i = threadIdx.x; i < len_vocab; i += blockDim.x) {
        out[offset + i] =
            ((float) logits[offset + i] - local_max) / temperature - logf(local_sum) + b;
    }
}

// gridDim (batch, 1, 1),   blockDim (1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(log_softmax_bias_without_temperature)(
    size_t len_vocab,
    const float* __restrict__ bias, // (batch)
    const T* __restrict__ logits,   // (batch, len_vocab)
    T* __restrict__ out             // (batch, len_vocab)
) {
    size_t offset = blockIdx.x * len_vocab;
    float local_max = -1e20;
    for (int i = threadIdx.x; i < len_vocab; i += blockDim.x) {
        local_max = fmaxf(local_max, logits[offset + i]);
    }
    local_max = functions::blockReduceMax<float>(local_max);
    float local_sum = 1e-20;
    for (int i = threadIdx.x; i < len_vocab; i += blockDim.x) {
        local_sum += expf((float) logits[offset + i] - local_max);
    }
    local_sum = functions::blockReduceSum<float>(local_sum);
    float b = bias[blockIdx.x];
    for (int i = threadIdx.x; i < len_vocab; i += blockDim.x) {
        out[offset + i] = (float) logits[offset + i] - local_max - logf(local_sum) + b;
    }
}

core::Tensor log_softmax_bias(
    const core::Context& ctx,
    const core::Tensor& logits, // half (batch, dim_logits)
    const core::Tensor& bias    // float32 (batch)
) {
    BM_ASSERT(logits.ndim() >= 2, "logits must be  2 or 3 dimensional");
    size_t dim_logits = logits.size(-1);
    int threads = round_up_thread(dim_logits);
    dim3 gridDim(logits.numel() / dim_logits);
    dim3 blockDim(threads);
    auto stream = ctx.current_stream()->ptr;
    auto out = ctx.tensor(logits.size(), logits.dtype());

    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(log_softmax_bias_without_temperature)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            dim_logits, bias.data<float>(), logits.data<scalar_t>(), out.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return out;
}

void log_softmax_bias(
    const core::Context& ctx,
    const core::Tensor& logits, // half (batch, dim_logits)
    const core::Tensor& bias,   // float32 (batch)
    float temperature,
    core::Tensor* out) {
    BM_ASSERT(logits.ndim() >= 2, "logits must be 2 or 3 dimensional");
    BM_ASSERT_EQ(logits.shape(), out->shape(), "logits and out has different shape");

    size_t dim_logits = logits.size(-1);
    int threads = round_up_thread(dim_logits);
    dim3 gridDim(logits.numel() / dim_logits);
    dim3 blockDim(threads);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(log_softmax_bias)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            dim_logits,
            temperature,
            bias.data<float>(),
            logits.data<scalar_t>(),
            out->mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

core::Tensor log_softmax_bias(
    const core::Context& ctx,
    const core::Tensor& logits, // half (batch, dim_logits)
    const core::Tensor& bias,   // float32 (batch)
    float temperature) {
    if (temperature == 0.0) {
        return std::move(log_softmax_bias(ctx, logits, bias));
    }

    auto out = ctx.tensor(logits.size(), logits.dtype());
    log_softmax_bias(ctx, logits, bias, temperature, &out);
    return std::move(out);
}

// gridDim(N / 1024, 1, 1)  blockDim(1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(gather_logits)(
    int N,
    const int32_t* __restrict__ indexes, // (N,)
    const T* __restrict__ logits_in,
    float* __restrict__ logits_out // (N,)
) {
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < N) {
        int32_t idx = indexes[offset];
        logits_out[offset] = (float) logits_in[idx];
    }
}

core::Tensor gather_logits(
    const core::Context& ctx, const core::Tensor& indexes, const core::Tensor& logits) {
    size_t N = indexes.numel();
    int threads = round_up(min(N, (size_t) 1024), 32);
    dim3 gridDim(round_up(N, threads) / threads, 1, 1);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    auto d_out = ctx.tensor(indexes.size(), core::DataType::kFloat);
    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(gather_logits)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            N, indexes.data<int32_t>(), logits.data<scalar_t>(), d_out.data<float>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return d_out;
}

// gridDim(N / 1024, 1, 1)  blockDim(1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(apply_gumbel)(
    size_t N,
    const float* __restrict__ uniform_eps, // (N,)
    const T* __restrict__ logits_in,       // (N,)
    T* __restrict__ logits_out             // (N,)
) {
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < N) {
        logits_out[offset] = (float) logits_in[offset] - logf(-logf(uniform_eps[offset]));
    }
}

core::Tensor apply_gumbel_softmax(
    const core::Context& ctx, curandGenerator_t& gen, const core::Tensor& logits) {
    size_t N = logits.numel();
    int threads = round_up(min(N, (size_t) 1024), 32);
    dim3 gridDim(round_up(N, threads) / threads, 1, 1);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    auto d_eps = ctx.tensor({ N }, core::DataType::kFloat);
    auto d_out = ctx.tensor(logits.size(), logits.dtype());
    CURAND_CHECK(curandGenerateUniform(gen, d_eps.data<float>(), N));
    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(apply_gumbel)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            N, d_eps.data<float>(), logits.data<scalar_t>(), d_out.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return d_out;
}

// gridDim (n / 1024, 1, 1),    blockDim(1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(beam_repetition_penalty)(
    int n,
    int vocab_size,
    const float* __restrict__ value,
    const float* __restrict__ presence_penalties,
    const int32_t* __restrict__ tokens,
    const int32_t* __restrict__ batch_id,
    T* __restrict__ logits // (batch_size, vocab_size)
) {
    int pos = threadIdx.x + blockDim.x * blockIdx.x;
    if (pos < n) {
        int32_t token = tokens[pos];
        int32_t batch_idx = batch_id[pos];

        T l = logits[batch_idx * vocab_size + token];
        float presence_penalty = presence_penalties ? presence_penalties[pos] : 0.;
        if (presence_penalty != 0.) {
            l = l - T(presence_penalty);
        } else {
            l = (l < T(0.)) ? (l * T(value[pos])) : (l / T(value[pos]));
        }
        logits[batch_idx * vocab_size + token] = l;
    }
}

// gridDim (n / 1024, 1, 1),    blockDim(1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(scatter_update) (
    int n,
    int stride,
    const float* __restrict__ values,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ batch_ids,
    T* __restrict__ logits, // (batch_size, stride)
    bool add
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (add) {
            logits[batch_ids[i] * stride + indices[i]] += T(values[i]);
        } else {
            logits[batch_ids[i] * stride + indices[i]] = T(values[i]);
        }
    }
}

void beam_repetition_penalty(
    const core::Context& ctx,
    const std::vector<float>& penalty_factor,
    const std::vector<int32_t>& tokens,
    const std::vector<int32_t>& batch_id,
    core::Tensor& logits,
    const std::vector<float>& presence_penalty) {
    BM_ASSERT(tokens.size() == batch_id.size(), "tokens and batch_id must have the same size");
    BM_ASSERT(tokens.size() > 0, "tokens and batch_id must have at least one element");
    auto device = logits.device();
    int n = tokens.size();
    int vocab_size = logits.size(-1);

    auto d_factor = ctx.tensor({ penalty_factor.size() }, core::DataType::kFloat);
    auto d_tokens = ctx.tensor({ tokens.size() }, core::DataType::kInt32);
    auto d_batch = ctx.tensor({ batch_id.size() }, core::DataType::kInt32);
    d_factor.from_buffer(penalty_factor.data());
    d_tokens.from_buffer(tokens.data());
    d_batch.from_buffer(batch_id.data());
    auto d_presence_penalty = presence_penalty.empty() ? Tensor() : ctx.tensor_of(presence_penalty);

    int threads = round_up(min(n, 1024), 32);
    dim3 gridDim(round_up(n, threads) / threads, 1, 1);
    dim3 blockDim(threads, 1, 1);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(beam_repetition_penalty)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            n,
            vocab_size,
            d_factor.data<float>(),
            d_presence_penalty.numel() ? d_presence_penalty.data<float>() : nullptr,
            d_tokens.data<int32_t>(),
            d_batch.data<int32_t>(),
            logits.mutable_data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

void scatter_update(
    const core::Context& ctx,
    const std::vector<float>& values,
    const std::vector<int32_t>& token_ids,  // indices[1]
    const std::vector<int32_t>& batch_ids,  // indices[0]
    core::Tensor& logits,
    bool add) {
    BM_ASSERT(token_ids.size() == batch_ids.size(), "tokens and batch_id must have the same size");
    BM_ASSERT(batch_ids.size() > 0, "tokens and batch_id must have at least one element");
    auto device = logits.device();
    int n = batch_ids.size();
    int vocab_size = logits.size(-1);

    auto d_values = ctx.tensor_of(values);
    auto d_tokens = ctx.tensor_of(token_ids);
    auto d_batches = ctx.tensor_of(batch_ids);

    int threads = round_up_thread(n);
    dim3 gridDim(round_up(n, threads) / threads);
    dim3 blockDim(threads);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(logits.dtype(), {
        BM_KERNEL(scatter_update)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
        n,
        vocab_size,
        d_values.data<float>(),
        d_tokens.data<int32_t>(),
        d_batches.data<int32_t>(),
        logits.mutable_data<scalar_t>(),
        add);
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

std::unordered_map<int, float> calc_repetition_ngram(
    const std::vector<int>& token_ids, float ngram_penalty) {
    if (token_ids.size() == 0)
        return {};
    std::vector<int> next(token_ids.size());
    next[0] = -1;
    for (int i = 0; i + 1 < token_ids.size(); i++) {
        int p = next[i];
        while (p >= 0) {
            if (token_ids[p + 1] == token_ids[i + 1])
                break;
            p = next[p];
        }
        if (token_ids[p + 1] == token_ids[i + 1]) {
            next[i + 1] = p + 1;
        } else {
            next[i + 1] = -1;
        }
    }

    std::vector<int> ngrams(token_ids.size(), 0);
    for (int i = 0; i < next.size(); i++) {
        int ngram = next[i] + 1;
        ngrams[i - ngram] = std::max(ngrams[i - ngram], ngram);
    }
    std::unordered_map<int, float> ret;
    for (int i = 0; i < token_ids.size(); i++) {
        int token = token_ids[i];
        ret[token] = std::max(ret[token], powf(ngram_penalty, ngrams[i] + 1));
    }
    return ret;
}

void apply_beam_repetition_penalty(
    model::ModelContext& ctx,
    const BeamBufferManager<int>& bm,
    const std::vector<int>& hypotheses_last_pos,
    float ngram_penalty,
    float repetition_penalty,
    Tensor* logits_all) {
    vector<float> value_penalty;
    vector<int32_t> tokens_penalty;
    vector<int32_t> batch_penalty;
    vector<int32_t> rev_token_ids;

    for (int i = 0; i < hypotheses_last_pos.size(); i++) {
        rev_token_ids.clear();
        bm.get_hypothesis_tokens(hypotheses_last_pos[i], &rev_token_ids, true);
        auto ngram_map = calc_repetition_ngram(rev_token_ids, ngram_penalty);
        for (const auto& kv : ngram_map) {
            tokens_penalty.push_back(kv.first);
            batch_penalty.push_back(i);
            value_penalty.push_back(kv.second * repetition_penalty);
        }
    }
    if (!tokens_penalty.empty()) {
        beam_repetition_penalty(ctx, value_penalty, tokens_penalty, batch_penalty, *logits_all);
    }
}

void batch_apply_repetition_penalty(
    model::ModelContext& ctx,
    const std::vector<std::vector<std::vector<int>>>& output_sequences, // [batch, hyp_num, tokens]
    float ngram_penalty,
    float repetition_penalty,
    Tensor& logits_all) { // [batch, 0]

    // TODO: rewrite with real batch.

    vector<core::Tensor> logits_chunks = logits_all.chunk();
    for (int b = 0; b < output_sequences.size(); b++) {
        vector<float> value_penalty;
        vector<int32_t> tokens_penalty;
        vector<int32_t> batch_penalty;
        auto output_hyps = output_sequences[b];
        for (int i = 0; i < output_hyps.size(); i++) {
            auto output_sequence = output_hyps[i];
            auto ngram_map = calc_repetition_ngram(output_sequence, ngram_penalty);
            for (const auto& kv : ngram_map) {
                tokens_penalty.push_back(kv.first);
                batch_penalty.push_back(i);
                value_penalty.push_back(kv.second * repetition_penalty);
            }
        }
        if (!tokens_penalty.empty()) {
            beam_repetition_penalty(
                ctx, value_penalty, tokens_penalty, batch_penalty, logits_chunks[b]);
        }
    }
}

void init_curand_gen(const core::Context& ctx, curandGenerator_t& gen, int seed) {
    CURAND_CHECK(curandSetStream(gen, ctx.current_stream()->ptr));
    CURAND_CHECK(curandSetGeneratorOffset(gen, 0));
    CURAND_CHECK(curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_BEST));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
}
}
