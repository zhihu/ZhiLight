#pragma once

#include <bmengine/core/core.h>

namespace nn {
namespace gptq {

using namespace bmengine;

core::Tensor gptq_gemm(
    const core::Context& ctx,
    core::Tensor a,
    core::Tensor b_q_weight,
    core::Tensor b_gptq_qzeros,
    core::Tensor b_gptq_scales,
    core::Tensor b_g_idx,
    bool use_exllama,
    int group_size,
    int size_n1,
    int size_n2
);

void gptq_shuffle(
    const core::Context& ctx,
    core::Tensor& q_weight,
    core::Tensor q_perm
);
void un_shuffle(
    const core::Context& ctx,
    core::Tensor& input
);

void increase_zero(
    const core::Context& ctx,
    core::Tensor& input
);

void subtract8(
    const core::Context& ctx,
    core::Tensor& input
);

core::Tensor shuffle_awq(
    const core::Context& ctx,
    core::Tensor& input,
    bool use_exllama
);
void reconstruct_exllama(
    const uint32_t* b_q_weight,
    const uint32_t* b_gptq_qzeros,
    const half* b_gptq_scales,
    const int* b_q_perm,
    half* out,
    int height,
    int width,
    int groups,
    const cudaStream_t stream,
    int size_n1,
    int size_n2
);

void reconstruct_gptq(
    const uint32_t* b_q_weight,
    const uint32_t* b_gptq_qzeros,
    const half* b_gptq_scales,
    const int* b_g_idx,
    half* out,
    int height, // k
    int width, // n
    int groups,
    const cudaStream_t stream
);

core::Tensor dequant_k_major(
    const core::Context& ctx,
    const core::Tensor& q_weight, // (N, K / 8)
    const core::Tensor& qzeros,   // (N, K / group_size)
    const core::Tensor& scales,   // (N, K / group_size)
    int out_type=0
);

core::Tensor gptq_gemm_k_major(
    const core::Context& ctx,
    const core::Tensor& a,        // (M, K)
    const core::Tensor& q_weight, // (N, K / 8)
    const core::Tensor& qzeros,   // (N, K / group_size)
    const core::Tensor& scales,   // (N, K / group_size)
    const core::Tensor& q_perm,   // (K)
    const core::Tensor& rev_perm, // (K)
    const core::Tensor* bias,
    bool sym,
    bool cache_only = false,
    core::Tensor* output = nullptr,
    const core::Tensor* precomputed_w8 = nullptr
);

core::Tensor gemm_fuse_gate_in(
    const core::Context& ctx,
    const core::Tensor& a,         // (M, K)
    const core::Tensor& q_weight1, // (N, K / 8)
    const core::Tensor& qzeros1,   // (N, K / group_size)
    const core::Tensor& scales1,   // (N, K / group_size)
    const core::Tensor& rev_perm1, // (K)
    const core::Tensor& q_weight2, // (N, K / 8)
    const core::Tensor& qzeros2,   // (N, K / group_size)
    const core::Tensor& scales2,   // (N, K / group_size)
    const core::Tensor& rev_perm2, // (K)
    bool sym
);

core::Tensor gemm_moe_up(
    const core::Context& ctx,
    const core::Tensor& a,          // (M, K)
    const core::Tensor& q_weight1,  // (EXP, N, K / 8)
    const core::Tensor& qzeros1,    // (EXP, N, K / group_size)
    const core::Tensor& scales1,    // (EXP, N, K / group_size)
    const core::Tensor& rev_perm1,  // (EXP, K)
    const core::Tensor& q_weight2,  // (EXP, N, K / 8)
    const core::Tensor& qzeros2,    // (EXP, N, K / group_size)
    const core::Tensor& scales2,    // (EXP, N, K / group_size)
    const core::Tensor& rev_perm2,  // (EXP, K)
    bool sym,
    const core::Tensor& expert_ids, // (M, top_k)
    int n_shared_expert = 0,
    bool exp_parallel = false
);
core::Tensor gemm_moe_down(
    const core::Context& ctx,
    const core::Tensor& topk_a,         // (M, TOP_K, K)
    const core::Tensor& q_weight,       // (NUM_EXP, N, K / 8)
    const core::Tensor& qzeros,         // (NUM_EXP, N, K / group_size)
    const core::Tensor& scales,         // (NUM_EXP, N, K / group_size)
    const core::Tensor& expert_ids,     // (M, TOP_K)
    const core::Tensor& expert_weights, // (M, TOP_K)
    bool sym,
    int n_shared_expert = 0,
    bool exp_parallel = false,
    core::Tensor* output = nullptr
);

// int32 => int8
core::Tensor q4_to_q8(
    const core::Context& ctx,
    const core::Tensor& input
);

// int8 => int8
core::Tensor q8_to_q4(
    const core::Context& ctx,
    const core::Tensor& input
);

core::Tensor int32_to_int16(
    const core::Context& ctx,
    const core::Tensor& input
);

core::Tensor reverse_perm(
    const core::Context& ctx,
    const core::Tensor& input
);

core::Tensor permute_input(
    const core::Context& ctx,
    const core::Tensor& input,
    const core::Tensor& q_perm
);

core::Tensor w4_to_int8(
    const core::Context& ctx,
    const core::Tensor& permuted_weight, // (N, K / 8)
    const core::Tensor& qzeros,          // (N, K / group_size)
    const core::Tensor* q_scale
);

core::Tensor gemm_groupwise_int8(
    const core::Context& ctx,
    const core::Tensor& a,        // (M, K) int8
    const core::Tensor& b,        // (N, K) int8
    const core::Tensor& q_scales, // (N, K / group_size) int8
    const core::Tensor& s_scales, // (N) half
    const core::Tensor& a_scales, // (M) float
    const core::Tensor* bias
);
}
}