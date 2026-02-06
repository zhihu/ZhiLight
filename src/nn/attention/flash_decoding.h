#pragma once
#include "bmengine/core/core.h"

#ifdef USE_FLASH_NS
namespace flash {
    struct Flash_fwd_params;
}
using namespace flash;
#else
struct Flash_fwd_params;
#endif

namespace nn {
class FlashDecoding : public bmengine::core::Layer {
    BM_LAYER_DEF_PUBLIC(FlashDecoding);
    using Tensor = bmengine::core::Tensor;
    using Context = bmengine::core::Context;

public:

    FlashDecoding(const Context& ctx);
    ~FlashDecoding();

    Tensor forward(
        const Context &ctx,
        Tensor &query,                          // (batch, len_q,  num_heads, dim_head), or (total_q, num_heads, dim_head) for varlen
        const Tensor &key,                      // (batch, len_kv, num_kv_heads, dim_head), or (total_kv, num_kv_heads, dim_head) for varlen
        const Tensor &value,                    // (batch, len_kv, num_kv_heads, dim_head), or (total_kv, num_kv_heads, dim_head) for varlen
        Tensor *fa_out = nullptr,               // (batch, len_q,  num_heads, dim_head), or (total_q, num_heads, dim_head) for varlen
        const Tensor *cu_seqlens_q = nullptr,   // (batch+1), int32, only for varlen
        const Tensor *cu_seqlens_k = nullptr,   // (batch+1), int32, only for varlen
        int max_seqlen_q = 0,                   // only for varlen
        int max_seqlen_k = 0,                   // only for varlen
        bool chunked_addrs = false,
        bool is_causal = true,
        int window_size_left = -1,
        int window_size_right = -1,
        float softmax_scale = 0.f);

    // Deprecated: only use when turn off dynamic batch
    Tensor compact_kv_fwd(
        const Context& ctx,
        const Tensor& query,      // (batch, len_q, num_heads, dim_head)
        const Tensor& key_cache,  // (batch, len_buf, num_kv_heads, dim_head)
        const Tensor& val_cache,  // (batch, len_buf, num_kv_heads, dim_head)
        const Tensor& seqlens_q,  // (batch, len_q,)    int32
        const Tensor& seqlens_kv, // (batch, len_q,)    int32
        const Tensor& new_k,
        const Tensor& new_v,
        const Tensor* block_table,
        Tensor out,               // (batch, len_q, num_heads, dim_head)
        float scale_softmax = 0);

private:

    Tensor mha_fwd(
        const Context &ctx,
        Tensor &q,               // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        Tensor *out_,            // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        Tensor *alibi_slopes,    // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax);

    Tensor mha_varlen_fwd(
        const Context &ctx,
        Tensor &q,                   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const Tensor &k,             // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        const Tensor &v,             // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        Tensor *out_,                // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const Tensor &cu_seqlens_q,  // b+1
        const Tensor &cu_seqlens_k,  // b+1
        bool chunked_addrs,
        Tensor *seqused_k,           // b. If given, only this many elements of each batch element's keys are used.
        Tensor *leftpad_k,           // batch_size
        Tensor *block_table,         // batch_size x max_num_blocks_per_seq
        Tensor *alibi_slopes,        // num_heads or b x num_heads
        int max_seqlen_q,
        const int max_seqlen_k,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax);

    void set_params_fprop(
        // sizes
        const size_t b,
        const size_t seqlen_q,
        const size_t seqlen_k,
        const size_t seqlen_q_rounded,
        const size_t seqlen_k_rounded,
        const size_t h,
        const size_t h_k,
        const size_t d,
        const size_t d_rounded,
        // device pointers
        const Tensor &q,
        const Tensor &k,
        const Tensor &v,
        Tensor &out,
        void *cu_seqlens_q_d,
        void *cu_seqlens_k_d,
        void *seqused_k,
        void *p_d,
        void *softmax_lse_d,
        float p_dropout,
        float softmax_scale,
        int window_size_left,
        int window_size_right,
        const float softcap,
        bool chunked_addrs = false,
        bool seqlenq_ngroups_swapped=false,
        const bool unpadded_lse=false);

    std::tuple<Tensor, Tensor> set_params_splitkv(
        const Context &ctx,
        const int batch_size,
        const int num_heads,
        const int head_size,
        const int max_seqlen_k,
        const int max_seqlen_q,
        const int head_size_rounded,
        const float p_dropout,
        const int num_splits);

    void set_params_alibi(
        Tensor *alibi_slopes,
        int batch_size,
        int num_heads);

    inline void reset_params();

    cudaDeviceProp dprops;
    Flash_fwd_params *params;
};
}