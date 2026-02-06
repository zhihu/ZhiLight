/*
 * Adapted from https://github.com/Dao-AILab/flash-attention/blob/v2.7.0.post2/csrc/flash_attn/flash_api.cpp
 */
#include <vector>
#include <iostream>
#include "flash_api.h"
#include "flash_decoding.h"
#include "bmengine/functions/transpose.h"

namespace nn {

using std::vector;
using bmengine::core::DataType;
using bmengine::functions::transpose_2_1;
#ifdef USE_FLASH_NS
using namespace flash;
#endif

FlashDecoding::FlashDecoding(const Context& ctx) {
    BM_CUDART_ASSERT(cudaGetDeviceProperties(&dprops, ctx.active_device()));
    params = new Flash_fwd_params();
}

FlashDecoding::~FlashDecoding() {
    if (params != nullptr) {
        delete params;
    }
}

FlashDecoding::Tensor FlashDecoding::forward(
    const Context& ctx,
    Tensor& query,                // (batch, len_q,  num_heads, dim_head), or (total_q, num_heads, dim_head) for varlen
    const Tensor& key,            // (batch, len_kv, num_kv_heads, dim_head), or (total_kv, num_kv_heads, dim_head) for varlen
    const Tensor& value,          // (batch, len_kv, num_kv_heads, dim_head), or (total_kv, num_kv_heads, dim_head) for varlen
    Tensor *fa_out,               // (batch, len_q,  num_heads, dim_head), or (total_q, num_heads, dim_head) for varlen
    const Tensor *cu_seqlens_q,   // (batch+1), int32, only for varlen
    const Tensor *cu_seqlens_k,   // (batch+1), int32, only for varlen
    int max_seqlen_q,             // only for varlen
    int max_seqlen_k,             // only for varlen
    bool chunked_addrs,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float softmax_scale) {
    // varlen fwd
    if (query.ndim() == 3) {
        BM_ASSERT(cu_seqlens_q != nullptr
                && cu_seqlens_k != nullptr
                && max_seqlen_q > 0
                && max_seqlen_k > 0,
                "Invalid arguments for FlashDecoding varlen fwd")
        return mha_varlen_fwd(ctx, query, key, value, fa_out, *cu_seqlens_q, *cu_seqlens_k,
                            chunked_addrs, nullptr, nullptr, nullptr, nullptr, max_seqlen_q,
                            max_seqlen_k, 0.f, softmax_scale, false, is_causal, window_size_left,
                            window_size_right, 0.f, false);
    // fixedlen fwd
    } else if (query.ndim() == 4) {
        return mha_fwd(ctx, query, key, value, fa_out, nullptr, 0.f, softmax_scale,
                    is_causal, window_size_left, window_size_right, 0.f, false);
    }
    BM_EXCEPTION("FlashDecoding only supports 3D and 4D tensors");
}

FlashDecoding::Tensor FlashDecoding::mha_fwd(
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
    const bool return_softmax) {

    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
    bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
    bool is_sm110 = dprops.major == 11 && dprops.minor == 0;
    BM_ASSERT(is_sm90 || is_sm8x || is_sm110, "FlashAttention only supports Ampere GPUs or newer.");
    // We will support Turing in the near future
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

    auto q_dtype = q.dtype();
    BM_ASSERT(q_dtype == DataType::kHalf || q_dtype == DataType::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    if (q_dtype == DataType::kBFloat16) {
        BM_ASSERT(is_sm90 || is_sm8x || is_sm110, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    BM_ASSERT(k.dtype() == q_dtype, "query and key must have the same dtype");
    BM_ASSERT(v.dtype() == q_dtype, "query and value must have the same dtype");

    BM_ASSERT(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    BM_ASSERT(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    BM_ASSERT(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.size();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    BM_ASSERT(batch_size > 0, "batch size must be positive");
    BM_ASSERT(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    BM_ASSERT(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    BM_ASSERT(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (softcap > 0.f) { BM_ASSERT(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && alibi_slopes == nullptr) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size % 8 == 0 && alibi_slopes == nullptr;
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = transpose_2_1(ctx, q.view({batch_size, num_heads_k, ngroups, head_size}));
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    BM_ASSERT(q.shape() == std::vector<size_t>({batch_size, seqlen_q, num_heads, head_size}), "mismatch q shape");
    BM_ASSERT(k.shape() == std::vector<size_t>({batch_size, seqlen_k, num_heads_k, head_size}), "mismatch k shape");
    BM_ASSERT(v.shape() == std::vector<size_t>({batch_size, seqlen_k, num_heads_k, head_size}), "mismatch v shape");

    Tensor out;
    if (out_ != nullptr) {
        BM_ASSERT(out_->dtype() == q_dtype, "Output must have the same dtype as inputs");
        BM_ASSERT(out_->stride(-1) == 1, "Output tensor must have contiguous last dimension");
        BM_ASSERT(out_->shape() == std::vector<size_t>({batch_size, sizes[1], sizes[2], head_size}), "mismatch out shape");
        if (seqlenq_ngroups_swapped) {
            //out = transpose_2_1(ctx, out.view({batch_size, num_heads_k, ngroups, head_size}));
            out = ctx.tensor(q.size(), q.dtype());
        } else {
            out = out_->view(q.size());
        }
    } else {
        out = ctx.tensor(q.size(), q_dtype);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = head_size <= 192 ? round_multiple(head_size, 32) : 256;
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    auto softmax_lse = ctx.tensor({batch_size, num_heads, seqlen_q}, DataType::kFloat);

    set_params_fprop(batch_size,
                    seqlen_q, seqlen_k,
                    seqlen_q_rounded, seqlen_k_rounded,
                    num_heads, num_heads_k,
                    head_size, head_size_rounded,
                    q, k, v, out,
                    /*cu_seqlens_q_d=*/nullptr,
                    /*cu_seqlens_k_d=*/nullptr,
                    /*seqused_k=*/nullptr,
                    nullptr,
                    softmax_lse.data(),
                    p_dropout,
                    softmax_scale,
                    window_size_left,
                    window_size_right,
                    softcap);

    // Keep references to these tensors to extend their lifetime
    Tensor softmax_lse_accum, out_accum;
    std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
        ctx, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
        head_size_rounded, p_dropout, /*num_splits*/ 0);

    set_params_alibi(alibi_slopes, batch_size, num_heads);

    run_mha_fwd(*params, ctx.current_stream()->ptr);

    if (seqlenq_ngroups_swapped) {
        vector<size_t> before_size = {batch_size, seqlen_q, num_heads_k, head_size};
        vector<size_t> tmp_size = {batch_size, num_heads_k, seqlen_q, head_size};
        vector<size_t> after_size = {batch_size, 1, seqlen_q * num_heads_k, head_size};
        Tensor tmp = out_ == nullptr ? ctx.tensor(tmp_size, q_dtype) : out_->view(tmp_size);
        out = transpose_2_1(ctx, out.view(before_size), &tmp).view(after_size);
    }
    return out;
}

FlashDecoding::Tensor FlashDecoding::mha_varlen_fwd(const Context &ctx,
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
    const bool return_softmax) {

    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
    bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
    bool is_sm110 = dprops.major == 11 && dprops.minor == 0;
    BM_ASSERT(is_sm90 || is_sm8x || is_sm110, "FlashAttention only supports Ampere GPUs or newer.");
    auto q_dtype = q.dtype();
    BM_ASSERT(q_dtype == DataType::kHalf || q_dtype == DataType::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    if (q_dtype == DataType::kBFloat16) {
        BM_ASSERT(is_sm90 || is_sm8x || is_sm110, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    BM_ASSERT(k.dtype() == q_dtype, "query and key must have the same dtype");
    BM_ASSERT(v.dtype() == q_dtype, "query and value must have the same dtype");
    BM_ASSERT(cu_seqlens_q.dtype() == DataType::kInt32, "cu_seqlens_q must have dtype int32");
    BM_ASSERT(cu_seqlens_k.dtype() == DataType::kInt32, "cu_seqlens_k must have dtype int32");

    const bool paged_KV = block_table != nullptr;
    if (paged_KV) {
        TORCH_CHECK(block_table->dtype() == DataType::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table->stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    BM_ASSERT(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    BM_ASSERT(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    BM_ASSERT(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.size();

    const int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int num_heads_k = paged_KV ? k.size(2) : k.size(1);

    if (softcap > 0.f) { BM_ASSERT(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table->size(1);
    const int num_blocks = !paged_KV ? 0 : k.size(0);
    const int page_block_size = !paged_KV ? 1 : k.size(1);
    BM_ASSERT(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");

    if (max_seqlen_q == 1 && alibi_slopes == nullptr) { is_causal = false; }  // causal=true is the same as causal=false in this case
    if (is_causal) { window_size_right = 0; }

    void *cu_seqlens_q_d = cu_seqlens_q.data();

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = max_seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size % 8 == 0 && alibi_slopes == nullptr;
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = transpose_2_1(ctx, q.view({batch_size, num_heads_k, ngroups, head_size}))
            .view({batch_size * ngroups, num_heads_k, head_size});
        max_seqlen_q = ngroups;
        num_heads = num_heads_k;
        cu_seqlens_q_d = nullptr;
    }

    const int total_q = q.size()[0];

    BM_ASSERT(batch_size > 0, "batch size must be positive");
    BM_ASSERT(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    BM_ASSERT(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    BM_ASSERT(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    BM_ASSERT(q.shape() == vector<size_t>({total_q, num_heads, head_size}), "mismatch q shape");
    if (!paged_KV) {
        const int total_k = k.size(0);
        BM_ASSERT(k.shape() == vector<size_t>({total_k, num_heads_k, head_size}), "mismatch k shape");
        BM_ASSERT(v.shape() == vector<size_t>({total_k, num_heads_k, head_size}), "mismatch v shape");
    } else {
        BM_ASSERT(k.shape() == vector<size_t>({num_blocks, page_block_size, num_heads_k, head_size}), "mismatch k shape");
        BM_ASSERT(v.shape() == vector<size_t>({num_blocks, page_block_size, num_heads_k, head_size}), "mismatch v shape");
        BM_ASSERT(block_table->shape() == vector<size_t>({batch_size, max_num_blocks_per_seq}), "mismatch block_table shape");
    }

    BM_ASSERT(cu_seqlens_q.shape() == vector<size_t>({batch_size + 1}), "mismatch cu_seqlens_q shape");
    BM_ASSERT(cu_seqlens_k.shape() == vector<size_t>({batch_size + 1}), "mismatch cu_seqlens_k shape");
    if (seqused_k){
        BM_ASSERT(seqused_k->dtype() == DataType::kInt32, "seqused_k must have dtype int32");
        BM_ASSERT(seqused_k->shape() == vector<size_t>({batch_size}), "mismatch seqused_k shape");
    }

    Tensor out;
    if (out_ != nullptr) {
        BM_ASSERT(out_->dtype() == q_dtype, "Output must have the same dtype as inputs");
        BM_ASSERT(out_->stride(-1) == 1, "Output tensor must have contiguous last dimension");
        BM_ASSERT(out_->shape() == vector<size_t>({sizes[0], sizes[1], head_size}), "mismatch out shape");
        if (seqlenq_ngroups_swapped) {
            out = ctx.tensor(q.size(), q_dtype);
        } else {
            out = out_->view(q.size());
        }
    } else {
        out = ctx.tensor(q.size(), q_dtype);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = head_size <= 192 ? round_multiple(head_size, 32) : 256;
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    auto softmax_lse = ctx.tensor({num_heads, total_q}, DataType::kFloat);

    set_params_fprop(batch_size,
                    max_seqlen_q, max_seqlen_k,
                    seqlen_q_rounded, seqlen_k_rounded,
                    num_heads, num_heads_k,
                    head_size, head_size_rounded,
                    q, k, v, out,
                    cu_seqlens_q_d,
                    cu_seqlens_k.data(),
                    seqused_k != nullptr ? seqused_k->data() : nullptr,
                    nullptr,
                    softmax_lse.data(),
                    p_dropout,
                    softmax_scale,
                    window_size_left,
                    window_size_right,
                    softcap,
                    chunked_addrs,
                    seqlenq_ngroups_swapped,
                    /*unpadded_lse*/true);
    params->total_q = total_q;

    if (paged_KV) {
        params->block_table = reinterpret_cast<int *>(block_table->data());
        params->block_table_batch_stride = block_table->stride(0);
        params->k_batch_stride = k.stride(0);
        params->v_batch_stride = v.stride(0);
    }
    params->page_block_size = page_block_size;
    // Keep references to these tensors to extend their lifetime
    Tensor softmax_lse_accum, out_accum;
    if (seqlenq_ngroups_swapped) {
        // Only apply split-k for decoding
        std::tie(softmax_lse_accum, out_accum) =
            set_params_splitkv(ctx, batch_size, num_heads, head_size,
                            max_seqlen_k, max_seqlen_q, head_size_rounded,
                            p_dropout, /*num_splits*/ 0);
    }

    if (leftpad_k != nullptr) {
        BM_ASSERT(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        BM_ASSERT(leftpad_k->dtype() == DataType::kInt32, "leftpad_k must have dtype int32");
        BM_ASSERT(leftpad_k->shape() == vector<size_t>({batch_size}), "mismatch leftpad shape");
        params->leftpad_k = static_cast<int *>(leftpad_k->data());
    }

    set_params_alibi(alibi_slopes, batch_size, num_heads);

    run_mha_fwd(*params, ctx.current_stream()->ptr, paged_KV);

    if (seqlenq_ngroups_swapped) {
        vector<size_t> before_size = {batch_size, max_seqlen_q, num_heads_k, head_size};
        vector<size_t> tmp_size = {batch_size, num_heads_k, max_seqlen_q, head_size};
        vector<size_t> after_size = {batch_size, max_seqlen_q * num_heads_k, head_size};
        Tensor tmp = out_ == nullptr ? ctx.tensor(tmp_size, q_dtype) : out_->view(tmp_size);
        out = transpose_2_1(ctx, out.view(before_size), &tmp).view(after_size);
    }

    return out;
}

void FlashDecoding::set_params_fprop(
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
    bool chunked_addrs,
    bool seqlenq_ngroups_swapped,
    const bool unpadded_lse) {

    // Reset the parameters
    reset_params();
    params->is_bf16 = q.dtype() == DataType::kBFloat16;

    // Set the pointers and strides.
    params->q_ptr = q.data();
    params->k_ptr = k.data();
    params->v_ptr = v.data();
    // All stride are in elements, not bytes.
    params->q_row_stride = q.stride(-3);
    params->k_row_stride = k.stride(-3);
    params->v_row_stride = v.stride(-3);
    params->q_head_stride = q.stride(-2);
    params->k_head_stride = k.stride(-2);
    params->v_head_stride = v.stride(-2);
    params->o_ptr = out.data();
    params->o_row_stride = out.stride(-3);
    params->o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params->q_batch_stride = q.stride(0);
        params->k_batch_stride = k.stride(0);
        params->v_batch_stride = v.stride(0);
        params->o_batch_stride = out.stride(0);
        if (seqlenq_ngroups_swapped) {
            params->q_batch_stride *= seqlen_q;
            params->o_batch_stride *= seqlen_q;
        }
    }

    params->cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params->cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params->seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params->p_ptr = p_d;

    // Softmax sum
    params->softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params->b = b;
    params->h = h;
    params->h_k = h_k;
    params->h_h_k_ratio = h / h_k;
    params->seqlen_q = seqlen_q;
    params->seqlen_k = seqlen_k;
    params->seqlen_q_rounded = seqlen_q_rounded;
    params->seqlen_k_rounded = seqlen_k_rounded;
    params->d = d;
    params->d_rounded = d_rounded;

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
        BM_ASSERT(softcap <= 0.0, "This flash attention build does not support softcap.");
    #endif
    if (softmax_scale == 0.0) {
        softmax_scale = 1 / sqrtf(d);
    }
    if (softcap > 0.0) {
        params->softcap = softmax_scale / softcap;
        params->scale_softmax = softcap;
        params->scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params->softcap = 0.0;
        params->scale_softmax = softmax_scale;
        params->scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params->p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params->p_dropout_in_uint = uint32_t(std::floor(params->p_dropout * 4294967295.0));
    // params->p_dropout_in_uint16_t = uint16_t(std::floor(params->p_dropout * 65535.0));
    params->p_dropout_in_uint8_t = uint8_t(std::floor(params->p_dropout * 255.0));
    params->rp_dropout = 1.f / params->p_dropout;
    params->scale_softmax_rp_dropout = params->rp_dropout * params->scale_softmax;
    BM_ASSERT(p_dropout < 1.f, "Dropout need less than 1.f.");
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        BM_ASSERT(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params->is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params->window_size_left = window_size_left;
    params->window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        BM_ASSERT(params->is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    params->is_seqlens_k_cumulative = true;

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        BM_ASSERT(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif

    // TODO: tricky, reuse a variable when fwd no used for compatible
    params->p_dropout_in_uint8_t = chunked_addrs;
    params->unpadded_lse = unpadded_lse;
    params->seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

std::tuple<FlashDecoding::Tensor, FlashDecoding::Tensor> FlashDecoding::set_params_splitkv(
    const Context &ctx,
    const int batch_size,
    const int num_heads,
    const int head_size,
    const int max_seqlen_k,
    const int max_seqlen_q,
    const int head_size_rounded,
    const float p_dropout,
    const int num_splits) {

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params->num_splits = num_splits;
    Tensor softmax_lse_accum;
    Tensor out_accum;

    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) {
            // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
            params->num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, dprops.multiProcessorCount * 2, num_n_blocks, 128);
        }
        if (params->num_splits > 1) {
            // TODO(wjj): if need zero???
            softmax_lse_accum = ctx.tensor({params->num_splits, batch_size, num_heads, max_seqlen_q}, DataType::kFloat);
            out_accum = ctx.tensor({params->num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, DataType::kFloat);
            params->softmax_lseaccum_ptr = softmax_lse_accum.data();
            params->oaccum_ptr = out_accum.data();
        }
        BM_ASSERT(params->num_splits <= 128, "num_splits > 128 not supported");
    }

    return std::make_tuple(softmax_lse_accum, out_accum);
}

void FlashDecoding::set_params_alibi(
    Tensor *alibi_slopes,
    int batch_size,
    int num_heads){
#ifdef FLASHATTENTION_DISABLE_ALIBI
    BM_ASSERT(!alibi_slopes, "This flash attention build does not support alibi.");
    params->alibi_slopes_ptr = nullptr;
#else
    if (alibi_slopes != nullptr) {
        BM_ASSERT(alibi_slopes->dtype() == DataType::kFloat, "ALiBi slopes must have dtype fp32");
        BM_ASSERT(alibi_slopes->stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        params->alibi_slopes_ptr = alibi_slopes->data();
        params->alibi_slopes_batch_stride = alibi_slopes->ndim() == 2 ? alibi_slopes->stride(0) : 0;
    } else {
        params->alibi_slopes_ptr = nullptr;
    }
#endif
}

// Deprecated: this code is only used when turn off dynamic batch, it will be removed in the future.
FlashDecoding::Tensor FlashDecoding::compact_kv_fwd(
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
    float scale_softmax) {

    BM_ASSERT_EQ(query.ndim(), 4, "Wrong query dim");
    uint32_t batch_size = seqlens_kv.numel() ? seqlens_kv.size(0) : 1;
    size_t len_q = query.size(1);
    size_t len_buf = key_cache.size(1);
    size_t num_heads = query.size(2);
    size_t num_kv_heads = key_cache.size(2);
    size_t dim_head = key_cache.size(3);
    size_t num_head_groups = num_heads / num_kv_heads;
    auto dtype = query.dtype();
    const bool paged_KV = block_table != nullptr;
    Tensor h_q = query.view({ batch_size, len_q, num_heads, dim_head });

    Tensor attn_res;
    unsigned int num_heads_g = num_heads;
    int seqlenq_ngroups_swapped;
    {
        const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table->size(1);
        const int num_blocks = !paged_KV ? 0 : key_cache.size(0);
        const int page_block_size = !paged_KV ? 1 : key_cache.size(1);
        const int seqlen_k = !paged_KV ? len_buf : max_num_blocks_per_seq * page_block_size;
        const int batch_size_c = !paged_KV ? key_cache.size(0) : batch_size;

        int window_size_right = -1;
        int window_size_left = -1;
        bool is_causal = true;
        if (len_q == 1) {
            is_causal = false;
        } // causal=true is the same as causal=false in this case
        if (is_causal) {
            window_size_right = 0;
        }
        /*
            * single-token query optimization:
            * 1. treat MQA groups as query.
            * 2. sampling-parallism(num_results > 1) as batch dim to work with FA's mask
            * memchanism.
            */
        seqlenq_ngroups_swapped = len_q == 1 && num_heads > num_kv_heads && window_size_left < 0
                                && window_size_right < 0 && dim_head % 8 == 0;
        if (seqlenq_ngroups_swapped) {
            h_q = transpose_2_1(
                ctx, h_q.view({ batch_size, num_kv_heads, num_head_groups, dim_head }));
            len_q = num_head_groups;
            num_heads_g = num_kv_heads;
        }
        attn_res = out.numel() ?
                out.view({ batch_size, len_q, num_heads_g, dim_head }) :
                ctx.tensor({ batch_size, len_q, num_heads_g, dim_head }, dtype);
        auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
        const int head_size = round_multiple(dim_head, 8);
        const int head_size_rounded = round_multiple(head_size, 32);
        const int seqlen_q_rounded = round_multiple(len_q, 128);
        const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

        reset_params();
        params->is_bf16 = (dtype == DataType::kBFloat16 ? true : false);
        params->q_ptr = h_q.data();
        params->k_ptr = key_cache.data();
        params->v_ptr = val_cache.data();

        params->q_row_stride = h_q.stride(1);
        params->k_row_stride = key_cache.stride(1);
        params->v_row_stride = val_cache.stride(1);

        params->q_head_stride = h_q.stride(2);
        params->k_head_stride = key_cache.stride(2);
        params->v_head_stride = val_cache.stride(2);

        params->o_ptr = attn_res.data();
        params->o_row_stride = attn_res.stride(1);
        params->o_head_stride = attn_res.stride(2);

        params->q_batch_stride = h_q.stride(0);
        params->k_batch_stride = key_cache.stride(0);
        params->v_batch_stride = val_cache.stride(0);
        params->o_batch_stride = attn_res.stride(0);

        if (seqlens_q.numel() > 0) {
            params->cu_seqlens_q = static_cast<int*>(seqlens_q.data());
        } else
            params->cu_seqlens_q = nullptr;
        params->p_ptr = nullptr;
        if (new_k.numel() > 0) {
            params->seqlen_knew = new_k.size(1);
            params->knew_ptr = new_k.data();
            params->vnew_ptr = new_v.data();
            params->knew_batch_stride = new_k.stride(0);
            params->vnew_batch_stride = new_v.stride(0);
            params->knew_row_stride = new_k.stride(1);
            params->vnew_row_stride = new_v.stride(1);
            params->knew_head_stride = new_k.stride(2);
            params->vnew_head_stride = new_v.stride(2);
        }

        params->b = batch_size;
        params->h = num_heads_g;
        params->h_k = num_kv_heads;
        params->h_h_k_ratio = num_heads_g / num_kv_heads;
        params->seqlen_q = len_q;
        params->seqlen_k = seqlen_k;
        params->seqlen_q_rounded = seqlen_q_rounded;
        params->seqlen_k_rounded = seqlen_k_rounded;
        params->d = head_size;
        params->d_rounded = head_size_rounded;
        params->softcap = 0.0;
        params->scale_softmax = scale_softmax == 0 ? 1 / sqrtf(dim_head) : scale_softmax;
        params->scale_softmax_log2 = params->scale_softmax * M_LOG2E;
        params->p_dropout = 1.f;
        params->p_dropout_in_uint8_t = uint8_t(std::floor(params->p_dropout * 255.0));
        params->rp_dropout = 1.f / params->p_dropout;
        params->scale_softmax_rp_dropout = params->rp_dropout * params->scale_softmax;
        // casual mask in SWA form, no right window and maximized left window.
        params->is_causal = window_size_left < 0 && window_size_right == 0;
        if (window_size_left < 0 && window_size_right >= 0) {
            window_size_left = seqlen_k;
        }
        if (window_size_left >= 0 && window_size_right < 0) {
            window_size_right = seqlen_k;
        }
        params->window_size_left = window_size_left;
        params->window_size_right = window_size_right;

        if (paged_KV) {
            params->block_table = block_table->data<int>();
            params->block_table_batch_stride = block_table->stride(0);
        }
        params->page_block_size = page_block_size;

        auto softmax_lse =
            ctx.tensor({ batch_size, num_heads_g, len_q }, DataType::kFloat);
        params->softmax_lse_ptr = softmax_lse.data();
        params->cu_seqlens_k = seqlens_kv.numel() ? seqlens_kv.data<int>() : nullptr;
        params->is_seqlens_k_cumulative = false;
        params->unpadded_lse = false;
        params->rotary_dim = 0;

        const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
        const int num_n_blocks = (seqlen_k + block_n - 1) / block_n;
        // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
        // In any case we don't expect seqlen_q to be larger than 64 for inference.
        const int num_m_blocks = (len_q + 64 - 1) / 64;
        params->num_splits = num_splits_heuristic(
            batch_size * num_heads_g * num_m_blocks, dprops.multiProcessorCount * 2, num_n_blocks, 128);
        if (params->num_splits > 1) {
            Tensor softmax_lse_accum = ctx.tensor(
                { (size_t) params->num_splits, batch_size, num_heads_g, len_q },
                DataType::kFloat);
            Tensor out_accum = ctx.tensor(
                { (size_t) params->num_splits,
                    batch_size,
                    num_heads_g,
                    len_q,
                    size_t(head_size_rounded) },
                DataType::kFloat);
            params->softmax_lseaccum_ptr = softmax_lse_accum.data();
            params->oaccum_ptr = out_accum.data();
        }
        params->alibi_slopes_ptr = nullptr;
        run_mha_fwd(*params, ctx.current_stream()->ptr, true);
    }

    if (seqlenq_ngroups_swapped) {
        attn_res = transpose_2_1(ctx, attn_res);
        attn_res = attn_res.view({ batch_size, 1, num_heads * dim_head }); // (len_q, dim_model)
    } else {
        attn_res =
            attn_res.view({ batch_size, len_q, num_heads * dim_head }); // (len_q, dim_model)
    }
    return attn_res;
}

inline void FlashDecoding::reset_params() {
    memset(params, 0, sizeof(Flash_fwd_params));
}

}