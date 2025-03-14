// Adapted from https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/flash_api.cpp
// which adapted from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
#include "ds_flash_mla_api.h"
#include "flash_mla.h"
#include "bmengine/functions/init.h"
#include "bmengine/logger/std_log_op.hpp"

#define CHECK_DEVICE(x) {}
#define CHECK_SHAPE(x, ...) BM_ASSERT_EQ(x.shape(), std::vector<size_t>({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) {}

namespace ds {

template<class T>
static inline T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

std::tuple<core::Tensor, core::Tensor> get_mla_metadata(
    const core::Context& ctx,
    const core::Tensor& seqlens_k,
    const size_t num_heads_per_head_k,
    const size_t num_heads_k
) {
    // This should match the logic in the MLA kernel.
    static constexpr size_t block_size_m = 64;
    static constexpr size_t block_size_n = 64;
    static constexpr size_t fixed_overhead_num_blocks = 5;

    BM_ASSERT_EQ(seqlens_k.dtype(), core::DataType::kInt32, "seqlens_k is not int");

    int batch_size = seqlens_k.size(0);
    int *seqlens_k_ptr = seqlens_k.data<int>();

    int sm_count = ctx.get_mp_count();
    int num_sm_parts = sm_count / num_heads_k / ceil_div(num_heads_per_head_k, block_size_m);

    auto tile_scheduler_metadata = ctx.tensor({num_sm_parts, TileSchedulerMetaDataSize}, core::DataType::kInt32);
    auto num_splits = ctx.tensor({batch_size + 1}, core::DataType::kInt32);
    bmengine::functions::zeros_(ctx, tile_scheduler_metadata);
    bmengine::functions::zeros_(ctx, num_splits);
    int *tile_scheduler_metadata_ptr = tile_scheduler_metadata.data<int>();
    int *num_splits_ptr = num_splits.data<int>();

    Mla_metadata_params params = {};
    params.seqlens_k_ptr = seqlens_k_ptr;
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    params.num_splits_ptr = num_splits_ptr;
    params.batch_size = batch_size;
    params.block_size_n = block_size_n;
    params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
    params.num_sm_parts = num_sm_parts;
    get_mla_metadata_func(params, ctx.current_stream()->ptr);

//    std::cout << "get_mla_metadata: \n";

    return {tile_scheduler_metadata, num_splits};
}

std::tuple<core::Tensor, core::Tensor>
mha_fwd_kvcache_mla(
    const core::Context& ctx,
    core::Tensor& q,                               // batch_size x seqlen_q x num_heads x head_size
    const core::Tensor& kcache,                    // num_blocks x page_block_size x num_heads_k x head_size
//    std::optional<const core::Tensor>& vcache_,  // num_blocks x page_block_size x num_heads_k x head_size_v
    const size_t head_size_v,
    const core::Tensor& seqlens_k,                 // batch_size
    const core::Tensor& block_table,               // batch_size x max_num_blocks_per_seq
    const float softmax_scale,
    bool is_causal,
    const core::Tensor& tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    const core::Tensor& num_splits,                // batch_size + 1
    core::Tensor out_org
) {
//    core::Tensor vcache = !vcache_.empty() ? vcache_.value() : kcache;
    core::Tensor vcache = kcache;

    auto q_dtype = q.dtype();
    BM_ASSERT_EQ(kcache.dtype(), q_dtype, "query and key must have the same dtype");

    BM_ASSERT(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    BM_ASSERT(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    BM_ASSERT(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    BM_ASSERT_EQ(block_table.ndim(), 2, "block_table must be 2D");
    BM_ASSERT_EQ(block_table.dtype(), core::DataType::kInt32, "block_table must have dtype torch.int32");
    BM_ASSERT(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");

    BM_ASSERT_EQ(kcache.ndim(), 4, "kcache is not 4D");
    BM_ASSERT_EQ(kcache.size(1), 64, "block_size must be 64");
    BM_ASSERT_EQ(kcache.size(2), 1, "num_heads_k must be 1");
    BM_ASSERT_EQ(kcache.size(3), 576, "head_size must be 576");

    const auto sizes = q.size();
    const size_t batch_size = sizes[0];
    const size_t seqlen_q_ori = sizes[1];
    const size_t num_heads_ori = sizes[2];
    const size_t head_size = sizes[3];
    BM_ASSERT(head_size % 8 == 0, "head_size should be a multiple of 8");
    BM_ASSERT(head_size_v % 32 == 0, "head_size_v should be a multiple of 32");

    const size_t max_num_blocks_per_seq = block_table.size(1);
    const size_t num_blocks = kcache.size(0);
    const size_t page_block_size = kcache.size(1);
    const size_t num_heads_k = kcache.size(2);
    BM_ASSERT(batch_size > 0, "batch size must be positive");
    BM_ASSERT(num_heads_ori % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (seqlen_q_ori == 1) { is_causal = false; }

    // 注意：这里 seqlen_q 和 num_heads 互换了 ！！！
    const size_t ngroups = num_heads_ori / num_heads_k;
    const size_t seqlen_q = seqlen_q_ori * ngroups;
    const size_t num_heads = num_heads_k;
    q = q.view({batch_size, seqlen_q, num_heads, head_size});
//    q = q.view({batch_size, seqlen_q_ori, num_heads_k, ngroups, head_size}).transpose(2, 3)
//        .reshape({batch_size, seqlen_q, num_heads, head_size});

    size_t head_size_k = head_size;
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_k);
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);


    BM_ASSERT(seqlens_k.dtype() == core::DataType::kInt32, "seqlens_k must have dtype int32");
    CHECK_SHAPE(seqlens_k, batch_size);

    core::Tensor out = out_org.view({batch_size, seqlen_q, num_heads, head_size_v});
    core::Tensor softmax_lse = ctx.tensor({batch_size, num_heads, seqlen_q}, core::DataType::kFloat);

    Flash_fwd_mla_params params = {};
    // Set the sizes.
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.cu_seqlens_k = seqlens_k.data<int>();
    params.h = num_heads;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.ngroups = ngroups;
    params.is_causal = is_causal;
    params.d = head_size;
    params.d_v = head_size_v;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = float(softmax_scale * M_LOG2E);
    // Set the pointers and strides.
    params.q_ptr = q.data();
    params.k_ptr = kcache.data();
    params.v_ptr = vcache.data();
    params.o_ptr = out.data();
    params.softmax_lse_ptr = softmax_lse.data();
    // All stride are in elements, not bytes.
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = kcache.stride(0);
    params.v_batch_stride = vcache.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = kcache.stride(-3);
    params.v_row_stride = vcache.stride(-3);
    params.o_row_stride = out.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = kcache.stride(-2);
    params.v_head_stride = vcache.stride(-2);
    params.o_head_stride = out.stride(-2);

    params.block_table = block_table.data<int>();
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = page_block_size;

    BM_ASSERT(tile_scheduler_metadata.dtype() == core::DataType::kInt32, "tile_scheduler_metadata must have dtype int32");
    BM_ASSERT_EQ(tile_scheduler_metadata.size(1), TileSchedulerMetaDataSize, "");
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data<int>();
    params.num_sm_parts = tile_scheduler_metadata.size(0);
    BM_ASSERT(num_splits.dtype() == core::DataType::kInt32, "num_splits must have dtype int32");
    params.num_splits_ptr = num_splits.data<int>();

    core::Tensor softmax_lse_accum = ctx.tensor({batch_size + params.num_sm_parts, num_heads, seqlen_q},
                                                  core::DataType::kFloat);
    core::Tensor out_accum = ctx.tensor({batch_size + params.num_sm_parts, num_heads, seqlen_q, head_size_v},
                                          core::DataType::kFloat);
    params.softmax_lseaccum_ptr = softmax_lse_accum.data();
    params.oaccum_ptr = out_accum.data();

    auto stream = ctx.current_stream()->ptr;
    BM_ASSERT(head_size == 576, "");

    if (q_dtype == core::DataType::kBFloat16) {
        run_mla_fwd_splitkv_bf16(params, stream);
    } else if (q_dtype == core::DataType::kHalf) {
        run_mla_fwd_splitkv_f16(params, stream);
    }
    else {
        BM_ASSERT(false, "Unsupported tensor dtype for query");
    }

//    out = out.view({batch_size, seqlen_q_ori, ngroups, num_heads_k, head_size_v});.transpose(2, 3)
//        .reshape({batch_size, seqlen_q_ori, num_heads_ori, head_size_v});
//    softmax_lse = softmax_lse.view({batch_size, num_heads_k, seqlen_q_ori, ngroups}).transpose(2, 3)
//        .reshape({batch_size, num_heads_ori, seqlen_q_ori});

    return {out_org, softmax_lse};
}

} // namespace ds