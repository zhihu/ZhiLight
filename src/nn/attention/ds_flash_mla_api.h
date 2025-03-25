#pragma one

#include "bmengine/core/core.h"
#include <tuple>

namespace ds {
using namespace bmengine;

std::tuple<core::Tensor, core::Tensor> get_mla_metadata(
    const core::Context& ctx,
    const core::Tensor& seqlens_k,
    const size_t num_heads_per_head_k,
    const size_t num_heads_k = 1
);

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
    core::Tensor out
);

}
