#pragma once
#include "bmengine/core/core.h"

namespace bmengine {
namespace functions {

// select sub tensors from given dim and index.
core::Tensor index_select(
    const core::Context& ctx,
    const core::Tensor& input,
    int dim,
    const core::Tensor& index,  // the 1-D tensor containing the indices to index
    core::Tensor* out = nullptr
);

// analog with torch.take_along_dim() without broadcast, dims < dim
// must have equal sizes between index and input.
core::Tensor index_along_dim(
    const core::Context& ctx,
    const core::Tensor& input,
    int dim,
    const core::Tensor& index // the n-D tensor containing the indices to index.
);

void index_along_dim(
    cudaStream_t stream,
    const core::Tensor& input,
    int dim,
    const core::Tensor& index, // the 1-D tensor containing the indices to index,
    core::Tensor& out);

void copy_last_dim(
    cudaStream_t stream,
    const core::Tensor& input,
    core::Tensor& output,
    int from,
    int to = -1,
    bool padding_zero = false
);

core::Tensor slice_last_dim(
    const core::Context& ctx,
    const core::Tensor& tensor,
    int from,
    int len
);
}  // namespace functions
}  // namespace bmengine
