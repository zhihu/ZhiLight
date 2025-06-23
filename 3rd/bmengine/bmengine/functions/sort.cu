#include "bmengine/functions/sort.h"

#include "bmengine/core/core.h"
#include "bmengine/functions/init.h"
//#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

namespace bmengine {
namespace functions {

using bmengine::core::Tensor;

// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
template<typename KeyT>
std::pair<core::Tensor, core::Tensor> sort_1d_temp(
    const core::Context& ctx,
    const core::Tensor& keys,
    const core::Tensor& values,
    int max_key)
{
    Tensor keys_out = ctx.tensor(keys.shape(), keys.dtype());
    Tensor values_out = ctx.tensor(values.shape(), values.dtype());

    int num_items = keys.numel();
    KeyT* d_keys_in = keys.data<KeyT>();
    KeyT* d_keys_out = keys_out.data<KeyT>();
    int end_bit = sizeof(KeyT) * 8;
    if (max_key > 0) {
        end_bit -= __builtin_clz(max_key);  // reduce end_bit to improve performance for int sort
    }
    auto stream = ctx.current_cuda_stream();

    size_t temp_storage_bytes = 0;

    BM_ASSERT_EQ(values.dtype(), core::DataType::kInt32, "Values int only");
//    BM_CORE_DTYPE_DISPATCH(values.dtype(),
    {
//        using ValueT = scalar_t;
        using ValueT = int;
        ValueT* d_values_in = values.data<ValueT>();
        ValueT* d_values_out = values_out.data<ValueT>();

        // get temp_storage_bytes
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
        Tensor temp_storage = ctx.tensor({temp_storage_bytes}, core::DataType::kInt8);
        void* d_temp_storage = temp_storage.data();

        // real sort
        BM_CUDART_ASSERT(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, 0, end_bit, stream));
    }
//    );
    BM_CUDART_ASSERT(cudaGetLastError());

    return {keys_out, values_out};
}

std::pair<core::Tensor, core::Tensor> sort_pair_1d(
    const core::Context& ctx,
    const core::Tensor& keys,
    const core::Tensor& values,
    int max_key)
{
    BM_ASSERT_EQ(keys.ndim(), 1, "sort_1d: key is not 1d");
    BM_ASSERT_EQ(values.ndim(), 1, "sort_1d: values is not 1d");
    BM_ASSERT_EQ(keys.numel(), values.numel(), "sort_1d: keys and values has different numel");

    Tensor keys_out;
    Tensor values_out;
    BM_CORE_DTYPE_DISPATCH(keys.dtype(), {
        std::tie(keys_out, values_out) = sort_1d_temp<scalar_t>(ctx, keys, values, max_key);
    });

    return {keys_out, values_out};
}

std::pair<core::Tensor, core::Tensor> sort_with_indices_1d(
    const core::Context& ctx,
    const core::Tensor& keys,
    int max_key) {
    Tensor indices = arange(ctx, 0, keys.numel());

    return sort_pair_1d(ctx, keys, indices, max_key);
}
}
}


