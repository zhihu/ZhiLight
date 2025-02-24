#include "nn/feedforward/feedforward.h"
#include "nn/linear/linear.h"
#include "nn/functions/activation.cuh"
#include "model/model_context.h"
#include "nn/quant/int8/quant_kernel.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/element.h>
#include <bmengine/functions/gemm.h>
#include <bmengine/functions/reduce.cuh>
#include <bmengine/functions/typecast.h>
#include <bmengine/logger/std_log_op.hpp>

#include <map>
#include <assert.h>

namespace nn {

using namespace bmengine;
using bmengine::core::DataType;
using bmengine::core::Tensor;
using model::ModelContext;

#define GELU 1
#define SILU 2

// gridDim(seq_len, 1, dim_ff / 1024) blockDim(1024)
template<typename T>
static __global__ void BM_KERNEL(gate_fuse)(
    const T* in,  // (seq_len, 2 * dim_ff)
    T* out,       // (seq_len, dim_ff)
    int dim_ff,
    int act_type) {
    unsigned int offset = blockIdx.x * dim_ff;
    int i = blockIdx.z * blockDim.x + threadIdx.x;
    if (i < dim_ff) {
        float x = in[offset * 2 + i];
        float y = in[offset * 2 + dim_ff + i];
        if (act_type == GELU) {
            x = gelu(x);
        } else if (act_type == SILU) {
            x = silu(x);
        } else if (threadIdx.x == 0) {
            assert(false);
        }
        out[offset + i] = T(x * y);
    }
}

Tensor gate_fuse(
    const core::Context& ctx,
    const Tensor& input,
    const std::string& act_fn_type
) {
    int act_type;
    if (act_fn_type == "gelu")
        act_type = GELU;
    else if (act_fn_type == "silu")
        act_type = SILU;
    else
        throw std::logic_error(act_fn_type + " activation is not supported");

    size_t seq_len = input.numel() / input.size(-1);
    size_t dim_ff = input.size(-1) / 2;

    auto shape = input.shape();
    shape[shape.size() - 1] /= 2;
    Tensor out = ctx.tensor(shape, input.dtype());

    dim3 gridDim(seq_len, 1, round_up(dim_ff, 1024) / 1024);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(input.dtype(), {
        BM_KERNEL(gate_fuse)<scalar_t><<<gridDim, 1024, 0, stream>>>(
        input.data<scalar_t>(),
        out.mutable_data<scalar_t>(),
        dim_ff,
        act_type);
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return out;
}

#define SOFTMAX 1
#define SIGMOID 2
#define LINEAR 3
static const std::map<std::string, int> scoring_func_map {
    { "", SOFTMAX },
    { "softmax", SOFTMAX },
    { "sigmoid", SIGMOID },
    { "linear", LINEAR },
};

#define MAX_TOP_K 16
#define MAX_INLINE_NUM 16
template<typename T, int BUF_LEN=MAX_TOP_K>
static __device__ __forceinline__ void DEV_insert_sort_topk(
    const T* data,
    const int num,
    const int k,
    float (&value)[BUF_LEN],
    int (&idx)[BUF_LEN]
) {
    assert(k < BUF_LEN);
    for (int j = 0; j < num; ++j) {
        float v = data[j];
        // insert sort for small k
        int i;
        for (i = k - 1; i >= 0; --i) {
            if (v > value[i]) {
                value[i + 1] = value[i];
                idx[i + 1] = idx[i];
            } else {
                value[i + 1] = v;
                idx[i + 1] = j;
                break;
            }
        }
        if (i < 0) {
            value[0] = v;
            idx[0] = j;
        }
    }
}

template<typename T>
static __device__ __forceinline__ void DEV_softmax_inplace(const T* logits, float* data, const int n) {
    float local_max = -1e20;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, data[i]);
    }
    if (blockDim.x > 32)
        local_max = functions::blockReduceMax(local_max);
    else
        local_max = functions::warpReduceMaxB(local_max);

    float local_sum = 1e-20;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        data[i] = expf(data[i] - local_max);
        local_sum += data[i];
    }
    if (blockDim.x > 32)
        local_sum = functions::blockReduceSum(local_sum);
    else
        local_sum = functions::warpReduceSumB(local_sum);

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        data[i] /= local_sum;
    }
    __syncthreads();
}

template<typename T>
static __device__ __forceinline__ void DEV_route_score(
    const T* logits, float* data, const int n, int scoring_type) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        data[i] = logits[i];
    }

    if (scoring_type == SOFTMAX) {
        DEV_softmax_inplace(logits, data, n);
    } else if (scoring_type == SIGMOID) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            data[i] = 1.f / (1.f + expf(data[i]));
        }
        __syncthreads();
    }
}

// gridDim(seq_len) blockDim(32)
template<typename T>
static __global__ void KERNEL_top_k_softmax(
    const T* logits, int len_q, int num_exp, int k, float* out_v, int* out_idx,
    bool renormalize,
    float weight_scale,
    int scoring_type,
    int top_k_ext,
    int* worker_load,
    int num_worker) {
    int q = blockIdx.x;
    logits += q * num_exp;
    out_v += q * top_k_ext;
    out_idx += q * top_k_ext;
    assert(k < MAX_TOP_K);

    static __shared__ float data[256];
    assert(num_exp < 256);
    DEV_route_score(logits, data, num_exp, scoring_type);

    if (threadIdx.x > 0) return;

    float value[MAX_TOP_K + 1];
    int idx[MAX_TOP_K + 1];
    // get top k
    for (int i = 0; i < k; ++i) {
        value[i] = -1e20;
    }
    DEV_insert_sort_topk(data, num_exp, k, value, idx);

    float sum_e = 1.;
    if (renormalize) {
        sum_e = 1.e-20;
        for (int i = 0; i < k; ++i) {
            sum_e += value[i];
        }
    }
    for (int i = 0; i < k; ++i) {
        out_v[i] = value[i] / sum_e * weight_scale;
        out_idx[i] = idx[i];
    }

    if (worker_load) {
        for (int i = 0; i < k; ++i) {
            int rank = idx[i] % num_worker;
            atomicAdd(&worker_load[rank], 1);
        }
    }
    for (int i = k; i < top_k_ext; ++i) {
        out_v[i] = 1;
    }
}

// Return (scores, indices)
std::tuple<core::Tensor, core::Tensor> top_k_softmax(
    const core::Context& ctx,
    const core::Tensor& input,
    const core::Tensor& worker_load,
    int top_k,
    int top_k_ext,
    bool norm_topk_prob,
    float weight_scale,
    const std::string& scoring_func
) {
    BM_ASSERT_EQ(input.ndim(), 2, "Wrong input dim");
    BM_ASSERT_LE(top_k, MAX_TOP_K, "k too big");
    auto out_shape = input.shape();
    out_shape[1] = top_k_ext;
    core::Tensor out = ctx.tensor(out_shape, DataType::kFloat);
    core::Tensor out_idx = ctx.tensor(out_shape, DataType::kInt32);;

    dim3 gridDim(input.size(0));
    auto stream = ctx.current_stream()->ptr;
    int scoring_type = scoring_func_map.at(scoring_func);
    BM_DTYPE_DISPATCH_FLOAT(input.dtype(), {
        KERNEL_top_k_softmax<scalar_t><<<gridDim, 32, 0, stream>>>(
        input.data<scalar_t>(),
        input.size(0),
        input.size(1),
        top_k,
        out.mutable_data<float>(),
        out_idx.mutable_data<int>(),
        norm_topk_prob,
        weight_scale,
        scoring_type,
        top_k_ext,
        worker_load.numel() ? worker_load.data<int>() : nullptr,
        ctx.world_size());
    });
    BM_CUDART_ASSERT(cudaGetLastError());

    return std::make_tuple(out, out_idx);
}

#define WARP_SIZE 32

template<typename T, int N=32>
static __device__ inline void warpBitonicSort(T& v1, int& pos, bool asc = false) {
    int lane_id = threadIdx.x & (N - 1);
#pragma unroll
    for (int k = 2; k <= N; k *= 2) {
        bool desc = ((lane_id & k) == 0) ^ asc;
#pragma unroll
        for (int j = k / 2; j > 0; j /= 2) {
            T v2 = __shfl_xor_sync(0xFFFFFFFF, v1, j);
            int pos2 = __shfl_xor_sync(0xFFFFFFFF, pos, j);
            bool upper = (lane_id & j) != 0;

            if (desc ^ (v1 > v2 || (v1 == v2 && pos < pos2)) ^ upper) {
                v1 = v2;
                pos = pos2;
            }
        }
    }
}

#define DEBUG_GROUP_TOPK 0
#define DEBUG_GROUP_BLOCK_ID 0

// gridDim(seq_len, 1, 1) blockDim(num_group * 32)
template<typename T>
static __global__ void KERNEL_group_topk(
    const T* logits,
    const float* correction_bias,
    int num_exp, int k, float* out_v, int* out_idx,
    bool renormalize,
    float weight_scale,
    int scoring_type,
    int num_group,
    int topk_group,
    int num_in_group,
    int top_k_ext,
    int* worker_load = nullptr,
    int num_worker = 0) {
    assert(k <= MAX_TOP_K);
    assert(num_in_group <= WARP_SIZE);

    int q = blockIdx.x;
    int g = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    logits += q * num_exp;
    out_v += q * top_k_ext;
    out_idx += q * top_k_ext;

    // softmax
    static __shared__ float data[512];
    assert(num_exp < 512);
    if (scoring_type == SIGMOID) {
        if (threadIdx.x < num_exp) {
            data[threadIdx.x] = 1.f / (1.f + expf(-float(logits[threadIdx.x])));
#if DEBUG_GROUP_TOPK
            assert(!isnan(data[threadIdx.x]));
#endif
        }
        __syncthreads();
    } else {
        if (threadIdx.x < num_exp) {
            data[threadIdx.x] = logits[threadIdx.x];
        }
        DEV_softmax_inplace(logits, data, num_exp);
    }

    __shared__ float shared_val[32];
    __shared__ int shared_pos[32];
    __shared__ int shared_gid[32];
    __shared__ int group_ranks[32];
    // get top k in group
    int exp_id = -1;
    float score = -1e20;
    if (lane_id < num_in_group) {
        exp_id = g * num_in_group + lane_id;
        assert(exp_id < num_exp);
        score = data[exp_id];
        if (correction_bias) {
            // plus bias for sort ONLY
            score += correction_bias[exp_id];
        }
    }
#if DEBUG_GROUP_TOPK
    if (q == DEBUG_GROUP_BLOCK_ID && g == 0 && lane_id < num_in_group) {
        printf("UnsortGroup lane_id=%d, exp_id=%d, score=%.5f\n", lane_id, exp_id, score);
    }
#endif
    warpBitonicSort(score, exp_id);
//    if (q == 0 && g == 0 && lane_id < num_in_group) {
//        printf("lane_id=%d, exp_id=%d, score=%.5f\n", lane_id, exp_id, score);
//    }
    if (lane_id == 0) {
        shared_val[g] = score; // max score in group
        shared_gid[g] = g; // group id
        group_ranks[g] = -1;
    }
    __syncthreads();

    // select topk_group group
    if (threadIdx.x < WARP_SIZE) {
        float group_score = (lane_id < num_group) ? shared_val[lane_id] : -1e20;
        int group_id = shared_gid[lane_id];
        assert(num_group <= 8);
#if DEBUG_GROUP_TOPK
        if (q == DEBUG_GROUP_BLOCK_ID && g == 0 && lane_id < topk_group) {
            printf("UnSort: lane_id=%d, group_id=%d, score=%.5f\n", lane_id, group_id, group_score);
        }
#endif
        warpBitonicSort<float, 8>(group_score, group_id);
#if DEBUG_GROUP_TOPK
        if (q == DEBUG_GROUP_BLOCK_ID && g == 0 && lane_id < topk_group) {
            printf("Sorted: lane_id=%d, group_id=%d, score=%.5f\n", lane_id, group_id, group_score);
        }
#endif
        if (threadIdx.x < topk_group) {
            // shared_gid[threadIdx.x] = group_id;
            group_ranks[group_id] = threadIdx.x;
        }
    }
    __syncthreads();

    // Copy topk in topk_group to shared
    assert(topk_group * k <= WARP_SIZE);
    int group_rank = group_ranks[g];
    if (group_rank >= 0 && lane_id < k) {
        shared_val[group_rank * k + lane_id] = score;
        shared_pos[group_rank * k + lane_id] = exp_id;
#if DEBUG_GROUP_TOPK
        if (q == DEBUG_GROUP_BLOCK_ID) {
            printf("Copy topk in topk_group: group_rank=%d, lane_id=%d, exp_id=%d, score=%.5f\n",
                   group_rank, lane_id, exp_id, score);
        }
#endif
    }
    __syncthreads();

    // Final sort topk_group * k in 1st warp
    if (threadIdx.x < WARP_SIZE) {
        score = (lane_id < topk_group * k) ? shared_val[lane_id] : -1e20;
        exp_id = shared_pos[lane_id];
        warpBitonicSort(score, exp_id);
    }
    if (threadIdx.x < k && correction_bias) {
        assert(exp_id >= 0);
#if DEBUG_GROUP_TOPK
        if (q == DEBUG_GROUP_BLOCK_ID) {
            printf("Final: k=%d, exp=%d, score=%.5f, original_score=%.5f, bias=%.5f\n",
                   threadIdx.x, exp_id, score, data[exp_id], correction_bias[exp_id]);
        }
#endif
        score = data[exp_id]; // use original score
    }
    float sum_e = 1.f;
    if (renormalize) {
        float tmp_score = threadIdx.x < k ? score : 0.f;
        sum_e = functions::warpReduceSumB(tmp_score) + 1e-20;
    }
    if (threadIdx.x < k) {
        out_v[lane_id] = score / sum_e * weight_scale;
        out_idx[lane_id] = exp_id;
#if DEBUG_GROUP_TOPK
        if (q == DEBUG_GROUP_BLOCK_ID) {
            printf("Write: k=%d, exp=%d, score=%.5f, sum_e=%.5f, weight_scale=%.5f\n",
                   threadIdx.x, exp_id, score, sum_e, weight_scale);
        }
#endif
        if (worker_load) {
            int rank = exp_id % num_worker;
            atomicAdd(&worker_load[rank], 1);
        }
//        if (q == 0) {
//            printf("lane_id=%d, exp_id=%d, score=%.5f\n", lane_id, exp_id, score);
//        }
    }

    if (threadIdx.x < top_k_ext - k) {
        out_v[k + threadIdx.x] = 1;
        out_idx[k + threadIdx.x] = 0;
    }
}

std::tuple<core::Tensor, core::Tensor> group_topk_softmax(
    const core::Context& ctx,
    const core::Tensor& input,
    const core::Tensor& score_correction_bias,
    const core::Tensor& worker_load,
    int num_group,
    int topk_group,
    int top_k,
    int top_k_ext,
    bool norm_topk_prob,
    float weight_scale,
    const std::string& scoring_func
) {
    BM_ASSERT_EQ(input.ndim(), 2, "Wrong input dim");
    BM_ASSERT_LE(num_group, WARP_SIZE, "num_group is too big"); // otherwise block reduce
    BM_ASSERT_LE(top_k, MAX_TOP_K, "k too big");
    if (!score_correction_bias.empty()) {
        BM_ASSERT_EQ(score_correction_bias.numel(), input.size(1), "wrong correction_bias numel");
        BM_ASSERT_EQ(score_correction_bias.dtype(), DataType::kFloat, "wrong correction_bias dtype");
    }
    auto out_shape = input.shape();
    out_shape[1] = top_k_ext;
    core::Tensor out = ctx.tensor(out_shape, DataType::kFloat);
    core::Tensor out_idx = ctx.tensor(out_shape, DataType::kInt32);;

    dim3 gridDim(input.size(0));
    auto stream = ctx.current_stream()->ptr;
    int scoring_type = scoring_func_map.at(scoring_func);
    int num_in_group = input.size(1) / num_group;
    BM_DTYPE_DISPATCH_FLOAT(input.dtype(), {
        KERNEL_group_topk<scalar_t><<<gridDim, num_group * WARP_SIZE, 0, stream>>>(
            input.data<scalar_t>(),
            score_correction_bias.data<float>(),
            input.size(1),
            top_k,
            out.mutable_data<float>(),
            out_idx.mutable_data<int>(),
            norm_topk_prob,
            weight_scale,
            scoring_type,
            num_group,
            topk_group,
            num_in_group,
            top_k_ext,
            worker_load.numel() ? worker_load.data<int>() : nullptr,
            ctx.world_size()
        );
    });
    BM_CUDART_ASSERT(cudaGetLastError());

    return std::make_tuple(out, out_idx);
}

// gridDim (seq_len, dim_model / 1024), blockDim (1024)
template<typename T>
static __global__ void KERNEL_sum_experts(
    const int dim_model,
    const int K,
    const T* __restrict__ input,      // (seq_len * k, dim_model)
    const int* __restrict__ index,    // (seq_len * k)
    const float* __restrict__ weight, // (seq_len * k)
    T* __restrict__ out               // (seq_len, dim_model)
) {
    int q = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= dim_model) return;
    float acc = 0;
    for (int i = 0; i < K; ++i) {
        int idx = index[q * K + i];
        float w = weight[q * K + i];
        acc += float(input[idx * dim_model + d]) * w;
    }
    out[q * dim_model + d] = acc;
}

// gridDim (seq_len, dim_model / 1024), blockDim (1024)
template<typename T, bool DEBUG = false>
static __global__ void KERNEL_sum_experts_arr(
    const int dim_model,
    const int NUM_EXP,
    const int K,
    const T** __restrict__ input_arr, // m => (0~seq_len, dim_model)
    const int* __restrict__ input_lens, // (seq_len * k)
    const int* __restrict__ experts,  // (seq_len * k)
    const int* __restrict__ index,    // (seq_len * k)
    const float* __restrict__ weight, // (seq_len * k)
    T* __restrict__ out,              // (seq_len, dim_model)
    bool exp_parallel,
    int world_size_mask,
    int local_rank
) {
    int q = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
//    if (q ==0 && d == 0 && local_rank==6) {
//        printf("#### i=%d, idx=%d\n", 21815, index[21815]);
//    }
    if (d >= dim_model) return;
    float acc = 0;
    for (int k = 0; k < K; ++k) {
        int i = q * K + k; // which token
        int exp = experts[i]; // [0, num_experts)
        assert(exp < NUM_EXP);
        if (exp_parallel && ((exp & world_size_mask) != local_rank)) {
            continue; // Skip NOT local experts
        }
//        if (d == 0) {
//            printf("world_size_mask %d, local_rank %d, exp %d, idx %d\n", world_size_mask, local_rank, exp, index[i]);
//        }
        const T* input = input_arr[exp];
        assert(input);
        if (gridDim.x == 1) { // seq_len == 1 => idx=0
            acc += float(input[d]) * weight[k];
        } else {
            int idx = index[i]; // index in one expert
            float w = weight[i];
            if (DEBUG && d == 0) {
                const int len = input_lens[exp];
                if (idx >= len) {
                    printf("i=%d, rank=%d, q=%d, k=%d, exp=%d, idx=%d, len=%d\n", i, local_rank, q, k, exp, idx, len);
                }
                assert(idx < len);
            }
            acc += float(input[idx * dim_model + d]) * w;
        }
    }
    assert(!isnan(acc));
    out[q * dim_model + d] = acc;
}

struct InlineAddress {
    void* addr[MAX_INLINE_NUM];
};
// gridDim (seq_len, dim_model / 1024), blockDim (1024)
template<typename T>
static __global__ void KERNEL_sum_experts_inline_arr(
    const int dim_model,
    const int K,
    InlineAddress input_arr,          // m => (0~seq_len, dim_model)
    const int* __restrict__ experts,  // (seq_len * k)
    const int* __restrict__ index,    // (seq_len * k)
    const float* __restrict__ weight, // (seq_len * k)
    T* __restrict__ out               // (seq_len, dim_model)
) {
    int q = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= dim_model) return;
    float acc = 0;
    for (int k = 0; k < K; ++k) {
        int i = q * K + k; // which token
        int exp = experts[i]; // [0, num_experts)
        const T* input = reinterpret_cast<T*>(input_arr.addr[exp]);
        if (gridDim.x == 1) { // seq_len == 1 => idx=0
            acc += float(input[d]) * weight[k];
        } else {
            int idx = index[i]; // index in one expert
            float w = weight[i];
            acc += float(input[idx * dim_model + d]) * w;
        }
    }
    out[q * dim_model + d] = acc;
}

// return (seq_len, dim_model)
core::Tensor sum_experts(
    const core::Context& ctx,
    const core::Tensor& input, // (seq_len * k, dim_model)
    const core::Tensor& index, // (seq_len * k)
    const core::Tensor& weights // (seq_len, k)
) {
    BM_ASSERT_EQ(input.ndim(), 2, "Wrong input dim");
    BM_ASSERT_EQ(weights.ndim(), 2, "Wrong weights dim");
    BM_ASSERT_EQ(weights.numel(), input.size(0), "Wrong weights size");
    BM_ASSERT_EQ(weights.numel(), index.numel(), "Wrong reverse_idx size");
    size_t dim_model = input.size(-1);
    size_t K = weights.size(1);
    BM_ASSERT_LE(K, MAX_TOP_K, "top k too big");
    BM_ASSERT_EQ(input.size(0) % K, 0, "Wrong input dim0");
    size_t seq_len = input.size(0) / K;

    std::vector<size_t> shape = { seq_len, dim_model };
    core::Tensor out = ctx.tensor(shape, input.dtype());

    size_t threads = round_up_thread(dim_model);
    dim3 gridDim(seq_len, round_up(dim_model, threads) / threads);
    auto stream = ctx.current_stream()->ptr;
    BM_DTYPE_DISPATCH_FLOAT(input.dtype(), {
        KERNEL_sum_experts<scalar_t><<<gridDim, threads, 0, stream>>>(
            dim_model,
            K,
            input.data<scalar_t>(),
            index.data<int>(),
            weights.data<float>(),
            out.mutable_data<scalar_t>()
        );
    });
    BM_CUDART_ASSERT(cudaGetLastError());
    return out;
}

// return (seq_len, dim_model)
core::Tensor sum_experts(
    const core::Context& ctx,
    std::vector<core::Tensor> inputs, // m => (0~seq_len, dim_model)
    const core::Tensor& concat_inputs,
    const core::Tensor& experts,      // (seq_len * k)
    const core::Tensor& index,        // (seq_len * k)
    const core::Tensor& weights,      // (seq_len, k)
    bool exp_parallel,
    int world_size,
    int local_rank
) {
    size_t dim_model = 0;
    std::vector<void*> ptr_vec;
    std::vector<int> len_vec;
    core::DataType dtype;
    InlineAddress inline_address;
    for (auto &input: inputs) {
        BM_ASSERT(input.ndim() == 2 || input.ndim() == 0, "Wrong input dim");
        void* addr = input.ndim() > 0 ? input.data(): nullptr;
        if (ptr_vec.size() < MAX_INLINE_NUM) inline_address.addr[ptr_vec.size()] = addr;
        ptr_vec.push_back(addr);
        len_vec.push_back(input.ndim() > 0 ? input.size(0) : 0);
        if (input.ndim() > 0) {
            BM_ASSERT(dim_model == input.size(-1) || dim_model == 0, "dim_model mismatch");
            dim_model = input.size(-1);
            dtype = input.dtype();
        }
    }
    BM_ASSERT(dim_model > 0, "all inputs is empty");

    BM_ASSERT_EQ(weights.ndim(), 2, "Wrong weights dim");
    BM_ASSERT_EQ(weights.numel(), experts.numel(), "Wrong weights size");
    size_t seq_len = weights.size(0);
    size_t K = weights.size(1);
    if (seq_len > 1)
        BM_ASSERT_EQ(weights.numel(), index.numel(), "Wrong reverse_idx size");
    BM_ASSERT_LE(K, MAX_TOP_K, "top k too big");

    core::Tensor out = ctx.tensor({ seq_len, dim_model }, dtype);

    size_t threads = round_up_thread(dim_model);
    dim3 gridDim(seq_len, round_up(dim_model, threads) / threads);
    auto stream = ctx.current_stream()->ptr;
    const int* index_ptr = index.numel() ? index.data<int>() : nullptr;
    if (ptr_vec.size() <= MAX_INLINE_NUM && !exp_parallel) {
        BM_DTYPE_DISPATCH_FLOAT(dtype, {
            KERNEL_sum_experts_inline_arr<scalar_t><<<gridDim, threads, 0, stream>>>(
                dim_model,
                K,
                inline_address,
                experts.data<int>(),
                index_ptr,
                weights.data<float>(),
                out.mutable_data<scalar_t>()
            );
        });
    } else {
        core::Tensor ptr_arr = ctx.tensor({ptr_vec.size()}, core::DataType::kDouble);
        ptr_arr.from_buffer(ptr_vec.data());
        // core::Tensor input_lens = ctx.tensor_of(len_vec);
        core::Tensor input_lens;
        BM_DTYPE_DISPATCH_FLOAT(dtype, {
            KERNEL_sum_experts_arr<scalar_t><<<gridDim, threads, 0, stream>>>(
                dim_model,
                inputs.size(),
                K,
                ptr_arr.data<const scalar_t *>(),
                input_lens.numel() ? input_lens.data<int>() : nullptr,
                experts.data<int>(),
                index_ptr,
                weights.data<float>(),
                out.mutable_data<scalar_t>(),
                exp_parallel,
                world_size - 1,
                local_rank
            );
        });
//        BM_CUDART_ASSERT(cudaStreamSynchronize(stream));
//        functions::check_numeric(ctx, out);
    }
    BM_CUDART_ASSERT(cudaGetLastError());
    return std::move(out);
}

// (seq_len), (1)
static __global__ void KERNEL_route_shared_lb_v1(
    int* exp_ids,
    float* exp_weights,
    int* worker_load,
    const int max_load,
    const int world_size,
    const int seq_len,
    const int top_k,
    const int top_k_ext,
    const int num_local_experts
) {
    if (blockIdx.y > 0) return;
    int rank = 0;
    int q = blockIdx.x;
    // for (int q = 0; q < seq_len; q++) {
    {
        for (int k = top_k; k < top_k_ext; ++k) {
            while (true) {
                if (atomicAdd(&worker_load[rank], 1) < max_load) {
                    exp_ids[q * top_k_ext + k] = (num_local_experts + k - top_k) * world_size + rank;
                    // exp_weights[q * top_k_ext + k] = 1;
                    break;
                } else {
                    rank++;
                    assert(rank < world_size);
                }
            }
        }
    }
}

// (seq_len, n_shared), (1)
static __global__ void KERNEL_route_shared_lb(
    int* exp_ids,
    float* exp_weights,
    int* worker_load,
    const int max_load,
    const int world_size,
    const int seq_len,
    const int top_k,
    const int top_k_ext,
    const int num_local_experts
) {
    int capacities[9];
    int ranks[9];
    for (int r = 0; r < world_size; ++r) {
        capacities[r] = worker_load[r] >= max_load ? 0 : max_load - worker_load[r];
        ranks[r] = r;
    }

    const int q = blockIdx.x;
    const int s = blockIdx.y;
    int r = 0;

    // skip seq_len * s
    int skip_len = seq_len * s + q;
    while (skip_len >= capacities[r]) {
        skip_len -= capacities[r];
        r++; // move to next worker
        assert(r < world_size);
    }
    assert(skip_len < capacities[r]);

    int rank = ranks[r];
    int exp_id = (num_local_experts + s) * world_size + rank;
    exp_ids[q * top_k_ext + top_k + s] = exp_id;
//    if (q == 0) {
//        printf("offset=%d, exp=%d\n", q * top_k_ext + s, exp_id);
//    }
}

void route_shared_lb(
    const core::Context& ctx,
    core::Tensor& exp_ids,
    core::Tensor& exp_weights,
    core::Tensor& worker_load,
    int top_k,
    int num_local_experts) {
    BM_ASSERT_EQ(exp_ids.ndim(), 2, "Wrong exp_ids dim");
    BM_ASSERT_EQ(exp_ids.shape(), exp_weights.shape(), "shape mismatch");
    BM_ASSERT_EQ(exp_ids.dtype(), core::DataType::kInt32, "");
    BM_ASSERT_EQ(exp_weights.dtype(), core::DataType::kFloat, "");
    BM_ASSERT_EQ(worker_load.numel(), ctx.world_size(), "");

    int seq_len = exp_ids.size(0);
    int top_k_ext = exp_ids.size(1);
    int max_load = round_up(exp_ids.numel(), worker_load.numel()) / worker_load.numel();
    BM_ASSERT_LE(top_k, top_k_ext - 1, "top k too big");

    auto stream = ctx.current_stream()->ptr;
    dim3 gridDim(seq_len, top_k_ext - top_k);
    auto kernel = KERNEL_route_shared_lb;
    if (seq_len == 1) kernel = KERNEL_route_shared_lb_v1;
    kernel<<<gridDim, 1, 0, stream>>>(
        exp_ids.data<int>(),
        exp_weights.data<float>(),
        worker_load.data<int>(),
        max_load,
        ctx.world_size(),
        seq_len,
        top_k,
        top_k_ext,
        num_local_experts);
    BM_CUDART_ASSERT(cudaGetLastError());
}
}
