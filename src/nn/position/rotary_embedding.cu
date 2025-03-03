#include "nn/position/rotary_embedding.h"
#include "nn/position/rope_common.cuh"
#include "bmengine/functions/index_select.h"
#include "bmengine/functions/utils.cuh"
#include "bmengine/functions/transpose.h"
#include "bmengine/logger/std_log_op.hpp"
#include "model/dyn_batch_context.h"
#include "model/model_context.h"
#include <numeric>
#include <assert.h>

namespace nn {

using namespace bmengine;
using bmengine::core::Tensor;

// gridDim (seq_len, batch, dim_model / 1024),   blockDim (1024, 1, 1)
template<typename T, bool IsDynamic = false>
static __global__ void KERNEL_rotary_embedding(
    int dim_model,
    int dim_head, // dim_model = (num_head, dim_head)
    size_t hidden_stride,
    size_t pos_stride,
    const int32_t* __restrict__ pos, // (batch, seq_len)
    const T* __restrict__ in,        // (batch, seq_len, dim_model)
    T* __restrict__ out,             // (batch, seq_len, dim_model)
    float rope_theta,
    int max_position_embeddings,
    float scaling_factor) {
    int batch_id = blockIdx.y;
    int target_pos = pos[batch_id * pos_stride + blockIdx.x];
    int ith = blockIdx.z * blockDim.x + threadIdx.x;
    int col = ith % dim_head;
    size_t offset = batch_id * hidden_stride + blockIdx.x * dim_model;

    if constexpr (IsDynamic) {
        int seq_len = pos[batch_id * pos_stride + gridDim.x - 1];
        if (seq_len > max_position_embeddings) {
            rope_theta *= powf(
                (scaling_factor * seq_len / max_position_embeddings) - (scaling_factor - 1.f),
                dim_head / (dim_head - 2));
        }
    }
    if (ith >= dim_model)
        return;
    int half_dim = dim_head / 2;
    if (col < half_dim) {
        float freq = target_pos * powf(rope_theta, -float(col * 2) / dim_head);
        float cos_freq = cos(freq);
        float sin_freq = sin(freq);
        out[offset + ith] =
            in[offset + ith] * T(cos_freq) - in[offset + ith + half_dim] * T(sin_freq);
    } else {
        float freq = target_pos * powf(rope_theta, -float((col - half_dim) * 2) / dim_head);
        float cos_freq = cos(freq);
        float sin_freq = sin(freq);
        out[offset + ith] =
            in[offset + ith] * T(cos_freq) + in[offset + ith - half_dim] * T(sin_freq);
    }
}

// gridDim (seq_len, num_heads),   blockDim (dim_head / 2)
template<typename T>
static __global__ void KERNEL_rotary_emb_NOT_neox(
    const int32_t* __restrict__ pos, // (seq_len)
    const T* __restrict__ g_in,      // (seq_len, num_heads, dim_head)
    T* __restrict__ g_out,           // (seq_len, num_heads, dim_head)
    uint32_t src_stride1,
    uint32_t src_stride2,
    uint32_t dst_stride1,
    uint32_t dst_stride2,
    float base,
    float scaling_factor,
    bool OUT_neox_style=false) {
    int m = pos[blockIdx.x];
    int dim_head = blockDim.x * 2;
    int i = threadIdx.x;

    size_t src_offset = size_t(blockIdx.x) * src_stride1 + blockIdx.y * src_stride2;
    size_t dst_offset = size_t(blockIdx.x) * dst_stride1 + blockIdx.y * dst_stride2;

    T in[2];
    T out[2];
    *reinterpret_cast<int*>(&in) = *reinterpret_cast<const int*>(&g_in[src_offset + i * 2]);
    float inv_freq = powf(base, -float(i * 2) / dim_head);
    float freq = m * inv_freq;
    float cos_freq = cos(freq);
    float sin_freq = sin(freq);

    out[0] = in[0] * T(cos_freq) - in[1] * T(sin_freq);  // x
    out[1] = in[0] * T(sin_freq) + in[1] * T(cos_freq);  // y
    if (g_in == g_out)
        __syncthreads(); // Need sync for inplace update
    if (OUT_neox_style) {
        g_out[dst_offset + i] = out[0];
        g_out[dst_offset + i + blockDim.x] = out[1];
    } else {
        *reinterpret_cast<int *>(&g_out[dst_offset + i * 2]) = *reinterpret_cast<int *>(&out);
    }
}

// gridDim (seq_len, num_heads),   blockDim (dim_head)
template<typename T>
static __global__ void KERNEL_rope_with_cache(
    const float *__restrict__ g_cos, // (seq_len, dim_head)
    const float *__restrict__ g_sin, // (seq_len, dim_head)
    const T* __restrict__ g_in,      // (seq_len, num_heads, dim_head)
    T* __restrict__ g_out,           // (seq_len, num_heads, dim_head)
    uint32_t src_stride1,
    uint32_t src_stride2,
    uint32_t dst_stride1,
    uint32_t dst_stride2,
    bool neox_style) {
    int dim_head = blockDim.x;
    int col = threadIdx.x;
    int half_dim = dim_head / 2;

    size_t src_offset = size_t(blockIdx.x) * src_stride1 + blockIdx.y * src_stride2 + col;
    size_t dst_offset = size_t(blockIdx.x) * dst_stride1 + blockIdx.y * dst_stride2 + col;

    float cos_freq = g_cos[blockIdx.x * dim_head + col];
    float sin_freq = g_sin[blockIdx.x * dim_head + col];

    float t = rope_one_value(g_in, src_offset, cos_freq, sin_freq, col, half_dim, neox_style);;

    if (g_in == g_out)
        __syncthreads(); // Need sync for inplace update
    g_out[dst_offset] = t;
}

class RotaryEmbedding::impl {
public:
    class NormalImpl;
    class DynamicNTKImpl;
    class YarnImpl;
    int dim_head;
    float rope_theta;
    std::string rope_scaling_type;
    float scaling_factor;
    int max_position_embeddings;
    bool neox_style = true;

    impl(const core::Context& ctx, model::ModelConfig cfg)
        : dim_head(cfg.dim_head),
          rope_theta(cfg.rope_theta),
          rope_scaling_type(cfg.rope_cfg.type),
          scaling_factor(cfg.rope_cfg.factor),
          max_position_embeddings(cfg.max_position_embeddings),
          neox_style(cfg.rope_cfg.neox_style) {
        if (cfg.qk_rope_head_dim > 0)
            dim_head = cfg.qk_rope_head_dim;
    }
    virtual ~impl() {}

    virtual std::tuple<core::Tensor, core::Tensor> forward(
        const core::Context& ctx,
        const core::Tensor& pos, // (batch, seq_len)
        const core::Tensor& q,   // (batch, seq_len, dim_model)
        const core::Tensor& k    // (batch, seq_len, dim_model)
        ) = 0;

    virtual Tensor rotate(
        const core::Context& ctx,
        const core::Tensor& pos, // (batch, seq_len)
        const core::Tensor& q,   // (batch, seq_len, dim_model)
        core::Tensor* output = nullptr
    ) {
        throw std::runtime_error("Unsupported");
    }
    virtual void rotate_inplace(
        const core::Context& ctx,
        const core::Tensor& pos, // (batch, seq_len)
        core::Tensor& q          // (batch, seq_len, dim_model)
    ) {
        throw std::runtime_error("Unsupported");
    }

    void print_cos_sin(const core::Context& ctx, size_t seq_len) {
        std::vector<int> v_pos(seq_len);
        std::iota(v_pos.begin(), v_pos.end(), 0);
        std::vector<float> v(seq_len * dim_head);
        for(int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < dim_head; ++j) {
                // Set: x=1, y=0
                if (neox_style) {
                    // x, x ..., y, y ....
                    v[i * dim_head + j] = j < dim_head / 2 ? 1 : 0;
                } else {
                    // x, y, x, y ...
                    v[i * dim_head + j] = (j % 2 == 0) ? 1 : 0;
                }
            }
        }

        Tensor pos = ctx.tensor_of(v_pos);
        Tensor q = ctx.tensor_of(v, {seq_len, size_t(dim_head)});
        Tensor out = rotate(ctx, pos, q);
        Tensor cos = functions::slice_last_dim(ctx, out, 0, dim_head / 2);
        Tensor sin = functions::slice_last_dim(ctx, out, dim_head / 2, dim_head / 2);
        std::cout << "cos: " << cos << std::endl;
        std::cout << "sin: " << sin << std::endl;
    }
};

class RotaryEmbedding::impl::NormalImpl : public RotaryEmbedding::impl {
public:
    NormalImpl(const core::Context& ctx, model::ModelConfig cfg) : impl(ctx, cfg) { }

    void rotate_NOT_neox_style(
        const core::Context& ctx,
        const core::Tensor& pos, // (batch? seq_len)
        const core::Tensor& q,   // (batch? seq_len, dim_model) or (batch?, seq_len, num_heads, dim_head)
        core::Tensor& out_q
    ) {
        BM_ASSERT_EQ(core::DataType::kInt32, pos.dtype(), "dtype mismatch");
        BM_ASSERT(q.ndim() <= pos.ndim() + 2, "Dim mismatch");
        BM_ASSERT(q.size(-1) % dim_head == 0, "dim_model mismatch");
        BM_ASSERT_EQ(q.size(0), pos.size(0), "batch or seq_len mismatch");
        if (pos.ndim() > 1)
            BM_ASSERT_EQ(q.size(1), pos.size(1), "seq_len mismatch");
        if (q.ndim() == pos.ndim() + 2)
            BM_ASSERT_EQ(q.size(-1), dim_head, "dim_head mismatch");

        uint32_t src_stride1 = q.stride(-2);
        uint32_t src_stride2 = dim_head;
        uint32_t dst_stride1 = out_q.stride(-2);
        uint32_t dst_stride2 = dim_head;
        if (q.ndim() == pos.ndim() + 2) {
            src_stride1 = q.stride(-3);
            src_stride2 = q.stride(-2);
            dst_stride1 = out_q.stride(-3);
            dst_stride2 = out_q.stride(-2);
        }

        size_t seq_len = pos.numel();
        size_t num_heads = (q.ndim() == pos.ndim() + 2) ? q.size(-2) : (q.size(-1) / dim_head);
//        std::cout << "seq_len: " << seq_len << std::endl;
//        std::cout << "num_heads: " << num_heads << std::endl;
        {
            dim3 gridDim(seq_len, num_heads);
            auto stream = ctx.current_stream()->ptr;
            BM_DTYPE_DISPATCH_HALF(q.dtype(), {
                KERNEL_rotary_emb_NOT_neox<<<gridDim, dim_head / 2, 0, stream>>>(
                    pos.data<int32_t>(),
                    q.data<scalar_t>(),
                    out_q.data<scalar_t>(),
                    src_stride1, src_stride2, dst_stride1, dst_stride2,
                    rope_theta,
                    scaling_factor);
            });
            BM_CUDART_ASSERT(cudaGetLastError());
        }
    }

    void rotate_with_cache(
        const core::Context& ctx,
        const core::Tensor& cos, // (batch? seq_len, dim_head)
        const core::Tensor& sin, // (batch? seq_len, dim_head)
        const core::Tensor& q,   // (batch? seq_len, dim_model) or (batch?, seq_len, num_heads, dim_head)
        core::Tensor& out_q
    ) {
        BM_ASSERT_EQ(core::DataType::kFloat, cos.dtype(), "dtype mismatch");
        BM_ASSERT_EQ(sin.shape(), cos.shape(), "shape mismatch");
        BM_ASSERT(q.ndim() <= cos.ndim() + 1, "Dim mismatch");
        BM_ASSERT(q.size(-1) % dim_head == 0, "dim_model mismatch");
        BM_ASSERT_EQ(q.size(0), cos.size(0), "batch or seq_len mismatch");
        if (cos.ndim() == 3)
            BM_ASSERT_EQ(q.size(1), cos.size(1), "seq_len mismatch");
        if (q.ndim() == cos.ndim() + 1)
            BM_ASSERT_EQ(q.size(-1), dim_head, "dim_head mismatch");

        uint32_t src_stride1 = q.stride(-2);
        uint32_t src_stride2 = dim_head;
        uint32_t dst_stride1 = out_q.stride(-2);
        uint32_t dst_stride2 = dim_head;
        if (q.ndim() == cos.ndim() + 1) {
            src_stride1 = q.stride(-3);
            src_stride2 = q.stride(-2);
            dst_stride1 = out_q.stride(-3);
            dst_stride2 = out_q.stride(-2);
        }

        size_t seq_len = cos.numel() / cos.size(-1);
        size_t num_heads = (q.ndim() == cos.ndim() + 1) ? q.size(-2) : (q.size(-1) / dim_head);
        if (ctx.is_layer(1000)) {
            std::cout << "seq_len: " << seq_len << std::endl;
            std::cout << "num_heads: " << num_heads << std::endl;
        }
        {
            dim3 gridDim(seq_len, num_heads);
            auto stream = ctx.current_stream()->ptr;
            BM_DTYPE_DISPATCH_HALF(q.dtype(), {
                KERNEL_rope_with_cache<<<gridDim, dim_head, 0, stream>>>(
                    cos.data<float>(),
                    sin.data<float>(),
                    q.data<scalar_t>(),
                    out_q.data<scalar_t>(),
                    src_stride1, src_stride2, dst_stride1, dst_stride2,
                    neox_style
                );
            });
            BM_CUDART_ASSERT(cudaGetLastError());
        }
    }

    void rotate_inplace(
        const core::Context& ctx,
        const core::Tensor& pos, // (batch, seq_len)
        core::Tensor& q          // (batch, seq_len, dim_model)
    ) override {
        auto m_ctx = model::ModelContext::cast(ctx);
        if (m_ctx && m_ctx->dyn_batch() && m_ctx->dyn_batch()->rope_cache.cos.numel() > 0) {
            auto& cos = m_ctx->dyn_batch()->rope_cache.cos;
            auto& sin = m_ctx->dyn_batch()->rope_cache.sin;
            BM_ASSERT_EQ(pos.size(0), cos.size(0), "pos and cos dim0 mismatch");
            Tensor out = q;
            rotate_with_cache(ctx, cos, sin, q, out);
        } else if (!neox_style) {
            Tensor out = q;
            rotate_NOT_neox_style(ctx, pos, q, out);
        } else {
            throw std::runtime_error("rotate_inplace is not supported in NormalImpl");
        }
    }


    Tensor rotate(
        const core::Context& ctx,
        const core::Tensor& pos, // (batch, seq_len)
        const core::Tensor& q,   // (batch, seq_len, dim_model)
        core::Tensor* output
        ) override {
        auto m_ctx = model::ModelContext::cast(ctx);
        auto out_q = output ? *output : ctx.tensor(q.size(), q.dtype());
        if (m_ctx && m_ctx->dyn_batch() && m_ctx->dyn_batch()->rope_cache.cos.numel() > 0) {
            auto& cos = m_ctx->dyn_batch()->rope_cache.cos;
            auto& sin = m_ctx->dyn_batch()->rope_cache.sin;
            BM_ASSERT_EQ(pos.size(0), cos.size(0), "pos and cos dim0 mismatch");
            // if (ctx.is_layer(0)) std::cout << "rotate_with_cache\n";
            rotate_with_cache(ctx, cos, sin, q, out_q);
            return out_q;
        }
        if (!neox_style) {
            rotate_NOT_neox_style(ctx, pos, q, out_q);
            return out_q;
        }
        BM_ASSERT_EQ(q.ndim(), pos.ndim() + 1, "Dim mismatch");
        BM_ASSERT(q.size(-1) % dim_head == 0, "dim_model mismatch");
        BM_ASSERT_EQ(q.size(0), pos.size(0), "shape mismatch");
        if (pos.ndim() > 1)
            BM_ASSERT_EQ(q.size(1), pos.size(1), "shape mismatch");
        if (output)
            BM_ASSERT_EQ(output->size(), q.size(), "shape mismatch");

        int batch = (q.ndim() == 2) ? 1 : q.size(0);
        int pos_stride = (pos.ndim() == 1) ? 0 : pos.stride(0);
        int seq_len = q.size(-2);
        {
            int hidden_stride = (q.ndim() == 2) ? 0 : q.stride(0);
            int dim_model = q.size(-1);
            int threads = min(1024, round_up(dim_model, 32));
            dim3 gridDim(seq_len, batch, round_up(dim_model, threads) / threads);
            dim3 blockDim(threads, 1, 1);
            auto stream = ctx.current_stream()->ptr;
            BM_DTYPE_DISPATCH_FLOAT(q.dtype(), {
                auto kernel = KERNEL_rotary_embedding<scalar_t, /*IsDynamic=*/false>;
                if (rope_scaling_type == "dynamic") {
                    kernel = KERNEL_rotary_embedding<scalar_t, /*IsDynamic=*/true>;
                }
                kernel<<<gridDim, blockDim, 0, stream>>>(
                    dim_model,
                    dim_head,
                    hidden_stride,
                    pos_stride,
                    pos.data<int32_t>(),
                    q.data<scalar_t>(),
                    out_q.data<scalar_t>(),
                    rope_theta,
                    max_position_embeddings,
                    scaling_factor);
            });
            BM_CUDART_ASSERT(cudaGetLastError());
        }
        return out_q;
    }
    std::tuple<core::Tensor, core::Tensor> forward(
        const core::Context& ctx,
        const core::Tensor& pos, // (batch, seq_len)
        const core::Tensor& q,   // (batch, seq_len, dim_model)
        const core::Tensor& k    // (batch, seq_len, dim_model)
    ) {
        Tensor out_q = rotate(ctx, pos, q, nullptr);
        Tensor out_k = rotate(ctx, pos, k, nullptr);
        return std::make_tuple(out_q, out_k);
    }
};

// gridDim (seq_len, num_heads),   blockDim (dim_head, 1, 1)
template<typename T>
static __global__ void KERNEL_yarn_rope_neox_style(
    const int32_t* __restrict__ pos, // (seq_len)
    const T* __restrict__ in,        // (seq_len, num_heads, dim_head)
    T* __restrict__ out,             // (seq_len, num_heads, dim_head)
    uint32_t src_stride1,
    uint32_t src_stride2,
    uint32_t dst_stride1,
    uint32_t dst_stride2,
    float base,
    float scaling_factor,
    float low,
    float high,
    float _mscale,
    bool ignore_neox_style=false) {
    int m = pos[blockIdx.x];
    int dim_head = blockDim.x;
    int half_dim = dim_head / 2;
    int col = threadIdx.x;
    int i = (col < half_dim) ? col : (col - half_dim);  // 0 ~ half_dim
    float i_f = float(i);

    float pos_freq = powf(base, float(i * 2) / dim_head);
    float inv_freq_extrapolation = 1.0f / pos_freq;
    float inv_freq_interpolation = 1.0f / (scaling_factor * pos_freq);

    float ramp_mask;
    if (i_f <= low) {
        ramp_mask = 0.f;
    } else if (i_f >= high) {
        ramp_mask = 1.f;
    } else {
        ramp_mask = (i_f - low) / (high - low);
    }
    float inv_freq_mask = 1.f - ramp_mask;
    float inv_freq = inv_freq_interpolation * ramp_mask + inv_freq_extrapolation * inv_freq_mask;

    size_t src_offset = size_t(blockIdx.x) * src_stride1 + blockIdx.y * src_stride2 + col;
    size_t dst_offset = size_t(blockIdx.x) * dst_stride1 + blockIdx.y * dst_stride2 + col;

    float freq = m * inv_freq;
    float cos_freq = cos(freq) * _mscale;
    float sin_freq = sin(freq) * _mscale;
    if (col < half_dim) {
        out[dst_offset] = in[src_offset] * T(cos_freq) - in[src_offset + half_dim] * T(sin_freq);
    } else {
        out[dst_offset] = in[src_offset] * T(cos_freq) + in[src_offset - half_dim] * T(sin_freq);
    }
}
// gridDim (seq_len, num_heads),   blockDim (dim_head / 2)
template<typename T>
static __global__ void KERNEL_yarn_rope(
    const int32_t* __restrict__ pos, // (seq_len)
    const T* __restrict__ g_in,      // (seq_len, num_heads, dim_head)
    T* __restrict__ g_out,           // (seq_len, num_heads, dim_head)
    uint32_t src_stride1,
    uint32_t src_stride2,
    uint32_t dst_stride1,
    uint32_t dst_stride2,
    float base,
    float scaling_factor,
    float low,
    float high,
    float _mscale,
    bool OUT_neox_style=false) {
    int m = pos[blockIdx.x];
    int dim_head = blockDim.x * 2;
    int i = threadIdx.x;
    float i_f = float(i);

    float pos_freq = powf(base, float(i * 2) / dim_head);
    float inv_freq_extrapolation = 1.0f / pos_freq;
    float inv_freq_interpolation = 1.0f / (scaling_factor * pos_freq);

    float ramp_mask;
    if (i_f <= low) {
        ramp_mask = 0.f;
    } else if (i_f >= high) {
        ramp_mask = 1.f;
    } else {
        ramp_mask = (i_f - low) / (high - low);
    }
    float inv_freq_mask = 1.f - ramp_mask;
    float inv_freq = inv_freq_interpolation * ramp_mask + inv_freq_extrapolation * inv_freq_mask;

    size_t src_offset = size_t(blockIdx.x) * src_stride1 + blockIdx.y * src_stride2;
    size_t dst_offset = size_t(blockIdx.x) * dst_stride1 + blockIdx.y * dst_stride2;

    T in[2];
    T out[2];
    *reinterpret_cast<int*>(&in) = *reinterpret_cast<const int*>(&g_in[src_offset + i * 2]);
    float freq = m * inv_freq;
    float cos_freq = cos(freq) * _mscale;
    float sin_freq = sin(freq) * _mscale;

    out[0] = in[0] * T(cos_freq) - in[1] * T(sin_freq);  // x
    out[1] = in[0] * T(sin_freq) + in[1] * T(cos_freq);  // y
    if (OUT_neox_style) {
        g_out[dst_offset + i] = out[0];
        g_out[dst_offset + i + blockDim.x] = out[1];
    } else {
        *reinterpret_cast<int *>(&g_out[dst_offset + i * 2]) = *reinterpret_cast<int *>(&out);
    }
}

class RotaryEmbedding::impl::YarnImpl : public RotaryEmbedding::impl {
    double base;
    float attn_factor;
    int beta_fast { 0 };
    int beta_slow { 0 };
    double mscale { 0 };
    double mscale_all_dim { 0 };
    int original_max_position { 0 };
    bool neox_style;

    float low;
    float high;
    float _mscale;

    double yarn_find_correction_dim(double num_rotations, int dim) {
        double PI = 3.141592653589793;
        double max_pos = original_max_position;
        return (dim * log(max_pos / (num_rotations * 2 * PI))) / ( 2 * log(base));
    }

    double yarn_get_mscale(double scale, double mscale=1.) {
        if (scale <= 1.)
            return 1.;
        return 0.1 * mscale * log(scale) + 1.0;
    }

public:
    YarnImpl(const core::Context& ctx, model::ModelConfig cfg)
      : impl(ctx, cfg),
        base(cfg.rope_theta),
        attn_factor(cfg.rope_cfg.attn_factor),
        beta_fast(cfg.rope_cfg.beta_fast),
        beta_slow(cfg.rope_cfg.beta_slow),
        mscale(cfg.rope_cfg.mscale),
        mscale_all_dim(cfg.rope_cfg.mscale_all_dim),
        original_max_position(cfg.rope_cfg.original_max_position)
    {
        BM_ASSERT_EQ(cfg.qk_rope_head_dim % 64, 0, "");
        double low1 = floor(yarn_find_correction_dim(beta_fast, dim_head));
        double high1 = ceil(yarn_find_correction_dim(beta_slow, dim_head));
        low = max(low1, 0.);  // Clamp values just in case
        high = min(high1, dim_head - 1.);

        _mscale = yarn_get_mscale(scaling_factor) * attn_factor;
        if (cfg.model_type == "deepseek_v2" || cfg.model_type == "deepseek_v3") {
            _mscale = yarn_get_mscale(scaling_factor, mscale) /
                      yarn_get_mscale(scaling_factor, mscale_all_dim) *
                      attn_factor;
        }
        // high = max(low + 0.001, high); // Prevent singularity
//        if (ctx.rank() == 0 && ctx.current_layer() == 0) {
//            std::cout
//                << "beta_fast: " << beta_fast
//                << ", beta_slow: " << beta_slow
//                << ", low: " << low
//                << ", high: " << high << std::endl;
//        }
    }

    void do_rotate(
        const core::Context& ctx,
        const core::Tensor& pos, // (batch? seq_len)
        const core::Tensor& q,   // (batch? seq_len, dim_model) or (batch?, seq_len, num_heads, dim_head)
        core::Tensor& out_q
    ) {
        BM_ASSERT_EQ(core::DataType::kInt32, pos.dtype(), "dtype mismatch");
        BM_ASSERT(q.ndim() <= pos.ndim() + 2, "Dim mismatch");
        BM_ASSERT(q.size(-1) % dim_head == 0, "dim_model mismatch");
        BM_ASSERT_EQ(q.size(0), pos.size(0), "batch or seq_len mismatch");
        if (pos.ndim() > 1)
            BM_ASSERT_EQ(q.size(1), pos.size(1), "seq_len mismatch");
        if (q.ndim() == pos.ndim() + 2)
            BM_ASSERT_EQ(q.size(-1), dim_head, "dim_head mismatch");

        uint32_t src_stride1 = q.stride(-2);
        uint32_t src_stride2 = dim_head;
        uint32_t dst_stride1 = out_q.stride(-2);
        uint32_t dst_stride2 = dim_head;
        if (q.ndim() == pos.ndim() + 2) {
            src_stride1 = q.stride(-3);
            src_stride2 = q.stride(-2);
            dst_stride1 = out_q.stride(-3);
            dst_stride2 = out_q.stride(-2);
        }

        size_t seq_len = pos.numel();
        size_t num_heads = (q.ndim() == pos.ndim() + 2) ? q.size(-2) : (q.size(-1) / dim_head);
        if (ctx.is_layer(600)) {
            std::cout << "q.shape() " << q.shape() << std::endl;
            std::cout << "low=" << low << ", high=" << high << "\n";
            std::cout << "seq_len: " << seq_len << std::endl;
            std::cout << "num_heads: " << num_heads << std::endl;
            std::cout << "src_stride1: " << src_stride1 << std::endl;
            std::cout << "src_stride2: " << src_stride2 << std::endl;
            std::cout << "dst_stride1: " << dst_stride1 << std::endl;
            std::cout << "dst_stride2: " << dst_stride2 << std::endl;
            std::cout << "rope_theta: " << rope_theta << std::endl;
            std::cout << "scaling_factor: " << scaling_factor << std::endl;
            std::cout << "_mscale: " << _mscale << std::endl;
        }
        {
            dim3 gridDim(seq_len, num_heads);
            auto stream = ctx.current_stream()->ptr;
            BM_DTYPE_DISPATCH_HALF(q.dtype(), {
                auto kernel = KERNEL_yarn_rope<scalar_t>;
                size_t thread = dim_head / 2;
                if (neox_style) {
                    kernel = KERNEL_yarn_rope_neox_style<scalar_t>;
                    thread = dim_head;
                }
                kernel<<<gridDim, thread, 0, stream>>>(
                    pos.data<int32_t>(),
                    q.data<scalar_t>(),
                    out_q.data<scalar_t>(),
                    src_stride1, src_stride2, dst_stride1, dst_stride2,
                    rope_theta,
                    scaling_factor,
                    low,
                    high,
                    _mscale,
                    false);
            });
            BM_CUDART_ASSERT(cudaGetLastError());
        }
    }
    Tensor rotate(
        const core::Context& ctx,
        const core::Tensor& pos, // (batch? seq_len)
        const core::Tensor& q,   // (batch? seq_len, dim_model) or (batch?, seq_len, num_heads, dim_head)
        Tensor* output
    ) override {
        if (output)
            BM_ASSERT_EQ(output->size(), q.size(), "shape mismatch");
        Tensor out = output ? *output : ctx.tensor(q.size(), q.dtype());
        do_rotate(ctx, pos, q, out);
        return out;
    }
    void rotate_inplace(
        const core::Context& ctx,
        const core::Tensor& pos, // (batch? seq_len)
        core::Tensor& q          // (batch? seq_len, dim_model) or (batch?, seq_len, num_heads, dim_head)
    ) override {
        do_rotate(ctx, pos, q, q);
    }
    std::tuple<core::Tensor, core::Tensor> forward(
        const core::Context& ctx,
        const core::Tensor& pos, // (batch, seq_len)
        const core::Tensor& q,   // (batch, seq_len, dim_model)
        const core::Tensor& k    // (batch, seq_len, dim_model)
    ) {
        Tensor out_q = rotate(ctx, pos, q, nullptr);
        Tensor out_k = rotate(ctx, pos, k, nullptr);
        return std::make_tuple(out_q, out_k);
    }
};

RotaryEmbedding::RotaryEmbedding(const core::Context& ctx, model::ModelConfig cfg) {
    if (cfg.rope_cfg.type == "yarn") {
        pimpl = std::make_unique<impl::YarnImpl>(ctx, cfg);
    } else {
        pimpl = std::make_unique<impl::NormalImpl>(ctx, cfg);
        if (cfg.model_type == "cohere")
            pimpl->neox_style = false;
    }
//    pimpl->print_cos_sin(ctx, 10);
};

RotaryEmbedding::~RotaryEmbedding() { }

std::tuple<core::Tensor, core::Tensor> RotaryEmbedding::forward(
    const core::Context& ctx,
    const core::Tensor& pos, // (batch?, seq_len)
    const core::Tensor& q,   // (batch?, seq_len, dim_model)
    const core::Tensor& k    // (batch?, seq_len, dim_model)
) {
    core::EventScope ev(ctx, "RotaryEmbedding2", 3);
    return pimpl->forward(ctx, pos, q, k);
}

core::Tensor RotaryEmbedding::rotate(
    const core::Context& ctx,
    const core::Tensor& pos, // (batch?, seq_len)
    const core::Tensor& q,   // (batch?, seq_len, dim_model) or (batch?, seq_len, num_heads, dim_head)
    core::Tensor* output
) {
    core::EventScope ev(ctx, "RotaryEmbedding", 3);
    return pimpl->rotate(ctx, pos, q, output);
}
void RotaryEmbedding::rotate_inplace(
    const core::Context& ctx,
    const core::Tensor& pos, // (batch?, seq_len)
    core::Tensor& q          // (batch?, seq_len, dim_model) or (batch?, seq_len, num_heads, dim_head)
) {
    core::EventScope ev(ctx, "RotaryEmbInplace", 3);
    pimpl->rotate_inplace(ctx, pos, q);
}

bool RotaryEmbedding::is_normal() const {
    return pimpl->rope_scaling_type.empty();
}
bool RotaryEmbedding::is_neox_style() const {
    return pimpl->neox_style;
}
}