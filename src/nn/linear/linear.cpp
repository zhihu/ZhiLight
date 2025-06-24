#include "nn/linear/linear.h"
#include "nn/linear/cuda_helper.h"
#include "nn/quant/awq/awq.h"
#include "nn/quant/gptq/gptq.h"
#include "nn/quant/fp8/fp8.h"
#include "nn/quant/marlin/marlin.h"
#include "nn/linear/activation_kernel.h"
#include "nn/functions/functions.h"
#include "nn/quant/int8/quant_kernel.h"
#include "model/model_context.h"
#include "model/dyn_batch_context.h"
#include <bmengine/functions/element.h>
#include <bmengine/functions/gemm.h>
#include <bmengine/functions/tensor_ops.h>
#include <bmengine/functions/transpose.h>
#include <bmengine/functions/typecast.h>
#include <bmengine/functions/init.h>
#include <bmengine/functions/all.h>
#include <bmengine/logger/std_log_op.hpp>
#include <bmengine/core/core.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <regex>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "utils/exception.h"
#include "utils/env.h"

#include "../../../3rd/deep_gemm/deep_gemm_api.h"

namespace nn {

using model::ModelContext;
using bmengine::core::DataType;
using bmengine::core::DistLayout;
using bmengine::core::ScopeDevice;
using bmengine::core::Tensor;
using std::string;
using std::unique_ptr;
using std::vector;

// clang-format off
// tensors will be updated to returned tensor's slice.
Tensor concat_dim0(const core::Context& ctx, std::vector<Tensor*> tensors, bool stack) {
    BM_ASSERT(!tensors.empty(), "");
    auto shape = tensors[0]->shape();
    auto shape_a = tensors[0]->shape();
    shape[0] = 0;
    std::vector<size_t> dim0s;
    std::vector<size_t> bytes;
    std::vector<void*> datas;
    for (Tensor* t : tensors) {
        if (stack)
            BM_ASSERT_EQ(shape_a, t->shape(), "shape mismatch");
        shape[0] += t->size(0);
        dim0s.push_back(t->size(0));
        bytes.push_back(t->nbytes());
        datas.push_back(t->data());
    }
    Tensor ret = ctx.tensor(shape, tensors[0]->dtype());

    auto stream = ctx.current_stream()->ptr;
    auto d2d = cudaMemcpyDeviceToDevice;
    char* dst = ret.data<char>();
    size_t dim0 = 0;
    std::vector<Tensor*> quant_scales;
    for (size_t i = 0; i < tensors.size(); ++i) {
        BM_CUDART_ASSERT(cudaMemcpyAsync(dst, datas[i], bytes[i], d2d, stream));
        auto name = tensors[i]->name();
        auto quant_scale = tensors[i]->quant_scale;
        if (quant_scale) quant_scales.push_back(quant_scale.get());
        *tensors[i] = ret.slice_dim0_len(dim0, dim0s[i]); // update tensors to slice
        tensors[i]->set_name(name);
        tensors[i]->quant_scale = quant_scale;
        dst += bytes[i];
        dim0 += dim0s[i];
    }
    if (stack) {
        shape[0] /= tensors.size();
        shape.insert(shape.begin(), tensors.size());
    }
    ret = ret.view(shape);
    if (!quant_scales.empty()) {
        Tensor fuse_scale = concat_dim0(ctx, quant_scales, stack);
        int8_op::set_quant_scale(ret, fuse_scale);
    }
    return ret;
}
static Tensor concat2_dim0(const core::Context& ctx, Tensor& a, Tensor& b) {
//    BM_ASSERT_EQ(a.shape(), b.shape(), "shape mismatch");
    if (a.empty()) return b;
    return concat_dim0(ctx, {&a, &b}, false);
}
static Tensor concat3_dim0(const core::Context& ctx, Tensor& q, Tensor& k, Tensor& v) {
    BM_ASSERT_EQ(q.shape(), k.shape(), "shape mismatch");
    BM_ASSERT_EQ(q.shape(), v.shape(), "shape mismatch");
    if (q.empty()) return {};
    return concat_dim0(ctx, {&q, &k, &v}, false);
}

static Tensor concat2_dim1(const core::Context& ctx, const Tensor& a, const Tensor& b) {
    return functions::concat_tensor(ctx, a, b, 1);
}
static Tensor concat3_dim1(const core::Context& ctx, const Tensor& q, const Tensor& k, const Tensor& v) {
    auto kv = functions::concat_tensor(ctx, k, v, 1);
    auto ret = functions::concat_tensor(ctx, q, kv, 1);
    return ret;
}

static vector<size_t> split_shape_dim0(const std::vector<size_t>& shape, size_t n_split) {
    BM_ASSERT_EQ(shape[0] % n_split, 0, "Not dividable");
    std::vector<size_t> new_shape = shape;
    new_shape[0] /= n_split;
    new_shape.insert(new_shape.begin(), n_split);
    return new_shape;
}

vector<Tensor> split_dim1(const core::Context& ctx, const Tensor& t, size_t n_split) {
    BM_ASSERT_EQ(t.size(1) % n_split, 0, "Not divisible");
    size_t part_size = t.size(1) / n_split;
    vector<Tensor> results(n_split);
    for (size_t i = 0; i < n_split; ++i) {
        results[i] = functions::slice_last_dim(ctx, t, i * part_size, part_size);
    }
    return results;
}

class Linear::impl {
public:
    class NormalLinear;
    class Int8Linear;
    class Fp8Linear;
    class Fp8Block;
    class Int4GPTQ;
    class GPTQMarlin;
    class AWQ;

    struct Fuser {
        typedef impl* (*FN_FUSE2)(const core::Context& ctx, impl& a, impl& b);
        typedef impl* (*FN_FUSE3)(const core::Context& ctx, impl& a, impl& b, impl& c);
        typedef impl* (*FN_FUSE_DIM0)(const core::Context& ctx, const std::vector<impl*>& layers);
        FN_FUSE2 fuse2 { nullptr };  // for FeedForward
        FN_FUSE3 fuse3 { nullptr }; // for Attention to fuse Q,K,V
        FN_FUSE_DIM0 fuse_dim0 { nullptr }; // MOE to fuse all experts
    };

    uint32_t dim_in;
    uint32_t dim_out;
    bool parallel;
    core::DistLayout dist_layout;
    std::string act_fn_type;
    bool weight_transposed;
    int quant;
    core::DataType dtype;
    bool has_bias { false };
    std::string prefix;

    impl(uint32_t dim_in, uint32_t dim_out, std::string act_fn, bool w_trans, int quant, DataType dtype, bool parallel)
        : dim_in(dim_in), dim_out(dim_out),
          act_fn_type(act_fn), weight_transposed(w_trans), quant(quant), dtype(dtype), parallel(parallel) { }
    virtual ~impl() = default;

    virtual void scale_output(float scale) = 0;
    virtual void set_output_type(core::DataType dtype) = 0;
    virtual void set_compute_type(cublasComputeType_t compute_type) {}

    virtual Fuser weight_fuser() const {
        return {nullptr, nullptr, nullptr};
    }

    virtual vector<impl*> split(const core::Context& ctx, size_t n_split, bool by_out) {
        return {};
    }

    virtual core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& input,
        const std::string& output_name,
        bool quant_back,
        Tensor* output) = 0;

    virtual core::Tensor& get_weight() = 0;
    virtual core::Tensor get_dequant_weight(const core::Context& ctx) {
        throw std::runtime_error("get_dequant_weight is not supported");
    };
    core::Tensor dequant_weight_by_mul_diag(const core::Context& ctx, size_t dim) {
        BM_ASSERT_LE(dim, 20000, "too big dim");
        std::vector<float> h_diag(dim * dim);
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                h_diag[i * dim + j] = i == j ? 1.f : 0.f;
            }
        }
        core::Tensor diag = ctx.tensor_of(h_diag, {dim, dim});
        diag = functions::typecast(ctx, diag, dtype);
        auto w = forward(ctx, diag, "", true, nullptr);
        w = functions::Transpose(ctx).forward(ctx, w);
        return w;
    };
    virtual core::Tensor* get_weight_scale() { return nullptr; }

    virtual void load_parameters(const core::Context& ctx, const std::string& prefix) {
        throw std::runtime_error("load_parameters for QuantImpl only");
    }

   virtual void load_state_dict(
        const core::Context& ctx,
        const std::map<std::string, const core::Tensor>& state_dict,
        const std::string& prefix,
        bool allow_missing) = 0;

    Tensor activate(const core::Context& ctx, const Tensor& ret) {
        // BM_ASSERT(act_fn_type.empty(), "");
        if (!act_fn_type.empty()) {
            ctx.recordEvent(act_fn_type, 2);
        }
        if (act_fn_type == "gelu") {
            gelu_inplace(ret, ctx.current_stream()->ptr);
        } else if (act_fn_type == "silu") {
            silu_inplace(ret, ctx.current_stream()->ptr);
        } else if (act_fn_type != "") {
            throw std::runtime_error(act_fn_type + " activation is not supported");
        }
        return ret;
    }

    virtual void set_has_bias(bool b) {
        if (b) throw std::runtime_error("Bias is not implemented");
    }
    Tensor add_bias(const core::Context& ctx, const Tensor& t, const Tensor& bias) {
        using BinaryOp = bmengine::functions::BinaryElementwiseOp;
        BinaryOp add_op(ctx, BinaryOp::Add);
        return add_op.broadcast_y(ctx, t, bias);
    }

    bool is_attn_proj() {
        return prefix.find("attn.project_") != string::npos;
    }
    bool is_ff_in() {
        return prefix.find(".w_in") != string::npos || prefix.find(".w_gated") != string::npos;
    }
    bool is_ff_out() {
        return prefix.find(".w_out") != string::npos;
    }
};

// =========================== normal linear ===========================
class Linear::impl::NormalLinear : public Linear::impl {
public:
    core::DistLayout dist_layout;
    float scale_factor;
    std::unique_ptr<Tensor> weight;
    Tensor bias;
    functions::Gemm gemm_A_B;
    functions::Gemm gemm_A_Btrans;

    NormalLinear(
        const core::Context& ctx,
        uint32_t dim_in,
        uint32_t dim_out,
        std::string act_fn_type,
        bool scale_weights,
        bool weight_transposed,
        core::DataType dtype,
        bool parallel,
        core::DistLayout dist_layout
        )
        : Linear::impl(dim_in, dim_out, act_fn_type, weight_transposed, 0, dtype, parallel),
          dist_layout(weight_transposed ? dist_layout : transpose_layout(dist_layout)),
          scale_factor(float(scale_weights ? 1.0 / sqrtf(dim_in) : 1.0)),
          gemm_A_B(ctx, dtype, false, false, scale_factor),
          gemm_A_Btrans(ctx, dtype, false, true, scale_factor)
    {
        std::vector<size_t> shape({
            weight_transposed ? dim_in : dim_out, // W^T
            weight_transposed ? dim_out : dim_in  // W
        });
        weight = std::make_unique<Tensor>(ctx.parameter(shape, dtype));
        if (ctx.high_precision() >= 1) {
            gemm_A_B.set_compute_type(CUBLAS_COMPUTE_32F);
            gemm_A_Btrans.set_compute_type(CUBLAS_COMPUTE_32F);
        }
    }

    ~NormalLinear() = default;

    void set_has_bias(bool b) override {
        has_bias = b;
    }

    Fuser weight_fuser() const override {
        return {
            reinterpret_cast<Fuser::FN_FUSE2>(NormalLinear::fuse2),
            reinterpret_cast<Fuser::FN_FUSE3>(NormalLinear::fuse3),
            nullptr
        };
    }

    static impl* fuse2(const core::Context& ctx, NormalLinear& a, NormalLinear& b) {
        BM_ASSERT_EQ(a.scale_factor, b.scale_factor, "scale_factor not equal");
        uint32_t dim_out = a.dim_out + b.dim_out;
        NormalLinear* ret = new NormalLinear(
            ctx, a.dim_in, dim_out, "", false, a.weight_transposed, a.dtype, false, core::DistLayout::ROW);
        Tensor weight = a.weight_transposed ?
                        concat2_dim1(ctx, *a.weight, *b.weight) :
                        concat2_dim0(ctx, *a.weight, *b.weight);
        ret->weight = std::make_unique<Tensor>(weight);
        ret->scale_factor = a.scale_factor;
        if (a.weight_transposed) {
            a.weight.reset();
            b.weight.reset();
        } else {
            BM_ASSERT(ret->weight->data() == a.weight->data(), "weight not match.");
        }
        if (a.has_bias)
            ret->bias = concat2_dim0(ctx, a.bias, b.bias);
        ret->has_bias = a.has_bias;
        return ret;
    }

    static NormalLinear* fuse3(const core::Context& ctx, NormalLinear& q, NormalLinear& k, NormalLinear& v) {
        BM_ASSERT_EQ(q.scale_factor, k.scale_factor, "scale_factor not equal");
        BM_ASSERT_EQ(q.scale_factor, v.scale_factor, "scale_factor not equal");
        uint32_t dim_out = q.dim_out + k.dim_out + v.dim_out;
        NormalLinear* ret = new NormalLinear(
            ctx, q.dim_in, dim_out, q.act_fn_type, false, q.weight_transposed, q.dtype, false, core::DistLayout::ROW);
        Tensor weight = q.weight_transposed ?
                        concat3_dim1(ctx, *q.weight, *k.weight, *v.weight) :
                        concat3_dim0(ctx, *q.weight, *k.weight, *v.weight);
        ret->weight = std::make_unique<Tensor>(weight);
        ret->scale_factor = q.scale_factor;
        if (q.weight_transposed) {
            q.weight.reset();
            k.weight.reset();
            v.weight.reset();
        } else {
            BM_ASSERT(ret->weight->data() == q.weight->data(), "weight not match.");
        }
        if (q.has_bias)
            ret->bias = concat3_dim0(ctx, q.bias, k.bias, v.bias);
        ret->has_bias = q.has_bias;
        return ret;
    }

    void load_state_dict(
        const core::Context& ctx,
        const std::map<std::string, const core::Tensor>& state_dict,
        const std::string& prefix,
        bool allow_missing) override {
        std::vector<size_t> shape({
            weight_transposed ? dim_in : dim_out, // W^T
            weight_transposed ? dim_out : dim_in  // W
        });
        weight = std::make_unique<Tensor>(ctx.parameter(shape, dtype));
        auto name = prefix + ".weight";
        ctx.load_parameter(weight.get(), name, state_dict, parallel, dist_layout);

        auto bias_layout = dist_layout == DistLayout::ROW ? DistLayout::COLUMNAR : DistLayout::REPLICATED;
        if (has_bias) {
            name = prefix + ".bias";
            bias = ctx.parameter({ dim_out }, dtype);
            ctx.load_parameter(&bias, name, state_dict, parallel, bias_layout);
        }
    }

    void scale_output(float scale) {
        gemm_A_B.scale_output(scale);
        gemm_A_Btrans.scale_output(scale);
    }
    void set_output_type(core::DataType dtype) {
        gemm_A_B.set_output_type(dtype);
        gemm_A_Btrans.set_output_type(dtype);
    }

    core::Tensor& get_weight() { return *weight; }
    core::Tensor get_dequant_weight(const core::Context& ctx) { return *weight; }

    core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& input,
        const std::string& output_name,
        bool quant_back,
        Tensor* output) override {
        /*
        Input: (seq_len, dim_in)
        Output: (seq_len, dim_out)
        */
        BM_ASSERT(input.ndim() == 2 || input.ndim() == 3, "Input must be 2D/3D");
        BM_ASSERT_EQ(input.dtype(), weight->dtype(), "Input data type mismatch");
        BM_ASSERT_EQ(input.device(), weight->device(), "Input and weight must be on the same device");

        core::Tensor ret; // (seq_len, dim_out)
        // x @ W^T
        if (!weight_transposed) {
            ret = gemm_A_Btrans.forward(ctx, input, *weight, output,  has_bias ? &bias : nullptr);
//            if (has_bias) {
//                ret = add_bias(ctx, ret, bias);
//            }
        } else {
            ret = gemm_A_B.forward(ctx, input, *weight);
        }

        // set name here to avoid memory allocation.
        ret.set_name(output_name);
        return activate(ctx, ret);
    }
};

// =========================== Int8Linear ===========================
class Linear::impl::Int8Linear : public Linear::impl {
public:
    core::Tensor weight; // (dim_out, dim_in)
    core::Tensor weight_scale;
    core::Tensor bias;
    std::string act_fn_type;
    bool scale_weights;
    bool weight_transposed;
    DistLayout dist_layout;
    int dev;

    cublasLtMatmulDesc_t matmul_desc;

    Fuser weight_fuser() const override {
        return {
            nullptr,
            reinterpret_cast<Fuser::FN_FUSE3>(Int8Linear::fuse3),
            nullptr
        };
    }
    static Int8Linear* fuse3(const core::Context& ctx, Int8Linear& q, Int8Linear& k, Int8Linear& v) {
        int dim_out = q.dim_out + k.dim_out + v.dim_out;
        auto dist_layout = transpose_layout(q.dist_layout);
        Int8Linear* ret = new Int8Linear(
            ctx, q.dim_in, dim_out, "", 2, false, false, q.dtype, q.parallel, dist_layout);
        ret->weight = concat3_dim0(ctx, q.weight, k.weight, v.weight);
        ret->weight_scale = concat3_dim0(ctx, q.weight_scale, k.weight_scale, v.weight_scale);
        if (q.has_bias)
            ret->bias = concat3_dim0(ctx, q.bias, k.bias, v.bias);
        ret->has_bias = q.has_bias;
        return ret;
    }

    Int8Linear(
        const core::Context& ctx,
        uint32_t dim_in,
        uint32_t dim_out,
        std::string act_fn_type,
        int quant,
        bool scale_weights,
        bool weight_transposed,
        core::DataType dtype,
        bool parallel,
        DistLayout dist_layout)
        : Linear::impl(dim_in, dim_out, act_fn_type, weight_transposed, quant, dtype, parallel),
          weight(ctx.parameter({ dim_out, dim_in }, core::DataType::kInt8)),
          weight_scale(ctx.parameter({ dim_out }, dtype)),
          scale_weights(scale_weights),
          weight_transposed(weight_transposed),
          dist_layout(weight_transposed ? dist_layout : transpose_layout(dist_layout)),
          dev(ctx.active_device_idx()) {
        BM_ASSERT(quant >= 0 && quant <= 2, "Wrong quant " + std::to_string(quant));

        BM_CUBLAS_ASSERT(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));

        cublasLtEpilogue_t epilogue;
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;

        BM_CUBLAS_ASSERT(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

        cublasOperation_t op_transpose = CUBLAS_OP_T;
        BM_CUBLAS_ASSERT(cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_transpose, sizeof(op_transpose)));
    }
    ~Int8Linear() {
        try {
            BM_CUBLAS_ASSERT(cublasLtMatmulDescDestroy(matmul_desc));
        } catch (const BMEngineException& e) { std::cerr << e.what() << std::endl; }
    }
    void set_has_bias(bool b) override {
        has_bias = b;
    }

    void scale_output(float scale) {
        // throw std::logic_error("Not supported");
    }
    void set_output_type(core::DataType dtype) {
        // throw std::logic_error("Not supported");
    }

    core::Tensor& get_weight() { return weight; }
    core::Tensor* get_weight_scale() { return &weight_scale; }

    void load_state_dict(
        const core::Context& ctx,
        const std::map<std::string, const core::Tensor>& state_dict,
        const std::string& prefix,
        bool allow_missing) override {
        if (quant == 2) {
            auto name = prefix + ".weight";
            Tensor tmp_weight = weight_transposed
                                ? ctx.tensor({dim_in, dim_out}, dtype)
                                : ctx.tensor({dim_out, dim_in}, dtype);
            ctx.load_parameter(&tmp_weight, name, state_dict, parallel, dist_layout);

            Tensor weight_t = weight_transposed
                              ? functions::Transpose(ctx).forward(ctx, tmp_weight)
                              : tmp_weight;

            Tensor w_q, w_scale;
            int8_op::quant_calc_scale(ctx, weight_t, &w_q, &w_scale);
            weight = w_q;

            if (scale_weights) {
                Tensor w_scale2 = ctx.tensor(w_scale.shape(), w_scale.dtype());
                multiply(ctx, w_scale, 1.0 / sqrt(dim_in), &w_scale2);
                w_scale = w_scale2;
            }
            weight_scale = functions::typecast(ctx, w_scale, dtype);
            auto bias_layout = dist_layout == DistLayout::ROW ? DistLayout::COLUMNAR : DistLayout::REPLICATED;
            if (has_bias) {
                name = prefix + ".bias";
                bias = ctx.parameter({ dim_out }, dtype);
                ctx.load_parameter(&bias, name, state_dict, parallel, bias_layout);
            }
        } else {
            throw ZhiLightException("quant value error: " + std::to_string(quant));
        }
    }

    /*
    Input: (seq_len, dim_in)
    Output: (seq_len, dim_out)
    */
    core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& input,
        const std::string& output_name,
        bool quant_back,
        Tensor* output) override {
        uint32_t dim_out = weight.size(0);
        uint32_t dim_in = weight.size(1);
        BM_ASSERT(input.ndim() == 2 || input.ndim() == 3, "Input must be 2D");
        BM_ASSERT_EQ(input.size(-1), dim_in, "Input size mismatch");
        BM_ASSERT_EQ(input.device(), weight.device(), "Input and weight must be on the same device");

        // quantization input
        core::Tensor input_quant; // (M, K)
        core::Tensor input_scale; // (M)
        if (input.quant_scale && input.quant_scale->dtype() == core::DataType::kInt8) {
            input_quant = *input.quant_scale;
            BM_ASSERT(input_quant.quant_scale, "quant tensor has no scale");
        } else if (!input.quant_scale) {
            core::EventScope event_scope(ctx, "quant_calc_scale");
            BM_ASSERT_EQ(input.dtype(), dtype, "Input data type mismatch1.");
            input_quant = int8_op::quant_calc_scale(ctx, input);
            int8_op::set_quant_scale(const_cast<Tensor&>(input), input_quant);
        } else {
            BM_ASSERT_EQ(input.dtype(), core::DataType::kInt8, "Input data type mismatch2");
            input_quant = input;
        }
        input_scale = *input_quant.quant_scale;

        auto ret_shape = input.shape();
        ret_shape[ret_shape.size() - 1] = dim_out;
        core::Tensor ret =
            ctx.tensor(ret_shape, core::DataType::kInt32, output_name, 32 * dim_out * sizeof(int));

        static int kAlignMInt8 = utils::get_int_env("GEMM_INT8_ALIGN_M", 32);
        size_t M = input.numel() / dim_in;
        M = kAlignMInt8 > 0 ? round_up(M, kAlignMInt8) : M;
        cublasLtMatrixLayout_t layout_weight, layout_input, layout_ret;
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutCreate(&layout_weight, CUDA_R_8I, dim_in, dim_out, dim_in));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutCreate(&layout_input, CUDA_R_8I, dim_in, M, dim_in));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutCreate(&layout_ret, CUDA_R_32I, dim_out, M, dim_out));

        int32_t alpha = 1, beta = 0;
        BM_CUBLAS_ASSERT(cublasLtMatmul(
            ctx.current_cublas_handle(),
            matmul_desc,
            &alpha,
            weight.data(),
            layout_weight,
            input_quant.data(),
            layout_input,
            &beta,
            ret.data(),
            layout_ret,
            ret.data(),
            layout_ret,
            nullptr,
            nullptr,
            0,
            ctx.current_stream()->ptr));

        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutDestroy(layout_weight));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutDestroy(layout_input));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutDestroy(layout_ret));

        // scale back
        if (!quant_back && !has_bias) {
            BM_ASSERT(act_fn_type.empty(), "Linear has activation");
            ret.quant_scale = std::make_unique<core::Tensor>();
            *ret.quant_scale = input_scale;
            return ret;
        }
        core::EventScope event_scope(ctx, "quant_scale_back");
        ret = int8_op::quant_scale_back(ctx, ret, &input_scale, &weight_scale, dtype);
        if (has_bias) {
            ret = add_bias(ctx, ret, bias);
        }
        return activate(ctx, ret);
    }
};

class Linear::impl::Int4GPTQ : public Linear::impl {
    const int group_size;
public:
    core::Tensor qweight; // GPTQ: (dim_in / 8, dim_out); AWQ: (dim_in, dim_out / 8)
    core::Tensor scales; // (dim_in / group_size, dim_out)
    core::Tensor qzeros; // (dim_in / group_size, dim_out / 8)
    core::Tensor g_idx;
    core::Tensor bias;
    core::Tensor q_perm_i16;
    core::Tensor rev_perm;

    bool scale_weights;
    DistLayout dist_layout;
    int dev;
    bool act_order { false };
    bool sym { false };
    bool use_exllama { false };
    bool dim_out_parallel {false };
    bool permute_ff_up_out;
    int size_n1 { 0 };
    int size_n2 { 0 };
    bool is_awq { false };
    bool new_kernel;
    bool trt_kernel;
    int w4_int8;
    int w4_fp8;
    bool loaded { false };

    Int4GPTQ(
        const core::Context& ctx,
        uint32_t dim_in,
        uint32_t dim_out,
        std::string act_fn_type,
        int quant,
        core::DataType dtype,
        bool parallel,
        DistLayout dist_layout,
        bool act_order,
        int group_size,
        bool sym)
        : Linear::impl(dim_in, dim_out, act_fn_type, false, quant, dtype, parallel),
          act_order(act_order),
          group_size(group_size),
          sym(sym),
          qweight(ctx.parameter({ dim_in / 8, dim_out }, DataType::kInt32)),
          qzeros(ctx.parameter({ dim_in / group_size, dim_out / 8 }, DataType::kInt32)),
          scales(ctx.parameter({ dim_in / group_size, dim_out }, dtype)),
          g_idx(ctx.parameter({ dim_in }, DataType::kInt32)),
          dist_layout(dist_layout),
          dev(ctx.active_device_idx()) {
        // DistLayout::COLUMNAR => split by dim_out
        use_exllama = dim_out_parallel = !parallel || !act_order || dist_layout == DistLayout::COLUMNAR;
        BM_ASSERT(quant >= 3, "Wrong quant type");
        BM_ASSERT(dim_in % group_size == 0, "dim_in % group_size != 0");
        permute_ff_up_out = act_order && utils::get_int_env("PERMUTE_FF_UP_OUT", 1) == 1;
        is_awq = utils::get_int_env("AWQ_USE_EXLLAMA", 0) == 1;
        if (is_awq) {
            qweight = ctx.parameter({dim_in, dim_out / 8}, DataType::kInt32);
            g_idx = Tensor();
        }
        int algo = utils::get_int_env("GPTQ_KERNEL_ALGO", 1);
        w4_int8 = utils::get_int_env("W4_INT8_ALGO", 0);
        w4_fp8 = utils::get_int_env("W4_FP8_ALGO", 0);
        BM_ASSERT(w4_int8 == 0 || w4_fp8 == 0, "Both W4_INT8_ALGO and W4_FP8_ALGO is set");
        new_kernel = algo >= 1;
        trt_kernel = sym && w4_int8 == 0 && w4_fp8 == 0 && algo == 2;
    }
    ~Int4GPTQ() override = default;

    void set_has_bias(bool b) override {
        if (b) {
            BM_ASSERT(new_kernel, "bias only support new kernel");
            has_bias = b;
            trt_kernel = false;
        }
    }

    void clear_weights() {
        qweight = scales = qzeros = g_idx = Tensor();
    }

    static impl* fuse_dim0(const core::Context& ctx, const std::vector<impl*>& layers) {
        BM_ASSERT(!layers.empty(), "");
        Int4GPTQ& a = *dynamic_cast<Int4GPTQ*>(layers[0]);
        BM_ASSERT(!a.has_bias, "");
        BM_ASSERT(!a.g_idx.numel(), "");
        std::vector<Tensor*> all_weights, all_zeros, all_scales;
        std::vector<Tensor*> all_g_idx, all_q_perm_i16, all_rev_perm;
        for (auto l: layers) {
            Int4GPTQ* m = dynamic_cast<Int4GPTQ*>(l);
            BM_ASSERT(m, "Wrong sub layers impl type");
            all_weights.push_back(&m->qweight);
            all_zeros.push_back(&m->qzeros);
            all_scales.push_back(&m->scales);
            all_g_idx.push_back(&m->g_idx);
            all_q_perm_i16.push_back(&m->q_perm_i16);
            all_rev_perm.push_back(&m->rev_perm);
        }

        Int4GPTQ* ret = new Int4GPTQ(
            ctx, a.dim_in, a.dim_out, "", a.quant, a.dtype, a.parallel, a.dist_layout, a.act_order, a.group_size, a.sym);
        ret->qweight = concat_dim0(ctx, all_weights);
        ret->qzeros = concat_dim0(ctx, all_zeros);
        ret->scales = concat_dim0(ctx, all_scales);
        if (a.g_idx.numel())
            ret->g_idx = concat_dim0(ctx, all_g_idx);
        if (a.q_perm_i16.numel())
            ret->g_idx = concat_dim0(ctx, all_q_perm_i16);
        if (a.rev_perm.numel())
            ret->g_idx = concat_dim0(ctx, all_rev_perm);
        ret->size_n1 = a.dim_out;
        ret->size_n2 = a.dim_out * 2;
        return ret;
    }

    vector<impl*> split(const core::Context& ctx, size_t n_split, bool by_out) override {
        BM_ASSERT(loaded, "");
        BM_ASSERT(new_kernel && use_exllama, "");
        BM_ASSERT(!has_bias, "");
        BM_ASSERT(!g_idx.numel(), "");
        std::vector<Int4GPTQ*> results(n_split);

        if (by_out) {
            BM_ASSERT_EQ(qweight.size(0) % n_split, 0, "Not divisible");
            auto v_weights = qweight.view(split_shape_dim0(qweight.shape(), n_split)).chunk();
            auto v_zeros = qzeros.view(split_shape_dim0(qzeros.shape(), n_split)).chunk();
            auto v_scales = scales.view(split_shape_dim0(scales.shape(), n_split)).chunk();

            for (size_t i = 0; i < n_split; ++i) {
                results[i] = new Int4GPTQ(
                    ctx, dim_in, dim_out / n_split, "", quant, dtype, parallel, dist_layout, act_order, group_size, sym);
                results[i]->qweight = ctx.copy(v_weights[i]);
                results[i]->qzeros = v_zeros[i];
                results[i]->scales = v_scales[i];
                results[i]->qweight.set_name(logger::str_cat(qweight.name(), "_part", i));
            }
        } else {
            auto v_weights = split_dim1(ctx, qweight, n_split);
            auto v_zeros = split_dim1(ctx, qzeros, n_split);
            auto v_scales = split_dim1(ctx, scales, n_split);

            for (size_t i = 0; i < n_split; ++i) {
                results[i] = new Int4GPTQ(
                    ctx, dim_in / n_split, dim_out, "", quant, dtype, parallel, dist_layout, act_order, group_size, sym);
                results[i]->qweight = v_weights[i];
                results[i]->qzeros = v_zeros[i];
                results[i]->scales = v_scales[i];
                results[i]->qweight.set_name(qweight.name());
            }
        }
        return *reinterpret_cast<vector<impl*>*>(&results);
    }

    Fuser weight_fuser() const override {
        return {
            reinterpret_cast<Fuser::FN_FUSE2>(Int4GPTQ::fuse2),
            reinterpret_cast<Fuser::FN_FUSE3>(Int4GPTQ::fuse3),
            Int4GPTQ::fuse_dim0,
        };
    }

    static Int4GPTQ* fuse2(const core::Context& ctx, Int4GPTQ& a, Int4GPTQ& b) {
        if (!a.use_exllama || !b.use_exllama || a.act_order)
            return nullptr;

        auto dim_out = a.dim_out + b.dim_out;
        Int4GPTQ* ret = new Int4GPTQ(
            ctx, a.dim_in, dim_out, "", a.quant, a.dtype, a.parallel, a.dist_layout, a.act_order, a.group_size, a.sym);
        if (a.new_kernel) {
            ret->qweight = concat2_dim0(ctx, a.qweight, b.qweight);
            ret->qzeros = concat2_dim0(ctx, a.qzeros, b.qzeros);
            ret->scales = concat2_dim0(ctx, a.scales, b.scales);
            if (a.g_idx.numel()) {
                ret->g_idx = concat2_dim0(ctx, a.g_idx, b.g_idx);
                ret->q_perm_i16 = concat2_dim0(ctx, a.q_perm_i16, b.q_perm_i16);
                ret->rev_perm = concat2_dim0(ctx, a.rev_perm, b.rev_perm); // TODO: check this
            }
            ret->qweight.set_name(a.qweight.name() + "[FUSED]");
        } else {
            ret->qweight = concat2_dim1(ctx, a.qweight, b.qweight);
            ret->qzeros = concat2_dim1(ctx, a.qzeros, b.qzeros);
            ret->scales = concat2_dim1(ctx, a.scales, b.scales);
            ret->g_idx = a.g_idx.numel() ? concat2_dim0(ctx, a.g_idx, b.g_idx) : Tensor();
            a.clear_weights();
            b.clear_weights();
        }
        if (a.has_bias)
            ret->bias = concat2_dim0(ctx, a.bias, a.bias);
        ret->has_bias = a.has_bias;
        ret->trt_kernel = a.trt_kernel;
        ret->size_n1 = a.dim_out;
        ret->size_n2 = a.dim_out + b.dim_out;
        ret->use_exllama = true;
        return ret;
    }

    static Int4GPTQ* fuse3(const core::Context& ctx, Int4GPTQ& q, Int4GPTQ& k, Int4GPTQ& v) {
        if (!q.use_exllama || !k.use_exllama || v.act_order)
            return nullptr;

        auto dim_out = q.dim_out + k.dim_out + v.dim_out;
        Int4GPTQ *ret = new Int4GPTQ(
            ctx, q.dim_in, dim_out, "", q.quant, q.dtype, q.parallel, q.dist_layout, q.act_order, q.group_size, q.sym);
        if (q.new_kernel) {
            ret->qweight = concat3_dim0(ctx, q.qweight, k.qweight, v.qweight);
            ret->qzeros = concat3_dim0(ctx, q.qzeros, k.qzeros, v.qzeros);
            ret->scales = concat3_dim0(ctx, q.scales, k.scales, v.scales);
            ret->qweight.set_name(q.qweight.name() + "kv[FUSED]");
        } else {
            ret->qweight = concat3_dim1(ctx, q.qweight, k.qweight, v.qweight);
            ret->qzeros = concat3_dim1(ctx, q.qzeros, k.qzeros, v.qzeros);
            ret->scales = concat3_dim1(ctx, q.scales, k.scales, v.scales);
            ret->g_idx = q.g_idx.numel() ? concat3_dim0(ctx, q.g_idx, k.g_idx, v.g_idx) : Tensor();
            q.clear_weights();
            k.clear_weights();
            v.clear_weights();
        }
        if (q.has_bias)
            ret->bias = concat3_dim0(ctx, q.bias, k.bias, v.bias);
        ret->has_bias = q.has_bias;
        ret->trt_kernel = q.trt_kernel;
        ret->size_n1 = q.dim_out;
        ret->size_n2 = q.dim_out + k.dim_out;
        ret->use_exllama = true;
        return ret;
    }

    void scale_output(float scale) {
        // throw std::logic_error("Not supported");
    }
    void set_output_type(core::DataType dtype) {
        // throw std::logic_error("Not supported");
    }

    core::Tensor& get_weight() { return qweight; }
    core::Tensor* get_weight_scale() { return &scales; }
    core::Tensor get_dequant_weight(const core::Context& ctx) {
        if (new_kernel && use_exllama) {
            return nn::gptq::dequant_k_major(ctx, qweight, qzeros, scales);
        }
        size_t K = qweight.size(0) * 8; // dim_in;
        size_t N = qweight.size(1); // dim_out;
        size_t num_group = qzeros.size(0);
        Tensor ret = ctx.tensor({K, N}, dtype); // Col Major
        if (use_exllama) {
            nn::gptq::reconstruct_exllama(
                qweight.data<uint32_t>(),
                qzeros.data<uint32_t>(),
                scales.data<half>(),
                g_idx.numel() ? g_idx.data<int>() : nullptr,
                ret.mutable_data<half>(),
                K,
                N,
                num_group,
                ctx.current_stream()->ptr,
                size_n1,
                size_n2
            );
        } else {
            BM_ASSERT(g_idx.numel() > 0, "");
            nn::gptq::reconstruct_gptq(
                qweight.data<uint32_t>(),
                qzeros.data<uint32_t>(),
                scales.data<half>(),
                g_idx.numel() ? g_idx.data<int>() : nullptr,
                ret.mutable_data<half>(),
                K,
                N,
                num_group,
                ctx.current_stream()->ptr
            );
        }
        // std::cout << "dequant: " << ret;
        ret = functions::Transpose(ctx).forward(ctx, ret);
        // functions::check_numeric(ctx, ret);
        return ret;
    }

    void dequant_cache_weight(ModelContext& ctx, const core::Tensor& fake_input) {
        if (new_kernel && use_exllama && !trt_kernel) {
            nn::gptq::gptq_gemm_k_major(
                ctx, fake_input, qweight, qzeros, scales, q_perm_i16, rev_perm, has_bias ? &bias : nullptr, sym, true);
        }
    }

    /*
    * Input: (seq_len, dim_in)
    * Output: (seq_len, dim_out)
    */
    core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& input,
        const std::string& output_name,
        bool quant_back,
        Tensor* output) override {
        ModelContext* m_ctx = const_cast<ModelContext*>(dynamic_cast<const ModelContext*>(&ctx));
        model::DynBatchContext *dyn_ctx = m_ctx ? m_ctx->dyn_batch().get() : nullptr;
        bool dual_stream = m_ctx ? m_ctx->dual_stream() : false;

        static int w4_int8_encode_only = utils::get_int_env("W4_INT8_ENCODE_ONLY", 1);
        bool w4_a8_enc_only = w4_int8 > 0 && w4_int8_encode_only > 0 && dyn_ctx;
        static int w4_int8_attn = utils::get_int_env("W4_INT8_ATTN", 0);
        bool skip_int8 = (w4_int8_attn == 0 && prefix.find(".attn.") != std::string::npos);

        bool size_legal = dyn_ctx
            && dyn_ctx->e_token.numel() && dyn_ctx->s_token.numel()
            && input.size(0) == dyn_ctx->e_token.size(0) + dyn_ctx->s_token.size(0);

        if (new_kernel && use_exllama) {
            size_t K = input.size(-1);
            uint32_t N = qweight.size(0);
            Tensor* bias_p = has_bias ? &bias : nullptr;

            if (w4_a8_enc_only && !skip_int8 && !has_bias && dyn_ctx->s_token.numel() && size_legal) {
                Tensor ret = ctx.tensor({input.size(0), N}, input.dtype());
                int num_e = dyn_ctx->e_token.size(0);
                int num_s = dyn_ctx->s_token.size(0);
                Tensor input_e = input.slice_dim0_len(0, num_e);
                Tensor ret_e = ret.slice_dim0_len(0, num_e);
                Tensor input_s = input.slice_dim0_len(num_e, num_s);
                Tensor ret_s = ret.slice_dim0_len(num_e, num_s);
                // W4A16 gemm decode
                ctx.recordEvent("Start>gemm_w16_decode", 2);
                m_ctx->set_dual_stream(false);
                Tensor ret1 = nn::gptq::gptq_gemm_k_major(
                    ctx, input_s, qweight, qzeros, scales, q_perm_i16, rev_perm, bias_p, sym, false, &ret_s);
                BM_ASSERT_EQ(ret1.data<char>(), ret_s.data<char>(), "");
                ctx.recordEvent("End>gemm_w16_decode", 2);
                // W4A8 gemm encode
                m_ctx->set_dual_stream(dual_stream);
                Tensor ret2 = nn::gptq::gptq_gemm_k_major(
                    ctx, input_e, qweight, qzeros, scales, q_perm_i16, rev_perm, bias_p, sym, false, &ret_e);
                BM_ASSERT_EQ(ret2.data<char>(), ret_e.data<char>(), "");
                return activate(ctx, ret);
            } else {
                core::Tensor ret = ctx.tensor({input.size(0), N}, input.dtype());
                ret = nn::gptq::gptq_gemm_k_major(
                    ctx, input, qweight, qzeros, scales, q_perm_i16, rev_perm, bias_p, sym, false, &ret);
                BM_ASSERT(ret.numel(), "");
                return activate(ctx, ret);
            }
        }

        uint32_t dim_out = qweight.size(-1);
        uint32_t dim_in = qweight.size(0) * 8;
//        std::cout << "g_idx " << g_idx.shape() << " world size " << ctx.world_size() << endl;

        BM_ASSERT(input.ndim() == 2, "Input must be 2D");
        BM_ASSERT_EQ(input.size(-1), dim_in, "Input size mismatch");
        BM_ASSERT_EQ(input.dtype(), DataType::kHalf,
                  "Input data type mismatch");
        BM_ASSERT_EQ(input.device(), qweight.device(), "Input and weight must be on the same device");

        BM_ASSERT(size_n1 > 0 && size_n2 > 0, "");
        core::Tensor ret = nn::gptq::gptq_gemm(
            ctx, input, qweight, qzeros, scales, g_idx, use_exllama, group_size, size_n1, size_n2);

        return activate(ctx, ret);
    }

    void load_parameters(const core::Context& ctx, const std::string& prefix) override {
        throw std::runtime_error("Unsupported");
    }

    std::vector<int> argsort_cpu(const core::Context& ctx, Tensor g) {
        int* group = g.data<int>();
        if (g.device() >= 0) {
            group = new int[g.numel()];
            g.to_buffer(group);
        }
        std::vector<int> idx(g.numel());
        std::vector<int> count(g.numel() / group_size, 0);
        int numel = g.numel();
        for (int i = 0; i < numel; ++i) {
            int x = group[i];
            int y = x * group_size + count[x]++;
            BM_ASSERT_LE(y, numel, "");
            BM_ASSERT_LE(count[x], group_size, "");
            idx[y] = i;
        }
        if (g.device() >= 0) delete[] group;
        return std::move(idx);
    }

    template<class T>
    Tensor load_permute_out(const core::Context& ctx, Tensor param, std::vector<int>& out_idx) {
        // BM_ASSERT_EQ(param.size(0), size_t(dim_in / 8), "dim_in mismatch");
        BM_ASSERT_EQ(param.size(-1), out_idx.size(), "dim_out mismatch");
        BM_ASSERT_EQ(sizeof(T), core::get_elem_size(param.dtype()), "type mismatch");
        size_t out_size = out_idx.size();
        size_t shard_size = out_size / (parallel ? ctx.world_size() : 1);
        if (parallel)
            BM_ASSERT_EQ(out_size % ctx.world_size(), 0, "");
        size_t num_row = param.size(0);
        const T* src = param.data<T>();
        Tensor weight = ctx.tensor({num_row, shard_size}, param.dtype());
        T* buf = new T[weight.numel()];
        size_t col_offset = parallel ? ctx.rank() * shard_size : 0;
        for (size_t r = 0; r < num_row; ++r) {
            // permute out dim in range [0, shard_size]
            for (int k = 0; k < shard_size; ++k) {
                int idx = out_idx[col_offset + k];
                BM_ASSERT_LE(idx, out_size, "");
                buf[r * shard_size + k] = src[r * out_size + idx];
            }
        }
        weight.from_buffer(buf);
        delete[] buf;
        return weight;
    }

    Tensor load_permute_q_zeros(const core::Context& ctx, Tensor param, std::vector<int>& out_idx) {
        // qzeros: (dim_in / group_size, dim_out / 8)
        BM_ASSERT_EQ(param.size(0), size_t(dim_in / group_size), "dim_in mismatch");
        BM_ASSERT_EQ(param.size(-1), out_idx.size() / 8, "dim_out mismatch");
        if (parallel)
            BM_ASSERT(param.size(-1) % ctx.world_size() == 0, "can't mod");
        size_t out_size = out_idx.size();
        size_t shard_size = out_size / (parallel ? ctx.world_size() : 1);
        size_t num_group = param.size(0);
        const uint32_t* src = param.data<uint32_t>();
        Tensor q_zeros = ctx.tensor({num_group, shard_size / 8}, param.dtype());
        std::vector<uint32_t> buf(num_group * shard_size / 8, 0);
        size_t col_offset = parallel ? ctx.rank() * shard_size : 0;
        for (size_t row = 0; row < num_group; ++row) {
            // permute out dim in range [0, shard_size]
            for (int k = 0; k < shard_size; ++k) {
                int idx = out_idx[col_offset + k];
                uint32_t z8 = src[row * out_size / 8 + idx / 8];
                uint32_t shift = (idx % 8) * 4;
                uint32_t v = (z8 >> shift) & 0x0f;
                shift = (k % 8) * 4;
                buf[row * shard_size / 8 + k / 8] |= v << shift;
            }
        }
        q_zeros.from_buffer(buf.data());
        return q_zeros;
    }

    void transpose_weight(const core::Context& ctx) {
        if (new_kernel && use_exllama) {
            qzeros = nn::gptq::q4_to_q8(ctx, qzeros);
            functions::Transpose transpose(ctx);
            qweight = transpose(ctx, qweight);
            qzeros = transpose(ctx, qzeros);
            scales = transpose(ctx, scales);
            if (act_order) {
                q_perm_i16 = nn::gptq::int32_to_int16(ctx, g_idx);
                rev_perm = nn::gptq::reverse_perm(ctx, g_idx);
            }
            calc_w4a8_scale(ctx);
        }
        qweight.set_name(prefix);
    }

    void calc_w4a8_scale(const core::Context& ctx) {
        if (w4_int8 >= 1) {
            Tensor w16 = nn::gptq::dequant_k_major(ctx, qweight, qzeros, scales);
            Tensor w_max = functions::reduce_abs_max(ctx, w16, 1);
            w_max = functions::typecast(ctx, w_max, DataType::kFloat);
            Tensor q_max = ctx.tensor_of(std::vector<float>(w_max.numel(), 127.f));
            functions::BinaryElementwiseOp div_op(ctx, functions::BinaryElementwiseOp::Div);
            Tensor q_scales = div_op.forward(ctx, w_max, q_max);
            int8_op::set_quant_scale(qweight, q_scales);
            if (ctx.rank() == 110 && prefix.find("project_k") != std::string::npos) {
                auto w8 = nn::gptq::dequant_k_major(ctx, qweight, qzeros, scales, 1);
                std::cout << "w_max: " << w_max << endl;
                std::cout << "q_scale: " << q_scales << endl;
                std::cout << "w8: " << w8 << endl;
                w8 = int8_op::quant_calc_scale(ctx, w16);
                std::cout << "w8_BASE: " << w8 << endl;
                std::cout << "base_scale: " << *w8.quant_scale << endl;
                std::cout << "w16: " << w16 << endl;
            }
        } else if (w4_fp8 == 1) {
            Tensor w16 = nn::gptq::dequant_k_major(ctx, qweight, qzeros, scales);
            static int MAX_WEIGHT_E4M3 = utils::get_int_env("MAX_WEIGHT_E4M3", 256); // 448
            Tensor fp8_scale = nn::fp8::calc_scale(ctx, w16, (float)MAX_WEIGHT_E4M3); // shape: (1) type: float
            if (ctx.rank() == 0 && ctx.current_layer() == 0 && prefix.find("project_k") != std::string::npos) {
                auto w8 = nn::gptq::dequant_k_major(ctx, qweight, qzeros, scales, 1);
                std::cout << "fp8_scale: " << fp8_scale << endl;
            }
            int8_op::set_quant_scale(qweight, fp8_scale);
        }
    }

    void preprocess_weight(const core::Context& ctx, bool trans=true) {
        if (is_awq) {
            // (dim_in, dim_out / 8) => (dim_in / 8, dim_out)
            qweight = nn::gptq::shuffle_awq(ctx, qweight, use_exllama);
            nn::gptq::un_shuffle(ctx, qzeros);
        } else if (use_exllama) {
            g_idx = act_order ? ctx.tensor_of(argsort_cpu(ctx, g_idx)) : Tensor();
            nn::gptq::gptq_shuffle(ctx, qweight, g_idx);
            nn::gptq::increase_zero(ctx, qzeros);
        } else {
            nn::gptq::increase_zero(ctx, qzeros);
        }
        if (trans) {
            transpose_weight(ctx); // new_kernel only
        }
        qweight.set_name(prefix);
        if (trans) {
            size_n2 = size_n1 = qweight.size(-1);
//            Tensor w = get_dequant_weight(ctx);
//            functions::check_numeric(ctx, w);
        }
    }

    void load_state_dict(
        const core::Context& ctx,
        const std::map<std::string, const core::Tensor>& state_dict,
        const std::string& prefix,
        bool allow_missing) override {
        this->prefix = prefix;
        if (permute_ff_up_out && is_ff_in()) {
            ctx.load_parameter(&g_idx, prefix + ".g_idx", state_dict, false, dist_layout);

            auto out_prefix = std::regex_replace(prefix, std::regex("w_in|w_gated"), "w_out");
            // if (ctx.current_layer() == 0) std::cout << "out_prefix " << out_prefix << endl;
            Tensor out_g_idx = state_dict.at(out_prefix + ".g_idx");
            auto out_idx = argsort_cpu(ctx, out_g_idx);

            Tensor weight_cpu = state_dict.at(prefix + ".qweight");
            Tensor scales_cpu = state_dict.at(prefix + ".scales");
            Tensor qzeros_cpu = state_dict.at(prefix + ".qzeros");
            // permute and slice dim_out
            qweight = load_permute_out<int>(ctx, weight_cpu, out_idx);
            scales = load_permute_out<short>(ctx, scales_cpu, out_idx);
            qzeros = load_permute_q_zeros(ctx, qzeros_cpu, out_idx);
            use_exllama = true;
            preprocess_weight(ctx);
            return;

        } else if (permute_ff_up_out && is_ff_out()) {
            ctx.load_parameter(&qweight, prefix + ".qweight", state_dict, false, dist_layout);
            ctx.load_parameter(&g_idx, prefix + ".g_idx", state_dict, false, dist_layout);
            ctx.load_parameter(&scales, prefix + ".scales", state_dict, parallel, dist_layout);
            ctx.load_parameter(&qzeros, prefix + ".qzeros", state_dict, parallel, dist_layout);
            use_exllama = true;
            preprocess_weight(ctx, false);  // gptq_shuffle by g_idx
            if (parallel) {
                // slice after shuffle
                BM_ASSERT(qweight.size(0) % ctx.world_size() == 0, "");
                size_t shard_size = qweight.size(0) / ctx.world_size();
                qweight = qweight.slice_dim0_len(ctx.rank() * shard_size, shard_size);
                qweight = ctx.copy(qweight);
            }
            // transpose after slice
            transpose_weight(ctx);
            // the input is already permuted, clear q_perm
            act_order = false;
            g_idx = Tensor();
            q_perm_i16 = Tensor();
            rev_perm = Tensor();
            size_n2 = size_n1 = qweight.size(-1);
            return;
        }

        // QKV, FF up: COLUMNAR parallel (ColMajor) => dim_out_parallel=True
        // ATTN_OUT, FF down: ROW parallel
        // g_idx is dim_in
        auto idx_layout = dist_layout == DistLayout::ROW ? DistLayout::COLUMNAR : DistLayout::REPLICATED;
        if (!parallel || dim_out_parallel) {
            ctx.load_parameter(&qweight, prefix + ".qweight", state_dict, parallel, dist_layout);
            ctx.load_parameter(&scales, prefix + ".scales", state_dict, parallel, dist_layout);
            ctx.load_parameter(&qzeros, prefix + ".qzeros", state_dict, parallel, dist_layout);
            if (act_order)
                ctx.load_parameter(&g_idx, prefix + ".g_idx", state_dict, false, idx_layout);
        } else {
            // dim_in/row parallel
            ctx.load_parameter(&qweight, prefix + ".qweight", state_dict, parallel, dist_layout);
            if (act_order) {
                // parallel by g_idx
                ctx.load_parameter(&g_idx, prefix + ".g_idx", state_dict, parallel, idx_layout);
                ctx.load_parameter(&scales, prefix + ".scales", state_dict, false, dist_layout);
                ctx.load_parameter(&qzeros, prefix + ".qzeros", state_dict, false, dist_layout);
            } else {
                ctx.load_parameter(&scales, prefix + ".scales", state_dict, parallel, dist_layout);
                ctx.load_parameter(&qzeros, prefix + ".qzeros", state_dict, parallel, dist_layout);
            }
        }

        auto bias_layout = dist_layout == DistLayout::COLUMNAR ? DistLayout::COLUMNAR : DistLayout::REPLICATED;
        if (has_bias) {
            bias = ctx.parameter({ dim_out }, dtype);
            ctx.load_parameter(&bias, prefix + ".bias", state_dict, parallel, bias_layout);
        }

        preprocess_weight(ctx);
        loaded = true;
    }
};

class Linear::impl::GPTQMarlin : public Linear::impl {
    const uint32_t pack_factor { 32 / 4 };  // 32 bits(int) / 4 bits
    const int group_size;
public:
    // (K / 16, N * 2) after repack
    core::Tensor qweight; // GPTQ: (dim_in / 8, dim_out); AWQ: (dim_in, dim_out / 8)
    core::Tensor scales; // (dim_in / group_size, dim_out)
    core::Tensor qzeros; // (dim_in / group_size, dim_out / 8)
    core::Tensor g_idx;
    core::Tensor bias;
    core::Tensor workspace;
    bool act_order;
    bool sym;
    bool is_k_full { true };
    functions::Gemm gemm;

    bool scale_weights;
    DistLayout dist_layout;
    int dev;
    bool fuse_qkv;
    bool fuse_ff_in;
    bool loaded { false };

    GPTQMarlin(
        const core::Context& ctx,
        uint32_t dim_in,
        uint32_t dim_out,
        std::string act_fn_type,
        int quant,
        core::DataType dtype,
        bool parallel,
        DistLayout dist_layout,
        bool act_order,
        int group_size,
        bool sym)
        : Linear::impl(dim_in, dim_out, act_fn_type, false, quant, dtype, parallel),
          group_size(group_size),
          act_order(act_order),
          sym(sym),
          qweight(ctx.parameter({ dim_in / pack_factor, dim_out }, DataType::kInt32)),
          qzeros(ctx.parameter({ dim_in / group_size, dim_out / pack_factor }, DataType::kInt32)),
          scales(ctx.parameter({ dim_in / group_size, dim_out }, dtype)),
          gemm(ctx, dtype, false, false),
          dist_layout(dist_layout),
          dev(ctx.active_device_idx()) {
        BM_ASSERT_EQ(quant, 8, "Wrong quant type");
        BM_ASSERT(dim_in % group_size == 0, "dim_in % group_size != 0");
        gemm.set_compute_type(CUBLAS_COMPUTE_32F);
        workspace = ctx.tensor({1024 * 1024}, DataType::kFloat);
        functions::zeros_(ctx, workspace);
        fuse_qkv = utils::get_int_env("CPM_FUSE_QKV", 0);
        fuse_ff_in = utils::get_int_env("CPM_FUSE_FF_IN", 0);
    }
    ~GPTQMarlin() = default;

    virtual void set_has_bias(bool b) {
        has_bias = b;
    }

    void clear_weights() {
        qweight = scales = qzeros = Tensor();
    }

    Fuser weight_fuser() const override {
        return {
            reinterpret_cast<Fuser::FN_FUSE2>(GPTQMarlin::fuse2),
            reinterpret_cast<Fuser::FN_FUSE3>(GPTQMarlin::fuse3),
            nullptr
        };
    }

    static GPTQMarlin* fuse2(const core::Context& ctx, GPTQMarlin& a, GPTQMarlin& b) {
        if (a.loaded || b.loaded) return nullptr;
        BM_ASSERT(!a.loaded, "");
        auto dim_out = a.dim_out + b.dim_out;
        GPTQMarlin* ret = new GPTQMarlin(
            ctx, a.dim_in, dim_out, "", a.quant, a.dtype, a.parallel, a.dist_layout, a.act_order, a.group_size, a.sym);
        ret->qweight = concat2_dim1(ctx, a.qweight, b.qweight);
        ret->qzeros = concat2_dim1(ctx, a.qzeros, b.qzeros);
        ret->scales = concat2_dim1(ctx, a.scales, b.scales);
        a.clear_weights();
        b.clear_weights();
        ret->post_load(ctx);
        if (a.has_bias)
            ret->bias = concat2_dim0(ctx, a.bias, a.bias);
        ret->has_bias = a.has_bias;
        return ret;
    }

    static GPTQMarlin* fuse3(const core::Context& ctx, GPTQMarlin& q, GPTQMarlin& k, GPTQMarlin& v) {
        BM_ASSERT(!q.loaded, "");
        auto dim_out = q.dim_out + k.dim_out + v.dim_out;
        GPTQMarlin* ret = new GPTQMarlin(
            ctx, q.dim_in, dim_out, "", q.quant, q.dtype, q.parallel, q.dist_layout, q.act_order, q.group_size, q.sym);
        ret->qweight = concat3_dim1(ctx, q.qweight, k.qweight, v.qweight);
        ret->qzeros = concat3_dim1(ctx, q.qzeros, k.qzeros, v.qzeros);
        ret->scales = concat3_dim1(ctx, q.scales, k.scales, v.scales);
        q.clear_weights();
        k.clear_weights();
        v.clear_weights();
        ret->post_load(ctx);
        if (q.has_bias)
            ret->bias = concat3_dim0(ctx, q.bias, k.bias, v.bias);
        ret->has_bias = q.has_bias;
        return ret;
    }

    void scale_output(float scale) {
        // throw std::logic_error("Not supported");
    }
    void set_output_type(core::DataType dtype) {
        // throw std::logic_error("Not supported");
    }

    core::Tensor& get_weight() { return qweight; }
    core::Tensor* get_weight_scale() { return &scales; }
    core::Tensor get_dequant_weight(const core::Context& ctx) {
        // TODO: Verify
        return dequant_weight_by_mul_diag(ctx, qweight.size(0) * 16);
    }

    core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& input,
        const std::string& output_name,
        bool quant_back,
        Tensor* output) override {
        size_t M = input.size(0);
        size_t N = qweight.size(-1) / 2;
        size_t K = qweight.size(0) * 16;
//        size_t N = qweight.size(-1);
//        size_t K = qweight.size(0) * 8;

        BM_ASSERT(loaded, "")
        BM_ASSERT(input.ndim() == 2, "Input must be 2D");
        BM_ASSERT_EQ(input.size(-1), K, "Input size mismatch");
        BM_ASSERT_EQ(input.dtype(), DataType::kHalf, "Input data type mismatch");
        BM_ASSERT_EQ(input.device(), qweight.device(), "Input and weight must be on the same device");

        Tensor perm;
//        Tensor workspace = ctx.tensor({1024 * 1024}, DataType::kFloat);
//        functions::zeros_(ctx, workspace);
        bool has_zp = false; // !awq
        Tensor qweight1 = qweight; // .view({qweight.size(0) / 2, qweight.size(1) *2});
//        auto stream = ctx.current_stream()->ptr;
//        setL2AccessPolicyWindow(stream, input.data(), input.nbytes());
        Tensor ret = gptq_marlin_gemm(ctx, input, qweight1, scales, qzeros, g_idx, perm, workspace, M, N, K, is_k_full, has_zp, true);
//        setL2AccessPolicyWindow(stream, input.data(), 0);
//        BM_CUDART_ASSERT(cudaStreamSynchronize(stream));
        if (has_bias) {
            ret = add_bias(ctx, ret, bias);
        }
        return activate(ctx, ret);
    }

    Tensor marlin_permute_scales(const core::Context& ctx, const Tensor& s) {
        vector<int> perm;
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                perm.push_back(i + 8 * j);
            }
        }
        BM_ASSERT_EQ(s.size(1) % perm.size(), 0, "dim_out can not divide 64");
        Tensor perm_d = ctx.tensor_of(perm);

        Tensor s1 = s.view({s.numel() / perm.size(), perm.size()});
        Tensor s2 = functions::index_select(ctx, s1, 1, perm_d);
        return s2.view(s.shape());
    }

    void post_load(const core::Context& ctx) {
        BM_ASSERT(!act_order, "");
        Tensor perm; // TODO: act_order
        size_t K = qweight.size(0) * 8;
        size_t N = qweight.size(1);
        // (K / 16, N * 2)
        // auto shape = qweight.shape();
        qweight = gptq_marlin_repack(ctx, qweight, perm, K, N, 4);
        // qweight = qweight.view(shape);
        scales = marlin_permute_scales(ctx, scales);
        loaded = true;
    }

    void load_state_dict(
        const core::Context& ctx,
        const std::map<std::string, const core::Tensor>& state_dict,
        const std::string& prefix,
        bool allow_missing) override {
        BM_ASSERT(group_size == 128, "");
        this->prefix = prefix;
        ctx.load_parameter(&qweight, prefix + ".qweight", state_dict, parallel, dist_layout);
        ctx.load_parameter(&scales, prefix + ".scales", state_dict, parallel, dist_layout);
        ctx.load_parameter(&qzeros, prefix + ".qzeros", state_dict, parallel, dist_layout);
        auto bias_layout = dist_layout == DistLayout::COLUMNAR ? DistLayout::COLUMNAR : DistLayout::REPLICATED;
        if (has_bias) {
            bias = ctx.parameter({ dim_out }, dtype);
            ctx.load_parameter(&bias, prefix + ".bias", state_dict, parallel, bias_layout);
        }
        if (fuse_qkv && is_attn_proj())
            return;
        if (fuse_ff_in && is_ff_in())
            return;
        post_load(ctx);
    }
};

class Linear::impl::AWQ : public Linear::impl {
    const uint32_t pack_factor { 32 / 4 };  // 32 bits(int) / 4 bits
    const int group_size;
public:
    core::Tensor qweight;
    core::Tensor scales;
    core::Tensor qzeros;
    functions::Gemm gemm;

    bool scale_weights;
    DistLayout dist_layout;
    int dev;

    AWQ(
        const core::Context& ctx,
        uint32_t dim_in,
        uint32_t dim_out,
        std::string act_fn_type,
        int quant,
        core::DataType dtype,
        bool parallel,
        DistLayout dist_layout,
        int group_size)
        : Linear::impl(dim_in, dim_out, act_fn_type, false, quant, dtype, parallel),
          group_size(group_size),
          qweight(ctx.parameter({ dim_in, dim_out / pack_factor }, DataType::kInt32)),
          qzeros(ctx.parameter({ dim_in / group_size, dim_out / pack_factor }, DataType::kInt32)),
          scales(ctx.parameter({ dim_in / group_size, dim_out }, dtype)),
          gemm(ctx, dtype, false, false),
          dist_layout(dist_layout),
          dev(ctx.active_device_idx()) {
        BM_ASSERT_EQ(quant, 6, "Wrong quant type");
        BM_ASSERT(dim_in % group_size == 0, "dim_in % group_size != 0");
        gemm.set_compute_type(CUBLAS_COMPUTE_32F);
    }
    ~AWQ() = default;

    void clear_weights() {
        qweight = scales = qzeros = Tensor();
    }

    Fuser weight_fuser() const override {
        return {
            reinterpret_cast<Fuser::FN_FUSE2>(AWQ::fuse2),
            reinterpret_cast<Fuser::FN_FUSE3>(AWQ::fuse3),
            nullptr
        };
    }

    static AWQ* fuse2(const core::Context& ctx, AWQ& a, AWQ& b) {
        auto dim_out = a.dim_out + b.dim_out;
        AWQ* ret = new AWQ(
            ctx, a.dim_in, dim_out, "", a.quant, a.dtype, a.parallel, a.dist_layout, a.group_size);
        ret->qweight = concat2_dim1(ctx, a.qweight, b.qweight);
        ret->qzeros = concat2_dim1(ctx, a.qzeros, b.qzeros);
        ret->scales = concat2_dim1(ctx, a.scales, b.scales);
        a.clear_weights();
        b.clear_weights();
        return ret;
    }

    static AWQ* fuse3(const core::Context& ctx, AWQ& q, AWQ& k, AWQ& v) {
        auto dim_out = q.dim_out + k.dim_out + v.dim_out;
        AWQ* ret = new AWQ(
            ctx, q.dim_in, dim_out, "", q.quant, q.dtype, q.parallel, q.dist_layout, q.group_size);
        ret->qweight = concat3_dim1(ctx, q.qweight, k.qweight, v.qweight);
        ret->qzeros = concat3_dim1(ctx, q.qzeros, k.qzeros, v.qzeros);
        ret->scales = concat3_dim1(ctx, q.scales, k.scales, v.scales);
        q.clear_weights();
        k.clear_weights();
        v.clear_weights();
        return ret;
    }

    void scale_output(float scale) {
        // throw std::logic_error("Not supported");
    }
    void set_output_type(core::DataType dtype) {
        // throw std::logic_error("Not supported");
    }

    core::Tensor& get_weight() { return qweight; }
    core::Tensor* get_weight_scale() { return &scales; }

    /*
    * Input: (seq_len, dim_in)
    * Output: (seq_len, dim_out)
    */
    core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& input,
        const std::string& output_name,
        bool quant_back,
        Tensor* output) override {
        uint32_t dim_out = qweight.size(-1) * pack_factor;
        uint32_t dim_in = qweight.size(0);

        BM_ASSERT(input.ndim() == 2 || input.ndim() == 3, "Input must be 2D");
        BM_ASSERT_EQ(input.size(-1), dim_in, "Input size mismatch");
        BM_ASSERT_EQ(input.dtype(), DataType::kHalf,
                     "Input data type mismatch");
        BM_ASSERT_EQ(input.device(), qweight.device(), "Input and weight must be on the same device");

        auto ret_shape = input.shape();
        ret_shape[ret_shape.size() - 1] = dim_out;
        core::Tensor ret; // = ctx.tensor(ret_shape, input.dtype(), output_name);
        core::Tensor a = input.view({input.numel() / size_t(dim_in), dim_in});
        if (a.size(0) >= 256) {
            auto weight = nn::awq::awq_dequantize(ctx, qweight, scales, qzeros, 0, 0, 0);
            ret = gemm.forward(ctx, a, weight);
        } else {
            ret = nn::awq::awq_gemm(ctx, a, qweight, scales, qzeros, 32);
        }
        return activate(ctx, ret.view(ret_shape));
    }

    void load_state_dict(
        const core::Context& ctx,
        const std::map<std::string, const core::Tensor>& state_dict,
        const std::string& prefix,
        bool allow_missing) override {
        this->prefix = prefix;
//        auto fn = core::Layer::load_param_from_state_dict;
//        fn(ctx, state_dict, prefix + ".qweight", &qweight, allow_missing);
//        fn(ctx, state_dict, prefix + ".scales", &scales, allow_missing);
//        fn(ctx, state_dict, prefix + ".qzeros", &qzeros, allow_missing);

        ctx.load_parameter(&qweight, prefix + ".qweight", state_dict, parallel, dist_layout);
        ctx.load_parameter(&scales, prefix + ".scales", state_dict, parallel, dist_layout);
        ctx.load_parameter(&qzeros, prefix + ".qzeros", state_dict, parallel, dist_layout);

//        auto h_w = state_dict.at(prefix + ".qweight");
//        auto h_s = state_dict.at(prefix + ".scales");
//        auto h_z = state_dict.at(prefix + ".qzeros");
//        uint32_t q = *h_w.data<uint32_t>();
//        uint32_t z = *h_z.data<uint32_t>();
//        half* sp = h_s.data<half>();
//
//        auto weight = nn::awq::awq_dequantize(ctx, qweight, scales, qzeros, 0, 0, 0);
//
//        auto w1 = weight.slice_dim0_len(128, 10);
//        if (prefix.find("7.w_out") != std::string::npos) {
//            std::cout << std::setprecision(5);
//            // std::vector<int> sfl = {0, 2, 4, 6, 1, 3, 5, 7};
//            std::vector<uint32_t> sfl = {0, 4, 1, 5, 2, 6, 3, 7};
//            for (auto& s: sfl) { s *= 4; }
//            // for (int s = 0; s < 32; s += 4) {
//            for (int s = 0; s < 8; s += 1) {
//                float w = float(int(q >> sfl[s] & 0xF) - int(z >> sfl[s] & 0xF)) * __half2float(sp[s]);
//                // std::cout << "q=" << int(q >> s & 0xF) << ", z=" << int(z >> s & 0xF) << ", s=" << sp[s / 4] << ", w=" << w << ";     ";
//                std::cout << w << ",  ";
//            }
//            std::cout << "\n";
//
//            std::cout << prefix << " awq: " << weight << endl;
//        }
    }
};

class Linear::impl::Fp8Linear : public Linear::impl {
public:
    core::Tensor weight; // (dim_out, dim_in)
    core::Tensor weight_scale;
    core::Tensor bias;
    std::string act_fn_type;
    DistLayout dist_layout;

    Fp8Linear(
        const core::Context& ctx,
        uint32_t dim_in,
        uint32_t dim_out,
        std::string act_fn_type,
        int quant,
        core::DataType dtype,
        bool parallel,
        DistLayout dist_layout)
        : Linear::impl(dim_in, dim_out, act_fn_type, false, quant, dtype, parallel),
          weight(ctx.parameter({ dim_out, dim_in }, core::DataType::kInt8)),
          weight_scale(ctx.parameter({ 1 }, core::DataType::kFloat)),
          dist_layout(transpose_layout(dist_layout)) {
    }
    virtual ~Fp8Linear() = default;

    void set_has_bias(bool b) override {
        has_bias = b;
    }

    void scale_output(float scale) {
        // throw std::logic_error("Not supported");
    }
    void set_output_type(core::DataType dtype) {
        // throw std::logic_error("Not supported");
    }

    core::Tensor& get_weight() { return weight; }
    core::Tensor* get_weight_scale() { return &weight_scale; }

    void load_state_dict(
        const core::Context& ctx,
        const std::map<std::string, const core::Tensor>& state_dict,
        const std::string& prefix,
        bool allow_missing) override {
        this->prefix = prefix;
        ctx.load_parameter(&weight, prefix + ".weight", state_dict, parallel, dist_layout);
        ctx.load_parameter(&weight_scale, prefix + ".weight_scale", state_dict, false, dist_layout);
        auto bias_layout = dist_layout == DistLayout::ROW ? DistLayout::COLUMNAR : DistLayout::REPLICATED;
        if (has_bias) {
            bias = ctx.parameter({ dim_out }, dtype);
            ctx.load_parameter(&bias, prefix + ".bias", state_dict, parallel, bias_layout);
        }
    }

    core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& input,
        const std::string& output_name,
        bool quant_back,
        Tensor* output
    ) override {
        int ev_level = ctx.rank() == 0 && ctx.current_layer() == 300 ? 0 : 2;
        string ev_name = logger::str_cat("[M=", input.size(0), "] FP8::forward ", prefix);
        core::EventScope ev_scope(ctx, ev_name, ev_level);
        ctx.recordEvent("dynamic_scaled_quant", ev_level);
        auto a_quant = nn::fp8::dynamic_scaled_quant(ctx, input);
        ctx.recordEvent("gemm_fp8", ev_level);
        functions::Gemm gemm(ctx, core::DataType::kFP8_E4M3, false, true, 1.f);
        gemm.set_output_type(dtype);
        gemm.set_A_scale(*a_quant.quant_scale);
        gemm.set_B_scale(weight_scale);
        Tensor ret = gemm.forward(ctx, a_quant, weight, output, has_bias ? &bias : nullptr);
        static bool printed = true;
        if (!printed && ctx.rank() == 0) {
            printed = true;
            std::cout << "prefix: " << prefix << endl;
            std::cout << "input: " << input << endl;
            std::cout << "a_quant: " << a_quant << endl;
            std::cout << "a_scale: " << *a_quant.quant_scale << endl;
            std::cout << "w_scale: " << weight_scale << endl;
        }

        return activate(ctx, ret);
    }
};

class Linear::impl::Fp8Block : public Linear::impl {
public:
    core::Tensor weight; // (dim_out, dim_in)
    core::Tensor weight_scale;
    core::Tensor bias;
    DistLayout dist_layout;

    Fp8Block(
        const core::Context& ctx,
        uint32_t dim_in,
        uint32_t dim_out,
        std::string act_fn_type,
        int quant,
        core::DataType dtype,
        bool parallel,
        DistLayout dist_layout)
        : Linear::impl(dim_in, dim_out, act_fn_type, false, quant, dtype, parallel),
          weight(ctx.parameter({ dim_out, dim_in }, core::DataType::kInt8)),
          weight_scale(ctx.parameter({ round_up(dim_out, 128) / 128, round_up(dim_in, 128) / 128 }, core::DataType::kFloat)),
          dist_layout(transpose_layout(dist_layout)) {
    }
    virtual ~Fp8Block() = default;

    void set_has_bias(bool b) override {
        has_bias = b;
        if (b) throw std::runtime_error("Fp8Block doesn't support bias");
    }

    void scale_output(float scale) {
        // throw std::logic_error("Not supported");
    }
    void set_output_type(core::DataType dtype) {
        // throw std::logic_error("Not supported");
    }

    core::Tensor& get_weight() override { return weight; }
    core::Tensor* get_weight_scale() override { return &weight_scale; }
    core::Tensor get_dequant_weight(const core::Context& ctx) override {
        auto ret = fp8::dequant_fp8_block_weight(ctx, weight, weight_scale, dtype);
//        if (ctx.is_layer(0)) {
//            std::cout << "weight_scale: " << weight_scale << endl;
//        }
//        functions::check_numeric(ctx, ret);
        return ret;
    };

    Fuser weight_fuser() const override {
        return {
            reinterpret_cast<Fuser::FN_FUSE2>(Fp8Block::fuse),
            nullptr,
            Fp8Block::fuse_dim0
        };
    }

    static Fp8Block* fuse(const core::Context& ctx, Fp8Block& a, Fp8Block& b) {
        uint32_t dim_out = a.dim_out + b.dim_out;
        auto* ret = new Fp8Block(
            ctx, a.dim_in, dim_out, a.act_fn_type, a.quant, a.dtype, a.parallel, a.dist_layout);
        ret->weight = concat2_dim0(ctx, a.weight, b.weight);
        ret->weight_scale = concat2_dim0(ctx, a.weight_scale, b.weight_scale);
        return ret;
    }

    static impl* fuse_dim0(const core::Context& ctx, const std::vector<impl*>& layers) {
        BM_ASSERT(!layers.empty(), "");
        Fp8Block& a = *dynamic_cast<Fp8Block*>(layers[0]);
        BM_ASSERT(!a.has_bias, "");
        std::vector<Tensor*> all_scales;
        std::vector<Tensor*> all_weights;
        for (auto l: layers) {
            Fp8Block* m = dynamic_cast<Fp8Block*>(l);
            BM_ASSERT(m, "All sub layers impl type should be Fp8Block");
            all_weights.push_back(&m->weight);
            all_scales.push_back(&m->weight_scale);
        }

        auto* ret = new Fp8Block(
            ctx, a.dim_in, a.dim_out, a.act_fn_type, a.quant, a.dtype, a.parallel, a.dist_layout);
        ret->weight = concat_dim0(ctx, all_weights);
        ret->weight_scale = concat_dim0(ctx, all_scales);
        return ret;
    }

    vector<impl*> split(const core::Context& ctx, size_t n_split, bool by_out) override {
        BM_ASSERT(!has_bias, "");
        std::vector<Fp8Block*> results(n_split);

        if (by_out) {
            BM_ASSERT_EQ(weight_scale.size(0) % n_split, 0, "Not divisible");
            auto v_weights = weight.view(split_shape_dim0(weight.shape(), n_split)).chunk();
            auto v_scales = weight_scale.view(split_shape_dim0(weight_scale.shape(), n_split)).chunk();

            for (size_t i = 0; i < n_split; ++i) {
                results[i] = new Fp8Block(
                    ctx, dim_in, dim_out / n_split, act_fn_type, quant, dtype, parallel, dist_layout);
                results[i]->weight = ctx.copy(v_weights[i]);
                results[i]->weight.set_name(logger::str_cat(weight.name(), "_part", i));
                results[i]->weight_scale = ctx.copy(v_scales[i]);
            }
        } else {
            BM_ASSERT_EQ(weight.size(1) % (128 * n_split), 0, "Not divisible");
            auto v_weights = split_dim1(ctx, weight, n_split);
            auto v_scales = split_dim1(ctx, weight_scale, n_split);

            for (size_t i = 0; i < n_split; ++i) {
                results[i] = new Fp8Block(
                    ctx, dim_in / n_split, dim_out, act_fn_type, quant, dtype, parallel, dist_layout);
                results[i]->weight = v_weights[i];
                results[i]->weight.set_name(logger::str_cat(weight.name(), "_part", i));
                results[i]->weight_scale = v_scales[i];
            }
        }
        return *reinterpret_cast<vector<impl*>*>(&results);
    }

    void load_state_dict(
        const core::Context& ctx,
        const std::map<std::string, const core::Tensor>& state_dict,
        const std::string& prefix,
        bool allow_missing) override {
        this->prefix = prefix;
        ctx.load_parameter(&weight, prefix + ".weight", state_dict, parallel, dist_layout);
        ctx.load_parameter(&weight_scale, prefix + ".weight_scale_inv", state_dict, parallel, dist_layout);
    }

    Tensor quant_input(const core::Context& ctx,
                       const core::Tensor& input) {
        Tensor a_quant;
        if (!input.quant_scale) {
            ctx.recordEvent("dynamic_scaled_quant", 3);
            a_quant = nn::fp8::per_token_cast_to_fp8(ctx, input);
            const_cast<Tensor&>(input).set_quant_scale(a_quant); // cache quantized tensor
        } else {
            if (input.dtype() == core::DataType::kFP8_E4M3) {
                // input SELF is quantized tensor
                BM_ASSERT(input.quant_scale, "No quant_scale");
                BM_ASSERT_EQ(input.quant_scale->dtype(), core::DataType::kFloat, "Wrong scale dtype");
                a_quant = input;
            } else {
                // input.quant_scale is quantized tensor (cached)
                BM_ASSERT_EQ(input.quant_scale->dtype(), core::DataType::kFP8_E4M3, "Wrong fp8 dtype");
                a_quant = *input.quant_scale;
            }
        }
        return a_quant;
    }

    core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& input,
        const std::string& output_name,
        bool quant_back,
        Tensor* output
    ) override {
        int ev_level = ctx.is_layer(300) ? 0 : 2;
        size_t m = input.size(0);
        size_t k = input.size(1);
        size_t n = weight.size(0);
        string ev_name = logger::str_cat("[M=", m, "] FP8Block::forward ", prefix);
        core::EventScope ev_scope(ctx, ev_name, ev_level);

        Tensor q = quant_input(ctx, input);

        ctx.recordEvent("gemm_fp8_block", ev_level);
        if (output) {
            BM_ASSERT_EQ(output->size(0), m, "");
            BM_ASSERT_EQ(output->size(1), n, "");
            BM_ASSERT_EQ(output->dtype(), dtype, "");
        }
        Tensor ret = output ? *output : ctx.tensor({m, n}, dtype);
        int ret_code = deep_gemm_fp8_block_h20_group(
            q.data(),
            q.quant_scale->data(),
            weight.data(),
            weight_scale.data(),
            ret.data(),
            nullptr,
            ctx.current_stream()->ptr,
            m, n, k,
            64,
            1
        );
        if (ret_code == -1) {
            throw std::runtime_error(logger::str_cat("Unsupported shape N=", weight.size(0), ", K=", weight.size(1)));
        }

        return activate(ctx, ret);
    }

    core::Tensor grouped_gemm(
        const core::Context& ctx,
        const core::Tensor& input,
        const core::Tensor& m_indices,
        int num_groups) {
        BM_ASSERT_EQ(weight.ndim(), 3, "weight should be a fused 3D tensor");
        int ev_level = ctx.is_layer(300) ? 0 : 2;
        size_t m = input.size(0);
        size_t k = input.size(1);
        size_t n = weight.size(1);
//        if (ctx.is_layer(3)) std::cout << "weight: " << weight.shape() << endl;
        string ev_name = logger::str_cat("[M=", m, "] FP8Block::grouped_gemm ", prefix);
        core::EventScope ev_scope(ctx, ev_name, ev_level);

        Tensor q = quant_input(ctx, input);

        ctx.recordEvent("gemm_fp8_block", ev_level);
        Tensor ret = ctx.tensor({m, n}, dtype);
        int ret_code = deep_gemm_fp8_block_h20_group(
            q.data(),
            q.quant_scale->data(),
            weight.data(),
            weight_scale.data(),
            ret.data(),
            m_indices.data<int>(),
            ctx.current_stream()->ptr,
            m, n, k,
            64,
            num_groups
        );
        if (ret_code == -1) {
            throw std::runtime_error(logger::str_cat("Unsupported shape N=", weight.size(0), ", K=", weight.size(1)));
        }

        return activate(ctx, ret);
    }
};

Linear::Linear(
    const core::Context& ctx,
    int dim_in,
    int dim_out,
    std::string act_fn_type,
    model::QuantConfig quant_config,
    bool scale_weights,
    bool weight_transposed,
    bool parallel,
    core::DistLayout dist_layout,
    core::DataType dtype)
    : Layer() {
    int quant = static_cast<int>(quant_config.quant_type);
    if (quant_config.quant_type == model::QuantType::FP8) {
        pimpl.reset(new impl::Fp8Linear(
            ctx, dim_in, dim_out, act_fn_type, quant, dtype, parallel, dist_layout));
    } else if (quant_config.quant_type == model::QuantType::FP8_Block) {
        pimpl.reset(new impl::Fp8Block(
            ctx, dim_in, dim_out, act_fn_type, quant, dtype, parallel, dist_layout));
    } else if (quant_config.quant_type == model::QuantType::AWQ) {
        auto tmp = new impl::AWQ(
            ctx, dim_in, dim_out, act_fn_type, quant, dtype, parallel, dist_layout, quant_config.group_size);
        pimpl = std::unique_ptr<impl>((impl *) tmp);
    } else if (quant_config.quant_type == model::QuantType::GPTQ) {
        bool act_order = quant_config.act_order;
        BM_ASSERT_EQ(scale_weights, false, "Unsupported");
        BM_ASSERT_EQ(weight_transposed, false, "Unsupported");
        int group_size = quant_config.group_size;
        bool sym = quant_config.sym;
        auto tmp = new impl::Int4GPTQ(
            ctx, dim_in, dim_out, act_fn_type, quant, dtype, parallel, dist_layout, act_order, group_size, sym);
        pimpl = std::unique_ptr<impl>((impl *) tmp);
    } else if (quant_config.quant_type == model::QuantType::GPTQ_Marlin) {
        bool act_order = quant_config.act_order;
        int group_size = quant_config.group_size;
        bool sym = quant_config.sym;
        auto tmp = new impl::GPTQMarlin(
            ctx, dim_in, dim_out, act_fn_type, quant, dtype, parallel, dist_layout, act_order, group_size, sym);
        pimpl = std::unique_ptr<impl>((impl *) tmp);
    } else if (quant) {
        auto tmp = new impl::Int8Linear(
            ctx, dim_in, dim_out, act_fn_type, quant, scale_weights, weight_transposed, dtype, parallel, dist_layout);
        add_parameter("weight_quant", tmp->weight);
        add_parameter("weight_scale", tmp->weight_scale);
        pimpl = std::unique_ptr<impl>((impl*) tmp);
    } else {
        auto tmp = new impl::NormalLinear(
            ctx, dim_in, dim_out, act_fn_type, scale_weights, weight_transposed, dtype, parallel, dist_layout);
        add_parameter("weight", *tmp->weight);
        // gemm has no weight; add only for set prefix
        add_submodule("gemm_A_B", tmp->gemm_A_B);
        add_submodule("gemm_A_Btrans", tmp->gemm_A_Btrans);
        pimpl = std::unique_ptr<impl>((impl*) tmp);
    }
    pimpl->dist_layout = dist_layout;
}

Linear::Linear(
    const core::Context& ctx,
    int dim_in,
    int dim_out,
    model::QuantConfig quant_config,
    core::DistLayout dist_layout,
    core::DataType dtype)
    : Linear(ctx, dim_in, dim_out, "", quant_config, false, false, ctx.world_size() > 1, dist_layout, dtype) {}

Linear::Linear(
    const core::Context& ctx,
    const std::string& name,
    const core::Tensor& w)
    : Linear(ctx, w.size(1), w.size(0), "", 0, false, false, false, DistLayout::REPLICATED, w.dtype()) {
    BM_ASSERT_EQ(w.ndim(), 2, "");
    this->name = name;
    auto ptr = dynamic_cast<impl::NormalLinear*>(pimpl.get());
    BM_ASSERT(ptr, "Not NormalLinear");
    *ptr->weight = w;
}

void Linear::move(Linear& other) {
    pimpl = std::move(other.pimpl);
}

Linear::~Linear() = default;

void Linear::scale_output(float scale) {
    pimpl->scale_output(scale);
}
void Linear::set_output_type(core::DataType dtype) {
    pimpl->set_output_type(dtype);
}

core::Tensor Linear::forward(
    const core::Context& ctx, const core::Tensor& input, bool quant_back, Tensor* output) {
    size_t K = input.size(-1);
    size_t M = input.numel() / K;
    size_t N = (pimpl->parallel && DistLayout::COLUMNAR == pimpl->dist_layout)
        ? pimpl->dim_out / ctx.world_size()
        : pimpl->dim_out;
    auto name1 = "Linear(" + name + ")[M=";
    auto ev_name = logger::str_cat(name1, M, ",N=", N, ",K=", K, "]");
    size_t flops = 2UL * K * M * N;
    core::EventScope event_scope(ctx, ev_name, 2, flops);

    Tensor ret;
    if (input.ndim() == 2) {
        ret = pimpl->forward(ctx, input, output_name, quant_back, output);
    } else {
        core::Tensor input2d = input.view({input.numel() / input.size(-1), input.size(-1)});
        Tensor ret2d = pimpl->forward(ctx, input2d, output_name, quant_back, output);
        auto out_shape = input.shape();
        out_shape[out_shape.size() - 1] = ret2d.size(-1);
        ret = ret2d.view(out_shape);
    }

    // clear layer_cache to reduce memory usage if not dual_stream
    ModelContext* m_ctx = const_cast<ModelContext*>(dynamic_cast<const ModelContext*>(&ctx));
    if (m_ctx && !m_ctx->dual_stream())
        m_ctx->layer_cache().clear();
    return ret;
}

const core::Tensor& Linear::get_weight() const {
    return pimpl->get_weight();
}

core::Tensor Linear::get_dequant_weight(const core::Context& ctx) const {
    return pimpl->get_dequant_weight(ctx);
}

const core::Tensor* Linear::get_weight_scale() const {
    return pimpl->get_weight_scale();
}

void Linear::load_state_dict(
    const core::Context& ctx,
    const std::map<std::string, const core::Tensor>& state_dict,
    const std::string& prefix,
    bool allow_missing) {
    this->prefix = prefix;
    pimpl->load_state_dict(ctx, state_dict, prefix, allow_missing);

    bool dequant_desc_act = utils::get_int_env("DEQUANT_DESC_ACT", 0) > 0;
    impl::Int4GPTQ* q = dynamic_cast<impl::Int4GPTQ*>(pimpl.get());
    if (dequant_desc_act && q && q->parallel && q->act_order && !q->dim_out_parallel) {
        Tensor w = q->get_dequant_weight(ctx);
        auto new_p = new impl::NormalLinear(
            ctx, q->dim_in, q->dim_out, q->act_fn_type, false, false, q->dtype, q->parallel, q->dist_layout);
        new_p->weight = std::make_unique<Tensor>();
        *new_p->weight = w;
        pimpl.reset(new_p);
    }
}

Linear* Linear::fuse(const core::Context& ctx, Linear& q, Linear& k) {
    Linear::impl::Fuser fuser = q.pimpl->weight_fuser();
    if (fuser.fuse2) {
        auto a = fuser.fuse2(ctx, *q.pimpl.get(), *k.pimpl.get());
        if (a) {
            unique_ptr<Linear> ret(new Linear());
            ret->pimpl = std::unique_ptr<impl>(a);
            if (q.name == "w_in")
                ret->name = "FUSE_ff_in";
            return ret.release();
        }
    }
    return nullptr;
}

Linear* Linear::fuse(const core::Context& ctx, Linear& q, Linear& k, Linear& v) {
    Linear::impl::Fuser fuser = q.pimpl->weight_fuser();
    if (fuser.fuse3) {
        auto a = fuser.fuse3(ctx, *q.pimpl.get(), *k.pimpl.get(), *v.pimpl.get());
        if (a) {
            unique_ptr<Linear> ret(new Linear());
            ret->pimpl = std::unique_ptr<impl>(a);
            if (q.name == "project_q")
                ret->name = "FUSE_project_qkv";
            return ret.release();
        }
    }
    return nullptr;
}

Linear* Linear::fuse(const core::Context& ctx, const std::vector<Linear*>& layers) {
    BM_ASSERT(!layers.empty(), "no layers");
    Linear::impl::Fuser fuser = layers[0]->pimpl->weight_fuser();
    if (fuser.fuse_dim0) {
        std::vector<impl*> vec;
        for (auto layer: layers) {
            vec.push_back(layer->pimpl.get());
        }
        unique_ptr<Linear> ret(new Linear());
        auto a = fuser.fuse_dim0(ctx, vec);
        ret->pimpl = std::unique_ptr<impl>(a);
        return ret.release();
    }
    return nullptr;
}

std::vector<Linear*> Linear::split(const core::Context& ctx, size_t n_split, bool dim_out) {
    std::vector<Linear*> results;
    auto impls = pimpl->split(ctx, n_split, dim_out);
    for (auto a : impls) {
        Linear* linear = new Linear();
        linear->pimpl = std::unique_ptr<impl>(a);
        linear->name = name;
        results.push_back(linear);
    }
    return results;
}

bool Linear::support_fuse_gptq_gate_in(const Tensor& input) {
    static int fuse_w_in = utils::get_int_env("CPM_FUSE_FF_IN", 0);
    auto ptr = dynamic_cast<impl::Int4GPTQ*>(pimpl.get());
    return ptr
            && fuse_w_in == 2
            && ptr->use_exllama
            && ptr->new_kernel
            && !ptr->act_order
            && !ptr->trt_kernel
            && ptr->qweight.numel() > 0;
}

std::tuple<core::Tensor, core::Tensor, core::Tensor, bool> Linear::get_gptq_weights() {
    auto ptr = dynamic_cast<impl::Int4GPTQ*>(pimpl.get());
    if (!ptr) {
        return {Tensor(), Tensor(), Tensor(), false};
    }
    return {ptr->qweight, ptr->qzeros, ptr->scales, ptr->sym};
}

void Linear::set_has_bias(bool b) {
    pimpl->set_has_bias(b);
}

void Linear::dequant_cache_weight(core::Context& ctx, const core::Tensor& fake_input) {
    auto gptq_ptr = dynamic_cast<impl::Int4GPTQ*>(pimpl.get());
    model::ModelContext* m_ctx = dynamic_cast<model::ModelContext*>(&ctx);
    if (gptq_ptr && m_ctx) {
        gptq_ptr->dequant_cache_weight(*m_ctx, fake_input);
    }
}

core::Tensor Linear::grouped_gemm_fp8_block(
    const core::Context& ctx,
    const core::Tensor& input,
    const core::Tensor& m_indices,
    int num_groups) {
    auto ptr = dynamic_cast<impl::Fp8Block*>(pimpl.get());
    BM_ASSERT(ptr, "Not Fp8Block impl");
    return ptr->grouped_gemm(ctx, input, m_indices, num_groups);
}
}
