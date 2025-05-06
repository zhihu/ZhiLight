#pragma once
#include "model/model_config.hpp"
#include <bmengine/core/core.h>

namespace nn {
using namespace bmengine;

class Linear : public core::Layer {
    Linear() = default;
BM_LAYER_DEF(Linear);

public:
    Linear(
        const core::Context& ctx,
        int dim_in,
        int dim_out,
        std::string act_fn_type,
        model::QuantConfig quant,
        bool scale_weights = false,
        bool weight_transposed = true,
        bool parallel = false,
        core::DistLayout dist_layout = core::DistLayout::COLUMNAR,
        core::DataType dtype = core::DataType::kHalf);

    Linear(
        const core::Context& ctx,
        int dim_in,
        int dim_out,
        model::QuantConfig quant,
        core::DistLayout dist_layout,
        core::DataType dtype = core::DataType::kHalf);

    Linear(
        const core::Context& ctx,
        const std::string& name,
        const core::Tensor& w);

    void move(Linear& other);

    void scale_output(float scale);
    void set_output_type(core::DataType dtype);

    const core::Tensor& get_weight() const;
    core::Tensor get_dequant_weight(const core::Context& ctx) const;
    const core::Tensor* get_weight_scale() const; // for quant

    core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& x,
        bool quant_back = true,
        core::Tensor* output = nullptr);

    void load_state_dict(
        const core::Context& ctx,
        const std::map<std::string, const core::Tensor>& state_dict,
        const std::string& prefix,
        bool allow_missing) override;

    static Linear* fuse(const core::Context& ctx, Linear& a, Linear& b);
    static Linear* fuse(const core::Context& ctx, Linear& q, Linear& k, Linear& v);
    static Linear* fuse(const core::Context& ctx, const std::vector<Linear*>& layers);
    std::vector<Linear*> split(const core::Context& ctx, size_t n_split, bool dim_out);

    bool support_fuse_gptq_gate_in(const core::Tensor& input);
    std::tuple<core::Tensor, core::Tensor, core::Tensor, bool> get_gptq_weights();

    void set_has_bias(bool b = true);

    void dequant_cache_weight(core::Context& ctx, const core::Tensor& fake_input);

    core::Tensor grouped_gemm_fp8_block(
        const core::Context& ctx,
        const core::Tensor& input,
        const core::Tensor& m_indices,
        int num_groups);
};

core::Tensor concat_dim0(const core::Context& ctx, std::vector<core::Tensor*> tensors, bool stack=true);

}
