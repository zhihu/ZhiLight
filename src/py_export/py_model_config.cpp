#include "py_export/bind.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_fp16.h>
#include <sstream>
#include <iostream>
#include <bmengine/core/tensor.h>
#include "model/model.h"

namespace bind {

using bmengine::core::Context;
using bmengine::core::DataType;
using bmengine::core::Tensor;

template<typename T>
static T get_attr(const py::dict& cfg, const char* name, T def_val = T(0)) {
    if (cfg.template contains(name) && !py::isinstance<py::none>(cfg[name]))
        return cfg[name].cast<T>();
    return def_val;
}

template<typename T>
static T get_attr(const py::dict& cfg, const char* name, const char* alias) {
    if (alias && cfg.template contains(alias) && !py::isinstance<py::none>(cfg[alias]))
        return cfg[alias].cast<T>();
    if (cfg.template contains(name) && !py::isinstance<py::none>(cfg[name]))
        return cfg[name].cast<T>();
    throw std::runtime_error(std::string("No attribute found: ") + name);
}

static DataType get_dtype(const py::dict& cfg) {
    if (cfg.contains("force_half") && cfg["force_half"].cast<bool>())
        return DataType::kHalf;
    if (cfg.contains("torch_dtype") || cfg.contains("dtype")) {
        auto str_dtype = get_attr<std::string>(cfg, "dtype", "torch_dtype");
        auto data_type = bmengine::core::name_to_data_type(str_dtype);
        if (data_type != DataType::kBFloat16 && data_type != DataType::kHalf) {
            throw std::runtime_error(std::string("Unsupported dtype: ") + str_dtype);
        }
        return data_type;
    }
    // old way
    if (cfg.contains("bf16") && cfg["bf16"].cast<bool>() ||
        cfg.contains("bfloat16") && cfg["bfloat16"].cast<bool>() ||
        cfg.contains("bfloat") && cfg["bfloat"].cast<bool>())
        return DataType::kBFloat16;
    throw std::runtime_error(std::string("No attribute \"torch_dtype\" found"));
}

template<typename T>
static void set_attr(const py::dict& cfg, const char* name, T& attr) {
    if (cfg.template contains(name) && !py::isinstance<py::none>(cfg[name]))
        attr = cfg[name].cast<T>();
}

void pydict_to_rope_config(const py::dict& d, model::RopeConfig& config) {
    set_attr(d, "type", config.type);
    set_attr(d, "factor", config.factor);
    set_attr(d, "attn_factor", config.attn_factor);
    set_attr(d, "beta_fast", config.beta_fast);
    set_attr(d, "beta_slow", config.beta_slow);
    set_attr(d, "mscale", config.mscale);
    set_attr(d, "mscale_all_dim", config.mscale_all_dim);
    set_attr(d, "original_max_position_embeddings", config.original_max_position);
}

model::ModelConfig pydict_to_model_config(py::dict& cfg) {
    std::string model_type = get_attr<std::string>(cfg, "model_type", "");
    int num_layers = get_attr<int>(cfg, "num_layers", "num_hidden_layers");
    int dim_model = get_attr<int>(cfg, "dim_model", "hidden_size");
    int num_heads = get_attr<int>(cfg, "num_heads", "num_attention_heads");
    int dim_head = get_attr<int>(cfg, "dim_head", 128);
    int dim_ff = get_attr<int>(cfg, "dim_ff", "intermediate_size");
    int vocab_size = cfg["vocab_size"].cast<int>();
    float eps = get_attr(cfg, "rms_norm_eps", get_attr(cfg, "eps", 1e-5f));
    int num_kv_heads = get_attr(cfg, "num_kv_heads", get_attr(cfg, "num_key_value_heads", num_heads));
    auto data_type = get_dtype(cfg);

    model::ModelConfig config{ model_type, num_layers, dim_model, num_heads, dim_head, dim_ff, vocab_size,
             eps, num_kv_heads, data_type };

    config.activate_fn = get_attr(cfg, "activate_fn", get_attr<std::string>(cfg, "hidden_act", std::string("silu")));
    BM_ASSERT(config.activate_fn == "silu" || config.activate_fn == "gelu", "Unsupported activate_fn");

    // CPM deprecated
    set_attr(cfg, "scale_weights", config.scale_weights);
    set_attr(cfg, "weight_transposed", config.weight_transposed);
    set_attr(cfg, "pos_bias_type", config.pos_bias_type);
    BM_ASSERT(config.pos_bias_type == "rotary" || config.pos_bias_type == "relative",
              "Unsupported pos_bias_type");

    // MiniCPM
    set_attr(cfg, "dim_model_base", config.dim_model_base);
    set_attr(cfg, "scale_depth", config.scale_depth);
    set_attr(cfg, "scale_emb", config.scale_emb);

    set_attr(cfg, "tie_lm_head", config.tie_lm_head);

    // rope
    set_attr(cfg, "rope_theta", config.rope_theta);
    set_attr(cfg, "max_position_embeddings", config.max_position_embeddings);
    if (cfg.contains("rope_scaling") && py::isinstance<py::dict>(cfg["rope_scaling"])) {
        pydict_to_rope_config(cfg["rope_scaling"], config.rope_cfg);
    }

    // moe config
    set_attr(cfg, "moe_num_experts", config.moe_num_experts);
    set_attr(cfg, "num_local_experts", config.moe_num_experts);
    set_attr(cfg, "num_experts", config.moe_num_experts);
    set_attr(cfg, "n_routed_experts", config.moe_num_experts);
    set_attr(cfg, "moe_top_k", config.moe_top_k);
    set_attr(cfg, "num_experts_per_tok", config.moe_top_k);
    set_attr(cfg, "moe_intermediate_size", config.moe_intermediate_size);
    set_attr(cfg, "shared_expert_intermediate_size", config.shared_expert_intermediate_size);
    set_attr(cfg, "norm_topk_prob", config.norm_topk_prob);
    set_attr(cfg, "first_k_dense_replace", config.first_k_dense_replace);
    set_attr(cfg, "routed_scaling_factor", config.routed_scaling_factor);
    int n_shared_experts = 0;
    set_attr(cfg, "n_shared_experts", n_shared_experts);
    if (n_shared_experts > 0 && config.shared_expert_intermediate_size <= 0) {
        config.shared_expert_intermediate_size = n_shared_experts * config.moe_intermediate_size;
    }

    // MOE of DeepSeek
    set_attr(cfg, "n_group", config.moe_n_group);
    set_attr(cfg, "topk_group", config.moe_topk_group);

    // MLA config
    set_attr(cfg, "q_lora_rank", config.q_lora_rank);
    set_attr(cfg, "kv_lora_rank", config.kv_lora_rank);
    set_attr(cfg, "qk_nope_head_dim", config.qk_nope_head_dim);
    set_attr(cfg, "qk_rope_head_dim", config.qk_rope_head_dim);
    set_attr(cfg, "v_head_dim", config.v_head_dim);

    set_attr(cfg, "use_qk_norm", config.use_qk_norm);
    set_attr(cfg, "logit_scale", config.logit_scale);

    return config;
}
}
