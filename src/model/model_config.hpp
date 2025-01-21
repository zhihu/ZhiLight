#pragma once
#include <sstream>
#include <vector>
#include <bmengine/core/core.h>

namespace model {
using namespace bmengine;

struct RopeConfig {
    std::string type;
    float factor { 1.0 };
    float attn_factor { 1.0 };
    int beta_fast { 32 };
    int beta_slow { 1 };
    float mscale { 0 };
    float mscale_all_dim { 0 };
    int original_max_position { 0 };

    bool neox_style { true };
    std::vector<int> section;

    // LLaMA3
    float low_freq_factor { 0 };
    float high_freq_factor { 0 };
};

struct ModelConfig {
    std::string model_type;
    int num_layers;
    int dim_model;
    int num_heads;
    int dim_head;
    int dim_ff;
    int vocab_size;
    float eps;
    int num_kv_heads;
    core::DataType dtype;

    std::vector<std::vector<bool>> mask_modules;
    bool scale_weights { false };
    bool weight_transposed { false };
    int dim_model_base { 0 }; // cpm dragonfly
    float scale_emb { 1. }; // cpm dragonfly
    float scale_depth { 1. }; // cpm dragonfly

    std::string pos_bias_type { "rotary" };
    std::string activate_fn { "silu" };
    bool tie_lm_head { false };  // Whether the model's input and output embeddings should be tied.
    int max_position_embeddings { 8192 }; // max length trained.
    float rope_theta { 10000.0 }; // The base period of the RoPE embeddings
    RopeConfig rope_cfg;
    // MOE
    int moe_num_experts { -1 }; // Number of experts per Sparse MLP layer.
    int moe_top_k { -1 }; // The number of experts to root per-token
    int moe_intermediate_size { -1 };
    int shared_expert_intermediate_size { -1 };
    bool norm_topk_prob { true };
    int first_k_dense_replace { 0 };
    float routed_scaling_factor { 1 }; // Deep seek
    int moe_n_group { 1 }; // Deep seek
    int moe_topk_group { 1 }; // Deep seek
    // MLA
    int q_lora_rank { 0 };
    int kv_lora_rank { 0 };
    int qk_nope_head_dim { 0 };
    int qk_rope_head_dim { 0 };
    int v_head_dim { 0 };
    // Cohere
    bool use_qk_norm { false };
    float logit_scale { 1. };


    ModelConfig(
        std::string model_type,
        int num_layers,
        int dim_model,
        int num_heads,
        int dim_head,
        int dim_ff,
        int vocab_size,
        float eps = 1e-6,
        int num_kv_heads = -1,
        core::DataType dtype = core::DataType::kHalf)
        : model_type(model_type),
          num_layers(num_layers),
          dim_model(dim_model),
          num_heads(num_heads),
          dim_head(dim_head),
          dim_ff(dim_ff),
          vocab_size(vocab_size),
          eps(eps),
          num_kv_heads(num_kv_heads == -1 ? num_heads : num_kv_heads),
          dtype(dtype) {
        this->mask_modules.resize(num_layers);
        for (auto& it : this->mask_modules) {
            it.resize(2);
        }
    }

    std::string to_string() const {
        std::stringstream s;
        // clang-format off
        s << "Config(model_type=" << model_type
            << ", num_layers=" << num_layers
            << ", dim_model=" << dim_model
            << ", num_heads=" << num_heads
            << ", num_kv_heads=" << num_kv_heads
            << ", dim_head=" << dim_head
            << ", dim_ff=" << dim_ff
            << ", vocab_size=" << vocab_size
            << ", eps=" << eps
            << ", scale_weights=" << scale_weights
            << ", weight_transposed=" << weight_transposed
            << ", dim_model_base=" << dim_model_base
            << ", scale_depth=" << scale_depth
            << ", scale_emb=" << scale_emb
            << ", dtype=" << bmengine::core::get_data_type_name(dtype)
            << ", pos_bias_type=" << pos_bias_type
            << ", activate_fn=" << activate_fn
            << ", rope_theta=" << rope_theta
            << ", max_position_embeddings=" << max_position_embeddings
            << ")";
        // clang-format on
        return s.str();
    }
};

enum class QuantType {
    NoQuant = 0,
    AbsMax = 1,  // Load from quantized int8 weights and float16 scales
    AutoInt8 = 2,  // Load from float16 weights, do int8 quantization during loading model.
    Int4 = 3,
    AutoInt4 = 4,  // Only for speed test
    GPTQ = 5,
    AWQ = 6,
    FP8 = 7,
    GPTQ_Marlin = 8,
    AWQ_Marlin = 9,
};
struct QuantConfig {
    QuantType quant_type { QuantType::NoQuant };
    int quant_weight_kv { 1 };  // 0: no, 1: yes.
    bool act_order { false };  // https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda
    int group_size { 128 };
    bool sym { false };
    bool fuse_block_ = true;

    QuantConfig() = default;
    QuantConfig(int quant) : quant_type(static_cast<QuantType>(quant)) {}

    int quant() const {
        return static_cast<int>(quant_type);
    }
    bool is_int8() const {
        return quant_type == QuantType::AbsMax || quant_type == QuantType::AutoInt8;
    }
    bool is_int4() const {
        return quant_type == QuantType::Int4 || quant_type == QuantType::AutoInt4;
    }
    bool fuse_block() const {
        return is_int8() && fuse_block_;
    }
    bool fuse_ff() const {
        return is_int8();
    }
    bool fuse_ln_attn() const {
        return is_int8() && quant_weight_kv > 0;
    }
    bool fuse_ln_ff() const {
        return is_int8();
    }
};

}
