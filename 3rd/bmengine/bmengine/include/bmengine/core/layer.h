#pragma once
#include "bmengine/core/export.h"
#include <map>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand.h>
#include "bmengine/core/tensor.h"

namespace bmengine {

namespace core {

class Context;

class BMENGINE_EXPORT Layer {
public:
    std::map<std::string, Layer*> modules;
    std::map<std::string, Tensor*> parameters;
    // to record the order of children module or params
    std::vector<std::string> module_names;
    std::vector<std::string> param_names;

    std::string prefix;
    std::string output_name;
    std::string name;
    int dev { 0 };
    int output_dev { 0 };

    Layer() = default;
    virtual ~Layer() = default;
    Layer(const Layer&) = delete;
    Layer(Layer&&) = delete;

    void add_submodule(const std::string& name, Layer& module) { add_submodule(name, &module); }
    void add_submodule(const std::string& name, Layer*);
    void add_parameter(const std::string& name, Tensor&);
    std::map<const std::string, Tensor*> named_parameters(
        const std::string& prefix, bool recursive = true);
    virtual const char* layer_type() const = 0;

    BMENGINE_EXPORT friend std::ostream& operator<<(std::ostream& os, const Layer& layer);
    virtual void init_parameters(
        const Context& ctx, curandGenerator_t& gen, const std::string& prefix = "");

    // Load parameters recursively from external state_dict.
    //    Usually, state_dict is a map of 'references' to the underlying numpy arrays,
    //    which are passed in from python torch.load()
    virtual void load_state_dict(
        const Context& ctx,
        const std::map<std::string, const Tensor>& state_dict,
        const std::string& prefix,
        bool allow_missing = false);
    static void load_param_from_state_dict(
        const Context& ctx,
        const std::map<std::string, const Tensor>& state_dict,
        const std::string& name,
        Tensor* param,
        bool allow_missing = false);
    static void load_param_cast(
        const Context& ctx,
        const std::map<std::string, const Tensor>& state_dict,
        const std::string& name,
        Tensor* param,
        core::DataType cast_src_dtype);
};

} // namespace core

} // namespace bmengine
