#pragma once
#include <stddef.h>
#include <stdint.h>
#include <map>
#include <string>
#include <bmengine/core/core.h>
#include <ATen/ATen.h>

namespace bind {

using bmengine::core::Context;
using bmengine::core::DataType;
using bmengine::core::Tensor;

namespace py = pybind11;

const Tensor aten_to_tensor(const Context& ctx, const at::Tensor& at_tensor);

at::Tensor tensor_to_aten(const Context& ctx, const Tensor& tensor);

// convert every at::Tensor to core::Tensor
void load_at_state_dict(
    Context& ctx,
    const std::map<std::string, at::Tensor>& state_dict,
    std::map<const std::string, Tensor*> named_params,
    bool parallel = false);
};
