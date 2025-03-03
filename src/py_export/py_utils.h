#pragma once
#include <stddef.h>
#include <stdint.h>
#include <bmengine/core/core.h>
#include <pybind11/numpy.h>
#include <map>
#include <string>
#include "model/model.h"

namespace bind {

namespace py = pybind11;

void load_state_dict(
    bmengine::core::Context& ctx,
    const std::map<std::string, py::array>& state_dict,
    std::map<const std::string, bmengine::core::Tensor*> named_params,
    bool parallel = false);

bmengine::core::Tensor numpy_to_tensor(const std::string& name, const py::array& arr, bool copy=false);

std::map<std::string, const bmengine::core::Tensor> numpy_to_tensor(
    const std::map<std::string, py::array>& state_dict);

std::map<std::string, const bmengine::core::Tensor> numpy_to_tensor(py::dict state_dict);

bmengine::core::DataType numpy_dtype_to_bmengine(pybind11::dtype dtype);

model::ModelConfig pydict_to_model_config(py::dict& cfg);

}
