#include "bind_internal.h"
#include "internal_utils.h"
#include "layer_base.hpp"

#include "nn/nn.h"
#include "model/model.h"
#include "utils/exception.h"
#include "py_export/py_utils.h"
#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <tuple>
#include <iostream>
#include <random>

namespace py = pybind11;
using namespace bmengine;

class PyLinear : public PyLayerBase<nn::Linear> {

public:
    PyLinear(
        int dim_in,
        int dim_out,
        std::string activation,
        int quant,
        std::string& dtype_name)
        : PyLayerBase<nn::Linear>(
            dim_in,
            dim_out,
            activation,
            quant,
            false,
            false,
            false,
            core::DistLayout::COLUMNAR,
            core::name_to_data_type(dtype_name)) { }

    static PyLinear create(
        int dim_in,
        int dim_out,
        std::string activation,
        int quant,
        std::string dtype_name = "half") {
        auto ff =
            PyLinear(dim_in, dim_out, activation, quant, dtype_name);
        return ff;
    }

    at::Tensor forward(at::Tensor& input) {
        auto t_input = bind::aten_to_tensor(*ctx, input);
        auto out_data = layer->forward(*ctx, t_input);
        auto output = bind::tensor_to_aten(*ctx, out_data);
        return output;
    }
};

namespace bind {
void define_layer_linear(py::module_& layers_m) {

    py::class_<PyLinear>(layers_m, "Linear")
        .def(py::init(&PyLinear::create))
        .def("load_state_dict", &PyLinear::load_state_dict)
        .def("named_parameters", &PyLinear::named_parameters)
        .def("forward", &PyLinear::forward);
}
}
