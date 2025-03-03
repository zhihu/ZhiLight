#include "bind_internal.h"
// #include <pybind11/pybind11.h>
#include <stdexcept>
#include <iostream>
#include <torch/extension.h>
namespace py = pybind11;

PYBIND11_MODULE(internals_, m) {

    // layers
    py::module_ layers_m = m.def_submodule("layers", "internal layers for testing.");
    bind::define_layer_attention(layers_m);
    bind::define_layer_embedding(layers_m);
    bind::define_layer_feed_forward(layers_m);
    bind::define_layer_linear(layers_m);

    bind::define_functions(m);
}
