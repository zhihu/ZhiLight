#pragma once
#include "generator/generator.h"
#include <stddef.h>
#include <stdint.h>
#include <bmengine/core/core.h>
#include <pybind11/numpy.h>
#include <memory>

#include "py_export/py_utils.h"

namespace bind {

namespace py = pybind11;

void define_model_config(py::module_& m);
void define_quant_config(py::module_& m);
void define_dist_config(py::module_& m);

typedef std::shared_ptr<bmengine::core::Engine> PyEngine;
void define_engine(py::module_& m);

// models
void define_cpm_base(py::module_& m);
void define_llama(py::module_& m);
void define_llama_encoder_model(py::module_& m);
void define_encoder_output_head(py::module_& m);

// layers
void define_layer_attention(py::module_& m);
void define_layer_embedding(py::module_& m);
void define_layer_feed_forward(py::module_& m);
void define_layer_linear(py::module_& m);
void define_layer_position_embedding(py::module_& m);

void define_dynamic_batch(py::module_& m);

void define_functions(py::module_& m);

std::vector<int> to_int_vector(const py::list& data_list);
std::vector<std::string> to_string_vector(const py::list& data_list);

std::vector<std::vector<int>> to_2d_int_vector(const py::list& data_list);
std::vector<std::vector<bool>> to_2d_bool_vector(const py::list& data_list);

void convert_results(const std::vector<generator::SearchResults>& results, py::list* res);
void convert_multi_results(const std::vector<generator::SearchResults>& results, py::list* res);

uint16_t float2half(float x);
float half2float(uint16_t x);

// load recursively
void load_state_dict_new(
    bmengine::core::Context& ctx,
    const std::map<std::string, py::array>& state_dict,
    const std::string& prefix,
    bmengine::core::Layer* model);

int get_cpu_level();

}
