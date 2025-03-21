#include "py_export/py_utils.h"
#include "py_export/py_model_base.h"
#include "bmengine/core/engine.h"
#include "bmengine/core/exception.h"
#include "bmengine/logger/kernel_time_trace.hpp"
#include "model/model.h"
#include "utils/env.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

namespace bind {

namespace py = pybind11;
using bmengine::core::DeviceConfiguration;
using bmengine::core::DistConfiguration;
using bmengine::core::Engine;
using model::ModelConfig;
using model::QuantConfig;

model::ModelConfig create_config_from_dict(py::dict& cfg) {
    return bind::pydict_to_model_config(cfg);
}

void define_model_config(py::module_& m) {
    py::class_<model::ModelConfig>(m, "ModelConfig")
        .def(py::init(&create_config_from_dict));
}

QuantConfig create_quant_config(int i_quant_type, bool quant_weight_kv, bool act_order, int group_size, bool sym) {
    QuantConfig config{};
    config.quant_type = static_cast<model::QuantType>(i_quant_type);
    config.quant_weight_kv = quant_weight_kv;
    config.act_order = act_order;
    config.group_size = group_size;
    config.sym = sym;
    return config;
}

void define_quant_config(py::module_& m) {
    py::class_<QuantConfig>(m, "QuantConfig")
        .def(py::init(&create_quant_config));
}

DistConfiguration create_dist_config(int tp, std::string dist_init_addr, int nnodes, int node_rank) {
    DistConfiguration config{};
    config.tp = tp;
    config.dist_init_addr = dist_init_addr;
    config.nnodes = nnodes;
    config.node_rank = node_rank;
    return config;
}

void define_dist_config(py::module_& m) {
    py::class_<DistConfiguration>(m, "DistConfig")
        .def(py::init(&create_dist_config));
}

static std::vector<DeviceConfiguration> get_all_dev_config(size_t memory_limit) {
    std::vector<DeviceConfiguration> devices;
    int gpu_num;
    BM_CUDART_ASSERT(cudaGetDeviceCount(&gpu_num));
    for (int i = 0; i < gpu_num; ++i) {
        devices.emplace_back(i, memory_limit);
    }
    return devices;
}

std::shared_ptr<Engine> create_engine(int device_id, size_t memory_limit, DistConfiguration dist_cfg) {
    if (memory_limit == 0) {
        size_t free, total;
        BM_CUDART_ASSERT(cudaSetDevice(device_id < 0 ? 0 : device_id));
        BM_CUDART_ASSERT(cudaMemGetInfo(&free, &total));
        size_t def_reserve_mem = free > (36UL << 30UL) ? 1700 : 1024;
        size_t reserve_mem = utils::get_int_env("RESERVE_MEM_MB", def_reserve_mem);
        memory_limit = free - (reserve_mem << 20L);
    }

    std::vector<DeviceConfiguration> devices;
    if (device_id == -1) {
        // use all available devices
        devices = get_all_dev_config(memory_limit);
    } else {
        devices.emplace_back(device_id, size_t(memory_limit));
    }
    return std::make_shared<Engine>(devices, dist_cfg);
}

void define_engine(py::module_& m) {
    py::class_<Engine, std::shared_ptr<Engine>>(m, "Engine")
        .def(py::init(&create_engine));
}

} // namespace bind
