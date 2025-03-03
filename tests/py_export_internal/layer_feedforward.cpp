#include "bind_internal.h"
#include "internal_utils.h"

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
#include <thread>
#include <memory>
#include <csignal>

namespace py = pybind11;

class PyFeedForward {
private:
    std::vector<std::shared_ptr<nn::FeedForward>> mds;
    std::shared_ptr<bmengine::core::Engine> engine;
    int dim_model;
    int dim_ff;
    std::string act_fn_type;
    int quant;
    bool scale_weights;
    bool weight_transposed;

    PyFeedForward(
        int dim_model,
        int dim_ff,
        std::string act_fn_type,
        int quant,
        bool scale_weights = false,
        bool weight_transposed = true)
        : dim_model(dim_model),
          dim_ff(dim_ff),
          act_fn_type(act_fn_type),
          quant(quant),
          scale_weights(scale_weights),
          weight_transposed(weight_transposed) {
        std::vector<bmengine::core::DeviceConfiguration> devices;
        int gpu_num;
        BM_CUDART_ASSERT(cudaGetDeviceCount(&gpu_num));
        for (int i = 0; i < gpu_num; ++i) {
            devices.emplace_back(i, (size_t) 2 << 30);
        }
        engine = std::make_shared<bmengine::core::Engine>(devices);

        std::signal(SIGSEGV, [](int sig) { bmengine::print_demangled_trace(25); });
        std::signal(SIGABRT, [](int sig) { bmengine::print_demangled_trace(25); });

        std::vector<std::thread> threads;
        mds.resize(engine->num_gpus());
        model::ModelConfig model_config { "",
                                          0,
                                          dim_model,
                                          0,
                                          0,
                                          dim_ff,
                                          0};
        model_config.activate_fn = act_fn_type;
        model_config.scale_weights = scale_weights;
        model_config.weight_transposed = weight_transposed;
        for (int i = 0; i < engine->num_gpus(); ++i) {
            threads.emplace_back(
                [this, i, model_config, quant] {
                    auto ctx = engine->create_context({ i });
                    bmengine::core::WithDevice device(ctx, 0);
                    mds[i] = std::move(std::make_shared<nn::FeedForward>(
                        ctx,
                        model_config,
                        quant,
                        true));
                });
        }
        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }
    };

public:
    ~PyFeedForward() {
        // necessary to release md before engine, cycle reference.
        mds.clear();
    }
    PyFeedForward(const PyFeedForward& other) {
        mds = other.mds;
        engine = other.engine;
        dim_model = other.dim_model;
        dim_ff = other.dim_ff;
        act_fn_type = other.act_fn_type;
        quant = other.quant;
        scale_weights = other.scale_weights;
        weight_transposed = other.weight_transposed;
    }
    PyFeedForward(PyFeedForward&& other) {
        mds = std::move(other.mds);
        engine = std::move(other.engine);
        dim_model = other.dim_model;
        dim_ff = other.dim_ff;
        act_fn_type = other.act_fn_type;
        quant = other.quant;
        scale_weights = other.scale_weights;
        weight_transposed = other.weight_transposed;
    }
    PyFeedForward& operator=(const PyFeedForward& other) {
        mds = other.mds;
        engine = other.engine;
        dim_model = other.dim_model;
        dim_ff = other.dim_ff;
        act_fn_type = other.act_fn_type;
        quant = other.quant;
        scale_weights = other.scale_weights;
        weight_transposed = other.weight_transposed;
        return *this;
    }
    PyFeedForward& operator=(PyFeedForward&& other) {
        mds = std::move(other.mds);
        engine = std::move(other.engine);
        dim_model = other.dim_model;
        dim_ff = other.dim_ff;
        act_fn_type = other.act_fn_type;
        quant = other.quant;
        scale_weights = other.scale_weights;
        weight_transposed = other.weight_transposed;
        return *this;
    }
    static PyFeedForward create(
        int dim_model,
        int dim_ff,
        std::string act_fn_type,
        int quant,
        bool scale_weights = false,
        bool weight_transposed = true) {
        auto ff =
            PyFeedForward(dim_model, dim_ff, act_fn_type, quant, scale_weights, weight_transposed);
        return ff;
    };

    py::array forward(py::array& input) __attribute__((visibility("hidden"))) {
        // auto d = ctx.with_device(0);

        std::vector<std::thread> threads;
        auto buf = input.request();
        auto ndim = input.ndim();
        py::array_t<float> ndarray; // out
        ndarray.resize(buf.shape); // must resize in main thread

        // bmengine::core::Tensor out_data;
        for (int i = 0; i < engine->num_gpus(); ++i) {
            threads.emplace_back([this, i, &buf, ndim, &ndarray] {
                auto ctx = engine->create_context({ i });
                bmengine::core::WithDevice device(ctx, 0);

                std::vector<size_t> size;
                for (int d = 0; d < ndim; ++d) {
                    size.push_back(buf.shape[d]);
                }
                auto t_input = ctx.tensor(size, bmengine::core::DataType::kHalf);
                t_input.from_buffer(buf.ptr);

                // std::cout << t_input << std::endl;
                auto out_data = mds[i]->forward(ctx, t_input);
                // std::cout << out_data << std::endl;
                if (i == 0) {
                    auto converted = model::convert_fp32(ctx, out_data);
                    converted.to_buffer(ndarray.mutable_data());
                }
            });
        }
        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }
        // py::array_t<float> ndarray(out_data.size());
        // auto converted = model::convert_fp32(ctx, out_data);
        // converted.to_buffer(ndarray.mutable_data());
        return ndarray;
    };

    void init_parameters(int seed = 1024) {
        auto ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);
            curandGenerator_t gen;
            CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A));
            CURAND_CHECK(curandSetStream(gen, ctx.current_stream()->ptr));
            CURAND_CHECK(curandSetGeneratorOffset(gen, 0));
            CURAND_CHECK(curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_BEST));
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
            mds[0]->init_parameters(ctx, gen);
            curandDestroyGenerator(gen);
        }
    };

    void load_state_dict(const std::map<std::string, py::array>& state_dict)
        __attribute__((visibility("hidden"))) {

        std::vector<std::thread> threads;

        for (int i = 0; i < engine->num_gpus(); ++i) {

            threads.emplace_back([this, i, &state_dict] {
                auto ctx = engine->create_context({ i });
                bmengine::core::WithDevice device(ctx, 0);
                auto named_params = mds[i]->named_parameters("ff", true);
                bind::load_state_dict(ctx, state_dict, named_params);
            });
        }

        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }
    }

    std::map<const std::string, py::array_t<float>> named_parameters()
        __attribute__((visibility("hidden"))) {
        std::map<const std::string, py::array_t<float>> result;
        std::map<const std::string, bmengine::core::Tensor> res_tensors;
        std::vector<std::thread> threads;

        for (int i = 0; i < engine->num_gpus(); ++i) {

            threads.emplace_back([this, i, &res_tensors] {
                auto ctx = engine->create_context({ i });
                bmengine::core::WithDevice device(ctx, 0);

                auto named_params = mds[i]->named_parameters("ff", true);
                for (auto it : named_params) {
                    if (i == 0) {
                        auto converted = model::convert_fp32(ctx, *it.second);
                        res_tensors.emplace(it.first, std::move(converted));
                    }
                }
            });
        }

        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }

        auto ctx = engine->create_context({ 0 });
        bmengine::core::WithDevice device(ctx, 0);
        for (auto it : res_tensors) {
            py::array_t<float> ndarray(it.second.size());
            try {
                it.second.to_buffer(ndarray.mutable_data());
                result.emplace(it.first, std::move(ndarray));

            } catch (const BMEngineException& e) {
                std::cerr << e.what() << std::endl;
                throw std::logic_error(e.what());
            }
        }
        return result;
    }
};

namespace bind {
void define_layer_feed_forward(py::module_& layers_m) {
    py::class_<PyFeedForward>(layers_m, "FeedForward")
        .def(py::init(&PyFeedForward::create))
        .def("init_parameters", &PyFeedForward::init_parameters)
        .def("load_state_dict", &PyFeedForward::load_state_dict)
        .def("named_parameters", &PyFeedForward::named_parameters)
        .def("forward", &PyFeedForward::forward);
}
}
