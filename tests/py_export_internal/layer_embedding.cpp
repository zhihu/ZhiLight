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

namespace py = pybind11;

class PyEmbedding {
private:
    std::shared_ptr<nn::Embedding> md;
    std::shared_ptr<bmengine::core::Engine> engine;
    int dim_model;
    int vocab_size;
    bool scale_weights;
    PyEmbedding(int dim_model, int vocab_size, bool scale_weights = false)
        : dim_model(dim_model), scale_weights(scale_weights) {
        std::vector<bmengine::core::DeviceConfiguration> devices;
        devices.emplace_back(0, (size_t) 2 << 28);

        engine = std::make_shared<bmengine::core::Engine>(devices);

        auto ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);
            md = std::make_shared<nn::Embedding>(ctx, dim_model, vocab_size, scale_weights);
        }
    };

public:
    ~PyEmbedding() {
        // nessesary to release md before engine, cycle refeernce.
        md = nullptr;
    }
    PyEmbedding(const PyEmbedding& other) {
        md = other.md;
        engine = other.engine;
        dim_model = other.dim_model;
        vocab_size = other.vocab_size;
        scale_weights = other.scale_weights;
    }
    PyEmbedding(PyEmbedding&& other) {
        md = std::move(other.md);
        engine = std::move(other.engine);
        dim_model = other.dim_model;
        vocab_size = other.vocab_size;
        scale_weights = other.scale_weights;
    }
    PyEmbedding& operator=(const PyEmbedding& other) {
        md = other.md;
        engine = other.engine;
        dim_model = other.dim_model;
        vocab_size = other.vocab_size;
        scale_weights = other.scale_weights;
        return *this;
    }
    PyEmbedding& operator=(PyEmbedding&& other) {
        md = std::move(other.md);
        engine = std::move(other.engine);
        dim_model = other.dim_model;
        vocab_size = other.vocab_size;
        scale_weights = other.scale_weights;
        return *this;
    }
    static PyEmbedding create(int dim_model, int vocab_size, bool scale_weights = false) {
        auto ff = PyEmbedding(dim_model, vocab_size, scale_weights);
        return ff;
    };

    py::array forward(py::array& ids, py::array& ids_sub) {
        auto ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);

            auto ids_buf = ids.request();
            std::vector<size_t> ids_size;
            for (int i = 0; i < ids.ndim(); ++i) {
                ids_size.push_back(ids_buf.shape[i]);
            }
            auto t_ids = ctx.tensor(ids_size, bmengine::core::DataType::kInt32);
            t_ids.from_buffer(ids_buf.ptr);

            auto subs_buf = ids_sub.request();
            std::vector<size_t> subs_size;
            for (int i = 0; i < ids_sub.ndim(); ++i) {
                subs_size.push_back(subs_buf.shape[i]);
            }
            auto t_subs = ctx.tensor(subs_size, bmengine::core::DataType::kInt32);
            t_subs.from_buffer(subs_buf.ptr);

            auto out_data = md->forward(ctx, t_ids, t_subs);
            py::array_t<float> ndarray(out_data.size());
            auto converted = model::convert_fp32(ctx, out_data);
            converted.to_buffer(ndarray.mutable_data());
            return ndarray;
        }
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
            md->init_parameters(ctx, gen);
            curandDestroyGenerator(gen);
        }
    };

    void load_state_dict(const std::map<std::string, py::array>& state_dict) {

        auto ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);
            auto named_params = md->named_parameters("token_embedding", true);
            bind::load_state_dict(ctx, state_dict, named_params);
        }
    };

    std::map<const std::string, py::array_t<float>> named_parameters() {
        std::map<const std::string, py::array_t<float>> result;

        auto ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);

            auto named_params = md->named_parameters("token_embedding", true);
            for (auto it : named_params) {
                py::array_t<float> ndarray(it.second->size());
                auto converted = model::convert_fp32(ctx, *it.second);
                converted.to_buffer(ndarray.mutable_data());
                result.emplace(it.first, ndarray);
            }
            return result;
        }
    }
};

namespace bind {
void define_layer_embedding(py::module_& layers_m) {
    py::class_<PyEmbedding>(layers_m, "Embedding")
        .def(py::init(&PyEmbedding::create))
        .def("init_parameters", &PyEmbedding::init_parameters)
        .def("load_state_dict", &PyEmbedding::load_state_dict)
        .def("named_parameters", &PyEmbedding::named_parameters)
        .def("forward", &PyEmbedding::forward);
}
}
