#include "py_export/bind.h"
#include "py_export/py_model_base.h"
#include "model/llama.h"
#include "utils/array_guard.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/typecast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <random>

namespace py = pybind11;

using bmengine::core::Tensor;
using bmengine::core::DataType;
typedef std::map<std::string, py::array> NumpyMap;
typedef std::shared_ptr<bmengine::core::Engine> EnginePtr;
typedef py::object ContextObj;

class PyLLaMA : public PyModelBase {
private:
    EnginePtr engine_;
    model::ModelConfig model_config_;
    std::vector<model::ModelBase*> models_;

public:
    PyLLaMA(
        EnginePtr engine,
        model::ModelConfig model_config,
        model::QuantConfig quant_config,
        bmengine::core::DistConfiguration dist_config)
        : PyModelBase("llama", false), engine_(engine), model_config_(model_config) {
        std::cout << model_config.to_string() << std::endl;

        bool parallel = false;
        if (dist_config.tp > 1 || (dist_config.tp < 0 && (engine->num_gpus() > 1 || dist_config.nnodes > 1))) {
            parallel = true;
        }
        set_parallel(parallel);

        models_.resize(engine->local_ranks());
        engine->device_foreach([this, &model_config, quant_config, parallel](int i) {
            auto ctx = engine_->create_context_rank(i);
            auto with_device = ctx.with_device(0);
            models_[i] = new model::LLaMA(ctx, model_config, quant_config, parallel);
        });
    }

    ~PyLLaMA() {
        if (engine_ && !models_.empty()) {
            engine_->device_foreach([this](int i) {
                delete models_[i];
            });
            models_.clear();
        }
    }
    PyLLaMA(const PyLLaMA& other) = default;
    PyLLaMA(PyLLaMA&& other) = default;
    PyLLaMA& operator=(const PyLLaMA& other) = default;
    PyLLaMA& operator=(PyLLaMA&& other) = default;

    static PyLLaMA create(
        EnginePtr engine,
        model::ModelConfig& model_config,
        model::QuantConfig quant_config,
        bool parallel = false) {
        return PyLLaMA(engine, model_config, quant_config, parallel);
    }

    virtual bmengine::core::Engine* engine() {
        return engine_.get();
    }
    virtual std::vector<model::ModelBase*> par_models() {
        return models_;
    }

    void load_state_dict(const std::map<std::string, py::array>& state_dict) {
        auto tensor_dict = bind::numpy_to_tensor(state_dict);
        engine_->device_foreach([this, &tensor_dict](int i) {
            auto ctx = engine_->create_context_rank(i);
            auto with_device = ctx.with_device(0);
            // load params recursively
            models_[i]->load_state_dict(ctx, tensor_dict, prefix);
        });
        on_load();
    }

    ContextInfoPtr parse_context(const ContextObj& obj) {
        // Reserve for future usage
        return ContextInfoPtr();
    }

    std::vector<int8_t> get_mask(int len_seq) {
        std::vector<int8_t> mask(len_seq * len_seq);
        for (size_t i = 0; i < len_seq; i++) {
            for (size_t j = 0; j < len_seq; j++) {
                mask[i * len_seq + j] = i < j ? 0 : 1;
            }
        }
        return mask;
    }

    model::ModelContext create_ctx(int dev, model::LLaMA* model, int batch_size = 1) {
        bmengine::core::Context c_ctx = is_parallel() ?
                                        engine()->create_context({ dev }) :
                                        engine()->create_context();
        return model::ModelContext(std::move(c_ctx), *model, batch_size, is_parallel());
    }

    void run(std::function<void(int)> fn) {
        py::gil_scoped_release release;
        if (is_parallel()) {
            engine()->device_foreach(fn);
        } else {
            fn(0);
        }
    }

    py::dict calc_act_scales(py::list data_list) {
        std::vector<std::vector<int>> c_data_list = bind::to_2d_int_vector(data_list);
        std::map<std::string, std::vector<float>> map;
        auto fn = [&, this](int i) {
            model::LLaMA* model = dynamic_cast<model::LLaMA*>(get_model(i));
            auto ctx = create_ctx(i, model);
            auto d = ctx.with_device(0);

            ctx.set_calc_act_scales(true);
            for (auto& data: c_data_list) {
                BM_ASSERT(data.size() > 0, "data is empty");
                size_t len_seq = data.size();
                auto inp = ctx.tensor_of(data);
                std::vector<int> pos(len_seq);
                std::iota(pos.begin(), pos.end(), 0);
                auto d_pos = ctx.tensor_of(pos);
                auto mask = ctx.tensor_of(get_mask(len_seq), {len_seq, len_seq});
                ctx.resize_transformer_buf(len_seq);
                model->encode(ctx, inp, d_pos, Tensor(), Tensor(), mask, d_pos, Tensor());
            }
            auto map1 = ctx.get_act_scale_map();
            BM_ASSERT(!map1.empty(), "act_scale_map is empty");
            if (!is_parallel() || ctx.rank() == 0) {
                for (auto& item: map1) {
                    const Tensor* tensor = ctx.identity(&item.second, "");
                    Tensor f_t = bmengine::functions::typecast(
                        ctx, *tensor, bmengine::core::DataType::kFloat);
                    BM_ASSERT_EQ(f_t.ndim(), 1, "not 1-D tensor");
                    std::vector<float> vec(f_t.numel());
                    f_t.to_buffer(vec.data());
                    map.emplace(item.first, vec);
                }
            }
        };
        run(fn);

        py::dict dict;
        for (auto& item: map) {
            py::array_t<float> np_arr(py::ssize_t(item.second.size()), item.second.data());
            dict[item.first.c_str()] = np_arr;
        }
        return dict;
    }

    py::array get_input_embeddings(py::list data_list) {
        if (!loaded_) throw std::runtime_error("Model is not loaded!!!");
        std::vector<int> h_ids = bind::to_int_vector(data_list);
        py::array_t<float> out_emb({h_ids.size(), size_t(models_[0]->dim_model)});
        std::map<std::string, std::vector<float>> map;
        auto fn = [&, this](int i) {
            model::LLaMA* model = dynamic_cast<model::LLaMA*>(get_model(i));
            auto ctx = create_ctx(i, model);
            auto d = ctx.with_device(0);

            Tensor ids = ctx.tensor_of(h_ids);
            Tensor embeddings = model->get_input_embeddings(ctx, ids);
            BM_ASSERT_EQ(embeddings.size(-1), models_[0]->dim_model, "dim mismatch");
            if (!is_parallel() || ctx.rank() == 0) {
                model::convert_fp32(ctx, embeddings).to_buffer(out_emb.mutable_data());
            }
        };
        run(fn);
        return out_emb;
    }

    void load_with_smooth_quant(
        NumpyMap& state_dict, NumpyMap& scale_dict,
        float alpha,
        float min_scale,
        float max_scale) {
        auto tensor_dict = bind::numpy_to_tensor(state_dict);
        auto scale_map = bind::numpy_to_tensor(scale_dict);
        auto fn = [&, this](int i) {
            model::LLaMA* model = dynamic_cast<model::LLaMA*>(get_model(i));
            auto ctx = create_ctx(i, model);
            auto d = ctx.with_device(0);

            ctx.set_smooth_quant(
                reinterpret_cast<std::map<std::string, Tensor>&>(scale_map),
                alpha,
                min_scale,
                max_scale);
            model->load_state_dict(ctx, tensor_dict, prefix);
        };
        run(fn);
        on_load();
    }

};

namespace bind {
void define_llama(py::module_& m) {
    py::class_<PyLLaMA, PyModelBase>(m, "LLaMA")
        .def(py::init(&PyLLaMA::create))
        .def("get_input_embeddings", &PyLLaMA::get_input_embeddings)
        .def("load_state_dict", &PyLLaMA::load_state_dict)
        .def("load_with_smooth_quant", &PyLLaMA::load_with_smooth_quant)
        .def("calc_act_scales", &PyLLaMA::calc_act_scales);
}
}