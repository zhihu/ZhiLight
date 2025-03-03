#include "bind_internal.h"
#include "internal_utils.h"

#include "nn/nn.h"
#include "nn/position/rotary_embedding.h"
#include "model/model.h"
#include "utils/exception.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/all.h>
#include <pybind11/numpy.h>
#include <csignal>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <tuple>
#include <torch/extension.h>
#include <ATen/ATen.h>

namespace py = pybind11;

using bmengine::core::Engine;
using std::shared_ptr;
class PyFunctions {

    static shared_ptr<Engine> create_test_engine() {
        std::vector<bmengine::core::DeviceConfiguration> devices;
        devices.emplace_back(0, (size_t) 2 << 30);
        return std::make_shared<Engine>(devices);
    }

public:
    static std::tuple<float, std::vector<float>> log_prob(
        std::vector<float> inputs,
        std::vector<size_t> input_size,
        std::vector<float> inputs_ext,
        std::vector<size_t> input_ext_size,
        std::vector<int> labels) {
        auto engine = create_test_engine();
        bmengine::core::Context ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);
            auto input_tensor = ctx.tensor(input_size, bmengine::core::DataType::kFloat);
            input_tensor.from_buffer(inputs.data());
            auto input_ext_tensor = ctx.tensor(input_ext_size, bmengine::core::DataType::kFloat);
            input_ext_tensor.from_buffer(inputs_ext.data());
            auto label_tensor = ctx.tensor({ input_size[0] }, bmengine::core::DataType::kInt32);
            label_tensor.from_buffer(labels.data());
            auto log_prob = nn::log_prob(
                ctx, std::make_tuple(input_tensor, input_ext_tensor), label_tensor, -100);
            auto out_prob = std::get<0>(log_prob);
            auto out_data = std::get<1>(log_prob);
            std::vector<float> output;
            output.resize(out_data.numel());
            out_data.to_buffer(output.data());
            float res[1];
            out_prob.to_buffer(&res);
            return std::make_tuple(res[0], output);
        }
    }

    static std::tuple<py::array, py::array> rotary_embedding_2(
        int dim_head, py::array& pos, py::array& q, py::array& k) {
        auto engine = create_test_engine();

        model::ModelConfig model_config("", 1, dim_head, 1, dim_head, dim_head, 4096);

        auto ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);

            auto pos_buf = pos.request();
            std::vector<size_t> pos_size;
            for (int i = 0; i < pos.ndim(); ++i) {
                pos_size.push_back(pos_buf.shape[i]);
            }
            auto t_pos = ctx.tensor(pos_size, bmengine::core::DataType::kInt32);
            t_pos.from_buffer(pos_buf.ptr);

            auto q_buf = q.request();
            std::vector<size_t> q_size;
            for (int i = 0; i < q.ndim(); ++i) {
                q_size.push_back(q_buf.shape[i]);
            }
            auto t_q = ctx.tensor(q_size, bmengine::core::DataType::kHalf);
            t_q.from_buffer(q_buf.ptr);

            auto k_buf = k.request();
            std::vector<size_t> k_size;
            for (int i = 0; i < k.ndim(); ++i) {
                k_size.push_back(k_buf.shape[i]);
            }
            auto t_k = ctx.tensor(k_size, bmengine::core::DataType::kHalf);
            t_k.from_buffer(k_buf.ptr);

            auto rotary_embedding = nn::RotaryEmbedding(ctx, model_config);
            bmengine::core::Tensor h_q, h_k;
            std::tie(h_q, h_k) = rotary_embedding(ctx, t_pos, t_q, t_k);

            py::array_t<float> out_q(h_q.size());
            model::convert_fp32(ctx, h_q).to_buffer(out_q.mutable_data());
            py::array_t<float> out_k(h_k.size());
            model::convert_fp32(ctx, h_k).to_buffer(out_k.mutable_data());
            return std::make_tuple(out_q, out_k);
        }
    }

    static py::array attn_softmax(
        float scale, py::array& attn_score, py::array& mask, py::array& position_bias) {
        bmengine::core::Storage* storage;
        bmengine::core::Engine* engine;
        std::vector<bmengine::core::DeviceConfiguration> devices;
        devices.emplace_back(0, (size_t) 2 << 30);

        engine = new bmengine::core::Engine(devices);

        auto ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);

            auto attn_score_buf = attn_score.request();
            std::vector<size_t> attn_score_size;
            for (int i = 0; i < attn_score.ndim(); ++i) {
                attn_score_size.push_back(attn_score_buf.shape[i]);
            }
            auto t_attn_score = ctx.tensor(attn_score_size, bmengine::core::DataType::kHalf);
            t_attn_score.from_buffer(attn_score_buf.ptr);

            auto mask_buf = mask.request();
            std::vector<size_t> mask_size;
            for (int i = 0; i < mask.ndim(); ++i) {
                mask_size.push_back(mask_buf.shape[i]);
            }
            auto t_mask = ctx.tensor(mask_size, bmengine::core::DataType::kInt8);
            t_mask.from_buffer(mask_buf.ptr);

            auto position_bias_buf = position_bias.request();
            std::vector<size_t> position_bias_size;
            for (int i = 0; i < position_bias.ndim(); ++i) {
                position_bias_size.push_back(position_bias_buf.shape[i]);
            }
            auto t_position_bias = ctx.tensor(position_bias_size, bmengine::core::DataType::kHalf);
            t_position_bias.from_buffer(position_bias_buf.ptr);

            nn::attn_softmax(
                ctx, scale, t_attn_score, t_mask, t_position_bias

            );

            py::array_t<float> out(t_attn_score.size());
            model::convert_fp32(ctx, t_attn_score).to_buffer(out.mutable_data());
            return out;
        }
    }

    static py::array concat_tensor(py::array& A, py::array& B, int dim) {
        auto engine = create_test_engine();

        auto ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);

            auto A_buf = A.request();
            std::vector<size_t> A_size;
            for (int i = 0; i < A.ndim(); ++i) {
                A_size.push_back(A_buf.shape[i]);
            }
            auto t_A = ctx.tensor(A_size, bmengine::core::DataType::kHalf);
            t_A.from_buffer(A_buf.ptr);
            auto B_buf = B.request();
            std::vector<size_t> B_size;
            for (int i = 0; i < B.ndim(); ++i) {
                B_size.push_back(B_buf.shape[i]);
            }
            auto t_B = ctx.tensor(B_size, bmengine::core::DataType::kHalf);
            t_B.from_buffer(B_buf.ptr);

            auto t_C = bmengine::functions::concat_tensor(ctx, t_A, t_B, dim);

            py::array_t<float> out(t_C.size());
            model::convert_fp32(ctx, t_C).to_buffer(out.mutable_data());
            return out;
        }
    }

    static at::Tensor index_along_dim(at::Tensor input, int dim, at::Tensor index) {
        auto engine = create_test_engine();

        auto ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);
            auto t_input = bind::aten_to_tensor(ctx, input);
            auto t_index = bind::aten_to_tensor(ctx, index);
            auto t_output = bmengine::functions::index_along_dim(ctx, t_input, dim, t_index);
            return bind::tensor_to_aten(ctx, t_output);
        }
    }

    static at::Tensor sum(at::Tensor input) {
        auto engine = create_test_engine();

        auto ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);
            auto t_input = bind::aten_to_tensor(ctx, input);
            auto t_output = bmengine::functions::sum(ctx, t_input);
            return bind::tensor_to_aten(ctx, t_output);
        }
    }

    static at::Tensor div(at::Tensor input_a, at::Tensor input_b, float eps) {
        auto engine = create_test_engine();

        auto ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);
            auto t_input_a = bind::aten_to_tensor(ctx, input_a);
            auto t_input_b = bind::aten_to_tensor(ctx, input_b);
            auto t_output = bmengine::functions::div(ctx, t_input_a, t_input_b, eps);
            return bind::tensor_to_aten(ctx, t_output);
        }
    }

    static at::Tensor softmax(at::Tensor scores, float temperature) {
        auto engine = create_test_engine();

        auto ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);
            at::Tensor results = torch::empty_like(scores);
            auto input = bind::aten_to_tensor(ctx, scores);
            auto output = bind::aten_to_tensor(ctx, results);
            nn::softmax(ctx, input, output, temperature);
            return results;
        }
    }

    static at::Tensor amax(at::Tensor input) {
        auto engine = create_test_engine();

        auto ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);
            auto t_input = bind::aten_to_tensor(ctx, input);
            auto t_output = bmengine::functions::amax(ctx, t_input);
            return bind::tensor_to_aten(ctx, t_output);
        }
    }

    static at::Tensor amin(at::Tensor input) {
        auto engine = create_test_engine();

        auto ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);
            auto t_input = bind::aten_to_tensor(ctx, input);
            auto t_output = bmengine::functions::amin(ctx, t_input);
            return bind::tensor_to_aten(ctx, t_output);
        }
    }

    static at::Tensor add(at::Tensor a, float b) {
        auto engine = create_test_engine();

        auto ctx = engine->create_context({ 0 });
        {
            auto d = ctx.with_device(0);
            auto t_a = bind::aten_to_tensor(ctx, a);
            auto t_output = bmengine::functions::add(ctx, t_a, b);
            return bind::tensor_to_aten(ctx, t_output);
        }
    }
};

namespace bind {
void define_functions(py::module_& m) {
    py::module_ funcs_m = m.def_submodule("functions", "internal functions for testing ");
    funcs_m.def("log_prob", &PyFunctions::log_prob);
    funcs_m.def("rotary_embedding_2", &PyFunctions::rotary_embedding_2);
    funcs_m.def("attn_softmax", &PyFunctions::attn_softmax);
    funcs_m.def("concat_tensor", &PyFunctions::concat_tensor);
    funcs_m.def("softmax", &PyFunctions::softmax);
    funcs_m.def("sum", &PyFunctions::sum);
    funcs_m.def("div", &PyFunctions::div);
    funcs_m.def("amax", &PyFunctions::amax);
    funcs_m.def("amin", &PyFunctions::amin);
    funcs_m.def("add", &PyFunctions::add);
    funcs_m.def("index_along_dim", &PyFunctions::index_along_dim);
}
}
