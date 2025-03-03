#include "py_export/bind.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_fp16.h>
#include <sstream>
#include <iostream>
#include <bmengine/core/tensor.h>
#include "model/model.h"

namespace bind {

using bmengine::core::Context;
using bmengine::core::DataType;
using bmengine::core::Tensor;

// Create a tensor reference to numpy array's underlying data.
// No data coping happens if copy==false.
Tensor numpy_to_tensor(const std::string& name, const py::array& arr, bool copy) {
    py::format_descriptor<float>::format();
    py::buffer_info buf = arr.request();
    auto dtype = numpy_dtype_to_bmengine(arr.dtype());
    void* ptr = buf.ptr;
    if (copy) {
        ptr = new char[arr.nbytes()];
        memcpy(ptr, buf.ptr, arr.nbytes());
    }
    auto tensor = Tensor::from_external(
        *reinterpret_cast<std::vector<size_t>*>(&buf.shape),
        dtype,
        ptr,
        arr.nbytes(),
        -1,
        copy);
    tensor.set_name(name);
    return std::move(tensor);
}

std::map<std::string, const Tensor> numpy_to_tensor(
    const std::map<std::string, py::array>& state_dict) {
    std::map<std::string, const Tensor> tensor_dict;
    for (auto& it : state_dict) {
        tensor_dict.emplace(it.first, std::move(bind::numpy_to_tensor(it.first, it.second)));
    }
    return tensor_dict;
}

std::map<std::string, const bmengine::core::Tensor> numpy_to_tensor(py::dict state_dict) {
    std::map<std::string, const Tensor> tensor_dict;
    for (auto& it : state_dict) {
        std::string name = py::str(it.first);
        tensor_dict.emplace(name, std::move(bind::numpy_to_tensor(name, reinterpret_cast<const py::array&>(it.second))));
    }
    return tensor_dict;
}

// load recursively
void load_state_dict_new(
    bmengine::core::Context& ctx,
    const std::map<std::string, py::array>& state_dict,
    const std::string& prefix,
    bmengine::core::Layer* model) {
    std::map<std::string, const Tensor> tensor_dict = numpy_to_tensor(state_dict);

    // load params recursively
    model->load_state_dict(ctx, tensor_dict, prefix);

    // check all params was initialized
    auto named_parameters = model->named_parameters(prefix);
    for (auto& p : named_parameters) {
        if (p.second->data() == nullptr) {
            throw std::runtime_error("state_dict missing: " + p.first);
        }
    }
}

void load_state_dict(
    bmengine::core::Context& ctx,
    const std::map<std::string, py::array>& state_dict,
    std::map<const std::string, bmengine::core::Tensor*> named_params,
    bool parallel) {
    for (auto it : named_params) {
        BM_ASSERT(it.second, it.first + std::string(" not in named_params"));
        auto p = state_dict.find(it.first);
        if (p != state_dict.end()) {
            if (!parallel || ctx.rank() == 0) {

                auto buf = p->second.request();
                BM_ASSERT(
                    it.second->ndim() == buf.ndim,
                    it.first + " ndim miss match: " + std::to_string(it.second->ndim())
                        + " != " + std::to_string(buf.ndim));
                for (int i = 0; i < it.second->ndim(); ++i) {
                    std::stringstream ss;
                    ss << "model[" << i << "]=" << it.second->shape()[i] << ", state[" << i
                       << "]=" << buf.shape[i];
                    // std::cout << ss.str() + "=>wjj" << std::endl;
                    BM_ASSERT(
                        it.second->shape()[i] == buf.shape[i],
                        "Parameter `" + it.first + "` has different shape" + ss.str());
                }
                BM_ASSERT(
                    it.second->nbytes() == (buf.size * buf.itemsize),
                    it.first + " size miss match: " + std::to_string(it.second->nbytes())
                        + " != " + std::to_string(buf.size * buf.itemsize));
                // TODO add dtype check with numpy.
                // BM_ASSERT(py::dtype == p->second.dtype().num(), it.first + " dtype miss match" +
                // std::string(get_data_type_name(it.second->dtype())) + p->second.dtype().char_());
                ctx.init_parameter(it.first, it.second);
                it.second->from_buffer(buf.ptr);
            } else {
                it.second->from_buffer(nullptr);
            }

        } else {
            std::stringstream ss;
            ss << "state_dict missing: " << it.first;
            throw std::runtime_error(ss.str());
        }
    }
}

void convert_results(const std::vector<generator::SearchResults>& results_vec, py::list* res) {
    for (const auto& results : results_vec) {
        const generator::SearchResult& result = results.results[0];
        res->append(py::make_tuple(
            py::cast(result.tokens),
            py::cast(result.logprobs),
            py::cast(result.cumulative_logprob),
            py::cast(result.score)));
    }
}

void convert_multi_results(
    const std::vector<generator::SearchResults>& results_vec, py::list* res) {
    for (const auto& results : results_vec) {
        py::list l;
        for (auto result : results.results) {
            l.append(py::make_tuple(
                py::cast(result.tokens),
                py::cast(result.logprobs),
                py::cast(result.cumulative_logprob),
                py::cast(result.score)));
        }
        res->append(l);
    }
}

template<typename T>
std::vector<T> to_1d_vector(const py::list& z) {
    std::vector<T> v;
    for (const auto& it : z) {
        v.emplace_back(it.cast<T>());
    }
    return v;
}

std::vector<int> to_int_vector(const py::list& data_list) {
    return to_1d_vector<int>(data_list);
}

std::vector<std::string> to_string_vector(const py::list& data_list) {
    return to_1d_vector<std::string>(data_list);
}

template<typename T>
std::vector<std::vector<T>> to_2d_vector(const py::list& data_list) {
    std::vector<std::vector<T>> c_data_list;
    for (const auto& it : data_list) {
        std::vector<T> token_ids;
        for (const auto& jt : it) {
            token_ids.emplace_back(jt.cast<T>());
        }
        c_data_list.emplace_back(token_ids);
    }
    return c_data_list;
}

std::vector<std::vector<int>> to_2d_int_vector(const py::list& data_list) {
    return to_2d_vector<int>(data_list);
}

std::vector<std::vector<bool>> to_2d_bool_vector(const py::list& data_list) {
    return to_2d_vector<bool>(data_list);
}

bmengine::core::DataType numpy_dtype_to_bmengine(py::dtype dtype) {
    switch (dtype.char_()) {
        case 'd': return bmengine::core::DataType::kDouble;
        case 'f': return bmengine::core::DataType::kFloat;
        case 'e': return bmengine::core::DataType::kHalf;
        case 'b': return bmengine::core::DataType::kInt8;
        case 'h': return bmengine::core::DataType::kInt16;
        case 'i': return bmengine::core::DataType::kInt32;
        default: break;
    }
    throw std::runtime_error(
        std::string("can't convert np.ndarray of type ") + dtype.char_()
        + "The only supported types are: "
          "float63, float32, float16, int32, int16, half and int8.");
}

}
