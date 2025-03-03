#include "bind_internal.h"
#include "internal_utils.h"

#include <bmengine/core/core.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_fp16.h>
#include <sstream>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>

namespace bind {

using bmengine::core::Context;
using bmengine::core::DataType;
using bmengine::core::Tensor;

DataType aten_typemeta_to_bmengine(caffe2::TypeMeta type_meta) {
    auto scalar_type = type_meta.toScalarType();
    switch (scalar_type) {
        case at::ScalarType::Double: return DataType::kDouble;
        case at::ScalarType::Float: return DataType::kFloat;
        case at::ScalarType::Half: return DataType::kHalf;
        case at::ScalarType::Char: return DataType::kInt8;
        case at::ScalarType::Short: return DataType::kInt16;
        case at::ScalarType::Int: return DataType::kInt32;
        case at::ScalarType::BFloat16: return DataType::kBFloat16;
        default: break;
    }
    std::stringstream ss;
    ss << std::string("can't convert at::Tensor of scalar_type ") << type_meta.name()
       << "The only supported types are: "
          "Double, Float, Half, Chat, Short, Int and BFloat16.";
    throw std::runtime_error(ss.str());
}

at::ScalarType bmengine_to_aten_scalertype(DataType dtype) {
    switch (dtype) {
        case DataType::kDouble: return at::ScalarType::Double;
        case DataType::kFloat: return at::ScalarType::Float;
        case DataType::kHalf: return at::ScalarType::Half;
        case DataType::kInt8: return at::ScalarType::Char;
        case DataType::kInt16: return at::ScalarType::Short;
        case DataType::kInt32: return at::ScalarType::Int;
        case DataType::kBFloat16: return at::ScalarType::BFloat16;
        default: break;
    }
    throw std::runtime_error(
        std::string("can't convert dtype ") + get_data_type_name(dtype)
        + "The only supported types are: "
          "kDouble, kFloat, kHalf, kInt8, kInt16, kInt32 and kBFloat16.");
}

// Create a tensor reference to pytorch tensor's underlying data.
// No data coping happens.
const Tensor aten_to_tensor(const Context& ctx, const at::Tensor& at_tensor) {
    if (at_tensor.numel() == 0) {
        return Tensor();
    }
    auto shape = at_tensor.sizes().vec();
    std::vector<size_t> sizes(shape.begin(), shape.end());
    auto dtype = aten_typemeta_to_bmengine(at_tensor.dtype());
    if (at_tensor.is_cpu()) {
        auto tensor = ctx.tensor(sizes, dtype);
        tensor.from_buffer(at_tensor.data_ptr());
        return std::move(tensor);
    } else {
        BM_ASSERT(
            at_tensor.is_cuda() && at_tensor.get_device() == ctx.active_device(),
            "tensor device miss match.");

        const auto tensor = Tensor::from_external(
            sizes, dtype, at_tensor.data_ptr(), at_tensor.nbytes(), ctx.active_device());
        return std::move(tensor);
    }
}

at::Tensor tensor_to_aten(const Context& ctx, const Tensor& tensor) {
    auto sizes = tensor.shape();
    at::Tensor at_tensor = at::empty(
        at::IntArrayRef(std::vector<int64_t>(sizes.begin(), sizes.end())),
        at::initialTensorOptions()
            .device(at::Device(at::DeviceType::CUDA, tensor.device()))
            .dtype(bmengine_to_aten_scalertype(tensor.dtype())));
    BM_CUDART_ASSERT(
        cudaMemcpy(at_tensor.data_ptr(), tensor.data(), tensor.nbytes(), cudaMemcpyDeviceToDevice));
    return std::move(at_tensor);
}

void load_at_state_dict(
    bmengine::core::Context& ctx,
    const std::map<std::string, at::Tensor>& state_dict,
    std::map<const std::string, bmengine::core::Tensor*> named_params,
    bool parallel) {
    for (auto it : named_params) {
        auto p = state_dict.find(it.first);
        if (p != state_dict.end()) {
            if (!parallel || ctx.rank() == 0) {
                auto tensor = aten_to_tensor(ctx, p->second);
                *it.second = tensor;
            } else {
                throw std::runtime_error("parallel not supported yet.");
            }

        } else {
            std::stringstream ss;
            ss << "state_dict missing: " << it.first;
            throw std::runtime_error(ss.str());
        }
    }
}
}
