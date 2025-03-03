#include "bmengine/core/tensor.h"
#include "bmengine/core/context.h"
#include "bmengine/core/exception.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sstream>

namespace bmengine {
namespace core {

inline float int2float(uint32_t x) {
    return *reinterpret_cast<float*>(&x);
}
inline uint32_t float2int(float x) {
    return *reinterpret_cast<uint32_t*>(&x);
}
float half2float(uint16_t h) {
    uint32_t x = h;
    const float magic = int2float((254 - 15) << 23);
    const float was_infnan = int2float((127 + 16) << 23);
    uint32_t out;
    out = (x & 0x7fff) << 13;
    out = float2int(magic * int2float(out));
    if (int2float(out) >= was_infnan) {
        out |= 255 << 23;
    }
    out |= (x & 0x8000) << 16;
    return int2float(out);
}
uint16_t float2half(float f) {
    uint32_t f32infty = 255 << 23;
    uint32_t f16max = (127 + 16) << 23;
    float denorm_magic = int2float(((127 - 15) + (23 - 10) + 1) << 23);
    uint32_t sign_mask = 0x80000000u;
    uint16_t o = 0;
    uint32_t f_u = float2int(f);
    uint32_t sign = f_u & sign_mask;
    f_u ^= sign;

    if (f_u >= f16max)
        o = (f_u > f32infty) ? 0x7e00 : 0x7c00;
    else {
        if (f_u < (113 << 23)) {
            f_u = float2int(int2float(f_u) + denorm_magic);
            o = f_u - float2int(denorm_magic);
        } else {
            uint32_t mant_odd = (f_u >> 13) & 1;
            f_u += ((15 - 127) << 23) + 0xfff;
            f_u += mant_odd;
            o = f_u >> 13;
        }
    }
    o |= sign >> 16;
    return o;
}
uint16_t float2bfloat(float f){
  return *reinterpret_cast<uint32_t*>(&f) >> 16;
} 

// bf16 -> fp32
float bfloat2float(uint16_t h){
    uint32_t src = h;
    src <<= 16;
    return *reinterpret_cast<float*>(&src);
}
struct alignas(2) Half {
    uint16_t val;
    Half() = default;
    Half(float x) { val = float2half(x); }
};
struct alignas(2) BFloat {
    uint16_t val;
    BFloat() = default;
    BFloat(float x) { val = float2bfloat(x); }
};

template<typename T, int width, int precision>
std::ostream& print_value(std::ostream& os, const T& v) {
    os << std::setw(width) << std::setprecision(precision) << v;
    return os;
}

template<typename T, int width, int precision>
std::ostream& print_value(std::ostream& os, const int8_t& v) {
    os << std::setw(width) << std::setprecision(precision) << static_cast<int16_t>(v);
    return os;
}

template<typename T, int width, int precision>
std::ostream& print_value(std::ostream& os, const Half& v) {
    os << std::setw(width) << std::setprecision(precision) << half2float(v.val);
    return os;
}

template<typename T, int width, int precision>
std::ostream& print_value(std::ostream& os, const BFloat& v) {
    os << std::setw(width) << std::setprecision(precision) << bfloat2float(v.val);
    return os;
}

template<typename T, int width, int precision>
void print_buffer(std::ostream& os, T* buffer, const std::vector<size_t>& shape, int dim) {
    if (dim + 1 == shape.size()) {
        // last dim
        os << "[";
        if (shape[dim] > 16) {
            for (int i = 0; i < 7; ++i) {
                print_value<T, width, precision>(os, buffer[i]) << ", ";
            }
            os << "... ";
            for (int i = shape[dim] - 7; i < shape[dim]; ++i) {
                print_value<T, width, precision>(os, buffer[i]);
                if (i + 1 < shape[dim]) {
                    os << ", ";
                }
            }
        } else {
            for (int i = 0; i < shape[dim]; ++i) {
                print_value<T, width, precision>(os, buffer[i]);
                if (i + 1 < shape[dim]) {
                    os << ", ";
                }
            }
        }
        os << "]";
    } else {
        os << "[";
        size_t stride = 1;
        for (int i = dim + 1; i < shape.size(); ++i) {
            stride *= shape[i];
        }
        if (shape[dim] > 8) {
            for (int i = 0; i < 5; ++i) {
                if (i > 0)
                    os << std::setw(dim + 1) << "";
                print_buffer<T, width, precision>(os, buffer + i * stride, shape, dim + 1);
                os << "," << std::endl;
                if (dim + 2 < shape.size())
                    os << std::endl;
            }
            os << std::setw(dim + 1) << "";
            os << "..." << std::endl;
            if (dim + 2 < shape.size())
                os << std::endl;

            for (int i = shape[dim] - 5; i < shape[dim]; ++i) {
                os << std::setw(dim + 1) << "";
                print_buffer<T, width, precision>(os, buffer + i * stride, shape, dim + 1);
                if (i + 1 < shape[dim]) {
                    os << "," << std::endl;
                    if (dim + 2 < shape.size())
                        os << std::endl;
                }
            }
        } else {
            for (int i = 0; i < shape[dim]; ++i) {
                if (i > 0)
                    os << std::setw(dim + 1) << "";
                print_buffer<T, width, precision>(os, buffer + i * stride, shape, dim + 1);
                if (i + 1 < shape[dim]) {
                    os << "," << std::endl;
                    if (dim + 2 < shape.size())
                        os << std::endl;
                }
            }
        }
        os << "]";
    }
}

template<typename T, int width = 6, int precision = 4>
void print_tensor(std::ostream& os, const Tensor& tensor) {
    if (tensor.ndim() > 0) {
        T* buffer = new T[tensor.numel()];
        BM_CUDART_ASSERT(
            cudaMemcpy(buffer, tensor.data(), tensor.nbytes(), cudaMemcpyDeviceToHost));
        print_buffer<T, width, precision>(os, buffer, tensor.size(), 0);
        delete[] buffer;
    }
    os << std::endl;
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.info() << std::endl;
    switch (tensor.dtype()) {
        case DataType::kDouble: print_tensor<double>(os, tensor); break;
        case DataType::kFloat: print_tensor<float>(os, tensor); break;
        case DataType::kHalf: print_tensor<Half>(os, tensor); break;
        case DataType::kBFloat16: print_tensor<BFloat>(os, tensor); break;
        case DataType::kInt8: print_tensor<int8_t, 4>(os, tensor); break;
        case DataType::kInt16: print_tensor<int16_t>(os, tensor); break;
        case DataType::kInt32: print_tensor<int32_t>(os, tensor); break;
        default: break;
    }
    return os;
}

const char* get_data_type_name(DataType dtype) {
    switch (dtype) {
        case DataType::kDouble: return "double";
        case DataType::kFloat: return "float";
        case DataType::kHalf: return "half";
        case DataType::kBFloat16: return "bfloat";
        case DataType::kInt8: return "int8";
        case DataType::kInt16: return "int16";
        case DataType::kInt32: return "int32";
        default: return "unknown";
    }
}

static const std::map<const std::string, DataType> datatype_name_mapping {
    { "double", DataType::kDouble },
    { "float", DataType::kFloat },
    { "half", DataType::kHalf },
    { "float16", DataType::kHalf },
    { "bfloat", DataType::kBFloat16 },
    { "bfloat16", DataType::kBFloat16 },
    { "int8", DataType::kInt8 },
    { "int16", DataType::kInt16 },
    { "int32", DataType::kInt32 },
};

DataType name_to_data_type(const std::string& name) {
    if (datatype_name_mapping.count(name)) {
        return datatype_name_mapping.at(name);
    }
    BM_EXCEPTION("unknown datatype name: " + std::string(name));
}

const char* get_dist_layout_name(DistLayout dist_layout) {
    switch (dist_layout) {
        case DistLayout::COLUMNAR: return "COLUMNAR";
        case DistLayout::ROW: return "ROW";
        case DistLayout::REPLICATED: return "REPLICATED";

        default: return "unknown";
    }
}

} // namespace core
} // namespace bmengine
