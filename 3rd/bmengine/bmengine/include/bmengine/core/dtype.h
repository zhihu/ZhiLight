#pragma once
#include "bmengine/core/export.h"
#include <memory>
#include <initializer_list>
#include <string>
#include <vector>
#include <stdexcept>

namespace bmengine {
namespace core {

enum class DataType {
    kDouble,
    kFloat,
    kHalf,
    kInt8,
    kInt16,
    kInt32,
    kBFloat16,
    kFP8_E4M3,
    kFP8_E5M2,
};

BMENGINE_EXPORT const char* get_data_type_name(DataType dtype);
BMENGINE_EXPORT DataType name_to_data_type(const std::string& name);

template<typename T>
struct DTypeDeducer {
    static DataType data_type() { throw std::runtime_error("data_type must be overwrite"); }
};

template<>
struct DTypeDeducer<int> {
    static DataType data_type() { return DataType::kInt32; }
};
template<>
struct DTypeDeducer<int8_t> {
    static DataType data_type() { return DataType::kInt8; }
};
template<>
struct DTypeDeducer<float> {
    static DataType data_type() { return DataType::kFloat; }
};
template<>
struct DTypeDeducer<void*> {
    // TODO: define kLong to represent address
    static DataType data_type() { return DataType::kDouble; }
};

#define BM_PRIVATE_DTYPE_DISPATCH_CASE(dtypename, realname, ...)                                   \
    case bmengine::core::DataType::dtypename: {                                                    \
        using scalar_t = realname;                                                                 \
        { __VA_ARGS__ }                                                                            \
    } break;

#define BM_DTYPE_DISPATCH(dtype, ...)                                                              \
    do {                                                                                           \
        const auto& _dtype = dtype;                                                                \
        switch (_dtype) {                                                                          \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kDouble, double, __VA_ARGS__)                           \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kFloat, float, __VA_ARGS__)                             \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kHalf, half, __VA_ARGS__)                               \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kInt8, int8_t, __VA_ARGS__)                             \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kInt16, int16_t, __VA_ARGS__)                           \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kInt32, int32_t, __VA_ARGS__)                           \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kBFloat16, nv_bfloat16, __VA_ARGS__)                    \
            default: BM_EXCEPTION("Unsupported data type");                                        \
        };                                                                                         \
    } while (0)

#define BM_DTYPE_DISPATCH_FOR_COPY(dtype, ...)                                                              \
    do {                                                                                           \
        const auto& _dtype = dtype;                                                                \
        switch (_dtype) {                                                                          \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kDouble, double, __VA_ARGS__)                           \
            case bmengine::core::DataType::kFloat: /* fall through */                              \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kInt32, int32_t, __VA_ARGS__)                           \
            case bmengine::core::DataType::kHalf: /* fall through */                               \
            case bmengine::core::DataType::kBFloat16: /* fall through */                           \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kInt16, int16_t, __VA_ARGS__)                           \
            case bmengine::core::DataType::kFP8_E4M3: /* fall through */                           \
            case bmengine::core::DataType::kFP8_E5M2: /* fall through */                           \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kInt8, int8_t, __VA_ARGS__)                             \
            default: BM_EXCEPTION("Unsupported data type");                                        \
        };                                                                                         \
    } while (0)

#define BM_DTYPE_DISPATCH_FLOAT(dtype, ...)                                                        \
    do {                                                                                           \
        const auto& _dtype = dtype;                                                                \
        switch (_dtype) {                                                                          \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kFloat, float, __VA_ARGS__)                             \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kHalf, half, __VA_ARGS__)                               \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kBFloat16, nv_bfloat16, __VA_ARGS__)                    \
            default: BM_EXCEPTION(std::string("Unsupported float data type ") + core::get_data_type_name(_dtype));  \
        };                                                                                         \
    } while (0)

#define BM_DTYPE_DISPATCH_HALF(dtype, ...)                                                         \
    do {                                                                                           \
        const auto& _dtype = dtype;                                                                \
        switch (_dtype) {                                                                          \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kHalf, half, __VA_ARGS__)                               \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kBFloat16, nv_bfloat16, __VA_ARGS__)                    \
            default: BM_EXCEPTION("data type is not half or BF16");                                \
        };                                                                                         \
    } while (0)

#define BM_CORE_DTYPE_DISPATCH(dtype, ...)                                                              \
    do {                                                                                           \
        const auto& _dtype = dtype;                                                                \
        switch (_dtype) {                                                                          \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kInt32, int32_t, __VA_ARGS__)                           \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kFloat, float, __VA_ARGS__)                             \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kHalf, half, __VA_ARGS__)                               \
            BM_PRIVATE_DTYPE_DISPATCH_CASE(kBFloat16, nv_bfloat16, __VA_ARGS__)                    \
            default: BM_EXCEPTION(std::string("Unsupported core data type ") + core::get_data_type_name(_dtype));  \
        };                                                                                         \
    } while (0)


} // namespace core
} // namespace bmengine

namespace std {
static inline std::string to_string(bmengine::core::DataType dt) {
    return bmengine::core::get_data_type_name(dt);
}
}
