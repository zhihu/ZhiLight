#pragma once
#include "bmengine/core/core.h"

namespace bmengine {

namespace functions {

class Gemm : public core::Layer {
    BM_LAYER_DEF(Gemm)

    Gemm(
        const core::Context& ctx,
        core::DataType dtype,
        bool transA,
        bool transB,
        float alpha = 1.0);

    void scale_output(float factor);
    void set_output_type(core::DataType dtype);

    // set to CUBLAS_COMPUTE_32F for half gemm to use float accumulator
    void set_compute_type(cublasComputeType_t compute_type);

    void set_algo_id(int id, int num_search = 20, bool restrict = false);

    void set_A_scale(const core::Tensor& A_scale);
    void set_B_scale(const core::Tensor& B_scale);

    core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& A,
        const core::Tensor& B,
        core::Tensor* output = nullptr,
        const core::Tensor* bias = nullptr
    );

    core::Tensor batch_3d(
        const core::Context& ctx,
        const core::Tensor& A,
        const core::Tensor& B,
        core::Tensor* output = nullptr
    );
};

} // namespace functions

} // namespace bmengine
