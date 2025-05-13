#include "bmengine/functions/gemm.h"
#include "bmengine/core/exception.h"
#include "bmengine/logger/kernel_time_trace.hpp"
#include "bmengine/logger/std_log_op.hpp"
#include <iostream>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

static inline int get_int_env(const char* name, int def_val = 0) {
    char* env_str = std::getenv(name);
    return env_str != nullptr ? std::atoi(env_str) : def_val;
}

namespace bmengine {
namespace functions {

using bmengine::core::Tensor;

#define ADD_KEY_AND_STR(x) {x, #x}

struct LtGemmAlgoAttr {
    int algo_id;
    int tile_id;
    int stage_id;
    int split_k;
    int reduction_scheme;
    int swizzle;
    int custom_option;
    size_t workspace_size = 0;
    int wave_count;
};

LtGemmAlgoAttr get_algo_attr(const cublasLtMatmulAlgo_t* algo);
std::string tile_to_str(int id);
std::string stage_to_str(int id);

// clang-format off
class Gemm::impl {
public:
    bool transA, transB;
    core::DataType data_type;
    float alpha_, beta_;
    __half half_alpha, half_beta;
    int int_alpha { 1 }, int_beta { 0 };
    void* a_scale { nullptr };
    void* b_scale { nullptr };

    cublasComputeType_t compute_type;
    cudaDataType_t scale_type;
    cudaDataType_t in_type;
    cudaDataType_t out_type;

    int algo_id { -1 };
    int algo_num_search { 20 };
    int algo_restrict { false };
    cublasLtMatmulAlgo_t algo;
    bool algo_found { false };
    size_t last_m { 0 };
    size_t last_batch { 0 };
    std::string prefix;
    bool is_fp8 { false };

public:
    impl(const core::Context& ctx, core::DataType data_type, bool transA, bool transB, float alpha)
        : data_type(data_type), transA(transA), transB(transB), alpha_(alpha), beta_(0) {
        half_alpha = __float2half(alpha);
        half_beta = __float2half(beta_);

        if (data_type == core::DataType::kFloat) {
            compute_type = CUBLAS_COMPUTE_32F;
            scale_type = in_type = out_type = CUDA_R_32F;
        } else if (data_type == core::DataType::kHalf) {
            compute_type = CUBLAS_COMPUTE_16F;
            scale_type = in_type = out_type = CUDA_R_16F;
        } else if (data_type == core::DataType::kBFloat16) {
            // Only support 32F
            compute_type = CUBLAS_COMPUTE_32F;
            scale_type = CUDA_R_32F;
            in_type = out_type = CUDA_R_16BF;
        } else if (data_type == core::DataType::kInt8) {
            // https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
            // To use IMMA kernels, requirement set 1:
            //   Only the “TN” format is supported - A must be transposed and B non-transposed
            BM_ASSERT(!transA && transB, "imma need NT");
            compute_type = CUBLAS_COMPUTE_32I;
            scale_type = out_type = CUDA_R_32I;
            in_type = CUDA_R_8I;
            this->data_type = core::DataType::kInt32; // result type
#if CUDART_VERSION >= 12010
        } else if (data_type == core::DataType::kFP8_E4M3 || data_type == core::DataType::kFP8_E5M2) {
            BM_ASSERT(!transA && transB, "fp8 need NT");
            // Only support 32F
            compute_type = CUBLAS_COMPUTE_32F;
            scale_type = CUDA_R_32F;
            in_type = data_type == core::DataType::kFP8_E4M3 ? CUDA_R_8F_E4M3 : CUDA_R_8F_E5M2;
            out_type = CUDA_R_16F;
            this->data_type = core::DataType::kHalf; // result type
            is_fp8 = true;
#endif
        } else {
            BM_EXCEPTION(std::string("Unsupported data type ") + core::get_data_type_name(data_type));
        }
    }

    impl(const impl&) = delete;
    impl(impl&&) = delete;
    ~impl() {}

    void scale_output(float factor) {
        alpha_ *= factor;
        half_alpha = __float2half(alpha_);
    }

    cublasLtMatmulDesc_t create_desc() {
        cublasLtMatmulDesc_t matmul_desc;
        BM_CUBLAS_ASSERT(cublasLtMatmulDescCreate(&matmul_desc, compute_type, scale_type));

        cublasOperation_t op_transpose = CUBLAS_OP_T;
        if (transB) {
            BM_CUBLAS_ASSERT(cublasLtMatmulDescSetAttribute(
                matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_transpose, sizeof(op_transpose)));
        }
        if (transA) {
            BM_CUBLAS_ASSERT(cublasLtMatmulDescSetAttribute(
                matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_transpose, sizeof(op_transpose)));
        }
#if CUDART_VERSION >= 12010
        if (a_scale) {
            BM_CUBLAS_ASSERT(cublasLtMatmulDescSetAttribute(
                matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a_scale, sizeof(a_scale)));
        }
        if (b_scale) {
            BM_CUBLAS_ASSERT(cublasLtMatmulDescSetAttribute(
                matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &b_scale, sizeof(b_scale)));
        }
#endif
        return matmul_desc;
    }

    void set_output_type(core::DataType dtype) {
        if (dtype == core::DataType::kFloat) {
            data_type = dtype;
            out_type = CUDA_R_32F;
            set_compute_type(CUBLAS_COMPUTE_32F);
        } else if (is_fp8) {
            BM_ASSERT(dtype == core::DataType::kHalf || dtype == core::DataType::kBFloat16, "");
            data_type = dtype;
            out_type = dtype == core::DataType::kHalf ? CUDA_R_16F : CUDA_R_16BF;
        } else {
            BM_EXCEPTION(std::string("Unsupported data type ") + std::to_string(compute_type));
        }
    }

    void set_compute_type(cublasComputeType_t compute_type) {
        if (compute_type == CUBLAS_COMPUTE_32F) {
            this->compute_type = compute_type;
            scale_type = CUDA_R_32F;
        } else {
            BM_EXCEPTION(std::string("Unsupported data type ") + std::to_string(compute_type));
        }
    }

    void get_scale(const void *& p_alpha, const void *& p_beta) {
        if (scale_type == CUDA_R_32F) {
            p_alpha = &alpha_, p_beta = &beta_;
        } else if (scale_type == CUDA_R_16F) {
            p_alpha = &half_alpha, p_beta = &half_beta;
        } else if (scale_type == CUDA_R_32I) {
            p_alpha = &int_alpha, p_beta = &int_beta;
        } else {
            BM_EXCEPTION("Unsupported data type");
        }
    }

    void find_algo(
        const core::Context& ctx,
        cublasLtMatrixLayout_t layout_A,
        cublasLtMatrixLayout_t layout_B,
        cublasLtMatrixLayout_t layout_C,
        uint32_t M,
        uint32_t K,
        uint32_t N,
        uint32_t batch = 0) {
        if (algo_id != -1 && (last_m != M || last_batch != batch)) {
            algo_found = false;
            if ((batch && N % 16 != 0 && M % 16 != 0) || (!batch && N < 128 && K < 128 && M < 128))
                return;
            cublasLtMatmulPreference_t preference;
            BM_CUBLAS_ASSERT(cublasLtMatmulPreferenceCreate(&preference));
            cublasLtMatmulHeuristicResult_t results[algo_num_search];
            for (int i = 0; i < algo_num_search; ++i) {
                BM_CUBLAS_ASSERT(cublasLtMatmulAlgoInit(
                    ctx.current_cublas_handle(),
                    compute_type,
                    scale_type,
                    in_type,
                    in_type,
                    out_type,
                    out_type,
                    algo_id,
                    &results[i].algo));
            }
            if (algo_restrict) {
                cublasLtMatmulSearch_t search_mode = CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID;
                BM_CUBLAS_ASSERT(cublasLtMatmulPreferenceSetAttribute(
                    preference, CUBLASLT_MATMUL_PREF_SEARCH_MODE, &search_mode, sizeof(search_mode)));
            }
            size_t workspace_size = 0;
            BM_CUBLAS_ASSERT(cublasLtMatmulPreferenceSetAttribute(
                preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(size_t)));

            int nb_result = 0;
            cublasLtMatmulDesc_t matmul_desc = create_desc();
            auto status = (cublasLtMatmulAlgoGetHeuristic(
                ctx.current_cublas_handle(), matmul_desc,
                layout_B, layout_A, layout_C, layout_C,
                preference, algo_num_search, &results[0], &nb_result));
            if (CUBLAS_STATUS_SUCCESS != status) {
                std::cerr << prefix << ctx.current_layer()
                << " not support, N=" << N << ", K=" << K << ", M=" << M << ", batch=" << batch << "\n";
                return;
            }
            BM_ASSERT(nb_result > 0, "nb_result is 0");
            int target_algo_id = algo_id == 13 && M > 1 ? 24 : algo_id;
            for (int i = 0; i < nb_result; ++i) {
                auto attr = get_algo_attr(&results[i].algo);
                if (ctx.debug() > 1 && algo_id == 13) {
                    std::cout << prefix << ", N=" << N << ", K=" << K << ", M=" << M
                              << ", algo=" << attr.algo_id << ", tile=" << tile_to_str(attr.tile_id)
                              << ", stage=" << stage_to_str(attr.stage_id)
                              << ", split=" << attr.split_k << ", scheme=" << attr.reduction_scheme << "\n";
                }
                if (attr.algo_id != target_algo_id) {
                    continue;
                }
                algo = results[i].algo;
                algo_found = true;
                break;
            }
            cublasLtMatmulPreferenceDestroy(preference);
            if (!algo_found && last_m == 0) {
                std::cerr << prefix << ", algo_id=" << algo_id << ", restrict=" << algo_restrict
                    << ". Algo not found!!!\n";
            }
            auto attr = get_algo_attr(algo_found ? &algo : &results[0].algo);
            if (!algo_found || ctx.debug() > 1 && ctx.current_layer() == 0) {
                std::cout << prefix << ctx.current_layer() << ", N=" << N << ", K=" << K << ", M=" << M
                          << ", algo=" << attr.algo_id << ", tile=" << tile_to_str(attr.tile_id)
                          << ", stage=" << stage_to_str(attr.stage_id)
                          << ", split=" << attr.split_k << ", scheme=" << attr.reduction_scheme << "\n";
            }
            last_m = M;
            last_batch = batch;
        }
    }

    core::Tensor gemm(
        const core::Context& ctx,
        const core::Tensor& A,
        const core::Tensor& B,
        core::Tensor* output,
        const Tensor* bias=nullptr) {
        auto stream = ctx.current_stream()->ptr;
        cublasLtMatrixLayout_t layout_A, layout_B, layout_C;
        unsigned int M, N, K1, K2;
        if (transA) {
            BM_ASSERT(A.ndim() == 2, "A should be a 2D matrix");
            M = A.size(1);
            K1 = A.size(0);
        } else {
            K1 = A.size(-1);
            M = A.numel() / K1;
        }
        if (transB) {
            K2 = B.size(1);
            N = B.size(0);
        } else {
            K2 = B.size(0);
            N = B.size(1);
        }
        static int kAlignMInt8 = get_int_env("GEMM_INT8_ALIGN_M", 0);
        bool align_m = is_fp8 || kAlignMInt8 > 0 && (compute_type == CUBLAS_COMPUTE_32I);
        // bool align_m = kAlignMInt8 > 0 && (compute_type == CUBLAS_COMPUTE_32I || in_type == CUDA_R_8F_E4M3 || in_type == CUDA_R_8F_E5M2);
        if (align_m) {
            kAlignMInt8 = kAlignMInt8 == 0 ? 32 : kAlignMInt8;
            BM_ASSERT(A.mem_bytes() % (kAlignMInt8 * K1) == 0, "input size0 isn't aligned.");
            size_t old_M = M;
            M = round_up(M, kAlignMInt8);
            if (M > old_M)
                BM_CUDART_ASSERT(cudaMemsetAsync(A.data<char>() + A.nbytes(), 0, A.mem_bytes() - A.nbytes(), stream));
        }

        BM_ASSERT_EQ(K1, K2, "Matrix dimensions mismatch");
        cublasLtMatmulDesc_t matmul_desc = create_desc();
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutCreate(&layout_A, in_type, A.size(-1), transA ? K1 : M, A.stride(-2)));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutCreate(&layout_B, in_type, B.size(1), B.size(0), B.stride(-2)));
        if (bias) {
            cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            BM_CUBLAS_ASSERT(cublasLtMatmulDescSetAttribute(
                matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
            void* bias_ptr = bias->data();
            BM_CUBLAS_ASSERT(cublasLtMatmulDescSetAttribute(
                matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
        }

        const void *p_alpha, *p_beta;
        get_scale(p_alpha, p_beta);

        std::vector<size_t> c_shape = A.shape();
        c_shape[c_shape.size() - 1] = N;
        if (output) {
            BM_ASSERT_EQ(c_shape, output->shape(), "shape mismatch");
        }
        size_t round_up_n = align_m ? kAlignMInt8 * N * sizeof(int) : 1024;
        core::Tensor ret = (output != nullptr) ? *output : ctx.tensor(c_shape, data_type, "", round_up_n);
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutCreate(&layout_C, out_type, N, M, ret.stride(-2)));

        find_algo(ctx, layout_A, layout_B, layout_C, M, K1, N);

        BM_CUBLAS_ASSERT(cublasLtMatmul(
            ctx.current_cublas_handle(),
            matmul_desc,
            p_alpha,
            B.data(),
            layout_B,
            A.data(),
            layout_A,
            p_beta,
            ret.data(),
            layout_C,
            ret.data(),
            layout_C,
            algo_found ? &algo : nullptr,
            NULL,
            0,
            stream));

        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutDestroy(layout_A));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutDestroy(layout_B));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutDestroy(layout_C));
        BM_CUBLAS_ASSERT(cublasLtMatmulDescDestroy(matmul_desc));
        return ret;
    }

    core::Tensor gemm_batched(
        const core::Context& ctx,
        const core::Tensor& A,
        const core::Tensor& B,
        core::Tensor* output) {
        cublasLtMatrixLayout_t layout_A, layout_B, layout_C;
        unsigned int M, N, K1, K2;
        if (transA) {
            M = A.size(-1);
            K1 = A.size(-2);
        } else {
            M = A.size(-2);
            K1 = A.size(-1);
        }
        if (transB) {
            K2 = B.size(-1);
            N = B.size(-2);
        } else {
            K2 = B.size(-2);
            N = B.size(-1);
        }
        BM_ASSERT_EQ(K1, K2, "Matrix dimensions mismatch");
        BM_ASSERT(
            A.size(0) == 1 || (B.size(0) == 1 || B.ndim() == 2) || A.size(0) == B.size(0),
            "Batch dimensions mismatch");

        uint32_t batch_A = A.size(0);
        uint32_t batch_B = (B.ndim() <= 2) ? 1 : B.size(0);
        uint32_t batch_count = std::max(batch_A, batch_B);  // buggy here
        int64_t stride_A = (A.size(0) == 1) ? 0 : A.stride(0);
        int64_t stride_B = (B.ndim() <= 2) ? 0 : B.stride(0);
        int64_t stride_C = M * N;

        std::vector<size_t> c_shape = {batch_A, M, N};
        if (output) {
            BM_ASSERT_EQ(c_shape, output->shape(), "shape mismatch");
            stride_C = output->stride(0);
        }
        const core::Tensor& ret = (output != nullptr) ? *output : ctx.tensor(c_shape, data_type);

        cublasLtMatmulDesc_t matmul_desc = create_desc();
        BM_CUBLAS_ASSERT(
            cublasLtMatrixLayoutCreate(&layout_A, in_type, A.size(-1), A.size(-2), A.stride(-2)));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutSetAttribute(
            layout_A, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutSetAttribute(
            layout_A, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_A, sizeof(stride_A)));

        BM_CUBLAS_ASSERT(
            cublasLtMatrixLayoutCreate(&layout_B, in_type, B.size(-1), B.size(-2), B.stride(-2)));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutSetAttribute(
            layout_B, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutSetAttribute(
            layout_B, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_B, sizeof(stride_B)));

        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutCreate(&layout_C, out_type, N, M, ret.stride(-2)));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutSetAttribute(
            layout_C, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutSetAttribute(
            layout_C, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_C, sizeof(stride_C)));

        const void *p_alpha, *p_beta;
        get_scale(p_alpha, p_beta);

        find_algo(ctx, layout_A, layout_B, layout_C, M, K1, N, batch_A);

        BM_CUBLAS_ASSERT(cublasLtMatmul(
            ctx.current_cublas_handle(),
            matmul_desc,
            p_alpha,
            B.data(),
            layout_B,
            A.data(),
            layout_A,
            p_beta,
            ret.data(),
            layout_C,
            ret.data(),
            layout_C,
            algo_found ? &algo : nullptr,
            NULL,
            0,
            ctx.current_stream()->ptr));

        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutDestroy(layout_A));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutDestroy(layout_B));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutDestroy(layout_C));
        BM_CUBLAS_ASSERT(cublasLtMatmulDescDestroy(matmul_desc));
        return ret;
    }

    static void set_batch_layout(cublasLtMatrixLayout_t layout, uint32_t batch, int64_t stride) {
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutSetAttribute(
            layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutSetAttribute(
            layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
    }

    core::Tensor batch_3d(
        const core::Context& ctx,
        const core::Tensor& A,
        const core::Tensor& B,
        core::Tensor* output) {
        size_t batch = A.size(0);
        size_t M = A.size(transA ? -1 : -2);
        size_t K1 = A.size(transA ? -2 : -1);
        size_t N = B.size(transB ? -2 : -1);
        size_t K2 = B.size(transB ? -1 : -2);
        BM_ASSERT_EQ(K1, K2, "Matrix K1, K2 mismatch");

        std::vector<size_t> c_shape = {batch, M, N};
        if (output) {
            BM_ASSERT_EQ(c_shape, output->shape(), "shape mismatch");
        }
        const core::Tensor& ret = (output != nullptr) ? *output : ctx.tensor(c_shape, data_type);

        cublasLtMatmulDesc_t matmul_desc = create_desc();
        cublasLtMatrixLayout_t layout_A, layout_B, layout_C;
        BM_CUBLAS_ASSERT(
            cublasLtMatrixLayoutCreate(&layout_A, in_type, A.size(-1), A.size(-2), A.stride(-2)));

        BM_CUBLAS_ASSERT(
            cublasLtMatrixLayoutCreate(&layout_B, in_type, B.size(-1), B.size(-2), B.stride(-2)));

        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutCreate(&layout_C, out_type, N, M, ret.stride(-2)));

        set_batch_layout(layout_A, batch, A.stride(0));
        set_batch_layout(layout_B, batch, B.stride(0));
        set_batch_layout(layout_C, batch, ret.stride(0));

        const void *p_alpha, *p_beta;
        get_scale(p_alpha, p_beta);

        BM_CUBLAS_ASSERT(cublasLtMatmul(
            ctx.current_cublas_handle(),
            matmul_desc,
            p_alpha,
            B.data(),
            layout_B,
            A.data(),
            layout_A,
            p_beta,
            ret.data(),
            layout_C,
            ret.data(),
            layout_C,
            algo_found ? &algo : nullptr,
            nullptr,
            0,
            ctx.current_cuda_stream()));

        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutDestroy(layout_A));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutDestroy(layout_B));
        BM_CUBLAS_ASSERT(cublasLtMatrixLayoutDestroy(layout_C));
        BM_CUBLAS_ASSERT(cublasLtMatmulDescDestroy(matmul_desc));
        return ret;
    }
};

Gemm::Gemm(const core::Context& ctx, core::DataType dtype, bool transA, bool transB, float alpha)
    : pimpl(new impl(ctx, dtype, transA, transB, alpha)), Layer() { }

Gemm::~Gemm() = default;

core::Tensor Gemm::forward(
    const core::Context& ctx, const Tensor& A, const Tensor& B, Tensor* output, const Tensor* bias) {
    pimpl->prefix = prefix;
    if (A.ndim() == 4 && B.ndim() == 4) {
        BM_ASSERT_EQ(A.size(0), B.size(0), "Matrix size(0) mismatch");
        BM_ASSERT_EQ(A.size(1), B.size(1), "Matrix size(1) mismatch");
        // convert to 3d
        size_t fused_batch = A.size(0) * A.size(1);
        const core::Tensor a3d = A.view({ fused_batch, A.size(2), A.size(3) });
        const core::Tensor b3d = B.view({ fused_batch, B.size(2), B.size(3) });
        core::Tensor tmp;
        core::Tensor* ptr = nullptr;
        if (output) {
            tmp = output->view({ fused_batch, output->size(2), output->size(3) });
            ptr = &tmp;
        }
        core::Tensor ret = pimpl->gemm_batched(ctx, a3d, b3d, ptr);
        // convert back to 4d
        std::vector<size_t> out_shape = { A.size(0), A.size(1), ret.size(-2), ret.size(-1) };
        if (output) {
            *output = output->view(out_shape);
        }
        return ret.view(out_shape);
    }

    BM_ASSERT(A.ndim() >= B.ndim(), "Matrix dimensions mismatch");
    BM_ASSERT(A.ndim() == 2 || A.ndim() == 3, "Matrix dimensions mismatch");
    if (A.ndim() == 2 || (B.ndim() == 2 && !pimpl->transA)) {
        return pimpl->gemm(ctx, A, B, output, bias);
    } else {
        return pimpl->gemm_batched(ctx, A, B, output);
    }
}

core::Tensor Gemm::batch_3d(
    const core::Context& ctx, const Tensor& A, const Tensor& B, Tensor* output) {
    pimpl->prefix = prefix;
    BM_ASSERT_EQ(A.ndim(), 3, "MatrixA is not 3D");
    BM_ASSERT_EQ(B.ndim(), 3, "MatrixB is not 3D");
    BM_ASSERT_EQ(A.size(0), B.size(0), "Matrix batch=size(0) mismatch");
    if (output) {
        BM_ASSERT_EQ(output->ndim(), 3, "MatrixD is not 3D");
        BM_ASSERT_EQ(A.size(0), output->size(0), "Matrix batch=size(0) mismatch");
    }
//    if (A.size(0) == 1 && B.size(0) == 1 && A.is_continuous() && B.is_continuous()) {
//        // squeeze batch, convert to 2d
//        const core::Tensor a2d = A.view({ A.size(1), A.size(2) });
//        const core::Tensor b2d = B.view({ B.size(1), B.size(2) });
//        core::Tensor tmp;
//        core::Tensor* ptr = nullptr;
//        if (output) {
//            tmp = output->view({ output->size(1), output->size(2) });
//            ptr = &tmp;
//        }
//        core::Tensor ret = pimpl->gemm(ctx, a2d, b2d, ptr);
//        // convert back to 3d
//        std::vector<size_t> out_shape = { 1UL, ret.size(0), ret.size(1) };
//        if (output) {
//            *output = output->view(out_shape);
//        }
//        return ret.view(out_shape);
//    }
    {
        return pimpl->batch_3d(ctx, A, B, output);
    }
}

void Gemm::scale_output(float alpha) {
    pimpl->scale_output(alpha);
}

void Gemm::set_output_type(core::DataType dtype) {
    pimpl->set_output_type(dtype);
}

void Gemm::set_compute_type(cublasComputeType_t compute_type) {
    pimpl->set_compute_type(compute_type);
}

void Gemm::set_algo_id(int id, int num_search, bool restrict) {
    pimpl->algo_id = id;
    pimpl->algo_num_search = num_search;
    pimpl->algo_restrict = restrict;
}

void Gemm::set_A_scale(const core::Tensor& A_scale) {
    pimpl->a_scale = A_scale.data();
}
void Gemm::set_B_scale(const core::Tensor& B_scale) {
    pimpl->b_scale = B_scale.data();
}

LtGemmAlgoAttr get_algo_attr(const cublasLtMatmulAlgo_t* algo) {
    LtGemmAlgoAttr attr;
    BM_CUBLAS_ASSERT(cublasLtMatmulAlgoConfigGetAttribute(
        algo, CUBLASLT_ALGO_CONFIG_ID, &attr.algo_id, sizeof(int), nullptr));
    BM_CUBLAS_ASSERT(cublasLtMatmulAlgoConfigGetAttribute(
        algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &attr.tile_id, sizeof(int), nullptr));
    BM_CUBLAS_ASSERT(cublasLtMatmulAlgoConfigGetAttribute(
        algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &attr.stage_id, sizeof(int), nullptr));
    BM_CUBLAS_ASSERT(cublasLtMatmulAlgoConfigGetAttribute(
        algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &attr.split_k, sizeof(int), nullptr));
    BM_CUBLAS_ASSERT(cublasLtMatmulAlgoConfigGetAttribute(
        algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &attr.reduction_scheme, sizeof(int), nullptr));
    BM_CUBLAS_ASSERT(cublasLtMatmulAlgoConfigGetAttribute(
        algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &attr.swizzle, sizeof(int), nullptr));
    BM_CUBLAS_ASSERT(cublasLtMatmulAlgoConfigGetAttribute(
        algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &attr.custom_option, sizeof(int), nullptr));
    return attr;
}

std::string tile_to_str(int id) {
    const static std::map<cublasLtMatmulTile_t, std::string> tile_map{
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_UNDEFINED),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_8x8),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_8x16),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_16x8),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_8x32),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_16x16),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_32x8),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_8x64),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_16x32),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_32x16),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_64x8),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_32x32),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_32x64),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_64x32),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_32x128),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_64x64),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_128x32),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_64x128),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_128x64),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_64x256),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_128x128),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_256x64),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_64x512),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_128x256),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_256x128),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_512x64),
    };
    return tile_map.at(static_cast<cublasLtMatmulTile_t>(id)).substr(21);
}

std::string stage_to_str(int id) {
    const static std::map<cublasLtMatmulStages_t, std::string> stage_map{
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_UNDEFINED),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_16x1),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_16x2),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_16x3),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_16x4),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_16x5),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_16x6),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_32x1),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_32x2),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_32x3),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_32x4),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_32x5),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_32x6),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_64x1),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_64x2),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_64x3),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_64x4),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_64x5),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_64x6),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_128x1),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_128x2),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_128x3),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_128x4),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_128x5),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_128x6),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_32x10),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_8x4),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_STAGES_16x10),
    };
    return stage_map.at(static_cast<cublasLtMatmulStages_t>(id)).substr(23);
}
// clang-format on

} // namespace functions
} // namespace bmengine
